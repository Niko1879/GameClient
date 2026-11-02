"""Resumable bundle upload over tar and ssh, for workstations without rsync.

The bundle goes up as parts of at most --part-files files. A part is marked done
on the instance only after its tar stream has been fully extracted, so an
interrupted upload is rerun with the same command and resends one part at most.
--verify then hashes every shard on the instance against data/dataset.json.

  python -B scripts/cloud_upload.py --bundle .cache/cloud-bundles/v7-45k \
      --host root@1.2.3.4 --port 40758 --identity ~/.ssh/vast_ai --verify
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading
import time

from engine_runtime import disable_power_throttling


def parts(bundle, part_files):
    """Small files first, so the code and manifest arrive before the shards."""
    files = sorted(p.relative_to(bundle).as_posix() for p in bundle.iterdir() if p.is_file())
    files += sorted('code/'+p.name for p in (bundle/'code').iterdir())
    yield files+['data/dataset.json']
    doc = json.loads((bundle/'data/dataset.json').read_text(encoding='utf-8'))
    shards = sorted('data/'+e['path'] for e in doc['entries'])
    for start in range(0, len(shards), part_files):
        yield shards[start:start+part_files]


def part_id(names):
    return hashlib.sha256('\n'.join(names).encode()).hexdigest()[:24]


def manifest_id(bundle, names):
    """By content: a rebuilt bundle has the same names, and its code and config must be resent."""
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode()+b'\0'+(bundle/name).read_bytes())
    return digest.hexdigest()[:24]


def size(bundle, names):
    return sum((bundle/name).stat().st_size for name in names)


def upload_part(args, bundle, ssh, names, marker):
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(names)+'\n')
        listing = Path(stream.name)
    remote = shlex.quote(args.remote)
    command = f'tar --no-same-owner -xf - -C {remote} && touch {remote}/.upload-done/{marker}'
    try:
        tar = subprocess.Popen(['tar', '-C', str(bundle), '-cf', '-', '-T', str(listing)], stdout=subprocess.PIPE)
        receiver = subprocess.Popen(ssh+[command], stdin=tar.stdout)
        disable_power_throttling(int(tar._handle))
        disable_power_throttling(int(receiver._handle))
        tar.stdout.close()
        receiver_code = receiver.wait()
        tar_code = tar.wait()
        return tar_code == 0 and receiver_code == 0
    finally:
        listing.unlink()


def verify(args, bundle, ssh):
    doc = json.loads((bundle/'data/dataset.json').read_text(encoding='utf-8'))
    lines = ''.join(f"{e['archive_sha256']}  {e['path']}\n" for e in doc['entries'])
    command = f'cd {shlex.quote(args.remote)}/data && sha256sum -c --quiet -'
    result = subprocess.run(ssh+[command], input=lines.encode())
    if result.returncode:
        raise SystemExit('Shard verification failed on the instance')
    print(json.dumps({'verified': len(doc['entries'])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--host', required=True, help='user@address')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--remote', default='bundle', help='Directory on the instance, relative to the login directory')
    parser.add_argument('--part-files', type=int, default=100)
    parser.add_argument('--streams', type=int, default=12,
                        help='Parallel connections; rented hosts can limit each one to ~10 Mbit/s')
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--limit-bytes', type=int, help='Stop after about this many bytes: a speed test')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    disable_power_throttling()
    bundle = Path(args.bundle).resolve()
    ssh = ['ssh', '-i', str(Path(args.identity).expanduser()), '-p', str(args.port),
           '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'Compression=no',
           '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6', args.host]
    remote = shlex.quote(args.remote)
    listing = subprocess.run(ssh+[f'mkdir -p {remote}/data {remote}/code {remote}/.upload-done && ls {remote}/.upload-done'],
                             check=True, capture_output=True, text=True)
    done = set(listing.stdout.split())
    manifest, *shard_parts = parts(bundle, args.part_files)
    markers = {tuple(manifest): manifest_id(bundle, manifest)}
    markers.update((tuple(names), part_id(names)) for names in shard_parts)
    pending = [list(names) for names, marker in markers.items() if marker not in done]
    total = sum(size(bundle, names) for names in pending)
    sent = 0
    started = time.time()
    print(json.dumps({'pending_parts': len(pending), 'pending_gib': round(total/2**30, 2)}), flush=True)
    lock = threading.Lock()

    def send(names):
        nonlocal sent
        if args.limit_bytes and sent >= args.limit_bytes:
            return
        marker = markers[tuple(names)]
        for attempt in range(args.retries+1):
            if upload_part(args, bundle, ssh, names, marker):
                break
            if attempt == args.retries:
                raise RuntimeError(f'Part {marker} failed {attempt+1} times; rerun to resume')
            time.sleep(min(60, 5*2**attempt))
        with lock:
            sent += size(bundle, names)
            rate = sent/max(time.time()-started, 1e-3)
            print(json.dumps({'sent_gib': round(sent/2**30, 2), 'total_gib': round(total/2**30, 2),
                              'mbit_s': round(rate*8/1e6, 1), 'eta_hours': round((total-sent)/max(rate, 1)/3600, 2)}), flush=True)

    # The code and manifest part goes first, alone, so a partial upload is usable.
    if manifest in pending:
        send(manifest)
    with ThreadPoolExecutor(args.streams) as pool:
        futures = []
        for ordinal, names in enumerate(names for names in pending if names != manifest):
            futures.append(pool.submit(send, names))
            # sshd's MaxStartups refuses connections that all arrive together.
            if ordinal < args.streams:
                time.sleep(1)
        for future in as_completed(futures):
            future.result()
    if args.limit_bytes and sent >= args.limit_bytes:
        return
    if args.verify:
        verify(args, bundle, ssh)


if __name__ == '__main__':
    main()
