"""Stage a cloud bundle in object storage as a few tar parts, then pull them onto the instance.

The home uplink is the slow leg either way; staging ahead means no GPU is billed while shards
trickle up, and a staged bundle serves any number of instances (user, 17 Nov 2026). The parts
live in the user's IDrive e2 bucket (rclone remote `r2`, type s3, provider IDrive): `pack`
writes them locally and the user uploads them with their own rclone.

  python -B scripts/cloud_storage.py pack --bundle .cache/cloud-bundles/v11-1500 \
      --output .cache/cloud-bundles/v11-1500-parts

  python -B scripts/cloud_storage.py pull --source r2:training-data \
      --host root@1.2.3.4 --port 40758 --identity ~/.ssh/vast_ai

`pull` installs rclone RCLONE_VERSION (the release the user verified) on the instance, copies only
that remote's section of the local rclone config there (never the rest of the file), streams each
part through tar into the bundle, deletes the config, and hashes every shard against
data/dataset.json on the instance (cloud_upload --verify does the same). It is resumable: a part is marked done once fully
extracted, so a rerun fetches only the rest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

from cloud_upload import parts, shard_source
from engine_runtime import disable_power_throttling

RCLONE_VERSION = 'v1.76.0'
RCLONE = Path.home()/'Downloads/rclone-v1.76.0-windows-amd64/rclone-v1.76.0-windows-amd64/rclone.exe'


def run(command, **kwargs):
    process = subprocess.Popen(command, **kwargs)
    disable_power_throttling(int(process._handle))
    if process.wait():
        raise SystemExit(f'Failed ({process.returncode}): {" ".join(map(str, command[:3]))}')


def tar(directory, names, output):
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(names)+'\n')
        listing = Path(stream.name)
    partial = output.with_suffix('.partial')
    try:
        run(['tar', '-C', str(directory), '-cf', str(partial), '-T', str(listing)])
        partial.replace(output)  # Only whole parts get a final name, so an upload never takes half of one.
    finally:
        listing.unlink()


def pack(args):
    bundle = Path(args.bundle).resolve()
    source = shard_source(bundle)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, *_ = parts(bundle, 1)
    doc = json.loads((bundle/'data/dataset.json').read_text(encoding='utf-8'))
    groups, group, size = [], [], 0
    for path in sorted(e['path'] for e in doc['entries']):
        group.append(path)
        size += (source/path).stat().st_size
        if size >= args.part_gib*2**30:
            groups.append(group); group, size = [], 0
    if group:
        groups.append(group)
    jobs = [('bundle.tar', bundle, manifest)]+[(f'shards-{i:03}.tar', source, names) for i, names in enumerate(groups)]
    for ordinal, (name, directory, names) in enumerate(jobs, 1):
        if not (output/name).exists():  # Rerun to resume after an interruption.
            tar(directory, names, output/name)
        print(json.dumps({'part': name, 'done': ordinal, 'total': len(jobs)}), flush=True)
    (output/'parts.txt').write_text(''.join(name+'\n' for name, *_ in jobs), encoding='utf-8', newline='\n')


def remote_section(rclone, name):
    """Only the named remote's section of the local rclone config: other remotes' tokens stay here."""
    path = Path(subprocess.run([str(rclone), 'config', 'file'], capture_output=True, text=True,
                               check=True).stdout.strip().splitlines()[-1])
    section, keep = [], False
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('['):
            keep = line.strip() == f'[{name}]'
        if keep:
            section.append(line)
    if not section:
        raise SystemExit(f'No [{name}] remote in {path}')
    return '\n'.join(section)+'\n'


def pull(args):
    ssh = ['ssh', '-i', str(Path(args.identity).expanduser()), '-p', str(args.port), '-o', 'BatchMode=yes',
           '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ServerAliveInterval=30', args.host]
    remote_name, _, folder = args.source.partition(':')
    names = subprocess.run([str(args.rclone), 'lsf', '--files-only', args.source], capture_output=True, text=True,
                           check=True).stdout.split()
    tars = [name for name in names if name.endswith('.tar')]
    expected = subprocess.run([str(args.rclone), 'cat', args.source.rstrip('/')+'/parts.txt'], capture_output=True,
                              text=True, check=True).stdout.split()
    if sorted(tars) != sorted(expected):
        raise SystemExit(f'{args.source} lacks parts: {sorted(set(expected)-set(tars))}')
    remote = shlex.quote(args.remote)
    url = f'https://downloads.rclone.org/{RCLONE_VERSION}/rclone-{RCLONE_VERSION}-linux-amd64.zip'
    install = (f'set -euo pipefail; mkdir -p .rclone && chmod 700 .rclone; '
               f'if [ ! -x .rclone/rclone ]; then cd .rclone; '
               f'curl -fsSLO {url} && curl -fsSL https://downloads.rclone.org/{RCLONE_VERSION}/SHA256SUMS -o SUMS; '
               f'grep " rclone-{RCLONE_VERSION}-linux-amd64.zip$" SUMS | sha256sum -c -; '
               f'python3 -m zipfile -e rclone-{RCLONE_VERSION}-linux-amd64.zip x; '
               f'mv x/*/rclone rclone && chmod 700 rclone && rm -rf x SUMS *.zip; cd ..; fi; '
               f'umask 077; cat > .rclone/rclone.conf')
    result = subprocess.run(ssh+[install], input=remote_section(args.rclone, remote_name).encode())
    if result.returncode:
        raise SystemExit('Installing rclone on the instance failed')
    source = shlex.quote(args.source.rstrip('/'))
    def fetch(name):
        target = '.' if name == 'bundle.tar' else 'data'
        done = shlex.quote(f'.pull-done/{name}')
        return (f'[ -e {done} ] || {{ "$HOME/.rclone/rclone" --config "$HOME/.rclone/rclone.conf" cat {source}/{shlex.quote(name)} '
                f'| tar --no-same-owner -xf - -C {target} && touch {done}; }}')
    rest = ' '.join(shlex.quote(fetch(name)) for name in sorted(tars) if name != 'bundle.tar')
    script = (f'set -euo pipefail; mkdir -p {remote}/data {remote}/.pull-done; cd {remote}; {fetch("bundle.tar")}; '
              f'printf "%s\\0" {rest} | xargs -0 -P {args.streams} -I{{}} bash -o pipefail -c {{}}; '
              f'chmod +x *.sh; echo "parts done: $(ls .pull-done | wc -l)"')
    try:
        result = subprocess.run(ssh+[script])
    finally:
        # The bucket key never stays on a rented machine, also when the pull fails.
        subprocess.run(ssh+['rm -f .rclone/rclone.conf'])
    if result.returncode:
        raise SystemExit('Pull failed; rerun to resume')
    # Hash the shards on the instance against its own manifest. Copying the ~0.5 GB manifest home first
    # took hours on a host with ~270 KB/s egress (20 Nov 2026) and checked against the same file anyway.
    check = ('import json, subprocess, sys; d = json.load(open("data/dataset.json")); '
             'lines = "".join(e["archive_sha256"]+"  "+e["path"]+"\\n" for e in d["entries"]); '
             'sys.exit(subprocess.run(["sha256sum", "-c", "--quiet", "-"], input=lines.encode(), cwd="data").returncode '
             'or print(json.dumps({"verified": len(d["entries"])})))')
    if subprocess.run(ssh+[f'cd {remote} && python3 -c {shlex.quote(check)}']).returncode:
        raise SystemExit('Shard verification failed on the instance')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    packer = commands.add_parser('pack')
    packer.add_argument('--bundle', required=True)
    packer.add_argument('--output', required=True)
    packer.add_argument('--part-gib', type=float, default=4.0)
    puller = commands.add_parser('pull')
    puller.add_argument('--source', required=True, help='rclone path of the parts, e.g. r2:training-data')
    puller.add_argument('--rclone', type=Path, default=RCLONE, help='The verified local rclone.exe')
    puller.add_argument('--host', required=True, help='user@address')
    puller.add_argument('--port', type=int, required=True)
    puller.add_argument('--identity', required=True)
    puller.add_argument('--remote', default='bundle', help='Directory on the instance, relative to the login directory')
    puller.add_argument('--streams', type=int, default=8)
    args = parser.parse_args()
    disable_power_throttling()
    (pack if args.command == 'pack' else pull)(args)


if __name__ == '__main__':
    main()
