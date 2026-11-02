"""Keep a local copy of a cloud run's checkpoints, for the evaluation and to survive losing the instance.

Each pull hard-links latest.npz, best.npz and report.json on the instance first,
so the three are one consistent state, which a later --resume requires. Local
files are replaced only after a complete transfer. Runs until stopped; connection
failures are retried at the next interval.

  python -B scripts/cloud_download.py --host root@1.2.3.4 --port 40758 \
      --identity ~/.ssh/vast_ai --output .cache/cloud-runs/v7-45k-10m
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from engine_runtime import disable_power_throttling

FILES = 'latest.npz', 'best.npz', 'report.json'


def state(ssh, run):
    """The latest checkpoint's modification time, or '' before the first one."""
    command = f'stat -c %Y {shlex.quote(run)}/latest.npz 2>/dev/null || true'
    return subprocess.run(ssh+[command], check=True, capture_output=True, text=True, timeout=120).stdout.strip()


def pull(ssh, run, output):
    # The learner moves latest.npz aside while writing the next checkpoint: fail then, and retry.
    links = ' '.join(f'if [ -e {name} ]; then ln {name} .pull/; fi;' for name in FILES[1:])
    command = (f'set -e; cd {shlex.quote(run)}; rm -rf .pull; mkdir .pull; ln latest.npz .pull/; {links} '
               'tar -cf - -C .pull .')
    partial = output/'.partial'
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    process = subprocess.Popen(ssh+[command], stdout=subprocess.PIPE)
    disable_power_throttling(int(process._handle))
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            archive.extractall(partial, filter='data')
    finally:
        process.stdout.close()
    if process.wait():
        raise RuntimeError(f'Pull exited with {process.returncode}')
    pulled = sorted(path.name for path in partial.iterdir())
    for name in pulled:
        os.replace(partial/name, output/name)
    partial.rmdir()
    return pulled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', required=True, help='user@address')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run', default='bundle/run', help='Run directory on the instance, relative to the login directory')
    parser.add_argument('--interval', type=int, default=1800, help='Seconds between checks')
    args = parser.parse_args()
    disable_power_throttling()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ssh = ['ssh', '-i', str(Path(args.identity).expanduser()), '-p', str(args.port),
           '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'Compression=no',
           '-o', 'ConnectTimeout=30', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6', args.host]
    last = ''
    while True:
        try:
            current = state(ssh, args.run)
            if current and current != last:
                pulled = pull(ssh, args.run, output)
                last = current
                print(json.dumps({'time': time.strftime('%H:%M:%S'), 'pulled': pulled,
                                  'latest_mib': round((output/'latest.npz').stat().st_size/2**20)}), flush=True)
        except (OSError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as error:
            print(json.dumps({'time': time.strftime('%H:%M:%S'), 'retry': str(error)}), flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
