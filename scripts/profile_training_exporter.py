"""Profile one native replay export without changing the corpus or catalogue."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from types import SimpleNamespace

from event_resources import snapshot
from harvest_china_replays import parse_replay
from engine_runtime import isolation_profile
from replay_bootstrap import data_pack
from replay_training import export, file_sha256, records

ROOT = Path(__file__).resolve().parents[1]


def semantic_digest(trajectory):
    """Keep every observation/action/label; exclude only build identity and clocks."""
    digest = hashlib.sha256()
    for record in records(trajectory):
        if record['record'] == 'header':
            for key in ('engine_exe_crc', 'engine_git_sha', 'engine_git_dirty'):
                record.pop(key, None)
        elif record['record'] == 'sample':
            for grid in record.get('observation', {}).get('build_placements', ()):
                grid.pop('seconds', None)
        digest.update(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n')
    return digest.hexdigest()


def profile(args):
    exe, replay, source = (Path(value).resolve() for value in (args.exe, args.replay, args.source_profile))
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Choose a new report filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    header = parse_replay(replay.read_bytes())
    pack = data_pack(dict(header=header, metadata={}), Path(args.revision_data).resolve(), False)
    identities = dict(engine_sha256=file_sha256(exe), replay_sha256=file_sha256(replay),
                      data_pack_sha256=file_sha256(pack), mode=args.placement_mode, interval_ms=args.interval_ms,
                      storage=args.storage, quiet_capture=args.quiet_capture)
    # The directory is an owned disposable experiment, never an existing dataset.
    scratch_root = ROOT/'.cache'
    scratch_root.mkdir(exist_ok=True)
    result = dict(schema=1, kind='native_export_profile', **identities, complete=False, runs=[])
    with tempfile.TemporaryDirectory(prefix='native-export-profile-', dir=scratch_root) as temporary:
        work = Path(temporary).resolve()
        assert work.parent == scratch_root.resolve()
        private = isolation_profile(source, work/'profile', maps=[header['map']])
        keys = ('GENERALS_AI_USER_DATA', 'GENERALS_AI_PROFILE_EXPORT', 'GENERALS_AI_PROFILE_ALLOCATIONS')
        previous = {key: os.environ.get(key) for key in keys}
        try:
            os.environ.update(dict(zip(keys, (str(private), '1', '1'))))
            for number in range(args.repeat):
                if snapshot(work).get('available_ram_bytes', args.minimum_ram) < args.minimum_ram:
                    raise RuntimeError('Insufficient available RAM for this profiling run')
                started = time.perf_counter()
                with (work/'audit.log').open('w') as log, redirect_stdout(log):
                    manifest = export(SimpleNamespace(exe=str(exe), replay=str(replay), output=str(work/str(number)),
                        user_data=str(private), data_pack=str(pack), interval_ms=args.interval_ms,
                        timeout=args.timeout, storage=args.storage, ticks_per_second=60, placement_mode=args.placement_mode,
                        quiet_capture=args.quiet_capture, capture_policy=ROOT/'replay-runtime/policy.json'))
                wall = time.perf_counter()-started
                native_log = next((work/str(number)).rglob('*.engine.log'))
                text = native_log.read_text(encoding='utf-8', errors='replace')
                def report(prefix):
                    rows = [json.loads(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]
                    if not rows:
                        raise ValueError('Executable lacks opt-in exporter counters: '+prefix.strip())
                    return rows[-1]
                measured = dict(timings=report('TrainingExportProfile '),
                                allocations=report('TrainingExportAllocations '), wrapper_seconds=wall,
                                semantic_sha256=semantic_digest(manifest['trajectory']))
                measured['validation'] = manifest['native_validation']
                measured['audit'] = manifest['audit']
                measured['reconciliation'] = manifest['command_reconciliation']
                if result['runs'] and measured['semantic_sha256'] != result['runs'][0]['semantic_sha256']:
                    raise ValueError('Repeated exports produced different observations/actions')
                result['runs'].append(measured)
                # Keep proofs/logs flat; bulky raw payloads belong only to this experiment.
                shutil.copy2(native_log, output.with_name(output.stem+f'-{number+1}.log'))
                output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
                Path(manifest['trajectory']).unlink()
                print(json.dumps(dict(run=number+1, native_seconds=measured['timings']['seconds'],
                                      **measured['allocations'])), flush=True)
        except Exception as error:
            result['error'] = str(error)
            for number, log in enumerate(work.rglob('*.engine.log'), 1):
                shutil.copy2(log, output.with_name(output.stem+f'-failed-{number}.log'))
            output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
            raise
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    result['median_native_seconds'] = statistics.median(run['timings']['seconds'] for run in result['runs'])
    if (file_sha256(exe) != identities['engine_sha256'] or file_sha256(replay) != identities['replay_sha256']
            or file_sha256(pack) != identities['data_pack_sha256']):
        raise ValueError('Benchmark executable, replay or INI data changed')
    result['complete'] = True
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--source-profile', default=str(ROOT/'replay-runtime/profile'))
    parser.add_argument('--revision-data', default=str(ROOT/'replay-runtime/revision-data'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--placement-mode', choices=('fast', 'full_mask'), default='fast')
    parser.add_argument('--storage', choices=('binary', 'jsonl'), default='binary',
                        help='binary is the converter direct-capture path')
    parser.add_argument('--quiet-capture', choices=('all', 'scheduled'), default='scheduled',
                        help='scheduled is the converter; all keeps every quiet observation')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--interval-ms', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--min-free-ram-mib', type=int, default=2304)
    args = parser.parse_args()
    if args.repeat < 1 or not 1 <= args.interval_ms <= 10000 or min(args.timeout, args.min_free_ram_mib) <= 0:
        parser.error('Invalid profiling budget')
    args.minimum_ram = args.min_free_ram_mib*1024**2
    result = profile(args)
    print(json.dumps(dict(report=args.output, median_native_seconds=result['median_native_seconds'])))


if __name__ == '__main__':
    main()
