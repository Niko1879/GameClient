"""Profile real event preparation, including spawned workers, without publishing data.

Pass an already validated native dataset. All prepared ZIPs are disposable; only
the report and merged cProfile files remain. No catalogue or corpus is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import cProfile
from contextlib import redirect_stdout
import hashlib
import json
import multiprocessing.util
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def save(path, document):
    Path(path).write_text(json.dumps(document, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def worker_finished(output, started, cpu):
    save(Path(output)/f'worker-{os.getpid()}.json', dict(pid=os.getpid(),
        lifetime_wall_seconds=time.perf_counter()-started,
        lifetime_cpu_seconds=time.process_time()-cpu))


def worker_initializer(initializer, initargs, output):
    global WORKER_OUTPUT
    WORKER_OUTPUT = Path(output)
    started, cpu = time.perf_counter(), time.process_time()
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        import replay_metadata
        replay_metadata.CACHE_DIRECTORY = WORKER_OUTPUT/'metadata-cache'
        initializer(*initargs)
    finally:
        profiler.disable()
        profiler.dump_stats(str(WORKER_OUTPUT/f'initializer-{os.getpid()}.pstats'))
    multiprocessing.util.Finalize(None, worker_finished, args=(output, started, cpu), exitpriority=10)


def profiled_job(job, catalog):
    import event_store
    started, cpu = time.perf_counter(), time.process_time()
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        return event_store._preparation_job(job, catalog)
    finally:
        profiler.disable()
        stem = WORKER_OUTPUT/f'job-{os.getpid()}-{job["ordinal"]}'
        profiler.dump_stats(str(stem.with_suffix('.pstats')))
        save(stem.with_suffix('.json'), dict(pid=os.getpid(), match_key=job['match_key'],
            ordinal=job['ordinal'], wall_seconds=time.perf_counter()-started,
            cpu_seconds=time.process_time()-cpu))


class ProfiledExecutor(ProcessPoolExecutor):
    def __init__(self, *args, initializer, initargs, profile_output, **kwargs):
        super().__init__(*args, initializer=worker_initializer,
            initargs=(initializer, initargs, profile_output), **kwargs)

    def submit(self, function, *args, **kwargs):
        import event_store
        if function is not event_store._preparation_job:
            raise ValueError('Unexpected preparation worker function')
        return super().submit(profiled_job, *args, **kwargs)


def payloads(manifest):
    """Compare exact uncompressed members; ZIP timestamps may differ."""
    from event_dataset import EventDataset
    data, result = EventDataset(manifest), {}
    for entry in data.entries:
        with zipfile.ZipFile(data.shard_path(entry)) as archive:
            members = [(name, hashlib.sha256(archive.read(name)).hexdigest())
                       for name in sorted(archive.namelist())]
            if archive.testzip() is not None:
                raise ValueError('Prepared ZIP CRC failed')
        result[entry['match_key']] = dict(
            payload_sha256=hashlib.sha256(json.dumps(members, separators=(',', ':')).encode()).hexdigest(),
            rows=sum(episode['length'] for episode in entry['index']['episodes']),
            bytes=data.shard_path(entry).stat().st_size, coverage=entry['index']['coverage'],
            feature_key=entry['index']['legality_key'])
    decoded = sum(len(data.read(reference)) for split in ('train', 'validation', 'test')
                  for reference in data.chunks(split))
    if decoded != sum(entry['rows'] for entry in result.values()):
        raise ValueError('Decoded row count differs from manifest')
    return result


class MemoryTrace:
    def __init__(self):
        self.stopped, self.samples, self.near_peak = threading.Event(), [], None

    def start(self):
        import tracemalloc
        tracemalloc.start(8)
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        import tracemalloc
        captured = 0
        while not self.stopped.wait(.2):
            current, peak = tracemalloc.get_traced_memory()
            self.samples.append([time.perf_counter()-self.started, current, peak])
            if current > captured+8*1024**2:
                self.near_peak = tracemalloc.take_snapshot()
                captured = current

    def finish(self):
        import tracemalloc
        self.stopped.set()
        self.thread.join()
        current, peak = tracemalloc.get_traced_memory()
        snapshot = self.near_peak or tracemalloc.take_snapshot()
        top = [dict(file=str(item.traceback[0].filename), line=item.traceback[0].lineno,
                    bytes=item.size, allocations=item.count)
               for item in snapshot.statistics('lineno')[:40]]
        result = dict(current_bytes=current, peak_bytes=peak,
            profiler_metadata_bytes=tracemalloc.get_tracemalloc_memory(),
            near_peak_live_bytes=sum(item.size for item in snapshot.statistics('filename')),
            top_live_allocations=top, samples=self.samples,
            scope='Traced live allocations during serial preparation, excluding imports; not cumulative allocation traffic.')
        tracemalloc.stop()
        return result


def child(args):
    started, cpu = time.perf_counter(), time.process_time()
    output = Path(args.output)
    output.mkdir(parents=True)
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    import event_store
    import event_ingest
    import replay_metadata
    replay_metadata.CACHE_DIRECTORY = output/'metadata-cache'
    imports = time.perf_counter()-started
    if args.profile and args.workers > 1:
        def pool(*a, **kw):
            return ProfiledExecutor(*a, profile_output=str(output), **kw)
        event_store.ProcessPoolExecutor = pool
    trace = MemoryTrace() if args.memory else None
    if trace:
        trace.start()
    operation = time.perf_counter()
    with (output/'preparation.log').open('w', encoding='utf-8') as log, redirect_stdout(log):
        if args.stage == 'bootstrap':
            manifest = event_store.prepare_dataset(args.dataset, output/'prepared',
                action_catalog=args.action_catalog, minimum_ram=args.minimum_ram,
                workers=args.workers, placement_mode=args.placement_mode)
        else:
            manifest = event_ingest.prepare_batch(args.dataset, output/'prepared', args.config,
                action_catalog=args.action_catalog, minimum_ram=args.minimum_ram, workers=args.workers)
    operation_wall = time.perf_counter()-operation
    elapsed, process_cpu = time.perf_counter()-started, time.process_time()-cpu
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(output/'parent.pstats'))
    memory = trace.finish() if trace else None
    validation_started = time.perf_counter()
    prepared = payloads(manifest)
    save(output/'result.json', dict(stage=args.stage, workers=args.workers, profiled=args.profile,
        imports_seconds=imports, operation_wall_seconds=operation_wall, wall_seconds=elapsed,
        parent_cpu_seconds=process_cpu, payloads=prepared, memory=memory,
        validation_seconds=time.perf_counter()-validation_started,
        config=json.loads(Path(manifest).read_text())['config']))


def category(filename, name):
    file = filename.replace('\\', '/').lower()
    base = file.rsplit('/', 1)[-1]
    if base in ('generals_policy.py', 'unit_tasks.py', 'build_placement.py'):
        return 'features_targets_masks'
    if base == 'training_legality.py':
        return 'training_legality'
    if base in ('tensor_replay_store.py', 'prepared_io.py'):
        return 'column_packing_and_store_helpers'
    if base in ('replay_actions.py', 'replay_sequences.py', 'replay_events.py', 'generals_identity.py'):
        return 'event_selection_and_action_adaptation'
    if '/json/' in file or 'json' in name.lower():
        return 'json'
    if base in ('gzip.py', '_compression.py', 'zipfile.py') or any(word in name for word in ('compress', 'decompress')):
        return 'compression_and_archive'
    if '/numpy/' in file or 'numpy' in name:
        return 'numpy_other'
    if 'hashlib' in name or 'sha256' in name:
        return 'hashing'
    if base in ('replay_training.py', 'replay_metadata.py', 'replay_loading.py', 'replay_bootstrap.py',
                'audit_replay_commands.py', 'replay_storage.py', 'binary_replay.py'):
        return 'source_reading_and_proofs'
    if 'importlib' in file or '/torch/' in file:
        return 'imports_and_torch'
    if base in ('event_store.py', 'event_ingest.py'):
        return 'preparation_orchestration'
    return 'other'


def summarize(paths, destination):
    if not paths:
        return None
    stats = pstats.Stats(str(paths[0]))
    for path in paths[1:]:
        stats.add(str(path))
    if destination is not None:
        stats.dump_stats(str(destination))
    functions, categories = [], {}
    for (filename, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
        group = category(filename, name)
        categories[group] = categories.get(group, 0)+own
        functions.append(dict(file=filename, line=line, function=name, calls=calls,
            primitive_calls=primitive, self_seconds=own, cumulative_seconds=cumulative))
    return dict(total_calls=stats.total_calls, total_self_seconds=stats.total_tt,
        exclusive_categories_seconds=categories,
        by_self=sorted(functions, key=lambda row: row['self_seconds'], reverse=True)[:80],
        by_cumulative=sorted(functions, key=lambda row: row['cumulative_seconds'], reverse=True)[:100],
        project=sorted((row for row in functions if str(ROOT).lower() in row['file'].lower()),
                       key=lambda row: row['cumulative_seconds'], reverse=True)[:100])


def run_case(args, work, name, *, workers, stage='batch', profile=False, dataset=None, memory=False):
    from pipeline_resources import tree
    output = work/name
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--child', '--stage', stage,
        '--dataset', str(dataset or args.dataset), '--output', str(output), '--config', str(work/'config.json'),
        '--workers', str(workers), '--min-free-ram-mib', str(args.min_free_ram_mib),
        '--action-catalog', str(args.action_catalog), '--placement-mode', args.placement_mode]
    if profile:
        command.append('--profile')
    if memory:
        command.append('--memory')
    samples, stopped = [], threading.Event()
    def sample():
        while not stopped.is_set():
            samples.append(tree())
            stopped.wait(.2)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    finally:
        stopped.set()
        thread.join()
    outer_wall = time.perf_counter()-started
    result = json.loads((output/'result.json').read_text())
    result.update(name=name, subprocess_wall_seconds=outer_wall,
        resource_samples=len(samples),
        peak_tree_private_bytes=max(row.get('aggregate_private_bytes', 0) for row in samples),
        peak_tree_working_bytes=max(row.get('aggregate_working_bytes', 0) for row in samples),
        minimum_available_ram_bytes=min(row.get('available_ram_bytes', 0) for row in samples),
        maximum_process_count=max(row.get('process_count', 0) for row in samples))
    result['jobs'] = [json.loads(path.read_text()) for path in output.glob('job-*.json')]
    result['worker_lifetimes'] = [json.loads(path.read_text()) for path in output.glob('worker-*.json')]
    if profile:
        stem = Path(args.output).with_suffix('')
        result['parent_profile'] = summarize([output/'parent.pstats'], stem.with_name(stem.name+'-'+name+'-parent.pstats'))
        worker_profiles = sorted(output.glob('job-*.pstats'))+sorted(output.glob('initializer-*.pstats'))
        result['worker_profile'] = summarize(worker_profiles, stem.with_name(stem.name+'-'+name+'-workers.pstats'))
        for job in result['jobs']:
            job['profile'] = summarize([output/f'job-{job["pid"]}-{job["ordinal"]}.pstats'], None)
            for key, limit in (('by_self', 20), ('by_cumulative', 20), ('project', 30)):
                job['profile'][key] = job['profile'][key][:limit]
    print(json.dumps(dict(case=name, seconds=result['operation_wall_seconds'],
        cold_seconds=result['wall_seconds'], peak_tree_mib=result['peak_tree_private_bytes']/1024**2)), flush=True)
    return result


def main(args):
    if args.child:
        child(args)
        return
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Choose a new profiling report filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    args.dataset, args.action_catalog, args.output = map(lambda path: str(Path(path).resolve()),
        (args.dataset, args.action_catalog, args.output))
    document = json.loads(Path(args.dataset).read_text())
    report = dict(schema=1, kind='event_preparation_profile', complete=False,
        dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        matches=[entry['match_key'] for entry in document['entries']], python=sys.version,
        platform=platform.platform(), processor=platform.processor(), logical_cpus=os.cpu_count(),
        capability_catalog_sha256=hashlib.sha256(Path(args.action_catalog).read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
            for name in ('profile_event_preparation.py', 'event_store.py', 'event_ingest.py',
                'replay_actions.py', 'replay_events.py', 'replay_sequences.py', 'replay_loading.py',
                'replay_metadata.py', 'tensor_replay_store.py', 'training_legality.py',
                'generals_policy.py', 'unit_tasks.py')},
        minimum_ram_bytes=args.minimum_ram, runs=[],
        notes=['Fresh child process and new prepared output for every run; native metadata receipts already exist.',
               'Parent cProfile includes imports and coordinator waits; worker profiles cover initialization and jobs, excluding idle waits and pre-initializer imports.',
               'Exclusive categories sum self times and are approximate attribution; cumulative times overlap.',
               'Tree memory includes the lightweight driver; tracemalloc uses a separate serial run.'])
    try:
        with tempfile.TemporaryDirectory(prefix='event-preparation-', dir=ROOT/'.cache') as temporary:
            work = Path(temporary)
            bootstrap = run_case(args, work, 'bootstrap', workers=4, stage='bootstrap', profile=True)
            config = bootstrap.pop('config')
            report['config'] = config
            report['runs'].append(bootstrap)
            save(work/'config.json', config)
            expected = {key: value['payload_sha256'] for key, value in bootstrap['payloads'].items()}
            save(output, report)
            cases = [(f'baseline-{workers}-{repeat+1}', workers, False)
                     for repeat in range(args.repeat) for workers in (1, 4)]
            cases += [('cpu-1', 1, True), ('cpu-4', 4, True)]
            for name, workers, profile in cases:
                result = run_case(args, work, name, workers=workers, profile=profile)
                if result.pop('config') != config:
                    raise ValueError('Preparation changed the frozen configuration')
                if {key: value['payload_sha256'] for key, value in result['payloads'].items()} != expected:
                    raise ValueError('Profiling/worker count changed prepared shard payloads')
                report['runs'].append(result)
                save(output, report)
            if not args.skip_memory:
                selected = next((entry for entry in document['entries'] if entry['match_key']==args.memory_match),
                                document['entries'][len(document['entries'])//2])
                save(work/'memory-dataset.json', {**document, 'entries':[selected]})
                result = run_case(args, work, 'memory', workers=1, dataset=work/'memory-dataset.json', memory=True)
                if result.pop('config') != config:
                    raise ValueError('Memory profiling changed the frozen configuration')
                if result['payloads'][selected['match_key']]['payload_sha256'] != expected[selected['match_key']]:
                    raise ValueError('Memory profiler changed prepared data')
                report['runs'].append(result)
                save(output, report)
        medians = {str(workers):statistics.median(row['operation_wall_seconds'] for row in report['runs']
                   if row['name'].startswith(f'baseline-{workers}-')) for workers in (1,4)}
        report.update(complete=True, baseline_operation_medians=medians,
            four_worker_speedup=medians['1']/medians['4'], payload_equivalence=True)
    except BaseException as error:
        report['error'] = repr(error)
        save(output, report)
        raise
    save(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--action-catalog', default=str(ROOT/'replay-runtime/action-catalog.json'))
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--placement-mode', choices=('fast', 'full_mask'), default='fast')
    parser.add_argument('--skip-memory', action='store_true')
    parser.add_argument('--memory-match', default='pgo:4269520')
    parser.add_argument('--min-free-ram-mib', type=int, default=1024)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--stage', choices=('bootstrap','batch'), default='batch', help=argparse.SUPPRESS)
    parser.add_argument('--config', help=argparse.SUPPRESS)
    parser.add_argument('--workers', type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument('--profile', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--memory', action='store_true', help=argparse.SUPPRESS)
    parsed = parser.parse_args()
    if min(parsed.repeat, parsed.workers, parsed.min_free_ram_mib) < 1:
        parser.error('Positive repeat count, worker count and RAM reserve required')
    parsed.minimum_ram = parsed.min_free_ram_mib*1024**2
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    main(parsed)
