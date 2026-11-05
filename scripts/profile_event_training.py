"""Bounded, real-shard profiling of the event learner; never resumes a training run.

CPU profiles and synchronised phase timings are separate from uninstrumented
throughput. This Windows HIP build cannot collect GPU kernel traces; phase
timings include host work/launch overhead and must not be called GPU busy time.
"""
from __future__ import annotations

import argparse
import atexit
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
import cProfile
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import tempfile
import time
import types
from unittest.mock import patch
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parent.parent
for variable, name in (('MIOPEN_USER_DB_PATH', 'user-db'), ('MIOPEN_CUSTOM_CACHE_DIR', 'kernels')):
    cache = ROOT/'.cache/miopen'/name
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(cache))

import numpy as np
from event_resources import snapshot, within_budget
from event_dataset import EventDataset
from generals_policy import NeuralPolicy
from train_events import EventLearner


class ResourceLimit(RuntimeError):
    """A transient resource stop, distinct from corrupt data or a code failure."""


def write(path, document):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(document, indent=2), encoding='utf-8')
    temporary.replace(path)


def confirmation_suite(args):
    """Finish bounded qualification in fresh processes; retry resource stops only."""
    from replay_converter import InstanceLock, is_running
    with InstanceLock(args.output/'confirmation.lock'):
        if is_running():
            raise RuntimeError('Stop replay conversion before isolated learner profiling')
        state = dict(status='running', pid=os.getpid(), results={}, attempts={},
                     dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest())
        paths = [p for p in (ROOT/'scripts').glob('*.py') if not p.name.startswith('test_')]
        pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        state['sources'] = pins
        control = ROOT/'replay-runtime/converter.json'
        initial_control = control.read_bytes() if control.exists() else None
        target = args.output/'confirmation-status.json'

        def check_inputs():
            if (hashlib.sha256(args.dataset.read_bytes()).hexdigest() != state['dataset_sha256']
                    or any(hashlib.sha256(p.read_bytes()).hexdigest() != pins[p.name] for p in paths)):
                raise RuntimeError('Profiling inputs changed; stop and restart qualification')

        try:
            for label, count, split, instrument, checkpoint in (
                    ('final-confirm', 512, 'train', 'none', True),
                    ('final-cpu', 64, 'train', 'cpu', False),
                    ('final-reader', 64, 'train', 'cpu-reader', False),
                    ('final-validation', 128, 'validation', 'none', False)):
                expected = dict(batches=count, warmup=8, prefetch=1, pair_budget=args.pair_budget,
                    batch_size=args.batch_size, buffer_mib=args.buffer_mib, threads=args.threads,
                    device=args.device, seed=args.seed, split=split, instrument=instrument,
                    checkpoint_benchmark=checkpoint, loader_only=False)
                prior_path = args.output/(label+'.json')
                prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
                reusable = (prior.get('dataset_sha256') == state['dataset_sha256']
                    and all(prior.get('settings', {}).get(key) == value for key, value in expected.items())
                    and bool(prior.get('sources'))
                    and all(pins.get(name) == digest for name, digest in prior.get('sources', {}).items())
                    and (not checkpoint or 'checkpoint' in prior))
                while True:
                    check_inputs()
                    if reusable:
                        result = prior
                        state.setdefault('reused', []).append(label)
                        break
                    state.update(status='running', current=label, updated=time.time())
                    state['attempts'][label] = state['attempts'].get(label, 0)+1
                    write(target, state)
                    command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                        '--dataset', str(args.dataset), '--output', str(args.output), '--label', label,
                        '--batches', str(count), '--warmup', '8', '--prefetch', '1',
                        '--pair-budget', str(args.pair_budget), '--batch-size', str(args.batch_size),
                        '--buffer-mib', str(args.buffer_mib), '--threads', str(args.threads),
                        '--device', args.device, '--seed', str(args.seed),
                        '--split', split, '--instrument', instrument]
                    if checkpoint:
                        command.append('--checkpoint-benchmark')
                    completed = subprocess.run(command, cwd=ROOT)
                    if completed.returncode == 75:
                        state.update(status='retrying_resources', updated=time.time())
                        write(target, state)
                        time.sleep(15)
                        continue
                    completed.check_returncode()
                    check_inputs()
                    result = json.loads((args.output/(label+'.json')).read_text())
                    if result['dataset_sha256'] != state['dataset_sha256']:
                        raise RuntimeError('Qualification report uses a different dataset')
                    break
                state['results'][label] = {name: result[name] for name in
                    ('measured_seconds', 'examples_per_second', 'counts', 'resources', 'checkpoint')
                    if name in result}
            if args.restore_converter_workers:
                current_control = control.read_bytes() if control.exists() else None
                if current_control != initial_control or is_running():
                    state['converter_restoration'] = 'skipped_newer_user_control'
                else:
                    converter = [sys.executable, '-B', str(ROOT/'scripts/replay_converter.py')]
                    subprocess.run([*converter, 'set', '--workers', str(args.restore_converter_workers),
                                    '--resume'], cwd=ROOT, check=True)
                    subprocess.run([*converter, 'start'], cwd=ROOT, check=True)
                    state['restored_converter_workers'] = args.restore_converter_workers
            state.update(status='complete', current=None, updated=time.time())
            write(target, state)
        except BaseException as error:
            state.update(status='failed', error=str(error), updated=time.time())
            write(target, state)
            raise


def quantiles(values):
    if not values:
        return {}
    return dict(zip(('min', 'p50', 'p90', 'p99', 'max'),
                    map(float, np.percentile(values, [0, 50, 90, 99, 100]))))


@contextmanager
def module_overrides(modules):
    """Swap archived policy modules without undoing Torch's lazy imports."""
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def original_learner(config, args):
    """Load the cancelled run's actual implementation, without extracting files."""
    modules = {}
    with zipfile.ZipFile(ROOT/'.cache/training-profile/baseline-sources.zip') as archive:
        for name in ('torch_training_inputs', 'torch_policy_v3', 'train_events'):
            module = types.ModuleType('_profile_original_'+name)
            module.__file__ = '<baseline/'+name+'.py>'
            exec(compile(archive.read(name+'.py'), module.__file__, 'exec'), module.__dict__)
            modules[name] = module
    with module_overrides(modules):
        learner = modules['train_events'].EventLearner(NeuralPolicy(config, args.seed),
                    device=args.device, threads=args.threads, pair_budget=2_000_000)
    return learner, modules


def abba(dataset, args, report):
    """Matched whole-pipeline runs, alternating order and amortising verification."""
    import torch
    from event_training_batches import PrefetchedBatches
    # Verify the sampled archives once, just as a long-lived epoch reader does.
    source = dataset.batches(args.split, seed=args.seed, epoch=0, batch_size=args.batch_size,
                             buffer_bytes=args.buffer_mib*1024**2)
    for _ in range(args.warmup+args.batches):
        next(source)
    source.close()
    results = []

    def condition(name):
        modules = {}
        if name == 'original':
            learner, modules = original_learner(dataset.config, args)
        else:
            learner = EventLearner(NeuralPolicy(dataset.config, args.seed),
                device=args.device, threads=args.threads, pair_budget=args.pair_budget)
        source = dataset.batches(args.split, seed=args.seed, epoch=0, batch_size=args.batch_size,
                                 buffer_bytes=args.buffer_mib*1024**2)
        reader = PrefetchedBatches(source) if name == 'candidate' else source
        try:
            with module_overrides(modules):
                for _ in range(args.warmup):
                    learner.batch(next(reader)[0], train=args.split == 'train')
                torch.cuda.synchronize()
                start = time.perf_counter();totals = Counter()
                for _ in range(args.batches):
                    resource = snapshot(args.output, learner.model)
                    if not within_budget(resource, minimum_ram=1024**3):
                        raise ResourceLimit('ABBA profile reached resource boundary: '+json.dumps(resource))
                    totals.update(learner.batch(next(reader)[0], train=args.split == 'train'))
                torch.cuda.synchronize();duration = time.perf_counter()-start
                return dict(condition=name, seconds=duration, examples=totals['decisions'],
                            examples_per_second=totals['decisions']/duration, counts=dict(totals),
                            resources=snapshot(args.output, learner.model))
        finally:
            reader.close()

    for name in ('original', 'candidate', 'candidate', 'original'):
        result = condition(name)
        results.append(result)
        gc.collect()
        report.update(abba=results, status='partial')
        write(args.output/(args.label+'.partial.json'), report)
        print(json.dumps(dict(label=args.label, condition=name, seconds=result['seconds'],
                             examples_per_second=result['examples_per_second'])), flush=True)
    assert len({r['examples'] for r in results}) == 1
    original = np.mean([r['seconds'] for r in results if r['condition'] == 'original'])
    current = np.mean([r['seconds'] for r in results if r['condition'] == 'candidate'])
    report.update(abba=results, abba_speedup=original/current, status='complete',
                  verification_scope='sampled shards verified once before all four timed passes')
    write(args.output/(args.label+'.json'), report)
    print(json.dumps(dict(label=args.label, abba_speedup=report['abba_speedup'])), flush=True)


class Phases:
    def __init__(self, learner):
        self.learner, self.results = learner, defaultdict(lambda: dict(calls=0, wall_seconds=0.0, stream_ms=0.0))
        self.stack = ExitStack()

    def wrap(self, obj, name, label, device=False):
        original = getattr(obj, name)
        torch = self.learner.torch
        device = device and self.learner.model.device.type == 'cuda'

        def measured(*args, **kwargs):
            if device:
                torch.cuda.synchronize()
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin.record()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            if device:
                end.record()
                end.synchronize()
            elapsed = time.perf_counter()-start
            stats = self.results[label]
            stats['calls'] += 1
            stats['wall_seconds'] += elapsed
            if device:
                stats['stream_ms'] += begin.elapsed_time(end)
            return result

        self.stack.enter_context(patch.object(obj, name, measured))

    def __enter__(self):
        import train_events
        import torch_training_inputs
        import training_preparation
        torch, model = self.learner.torch, self.learner.model
        for obj, name, label, device in (
                (model, 'events', 'encoder_and_trunk', True),
                (model, 'teacher_heads', 'argument_heads', True),
                (model.stager, 'stage', 'input_staging_nested', True),
                (training_preparation, 'teacher_plan', 'teacher_cpu_plan_nested', False),
                (train_events, 'event_losses', 'loss', True),
                (torch.Tensor, 'backward', 'backward', True),
                (torch.Tensor, 'cpu', 'readback_nested', False),
                (torch.nn.utils, 'clip_grad_norm_', 'gradient_clip', True),
                (self.learner.optimizer, 'step', 'adam', True),
                (self.learner.optimizer, 'zero_grad', 'zero_grad', False),
                (model, 'regularization_loss', 'regularization', True)):
            self.wrap(obj, name, label, device)
        return self

    def __exit__(self, *args):
        self.stack.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'training-data/.model-training.json')
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/training-profile')
    parser.add_argument('--label', default='baseline')
    parser.add_argument('--batches', type=int, default=128)
    parser.add_argument('--full-epoch', action='store_true',
                        help='Measure every example in the split; warm-up uses a separate iterator')
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--pair-budget', type=int, default=8_000_000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-mib', type=int, default=64)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--encoder-source', type=Path,
                        help='Benchmark torch_policy_v3.py from a local source ZIP')
    parser.add_argument('--verify-updates', action='store_true', help='Compare complete real-data updates with archived original')
    parser.add_argument('--compare', action='store_true', help='Paired learner timings against archived original source')
    parser.add_argument('--prefetch', type=int, default=0)
    parser.add_argument('--checkpoint-benchmark', action='store_true')
    parser.add_argument('--gil-ms', type=float, default=0, help='Experimental Python thread switch interval; 0 keeps default')
    parser.add_argument('--abba', action='store_true', help='Repeated original/current whole-pipeline comparison')
    parser.add_argument('--confirmation-suite', action='store_true', help='Queued final timing, CPU, checkpoint and validation checks')
    parser.add_argument('--background', action='store_true', help='Detach the confirmation suite on Windows')
    parser.add_argument('--restore-converter-workers', type=int, default=0,
                        help='Restore this converter worker limit after successful qualification')
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--instrument', choices=('none', 'cpu', 'cpu-reader', 'phases', 'torch'), default='none')
    parser.add_argument('--loader-only', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    if args.gil_ms:
        if args.gil_ms <= 0:
            parser.error('Positive GIL switch interval required')
        sys.setswitchinterval(args.gil_ms/1000)
    if min(args.batches, args.batch_size, args.buffer_mib, args.threads, args.pair_budget) < 1 or args.warmup < 0:
        parser.error('Positive profile budgets required')
    if args.full_epoch and (args.compare or args.abba or args.verify_updates or args.confirmation_suite):
        parser.error('--full-epoch requires the ordinary profiling path')
    if not 0 <= args.restore_converter_workers <= 64:
        parser.error('Converter workers must be between 0 and 64')
    if (args.background or args.restore_converter_workers) and not args.confirmation_suite:
        parser.error('Background execution/converter restoration require --confirmation-suite')
    from engine_runtime import disable_power_throttling
    disable_power_throttling()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.background:
        if os.name != 'nt':
            parser.error('Detached qualification currently requires Windows')
        arguments = [argument for argument in sys.argv[1:] if argument != '--background']
        with (args.output/'confirmation.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', '-B', str(Path(__file__).resolve()), *arguments],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        print(json.dumps(dict(started_pid=process.pid, status=str(args.output/'confirmation-status.json'))))
        return
    if args.confirmation_suite:
        confirmation_suite(args)
        return
    # Use the existing training admission budget; never load PyTorch into a
    # resource wait or let a transient RAM shortage discard a profiling job.
    while not within_budget(snapshot(args.output), minimum_ram=int(2.5*1024**3)):
        print(json.dumps(dict(label=args.label, status='waiting_for_resources',
                              resources=snapshot(args.output))), flush=True)
        time.sleep(15)
    started = time.perf_counter()
    dataset = EventDataset(args.dataset)
    settings = vars(args).copy()
    settings.update(dataset=str(args.dataset.resolve()), output=str(args.output.resolve()))
    settings['encoder_source'] = str(args.encoder_source.resolve()) if args.encoder_source else None
    settings['gil_switch_seconds'] = sys.getswitchinterval()
    report = dict(settings=settings, dataset_sha256=dataset.digest,
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                      ROOT/'scripts/train_events.py', ROOT/'scripts/torch_policy_v3.py',
                      ROOT/'scripts/torch_training_inputs.py', ROOT/'scripts/event_store.py',
                      ROOT/'scripts/event_training_batches.py', Path(__file__).resolve())},
                  dataset_load_seconds=time.perf_counter()-started)
    if args.abba:
        abba(dataset, args, report)
        return
    learner = None
    if not args.loader_only:
        import torch
        import torch_policy_v3
        if args.encoder_source:
            with zipfile.ZipFile(args.encoder_source) as archive:
                source = archive.read('torch_policy_v3.py')
            encoder = types.ModuleType('profile_encoder_source')
            exec(compile(source, str(args.encoder_source)+'!/torch_policy_v3.py', 'exec'), encoder.__dict__)
            sys.modules['torch_policy_v3'] = torch_policy_v3 = encoder
            report['sources']['torch_policy_v3.py'] = hashlib.sha256(source).hexdigest()
            report['encoder_source_archive_sha256'] = hashlib.sha256(args.encoder_source.read_bytes()).hexdigest()
        torch.manual_seed(args.seed)
        reference = NeuralPolicy(dataset.config, args.seed)
        learner = EventLearner(reference, device=args.device, threads=args.threads, pair_budget=args.pair_budget)
        report.update(torch=torch.__version__, gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                      supported_activities=list(map(str, torch.profiler.supported_activities())))
    reader_profiler = None
    if args.prefetch:
        from event_training_batches import PrefetchedBatches
        # This Python build cannot enable two cProfile instances concurrently.
        # Profile the learner and reader in separate passes.
        reader_profiler = cProfile.Profile() if args.instrument == 'cpu-reader' else None

    def make_iterator(*, measured=True):
        source = iter(dataset.batches(args.split, seed=args.seed, epoch=0, batch_size=args.batch_size,
                                      buffer_bytes=args.buffer_mib*1024**2))
        if args.prefetch:
            source = PrefetchedBatches(source, args.prefetch,
                                       profiler=reader_profiler if measured else None)
        return source

    iterator = make_iterator(measured=not args.full_epoch)
    atexit.register(iterator.close)
    if args.compare:
        original, modules = original_learner(dataset.config, args)
        records = []
        for i in range(args.warmup+args.batches):
            if not within_budget(snapshot(args.output), minimum_ram=1024**3):
                raise ResourceLimit('Paired profiling reached the resource reserve')
            start = time.perf_counter();rows, _ = next(iterator);read_seconds = time.perf_counter()-start
            result = dict(examples=len(rows), read_seconds=read_seconds)
            for label in (('original', 'candidate') if i % 2 else ('candidate', 'original')):
                if args.device == 'cuda':torch.cuda.synchronize()
                start = time.perf_counter()
                if label == 'original':
                    with module_overrides(modules):stats = original.batch(rows, train=args.split == 'train')
                else:stats = learner.batch(rows, train=args.split == 'train')
                if args.device == 'cuda':torch.cuda.synchronize()
                result[label+'_seconds'] = time.perf_counter()-start
                result[label+'_microbatches'] = stats['microbatches']
            if i >= args.warmup:
                records.append(result)
            if len(records) and len(records) % 16 == 0:
                print(json.dumps(dict(label=args.label, paired_batches=len(records))), flush=True)
        samples = sum(r['examples'] for r in records)
        timings = {label:sum(r[label+'_seconds'] for r in records) for label in ('original','candidate')}
        report.update(paired_batches=records, counts=dict(decisions=samples), paired_learner_seconds=timings,
                      paired_learner_speedup=timings['original']/timings['candidate'])
        write(args.output/(args.label+'.json'), report)
        print(json.dumps(dict(label=args.label, examples=samples, **timings,
                             speedup=report['paired_learner_speedup'])), flush=True)
        return
    if args.verify_updates:
        other, modules = original_learner(dataset.config, args)
        differences = []
        for i in range(args.batches):
            rows, _ = next(iterator)
            with module_overrides(modules):left = other.batch(rows, train=True)
            right = learner.batch(rows, train=True)
            original, candidate = other, learner
            for name, value in original.model.weights.items():
                torch.testing.assert_close(candidate.model.weights[name], value, atol=5e-6, rtol=3e-4)
            differences.append(max((candidate.model.weights[name]-value).abs().max().item()
                                   for name, value in original.model.weights.items()))
            assert left['decisions'] == right['decisions']
        write(args.output/(args.label+'.json'), dict(real_data_updates=args.batches,
              max_parameter_differences=differences, dataset_sha256=dataset.digest))
        print(json.dumps(dict(label=args.label, agreement='passed', max_parameter_difference=max(differences))), flush=True)
        return
    warm_start = time.perf_counter()
    for _ in range(args.warmup):
        rows, _ = next(iterator)
        if learner:
            learner.batch(rows, train=args.split == 'train')
    if learner and args.device == 'cuda':
        learner.torch.cuda.synchronize()
        learner.torch.cuda.reset_peak_memory_stats()
    if learner:
        learner.model.stager.stats.update(batches=0, device_copies=0, host_bytes=0)
    report['warmup_seconds'] = time.perf_counter()-warm_start
    if args.full_epoch:
        # The measured pass must include the warm-up prefix again. Reset shard
        # verification too, so its full cost is included in each full epoch.
        iterator.close()
        dataset.verified.clear()
        dataset.shared.clear()
        iterator = make_iterator()
        atexit.register(iterator.close)
        report['expected_examples'] = sum(episode['length'] for entry in dataset.entries
            if entry['split'] == args.split for episode in entry['index']['episodes'])
        if not report['expected_examples']:
            raise ValueError('Cannot profile an empty epoch')
    totals, batches, loading, running, entity_counts = Counter(), [], 0.0, 0.0, []
    profiler, phases, trace = None, None, None
    with ExitStack() as stack:
        if args.instrument == 'cpu':
            profiler = cProfile.Profile()
            profiler.enable()
        elif args.instrument == 'phases':
            phases = stack.enter_context(Phases(learner))
        elif args.instrument == 'torch':
            trace = stack.enter_context(learner.torch.profiler.profile(
                activities=[learner.torch.profiler.ProfilerActivity.CPU], record_shapes=True))
        measurement_start = time.perf_counter()
        for i in itertools.count() if args.full_epoch else range(args.batches):
            resource = snapshot(args.output, learner.model if learner else None)
            if not within_budget(resource, minimum_ram=1024**3):
                report.update(status='resource_limit', completed_batches=i, resources=resource)
                write(args.output/(args.label+'.partial.json'), report)
                raise ResourceLimit('Profiling reached the existing resource boundary: '+json.dumps(resource))
            start = time.perf_counter()
            try:
                rows, cursor = next(iterator)
            except StopIteration:
                if args.full_epoch:
                    break
                raise ValueError('Dataset ended before the requested batch prefix') from None
            loaded = time.perf_counter()-start
            counts = [len(row['encoded']['numeric']) for row in rows]
            entity_counts.extend(counts)
            start = time.perf_counter()
            stats = learner.batch(rows, train=args.split == 'train') if learner else dict(decisions=len(rows))
            duration = time.perf_counter()-start
            totals.update(stats)
            totals['entity_tokens'] += sum(counts)
            totals['unpadded_entity_pairs'] += sum(count*count for count in counts)
            loading += loaded
            running += duration
            batches.append(dict(examples=len(rows), max_entities=max(counts), read_seconds=loaded,
                                learn_seconds=duration, microbatches=stats.get('microbatches'), cursor=cursor))
            if (i+1) % 16 == 0:
                print(json.dumps(dict(label=args.label, batches=i+1, examples=totals['decisions'],
                      seconds=time.perf_counter()-measurement_start)), flush=True)
        if learner and args.device == 'cuda':
            learner.torch.cuda.synchronize()
        wall = time.perf_counter()-measurement_start
        if args.prefetch:
            iterator.close()
        if profiler:
            profiler.disable()
    if args.full_epoch:
        if totals['decisions'] != report['expected_examples']:
            raise ValueError('Full epoch omitted or duplicated examples')
        report.update(full_epoch_complete=True, verified_shards=len(dataset.verified))
    report.update(measured_seconds=wall, loader_seconds=loading, learner_seconds=running,
                  examples_per_second=totals['decisions']/wall, counts=dict(totals),
                  batch_sizes=quantiles([b['examples'] for b in batches]), entities=quantiles(entity_counts),
                  batch_times=quantiles([b['learn_seconds'] for b in batches]), batches=batches,
                  resources=snapshot(args.output, learner.model if learner else None))
    if learner:
        report['staging'] = learner.model.stager.stats
    if phases:
        report['synchronised_phases'] = dict(phases.results)
        report['phase_warning'] = 'Nested totals overlap. Stream elapsed includes host launch/idle time, not GPU kernel busy time.'
    if profiler:
        profiler.dump_stats(str(args.output/(args.label+'.prof')))
        stats = pstats.Stats(profiler)
        report['cpu_cumulative'] = [dict(file=k[0], line=k[1], function=k[2], primitive=v[0], calls=v[1],
            self_seconds=v[2], cumulative_seconds=v[3]) for k, v in sorted(stats.stats.items(), key=lambda item:-item[1][3])[:90]]
        report['cpu_self'] = [dict(file=k[0], line=k[1], function=k[2], calls=v[1],
            self_seconds=v[2]) for k, v in sorted(stats.stats.items(), key=lambda item:-item[1][2])[:60]]
    if reader_profiler:
        reader_profiler.dump_stats(str(args.output/(args.label+'.reader.prof')))
        stats = pstats.Stats(reader_profiler)
        report['reader_cpu_cumulative'] = [dict(file=k[0], line=k[1], function=k[2], calls=v[1],
            self_seconds=v[2], cumulative_seconds=v[3]) for k,v in sorted(stats.stats.items(),
                key=lambda item:-item[1][3])[:70]]
    if trace:
        trace.export_chrome_trace(str(args.output/(args.label+'.trace.json')))
        report['torch_cpu_ops'] = [dict(name=e.key, count=e.count, self_cpu_us=e.self_cpu_time_total,
            total_cpu_us=e.cpu_time_total, input_shapes=e.input_shapes) for e in sorted(
                trace.key_averages(group_by_input_shape=True), key=lambda e:-e.self_cpu_time_total)[:120]]
    if args.checkpoint_benchmark:
        import tracemalloc
        metadata = dict(sources=[entry['index']['source'] for entry in dataset.entries],
                        event_dataset_sha256=dataset.digest, event_settings=settings,
                        event_match_partitions={e['match_key']:e['split'] for e in dataset.entries},
                        event_cursor=dict(epoch=0, position=list(cursor), validations=0),
                        event_round_counts=dict(totals))
        tracemalloc.start()
        with tempfile.TemporaryDirectory(dir=args.output) as temporary:
            path = Path(temporary)/'checkpoint.npz'
            start = time.perf_counter();learner.save(path, metadata)
            report['checkpoint'] = dict(seconds=time.perf_counter()-start, bytes=path.stat().st_size,
                                         traced_peak_bytes=tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    write(args.output/(args.label+'.json'), report)
    print(json.dumps(dict(label=args.label, examples=totals['decisions'], seconds=wall,
                         examples_per_second=report['examples_per_second'], output=str(args.output/(args.label+'.json')))), flush=True)


if __name__ == '__main__':
    try:
        main()
    except ResourceLimit as error:
        print(json.dumps(dict(status='resource_limit', message=str(error))), flush=True)
        sys.exit(75)
