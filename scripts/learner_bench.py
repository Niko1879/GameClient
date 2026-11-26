"""Learner throughput benchmark for a cloud GPU, and the small real-data bundle it runs on.

  # Workstation: a bench bundle from a checkpoint's recorded split. Catalogue
  # reads are primary-key/indexed lookups only, so the converter can keep
  # running; shards stay in training-data/ (cloud_upload.py streams them).
  python -B scripts/learner_bench.py bundle --checkpoint .cache/cloud-runs/v9-10k/best.npz \\
      --training 600 --output .cache/cloud-bundles/bench-v9 --baseline fd3eeed36
  # Instance, after setup.sh: the whole matrix (bench.sh), or one condition.
  .venv/bin/python -B code/learner_bench.py run --dataset data/dataset.json --precision bf16 --compile
  # A checkpoint's match split, for cloud_bundle.py --validation-from.
  python -B scripts/learner_bench.py splits --checkpoint .cache/cloud-runs/v9-10k/best.npz --output split.json

`run --code DIR` imports the learner from DIR (sources with the same lane API)
instead of this directory, so two versions are timed on the same instance, data
and settings. Updates are the ones train.sh runs: the same lane sampler, readers,
prefetch and learner; nothing is checkpointed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

ROOT = Path(__file__).resolve().parents[1]
LEARNER = ('train_events.py', 'torch_training_inputs.py', 'torch_policy_v3.py', 'training_preparation.py',
           'event_training_batches.py', 'generals_policy.py')

BENCH = """#!/usr/bin/env bash
# Learner timing for the AlphaGenerals sequence learner: lanes x unroll decisions per
# update (train_events --lanes/--unroll), train.sh's other settings. Results land in
# results/; results/summary.txt compares them. About 20-30 minutes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
export TORCHINDUCTOR_COMPILE_THREADS=1  # One ~0.5 GB compile worker per CPU thread otherwise (11 Nov).
# Readers each load the dataset index (~3.3 GB at 112.7k matches): cap by the memory limit.
MEM_LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
[ "$MEM_LIMIT" = max ] && MEM_LIMIT=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) * 1024 ))
MEM_READERS=$(( (MEM_LIMIT / 1048576 - 12288) / 3584 ))
CPU_READERS=$(( $(nproc) >= 12 ? 8 : ($(nproc) > 4 ? 4 : $(nproc) - 1) ))
READERS=${READERS:-$(( MEM_READERS < CPU_READERS ? (MEM_READERS > 1 ? MEM_READERS : 1) : CPU_READERS ))}
UPDATES=${UPDATES:-40}
# Compilation recompiles for new shapes during the first updates; an update is 15,360 decisions.
WARMUP=${WARMUP:-20}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee results/gpu.txt
lscpu | grep -E 'Model name|^CPU\\(s\\)|MHz' | tee results/cpu.txt
run() {
  label=$1; shift
  .venv/bin/python -B code/learner_bench.py run --dataset data/dataset.json --readers "$READERS" --prefetch 2 \\
    --precision bf16 --updates "$UPDATES" --label "$label" --output "results/$label.json" "$@" \\
    > "results/$label.log" 2>&1 && tail -1 "results/$label.log" || echo "$label failed: see results/$label.log"
}
run eager --warmup "$WARMUP"
run compile --compile --warmup "$WARMUP"
# Half the lanes: the update's fixed costs (the LSTM's 64 sequential steps) against its size.
run compile-lanes120 --compile --lanes 120 --warmup "$WARMUP"
run compile-repeat --compile --warmup "$WARMUP"
.venv/bin/python -B code/learner_bench.py summary results/*.json | tee results/summary.txt
"""


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def partitions(checkpoint):
    """(match key -> split in the dataset's own order, training metadata) of a checkpoint."""
    import numpy as np
    with np.load(checkpoint, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))['metadata']
    return metadata['event_match_partitions'], metadata


def splits(args):
    split, metadata = partitions(args.checkpoint)
    document = {'kind': 'generals_event_split', 'checkpoint': str(Path(args.checkpoint).resolve()),
                'event_dataset_sha256': metadata['event_dataset_sha256'],
                'entries': [{'match_key': key, 'split': value} for key, value in split.items()]}
    output = Path(args.output)
    if output.exists():
        raise ValueError('Choose a new split file')
    output.write_text(json.dumps(document, indent=1)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({'output': str(output), **Counter(split.values())}))


def bundle(args):
    import sqlite3
    import zipfile
    import run_prepared_training as experiment
    from cloud_bundle import SETUP, check_shards
    from event_dataset import EventDataset
    from event_store import feature_identity
    from generals_policy import NeuralPolicy
    from replay_catalog import connect
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Choose a new bundle directory')
    split, _ = partitions(args.checkpoint)
    keys = ([key for key, value in split.items() if value == 'train'][:args.training]
            + [key for key, value in split.items() if value == 'validation'][:args.validation])
    policy = experiment.load(ROOT/'replay-runtime/policy.json')
    model = NeuralPolicy(policy, parameters=False)
    key, engine = feature_identity(model), experiment.digest(experiment.ENGINE)
    baseline = subprocess.run(['git', 'rev-parse', '--verify', args.baseline+'^{commit}'], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    db = connect(experiment.CATALOG)
    db.row_factory = sqlite3.Row
    try:
        selected = []
        for match_key in keys:
            # prepared_shard_replay indexes (match_key, feature_key).
            rows = [dict(row) for row in db.execute('SELECT * FROM prepared_shard WHERE match_key=? AND feature_key=?',
                                                    (match_key, key)) if row['available'] and row['engine_sha256'] == engine]
            if len(rows) != 1:
                raise ValueError(f'{match_key}: {len(rows)} available shards with this feature key and engine')
            selected.append({**rows[0], 'split': split[match_key]})
        doc = experiment.document(db, selected, policy, model)
    finally:
        db.close()
    doc['selection'] = {'source_checkpoint': str(Path(args.checkpoint).resolve()), 'training': args.training,
                        'validation': args.validation, 'feature_key': key, 'purpose': 'learner benchmark'}
    (output/'data').mkdir(parents=True)
    (output/'code').mkdir()
    experiment.save(output/'data/dataset.json', doc)
    EventDataset(output/'data/dataset.json')  # The manifest's own checks.
    # Shards stay in training-data/; cloud_upload.py streams them from there.
    shard_bytes = check_shards(doc['entries'], experiment.DATA)
    for script in sorted((ROOT/'scripts').glob('*.py')):
        (output/'code'/script.name).write_bytes(script.read_bytes())
    archive = subprocess.run(['git', 'archive', '--format=zip', baseline, 'scripts'], cwd=ROOT, check=True,
                             capture_output=True).stdout
    (output/'baseline-code.zip').write_bytes(archive)
    with zipfile.ZipFile(output/'baseline-code.zip') as check:
        if 'scripts/train_events.py' not in check.namelist():
            raise ValueError('Baseline archive has no learner')
    for name, text in (('setup.sh', SETUP), ('bench.sh', BENCH)):
        (output/name).write_text(text, encoding='utf-8', newline='\n')
    summary = {'kind': 'generals_learner_bench_bundle', 'created': time.time(), 'baseline': baseline,
               'feature_key': key, 'training': args.training, 'validation': args.validation,
               'dataset_sha256': digest(output/'data/dataset.json'),
               'shard_source': str(experiment.DATA), 'shard_bytes': shard_bytes}
    (output/'bundle.json').write_text(json.dumps(summary)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(summary))


def model_widths(config, widths=None):
    """Model-only config overrides (event_store.model_only): a cloud_bundle preset name or JSON
    object. Shards and the feature key are unchanged. AlphaGenerals' network widths
    (generals_policy.WIDTHS) are fixed: changing them would change the feature key."""
    from event_store import model_only
    if widths is None:
        return {}
    from cloud_bundle import WIDTHS
    chosen = WIDTHS[widths] if widths in WIDTHS else json.loads(widths)
    if not isinstance(chosen, dict) or not model_only(chosen):
        raise ValueError('Widths must be model-only settings')
    return chosen


def cpu_name():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


class GpuSampler:
    """nvidia-smi's GPU busy percentage (share of time a kernel ran), sampled through the timed updates.

    Below ~85% busy the GPU waits on the host: the learner is CPU-bound and a
    faster single core, not a faster GPU, raises throughput.
    """
    BOUND = 85

    def __init__(self, command):
        import threading
        self.samples = []
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.process = None
            return
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    @classmethod
    def nvidia(cls, index):
        return cls(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits',
                    '-i', str(index), '-lms', '250'])

    def _read(self):
        for line in self.process.stdout:
            try:
                self.samples.append(float(line.strip()))
            except ValueError:
                pass

    def stop(self):
        """{mean, p10, p50, p90, samples, cpu_bound} or None without nvidia-smi or samples."""
        if self.process is None:
            return None
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(timeout=10)
        self.process.stdout.close()
        values = sorted(self.samples)
        if not values:
            return None
        def quantile(q):
            return values[min(len(values)-1, int(q*len(values)))]
        mean = sum(values)/len(values)
        return {'mean': mean, 'p10': quantile(.1), 'p50': quantile(.5), 'p90': quantile(.9),
                'samples': len(values), 'cpu_bound': mean < self.BOUND}


def launches(profiler, updates):
    """Kernel launches, copies and synchronisations per update, from the CPU-side runtime calls."""
    counts = Counter()
    for event in profiler.key_averages():
        name = event.key
        if 'LaunchKernel' in name:
            counts['kernel_launches'] += event.count
        elif 'Memcpy' in name:
            counts['memcpy_calls'] += event.count
        elif 'Synchronize' in name:
            counts['synchronize_calls'] += event.count
    return {name: value/updates for name, value in counts.items()}


class _Solo:
    """One process (also for --code baselines, whose train_events has no rank support)."""
    rank, world, share = 0, 1, None

    def barrier(self):
        pass

    def merge(self, counts):
        return counts


def run(args, ranks=None):
    """Time updates; with ranks (--world-size), every rank trains its share and rank 0 reports the total."""
    if args.code:
        sys.path.insert(0, str(Path(args.code).resolve()))
    import numpy as np
    import torch
    import train_events
    from event_dataset import EventDataset
    from event_training_batches import PreparedReader, prepared as prepared_batches
    from generals_policy import NeuralPolicy
    from training_preparation import PreparationSpec
    source = Path(train_events.__file__).resolve().parent
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = EventDataset(args.dataset)
    widths = model_widths(dataset.config, args.widths)
    reference = NeuralPolicy({**dataset.config, **widths}, args.seed)
    parameters = sum(value.data.size for value in reference.parameters.values())
    ranks = ranks or _Solo()
    share = {'share': ranks.share} if ranks.share else {}
    learner = train_events.EventLearner(reference, device=args.device, threads=1,
                                        pair_budget=args.pair_budget, precision=args.precision,
                                        compile_encoder=args.compile, **({'ranks': ranks} if ranks.share else {}))
    spec = PreparationSpec(dataset.config, dataset.key, float_dtype=learner.model.float_dtype,
                           device_type=learner.model.device_type)
    cuda = learner.model.device.type == 'cuda'
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    reader = (PreparedReader(dataset, spec, args.pair_budget, args.prefetch, args.readers, *share.values())
              if args.prefetch else None)
    stream = prepared_batches(dataset, 'train', spec=spec, pair_budget=args.pair_budget, reader=reader, seed=args.seed,
                              epoch=0, lanes=args.lanes, steps=args.unroll, part_lanes=args.part_lanes,
                              cache_bytes=args.cache_mib*1024*1024, cursor=0, **share)
    if hasattr(learner, 'staged'):  # Older --code baselines upload on the main thread.
        stream = learner.staged(stream)
    measured, waits, losses, profile, busy, sampler = Counter(), [], [], None, None, None
    # NVIDIA only: the ROCm build reports device type 'cuda' too, without nvidia-smi.
    nvidia = cuda and torch.version.hip is None
    total = args.warmup+args.updates+args.profile
    try:
        updates = iter(stream)
        for index in range(total):
            if index == args.warmup:
                sync(); ranks.barrier()  # Ranks start the timed updates together.
                start, waits = time.perf_counter(), []
                if nvidia:
                    sampler = GpuSampler.nvidia(torch.cuda.current_device())
            if args.profile and index == args.warmup+args.updates:
                sync(); ranks.barrier()
                elapsed = time.perf_counter()-start
                if sampler:
                    busy, sampler = sampler.stop(), None
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]
                                                  + ([torch.profiler.ProfilerActivity.CUDA] if cuda else []))
                profiler.__enter__()
            began = time.perf_counter()
            try:
                inputs, _ = next(updates)
            except StopIteration:
                raise ValueError(f'The dataset ran out after {index} updates; bundle more matches') from None
            waits.append(time.perf_counter()-began)
            totals = learner.run(inputs, train=True)
            losses.append(totals['loss_sum']/max(1, totals['decisions']))
            if args.warmup <= index < args.warmup+args.updates:
                measured.update(totals)
        sync(); ranks.barrier()
        if args.profile:
            profiler.__exit__(None, None, None)
            profile = launches(profiler, args.profile)
        else:
            elapsed = time.perf_counter()-start
            if sampler:
                busy, sampler = sampler.stop(), None
    finally:
        if sampler:
            sampler.stop()
        stream.close()
        if reader is not None:
            reader.close()
    measured_waits = waits[:args.updates]
    measured = ranks.merge(measured)  # Every rank's decisions; one update per step.
    if ranks.rank:
        return None
    report = {'label': args.label or source.name, 'code': str(source),
              'code_sha256': {name: digest(source/name) for name in LEARNER if (source/name).exists()},
              'settings': {k: v for k, v in vars(args).items() if k not in ('command', 'output', 'label')},
              'updates': measured['updates'], 'decisions': measured['decisions'], 'seconds': elapsed,
              'decisions_per_second': measured['decisions']/elapsed, 'updates_per_second': measured['updates']/elapsed,
              'reader_wait_share': sum(measured_waits)/elapsed,
              'mean_loss': measured['loss_sum']/max(1, measured['decisions']), 'update_losses': losses,
              'skipped_updates': measured['skipped_updates'], 'kernels': profile, 'gpu_busy': busy,
              'max_gpu_mib': torch.cuda.max_memory_allocated()/2**20 if cuda else None,
              'widths': widths, 'parameters': parameters,
              'world_size': ranks.world, 'ranks_per_gpu': getattr(args, 'ranks_per_gpu', 1),
              'gpu': torch.cuda.get_device_name() if cuda else None, 'cpu': cpu_name(),
              'torch': torch.__version__, 'python': platform.python_version()}
    if args.output:
        Path(args.output).write_text(json.dumps(report)+'\n', encoding='utf-8', newline='\n')
    brief = {k: report[k] for k in ('label', 'parameters', 'decisions_per_second', 'updates_per_second', 'reader_wait_share',
                                    'mean_loss', 'max_gpu_mib', 'gpu_busy', 'kernels')}
    print(json.dumps(brief), flush=True)
    return report


def summary(args):
    reports = [json.loads(Path(path).read_text(encoding='utf-8')) for path in args.reports]
    timed = {r['label']: r for r in reports if r['updates']}
    # The scaling matrix (SCALE=1) is relative to its own one-GPU runs.
    prefix = 'baseline-eager' if any(label.startswith('baseline-eager') for label in timed) else 'scale-w1-b256'
    base = [r['decisions_per_second'] for label, r in timed.items() if label.startswith(prefix)]
    reference = sum(base)/len(base) if base else None
    print(f"{'label':28s} {'dec/s':>8s} {'x base':>7s} {'upd/s':>6s} {'wait':>5s} {'GPU MiB':>8s} "
          f"{'busy%':>6s} {'p10-p90':>8s} {'loss':>8s}  kernels/update")
    for label, r in timed.items():
        ratio = f"{r['decisions_per_second']/reference:7.2f}" if reference else '      -'
        gpu = f"{r['max_gpu_mib']:8.0f}" if r['max_gpu_mib'] is not None else '       -'
        use = r.get('gpu_busy')
        busy = f"{use['mean']:6.0f} {use['p10']:>3.0f}-{use['p90']:<4.0f}" if use else '     -        -'
        kernels = ', '.join(f'{k} {v:.0f}' for k, v in (r['kernels'] or {}).items())
        print(f"{label:28s} {r['decisions_per_second']:8.0f} {ratio} {r['updates_per_second']:6.2f} "
              f"{r['reader_wait_share']:5.2f} {gpu} {busy} {r['mean_loss']:8.4f}  {kernels}")
    for label, r in timed.items():
        if r.get('parameters'):
            print(f"{label}: {r['parameters']/1e6:.1f}M parameters, widths {r.get('widths') or 'dataset default'}")
    # The verdict per run: a GPU idle for long stretches is waiting on the host.
    for label, r in timed.items():
        use = r.get('gpu_busy')
        if use:
            verdict = ('CPU-bound: a faster single core raises throughput, more cores do not'
                       if use['cpu_bound'] else 'GPU-bound: host CPU speed matters little')
            print(f"{label}: GPU busy {use['mean']:.0f}% over {use['samples']} samples -> {verdict}")
    # Same seed, data and initial weights: equivalent code gives the same first
    # losses to rounding. Later updates drift apart even between two runs of the
    # same code (GPU atomics in index_add/scatter_add backward), so the drift of
    # the baseline against itself is printed for scale.
    pairs = [(a, b) for a, b in (('baseline-eager', 'current-eager'), ('baseline-compile', 'current-compile'),
                                 ('baseline-eager', 'baseline-eager-2')) if a in timed and b in timed]
    for a, b in pairs:
        left, right = timed[a]['update_losses'], timed[b]['update_losses']
        first = max(abs(x-y)/max(abs(x), 1e-9) for x, y in zip(left[:2], right[:2]))
        print(f'{a} vs {b}: first 2 updates differ by at most {first:.2e} (relative); '
              f'mean loss over the timed updates {timed[a]["mean_loss"]:.4f} vs {timed[b]["mean_loss"]:.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    split = sub.add_parser('splits', help="Write a checkpoint's match split (for cloud_bundle --validation-from)")
    split.add_argument('--checkpoint', required=True)
    split.add_argument('--output', required=True)
    pack = sub.add_parser('bundle', help='Bench bundle from a checkpoint split; live-catalogue safe')
    pack.add_argument('--checkpoint', required=True)
    pack.add_argument('--output', required=True)
    pack.add_argument('--training', type=int, default=600)
    pack.add_argument('--validation', type=int, default=0)
    pack.add_argument('--baseline', required=True, help='Git commit whose scripts/ are the timing baseline')
    timing = sub.add_parser('run', help='Time steady-state training updates')
    timing.add_argument('--dataset', required=True)
    timing.add_argument('--code', help='Import the learner from this directory instead')
    timing.add_argument('--label')
    timing.add_argument('--output')
    timing.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    timing.add_argument('--precision', default='bf16')
    timing.add_argument('--compile', action='store_true')
    timing.add_argument('--warmup', type=int, default=60)
    timing.add_argument('--updates', type=int, default=400)
    timing.add_argument('--profile', type=int, default=0, help='Updates to profile after the timed ones')
    timing.add_argument('--lanes', type=int, default=240, help='Episodes per update (train_events --lanes)')
    timing.add_argument('--unroll', type=int, default=64, help='Decisions per lane and update')
    timing.add_argument('--part-lanes', type=int, default=8, help='Lanes per reader part')
    timing.add_argument('--widths', help="Model-only widths: cloud_bundle preset ('default', 'big') or JSON")
    timing.add_argument('--pair-budget', type=int, default=8_000_000)
    timing.add_argument('--cache-mib', type=int, default=256)
    timing.add_argument('--prefetch', type=int, default=2)
    timing.add_argument('--readers', type=int, default=4)
    timing.add_argument('--seed', type=int, default=123)
    timing.add_argument('--world-size', type=int, default=1, help='Data-parallel ranks (train_events --world-size)')
    timing.add_argument('--ranks-per-gpu', type=int, default=1,
                        help='Ranks sharing each GPU (gloo gradient sums): tests a CPU-bound learner on one card')
    report = sub.add_parser('summary', help='Compare run reports')
    report.add_argument('reports', nargs='+')
    args = parser.parse_args()
    if args.command == 'run' and args.world_size > 1:
        if args.code:
            parser.error('--world-size times this directory\'s learner; --code is for single-process baselines')
        from train_events import spawn_ranks
        spawn_ranks(run, args, args.world_size, args.ranks_per_gpu)
        return
    {'splits': splits, 'bundle': bundle, 'run': run, 'summary': summary}[args.command](args)


if __name__ == '__main__':
    main()
