"""Would static shapes and recorded CUDA graphs speed the learner up? Two measurements.

  # Any machine, no GPU: the shapes real updates produce, and what padding them to a
  # small set of fixed shapes would cost in extra GPU work.
  python -B scripts/graph_probe.py shapes --dataset <bundle>/data/dataset.json --shards training-data
  # A GPU: real microbatches run eagerly (as train_events does) against the same
  # microbatches recorded once as a CUDA graph and replayed. No bucketing is needed
  # for this: each microbatch is replayed with its own inputs. The ratio is the
  # ceiling a static-shape learner could reach, before padding costs.
  python -B scripts/graph_probe.py replay --dataset <bundle>/data/dataset.json --precision bf16

Nothing here trains or writes a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

# The fixed-shape scheme `shapes` prices: today's microbatches, with every size rounded up to a bucket.
SPATIAL_STEPS = 8   # Position rows padded to a multiple of an eighth of the batch.


def entity_bucket(count):
    """Entity counts rounded up: steps of 8 to 64, then about 12% apart."""
    bucket = 8
    while bucket < count:
        bucket += 8 if bucket < 64 else max(8, int(bucket*.125)//8*8)
    return bucket


def power_bucket(value):
    return 0 if value <= 0 else 1 << (int(value)-1).bit_length()


def dimensions(plan, rows, entities):
    """The sizes one microbatch's kernels depend on."""
    values = plan['values']
    commands = plan['command_count']
    steps = (max(values['unit_prefixes'][0])+1) if commands and plan.get('unit_tokens') else 0
    return {'rows': rows, 'entities': entities, 'commands': commands, 'unit_steps': steps,
            'unit_tokens': plan.get('unit_tokens', 0) if commands else 0,
            'spatial': len(plan.get('spatial_rows', ())), 'targets': len(plan.get('target_rows', ())),
            'templates': len(plan.get('template_rows', ())), 'angles': len(plan.get('angle_rows', ()))}


def work(d):
    """GPU-work proxies: attention pairs, map-encoder rows, location-head rows, selection LSTM steps."""
    return Counter(pairs=d['rows']*d['entities']**2, map_rows=d['rows'], location_rows=d['spatial'],
                   lstm_steps=d['commands']*d['unit_steps'])


def open_dataset(args):
    from pathlib import Path
    from event_dataset import EventDataset
    dataset = EventDataset(args.dataset)
    if args.shards:  # A workstation bundle lists shards that stay in training-data/.
        dataset.shard_path = lambda entry: Path(args.shards).resolve()/Path(entry['path']).name
    return dataset


def shapes(args):
    from training_preparation import PreparationSpec, microbatches, teacher_plan
    dataset = open_dataset(args)
    spec = PreparationSpec(dataset.config, dataset.key)
    now, fixed, seen, growth = Counter(), Counter(), Counter(), []
    current_shapes, parts_now, parts_fixed, rows_total = set(), 0, 0, 0
    stream = dataset.batches('train', seed=args.seed, epoch=0, batch_size=args.batch_size,
                             buffer_bytes=args.buffer_mib*1024*1024, cursor=(0, 0))
    for index, (rows, _) in enumerate(stream):
        if index == args.updates:
            break
        rows_total += len(rows)

        def plan_of(chunk, entities):
            return teacher_plan(spec, [row['encoded'] for row in chunk], [row['target'] for row in chunk], entities)
        for chunk in microbatches(rows, args.pair_budget):
            entities = max(len(row['encoded']['numeric']) for row in chunk)
            d = dimensions(plan_of(chunk, entities), len(chunk), entities)
            now.update(work(d)); parts_now += 1
            current_shapes.add(tuple(d.values()))
            # The same microbatch at fixed sizes: rows padded to the batch size (a split
            # update's parts to a power of two), every other size rounded up to a bucket.
            size = args.batch_size if 2*len(chunk) > args.batch_size else power_bucket(len(chunk))
            step = max(1, size//SPATIAL_STEPS)
            padded = {**d, 'rows': size, 'entities': entity_bucket(entities), 'commands': size,
                      'unit_steps': power_bucket(d['unit_steps']), 'spatial': -(-d['spatial']//step)*step}
            fixed.update(work(padded)); parts_fixed += 1
            seen[(size, padded['entities'], padded['unit_steps'], padded['spatial'])] += 1
        growth.append(len(seen))
    updates = len(growth)
    covered = sorted(seen.values(), reverse=True)
    report = {'updates': updates, 'mean_rows': rows_total/max(1, updates),
              'microbatches_per_update': {'current': parts_now/updates, 'fixed': parts_fixed/updates},
              'distinct_shapes': {'current': len(current_shapes), 'fixed': len(seen)},
              'fixed_shapes_seen_after': {str(n): growth[n-1] for n in (10, 25, 50, 100, 200, 400) if n <= updates},
              'fixed_shapes_covering': {f'{int(share*100)}%': next(i+1 for i in range(len(covered))
                                                                 if sum(covered[:i+1]) >= share*sum(covered))
                                        for share in (.5, .9, .99)},
              'work_ratio_fixed_over_current': {name: fixed[name]/max(1, now[name]) for name in now}}
    print(json.dumps(report, indent=1))


def replay(args):
    import torch
    import train_events
    from generals_policy import NeuralPolicy
    from training_preparation import PreparationSpec
    from event_training_batches import prepared as prepared_batches
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dataset = open_dataset(args)
    learner = train_events.EventLearner(NeuralPolicy(dataset.config, args.seed), device='cuda', threads=1,
                                        pair_budget=args.pair_budget, precision=args.precision,
                                        compile_encoder=args.compile)
    model = learner.model
    spec = PreparationSpec(dataset.config, dataset.key, float_dtype=model.float_dtype, device_type=model.device_type)
    stream = prepared_batches(dataset, 'train', spec=spec, pair_budget=args.pair_budget, seed=args.seed, epoch=0,
                              batch_size=args.batch_size, buffer_bytes=args.buffer_mib*1024*1024, cursor=(0, 0))
    parts = []
    for index, (update, _) in enumerate(stream):
        if index == args.updates:
            break
        parts.extend(update['microbatches'])
    stream.close()
    stage = model.stager.stage
    for part in parts:
        part['staged_values'] = stage(part['values'])
        part['plan']['staged'] = stage(part['plan']['values'])
    torch.cuda.synchronize()

    # event_losses uploads its per-row factors from the host; a recorded graph must not
    # contain that copy, so each microbatch's uploads are made once and reused.
    uploads = {}

    def step(index):
        part, calls = parts[index], uploads.setdefault(index, [])
        cursor = iter(list(calls))

        def reuse(values):
            cached = next(cursor, None)
            if cached is None:
                cached = stage(values)
                calls.append(cached)
            return cached
        model.stager.stage = reuse
        try:
            with torch.autocast('cuda', dtype=learner.autocast or torch.float16, enabled=learner.autocast is not None):
                trunk = model.events(prepared=part)
            hidden, entities, spatial = (value.float() for value in trunk)
            output = model.teacher_heads(None, [row['target'] for row in part['rows']], hidden, entities, spatial,
                                         loss_only=True, plan=part['plan'])
            losses = train_events.event_losses(model, part['rows'], output)
            (losses.sum()/args.batch_size).backward()
            return losses.detach()
        finally:
            model.stager.stage = stage

    def zero():
        for parameter in learner.parameter_list:
            if parameter.grad is not None:
                parameter.grad.zero_()

    def gradients():
        return [parameter.grad.clone() for parameter in learner.parameter_list if parameter.grad is not None]

    for index in range(len(parts)):  # Warm-up: lazy initialisation, convolution choices, compilation, .grad buffers.
        for _ in range(3):
            step(index)
    torch.cuda.synchronize()
    graphs, failed, worst = {}, Counter(), 0.0
    pool = torch.cuda.graph_pool_handle()
    for index in range(len(parts)):
        zero(); eager_loss = step(index); eager = gradients()
        zero()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, pool=pool):
                recorded_loss = step(index)
        except Exception as error:  # noqa: BLE001 - which operations refuse recording is the finding
            failed[f'{type(error).__name__}: {str(error)[:200]}'] += 1
            torch.cuda.synchronize()
            continue
        zero(); graph.replay(); torch.cuda.synchronize()
        scale = max(float(g.abs().max()) for g in eager)
        worst = max(worst, max(float((a-b).abs().max()) for a, b in zip(eager, gradients()))/scale,
                    float((eager_loss-recorded_loss).abs().max()/eager_loss.abs().max()))
        graphs[index] = graph
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'precision': args.precision,
              'compile': args.compile, 'microbatches': len(parts), 'recorded': len(graphs), 'failures': dict(failed),
              'largest_relative_difference': worst}
    if graphs:
        order = list(graphs)

        def timed(run):
            best = float('inf')
            for _ in range(args.repeats):
                zero(); torch.cuda.synchronize()
                start = time.perf_counter()
                for index in order:
                    run(index)
                torch.cuda.synchronize()
                best = min(best, time.perf_counter()-start)
            return best
        eager_seconds = timed(step)
        graph_seconds = timed(lambda index: graphs[index].replay())
        rows = sum(len(parts[index]['rows']) for index in order)
        report.update(eager_ms_per_microbatch=1000*eager_seconds/len(order),
                      graph_ms_per_microbatch=1000*graph_seconds/len(order),
                      eager_rows_per_second=rows/eager_seconds, graph_rows_per_second=rows/graph_seconds,
                      speedup=eager_seconds/graph_seconds)
    print(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('shapes', 'replay'):
        command = sub.add_parser(name)
        command.add_argument('--dataset', required=True)
        command.add_argument('--shards', help='Directory holding the shards, when not beside the dataset')
        command.add_argument('--updates', type=int, default=200 if name == 'shapes' else 6)
        command.add_argument('--batch-size', type=int, default=256)
        command.add_argument('--pair-budget', type=int, default=8_000_000)
        command.add_argument('--buffer-mib', type=int, default=512)
        command.add_argument('--seed', type=int, default=123)
    command.add_argument('--precision', default='bf16')
    command.add_argument('--compile', action='store_true', help='Eager side as train.sh runs it (NVIDIA only)')
    command.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    {'shapes': shapes, 'replay': replay}[args.command](args)


if __name__ == '__main__':
    main()
