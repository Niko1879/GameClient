"""Score training decisions with a checkpoint and report extreme-loss decisions.

Runs the trainer's own validation pass (EventLearner.batch, train=False) and
appends every decision above --threshold (train_events outliers: match, frame,
per-head loss terms, target) to --output. Shards are read in place; nothing is
written next to them. Two ways to choose the decisions:

  # a seeded random sample of training matches
  python -B scripts/scan_outliers.py --checkpoint .cache/cloud-runs/v9-all/best.npz \\
      --manifest .cache/cloud-bundles/v9-all/data/dataset.json --shards training-data \\
      --min-rating 1400 --matches 2000 --output .cache/outlier-scan/v9-all.jsonl
  # exactly the shuffle windows one training round used (train_events' seed, buffer and
  # validation marks), filtered to a game-time range and/or the acting player's rating band
  python -B scripts/scan_outliers.py ... --epoch 2 --segment 1 --minutes 30: --rating-band 1700+
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import random
import time

_WORKER = {}


def sample(entries, *, min_rating, matches, seed, keys=None):
    """Training entries at or above min_rating (mean of rated players) and in keys, in seeded random order."""
    from subset_manifest import match_rating
    pool = [entry for entry in entries if entry['split'] == 'train' and (keys is None or entry['match_key'] in keys)
            and (min_rating is None or ((rating := match_rating(entry)) is not None and rating >= min_rating))]
    random.Random(seed).shuffle(pool)
    return pool[:matches]


def minutes_range(text):
    """'30:' -> (30, None); '5:10' -> (5, 10); None -> (None, None)."""
    if not text:
        return None, None
    low, _, high = text.partition(':')
    return (float(low) if low else None), (float(high) if high else None)


def keep(row, minutes, band):
    """Whether a row falls in the game-time range [low, high) and the acting player's rating band."""
    from train_events import rating_band
    low, high = minutes
    if low is not None or high is not None:
        if row.get('frame') is None:
            return False
        minute = row['frame']/row.get('source_ticks_per_second', 60)/60
        if (low is not None and minute < low) or (high is not None and minute >= high):
            return False
    return band is None or rating_band(row.get('rating')) == band


def round_windows(dataset, *, seed, epoch, segment, validations, buffer_bytes):
    """The shuffle windows of one training round: segment k of epoch (0-based), as train_events splits it."""
    from train_events import validation_marks
    windows = dataset.windows('train', seed=seed, epoch=epoch, buffer_bytes=buffer_bytes)
    bounds = [0]+validation_marks(dataset, seed, epoch, buffer_bytes, validations)+[len(windows)]
    if not 0 <= segment < len(bounds)-1:
        raise ValueError(f'Epoch has {len(bounds)-1} rounds')
    return windows[bounds[segment]:bounds[segment+1]]


def _start_worker(shards, key, minutes, band):
    from engine_runtime import disable_power_throttling
    disable_power_throttling()
    from event_dataset import EventDataset
    dataset = object.__new__(EventDataset)
    dataset.path = Path(shards).resolve()/'.outlier-scan.json'
    dataset.verified, dataset.shared, dataset.key = {}, {}, key
    _WORKER.update(dataset=dataset, minutes=minutes, band=band)


def _read_window(task):
    """(rows of one window kept by the filters, rows read)."""
    references, entries = task
    dataset = _WORKER['dataset']
    dataset.entries = entries  # {source: entry}: read() indexes entries by source.
    kept, total = [], 0
    for reference in references:
        rows = dataset.read(reference)
        total += len(rows)
        kept.extend(row for row in rows if keep(row, _WORKER['minutes'], _WORKER['band']))
    return kept, total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True, help='A dataset.json naming the shards (a cloud bundle\'s)')
    parser.add_argument('--shards', required=True, help='Directory holding the shard files')
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-rating', type=float, help='Sampling: mean match rating at least this')
    parser.add_argument('--matches', type=int, default=1000)
    parser.add_argument('--keys', help='Sampling: JSON list of match keys to draw from')
    parser.add_argument('--seed', type=int, default=1, help='Sampling seed')
    parser.add_argument('--epoch', type=int, help='Replay: 0-based training epoch')
    parser.add_argument('--segment', type=int, help='Replay: 0-based round within the epoch')
    parser.add_argument('--train-seed', type=int, default=123)
    parser.add_argument('--validations', type=int, default=4, help='The run\'s --validations-per-epoch')
    parser.add_argument('--buffer-mib', type=int, default=64, help='The run\'s --buffer-mib')
    parser.add_argument('--minutes', help='Keep decisions in this game-time range, e.g. 30: or 5:10')
    parser.add_argument('--rating-band', help='Keep the acting player\'s rating band, e.g. 1700+')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--threshold', type=float, default=20.0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--precision', default='bf16', help='bf16 matches the cloud training run (CUDA only)')
    parser.add_argument('--threads', type=int, default=1, help='CPU threads for --device cpu')
    parser.add_argument('--low-priority', action='store_true', help='Below-normal CPU priority: yields to other work')
    args = parser.parse_args()
    from engine_runtime import disable_power_throttling
    disable_power_throttling()
    if args.low_priority and os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    os.environ.setdefault('MIOPEN_FIND_MODE', 'FAST')
    import train_events
    from event_dataset import EventDataset
    from generals_policy import NeuralPolicy

    train_events.OUTLIER_NLL = args.threshold
    minutes = minutes_range(args.minutes)
    document = json.loads(Path(args.manifest).read_text())
    key = document['entries'][0]['index']['legality_key']
    policy, _, _ = NeuralPolicy.load(args.checkpoint)
    if policy.config != document['config']:
        raise ValueError('Checkpoint and manifest have different model or feature configurations')
    # The trainer's reader over shards in place (as validate_shards does): no manifest file is written.
    dataset = object.__new__(EventDataset)
    dataset.path = Path(args.shards).resolve()/'.outlier-scan.json'
    dataset.verified, dataset.shared, dataset.key = {}, {}, key
    learner = train_events.EventLearner(policy, device=args.device, threads=args.threads,
                                        precision=args.precision if args.device == 'cuda' else 'fp32')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    totals, started, pending = {'units': 0, 'read': 0, 'decisions': 0, 'loss_sum': 0.0, 'outliers': 0}, time.monotonic(), []

    def flush():
        counts = learner.batch(pending, train=False)
        totals['decisions'] += counts['decisions']
        totals['loss_sum'] += counts['loss_sum']
        totals['outliers'] += counts.get('outliers', 0)
        with open(output, 'a') as stream:
            stream.writelines(json.dumps(record)+'\n' for record in learner.outliers)
        learner.outliers.clear()
        pending.clear()

    def add(rows, read, of):
        totals['read'] += read
        for row in rows:
            pending.append(row)
            if len(pending) == args.batch_size:
                flush()
        totals['units'] += 1
        if totals['units'] % 25 == 0 or totals['units'] == of:
            if pending and totals['units'] == of:
                flush()
            print(json.dumps({**totals, 'mean_nll': totals['loss_sum']/max(1, totals['decisions']),
                              'decisions_per_second': round(totals['decisions']/(time.monotonic()-started), 1),
                              'of': of}), flush=True)

    if args.epoch is None:
        keys = set(json.loads(Path(args.keys).read_text())) if args.keys else None
        dataset.entries = sample(document['entries'], min_rating=args.min_rating, matches=args.matches, seed=args.seed,
                                 keys=keys)
        for source, entry in enumerate(dataset.entries):
            rows = [row for episode in entry['index']['episodes'] for descriptor in episode['chunks']
                    for row in dataset.read((source, episode['shared'], descriptor)) if keep(row, minutes, args.rating_band)]
            add(rows, len(rows), len(dataset.entries))
    else:
        dataset.entries = document['entries']  # Every entry, in order: the windows depend on it.
        windows = round_windows(dataset, seed=args.train_seed, epoch=args.epoch, segment=args.segment,
                                validations=args.validations, buffer_bytes=args.buffer_mib*1024*1024)
        tasks = ((window, {source: document['entries'][source] for source in {ref[0] for ref in window}})
                 for window in windows)
        with multiprocessing.Pool(args.workers, _start_worker, (args.shards, key, minutes, args.rating_band)) as pool:
            for rows, read in pool.imap(_read_window, tasks, chunksize=1):
                add(rows, read, len(windows))
    if pending:
        flush()
    print(json.dumps({'done': True, **totals}), flush=True)


if __name__ == '__main__':
    main()
