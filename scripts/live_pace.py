"""Live pace of a play_head_to_head folder: orders per game-minute and the act-after-act rate.

The no-op design's live acceptance check (notes/wait_snapshots.md): the same model at different look intervals
should keep its pace (within 10%) near the human ~52 orders/min, and acting should not feed itself: P(act | the
previous look acted) near the human follow-up rate (~0.28 within 0.2 s), not 0.94 as pilot 2 (23 Nov).

  python -B scripts/live_pace.py .cache/noop-pilot/live/look200 [--look-ticks 12]
"""
import argparse
import gzip
import json
import statistics
from pathlib import Path

import numpy as np


def folder_pace(root, look_ticks):
    rates, counts = [], dict(after_act=0, acted=0, after_idle=0, idle=0)
    for path in sorted(Path(root).glob('games/*/episode.jsonl.gz')):
        acted, last = {}, 0
        with gzip.open(path, 'rt') as lines:
            for line in lines:
                record = json.loads(line)
                if record.get('record') != 'transition':
                    continue
                last = int(record['frame'])
                for player, actions in record['actions'].items():
                    acted.setdefault(player, set())
                    if actions:
                        acted[player].add(last//look_ticks)
        minutes = last/3600
        for slots in acted.values():
            looks = np.zeros(last//look_ticks+1, bool)
            looks[list(slots)] = True
            before, after = looks[:-1], looks[1:]
            counts['after_act'] += int((before & after).sum()); counts['acted'] += int(before.sum())
            counts['after_idle'] += int((~before & after).sum()); counts['idle'] += int((~before).sum())
        summary = json.loads((path.parent/'summary.json').read_text())
        rates += [player['commands']/minutes for player in summary['event_scheduling'].values()]
    return dict(folder=str(root), player_games=len(rates),
                orders_per_minute=round(statistics.mean(rates), 1) if rates else None,
                sd=round(statistics.pstdev(rates), 1) if rates else None,
                p_act_after_act=round(counts['after_act']/max(1, counts['acted']), 3),
                p_act_after_idle=round(counts['after_idle']/max(1, counts['idle']), 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--look-ticks', type=int, default=12, help='Ticks between looks (12 = 200 ms at 60 Hz)')
    args = parser.parse_args()
    for folder in args.folders:
        print(json.dumps(folder_pace(folder, args.look_ticks)), flush=True)


if __name__ == '__main__':
    main()
