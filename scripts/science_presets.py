"""Science purchase orders of top-rated vanilla China players, per matchup (scripted science buyer presets).

The policy has no science action (re-export fix 2, user 11 Nov 2026): in live play a script buys
sciences from a fixed priority list per matchup. This derives candidate lists from raw replays.

Purchases come from MSG_PURCHASE_SCIENCE in the replay stream. Its argument is the engine's
ScienceType, a NameKey (Science.cpp), so the number depends on name-key order, not on the science
alone. SCIENCE_IDS below was calibrated 12 Nov 2026 against v6 shards (owned-science flags in
player_flags, first row after each purchase): Generals Online and retail 1.04 builds agree.
Unmapped IDs (14, 49, 54, 75: 16 purchases in the top 10%) are counted and skipped.

Most games end before rank 5, so an order is a priority list that is cut wherever a game ends.
Orders are ranked by a prefix tree: a prefix's support is the number of players whose purchases
start with it. The top orders are the most supported prefixes at a depth most players still reach
(`--depth`), each extended by its most common continuation while at least `--min-support` players
continue it. Repeated purchases of an owned science (rejected spam) are dropped.

  python -B scripts/science_presets.py --min-rating 1574 --output .cache/science-presets/top10.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import replay_cleaning as rc

SCIENCE_IDS = {
    31: 'SCIENCE_RedGuardTraining', 33: 'SCIENCE_ClusterMines', 34: 'SCIENCE_ArtilleryTraining',
    35: 'SCIENCE_NukeLauncher', 36: 'SCIENCE_ArtilleryBarrage1', 37: 'SCIENCE_ArtilleryBarrage2',
    38: 'SCIENCE_ArtilleryBarrage3', 39: 'SCIENCE_Frenzy1', 40: 'SCIENCE_Frenzy2', 41: 'SCIENCE_Frenzy3',
    45: 'SCIENCE_CashHack1', 46: 'SCIENCE_CashHack2', 47: 'SCIENCE_CashHack3', 48: 'SCIENCE_EMPPulse',
    50: 'Early_SCIENCE_ChinaCarpetBomb', 69: 'SCIENCE_EmergencyRepair1', 70: 'SCIENCE_EmergencyRepair2',
    71: 'SCIENCE_EmergencyRepair3',
}

# Promotion rules (Rank.ini and Science.ini, identical in retail 1.04 and the Generals Online patch; read
# 12 Nov 2026): ranks at 0/800/1500/2500/5000 skill points grant 1/1/1/1/3 purchase points (7 in all).
# Every vanilla China science costs 1 point. Rank 2 and rank 4 unlock nothing new.
POINTS_BY_RANK = {1: 1, 2: 1, 3: 1, 4: 1, 5: 3}
SCIENCE_RULES = {  # science: (minimum rank, prerequisite science or None)
    'SCIENCE_RedGuardTraining': (1, None), 'SCIENCE_ArtilleryTraining': (1, None), 'SCIENCE_NukeLauncher': (1, None),
    'SCIENCE_ClusterMines': (3, None), 'Early_SCIENCE_ChinaCarpetBomb': (3, None), 'SCIENCE_EMPPulse': (5, None),
    'SCIENCE_ArtilleryBarrage1': (3, None), 'SCIENCE_ArtilleryBarrage2': (3, 'SCIENCE_ArtilleryBarrage1'),
    'SCIENCE_ArtilleryBarrage3': (3, 'SCIENCE_ArtilleryBarrage2'), 'SCIENCE_Frenzy1': (3, None),
    'SCIENCE_Frenzy2': (3, 'SCIENCE_Frenzy1'), 'SCIENCE_Frenzy3': (3, 'SCIENCE_Frenzy2'),
    'SCIENCE_CashHack1': (3, None), 'SCIENCE_CashHack2': (3, 'SCIENCE_CashHack1'),
    'SCIENCE_CashHack3': (3, 'SCIENCE_CashHack2'), 'SCIENCE_EmergencyRepair1': (3, None),
    'SCIENCE_EmergencyRepair2': (3, 'SCIENCE_EmergencyRepair1'), 'SCIENCE_EmergencyRepair3': (3, 'SCIENCE_EmergencyRepair2'),
}


def schedule(order):
    """Rank at which a buyer that waits for the next item (never skips) buys each pick; None if never."""
    owned, ranks, points, i = set(), [], 0, 0
    for rank in range(1, 6):
        points += POINTS_BY_RANK[rank]
        while i < len(order) and points:
            need, prerequisite = SCIENCE_RULES[order[i]]
            if rank < need or (prerequisite and prerequisite not in owned):
                break
            owned.add(order[i]); ranks.append(rank); points -= 1; i += 1
    return ranks + [None]*(len(order)-len(ranks))


def purchases(metrics_row, names):
    """{player index: [(seconds, science name)]} for the row's China players; unknown IDs counted apart."""
    from harvest_china_replays import parse_replay
    data = (rc.ROOT/metrics_row['path']).read_bytes()
    header = parse_replay(data)
    china = {p['player_index'] for p in metrics_row['players'] if rc.is_china(p)}
    bought, unknown = {p: [] for p in china}, collections.Counter()
    for frame, name, index, args in rc.messages(data, header['commands_offset'], names):
        if name != 'MSG_PURCHASE_SCIENCE' or index not in china:
            continue
        science = SCIENCE_IDS.get(args[0])
        if science is None:
            unknown[args[0]] += 1
        elif science not in {s for _, s in bought[index]}:
            bought[index].append((frame/metrics_row['rate'], science))
    return bought, unknown


def prefix_support(sequences):
    support = collections.Counter()
    for seq in sequences:
        for d in range(1, len(seq)+1):
            support[tuple(seq[:d])] += 1
    return support


def set_next(sequences):
    """frozenset of the first k purchases -> Counter of purchase k+1 (order within the set ignored)."""
    table = collections.defaultdict(collections.Counter)
    for seq in sequences:
        for k in range(len(seq)):
            table[frozenset(seq[:k])][seq[k]] += 1
    return table


def orders(sequences, pools, depth, min_support, tail_support, top=3):
    """The `top` most supported orders, each completed to the end of the science tree where data allows.

    Heads: the most supported prefixes at `depth`; if fewer than `top` reach min_support there,
    shorter prefixes not already covered fill in. Each step after the head is taken from, in turn:
    'exact' (this matchup's players with this exact order so far, >= min_support), 'matchup', 'side'
    or 'all' (players owning the same set, any order; matchup >= min_support, pools >= tail_support).
    """
    support = prefix_support(sequences)
    heads = []
    for d in range(depth, 0, -1):
        for p, n in support.most_common():
            if len(heads) < top and len(p) == d and n >= min_support and not any(h[:d] == p for h in heads):
                heads.append(p)
    tables = [('matchup', set_next(sequences), min_support)] + [(name, set_next(seqs), tail_support) for name, seqs in pools]
    result = []
    for head in heads:
        order, steps = list(head), [{'science': s, 'source': 'exact', 'support': support[tuple(head[:i+1])]}
                                    for i, s in enumerate(head)]
        while True:
            nxt = [(n, p[-1]) for p, n in support.items() if len(p) == len(order)+1 and list(p[:-1]) == order]
            if nxt and max(nxt)[0] >= min_support:
                n, science = max(nxt); source = 'exact'
            else:
                for source, table, floor in tables:
                    counts = table.get(frozenset(order), collections.Counter())
                    counts = [(c, s) for s, c in counts.items() if s not in order]
                    if counts and max(counts)[0] >= floor:
                        n, science = max(counts)
                        break
                else:
                    break
            order.append(science)
            steps.append({'science': science, 'source': source, 'support': n})
        for step, rank in zip(steps, schedule(order)):
            step['bought_at_rank'] = rank
        result.append({'order': order, 'steps': steps})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--min-rating', type=float, default=1574,
                        help="China player's own rating (estimates count); 1574 = top 10%% of all replays")
    parser.add_argument('--depth', type=int, default=3, help='Prefix length the top orders are ranked at')
    parser.add_argument('--min-support', type=int, default=10, help='Players needed for a step from this matchup')
    parser.add_argument('--tail-support', type=int, default=5, help='Players needed for a step from a pooled side/all table')
    parser.add_argument('--output', required=True)
    parser.add_argument('--write-orders', help='Also write the live buyer file (#1 order per native opponent faction), '
                        'e.g. replay-runtime/science-orders.json')
    args = parser.parse_args()
    from engine_runtime import disable_power_throttling
    disable_power_throttling()
    from audit_replay_commands import message_names
    names = message_names()
    ratings = rc.player_ratings()
    dropped, _, _ = rc.selection_filter()
    by_matchup, unknown, players, minutes = collections.defaultdict(list), collections.Counter(), 0, []
    first_times = collections.defaultdict(list)
    for line in Path(rc.METRICS).open(encoding='utf-8'):
        row = json.loads(line)
        if 'error' in row or row['match_key'] in dropped or len(row['players']) != 2:
            continue
        own = {p['player_index']: (ratings.get(row['match_key'], {}).get(str(p['player_index'])) or {}).get('rating')
               for p in row['players'] if rc.is_china(p)}
        if not any(r is not None and r >= args.min_rating for r in own.values()):
            continue
        bought, missing = purchases(row, names)
        unknown.update(missing)
        for player in row['players']:
            index = player['player_index']
            if index not in bought or own[index] is None or own[index] < args.min_rating:
                continue
            opponent = next(p for p in row['players'] if p is not player)['faction']
            seq = [s for _, s in bought[index]]
            by_matchup[opponent].append(seq)
            for t, s in bought[index]:
                first_times[s].append(t)
            players += 1
            minutes.append(max(row['frames'], row['last_command_frame'])/row['rate']/60)
    report = {'min_rating': args.min_rating, 'players': players, 'unknown_ids': dict(unknown),
              'median_game_minutes': sorted(minutes)[len(minutes)//2] if minutes else None,
              'median_purchase_seconds': {s: round(sorted(t)[len(t)//2]) for s, t in sorted(first_times.items())},
              'matchups': {}}
    side = lambda faction: faction.split()[0]
    by_side = collections.defaultdict(list)
    for opponent, seqs in by_matchup.items():
        by_side[side(opponent)] += seqs
    everyone = [s for seqs in by_matchup.values() for s in seqs]
    for opponent, seqs in sorted(by_matchup.items(), key=lambda kv: -len(kv[1])):
        if opponent == 'Random':
            continue  # opponent side never resolved
        lengths = collections.Counter(min(len(s), 6) for s in seqs)
        report['matchups'][opponent] = {
            'players': len(seqs),
            'purchases_made': {('6+' if k == 6 else str(k)): v for k, v in sorted(lengths.items())},
            'first_purchase': collections.Counter(s[0] for s in seqs if s).most_common(5),
            'orders': orders(seqs, [('side', by_side[side(opponent)]), ('all', everyone)],
                             args.depth, args.min_support, args.tail_support)}
    # An opponent of unknown faction (random lobby pick) gets the #1 order of all players pooled.
    report['default'] = orders(everyone, [], args.depth, args.min_support, args.tail_support, top=1)[0]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.write_orders:
        from enemy_side import NATIVE
        live = {'schema': 1, 'kind': 'generals_science_orders', 'source': 'scripts/science_presets.py',
                'min_rating': args.min_rating, 'players': players,
                'note': 'Fixed priority list per opponent faction; the buyer waits for the next science, never skips '
                        '(user sign-off 12 Nov 2026: #1 order per matchup)',
                'default': report['default']['order'],
                'orders': {NATIVE[name]: m['orders'][0]['order'] for name, m in report['matchups'].items()}}
        Path(args.write_orders).write_text(json.dumps(live, indent=1)+chr(10), encoding='utf-8')
    Path(args.output).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
