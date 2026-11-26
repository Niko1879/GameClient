"""Top-N clean matches by China match rating across Generals Online, gamereplays and special (user, 23 Nov 2026).

Rule per China-side player: GO ladder rating (rating_of); special: the folder expert's confirmed ladder rating
(expert-accounts.csv use_rating) when the player is that expert, else the style estimate; gamereplays: style
estimate. GenTool is left out. Special matches with the expert on a China side come first (user, 23 Nov). Writes JSON.

    python -B scripts/top_matches.py <N> <output.json or -> [<archive offset, e.g. -300>]"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, 'scripts')
import replay_cleaning as rcl

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '-' else None
# Added to archive (gamereplays, special) style estimates: the estimator rates special experts' own games ~300
# above their confirmed Generals Online ladder ratings (expert-accounts.csv), so the scales differ.
OFFSET = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

ratings, source, est = collections.defaultdict(dict), {}, {}


def offer(key, player, value, estimated, src):
    if value is None:
        return
    if estimated and src in ('gamereplays', 'special'):
        value += OFFSET
    ratings[key][player] = (value, estimated)
    source[key] = src


# Generals Online and gamereplays
with open(rcl.ESTIMATES, encoding='utf-8') as stream:
    for line in stream:
        row = json.loads(line)
        if row['faction'] not in rcl.CHINA_SIDES or row['collection'] == 'gentool' or row['archive'] == 'gentool':
            continue
        value, estimated = rcl.rating_of(row)
        offer(row['match_key'], row['player_index'], value, estimated,
              'gamereplays' if row['archive'] == 'gamereplays' else row['collection'])

# Metrics: paths of special matches, and GO matches scanned after the estimates were made
special_folder, scanned_go = {}, 0
with open(rcl.METRICS, encoding='utf-8') as stream:
    for line in stream:
        row = json.loads(line)
        if 'error' in row:
            continue
        if row.get('archive') == 'special':
            special_folder[row['match_key']] = Path(row['path'].replace('\\', '/')).parts[3]
        elif row.get('archive') is None and row['match_key'] not in ratings:
            for p in row['players']:
                if p['faction'] in rcl.CHINA_SIDES and (p.get('rating') or {}).get('overall') is not None:
                    offer(row['match_key'], p['player_index'], p['rating']['overall'], False,
                          Path(row['path'].replace('\\', '/')).parts[1])
                    scanned_go += 1

# Special: experts' confirmed ratings by recorded name
experts, recorded = {}, {}
with open('replays/special/expert-accounts.csv', encoding='utf-8-sig', newline='') as stream:
    for row in csv.DictReader(stream):
        names = {re.sub(r'\s*\(\d+\)$', '', n.strip()) for n in row['recorded_as'].split(';')}
        recorded[row['folder']] = names
        if row['use_rating']:
            experts[row['folder']] = (names, float(row['use_rating']))
# User, 23 Nov 2026: every clean special match whose China side includes the folder's expert (mirrors too) is in,
# whatever its rating; the rest of the N by rating.
expert_hits, china_sides, guaranteed = 0, collections.Counter(), set()
with open('replays/catalog/rating-estimates-special.jsonl', encoding='utf-8') as stream:
    for line in stream:
        row = json.loads(line)
        if row['faction'] not in rcl.CHINA_SIDES:
            continue
        china_sides[row['match_key']] += 1
        if row['name'] in recorded.get(special_folder.get(row['match_key']), ()) or china_sides[row['match_key']] == 2:
            guaranteed.add(row['match_key'])
        value, estimated = row['style_estimate'], True
        names, rating = experts.get(special_folder.get(row['match_key']), (set(), None))
        if row['name'] in names:
            value, estimated = rating, False
            expert_hits += 1
        if row['match_key'] in ratings and source[row['match_key']] != 'special':
            continue  # the same match from another source keeps that source's rating
        offer(row['match_key'], row['player_index'], value, estimated, 'special')

dropped, scanned, provenance = rcl.selection_filter()
match = {k: max(v.values()) for k, v in ratings.items() if v}
clean = {k: v for k, v in match.items() if k in scanned and k not in dropped}
print(json.dumps(dict(rated=len(match), clean=len(clean), late_go_players=scanned_go, special_expert_players=expert_hits,
                      clean_by_source=collections.Counter(source[k] for k in clean))))
guaranteed &= set(clean)
order = sorted(clean, key=lambda k: (k in guaranteed, clean[k][0]), reverse=True)
print('guaranteed special', len(guaranteed))
for n in sorted({2500, 5000, 7500, N}):
    top = order[:n]
    print(n, 'cut', clean[top[-1]][0], dict(collections.Counter(source[k] for k in top)),
          'estimated', sum(clean[k][1] for k in top))
if OUT:
    OUT.write_text(json.dumps(dict(rule=__doc__, n=N, cleaning=provenance, matches=[
        dict(match_key=k, rating=clean[k][0], estimated=clean[k][1], source=source[k]) for k in order[:N]])))
    print('wrote', OUT)
