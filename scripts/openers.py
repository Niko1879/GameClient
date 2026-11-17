"""Strategy input z: each player's opener, as building counts over their first OPENER_BUILDINGS buildings.

Decided with the user 15-17 Nov 2026 (notes: memory next-big-run). The model sees the counts of each building
template among a player's first six build orders (e.g. 1 power plant, 3 supply centres, 2 war factories), only
during the opening window (generals_policy.OPENER_WINDOW_SECONDS), so it holds one opener plan instead of averaging
several; after the window it plays from the game state alone. A player with fewer than six buildings in the whole
game gets no z (a partial count would tell the model the game ends early), and OPENER_BLANK of the other player-games
get none either, deterministically by match and player, so the model also plays without one.

The counts come from the shards' own build targets (templates are already folded to the vanilla twins), read once
per shard at selection time from the chunk headers only, and cached by shard hash in training-data/openers.json.
Not a feature source: shards and the feature key are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from generals_policy import OPENER_BUILDINGS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'training-data'/'openers.json'
# The same building ordered again within this many seconds is a re-click after a refused placement; it counts once.
REPEAT_SECONDS = 3
OPENER_BLANK = 0.5
BLANK_SALT = 'opener-blank-v1'


def build_orders(rows, limit=OPENER_BUILDINGS):
    """(template, seconds) of a player's first `limit` build orders from rows in frame order."""
    orders, last = [], {}
    for row in rows:
        target = row['target']
        if target.get('kind') != 'build':
            continue
        template = int(target['template'])
        seconds = row['seconds'] if 'seconds' in row else row['frame']/row.get('source_ticks_per_second', 60)
        if template in last and seconds-last[template] < REPEAT_SECONDS:
            last[template] = seconds
            continue
        last[template] = seconds
        orders.append((template, seconds))
        if len(orders) == limit:
            break
    return orders


def _header_rows(archive, member):
    with archive.open(member) as stream:
        length, = struct.unpack('<Q', stream.read(8))
        return [meta['row'] for meta in json.loads(stream.read(length))['rows']]


def shard_openers(path, index):
    """{player (str): {'templates': [...], 'frames': [...]}} for each episode with OPENER_BUILDINGS builds,
    {player: None} for the others. Reads chunk headers in order and stops at the sixth building."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for episode in index['episodes']:
            orders = []
            for descriptor in sorted(episode['chunks'], key=lambda c: c['start']):
                rows = _header_rows(archive, descriptor['member'])
                if any(row['player'] != episode['player'] for row in rows):
                    raise ValueError('Episode chunk holds another player')
                orders = build_orders(_chain(orders, rows))
                if len(orders) == OPENER_BUILDINGS:
                    break
            result[str(episode['player'])] = ({'templates': [t for t, _ in orders], 'seconds': [round(s, 2) for _, s in orders]}
                                              if len(orders) == OPENER_BUILDINGS else None)
    return result


def _chain(orders, rows):
    # Earlier orders replayed as rows keep the re-click rule working across a chunk boundary.
    return [{'target': {'kind': 'build', 'template': t}, 'seconds': s} for t, s in orders]+list(rows)


def _job(job):
    path, index = job
    return shard_openers(path, index)


def attach(doc, data_dir, workers=8):
    """Set entry['openers'] on every dataset entry, from the cache or the shard; returns counts for reporting."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    missing = [e for e in doc['entries'] if e['archive_sha256'] not in cache]
    if missing:
        jobs = [(str(Path(data_dir)/e['path']), e['index']) for e in missing]
        with ProcessPoolExecutor(workers) as pool:
            for entry, openers in zip(missing, pool.map(_job, jobs, chunksize=16)):
                cache[entry['archive_sha256']] = openers
        temporary = CACHE.with_suffix('.tmp')
        temporary.write_text(json.dumps(cache, separators=(',', ':')))
        temporary.replace(CACHE)
    players = given = 0
    for entry in doc['entries']:
        entry['openers'] = cache[entry['archive_sha256']]
        players += len(entry['openers']); given += sum(v is not None for v in entry['openers'].values())
    return dict(players=players, with_opener=given, computed=len(missing))


def blanked(match_key, player):
    """Deterministic per player-game: OPENER_BLANK of them train without z."""
    digest = hashlib.sha256(f'{BLANK_SALT}|{match_key}|{player}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') < OPENER_BLANK*2**64


# Named openers for live play (user's taxonomy, 15 Nov 2026; the most common top-20% count vectors, 17 Nov sample of
# 3,494 players: 2sup_bar_wf_air 13%, 2sup_bar_2wf 11%, 2sup_bar_wf_bunker 10%, 3sup_2wf 8%, 3sup_bar_wf 7%,
# 2sup_2wf_prop 5%). Building counts by template name; each totals OPENER_BUILDINGS.
NAMED = {
    '2sup_bar_wf_air': {'ChinaPowerPlant': 1, 'ChinaSupplyCenter': 2, 'ChinaBarracks': 1, 'ChinaWarFactory': 1,
                        'ChinaAirfield': 1},
    '2sup_bar_2wf': {'ChinaPowerPlant': 1, 'ChinaSupplyCenter': 2, 'ChinaBarracks': 1, 'ChinaWarFactory': 2},
    '2sup_bar_wf_bunker': {'ChinaPowerPlant': 1, 'ChinaSupplyCenter': 2, 'ChinaBarracks': 1, 'ChinaWarFactory': 1,
                           'ChinaBunker': 1},
    '3sup_2wf': {'ChinaPowerPlant': 1, 'ChinaSupplyCenter': 3, 'ChinaWarFactory': 2},
    '3sup_bar_wf': {'ChinaPowerPlant': 1, 'ChinaSupplyCenter': 3, 'ChinaBarracks': 1, 'ChinaWarFactory': 1},
    '2sup_2wf_prop': {'ChinaPowerPlant': 1, 'ChinaSupplyCenter': 2, 'ChinaWarFactory': 2, 'ChinaPropagandaCenter': 1},
}


def named_templates(config, name):
    """Policy template indices of a named opener (or None for 'none'), for EventActor.configure_opener."""
    if name in (None, 'none'):
        return None
    if name not in NAMED:
        raise ValueError('Unknown opener %r (choose from %s)' % (name, ', '.join(sorted(NAMED))))
    templates = [config['templates'].index(template) for template, count in NAMED[name].items() for _ in range(count)]
    if len(templates) != OPENER_BUILDINGS:
        raise ValueError('Opener %s is not %d buildings' % (name, OPENER_BUILDINGS))
    return templates
