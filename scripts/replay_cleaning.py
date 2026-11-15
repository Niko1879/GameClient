"""Data-cleaning metrics and filters over raw .rep files, before any export.

`scan` reads each replay's command stream once and writes per-player metrics
(JSON lines). `report` applies tunable thresholds to that file, so changing a
cutoff never needs a rescan. Checks (all thresholds are flags of `report`):

- afk: a player issued at most --afk-max-orders orders (selection, hotkey,
  camera and CRC messages are not orders).
- early_end: the recording ends before --early-end-minutes. The cause is kept
  (which player self-destructed, i.e. surrendered/quit, or none recorded).
- opening_dozers: a China player queued more than --max-opening-dozers dozers
  in the first --opening-seconds (cancelled queues are netted out).
- map: the game's map is not official, not a ranked-pack map (by CRC,
  replay-runtime/retail-104/ranked-maps.json) and not a map of any rated ladder
  game (by CRC). Joke/4fun maps (no money, casino, cxn, mods) only appear here.
  Applied by default (user decision, 7 Nov 2026: unrated games are kept, with the
  model's rated flag off, under the same map rule as GenTool's harvest).
- slots: the map does not have exactly 2 player slots, or no source verifies its
  slot count (scripts/map_slots.py: MapCache.ini numPlayers or the .map file's
  start waypoints; table replay-runtime/map-slots.json, rebuilt by `map-slots`).
  A competitive 1v1 is played on a 2-slot map (user decision, 10 Nov 2026).
  "[RANK]" is not evidence: GenTool's ranked pack includes team maps.
- unrated: the China player has no pre-match Overall rating in the sidecar's
  Matches listing text, i.e. the match was not a rated ladder game (custom
  lobbies, stats tracking disabled). Reported only since 7 Nov 2026 (applied
  5-7 Nov: ranked matches only); pass it in --checks to train ranked-only.
- cash: starting cash is not --starting-cash ($10,000, the ladder setting).
  Unrated lobbies hold most 4fun cash games ($1M, $50k, $100k: 6,046 of 6,275
  on 5 Nov), and those can be on normal maps. Applied by default with map.
- low_rating: that rating is below --min-china-rating (default 0: reported only).
- extra_trucks: against a non-GLA opponent, a China player queued more than
  --max-trucks-per-center supply trucks from one supply center in the first
  --truck-minutes (later queues are usually replacements for dead trucks).

Raw-stream limits: a queue the engine refused (no money, full queue) still
counts, and production from a multi-object selection is not attributed to a
single producer. Template IDs come from the pinned engine's catalogue
(replay-runtime/action-catalog.json); `scan` records each player's opening
queue so the IDs can be checked against every INI revision in the corpus.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import struct
import sys

from audit_replay_commands import message_names
import map_slots
from engine_runtime import disable_power_throttling
from harvest_china_replays import parse_replay
from replay_catalog import replay_key

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT/'replays/vanilla-china'
# gamereplays.org retail 1.04 archive (scripts/retail_conversion.py): top-level play with no
# listing, so no ladder rating. User decisions, 6 Nov 2026: one training batch with the ladder
# data; the experts named in Ratings.txt (and their aliases in player-names.csv) rate 2000,
# slightly below the current top 25 since play improves over time; everyone else 1700. Each
# player in each match gets a fixed jitter so the model does not key on exact values.
ARCHIVE = ROOT/'replays/gamereplays'
ARCHIVE_EXPERTS = ARCHIVE/'Ratings.txt'
ARCHIVE_ALIASES = ARCHIVE/'player-names.csv'
# Second gamereplays.org download (14 Nov 2026: vanilla, Tank and Nuke China 1v1s); same source and rules.
ARCHIVE3 = ROOT/'replays/gamereplays3'
EXPERT_RATING, ARCHIVE_RATING, RATING_JITTER = 2000, 1700, 50
# GenTool's upload archive (scripts/harvest_gentool.py): retail 1.04 games of any level,
# mostly casual. User, 7 Nov 2026: do not assume these are strong. Skill is unknown, so
# players are unrated (the model's rated flag is off) unless named in Ratings.txt, and the
# unrated check does not drop these matches; every other check applies.
GENTOOL = ROOT/'replays/gentool'
# Tank/Nuke-general China games from Generals Online (scripts/watch_china_replays.py; 13 Nov 2026): same crawler
# sidecars as CORPUS, so they scan like it.
CHINA_GENERALS = ROOT/'replays/china-generals'
GENTOOL_RATING = None


def archive_source(path):
    """'gamereplays', 'gentool' or None for a replay path."""
    parent = Path(path).resolve().parent
    if parent in (ARCHIVE.resolve(), ARCHIVE3.resolve()):
        return 'gamereplays'
    return 'gentool' if parent.parent == GENTOOL.resolve() else None
METRICS = ROOT/'replays/catalog/cleaning-metrics.jsonl'
# scripts/rating_estimator.py joint: one row per player of every scanned match.
ESTIMATES = ROOT/'replays/catalog/rating-estimates.jsonl'
RANKED_MAPS = ROOT/'replay-runtime/retail-104/ranked-maps.json'
_MAP_PREFIX = re.compile(r'^[0-9A-Fa-f]{2}(?=(?:userdata/)?maps/)', re.I)


def map_identity(header):
    """(map path without the header's 2-hex prefix, '%08X' CRC) of a parsed replay header."""
    return _MAP_PREFIX.sub('', header['map'].replace('\\', '/')), '%08X' % int(header['map_crc'], 16)


def ranked_map_crcs():
    return set(json.loads(RANKED_MAPS.read_text(encoding='utf-8'))['maps'])


def allowed_map(path, crc, allowed):
    """Shipped with the game (maps/...), or a custom map whose exact CRC is in allowed."""
    return path.casefold().startswith('maps/') or crc in allowed
ACTION_CATALOG = ROOT/'replay-runtime/action-catalog.json'
WIDTHS = (4, 4, 1, 4, 4, 4, 12, 8, 16, 4, 2)  # argument types in Recorder order
INT_TYPES = {0, 3, 4, 5, 9}  # integer, object, drawable, team, timestamp
NOT_ORDERS = {'MSG_BEGIN_NETWORK_MESSAGES', 'MSG_LOGIC_CRC', 'MSG_SET_REPLAY_CAMERA',
              'MSG_CREATE_SELECTED_GROUP', 'MSG_CREATE_SELECTED_GROUP_NO_SOUND',
              'MSG_DESTROY_SELECTED_GROUP', 'MSG_REMOVE_FROM_SELECTED_GROUP',
              'MSG_SELECTED_GROUP_COMMAND'}
WORKER_NAMES = ('Dozer', 'SupplyTruck', 'Chinook', 'Worker')
GLA = 'FactionGLA'


def tracked_templates(path=ACTION_CATALOG):
    """Template ID -> name for builders and harvesters, the only queues kept."""
    catalog = json.loads(Path(path).read_text(encoding='utf-8'))['template_catalog']
    return {int(k): v for k, v in catalog.items()
            if any(w in v for w in WORKER_NAMES) and 'Dead' not in v and 'Hulk' not in v
            and 'Rubble' not in v and not v.startswith(('CINE_', 'GC_'))}


def messages(data, position, names):
    """(frame, name, player, integer-like args) for each recorded message."""
    while position < len(data):
        if len(data) - position < 13:
            raise ValueError('Truncated replay message header')
        frame, kind, player, count = struct.unpack_from('<IiiB', data, position)
        position += 13
        descriptors = struct.unpack_from(f'<{2*count}B', data, position)
        position += 2*count
        args = []
        for dtype, repeat in zip(descriptors[::2], descriptors[1::2]):
            if dtype >= len(WIDTHS) or repeat == 0:
                raise ValueError('Unknown replay argument descriptor')
            if dtype in INT_TYPES:
                args.extend(struct.unpack_from(f'<{repeat}i', data, position))
            elif dtype == 2:
                args.extend(data[position:position+repeat])
            position += WIDTHS[dtype]*repeat
        if position > len(data):
            raise ValueError('Truncated replay arguments')
        yield frame, names.get(kind, str(kind)), player, args


def logic_rate(header):
    """Logic frames per second: 60 on current revisions, 30 on old ones."""
    from random_sides import RETAIL_RELEASES
    if header['exe_crc'] in RETAIL_RELEASES:
        return 30
    wall = header['end_time'] - header['start_time']
    return 30 if wall > 0 and header['frames']/wall < 42 else 60


RATING = re.compile(r'(Seasonal|Overall|World Series): (\d+) → (\d+)')


def listing_ratings(text, factions):
    """Pre-match ratings per listing side ({} for an unrated side), or None.

    The Matches listing reads '<n> <faction> <name> +d Seasonal: a -> b +d
    Overall: a -> b ... VS <faction> <name> ...'; unrated matches have no
    ladder figures. Listing sides follow replay slot order (78,628 of 78,629
    named China players matched their own slot on 3 Nov 2026).
    """
    parts = re.sub(r'^\d+ (This match does not count towards player stats\. )?', '', text or '').split(' VS ', 1)
    if len(parts) != 2 or not factions or len(factions) != 2:
        return None
    ratings = []
    for part, faction in zip(parts, factions):
        if not part.startswith(faction+' '):
            return None
        ratings.append({k.lower().replace(' ', '_'): int(before) for k, before, _ in RATING.findall(part)})
    return ratings


def archive_experts():
    """Every archive name that belongs to an expert in Ratings.txt, aliases included."""
    import csv
    experts = {line.strip() for line in ARCHIVE_EXPERTS.read_text(encoding='utf-8-sig').splitlines()
               if line.strip() and not line.strip().endswith(':')}
    with ARCHIVE_ALIASES.open(encoding='utf-8-sig', newline='') as stream:
        aliases = {row['name'] for row in csv.DictReader(stream) if row['alias_of'] in experts}
    return experts | aliases


def archive_rating(name, match_key, slot, experts, default=ARCHIVE_RATING):
    import random
    base = EXPERT_RATING if name in experts else default
    if base is None:
        return None
    noise = random.Random(f'{match_key}:{slot}').gauss(0, RATING_JITTER)
    return round(base + max(-2*RATING_JITTER, min(2*RATING_JITTER, noise)))


def sidecar(replay):
    try:
        return json.loads(replay.with_suffix('.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


class Player:
    def __init__(self):
        self.orders = self.selections = 0
        self.first_order = self.last_order = self.self_destruct = None
        self.selection, self.teams = set(), {}
        self.touched_selection = False
        self.opening_queue = None  # template of the first queue from the implicit start CC
        self.queues = []  # [frame, template, producer, production id]
        self.cancelled = set()

    def producer(self):
        if not self.touched_selection:
            return 'start_cc'
        if len(self.selection) == 1:
            return str(next(iter(self.selection)))
        return 'multi' if self.selection else 'none'

    def handle(self, frame, name, args, tracked):
        if name.startswith(('MSG_CREATE_SELECTED_GROUP', 'MSG_DESTROY_SELECTED_GROUP',
                            'MSG_REMOVE_FROM_SELECTED_GROUP')):
            self.selections += 1
            self.touched_selection = True
            if name == 'MSG_DESTROY_SELECTED_GROUP':
                self.selection = set()
            elif name == 'MSG_REMOVE_FROM_SELECTED_GROUP':
                self.selection -= set(args)
            else:
                if args and args[0]:
                    self.selection = set()
                self.selection |= set(args[1:])
            return
        if name.startswith(('MSG_CREATE_TEAM', 'MSG_SELECT_TEAM', 'MSG_ADD_TEAM')):
            self.selections += 1
            team = name[-1]
            if name.startswith('MSG_CREATE_TEAM'):
                self.teams[team] = set(self.selection)
            else:
                self.touched_selection = True
                if name.startswith('MSG_SELECT_TEAM'):
                    self.selection = set(self.teams.get(team, ()))
                else:
                    self.selection |= self.teams.get(team, set())
            return
        if name in NOT_ORDERS:
            return
        self.orders += 1
        self.first_order = frame if self.first_order is None else self.first_order
        self.last_order = frame
        if name == 'MSG_SELF_DESTRUCT' and self.self_destruct is None:
            self.self_destruct = frame
        elif name == 'MSG_QUEUE_UNIT_CREATE' and len(args) >= 2:
            producer = self.producer()
            if producer == 'start_cc' and self.opening_queue is None:
                self.opening_queue = args[0]
            if args[0] in tracked:
                self.queues.append([frame, args[0], producer, args[1]])
        elif name == 'MSG_CANCEL_UNIT_CREATE' and args:
            self.cancelled.add(args[0])


def scan_one(path, tracked=None):
    path = Path(path)
    tracked = tracked if tracked is not None else tracked_templates()
    try:
        data = path.read_bytes()
        header = parse_replay(data)
        names = message_names()
        players = defaultdict(Player)
        last_frame = 0
        for frame, name, index, args in messages(data, header['commands_offset'], names):
            last_frame = max(last_frame, frame)
            players[index].handle(frame, name, args, tracked)
    except (OSError, ValueError, struct.error) as error:
        return {'path': str(path), 'match_key': replay_key(path), 'error': str(error)}
    meta = sidecar(path)
    listing = meta.get('listing_factions')
    ratings = listing_ratings(meta.get('listing_text'), listing) or [None]*len(header['players'])
    key = replay_key(path)
    source = archive_source(path)
    if source:
        from random_sides import RETAIL_RELEASES, resolved_factions
        from replay_bootstrap import match_key
        key = match_key(header, replay_path=None if header['exe_crc'] in RETAIL_RELEASES else path)  # as retail_conversion queues it; GO files carry the match ID
        experts = archive_experts()
        default = GENTOOL_RATING if source == 'gentool' else ARCHIVE_RATING
        ratings = [{'overall': archive_rating(p['name'], key, p['slot'], experts, default)} for p in header['players']]
        # An unrated archive player stays unrated rather than an invented rating.
        if header['exe_crc'] in RETAIL_RELEASES:
            listing = resolved_factions(header)
    rows = []
    for position, slot in enumerate(header['players']):
        player = players[slot['slot']+2]  # PlayerList: neutral, civilian, then slots
        faction = slot['header_faction']
        if faction == 'Random' and listing and len(listing) == len(header['players']):
            faction = listing[position]
        queues = [q+[q[3] in player.cancelled] for q in player.queues]
        rows.append({'slot': slot['slot'], 'player_index': slot['slot']+2, 'faction': faction,
                     'rating': ratings[position],
                     'header_faction': slot['header_faction'], 'orders': player.orders,
                     'selections': player.selections, 'first_order': player.first_order,
                     'last_order': player.last_order, 'self_destruct': player.self_destruct,
                     'opening_queue': player.opening_queue, 'queues': queues})
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'bytes': len(data), 'match_key': key, 'exe_crc': header['exe_crc'], 'ini_crc': header['ini_crc'],
            'frames': header['frames'], 'last_command_frame': last_frame, 'rate': logic_rate(header),
            'quit_early': header['quit_early'], 'archive': source,
            'map': map_identity(header)[0], 'map_crc': map_identity(header)[1],
            'starting_cash': int(header['starting_cash']) if header.get('starting_cash') else None, 'players': rows}


def _init_worker():
    disable_power_throttling()


def _scan_batch(paths):
    tracked = tracked_templates()
    return [scan_one(p, tracked) for p in paths]


def scan(args):
    """Rescan only new or changed replays (by path and size) unless --full."""
    paths = sorted(Path(args.corpus).glob('*/*.rep'))
    if Path(args.corpus).resolve() == CORPUS.resolve() and ARCHIVE.is_dir():
        paths += (sorted(ARCHIVE.glob('*.rep')) + sorted(ARCHIVE3.glob('*.rep')) + sorted(GENTOOL.glob('*/*.rep'))
                  + sorted(CHINA_GENERALS.glob('*/*.rep')))
    if args.limit:
        paths = paths[:args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if output.exists() and not args.full:
        with output.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if 'error' not in row and 'bytes' in row and 'starting_cash' in row and 'rating' in row['players'][0]:
                    previous[row['path']] = row
    keep, todo = [], []
    for path in paths:
        relative = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        row = previous.get(relative)
        # Archive ratings follow Ratings.txt, so those rows are always recomputed.
        if row and row['bytes'] == path.stat().st_size and not archive_source(path):
            keep.append(row)
        else:
            todo.append(path)
    partial = output.with_suffix('.partial')
    batches = [todo[i:i+200] for i in range(0, len(todo), 200)]
    done = errors = 0
    with partial.open('w', encoding='utf-8') as stream:
        for row in keep:
            stream.write(json.dumps(row, separators=(',', ':'))+'\n')
        if batches:
            with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
                for rows in pool.map(_scan_batch, batches):
                    for row in rows:
                        stream.write(json.dumps(row, separators=(',', ':'))+'\n')
                        errors += 'error' in row
                    done += len(rows)
                    if done % 10000 < 200:
                        print(f'{done}/{len(todo)} scanned, {errors} unreadable', flush=True)
    partial.replace(output)
    print(f'{len(keep)} reused, {done} scanned, {errors} unreadable -> {output}')


# --- thresholds -----------------------------------------------------------

# Listing names of the sides the policy learns from (13 Nov 2026: vanilla China plus the Tank and Nuke
# generals; generals_policy.CHINA_FACTIONS). Cleaning checks and match ratings cover all three.
CHINA_SIDES = ('China', 'China Tank', 'China Nuke')


def is_china(player):
    return player['faction'] in CHINA_SIDES


def opponent_native(row, player):
    from enemy_side import NATIVE
    others = [p for p in row['players'] if p is not player]
    return NATIVE.get(others[0]['faction']) if len(others) == 1 else None


def player_flags(row, player, opts, names):
    """Reasons to drop this player's demonstration (empty list: keep)."""
    flags = []
    rate = row['rate']
    overall = (player.get('rating') or {}).get('overall')
    if overall is None:
        if row.get('archive') != 'gentool':
            flags.append('unrated')
    elif overall < opts.min_china_rating:
        flags.append(f'low_rating:{overall}')
    if is_china(player):
        def kind(t):
            return names.get(t, '')
        dozer_window = opts.opening_seconds*rate
        dozers = sum(1 for f, t, _, _, cancelled in player['queues']
                     if f <= dozer_window and not cancelled and kind(t) == 'ChinaVehicleDozer')
        if dozers > opts.max_opening_dozers:
            flags.append(f'opening_dozers:{dozers}')
        enemy = opponent_native(row, player)
        if enemy is not None and not enemy.startswith(GLA):
            per_center = Counter(producer for f, t, producer, _, cancelled in player['queues']
                                 if f <= opts.truck_minutes*60*rate and not cancelled
                                 and kind(t) == 'ChinaVehicleSupplyTruck'
                                 and producer not in ('multi', 'none', 'start_cc'))
            worst = max(per_center.values(), default=0)
            if worst > opts.max_trucks_per_center:
                flags.append(f'extra_trucks:{worst}')
    return flags


def ladder_map_crcs(metrics):
    """CRCs of maps played in rated ladder games (a rated China player, Generals Online corpus)."""
    crcs = set()
    with Path(metrics).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if 'error' not in row and not row.get('archive') and any(
                    is_china(p) and (p.get('rating') or {}).get('overall') is not None for p in row['players']):
                crcs.add(row['map_crc'])
    return crcs


def game_flags(row, opts, allowed_maps=None, slot_table=None):
    flags = []
    rate = row['rate']
    if allowed_maps is not None and not allowed_map(row['map'], row['map_crc'], allowed_maps):
        flags.append(f"map:{row['map_crc']}")
    if slot_table is not None:
        count = map_slots.slots(slot_table, row['map'], row['map_crc'])
        if count != 2:
            flags.append(f"slots:{row['map_crc']}:{'unverified' if count is None else count}")
    cash = row.get('starting_cash')
    if cash is not None and cash != getattr(opts, 'starting_cash', 10000):
        flags.append(f'cash:{cash}')
    for player in row['players']:
        if player['orders'] <= opts.afk_max_orders:
            flags.append(f"afk:slot{player['slot']}:{player['orders']}")
    end = max(row['frames'], row['last_command_frame'])
    if end < opts.early_end_minutes*60*rate:
        quitters = [p for p in row['players'] if p['self_destruct'] is not None]
        cause = ','.join(f"slot{p['slot']}" for p in quitters) or 'no_quit'
        flags.append(f'early_end:{end/rate/60:.1f}min:{cause}')
    return flags


def verified_revisions(path, names, minimum=5):
    """INI revisions whose China opening queue is the catalogue's dozer.

    Template IDs follow INI load order, so a revision that adds objects shifts
    them (49 revisions in the corpus; a few hundred games use shifted IDs).
    Template checks only run where the pinned engine's IDs are confirmed.
    """
    dozer = next(k for k, v in names.items() if v == 'ChinaVehicleDozer')
    opening = defaultdict(Counter)
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            for player in row.get('players', ()):
                if is_china(player) and player['opening_queue'] is not None:
                    opening[row['ini_crc']][player['opening_queue']] += 1
    return {ini for ini, c in opening.items()
            if sum(c.values()) >= minimum and c.most_common(1)[0][0] == dozer}


DEFAULTS = dict(checks='afk,early_end,opening_dozers,map,slots,cash', starting_cash=10000, afk_max_orders=10, early_end_minutes=2.0,
                opening_seconds=60.0, max_opening_dozers=3, truck_minutes=6.0, max_trucks_per_center=1,
                min_china_rating=0)


def evaluate(metrics, opts):
    """Yield (row, China player, flags) per China player; (row, None, None) if unreadable."""
    names = tracked_templates()
    verified = verified_revisions(metrics, names)
    allowed = ranked_map_crcs() | ladder_map_crcs(metrics)
    slot_table = map_slots.load()
    with Path(metrics).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if 'error' in row:
                yield row, None, None
                continue
            shared = game_flags(row, opts, allowed, slot_table)
            for player in row['players']:
                if is_china(player):
                    own = player_flags(row, player, opts, names) if row['ini_crc'] in verified else []
                    yield row, player, shared+own


def selection_filter(metrics=METRICS, **overrides):
    """(match key -> flags to drop, scanned match keys, provenance) for training selection.

    Shards hold whole matches, so a match is dropped when any China player in
    it is flagged by an applied check (in a mirror this also drops the clean
    side). Thresholds are applied here, so any setting works on one scan.
    """
    opts = argparse.Namespace(**{**DEFAULTS, **overrides})
    applied = set(opts.checks.split(',')) if opts.checks else set()
    dropped, scanned = {}, set()
    for row, player, flags in evaluate(metrics, opts):
        scanned.add(row['match_key'])
        if player is None:
            dropped[row['match_key']] = ['unreadable']
            continue
        hits = [f for f in flags if f.split(':')[0] in applied]
        if hits:
            dropped.setdefault(row['match_key'], []).extend(hits)
    with Path(metrics).open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dropped, scanned, dict(metrics_sha256=sha, options=vars(opts))


# Estimate used for players without a real rating. User decision 13 Nov 2026: switch to the per-game
# 'style_estimate' (the joint fit gives an identity one rating across all years, ExCaL 2010 = ExCaL 2026;
# style accuracy holds across archive eras: winner prediction 62-69% for 2005-2026 games vs 71% for close
# strong GO games). Landed 15 Nov 2026 with the observation v11 re-export (projectiles, hazards, cues), as
# the user planned; bundles before it (top20R, the three-faction pilot) used 'joint_estimate'.
ESTIMATE_FIELD = 'style_estimate'


def rating_of(row):
    """(rating, estimated) for one rating_estimator.py row: the ladder rating when the match had one;
    else, for a Generals Online player, their own rating found by name (nearest rated game); else the
    model estimate named by ESTIMATE_FIELD, flagged estimated (gamereplays and GenTool players always)."""
    if row['rating'] is not None:
        return row['rating'], False
    if row['identity'].startswith('go:') and row['name_rating'] is not None:
        return row['name_rating'], False
    return row[ESTIMATE_FIELD], True


def vanilla_china_matches(estimates=ESTIMATES):
    """Match keys with a vanilla China player (the faction-mix pilot's --vanilla-from rule)."""
    found = set()
    with Path(estimates).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['faction'] == 'China':
                found.add(row['match_key'])
    return found


def player_ratings(estimates=ESTIMATES):
    """match key -> {player index (str): {'rating': value, 'estimated': bool}} (re-export fix 7; rule: rating_of).
    No player is unknown: ESTIMATES covers the whole catalogue (GO, GenTool, gamereplays, gamereplays3)."""
    ratings = {}
    with Path(estimates).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            value, estimated = rating_of(row)
            ratings.setdefault(row['match_key'], {})[str(row['player_index'])] = {'rating': value, 'estimated': estimated}
    return ratings


def china_match_ratings(estimates=ESTIMATES):
    """match key -> the highest rating among its China-side players, vanilla, Tank or Nuke (rule: rating_of).

    The rating ladder's match rating: the export queue is ordered by it (.cache/repin-qual-v6/rating_order.py)
    and subset_manifest.match_rating applies the same rule to a bundle's entries."""
    best = {}
    with Path(estimates).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['faction'] not in CHINA_SIDES:
                continue
            value, _ = rating_of(row)
            if value is not None:
                best[row['match_key']] = max(best.get(row['match_key'], value), value)
    return best


def report(args):
    applied = set(args.checks.split(','))
    reported, dropped, excluded = Counter(), Counter(), []
    games, china_players, kept_players, unreadable = set(), 0, 0, 0
    for row, player, flags in evaluate(args.metrics, args):
        if player is None:
            unreadable += 1
            continue
        games.add(row['match_key'])
        china_players += 1
        kinds = {flag.split(':')[0] for flag in flags}
        reported.update(kinds)
        if kinds & applied:
            dropped.update(kinds & applied)
            excluded.append({'match_key': row['match_key'], 'path': row['path'],
                             'player_index': player['player_index'], 'flags': flags})
        else:
            kept_players += 1
    summary = {'games_with_china': len(games), 'unreadable': unreadable, 'china_players': china_players,
               'kept_china_players': kept_players, 'excluded_china_players': len(excluded),
               'excluded_matches': len({e['match_key'] for e in excluded}),
               'applied_checks': sorted(applied), 'players_flagged': dict(reported),
               'players_dropped_by': dict(dropped),
               'thresholds': {k: v for k, v in vars(args).items()
                              if k not in ('metrics', 'output', 'func', 'checks')}}
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps({'summary': summary, 'excluded': excluded}, indent=1),
                                     encoding='utf-8')


def write_map_slots(args):
    table = map_slots.build()
    Path(args.output).write_text(json.dumps(table, indent=1)+'\n', encoding='utf-8')
    counts = Counter(table['by_crc'].values())
    print(json.dumps({'maps': len(table['by_crc']), 'official_names': len(table['official']),
                      'sources': table['sources'], 'by_slots': dict(sorted(counts.items()))}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    s = sub.add_parser('scan', help='Compute per-player metrics from raw replays')
    s.add_argument('--corpus', default=CORPUS)
    s.add_argument('--output', default=METRICS)
    s.add_argument('--workers', type=int, default=8)
    s.add_argument('--limit', type=int, default=0)
    s.add_argument('--full', action='store_true', help='Rescan every replay')
    s.set_defaults(func=scan)
    r = sub.add_parser('report', help='Apply thresholds to scanned metrics')
    r.add_argument('--metrics', default=METRICS)
    r.add_argument('--output', help='Write the exclusion list here (JSON)')
    r.add_argument('--checks', default=DEFAULTS['checks'],
                   help='Flags that exclude a player (extra_trucks is reported, not applied, by default)')
    for name, value in DEFAULTS.items():
        if name != 'checks':
            r.add_argument('--'+name.replace('_', '-'), type=type(value), default=value)
    r.set_defaults(func=report)
    m = sub.add_parser('map-slots', help='Rebuild the map CRC -> slot count table the slots check reads')
    m.add_argument('--output', default=map_slots.TABLE)
    m.set_defaults(func=write_map_slots)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
