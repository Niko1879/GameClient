"""Screen new replays before importing them: keep only new vanilla-China 1v1 games.

In order, each replay is
  1. not 1v1         - not exactly two human players on opposing teams (AI slots, FFA, 2v2...);
  1b. unsupported version - not a release we can convert: retail 1.04 (exe DA2B4B18/C5BD7FC3) with
                       unmodified game data (INI FEAAE3F3), or a Generals Online release the crawler holds
                       (replays/vanilla-china/<exe CRC>/, not blacklisted). 1.05b and other builds land here;
  2. no vanilla China - neither player is vanilla China (China Tank/Nuke/Infantry don't count; --sides widens
                       the wanted set, e.g. --sides China "China Tank" "China Nuke": "no wanted side"). A Random
                       pick is resolved from the game seed for retail 1.04 (random_sides, engine-matched);
                       other releases can't be resolved from the file, so an unresolved Random that could
                       be China is set aside as "random unresolved" for a manual look, never removed;
  0. first, copies of one game within the input are reduced to one (the longest recording);
  3. a duplicate      - of a replay we already hold, see below;
  4. new             - kept; "new: map missing" when a retail game's map isn't in the map store yet
                       (still kept: `retail_conversion.py maps <folder>` adds maps).
Report only by default; --apply move puts every non-new file under <folder>/_screened/<reason>/
(reversible), --apply delete removes them (random-unresolved files are only ever moved).

A replay counts as a duplicate when we already hold the same GAME, not only the same file:
the opponent's recording of a match, or a Generals Online match found on another site, has
different bytes. Same game = retail_conversion.match_identity: (game seed, map CRC, sorted
player names), read from the replay header.

Every held replay (replays/vanilla-china, gentool, gamereplays, ...) is fingerprinted from
its first 8 KB into .cache/replay-fingerprints.sqlite3; later runs read only new or changed
files. Catalogue status is looked up for duplicates with primary-key reads only, so this is
safe while the converter runs (scan the catalogue with nothing heavier: see the memory note
on heavy catalogue queries).

    python -B scripts/screen_replays.py D:/crawl/replays            # report a folder (recursive)
    python -B scripts/screen_replays.py D:/crawl/replays --apply move
    python -B scripts/screen_replays.py one.rep two.rep --list-new new.txt
    python -B scripts/screen_replays.py D:/crawl --json report.json

Exit code 0 always; the summary line counts new, duplicate and unreadable files.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time

from harvest_china_replays import FACTIONS, parse_replay
from harvest_china_replays import replay_release_exclusion
from random_sides import RETAIL_RELEASES, resolved_factions

ROOT = Path(__file__).resolve().parents[1]
HELD = ROOT/'replays'
INDEX = ROOT/'.cache/replay-fingerprints.sqlite3'
CATALOG = ROOT/'replays/catalog/replay-catalog.sqlite3'
HEADER_BYTES = 8192


def fingerprint(path):
    """(identity JSON, header summary) from the header alone; ValueError if it is not a replay."""
    with Path(path).open('rb') as stream:
        data = stream.read(HEADER_BYTES) + b'\0'*16
    header = parse_replay(data)
    options = dict(x.split('=', 1) for x in header['game_options'].split(';') if '=' in x)
    # retail_conversion.match_identity, plus the release: the same seed on another release is another game.
    identity = json.dumps([header['exe_crc'], options.get('SD'), header['map_crc'],
                           sorted(p['name'] for p in header['players'])])
    summary = dict(map=Path(header['map'].replace('\\', '/')).name, exe_crc=header['exe_crc'],
                   players=[f"{p['name']} ({p['header_faction']})" for p in header['players']],
                   start_time=header['start_time'], frames=header['frames'])
    return identity, summary, header


# parse_replay's refusals that mean "not a human 1v1", as opposed to "not a readable replay".
NOT_1V1 = ('AI slot', 'active humans', 'same team', 'more than eight slots')


RETAIL_INI = 'FEAAE3F3'  # retail_conversion.RETAIL_INI
ONLINE = ROOT/'replays/vanilla-china'


def unsupported(header):
    """Why no converter here can take this replay's game version, or None."""
    exe, ini = header['exe_crc'], header['ini_crc']
    if exe in RETAIL_RELEASES:
        return None if ini == RETAIL_INI else f'retail 1.04 with modified game data (INI {ini})'
    if (ONLINE/exe).is_dir() and not replay_release_exclusion(exe):
        return None
    return f'unknown or unsupported release (exe {exe}, INI {ini})'


def map_missing(header):
    if header['exe_crc'] not in RETAIL_RELEASES:
        return None
    from retail_conversion import missing_map
    return missing_map(header)


def factions(header, path):
    """(faction per player, resolved?): Random resolved from the seed on retail 1.04, else from a crawler
    sidecar (<replay>.json listing_factions, Generals Online), else left unresolved."""
    if header['exe_crc'] in RETAIL_RELEASES:
        return resolved_factions(header), True
    named = [p['header_faction'] for p in header['players']]
    sidecar = Path(path).with_suffix('.json')
    if 'Random' in named and sidecar.is_file():
        try:
            listed = list(json.loads(sidecar.read_text(encoding='utf-8')).get('listing_factions') or [])
        except (ValueError, OSError):
            listed = []
        # The listing gives the match's factions; fixed header picks take theirs, Random takes the rest.
        rest = list(listed)
        for name in named:
            if name != 'Random' and name in rest:
                rest.remove(name)
        if len(listed) == len(named) and len(rest) == named.count('Random'):
            fill = iter(rest)
            return [next(fill) if name == 'Random' else name for name in named], True
    return named, 'Random' not in named


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def open_index():
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(INDEX)
    db.execute('''CREATE TABLE IF NOT EXISTS held (path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER,
                  identity TEXT, sha256 TEXT)''')
    db.execute('CREATE INDEX IF NOT EXISTS held_identity ON held(identity)')
    return db


def refresh(db, roots, quiet=False):
    """Fingerprint held replays that are new or changed since the last run; forget deleted ones."""
    known = {path: (size, mtime) for path, size, mtime in db.execute('SELECT path,size,mtime_ns FROM held')}
    seen, added, started = set(), 0, time.time()
    for root in roots:
        for path in root.rglob('*'):
            if path.suffix.lower() != '.rep' or not path.is_file():
                continue
            key = str(path.resolve())
            seen.add(key)
            stat = path.stat()
            if known.get(key) == (stat.st_size, stat.st_mtime_ns):
                continue
            try:
                identity = fingerprint(path)[0]
            except (ValueError, OSError):
                identity = None
            db.execute('INSERT OR REPLACE INTO held VALUES (?,?,?,?,NULL)', (key, stat.st_size, stat.st_mtime_ns, identity))
            added += 1
            if added % 5000 == 0:
                db.commit()  # an interrupted first build keeps its progress
                if not quiet:
                    print(f'  fingerprinted {added} held replays ({time.time()-started:.0f} s)', file=sys.stderr, flush=True)
    gone = set(known)-seen
    db.executemany('DELETE FROM held WHERE path=?', [(path,) for path in gone])
    db.commit()
    return added, len(gone)


def held_sha(db, path):
    """A held file's full hash, computed once when a duplicate first needs it."""
    (known,) = db.execute('SELECT sha256 FROM held WHERE path=?', (path,)).fetchone()
    if known is None:
        known = sha256(path)
        db.execute('UPDATE held SET sha256=? WHERE path=?', (known, path))
    return known


class Catalog:
    """Conversion status of a held file by hash: primary-key lookups only (pipeline, replay_sha256)."""

    def __init__(self):
        self.db = None
        if CATALOG.exists():
            self.db = sqlite3.connect(f'file:{CATALOG}?mode=ro', uri=True, timeout=60)
            self.pipelines = [p for (p,) in self.db.execute('SELECT DISTINCT pipeline FROM conversion_job')]

    def status(self, sha, path):
        if self.db is None:
            return None
        rows = [row for pipeline in self.pipelines for row in self.db.execute(
            'SELECT status,match_key,message FROM conversion_job WHERE pipeline=? AND replay_sha256=?', (pipeline, sha))]
        found = re.fullmatch(r'match_(\d+)(?:_.*)?', Path(path).stem)
        if not rows:
            return {'conversion': 'not queued', 'match_key': f'pgo:{found[1]}' if found else None}
        order = ('done', 'running', 'pending', 'heldout', 'excluded', 'duplicate', 'rejected', 'failed')
        status, key, message = min(rows, key=lambda r: order.index(r[0]) if r[0] in order else len(order))
        return {'conversion': status, 'match_key': key, **({'message': message} if message else {})}


def apply(report, roots, action):
    """Move or delete every non-new file; random-unresolved files are moved even with delete."""
    folders = [Path(r).resolve() for r in roots if Path(r).is_dir()]
    for entry in report:
        if entry['result'].startswith('new'):
            continue
        path = Path(entry['path']).resolve()
        if action == 'delete' and entry['result'] != 'random unresolved':
            path.unlink()
            entry['applied'] = 'deleted'
            continue
        base = next((f for f in folders if f in path.parents), path.parent)
        reason = entry['result'].split(':')[0].replace(' ', '-')
        target = base/'_screened'/reason/path.relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        entry['applied'] = str(target)


def inputs(paths):
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(p for p in path.rglob('*') if p.suffix.lower() == '.rep' and p.is_file()
                              and '_screened' not in p.relative_to(path).parts)
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='Replay files and/or folders (searched recursively)')
    parser.add_argument('--held', nargs='*', type=Path, default=[HELD],
                        help='Folders of replays we already hold (default: replays/)')
    parser.add_argument('--list-new', type=Path, help='Write the paths of new replays here, one per line')
    parser.add_argument('--json', type=Path, help='Write the full per-file report here')
    parser.add_argument('--quiet', action='store_true', help='Only the summary line')
    parser.add_argument('--sides', nargs='+', default=['China'],
                        help='Factions that make a game wanted (default vanilla China; e.g. China "China Tank" "China Nuke")')
    parser.add_argument('--apply', choices=('move', 'delete'),
                        help='move: non-new files to <input folder>/_screened/<reason>/; delete: remove them')
    args = parser.parse_args()

    wanted = set(args.sides)
    unknown = wanted - FACTIONS
    if unknown:
        parser.error(f'unknown sides: {sorted(unknown)}')
    db = open_index()
    added, gone = refresh(db, args.held, args.quiet)
    held = {}  # identity -> held paths, filled on demand
    catalog = Catalog()
    report, readable = [], defaultdict(list)
    for path in inputs(args.paths):
        entry = {'path': str(path)}
        report.append(entry)
        try:
            identity, entry['header'], header = fingerprint(path)
        except (ValueError, OSError) as error:
            entry['result'] = 'not 1v1' if any(text in str(error) for text in NOT_1V1) else 'unreadable'
            entry['reason'] = str(error)
            continue
        readable[identity].append((entry, path, header))
    # First, one copy per game within the input: the longest recording (a player who left early stops theirs).
    keep = []
    for identity, copies in readable.items():
        copies.sort(key=lambda item: (-item[2]['frames'], item[0]['path']))
        keep.append((identity, *copies[0]))
        for entry, _, _ in copies[1:]:
            entry['result'], entry['same_as'] = 'duplicate within input', copies[0][0]['path']
    folders = [Path(r).resolve() for r in args.paths if Path(r).is_dir()]
    for identity, entry, path, header in keep:
        version = unsupported(header)
        if version:
            entry['result'], entry['reason'] = 'unsupported version', version
            continue
        sides, resolved = factions(header, path)
        entry['factions'] = sides
        if not wanted & set(sides) and resolved:
            entry['result'] = 'no vanilla China' if wanted == {'China'} else 'no wanted side'
            continue
        own = sha256(path)
        if identity not in held:
            held[identity] = [p for (p,) in db.execute('SELECT path FROM held WHERE identity=?', (identity,))]
        # The input folder itself is never "held", even when it sits under replays/.
        matches = [p for p in held[identity] if Path(p).resolve() != path.resolve()
                   and not any(f in Path(p).resolve().parents for f in folders)]
        if matches:
            exact = [p for p in matches if held_sha(db, p) == own]
            entry['result'] = 'duplicate: same file' if exact else 'duplicate: same game, other recording'
            first = (exact or matches)[0]
            entry['held'] = first
            entry['catalog'] = catalog.status(held_sha(db, first), first)
        elif not wanted & set(sides):
            # Not held, and a Random that might be China can't be resolved from this file: look manually.
            entry['result'] = 'random unresolved'
        else:
            missing = map_missing(header)
            entry['result'] = 'new: map missing' if missing else 'new'
            if missing:
                entry['reason'] = missing
    db.commit()

    counts = defaultdict(int)
    for entry in report:
        counts[entry['result']] += 1
        if args.quiet:
            continue
        line = f"{entry['result']:38} {entry['path']}"
        if 'header' in entry:
            h = entry['header']
            sides = entry.get('factions') or [p.split(' (')[-1].rstrip(')') for p in h['players']]
            line += f"  [{h['map']}; " + ', '.join(
                f"{p.split(' (')[0]} ({side})" for p, side in zip(h['players'], sides)) + ']'
        if 'reason' in entry:
            line += f"  [{entry['reason']}]"
        if 'held' in entry:
            status = entry.get('catalog') or {}
            line += f"\n{'':38}   held: {entry['held']}  ({status.get('conversion', '?')}"
            line += f", {status['match_key']})" if status.get('match_key') else ')'
        if 'same_as' in entry:
            line += f"\n{'':38}   same game as {entry['same_as']}"
        print(line)
    if args.apply:
        apply(report, args.paths, args.apply)
    if args.list_new:
        args.list_new.write_text(''.join(e['path']+'\n' for e in report if e['result'].startswith('new')), encoding='utf-8')
    if args.json:
        args.json.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({'checked': len(report), **dict(sorted(counts.items())),
                      'held_index': {'refreshed': added, 'forgotten': gone}}), flush=True)


if __name__ == '__main__':
    main()
