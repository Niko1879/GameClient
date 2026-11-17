"""Who won each retail 1.04 replay, by playing it back on the VC6 engine (16 Nov 2026).

Replays store no result. The engine's own multiplayer victory conditions decide it during playback (a player
with no victory-counting buildings left is defeated; a surrender/quit self-destructs the player's assets), and
the outcome build prints them after the replay (TheSuperHackers branch ai, Core ReplaySimulation.cpp,
`ReplayOutcome {...}`: per header slot victorious / eliminated / defeated_now / disconnected /
last_frame_in_game, plus the frame the game was decided). Playback only: no training export, no shards.

The engine is pinned separately from the export engine (OUTCOME_ENGINE); each worker plays from a cloned copy of
the retail runtime with its own copy of that exe, staging the replay's map the way the retail converter does.

Output: replays/catalog/replay-outcomes.jsonl, one line per replay file, appended and resumable (keyed by the
file's SHA-256). `winner_slot` is the single victorious slot, else null (the game ended undecided, e.g. the
recording stopped before anyone was defeated).

usage:
  python -B scripts/replay_outcomes.py run [--workers 20] [--limit N] [--paths FILE...]
  python -B scripts/replay_outcomes.py summary
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
OUT = ROOT/'replays/catalog/replay-outcomes.jsonl'
WORK = ROOT/'.cache/replay-outcomes'
OUTCOME_ENGINE = ROOT/'replay-runtime/retail-104/outcome/generalszh.exe'
TIMEOUT = 3600


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def pinned_sha():
    marker = OUTCOME_ENGINE.with_suffix('.sha256')
    return marker.read_text().strip() if marker.exists() else None


def done_hashes():
    done = set()
    if OUT.exists():
        with OUT.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    done.add(json.loads(line)['replay_sha256'])
    return done


def worker_init(slots):
    global RUNTIME, PROFILE, SLOT
    from engine_runtime import clone_file, disable_power_throttling, isolation_profile
    import retail_conversion as retail
    disable_power_throttling()
    SLOT = slots.get()
    directory = WORK/str(SLOT)
    RUNTIME, PROFILE = directory/'runtime', directory/'profile'
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)

    def ignore(parent, names):  # no exe is cloned: the pinned export engine must never be a hard link here
        return [n for n in names if Path(n).suffix.lower() in ('.exe', '.bak', '.log', '.dmp', '.tmp')]
    shutil.copytree(retail.RETAIL_ENGINE.parent, RUNTIME, copy_function=clone_file, ignore=ignore)
    shutil.copyfile(OUTCOME_ENGINE, RUNTIME/'generalszh.exe')  # a real copy, never a link
    isolation_profile(None, PROFILE)


def play(path):
    """One replay -> result row (never raises: failures are rows with an error)."""
    import replay_conversion as rc
    import retail_conversion as retail
    from harvest_china_replays import parse_replay
    from random_sides import RETAIL_RELEASES
    from replay_bootstrap import match_key
    from scheduled_capture import run_engine
    path = Path(path)
    row = {'path': str(path.relative_to(ROOT)), 'replay_sha256': sha256(path), 'engine_sha256': pinned_sha()}
    try:
        header = parse_replay(path.read_bytes())
        row.update(match_key=match_key(header, None, path), exe_crc=header['exe_crc'],
                   players=[{'slot': p['slot'], 'name': p['name']} for p in header['players']])
        if str(header['exe_crc']).upper() not in RETAIL_RELEASES:
            return dict(row, status='skipped', message='Not a retail 1.04 release')
        if str(header['ini_crc']).upper() != retail.RETAIL_INI:
            return dict(row, status='skipped', message='Modified INI')
        if header['desync']:
            return dict(row, status='skipped', message='Header flags a desync')
        retail.stage_retail_maps(header, PROFILE)
        log = WORK/str(SLOT)/'engine.log'
        command = [str(RUNTIME/'generalszh.exe'), '-headless', '-replay', os.path.relpath(path, PROFILE/'Replays')]
        os.environ['GENERALS_AI_USER_DATA'] = str(PROFILE)
        started = time.time()
        returncode = run_engine(command, cwd=RUNTIME, log_path=log, timeout=TIMEOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        text = log.read_text(encoding='utf-8', errors='replace')
        validation = [json.loads(l[len('ReplayValidation '):]) for l in text.splitlines() if l.startswith('ReplayValidation ')]
        outcome = [json.loads(l[len('ReplayOutcome '):]) for l in text.splitlines() if l.startswith('ReplayOutcome ')]
        row.update(returncode=returncode, seconds=round(time.time()-started, 1),
                   validation=validation[-1] if validation else None, outcome=outcome[-1] if outcome else None)
        if not outcome:
            return dict(row, status='failed', message='No ReplayOutcome line')
        winners = [p['slot'] for p in outcome[-1]['players'] if p['victorious']]
        row['winner_slot'] = winners[0] if len(winners) == 1 else None
        return dict(row, status='played')
    except subprocess.TimeoutExpired:
        return dict(row, status='failed', message='timeout')
    except Exception as error:
        return dict(row, status='failed', message=repr(error))


def run(args):
    import retail_conversion as retail
    expected = pinned_sha()
    if not expected or sha256(OUTCOME_ENGINE) != expected:
        raise SystemExit('Outcome engine missing or changed: run `pin` first')
    done = done_hashes()
    files = [Path(p).resolve() for p in args.paths] if args.paths else retail.archive_files()
    todo = [p for p in files if p.exists()]
    if done:
        todo = [p for p in todo if sha256(p) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print('replays to play', len(todo), flush=True)
    context = mp.get_context('spawn')
    slots = context.Queue()
    for slot in range(args.workers):
        slots.put(slot)
    counts, started = {}, time.time()
    with context.Pool(args.workers, initializer=worker_init, initargs=(slots,), maxtasksperchild=None) as pool, \
            OUT.open('a', encoding='utf-8') as out:
        for n, row in enumerate(pool.imap_unordered(play, [str(p) for p in todo]), 1):
            out.write(json.dumps(row, ensure_ascii=False)+'\n')
            out.flush()
            counts[row['status']] = counts.get(row['status'], 0)+1
            if n % 50 == 0 or n == len(todo):
                rate = n/(time.time()-started)*3600
                print('%d/%d %s %.0f/h' % (n, len(todo), counts, rate), flush=True)


def pin(_):
    OUTCOME_ENGINE.parent.mkdir(parents=True, exist_ok=True)
    built = ROOT.parent/'TheSuperHackers/build/vc6/GeneralsMD/generalszh.exe'
    shutil.copyfile(built, OUTCOME_ENGINE)
    digest = sha256(OUTCOME_ENGINE)
    OUTCOME_ENGINE.with_suffix('.sha256').write_text(digest+'\n')
    print('pinned', OUTCOME_ENGINE, digest)


def summary(_):
    rows = [json.loads(line) for line in OUT.open(encoding='utf-8') if line.strip()]
    status, decided, how = {}, 0, {}
    for row in rows:
        status[row['status']] = status.get(row['status'], 0)+1
        if row['status'] == 'played':
            decided += row.get('winner_slot') is not None
            end = 'decided' if row.get('winner_slot') is not None else 'undecided'
            how[end] = how.get(end, 0)+1
    print(json.dumps({'replays': len(rows), 'status': status, 'played': how}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--workers', type=int, default=20)
    r.add_argument('--limit', type=int, default=0)
    r.add_argument('--paths', nargs='*')
    sub.add_parser('pin', help='copy the VC6 build output to OUTCOME_ENGINE and record its SHA-256')
    sub.add_parser('summary')
    args = parser.parse_args()
    {'run': run, 'pin': pin, 'summary': summary}[args.command](args)


if __name__ == '__main__':
    main()
