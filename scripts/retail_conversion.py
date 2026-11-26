"""Convert the gamereplays.org archive (replays/gamereplays/). Windows only.

Retail Zero Hour 1.04 replays play on the VC6 TheSuperHackers build with the GeneralsAI
exporter (replay-runtime/retail-104/game), at their native 30 Hz; preparation puts them
on the canonical 60 Hz clock (replay_sequences.normalize_demonstration). Quiet rows are
captured on the scheduled-capture controller's request, as in the main pipeline (engine
9a4beb2a, qualified 7 Nov 2026; re-pinned 938e428b for the v6 re-export, 12 Nov 2026; re-pinned 3668fe7d
14 Nov 2026: China Tank and Nuke players exported too, as in the main engine; re-pinned 06911584 15 Nov
2026: observation v11, projectiles, ground hazards, firing/poisoned/burning, superweapon timers, power cues). Replays from a
Generals Online release join that release's queue in the main service's pipeline instead,
unless the crawler already collected the same match.

Custom maps come from replay-runtime/retail-104/maps/<map CRC>/<map folder>/, keyed by the
CRC in the replay header (the engine's MapCache CRC), so two versions of one map never mix.
A replay whose map is not there waits as `excluded` and returns to the queue once it is added.

    python scripts/retail_conversion.py maps <folder> [<folder> ...]   # add maps to the store
    python scripts/retail_conversion.py inventory                       # route and queue
    python scripts/retail_conversion.py run --workers 12                # convert retail jobs
    python scripts/retail_conversion.py report                          # status CSV
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import re
import shutil
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import replay_conversion as rc
from random_sides import RETAIL_RELEASES

SOURCE = rc.ROOT/'replays/gamereplays'
# Second gamereplays.org download (14 Nov 2026): the full crawl's China/Random links, screened to new
# vanilla, Tank and Nuke China 1v1s.
SOURCE3 = rc.ROOT/'replays/gamereplays3'
# GenTool's upload archive (scripts/harvest_gentool.py), one folder per day.
GENTOOL = rc.ROOT/'replays/gentool'
# Expert players' own folders (user-supplied, screened 14 Nov 2026): special/<download>/<player>/*.rep.
# Screened-out files sit under special/_screened/ and are never part of the archive.
SPECIAL = rc.ROOT/'replays/special'


def archive_files():
    return (sorted(SOURCE.glob('*.rep')) + sorted(SOURCE3.glob('*.rep')) + sorted(GENTOOL.glob('*/*.rep'))
            + sorted(p for p in SPECIAL.glob('*/*/*.rep') if p.parts[-3] != '_screened'))


def in_archive(path):
    path = Path(path).resolve()
    return (path.parent in (SOURCE.resolve(), SOURCE3.resolve()) or path.parent.parent == GENTOOL.resolve()
            or (path.parent.parent.parent == SPECIAL.resolve() and path.parent.parent.name != '_screened'))
RETAIL = rc.ROOT/'replay-runtime/retail-104'
RETAIL_ENGINE = RETAIL/'game/generalszh.exe'
# VC6 build of TheSuperHackers 85135752c with the uncommitted GeneralsAI exporter port
# (C:/repos/GeneralsAI/TheSuperHackers); in CRC sync on retail 1.04 replays.
RETAIL_ENGINE_SHA = '06911584c5b02aa18f882ff0be0297bbce58978a24362be23772973257d4f139'
MAPS = RETAIL/'maps'
# Generals Online's handshake: retail shards train a policy deployed on that engine.
DEPLOYMENT_CATALOG = rc.ROOT/'replay-runtime/action-catalog.json'
REPORT = rc.ROOT/'replays/gamereplays-conversion.csv'
RETAIL_INI = 'FEAAE3F3'
SETTINGS = dict(timeout_seconds=2400, audit_level='integrity', reserve_ram_mib=1024, reuse_engine=False)


def map_crc(path):
    """Common/crc.h CRC over the whole file, as MapCache and the replay header compute it."""
    import numpy as np
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    crc = 0
    for byte in data.tolist():
        crc = (((crc << 1) & 0xFFFFFFFF) + byte + (crc >> 31)) & 0xFFFFFFFF
    return crc


def custom_map(header):
    """(folder, CRC) of a user map, or None for a map shipped in the game's archives."""
    # M= is a two-digit hex prefix, then the path: userdata/maps/<folder> or maps/<folder>.
    path = re.sub(r'^[0-9A-Fa-f]{2}(?=(?:userdata/)?maps/)', '', header['map'].replace('\\', '/'), flags=re.I)
    if path.casefold().startswith('maps/'):
        return None
    return path.rsplit('/', 1)[-1], '%08X' % int(header['map_crc'], 16)


@functools.lru_cache(maxsize=None)
def _shipped_crc(path):
    return '%08X' % map_crc(path) if path.is_file() else None


def missing_map(header):
    """Why this replay's map cannot be played here, or None when it can.

    Shipped maps are loose folders in the runtime's Maps/ (it has no MapsZH.big). The
    engine's playback evidence names the map the header asks for, not the file it found,
    so a missing shipped map otherwise plays on without it and diverges or never CRCs.
    """
    wanted = custom_map(header)
    if wanted:
        return None if stored_map(*wanted) else 'Missing map %s %s' % wanted
    path = re.sub(r'^[0-9A-Fa-f]{2}(?=maps/)', '', header['map'].replace('\\', '/'), flags=re.I)
    folder, crc = path.rsplit('/', 1)[-1], '%08X' % int(header['map_crc'], 16)
    shipped = _shipped_crc(RETAIL_ENGINE.parent/'Maps'/folder/(folder+'.map'))
    # CRC 00000000: the recorder wrote none; the shipped map plays and the game's CRC checks verify it.
    if shipped is None or (shipped != crc and crc != '00000000'):
        return 'Missing map %s %s (shipped map, not in the retail runtime)' % (folder, crc)
    return None


def stored_map(folder, crc):
    """The stored folder holding this exact map: the same name if there is one, else any name.

    Players rename map folders (e.g. "[go][rank] vendetta zh v1"); the CRC covers the whole
    .map file, so a folder with this CRC under another name is the same map.
    """
    folders = sorted(p for p in (MAPS/crc).iterdir() if p.is_dir()) if (MAPS/crc).is_dir() else []
    named = [p for p in folders if p.name.casefold() == folder.casefold()]
    return (named or folders or [None])[0]


def add_maps(roots):
    """Copy every map folder under `roots` into the store under its CRC."""
    added = 0
    for root in roots:
        for path in Path(root).rglob('*.map'):
            if path.stem.casefold() != path.parent.name.casefold():
                continue
            target = MAPS/f'{map_crc(path):08X}'/path.parent.name
            if any(p.name.casefold() == path.parent.name.casefold() for p in target.parent.glob('*')):
                continue
            shutil.copytree(path.parent, target)
            added += 1
    return added


def stage_retail_maps(header, profile):
    # Only this replay's map: a different version of the same folder must not linger.
    maps = Path(profile)/'Maps'
    if maps.exists():
        shutil.rmtree(maps)
    missing = missing_map(header)
    if missing:
        raise ValueError(missing)
    wanted = custom_map(header)
    if wanted:
        folder = stored_map(*wanted)
        # The game opens <folder>/<folder>.map by the replay's own folder name.
        target = maps/wanted[0]
        target.mkdir(parents=True)
        for item in folder.iterdir():
            name = wanted[0]+item.suffix if item.stem.casefold() == folder.name.casefold() else item.name
            shutil.copy2(item, target/name)


RETAIL_BACKEND = dict(name='retail-104', engine=RETAIL_ENGINE, ticks_per_second=30, quiet_capture='scheduled',
                      native_ini_crc=RETAIL_INI, logic=False, stage_maps=stage_retail_maps)


def contract():
    from event_store import feature_identity
    from generals_policy import NeuralPolicy
    from replay_training import file_sha256
    if file_sha256(RETAIL_ENGINE) != RETAIL_ENGINE_SHA:
        raise ValueError('Retail executable changed')
    catalog = json.loads(DEPLOYMENT_CATALOG.read_text(encoding='utf-8'))
    if catalog.get('engine_exe_sha256') != rc.ENGINE_SHA:
        raise ValueError('Deployment catalogue is not the pinned Generals Online engine')
    config = json.loads(rc.POLICY.read_text(encoding='utf-8'))
    identity = dict(backend='retail-104', engine_sha256=RETAIL_ENGINE_SHA,
                    engine_path=RETAIL_ENGINE.relative_to(rc.ROOT).as_posix(),
                    feature_key=feature_identity(NeuralPolicy(config, parameters=False)),
                    policy_sha256=file_sha256(rc.POLICY), deployment_catalog_sha256=file_sha256(DEPLOYMENT_CATALOG),
                    rules={}, storage='binary', interval_ms=100, quiet_capture='scheduled', ticks_per_second=30)
    identity['pipeline'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity['source_pins'] = {name: file_sha256(Path(__file__).with_name(name))
                               for name in rc.source_closure('retail_conversion.py')}
    return identity


def collected_matches(release, wanted):
    """Which (seed, map CRC, players) keys the crawler already holds for one GO release."""
    from harvest_china_replays import parse_replay
    found = set()
    for path in (rc.RAW/release).glob('*.rep'):
        with path.open('rb') as stream:
            data = stream.read(8192) + b'\0'*16
        try:
            key = match_identity(parse_replay(data))
        except ValueError:
            continue
        if key in wanted:
            found.add(key)
    return found


def match_identity(header):
    options = dict(x.split('=', 1) for x in header['game_options'].split(';') if '=' in x)
    return options.get('SD'), header['map_crc'], tuple(sorted(p['name'] for p in header['players']))


def inventory(db, identity):
    """Route every archive replay; each gets one conversion_job row in its pipeline."""
    from harvest_china_replays import parse_replay
    from replay_bootstrap import match_key, partition
    from replay_catalog import open_failures
    from replay_training import file_sha256
    retail, online = identity['pipeline'], rc.contract()['pipeline']
    # Rows for this archive's files; another source with the same bytes makes a file a duplicate.
    known, elsewhere = {}, set()
    for sha, pipeline, source in db.execute(
            'SELECT replay_sha256,pipeline,source_path FROM conversion_job WHERE pipeline IN (?,?)', (retail, online)):
        if in_archive(source):
            known[sha] = pipeline
        else:
            elsewhere.add(sha)
    # One match recorded by both players, or found in both archives, is converted once.
    retail_matches = {key for (key,) in db.execute('SELECT match_key FROM conversion_job WHERE pipeline=?', (retail,))}
    failed = open_failures(db)
    collected = {sha for (sha,) in db.execute("SELECT raw_sha256 FROM replay WHERE raw_status='downloaded'")}
    rows, go = [], {}
    for path in archive_files():
        sha = file_sha256(path)
        header = parse_replay(path.read_bytes())
        rows.append((path, sha, header))
        if header['exe_crc'] not in RETAIL_RELEASES and sha not in known:
            go.setdefault(header['exe_crc'], {})[match_identity(header)] = sha
    duplicates = set()
    for release, keys in go.items():
        if (rc.RAW/release).is_dir():
            duplicates |= {keys[k] for k in collected_matches(release, set(keys))}
    added = dict(retail=0, online=0, duplicate=0, excluded=0)
    for path, sha, header in rows:
        key = match_key(header, replay_path=None if header['exe_crc'] in RETAIL_RELEASES else path)
        if sha in known:
            if known[sha] == retail:
                reopen_if_mapped(db, retail, sha, header)
            continue
        release = header['exe_crc']
        if release in RETAIL_RELEASES:
            pipeline, status, message = retail, 'pending', None
            missing = missing_map(header)
            if missing:
                status, message = 'excluded', missing
            elif header['ini_crc'] != RETAIL_INI:
                status, message = 'rejected', 'Modified game data: INI CRC %s (retail 1.04 is %s)' % (header['ini_crc'], RETAIL_INI)
            if key in retail_matches:
                status, message = 'duplicate', 'Same match as another archive recording'
            retail_matches.add(key)
        elif (rc.RAW/release).is_dir():
            pipeline, status, message = online, 'pending', None
            if sha in collected or sha in duplicates or sha in elsewhere:
                # Kept out of the Generals Online queue, which has its own copy.
                pipeline, status, message = retail, 'duplicate', 'Match already collected from Generals Online'
        else:
            pipeline, status, message = retail, 'excluded', 'Unknown release '+release
        if status == 'pending' and sha in failed:
            status, message = 'rejected', 'Known %s failure: %s' % failed[sha]
        db.execute('''INSERT OR IGNORE INTO conversion_job
            (pipeline,replay_sha256,match_key,source_path,split,priority,status,stage,message,updated)
            VALUES(?,?,?,?,?,?,?,?,?,?)''', (pipeline, sha, key, str(path.resolve()), partition(key), 0, status,
            'queued', message, time.time()))
        added['duplicate' if status == 'duplicate' else 'excluded' if status == 'excluded'
              else 'retail' if pipeline == retail else 'online'] += 1
    db.commit()
    return added


def reopen_if_mapped(db, pipeline, sha, header):
    row = db.execute('SELECT status,message FROM conversion_job WHERE pipeline=? AND replay_sha256=?', (pipeline, sha)).fetchone()
    if row['status'] == 'excluded' and (row['message'] or '').startswith('Missing map') and missing_map(header) is None:
        rc.update(db, pipeline, sha, status='pending', stage='queued', message=None)


# A side converter: its engine's folder is staged per worker and every replay runs in a fresh
# engine process.
RETAIL_SIDE = dict(backend=RETAIL_BACKEND, settings=SETTINGS, work='retail-workers')


def worker_main(slot, inbox, outbox, identity, side):
    if inbox.get() != 'boot':
        return
    import gc
    from engine_runtime import clone_file, disable_power_throttling, isolation_profile
    from replay_training import file_sha256
    from scheduled_capture import EngineSession

    def emit(**event):
        outbox.put(dict(slot=slot, pid=os.getpid(), **event))
    disable_power_throttling()
    engine = side['backend']['engine']
    directory = rc.WORK/side['work']/str(slot)
    directory.mkdir(parents=True, exist_ok=True)
    runtime, profile = directory/'runtime', directory/'profile'
    session = EngineSession()  # never started: side exports run one process per replay
    try:
        rc.source_guard(identity)
        if not runtime.exists():
            def ignore(parent, names):
                return [n for n in names if (Path(n).suffix.lower() == '.exe' and n != engine.name)
                        or Path(n).suffix.lower() in ('.log', '.dmp', '.tmp')]
            shutil.copytree(engine.parent, runtime, copy_function=clone_file, ignore=ignore)
        if file_sha256(runtime/engine.name) != identity['engine_sha256']:
            raise ValueError('Staged engine differs')
        isolation_profile(None, profile)
        os.environ['GENERALS_AI_USER_DATA'] = str(profile)
        catalog = directory/'action-catalog.json'
        shutil.copyfile(DEPLOYMENT_CATALOG, catalog)
        emit(event='ready')
        while True:
            job = inbox.get()
            if job is None:
                break
            result = rc.run_job(job, identity, side['settings'], runtime, profile, catalog, session, emit,
                                directory/'worker.log', side['backend'])
            gc.collect()
            emit(event='result', sha=job['replay_sha256'], result=result)
    except BaseException as error:
        emit(event='fatal', message=repr(error))
        raise


def run(identity, workers, limit=0, side=RETAIL_SIDE):
    from replay_converter import ProcessFamily, record_job_failure
    from replay_catalog import resolve_failures
    pipeline = identity['pipeline']
    db = rc.connect()
    rc.recover(db, pipeline)
    context = mp.get_context('spawn')
    outbox = context.Queue()
    children, finished, started = {}, 0, time.time()
    try:
        for slot in range(workers):
            inbox = context.Queue()
            process = context.Process(target=worker_main, args=(slot, inbox, outbox, identity, side), daemon=True)
            process.start()
            family = ProcessFamily(process.pid)
            inbox.put('boot')
            children[slot] = dict(process=process, family=family, inbox=inbox, job=None, ready=False)
        claimed, exhausted = 0, False
        while True:
            for slot, child in children.items():
                if child['ready'] and child['job'] is None and not exhausted:
                    job = rc.claim(db, pipeline, '%s-%d' % (side['work'], slot)) if not limit or claimed < limit else None
                    if job:
                        child['job'] = job
                        child['inbox'].put(job)
                        claimed += 1
                    else:
                        exhausted = True
            if exhausted and not any(child['job'] for child in children.values()):
                break
            try:
                event = outbox.get(timeout=5)
            except queue.Empty:
                if any(not c['process'].is_alive() for c in children.values()):
                    raise RuntimeError('A retail worker exited')
                continue
            child = children[event['slot']]
            if event['event'] == 'ready':
                child['ready'] = True
            elif event['event'] == 'fatal':
                raise RuntimeError('Worker %d: %s' % (event['slot'], event['message']))
            elif event['event'] == 'result':
                job, result = child['job'], event['result']
                if job is None or event['sha'] != job['replay_sha256']:
                    raise RuntimeError('Worker result does not own its queue job')
                child['job'] = None
                fields = {k: result[k] for k in ('status', 'stage', 'message') if k in result}
                fields.update(owner=None, retry_after=time.time()+30 if result['status'] == 'pending' else 0)
                if result.get('failure'):
                    record_job_failure(db, identity, job, result['message'], **result['failure'])
                elif result['status'] == 'done':
                    resolve_failures(db, job['replay_sha256'])
                rc.update(db, pipeline, job['replay_sha256'], **fields)
                finished += 1
                print('%5d %6.0fs %-8s %s %s' % (finished, time.time()-started, result['status'],
                      Path(job['source_path']).name, (result.get('message') or '')[:120]), flush=True)
    finally:
        for child in children.values():
            child['inbox'].put(None)
        for child in children.values():
            child['process'].join(10)
            child['family'].close()
        rc.recover(db, pipeline)
        db.close()
    return finished


def report(identity):
    """Every archive replay with its route and outcome; `delete` marks the rejected ones."""
    db = rc.connect()
    pipelines = {identity['pipeline']: 'retail-104', rc.contract()['pipeline']: 'generalsonline'}
    rows = db.execute('SELECT pipeline,replay_sha256,match_key,source_path,split,status,message FROM conversion_job '
                      'WHERE pipeline IN (?,?)', tuple(pipelines)).fetchall()
    db.close()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('w', newline='', encoding='utf-8') as stream:
        out = csv.writer(stream)
        out.writerow(['file', 'replay_sha256', 'match_key', 'engine', 'split', 'status', 'delete', 'message'])
        for row in sorted(rows, key=lambda r: Path(r['source_path']).name):
            if not in_archive(row['source_path']):
                continue
            # A file the replay reader cannot parse to its end is broken, not merely unconverted.
            delete = row['status'] in ('rejected', 'duplicate') or (row['message'] or '').startswith('Truncated replay')
            out.writerow([Path(row['source_path']).name, row['replay_sha256'], row['match_key'], pipelines[row['pipeline']],
                          row['split'], row['status'], 'yes' if delete else '', row['message'] or ''])
    counts = {}
    for row in rows:
        if in_archive(row['source_path']):
            counts[(pipelines[row['pipeline']], row['status'])] = counts.get((pipelines[row['pipeline']], row['status']), 0)+1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('maps').add_argument('roots', nargs='+')
    sub.add_parser('inventory')
    runner = sub.add_parser('run')
    runner.add_argument('--workers', type=int, default=12)
    runner.add_argument('--limit', type=int, default=0)
    sub.add_parser('report')
    args = parser.parse_args(argv)
    # Replay names can hold characters the Windows console code page can't encode; printing one killed whole runs
    # (UnicodeEncodeError, 24 Nov). The log is UTF-8 either way.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from replay_converter import is_running
    if args.command == 'maps':
        print(json.dumps(dict(added=add_maps(args.roots))))
        return
    identity = contract()
    if args.command in ('inventory', 'run') and is_running():
        raise SystemExit('Stop the replay converter first: both write the conversion queue')
    if args.command == 'inventory':
        db = rc.connect()
        try:
            print(json.dumps(inventory(db, identity)))
        finally:
            db.close()
    elif args.command == 'run':
        print(json.dumps(dict(finished=run(identity, args.workers, args.limit))))
    else:
        print(json.dumps({'%s %s' % k: v for k, v in sorted(report(identity).items())}, indent=1))


if __name__ == '__main__':
    main()
