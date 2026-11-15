"""Full validation pass over every prepared shard the trainer can select.

Run once the export is complete and every converter is stopped:

    & .venv-torch-rocm/Scripts/python.exe -B scripts/validate_shards.py

Scope is exactly run_prepared_training.selection's candidate query: available
prepared_shard rows with the current feature key, from the pinned, retail and
GO 062026 engines. Each shard goes through the trainer's own read path
(EventDataset.read: whole-archive SHA-256, member sizes and hashes, column
decoding, row assembly, enemy identity and ratings) and then the learner's CPU
preparation (training_preparation.prepare_update, which rejects targets outside
their legality masks), plus a finiteness check of every float array it builds.
Catalogue, index and zip-inventory checks run alongside.

The pass only reads. Results go to .cache/shard-validation/: results.jsonl (one
line per shard, appended as it finishes, so an interrupted run resumes; --fresh
starts over), invalid.tsv and summary.json. Exit status 1 if any shard fails.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import zipfile

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'training-data'
CATALOG = ROOT/'replays/catalog/replay-catalog.sqlite3'
OUT = ROOT/'.cache/shard-validation'
CONVERTER_SCRIPTS = ('replay_converter.py', 'retail_conversion.py', 'event_ingest.py')
# A validation result is only reused while the code that decides it is unchanged.
VALIDATOR_SOURCES = ('validate_shards.py', 'event_dataset.py', 'training_preparation.py', 'training_legality.py',
                     'tensor_replay_store.py', 'generals_policy.py', 'enemy_side.py', 'prepared_io.py')
MAX_PROBLEMS = 20


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def running_converters():
    """Command lines of live converter processes: the catalogue must be quiet (slow reads crash writers)."""
    found = []
    sys.path.insert(0, str(ROOT/'scripts'))
    from replay_converter import is_running
    if is_running():
        found.append('replay_converter.py service (service.lock held)')
    if os.name == 'nt':
        script = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                  "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }")
        listing = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True, text=True,
                                 creationflags=subprocess.CREATE_NO_WINDOW).stdout
        for line in listing.splitlines():
            pid, _, command = line.partition(' ')
            if pid != str(os.getpid()) and any(name in command for name in CONVERTER_SCRIPTS) and ' status' not in command:
                found.append(line.strip())
    return found


# ---------------------------------------------------------------- catalogue pass

def catalogue_pass(limit=None):
    """(jobs for the workers, catalogue-level report). Primary-key reads plus one candidate scan."""
    from event_store import feature_identity
    from generals_policy import NeuralPolicy
    from replay_catalog import document_bytes
    from replay_cleaning import player_ratings, selection_filter
    from retail_conversion import RETAIL_ENGINE_SHA
    import run_prepared_training as runner

    config = json.loads((ROOT/'replay-runtime/policy.json').read_text(encoding='utf-8-sig'))
    key = feature_identity(NeuralPolicy(config, parameters=False))
    live_engine = file_digest(runner.ENGINE)
    engines = {live_engine: 'pinned', RETAIL_ENGINE_SHA: 'retail'}

    db = sqlite3.connect(f'file:{CATALOG.as_posix()}?mode=ro', uri=True, timeout=60)
    db.row_factory = sqlite3.Row
    try:
        out_of_scope = Counter()
        for row in db.execute('SELECT feature_key,engine_sha256,count(*) AS n FROM prepared_shard '
                              'WHERE available=1 GROUP BY feature_key,engine_sha256'):
            if row['feature_key'] != key or row['engine_sha256'] not in engines:
                out_of_scope[f"{row['feature_key'][:12]}/{(row['engine_sha256'] or 'none')[:12]}"] += row['n']
        rows = [dict(row) for row in db.execute(
            'SELECT * FROM prepared_shard WHERE available=1 AND feature_key=? AND engine_sha256 IN (?,?)',
            (key, *engines))]
        if limit is not None:  # A spread-out sample: hash order mixes engines and sizes.
            rows = sorted(rows, key=lambda row: row['archive_sha256'])[:limit]
        registered = {Path(r[0]).name for r in db.execute('SELECT storage_path FROM prepared_shard')}
        protected = {r[0] for r in db.execute("SELECT match_key FROM replay_assignment WHERE split='test'")}
        profiles = {}
        for profile_id in {row['profile_id'] for row in rows}:
            sha = db.execute('SELECT document_sha256 FROM prepared_profile WHERE profile_id=?', (profile_id,)).fetchone()[0]
            profiles[profile_id] = json.loads(document_bytes(db, sha))
        dropped, scanned, cleaning = selection_filter()
        ratings = player_ratings()
        protected.update(r['match_key'] for r in json.loads((ROOT/'replays/test-reserve.json').read_text())['matches'])

        jobs, by_match, by_replay = [], defaultdict(list), defaultdict(list)
        for row in rows:
            problems = []
            path = Path(row['storage_path'])
            path = path if path.is_absolute() else ROOT/path
            entry = None
            try:
                if path.parent.resolve() != DATA.resolve():
                    problems.append(f'catalogue: shard outside training-data ({path})')
                elif not path.is_file():
                    problems.append('catalogue: shard file missing')
                elif path.stat().st_size != row['bytes']:
                    problems.append(f"catalogue: size {path.stat().st_size} != registered {row['bytes']}")
                entry = json.loads(document_bytes(db, row['entry_document_sha256']))
                index = entry['index']
                source = index['source']
                if entry['archive_sha256'] != row['archive_sha256']:
                    problems.append('entry: archive_sha256 differs from catalogue row')
                if entry['index_sha256'] != row['index_sha256']:
                    problems.append('entry: index_sha256 differs from catalogue row')
                if entry['match_key'] != row['match_key']:
                    problems.append('entry: match_key differs from catalogue row')
                if source.get('replay_sha256') != row['replay_sha256']:
                    problems.append('entry: replay_sha256 differs from catalogue row')
                if index.get('legality_key') != key:
                    problems.append('entry: index legality_key is not the current feature key')
                if source.get('match_key') != row['match_key']:
                    problems.append('entry: source match_key differs')
                original = source.get('original_replay')
                if not original or not Path(original).is_file():
                    problems.append(f'source: original replay missing ({original})')
                if not index.get('episodes'):
                    problems.append('index: no episodes')
            except Exception as error:  # noqa: BLE001 - every failure is a finding
                problems.append(f'catalogue: {type(error).__name__}: {error}')
            # Every shard trains under the live engine's deployment (run_prepared_training.document
            # refuses a selection that mixes deployments), whichever engine exported it.
            deployment = profiles[row['profile_id']].get('deployment') or {}
            if deployment.get('engine_sha256') != live_engine:
                problems.append(f"profile: deployment engine {str(deployment.get('engine_sha256'))[:12]} "
                                f"is not the live engine {live_engine[:12]}")
            if entry is not None:
                entry = dict(entry, path=path.name, split='train', ratings=ratings.get(row['match_key'], {}))
            by_match[row['match_key']].append(row['archive_sha256'])
            by_replay[row['replay_sha256']].append(row['archive_sha256'])
            jobs.append(dict(archive_sha256=row['archive_sha256'], match_key=row['match_key'], path=str(path),
                             bytes=row['bytes'], engine=engines[row['engine_sha256']], profile_id=row['profile_id'],
                             entry=entry, catalogue_problems=problems, cleaning_scanned=row['match_key'] in scanned,
                             cleaning_dropped=dropped.get(row['match_key']),
                             protected=row['match_key'] in protected))
    finally:
        db.close()

    deployments = Counter(json.dumps(p.get('deployment'), sort_keys=True) for p in profiles.values())
    catalogs = Counter(p.get('capability_catalog_sha256') for p in profiles.values())
    # Leftovers in training-data: names only (stat on 139k entries is slow).
    leftovers, unregistered = [], []
    with os.scandir(DATA) as entries:
        for item in entries:
            name = item.name
            if name.endswith('.part') or name.startswith('.conversion-'):
                leftovers.append(name)
            elif name.endswith('.zip') and name not in registered:
                unregistered.append(name)
    report = dict(
        feature_key=key, engines={v: k for k, v in engines.items()}, candidates=len(rows),
        available_out_of_scope=dict(out_of_scope),
        duplicate_match_keys={k: v for k, v in by_match.items() if len(v) > 1},
        duplicate_replays={k: v for k, v in by_replay.items() if len(v) > 1},
        profiles=len(profiles), deployments=len(deployments), capability_catalogs=len(catalogs),
        cleaning=dict(cleaning, dropped_candidates=sum(1 for j in jobs if j['cleaning_dropped'])),
        protected_candidates=sum(1 for j in jobs if j['protected']),
        leftovers=dict(count=len(leftovers), sample=leftovers[:20]),
        unregistered_shards=dict(count=len(unregistered), sample=unregistered[:20]))
    corpus_errors = []
    unscanned = sum(1 for job in jobs if not job['cleaning_scanned'])
    if unscanned:
        corpus_errors.append(f'{unscanned} candidates lack cleaning metrics; run scripts/replay_cleaning.py scan '
                             '(run_prepared_training.selection refuses the corpus until then)')
    report['unscanned_for_cleaning'] = unscanned
    report['corpus_errors'] = corpus_errors
    return jobs, report, config, key


# ---------------------------------------------------------------- shard pass (workers)

_SPEC = _KEY = _BUDGET = None


def _initialize(config, key, pair_budget):
    global _SPEC, _KEY, _BUDGET
    sys.path.insert(0, str(ROOT/'scripts'))
    from engine_runtime import disable_power_throttling
    disable_power_throttling()  # Spawned workers are windowless: opt out of EcoQoS (not inherited).
    from training_preparation import PreparationSpec
    _SPEC, _KEY, _BUDGET = PreparationSpec(config, key), key, pair_budget


def _non_finite(value, where, problems):
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'f' and not np.isfinite(value).all():
            problems.append(f'{where}: non-finite values')
    elif isinstance(value, dict):
        for name, item in value.items():
            _non_finite(item, f'{where}.{name}', problems)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _non_finite(item, where, problems)


def check_shard(job):
    from enemy_side import match_factions
    from event_dataset import EventDataset
    from event_store import SCHEMA
    from training_preparation import prepare_update

    started = time.perf_counter()
    problems, rows_seen, chunks_seen = list(job['catalogue_problems']), 0, 0
    entry = job['entry']

    def result():
        return dict(archive_sha256=job['archive_sha256'], match_key=job['match_key'], path=job['path'],
                    engine=job['engine'], profile_id=job['profile_id'], valid=not problems,
                    problems=problems[:MAX_PROBLEMS], problem_count=len(problems), rows=rows_seen, chunks=chunks_seen,
                    cleaning_dropped=job['cleaning_dropped'], cleaning_scanned=job['cleaning_scanned'],
                    protected=job['protected'], feature_key=_KEY,
                    seconds=round(time.perf_counter()-started, 3))

    if entry is None or not Path(job['path']).is_file():
        return result()
    index = entry['index']
    try:
        if index.get('schema') != SCHEMA or index.get('kind') != 'generals_event_shard':
            problems.append('index: wrong schema or kind')
        if index.get('float_dtype') != 'float32':
            problems.append(f"index: float_dtype {index.get('float_dtype')}")
        expected = {'index.json'} | {item['member'] for item in index['shared'].values()}
        for number, episode in enumerate(index['episodes']):
            if episode['shared'] not in index['shared']:
                problems.append(f'index: episode {number} references missing shared block')
            start = 0
            for chunk in episode['chunks']:
                expected.add(chunk['member'])
                if chunk['start'] != start or chunk['rows'] < 1:
                    problems.append(f"index: episode {number} chunk {chunk['member']} start/rows inconsistent")
                start += chunk['rows']
            if start != episode['length'] or start < 1:
                problems.append(f"index: episode {number} length {episode['length']} != chunk rows {start}")
        with zipfile.ZipFile(job['path']) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                problems.append('zip: duplicate member names')
            if set(names) != expected:
                missing, extra = sorted(expected-set(names)), sorted(set(names)-expected)
                problems.append(f'zip: members differ from index (missing {missing[:3]}, extra {extra[:3]})')
            if hashlib.sha256(archive.read('index.json')).hexdigest() != entry['index_sha256']:
                problems.append('zip: index.json does not match the registered index proof')
    except Exception as error:  # noqa: BLE001
        problems.append(f'structure: {type(error).__name__}: {error}')
    if problems:
        return result()

    # The trainer's own reader, over a one-entry dataset (EventDataset.__init__ needs a manifest file).
    dataset = object.__new__(EventDataset)
    dataset.path = DATA/'.shard-validation.json'
    dataset.entries, dataset.key, dataset.verified, dataset.shared = [entry], _KEY, {}, {}
    try:
        factions = match_factions(index['source']['original_replay'])
        entry['faction_identities'] = ([list(_SPEC.reference.encoder.identity.player_identity(f)) for f in factions]
                                       if factions else None)
    except Exception as error:  # noqa: BLE001
        problems.append(f'enemy identity: {type(error).__name__}: {error}')
        return result()
    for episode in index['episodes']:
        for descriptor in episode['chunks']:
            where = descriptor['member']
            try:
                rows = dataset.read((0, episode['shared'], descriptor))
                chunks_seen += 1; rows_seen += len(rows)
                if any(row['player'] != episode['player'] for row in rows):
                    problems.append(f'{where}: row player differs from episode player')
                prepared = prepare_update(_SPEC, rows, _BUDGET)
                if prepared['rows'] != len(rows):
                    problems.append(f'{where}: preparation dropped rows')
                for number, part in enumerate(prepared['microbatches']):
                    _non_finite(part, f'{where}#{number}', problems)
            except Exception as error:  # noqa: BLE001
                problems.append(f'{where}: {type(error).__name__}: {error}')
                if 'integrity mismatch' in str(error):
                    return result()  # The whole archive is wrong; no point reading on.
            if len(problems) >= MAX_PROBLEMS:
                return result()
    return result()


# ---------------------------------------------------------------- driver

def problem_kind(problem):
    """'catalogue', 'zip', ... or, for a chunk failure, the error itself without the member name."""
    head, _, rest = problem.partition(': ')
    if head.startswith(('chunks/', 'shared/')):
        return rest[:100]
    return head

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)-2))
    parser.add_argument('--pair-budget', type=int, default=8_000_000, help='as run_prepared_training passes the trainer')
    parser.add_argument('--limit', type=int, help='validate only this many shards (smoke test)')
    parser.add_argument('--allow-live', action='store_true',
                        help='with --limit only: run while converters are live (bounded, quick catalogue reads)')
    parser.add_argument('--fresh', action='store_true', help='ignore earlier results and start over')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT/'scripts'))
    from engine_runtime import disable_power_throttling
    disable_power_throttling()
    live = running_converters()
    if args.allow_live and args.limit is None:
        parser.error('--allow-live needs --limit: a full pass reads every entry document from the catalogue')
    if live and not args.allow_live:
        print('Refusing to run: converters are live and catalogue scans can crash them:\n  '+'\n  '.join(live))
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print('Catalogue pass...', flush=True)
    jobs, report, config, key = catalogue_pass(args.limit)
    print(f"{report['candidates']} candidate shards for feature key {key[:12]}", flush=True)

    validator = {name: file_digest(ROOT/'scripts'/name) for name in VALIDATOR_SOURCES}
    identity = hashlib.sha256(json.dumps([key, validator, args.pair_budget], sort_keys=True).encode()).hexdigest()
    results_path = args.output/'results.jsonl'
    state_path = args.output/'run.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() and not args.fresh else {}
    done = {}
    if previous.get('identity') == identity and results_path.exists():
        with results_path.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A line torn by an interruption is simply redone.
                done[item['archive_sha256']] = item
    else:
        results_path.unlink(missing_ok=True)
    save(state_path, dict(identity=identity, feature_key=key, validator_sources=validator, pair_budget=args.pair_budget,
                          started=started))

    wanted = {job['archive_sha256'] for job in jobs}
    done = {sha: item for sha, item in done.items() if sha in wanted}
    pending = [job for job in jobs if job['archive_sha256'] not in done]
    pending.sort(key=lambda job: job['bytes'], reverse=True)  # Big shards first keeps the tail short.
    print(f'{len(done)} already validated, {len(pending)} to go, {args.workers} workers', flush=True)

    if pending:
        context = multiprocessing.get_context('spawn')
        total_bytes = sum(job['bytes'] for job in pending)
        finished_bytes, invalid_now, last = 0, 0, time.time()
        with context.Pool(args.workers, initializer=_initialize, initargs=(config, key, args.pair_budget)) as pool, \
                results_path.open('a', encoding='utf-8') as stream:
            sizes = {job['archive_sha256']: job['bytes'] for job in pending}
            for number, item in enumerate(pool.imap_unordered(check_shard, pending, chunksize=1), 1):
                stream.write(json.dumps(item)+'\n'); stream.flush()
                done[item['archive_sha256']] = item
                finished_bytes += sizes[item['archive_sha256']]
                invalid_now += not item['valid']
                if not item['valid']:
                    print(f"INVALID {Path(item['path']).name}: {item['problems'][0]}", flush=True)
                if time.time()-last > 30 or number == len(pending):
                    last = time.time()
                    rate = finished_bytes/max(last-started, 1e-9)
                    eta = (total_bytes-finished_bytes)/max(rate, 1e-9)
                    print(f'{number}/{len(pending)} shards, {finished_bytes/2**30:.1f}/{total_bytes/2**30:.1f} GiB, '
                          f'{invalid_now} invalid, ETA {eta/3600:.1f} h', flush=True)

    results = list(done.values())
    invalid = [item for item in results if not item['valid']]
    with (args.output/'invalid.tsv').open('w', encoding='utf-8') as stream:
        stream.write('archive_sha256\tmatch_key\tpath\tproblem_count\tfirst_problem\n')
        for item in sorted(invalid, key=lambda item: item['match_key']):
            stream.write(f"{item['archive_sha256']}\t{item['match_key']}\t{item['path']}\t{item['problem_count']}\t"
                         f"{item['problems'][0]}\n")
    kinds = Counter(problem_kind(item['problems'][0]) for item in invalid)
    complete = len(results) == len(jobs)
    summary = dict(
        complete=complete, candidates=len(jobs), validated=len(results), valid=len(results)-len(invalid),
        invalid=len(invalid), first_problem_kinds=dict(kinds.most_common()),
        rows=sum(item['rows'] for item in results), chunks=sum(item['chunks'] for item in results),
        valid_after_cleaning=sum(1 for item in results if item['valid'] and item['cleaning_scanned']
                                 and not item['cleaning_dropped'] and not item['protected']),
        by_engine={engine: dict(valid=sum(1 for i in results if i['engine'] == engine and i['valid']),
                                invalid=sum(1 for i in results if i['engine'] == engine and not i['valid']))
                   for engine in sorted({i['engine'] for i in results})},
        catalogue=report, seconds=round(time.time()-started), finished=time.time())
    save(args.output/'summary.json', summary)
    print(json.dumps({name: summary[name] for name in ('complete', 'candidates', 'validated', 'valid', 'invalid',
                                                       'first_problem_kinds', 'rows', 'valid_after_cleaning',
                                                       'by_engine')}, indent=2))
    for name in ('corpus_errors',):
        for error in report[name]:
            print('CORPUS ERROR:', error)
    for name in ('duplicate_match_keys', 'duplicate_replays'):
        if report[name]:
            print(f'note: {len(report[name])} {name.replace("_", " ")} (selection keeps one per match)')
    for name in ('leftovers', 'unregistered_shards'):
        if report[name]['count']:
            print(f"note: {report[name]['count']} {name.replace('_', ' ')} in training-data, e.g. {report[name]['sample'][:3]}")
    print('Details:', args.output)
    return 0 if complete and not invalid and not report['corpus_errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
