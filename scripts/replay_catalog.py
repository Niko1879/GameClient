"""Durable replay outcomes and reusable prepared shards, independent of datasets."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import zlib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT/'replays/catalog/replay-catalog.sqlite3'
DATA = ROOT/'training-data'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def connect(path=DEFAULT_CATALOG):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute('PRAGMA busy_timeout=60000')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS catalog_document (
            sha256 TEXT PRIMARY KEY, uncompressed_bytes INTEGER NOT NULL,
            payload BLOB NOT NULL, imported_utc TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS catalog_evidence (
            document_sha256 TEXT NOT NULL REFERENCES catalog_document(sha256),
            source_path TEXT NOT NULL, kind TEXT NOT NULL,
            PRIMARY KEY(document_sha256,source_path,kind));
        CREATE TABLE IF NOT EXISTS export_result (
            result_id TEXT PRIMARY KEY, match_key TEXT NOT NULL, replay_sha256 TEXT,
            status TEXT NOT NULL, backend TEXT, engine_sha256 TEXT,
            export_sha256 TEXT, export_path TEXT, crc_comparisons INTEGER,
            crc_mismatch INTEGER, completed INTEGER,
            document_sha256 TEXT NOT NULL REFERENCES catalog_document(sha256));
        CREATE INDEX IF NOT EXISTS export_result_replay ON export_result(match_key,status);
        CREATE TABLE IF NOT EXISTS replay_failure (
            replay_sha256 TEXT NOT NULL, match_key TEXT NOT NULL,
            stage TEXT NOT NULL, reason TEXT NOT NULL, engine_sha256 TEXT NOT NULL,
            pipeline TEXT NOT NULL, message TEXT NOT NULL,
            evidence_sha256 TEXT REFERENCES catalog_document(sha256),
            first_utc TEXT NOT NULL, last_utc TEXT NOT NULL, occurrences INTEGER NOT NULL,
            resolved_utc TEXT,
            PRIMARY KEY(replay_sha256,stage,reason,engine_sha256));
        CREATE INDEX IF NOT EXISTS replay_failure_reason ON replay_failure(stage,reason);
        -- One row per collected recording: its current shard on disk, or its open failure.
        -- Shard availability is refreshed from disk before every conversion inventory.
        CREATE VIEW IF NOT EXISTS replay_status AS
        WITH shard AS (SELECT replay_sha256,storage_path,feature_key,MAX(rowid)
                       FROM prepared_shard WHERE available=1 AND replay_sha256 IS NOT NULL GROUP BY replay_sha256),
             failure AS (SELECT replay_sha256,stage,reason,message,MAX(last_utc)
                         FROM replay_failure WHERE resolved_utc IS NULL GROUP BY replay_sha256)
        SELECT r.match_key,r.raw_sha256 AS replay_sha256,r.raw_path,
            CASE WHEN shard.storage_path IS NOT NULL THEN 'converted'
                 WHEN failure.reason IS NOT NULL THEN 'failed' ELSE 'unconverted' END AS state,
            shard.storage_path AS shard_path,shard.feature_key AS shard_feature_key,
            failure.stage AS failure_stage,failure.reason AS failure_reason,failure.message AS failure_message
        FROM replay r LEFT JOIN shard ON shard.replay_sha256=r.raw_sha256
            LEFT JOIN failure ON failure.replay_sha256=r.raw_sha256
        WHERE r.raw_sha256 IS NOT NULL;
        CREATE TABLE IF NOT EXISTS export_result_evidence (
            result_id TEXT NOT NULL REFERENCES export_result(result_id),
            document_sha256 TEXT NOT NULL REFERENCES catalog_document(sha256),
            pointer TEXT NOT NULL, PRIMARY KEY(result_id,document_sha256,pointer));
        CREATE TABLE IF NOT EXISTS prepared_profile (
            profile_id TEXT PRIMARY KEY, feature_key TEXT NOT NULL,
            document_sha256 TEXT NOT NULL REFERENCES catalog_document(sha256));
        CREATE TABLE IF NOT EXISTS prepared_shard (
            archive_sha256 TEXT PRIMARY KEY, match_key TEXT NOT NULL,
            replay_sha256 TEXT, export_sha256 TEXT, engine_sha256 TEXT,
            feature_key TEXT NOT NULL, index_sha256 TEXT NOT NULL,
            storage_path TEXT NOT NULL, bytes INTEGER NOT NULL, available INTEGER NOT NULL,
            profile_id TEXT NOT NULL REFERENCES prepared_profile(profile_id),
            entry_document_sha256 TEXT NOT NULL REFERENCES catalog_document(sha256));
        CREATE INDEX IF NOT EXISTS prepared_shard_replay ON prepared_shard(match_key,feature_key);
        CREATE INDEX IF NOT EXISTS prepared_shard_raw ON prepared_shard(replay_sha256,available);
        CREATE TABLE IF NOT EXISTS prepared_membership (
            manifest_sha256 TEXT NOT NULL REFERENCES catalog_document(sha256),
            ordinal INTEGER NOT NULL, archive_sha256 TEXT NOT NULL REFERENCES prepared_shard(archive_sha256),
            split TEXT NOT NULL, PRIMARY KEY(manifest_sha256,ordinal));
        CREATE TABLE IF NOT EXISTS replay_assignment (
            match_key TEXT NOT NULL, split TEXT NOT NULL,
            evidence_sha256 TEXT NOT NULL REFERENCES catalog_document(sha256),
            PRIMARY KEY(match_key,split,evidence_sha256));
        CREATE TABLE IF NOT EXISTS native_payload (
            sha256 TEXT NOT NULL, source_path TEXT NOT NULL, bytes INTEGER,
            retired INTEGER NOT NULL,
            evidence_sha256 TEXT NOT NULL REFERENCES catalog_document(sha256),
            PRIMARY KEY(sha256,source_path,evidence_sha256));
        CREATE VIEW IF NOT EXISTS replay_processing AS
        WITH results AS (
            SELECT match_key,MAX(status='exported') AS exported,
                MAX(status='crc_diverged') AS crc_diverged,
                COUNT(*) AS recorded_results
            FROM export_result GROUP BY match_key),
        prepared AS (
            SELECT match_key,COUNT(*) AS prepared_shards,
                COUNT(DISTINCT feature_key) AS feature_versions
            FROM prepared_shard WHERE available=1 GROUP BY match_key),
        ids AS (SELECT match_key FROM export_result UNION SELECT match_key FROM prepared_shard)
        SELECT ids.match_key,COALESCE(exported,0) AS exported,
            COALESCE(crc_diverged,0) AS crc_diverged,
            COALESCE(recorded_results,0) AS recorded_results,
            COALESCE(prepared_shards,0) AS prepared_shards,
            COALESCE(feature_versions,0) AS feature_versions
        FROM ids LEFT JOIN results USING(match_key) LEFT JOIN prepared USING(match_key);
    ''')
    return db


def store_document(db, raw, source_path, kind):
    sha = digest(raw)
    if db.execute('SELECT 1 FROM catalog_document WHERE sha256=?',(sha,)).fetchone() is None:
        db.execute('INSERT OR IGNORE INTO catalog_document VALUES(?,?,?,?)',
                   (sha, len(raw), zlib.compress(raw, 1), now()))
    db.execute('INSERT OR IGNORE INTO catalog_evidence VALUES(?,?,?)',
               (sha, str(source_path), kind))
    return sha


def document_bytes(db, sha):
    row = db.execute('SELECT uncompressed_bytes,payload FROM catalog_document WHERE sha256=?', (sha,)).fetchone()
    if row is None:
        raise ValueError('Catalogue document is missing: '+sha)
    raw = zlib.decompress(row[1])
    if len(raw) != row[0] or digest(raw) != sha:
        raise ValueError('Catalogue document failed its checksum')
    return raw


def replay_key(path, explicit=None):
    if explicit:
        return explicit
    match = re.fullmatch(r'match_(\d+)\.rep', Path(path).name, re.IGNORECASE)
    return 'pgo:'+match[1] if match else 'file:'+digest(str(path).encode())


def record_export(db, row, *, match_key=None, replay_sha256=None, engine_sha256=None,
                  source_path='', evidence_sha=None, pointer=''):
    entry = row.get('entry', {})
    key = replay_key(row.get('replay', ''), match_key or row.get('match_key') or entry.get('match_key'))
    raw_sha = replay_sha256 or entry.get('replay_sha256') or row.get('replay_sha256')
    engine_sha = engine_sha256 or entry.get('engine_sha256') or row.get('engine_sha256')
    wrapped = dict(match_key=key, replay_sha256=raw_sha, engine_sha256=engine_sha, result=row)
    result_sha = store_document(db, canonical(wrapped), source_path, 'export-result')
    validation = row.get('validation') or {}
    db.execute('INSERT OR IGNORE INTO export_result VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
        (result_sha,key,raw_sha,row.get('status','unknown'),row.get('backend'),engine_sha,
         entry.get('sha256') or row.get('export_sha256'),
         entry.get('trajectory') or row.get('export_path'),validation.get('crc_comparisons'),
         int(bool(validation['crc_mismatch'])) if 'crc_mismatch' in validation else None,
         int(bool(validation['completed'])) if 'completed' in validation else None,result_sha))
    db.execute('INSERT OR IGNORE INTO export_result_evidence VALUES(?,?,?)',
               (result_sha,evidence_sha or result_sha,pointer))
    return result_sha


# Failure stages: header (engine-independent replay flags), playback (native
# export on one engine) and conversion (audit, rules, vocabulary, preparation).
# Every pipeline skips a replay with an open failure until someone retries it.


def failure_reason(message):
    """Short, groupable code for a conversion-stage error; the full message is kept alongside."""
    text = message.lower()
    for needle, reason in (('frozen event vocabulary', 'vocabulary'), ('missing historical data package', 'missing_data_package'),
                           ('worker exited', 'worker_exited'), ('commands were lost', 'command_reconciliation'),
                           ('changed', 'source_changed'), ('audit', 'audit')):
        if needle in text:
            return reason
    return 'error'


def record_failure(db, *, replay_sha256, match_key, stage, reason, engine_sha256, pipeline, message, evidence=None):
    """Durable, investigable failure outcome for one replay; repeats count occurrences."""
    engine_sha256 = '' if stage == 'header' else engine_sha256
    evidence_sha = store_document(db, canonical(evidence), match_key, 'conversion-failure') if evidence else None
    stamp = now()
    db.execute('''INSERT INTO replay_failure VALUES(?,?,?,?,?,?,?,?,?,?,1,NULL)
        ON CONFLICT(replay_sha256,stage,reason,engine_sha256) DO UPDATE SET
        match_key=excluded.match_key,pipeline=excluded.pipeline,message=excluded.message,
        evidence_sha256=COALESCE(excluded.evidence_sha256,evidence_sha256),last_utc=excluded.last_utc,
        occurrences=occurrences+1,resolved_utc=NULL''',
        (replay_sha256, match_key, stage, reason, engine_sha256, pipeline, message, evidence_sha, stamp, stamp))


def resolve_failures(db, replay_sha256):
    """A successful conversion or an explicit retry closes a replay's open failures."""
    db.execute('UPDATE replay_failure SET resolved_utc=? WHERE replay_sha256=? AND resolved_utc IS NULL',
               (now(), replay_sha256))


def open_failures(db):
    """{replay_sha256: (stage, reason)} for every replay with an unresolved failure."""
    return {sha: (stage, reason) for sha, stage, reason in db.execute(
        'SELECT replay_sha256,stage,reason FROM replay_failure WHERE resolved_utc IS NULL ORDER BY last_utc')}


def profile_for(db, doc, key, source_path):
    profile = {name:doc[name] for name in ('config','deployment','capability_catalog_sha256') if name in doc}
    sha = store_document(db,canonical(profile),source_path,'prepared-profile')
    db.execute('INSERT OR IGNORE INTO prepared_profile VALUES(?,?,?)',(sha,key,sha))
    return sha


def register_prepared(db, entry, doc, path, *, source_path='', verified=False, profile_id=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('Prepared shard missing: '+str(path))
    if not verified and digest(path.read_bytes()) != entry['archive_sha256']:
        raise ValueError('Prepared shard checksum failed: '+str(path))
    index = entry['index']
    if digest(canonical(index)) != entry['index_sha256']:
        raise ValueError('Prepared index checksum failed')
    key = index['legality_key']
    profile = profile_id or profile_for(db,doc,key,source_path)
    stored = dict(entry,path=path.name)
    entry_sha = store_document(db,canonical(stored),source_path,'prepared-entry')
    source = index['source']
    storage = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    db.execute('''INSERT INTO prepared_shard VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(archive_sha256) DO UPDATE SET storage_path=excluded.storage_path,
        bytes=excluded.bytes,available=1''',
        (entry['archive_sha256'],entry['match_key'],source.get('replay_sha256'),source.get('sha256'),
         source.get('engine_sha256'),key,entry['index_sha256'],storage,path.stat().st_size,1,profile,entry_sha))
    # A prepared shard carries its original full native audit and reconciliation.
    audit = source.get('audit', {})
    if (audit.get('replay_stream_eof') and audit.get('recorded_crc_messages_compared',0)>0
            and source.get('command_reconciliation',{}).get('all_controlled_commands_captured')):
        row = dict(status='exported',replay=source.get('original_replay',''),
            entry=dict(match_key=entry['match_key'],replay_sha256=source.get('replay_sha256'),
                       engine_sha256=source.get('engine_sha256'),sha256=source.get('sha256'),
                       trajectory=source.get('path')),audit=audit,
            validation=dict(completed=True,crc_mismatch=False,crc_comparisons=audit['recorded_crc_messages_compared']))
        record_export(db,row,source_path=source_path,evidence_sha=entry_sha,pointer='/index/source')
    return entry_sha


def import_prepared_manifest(db, path, *, verified_hashes=None):
    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc.get('kind') != 'generals_event_dataset':
        raise ValueError('Not a prepared event manifest')
    sha = store_document(db,raw,path,'prepared-manifest')
    checked = verified_hashes if verified_hashes is not None else set()
    keys = {entry['index']['legality_key'] for entry in doc['entries']}
    if len(keys)>1:
        raise ValueError('Prepared manifest mixes incompatible feature versions')
    profile = profile_for(db,doc,next(iter(keys)),path) if keys else None
    for ordinal,entry in enumerate(doc['entries']):
        shard = (path.parent/entry['path']).resolve()
        register_prepared(db,entry,doc,shard,source_path=path,
                          verified=entry['archive_sha256'] in checked,profile_id=profile)
        checked.add(entry['archive_sha256'])
        db.execute('INSERT OR REPLACE INTO prepared_membership VALUES(?,?,?,?)',
                   (sha,ordinal,entry['archive_sha256'],entry['split']))
        db.execute('INSERT OR IGNORE INTO replay_assignment VALUES(?,?,?)',
                   (entry['match_key'],entry['split'],sha))
    return sha


def import_retirement(db, path):
    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw)
    sha = store_document(db,raw,path,'retirement-journal')
    for item in doc.get('files',[]):
        db.execute('INSERT OR IGNORE INTO native_payload VALUES(?,?,?,?,?)',
                   (item['sha256'],item['path'],item.get('bytes'),int(bool(doc.get('complete'))),sha))
    return sha


def import_test_reserve(db, root=ROOT):
    """Protected test recordings are held out of every selection."""
    reserve = Path(root)/'replays/test-reserve.json'
    if not reserve.is_file():
        return 0
    raw = reserve.read_bytes()
    sha = store_document(db,raw,reserve,'test-reserve')
    matches = json.loads(raw).get('matches',[])
    for row in matches:
        db.execute('INSERT OR IGNORE INTO replay_assignment VALUES(?,?,?)',(row['match_key'],'test',sha))
    db.commit()
    return len(matches)


def import_prepared_directory(db, directory=DATA):
    directory = Path(directory)
    checked = set()
    count = 0
    for path in sorted(directory.glob('dataset-*.json')):
        import_prepared_manifest(db,path,verified_hashes=checked)
        db.commit()
        count += 1
        print('Catalogued prepared manifest:',path.name,'verified shards:',len(checked),flush=True)
    for path in directory.glob('metadata-*.json'):
        store_document(db,path.read_bytes(),path,'preparation-metadata')
    db.commit()
    return dict(manifests=count,verified_shards=len(checked))


def reusable_prepared(db, match_key, replay_sha256=None, feature_key=None, engine_sha256=None):
    """Locate existing physical shards before exporting a replay again.

    Without a requested feature version, report all versions so the caller can
    choose explicitly rather than accidentally export or mix configurations.
    With an engine, only that engine's exports qualify: training selects shards
    by export engine, so another engine's shard cannot stand in for this one.
    """
    rows = db.execute('''SELECT archive_sha256,storage_path,bytes,feature_key,replay_sha256,engine_sha256
        FROM prepared_shard WHERE match_key=?''', (match_key,)).fetchall()
    result = []
    for sha, stored, size, version, raw_sha, engine in rows:
        if feature_key is not None and version != feature_key:
            continue
        if engine_sha256 is not None and engine != engine_sha256:
            continue
        if replay_sha256 is not None and raw_sha is not None and raw_sha != replay_sha256:
            continue
        path = Path(stored) if Path(stored).is_absolute() else ROOT/stored
        if path.is_file() and path.stat().st_size == size:
            result.append(dict(archive_sha256=sha,path=str(path),feature_key=version))
    return result


def refresh_availability(db):
    """Mark each catalogued shard available only while its file is still at its path with its size."""
    lost = 0
    for sha,path,size,was in db.execute('SELECT archive_sha256,storage_path,bytes,available FROM prepared_shard').fetchall():
        path = Path(path) if Path(path).is_absolute() else ROOT/path
        available = int(path.is_file() and path.stat().st_size==size)
        if available != was:
            db.execute('UPDATE prepared_shard SET available=? WHERE archive_sha256=?',(available,sha))
            lost += was and not available
    db.commit()
    return lost


def snapshot_manifest(path, catalog=DEFAULT_CATALOG):
    """Called after the normal native/preparation checks have already passed."""
    db = connect(catalog)
    try:
        doc = json.loads(Path(path).read_bytes())
        checked = {e['archive_sha256'] for e in doc['entries']}
        sha = import_prepared_manifest(db,path,verified_hashes=checked)
        db.commit()
        return sha
    finally:
        db.close()


def restore_document(sha, output, catalog=DEFAULT_CATALOG):
    db = connect(catalog)
    try:
        raw = document_bytes(db,sha)
    finally:
        db.close()
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_name(output.name+'.part')
    temporary.write_bytes(raw)
    temporary.replace(output)
    return output


def build_dataset(db, feature_key, *, limit=None, split='train', seed=123):
    """Materialise an arbitrary compatible subset; historical datasets are optional."""
    rows = db.execute('''SELECT match_key,archive_sha256,storage_path,profile_id,entry_document_sha256
        FROM prepared_shard WHERE feature_key=? AND available=1 ORDER BY archive_sha256''',(feature_key,)).fetchall()
    assignments = {}
    for key,role in db.execute('SELECT DISTINCT match_key,split FROM replay_assignment'):
        assignments.setdefault(key,set()).add(role)
    selected = {}
    for row in rows:
        roles = assignments.get(row[0],set())
        role = 'test' if 'test' in roles else 'validation' if 'validation' in roles else 'train'
        if split != 'all' and role != split:
            continue
        selected.setdefault(row[0],(row,role))
    ordered = sorted(selected.values(),key=lambda item:digest(f'{seed}|{item[0][0]}'.encode()))
    if limit is not None:
        ordered = ordered[:limit]
    if not ordered:
        raise ValueError('No compatible prepared replays match this selection')
    profile_id = ordered[0][0][3]
    profile_sha = db.execute('SELECT document_sha256 FROM prepared_profile WHERE profile_id=?',(profile_id,)).fetchone()[0]
    profile = json.loads(document_bytes(db,profile_sha))
    entries = []
    for row,role in ordered:
        path = Path(row[2]) if Path(row[2]).is_absolute() else ROOT/row[2]
        if not path.is_file():
            raise ValueError('Catalogue shard is missing: '+str(path))
        entry = json.loads(document_bytes(db,row[4]))
        entry.update(path=path.name,split=role)
        entries.append(entry)
    return dict(schema=1,kind='generals_event_dataset',complete=True,**profile,entries=entries,
                selection=dict(feature_key=feature_key,seed=seed,split=split,limit=limit))
