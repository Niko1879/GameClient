"""Durable conversion queue and one-replay export -> verified shard transaction."""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import redirect_stdout
from itertools import groupby
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'.cache/replay-converter'
CATALOG = ROOT/'replays/catalog/replay-catalog.sqlite3'
POLICY = ROOT/'replay-runtime/policy.json'
ENGINE = ROOT/'replay-runtime/game/engine-ab1c9849e5c41595.exe'
ENGINE_SHA = 'ab1c9849e5c41595ca9f2dffc3c2de60de2d6e285aed07c0181c4659f963a0f5'
ENGINE_HAS_REPLAY_SERVER = True
DATA = ROOT/'training-data'
RAW = ROOT/'replays/vanilla-china'
RULES = ROOT/'replay-runtime/revision-data'
ACTIVE = ('running', 'exported', 'preparing')
# Releases whose replays need a historical engine build (a revision adapter) this
# pipeline lacks. They stay collected but are excluded from conversion, so they
# are not recorded as replay failures. Remove an entry to queue them again.
PARKED_RELEASES = {}
# Releases recorded before a simulation fix play back with that release's logic
# (engine -replayLogic, GeneralsMD/.../Common/ReplayLogic.h). Part of the pipeline identity.
REPLAY_LOGIC = {
    '3DF43954': '081326',  # 081326_QFE3, commit c929fa068; 6/6 test replays matched every CRC
}


def replay_logic(exe_crc):
    return REPLAY_LOGIC.get(str(exe_crc).upper(), 'current')


def release_exclusion(exe_crc):
    from harvest_china_replays import replay_release_exclusion
    return replay_release_exclusion(exe_crc) or PARKED_RELEASES.get(str(exe_crc).upper())


def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.part')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def connect(path=CATALOG):
    from replay_catalog import connect as catalog_connect
    db=catalog_connect(path)
    db.row_factory=__import__('sqlite3').Row
    db.executescript('''
        CREATE TABLE IF NOT EXISTS conversion_job (
            pipeline TEXT NOT NULL, replay_sha256 TEXT NOT NULL,
            match_key TEXT NOT NULL, source_path TEXT NOT NULL, split TEXT NOT NULL,
            priority INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0, stage TEXT NOT NULL DEFAULT 'queued',
            owner TEXT, retry_after REAL NOT NULL DEFAULT 0,
            message TEXT, updated REAL NOT NULL,
            PRIMARY KEY(pipeline,replay_sha256));
        CREATE INDEX IF NOT EXISTS conversion_pending
            ON conversion_job(pipeline,status,priority,retry_after);
        CREATE INDEX IF NOT EXISTS conversion_claim
            ON conversion_job(pipeline,status,(CASE WHEN stage!='queued' THEN 0 ELSE 1 END),priority);
    ''')
    return db


def source_closure(entry='replay_converter.py'):
    """Every local module the converter can import, including function-local imports."""
    directory, seen, pending = Path(__file__).parent, set(), [entry]
    while pending:
        name = pending.pop()
        if name in seen or not (directory/name).is_file():
            continue
        seen.add(name)
        for node in ast.walk(ast.parse((directory/name).read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name.split('.')[0]+'.py' for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                pending.append(node.module.split('.')[0]+'.py')
    return sorted(seen)


def runtime_deployment(catalog):
    """The live engine identity a checkpoint trained on these shards must play against."""
    return {'engine_sha256': catalog['engine_exe_sha256'], 'engine_ini_crc': catalog['engine_ini_crc'],
            'ticks_per_second': catalog['ticks_per_second']}


def contract():
    from event_store import feature_identity
    from generals_policy import NeuralPolicy
    from replay_training import file_sha256
    if file_sha256(ENGINE)!=ENGINE_SHA:
        raise ValueError('Direct-capture executable changed')
    config=json.loads(POLICY.read_text(encoding='utf-8'))
    rules={p.relative_to(ROOT).as_posix():file_sha256(p) for p in RULES.glob('*/*.big')}
    if not rules:raise ValueError('No installed replay rule packages')
    # Resource controls are intentionally absent: resizing does not change data.
    identity=dict(engine_sha256=ENGINE_SHA,feature_key=feature_identity(NeuralPolicy(config,parameters=False)),
                  policy_sha256=file_sha256(POLICY),rules=rules,storage='binary',interval_ms=100,
                  quiet_capture='scheduled',replay_logic=REPLAY_LOGIC)
    identity['pipeline']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    # Editing learner, evaluation or tooling sources never interrupts conversion.
    identity['source_pins']={name:file_sha256(Path(__file__).with_name(name)) for name in source_closure()}
    return identity


def source_guard(identity):
    from replay_training import file_sha256
    for name,sha in identity['source_pins'].items():
        if file_sha256(Path(__file__).with_name(name))!=sha:
            raise ValueError('Converter source changed; stop and restart: '+name)
    if file_sha256(POLICY)!=identity['policy_sha256'] or file_sha256(ENGINE)!=identity['engine_sha256']:
        raise ValueError('Converter policy/native inputs changed; stop and restart')
    for name,sha in identity['rules'].items():
        if file_sha256(ROOT/name)!=sha:raise ValueError('Replay rules changed: '+name)


def verified_catalog(metadata,executable,expected_sha):
    from replay_training import file_sha256
    actual=file_sha256(executable)
    if actual!=expected_sha:raise ValueError('Native executable changed during handshake')
    if metadata.get('event')!='ready' or metadata.get('protocol_version')!=1 or metadata.get('ticks_per_second')!=60:
        raise ValueError('Unexpected native handshake protocol/rate')
    # The native protocol exposes CRCs, not its own SHA-256. Bind the catalogue
    # to the executable we actually launched and independently hashed.
    return dict(metadata,engine_exe_sha256=actual)


def recover(db,pipeline):
    placeholders=','.join('?' for _ in ACTIVE)
    changed=db.execute(f'''UPDATE conversion_job SET status='pending',owner=NULL,
        message='Recovered interrupted conversion',updated=?
        WHERE pipeline=? AND status IN ({placeholders})''',(time.time(),pipeline,*ACTIVE)).rowcount
    db.commit();return changed


def refresh_collection(db,archive=RAW/'harvest.sqlite3'):
    """Import only changed crawler rows; the crawler database stays read-only.

    Compare all small (id, updated) pairs rather than assuming timestamps or
    match IDs arrive monotonically. Old failed downloads can be repaired later.
    """
    db.execute('''CREATE TABLE IF NOT EXISTS replay (
        match_key TEXT PRIMARY KEY,match_id INTEGER NOT NULL,raw_status TEXT NOT NULL,
        raw_path TEXT,raw_sha256 TEXT,raw_error TEXT,raw_updated TEXT,
        listing_json TEXT NOT NULL,synced_utc TEXT NOT NULL)''')
    archive=Path(archive)
    if not archive.exists():raise ValueError('Crawler collection database is missing: '+str(archive))
    known=dict(db.execute('SELECT match_id,raw_updated FROM replay'))
    source=sqlite3.connect(f'file:{archive.resolve().as_posix()}?mode=ro',uri=True)
    try:
        changed=[key for key,stamp in source.execute('SELECT id,updated FROM matches')
                 if key not in known or known[key]!=(str(stamp) if stamp is not None else None)]
        for offset in range(0,len(changed),500):
            ids=changed[offset:offset+500]
            rows=source.execute('SELECT id,candidate,status,target,sha256,error,updated FROM matches WHERE id IN ('+','.join('?' for _ in ids)+')',ids)
            db.executemany('''INSERT INTO replay VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(match_key) DO UPDATE SET match_id=excluded.match_id,
                raw_status=excluded.raw_status,raw_path=excluded.raw_path,raw_sha256=excluded.raw_sha256,
                raw_error=excluded.raw_error,raw_updated=excluded.raw_updated,
                listing_json=excluded.listing_json,synced_utc=excluded.synced_utc''',
                [(f'pgo:{key}',key,status,path,sha,error,str(stamp) if stamp is not None else None,
                  candidate,str(time.time())) for key,candidate,status,path,sha,error,stamp in rows])
            db.commit()
        return len(changed)
    finally:source.close()


def resolve_recording(text,archive_root):
    relative=Path(text or '')
    source=(relative if relative.is_absolute() else RAW/relative).resolve()
    if not source.is_relative_to(archive_root):raise ValueError('Raw replay path escapes the replay archive')
    return source


def inventory(db,identity,reserve=None):
    """Read the crawler catalogue, never its website or training manifests."""
    from replay_bootstrap import partition
    pipeline=identity['pipeline']
    roles={}
    for key,split in db.execute('SELECT match_key,split FROM replay_assignment'):
        if split=='test' or (split=='validation' and roles.get(key)!='test'):
            roles[key]=split
    if reserve is None:
        reserve=json.loads((ROOT/'replays/test-reserve.json').read_text())
    for row in reserve.get('matches',[]):roles[row['match_key']]='test'
    known={r[0] for r in db.execute("SELECT DISTINCT replay_sha256 FROM export_result WHERE status='exported'")}
    existing=dict(db.execute('SELECT replay_sha256,split FROM conversion_job WHERE pipeline=?',(pipeline,)))
    archive_root=(ROOT/'replays').resolve()
    from replay_catalog import open_failures,refresh_availability
    # Disk is the truth for exports: a deleted or moved shard no longer counts as converted.
    refresh_availability(db)
    # Training selects shards by export engine too, so only this engine's count.
    converted=dict(db.execute('''SELECT replay_sha256,storage_path FROM prepared_shard
        WHERE available=1 AND feature_key=? AND engine_sha256=? AND replay_sha256 IS NOT NULL''',
        (identity['feature_key'],identity['engine_sha256'])))
    failed=open_failures(db)
    def known_failure(sha):
        return 'Known %s failure: %s (see replay_failure; retry to re-attempt)'%failed[sha]
    sql="SELECT match_key,raw_path,raw_sha256 FROM replay WHERE raw_status='downloaded' AND raw_sha256 IS NOT NULL"
    params=[]
    if identity.get('only_match_keys'):
        sql+=' AND match_key IN ('+','.join('?' for _ in identity['only_match_keys'])+')'
        params=identity['only_match_keys']
    rows=db.execute(sql+' ORDER BY raw_sha256,match_key',params).fetchall()
    added=0
    for sha,group in groupby(rows,key=lambda r:r['raw_sha256']):
        aliases=list(group)
        splits={roles.get(r['match_key'],partition(r['match_key'])) for r in aliases}
        split='test' if 'test' in splits else 'validation' if 'validation' in splits else 'train'
        row=aliases[0]
        if sha in existing:
            if split=='test' and existing[sha]!='test':
                raise ValueError('New test-reserve alias conflicts with an existing conversion: '+row['match_key'])
            continue
        try:
            source=resolve_recording(row['raw_path'],archive_root)
            path_error=None
        except ValueError as error:
            source=RAW/'invalid-source';path_error=str(error)
        if path_error:status,message='blocked',path_error
        elif split=='test':status,message='heldout','Protected test recording'
        elif release_exclusion(source.parent.name):
            status,message='excluded',release_exclusion(source.parent.name)
        elif sha in converted:status,message='done','Existing shard: '+converted[sha]
        elif sha in failed:status,message='rejected',known_failure(sha)
        else:status,message='pending',None
        priority=0 if sha in known else 1 if source.parent.name.upper()=='216B3EEA' else 2
        added+=db.execute('''INSERT OR IGNORE INTO conversion_job
            (pipeline,replay_sha256,match_key,source_path,split,priority,status,stage,message,updated)
            VALUES(?,?,?,?,?,?,?,?,?,?)''',(pipeline,sha,row['match_key'],str(source),split,priority,status,
            'prepared' if status=='done' else 'queued',message,time.time())).rowcount
        existing[sha]=split
    # Queued work follows the catalogue too: a deleted shard is work to do again,
    # and a replay converted or failed since it was queued is not exported again.
    reopened=0
    for sha,status,source,message in db.execute("""SELECT replay_sha256,status,source_path,message FROM conversion_job
            WHERE pipeline=? AND status IN ('done','pending','excluded')""",(pipeline,)).fetchall():
        exclusion=release_exclusion(Path(source).parent.name)
        if status=='excluded':
            if not (message or '').startswith('Parked ') or exclusion:continue
            # An unparked release returns to the queue.
            change=dict(status='pending',stage='queued',message=None)
        elif status=='pending' and exclusion:
            change=dict(status='excluded',stage='queued',message=exclusion)
        elif status=='done' and sha not in converted:
            change=dict(status='pending',stage='queued',message='Prepared shard missing');reopened+=1
        elif status=='pending' and sha in converted:
            change=dict(status='done',stage='prepared',message='Existing shard: '+converted[sha])
        elif status=='pending' and sha in failed:
            change=dict(status='rejected',stage='rejected',message=known_failure(sha))
        else:continue
        db.execute('UPDATE conversion_job SET '+','.join(k+'=?' for k in change)+',updated=? WHERE pipeline=? AND replay_sha256=?',
                   (*change.values(),time.time(),pipeline,sha))
    db.commit();return dict(added=added,reopened=reopened)


def claim(db,pipeline,owner,match_keys=None,prefer_release=None):
    db.execute('BEGIN IMMEDIATE')
    try:
        sql="SELECT * FROM conversion_job WHERE pipeline=? AND status='pending' AND retry_after<=?"
        params=[pipeline,time.time()]
        if match_keys:
            sql+=' AND match_key IN ('+','.join('?' for _ in match_keys)+')';params+=match_keys
        # conversion_claim serves this order without sorting the whole queue, which
        # would hold the write lock for ~140 ms per claim on a 60,000-job queue.
        order=" ORDER BY CASE WHEN stage!='queued' THEN 0 ELSE 1 END,priority,rowid LIMIT 1"
        row=db.execute(sql+order,params).fetchone()
        # A reused engine serves one INI package: stay on its release while it has work,
        # among the jobs of the first resumption state and priority.
        if row and prefer_release:
            same=db.execute(sql+" AND CASE WHEN stage!='queued' THEN 0 ELSE 1 END=? AND priority=? AND instr(source_path,?)>0"+order,
                [*params,0 if row['stage']!='queued' else 1,row['priority'],os.sep+prefer_release+os.sep]).fetchone()
            row=same or row
        if row:
            db.execute("UPDATE conversion_job SET status='running',owner=?,attempts=attempts+1,updated=? WHERE pipeline=? AND replay_sha256=?",(owner,time.time(),pipeline,row['replay_sha256']))
        db.commit()
        return dict(row) if row else None
    except BaseException:db.rollback();raise


def update(db,pipeline,sha,**changes):
    changes['updated']=time.time()
    db.execute('UPDATE conversion_job SET '+','.join(k+'=?' for k in changes)+' WHERE pipeline=? AND replay_sha256=?',(*changes.values(),pipeline,sha))
    db.commit()


def counts(db,pipeline,match_keys=None):
    sql='SELECT status,count(*) FROM conversion_job WHERE pipeline=?';params=[pipeline]
    if match_keys:
        sql+=' AND match_key IN ('+','.join('?' for _ in match_keys)+')';params+=match_keys
    return dict(db.execute(sql+' GROUP BY status',params))


def due(db,pipeline,match_keys=None):
    sql="SELECT 1 FROM conversion_job WHERE pipeline=? AND status='pending' AND retry_after<=?"
    params=[pipeline,time.time()]
    if match_keys:
        sql+=' AND match_key IN ('+','.join('?' for _ in match_keys)+')';params+=match_keys
    return db.execute(sql+' LIMIT 1',params).fetchone() is not None


def job_directory(job,identity):
    sha=job['replay_sha256']
    if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('Invalid replay SHA-256')
    return WORK/'jobs'/(identity['pipeline'][:16]+'-'+sha)


def remove_owned(path,root=WORK):
    path=Path(path).resolve();root=Path(root).resolve()
    if path==root or not path.is_relative_to(root):raise ValueError('Unsafe converter cleanup path')
    for attempt in range(150):
        try:
            if path.exists():shutil.rmtree(path)
            return
        except OSError as error:
            if getattr(error,'winerror',None) not in (32,33) or attempt==149:raise
            time.sleep(.2)


def reusable(db,job,identity):
    from replay_catalog import reusable_prepared,document_bytes
    from event_ingest import verify_prepared
    matches=reusable_prepared(db,job['match_key'],job['replay_sha256'],identity['feature_key'],identity['engine_sha256'])
    for match in matches:
        row=db.execute('SELECT * FROM prepared_shard WHERE archive_sha256=?',(match['archive_sha256'],)).fetchone()
        profile=db.execute('SELECT document_sha256 FROM prepared_profile WHERE profile_id=?',(row['profile_id'],)).fetchone()
        doc=json.loads(document_bytes(db,profile[0]))
        entry=json.loads(document_bytes(db,row['entry_document_sha256']))
        path=Path(match['path']).resolve()
        if path.parent!=DATA.resolve():raise ValueError('Existing shard is outside the flat training-data directory')
        entry.update(path=path.name,split=job['split'])
        manifest=DATA/('.conversion-'+job['replay_sha256']+'.json')
        save(manifest,dict(schema=1,kind='generals_event_dataset',complete=True,**doc,entries=[entry]))
        verify_prepared(manifest)
        return manifest
    return None


def finish_files(job,manifest,identity):
    """Shard has passed full verification and catalogue registration at this point."""
    doc=json.loads(Path(manifest).read_text())
    for entry in doc['entries']:
        sidecar=DATA/Path(entry['path']).with_suffix('.json')
        if sidecar.parent.resolve()!=DATA.resolve():raise ValueError('Invalid shard sidecar')
        sidecar.unlink(missing_ok=True)
    Path(manifest).unlink(missing_ok=True)
    remove_owned(job_directory(job,identity))


def log_tail(path,lines=60):
    try:return Path(path).read_text(encoding='utf-8',errors='replace').splitlines()[-lines:]
    except OSError:return None


def convert(job,identity,settings,runtime,profile,catalog,progress,session=None):
    from harvest_china_replays import parse_replay
    from replay_training import export,file_sha256,ExportPlaybackError
    from replay_bootstrap import data_pack,verify_entry,KIND
    from replay_storage import manifest_path
    from event_ingest import prepare_batch,verify_prepared,retire_exports
    from replay_catalog import record_export,register_prepared,import_retirement
    from engine_runtime import isolation_profile
    source_guard(identity)
    db=connect()
    try:
        # Reuse is physical, source- and feature-bound, and fully verified.
        existing=reusable(db,job,identity)
        if existing:
            finish_files(job,existing,identity)
            return dict(status='done',stage='prepared',reused=True)
        source=Path(job['source_path'])
        if file_sha256(source)!=job['replay_sha256']:raise ValueError('Original replay hash changed')
        header=parse_replay(source.read_bytes())
        exclusion=release_exclusion(header['exe_crc'])
        if exclusion:return dict(status='excluded',stage='excluded',message=exclusion)
        if header['desync'] or any(header['player_disconnections']):
            return dict(status='rejected',stage='rejected',message='Replay header flags desync/disconnection',
                failure=dict(stage='header',reason='desync_or_disconnect',
                    evidence=dict(desync=header['desync'],player_disconnections=header['player_disconnections'])))
        sidecar=source.with_suffix('.json')
        metadata=json.loads(sidecar.read_text(encoding='utf-8-sig')) if sidecar.exists() else {}
        row=dict(replay=str(source),header=header,metadata=metadata)
        pack=data_pack(row,RULES,False)
        isolation_profile(ROOT/'replay-runtime/profile',profile,maps=[header['map']])
        owned=job_directory(job,identity);owned.mkdir(parents=True,exist_ok=True)
        trajectory=owned/'export'/header['exe_crc']/(source.name+'.training.bin')
        proof_path=manifest_path(trajectory)
        # A crash after a complete export resumes at preparation, not simulation.
        if trajectory.exists() and proof_path.exists():
            proof=json.loads(proof_path.read_text())
            if proof['engine_sha256']!=identity['engine_sha256'] or proof['replay_sha256']!=job['replay_sha256']:
                raise ValueError('Recovered export has a different native/source identity')
        else:
            if (owned/'export').exists():remove_owned(owned/'export')
            progress('exporting')
            try:
                proof=export(SimpleNamespace(exe=runtime/ENGINE.name,replay=source,output=owned/'export',
                    user_data=profile,data_pack=pack,interval_ms=100,timeout=settings['timeout_seconds'],
                    ticks_per_second=60,storage='binary',placement_mode='fast',audit_level=settings['audit_level'],
                    quiet_capture='scheduled',capture_policy=POLICY,replay_logic=replay_logic(header['exe_crc']),
                    engine_session=session))
            except ExportPlaybackError as error:
                # The job directory (and its engine log) is deleted below; keep the log tail.
                evidence=dict(error.evidence,log_tail=log_tail(error.evidence['log']))
                record_export(db,dict(replay=str(source),backend='direct-capture',**evidence,error=str(error)),
                    match_key=job['match_key'],replay_sha256=job['replay_sha256'],engine_sha256=identity['engine_sha256'])
                db.commit()
                remove_owned(owned)
                return dict(status='rejected',stage='rejected',message=str(error),reason=error.evidence.get('status'),
                    failure=dict(stage='playback',reason=error.evidence.get('status') or 'unknown',evidence=evidence))
        entry=dict(trajectory=proof['trajectory'],sha256=proof['trajectory_sha256'],replay_sha256=job['replay_sha256'],
            match_key=job['match_key'],replay_exe_crc=header['exe_crc'],replay_ini_crc=header['ini_crc'],
            engine_sha256=identity['engine_sha256'],backend='direct-capture',ticks_per_second=60,split=job['split'])
        verify_entry(entry)
        record_export(db,dict(status='exported',replay=str(source),entry=entry,validation=proof['native_validation']),
            match_key=job['match_key'],replay_sha256=job['replay_sha256'],engine_sha256=identity['engine_sha256'])
        db.commit();progress('exported')
        native=owned/'dataset.json'
        save(native,dict(schema_version=1,kind=KIND,bootstrap_mode='cross_version_transfer',entries=[entry],deployment=runtime_deployment(json.loads(catalog.read_text())),
            action_catalog=dict(path=str(catalog),sha256=file_sha256(catalog))))
        progress('preparing')
        manifest=prepare_batch(native,DATA,POLICY,action_catalog=catalog,minimum_ram=settings['reserve_ram_mib']*1024**2,
            workers=1,manifest_name='.conversion-'+job['replay_sha256']+'.json',flat=True)
        data=verify_prepared(manifest)
        doc=json.loads(manifest.read_text())
        source_guard(identity)
        for item in data.entries:
            register_prepared(db,item,doc,data.shard_path(item),verified=True)
        db.commit()
        # Journaled deletion follows the verified, durably registered shard.
        retire_exports(native,manifest,owned)
        import_retirement(db,owned/'retired-exports.json');db.commit()
        finish_files(job,manifest,identity)
        return dict(status='done',stage='prepared',reused=False)
    finally:db.close()


def worker_main(slot,inbox,outbox,identity):
    # The parent assigns this worker to a kill-on-close process family before
    # permitting any native process to be created.
    if inbox.get()!='boot':return
    def emit(**event):outbox.put(dict(slot=slot,pid=os.getpid(),**event))
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    import gc
    from engine_runtime import clone_file,disable_power_throttling,isolation_profile,preflight
    from event_resources import snapshot
    from replay_training import file_sha256
    from scheduled_capture import EngineSession
    disable_power_throttling()
    session=EngineSession()
    directory=WORK/'workers'/str(slot);directory.mkdir(parents=True,exist_ok=True)
    runtime,profile=directory/'runtime',directory/'profile'
    try:
        source_guard(identity)
        if not runtime.exists():
            def ignore(parent,names):
                return [n for n in names if (Path(n).suffix.lower()=='.exe' and n!=ENGINE.name)
                        or Path(n).suffix.lower() in ('.log','.dmp','.tmp')]
            shutil.copytree(ENGINE.parent,runtime,copy_function=clone_file,ignore=ignore)
        if file_sha256(runtime/ENGINE.name)!=identity['engine_sha256']:raise ValueError('Staged engine differs')
        isolation_profile(ROOT/'replay-runtime/profile',profile,maps=[])
        os.environ['GENERALS_AI_USER_DATA']=str(profile)
        preflight(runtime/ENGINE.name,profile,directory/'preflight.log',120)
        ready=[json.loads(s[4:]) for s in (directory/'preflight.log').read_text().splitlines() if s.startswith('GAI ')]
        metadata=verified_catalog(next(r for r in ready if r.get('event')=='ready'),runtime/ENGINE.name,identity['engine_sha256'])
        catalog=directory/'action-catalog.json';save(catalog,metadata)
        emit(event='ready')
        while True:
            request=inbox.get()
            if request is None:break
            if request=='release':
                session.close();continue
            job,settings=request
            session.max_games=settings['engine_games']
            if not settings['reuse_engine']:session.close()
            def progress(stage):emit(event='stage',sha=job['replay_sha256'],stage=stage)
            with (directory/'worker.log').open('w',encoding='utf-8') as log,redirect_stdout(log):
                try:
                    result=convert(job,identity,settings,runtime,profile,catalog,progress,
                        session if settings['reuse_engine'] else None)
                except (OSError,ValueError,RuntimeError) as error:
                    text=str(error)
                    resource=any(s in text.lower() for s in ('ram/disk','resource boundary','resource reserve','insufficient memory','not enough memory'))
                    result=dict(status='pending' if resource else 'blocked',stage='interrupted' if resource else 'blocked',message=text[:4000])
                    if not resource:
                        from replay_catalog import failure_reason
                        result['failure']=dict(stage='conversion',reason=failure_reason(text),
                            evidence=dict(error=text[:4000],traceback=traceback.format_exc().splitlines()[-40:]))
                        # Valid raw data for this policy cannot be prepared; its
                        # original recording and failure remain in the catalogue.
                        remove_owned(job_directory(job,identity))
                        (DATA/('.conversion-'+job['replay_sha256']+'.json')).unlink(missing_ok=True)
            # A finished job leaves reference cycles worth up to ~80 MiB; collecting takes ~10 ms.
            gc.collect()
            emit(event='result',sha=job['replay_sha256'],result=result,engine=session.live(),private=snapshot(directory).get('private_bytes',0))
    except BaseException as error:
        emit(event='fatal',message=repr(error))
        raise
    finally:session.close()
