"""Frozen prepared-data experiment: fresh split, early stopping, then Hard-AI play."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'model-training'
DATA=ROOT/'training-data'
CATALOG=ROOT/'replays/catalog/replay-catalog.sqlite3'
MANIFEST=DATA/'.model-training.json'
ENGINE=ROOT/'replay-runtime/game/engine-ab1c9849e5c41595.exe'
STOP=RUN/'stop.json'

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.part')
    with temporary.open('w',encoding='utf-8') as stream:json.dump(value,stream,allow_nan=False)
    os.replace(temporary,path)


def load(path,default=None):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def status(phase,**fields):
    save(RUN/'status.json',dict(phase=phase,pid=os.getpid(),updated=time.time(),**fields))


def choose(candidates,training,validation,seed,protected_keys=(),protected_shas=(),validation_keys=None):
    """Experiment roles are chosen here, independently of conversion hints.

    Validation is chosen first, so runs of different training sizes share it;
    validation_keys reuses an earlier run's exact validation matches.
    training=None takes every remaining compatible replay.
    """
    selected=[];keys=set();shas=set();wanted=set(validation_keys or ())
    ordered=sorted(candidates,key=lambda r:(hashlib.sha256(f'{seed}|{r["replay_sha256"]}'.encode()).digest(),r['archive_sha256']))
    if wanted:ordered=[r for r in ordered if r['match_key'] in wanted]+[r for r in ordered if r['match_key'] not in wanted]
    for row in ordered:
        key,sha=row['match_key'],row['replay_sha256']
        if key in protected_keys or sha in protected_shas or key in keys or sha in shas:continue
        if wanted and len(selected)<validation and key not in wanted:
            raise ValueError('Reused validation matches are not all prepared and available')
        keys.add(key);shas.add(sha)
        selected.append(dict(row,split='validation' if len(selected)<validation else 'train'))
        if training is not None and len(selected)==training+validation:break
    if training is None and len(selected)<=validation:
        raise ValueError(f'Need more than {validation} unique compatible prepared replays; found {len(selected)}')
    if training is not None and len(selected)!=training+validation:
        raise ValueError(f'Need {training+validation} unique compatible prepared replays; found {len(selected)}')
    return selected


def selection(args):
    """(dataset document, feature key) for args.training/validation/seed; shards stay in training-data/."""
    from replay_catalog import connect,document_bytes
    from event_store import feature_identity
    from event_dataset import EventDataset
    from generals_policy import NeuralPolicy
    policy=load(ROOT/'replay-runtime/policy.json')
    from enemy_side import enemy_faction
    model=NeuralPolicy(policy,parameters=False)
    key=feature_identity(model)
    db=connect(CATALOG);db.row_factory=__import__('sqlite3').Row
    try:
        protected={r[0] for r in db.execute("SELECT match_key FROM replay_assignment WHERE split='test'")}
        protected.update(r['match_key'] for r in load(ROOT/'replays/test-reserve.json')['matches'])
        protected_shas={row[0] for row in db.execute('SELECT raw_sha256 FROM replay WHERE match_key IN ('+','.join('?' for _ in protected)+')',list(protected))} if protected else set()
        candidates=[]
        for row in db.execute('SELECT * FROM prepared_shard WHERE available=1 AND feature_key=? AND engine_sha256=?',(key,digest(ENGINE))):
            path=Path(row['storage_path']);path=path if path.is_absolute() else ROOT/path
            if path.is_file() and path.parent.resolve()==DATA.resolve() and path.stat().st_size==row['bytes']:
                candidates.append(dict(row))
        reused=None
        if args.validation_from:
            reused={e['match_key'] for e in load(args.validation_from)['entries'] if e['split']=='validation'}
            if len(reused)!=args.validation:raise ValueError('Reused validation set has a different size')
        selected=choose(candidates,args.training,args.validation,args.seed,protected,protected_shas,reused)
        entries=[];profile_cache={};deployment=None
        for ordinal,row in enumerate(selected,1):
            if STOP.exists():raise InterruptedError('Experiment stop requested')
            if row['profile_id'] not in profile_cache:
                sha=db.execute('SELECT document_sha256 FROM prepared_profile WHERE profile_id=?',(row['profile_id'],)).fetchone()[0]
                profile_cache[row['profile_id']]=json.loads(document_bytes(db,sha))
            profile=profile_cache[row['profile_id']]
            if deployment is None:deployment=profile['deployment']
            if profile['deployment']!=deployment:raise ValueError('Selected shards have different live deployment identities')
            entry=json.loads(document_bytes(db,row['entry_document_sha256']))
            entry.update(path=Path(row['storage_path']).name,split=row['split'])
            if entry['index']['source']['replay_sha256']!=row['replay_sha256']:
                raise ValueError('Catalogue entry differs from original replay identity')
            entry['enemy_identity']=list(model.encoder.identity.player_identity(enemy_faction(entry['index']['source']['original_replay'])))
            entries.append(entry)
        doc=dict(schema=1,kind='generals_event_dataset',complete=True,config=policy,deployment=deployment,
            capability_catalog_sha256=next(iter(profile_cache.values()))['capability_catalog_sha256'],entries=entries,
            selection=dict(seed=args.seed,training=args.training,validation=args.validation,
                split_policy='fresh_whole_match_and_raw_hash',feature_key=key))
        return doc,key
    finally:db.close()


def freeze(args):
    from event_dataset import EventDataset
    doc,key=selection(args);entries=doc['entries']
    save(MANIFEST,doc)
    dataset=EventDataset(MANIFEST)
    for ordinal,entry in enumerate(dataset.entries,1):
        path=dataset.shard_path(entry)
        if digest(path)!=entry['archive_sha256']:raise ValueError('Prepared shard checksum mismatch: '+str(path))
        # Registration already checked the whole archive and its source.
        # Matching SHA-256 proves these are the same qualified bytes;
        # normal training reads still validate individual array members.
        if ordinal%100==0:status('verifying_selection',verified=ordinal,total=len(entries))
    plan=dict(schema=1,kind='prepared_early_stopping_experiment',training=args.training,validation=args.validation,
        seed=args.seed,patience=args.patience,max_epochs=args.max_epochs,device=args.device,
        lr_decay=args.lr_decay,min_lr=args.min_lr,
        dataset=str(MANIFEST),dataset_sha256=digest(MANIFEST),feature_key=key,
        native=dict(exe=str(ENGINE),sha256=digest(ENGINE)),gameplay=[],gameplay_attempts=[],
        gameplay_work_root=str(RUN/'gameplay'),created=time.time())
    plan['source_pins']={p.name:digest(p) for p in (ROOT/'scripts').glob('*.py') if not p.name.startswith('test_')}
    save(RUN/'plan.json',plan)
    return plan


def guard(plan):
    if digest(MANIFEST)!=plan['dataset_sha256'] or digest(ENGINE)!=plan['native']['sha256']:
        raise RuntimeError('Frozen experiment data/native identity changed')
    for name,sha in plan['source_pins'].items():
        if digest(ROOT/'scripts'/name)!=sha:raise RuntimeError('Experiment source changed: '+name)


def wait_for_ram(plan,minimum):
    from event_resources import snapshot
    while True:
        if STOP.exists():raise InterruptedError('Experiment stop requested')
        guard(plan)
        resources=snapshot(RUN)
        if resources.get('available_ram_bytes',0)>=minimum and resources['free_disk_bytes']>=80*1024**3:return
        status('waiting_for_resources',required_ram_mib=minimum//1024**2,resources=resources)
        time.sleep(15)


def phase(plan,label,command,*,minimum=int(2.5*1024**3),training=False):
    from pipeline_resources import tree
    wait_for_ram(plan,minimum)
    path=RUN/(label+'.log');begin=time.monotonic();limit=False;family=None;minimum_available=None
    with path.open('a' if training else 'w',encoding='utf-8') as log:
        process,family=spawn_owned_python(command,log)
        try:
            while process.poll() is None:
                resources=tree()
                available=resources.get('available_ram_bytes',0)
                minimum_available=available if minimum_available is None else min(minimum_available,available)
                stopping=STOP.exists()
                boundary=resources.get('available_ram_bytes',0)<1024**3 or resources.get('aggregate_private_bytes',0)>8*1024**3 or resources['free_disk_bytes']<80*1024**3
                status(label,elapsed_seconds=time.monotonic()-begin,resources=resources,active_pid=process.pid,
                    training_report=load(RUN/'report.json'),last_epoch=last_epoch(path) if training else None)
                if not training and (stopping or boundary):
                    family.close();limit=boundary
                    process.wait(timeout=30);break
                # The trainer checkpoints and exits itself at an update boundary.
                if training and (boundary or stopping):
                    save(RUN/'checkpoint-stop.json',dict(reason='user' if stopping else 'resource'))
                    limit=boundary
                time.sleep(3)
            code=process.wait(timeout=30)
        finally:
            if family:family.close()
            if process.poll() is None:process.kill();process.wait()
    if STOP.exists():raise InterruptedError('Experiment stopped after checkpointing')
    return dict(returncode=code,resource_interruption=limit,seconds=time.monotonic()-begin,log=str(path),
                minimum_available_ram_bytes=minimum_available)


def python_runtime():
    # Match multiprocessing's Windows venv handling: avoid the redirector EXE
    # so the owned PID is the interpreter, not a launcher with an orphan child.
    env=os.environ.copy();executable=sys.executable
    if os.path.normcase(executable)!=os.path.normcase(sys._base_executable):
        env['__PYVENV_LAUNCHER__']=sys.executable;executable=sys._base_executable
    return executable,env


def spawn_owned_python(command,log):
    from engine_runtime import disable_power_throttling
    from replay_converter import ProcessFamily
    command=[str(p) for p in command]
    if command[1]!='-B':raise ValueError('Expected a script invocation with -B')
    executable,env=python_runtime()
    barrier="import sys,runpy,os; token=sys.stdin.buffer.read(1); assert token==b'G'; sys.argv=sys.argv[1:]; sys.path.insert(0,os.path.dirname(os.path.abspath(sys.argv[0]))); runpy.run_path(sys.argv[0],run_name='__main__')"
    process=subprocess.Popen([executable,'-u','-B','-c',barrier,*command[2:]],cwd=ROOT,env=env,
        stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    family=None
    try:
        disable_power_throttling(int(process._handle))
        family=ProcessFamily(process.pid)
        process.stdin.write(b'G');process.stdin.flush();process.stdin.close()
        return process,family
    except BaseException:
        if family:family.close()
        process.kill();process.wait();raise


def last_epoch(path):
    with Path(path).open('rb') as stream:
        stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-65536))
        lines=stream.read().decode('utf-8',errors='replace').splitlines()
    for line in reversed(lines):
        try:row=json.loads(line)
        except ValueError:continue
        if isinstance(row,dict) and isinstance(row.get('epoch'),int):return row
    return None


def run(args):
    from engine_runtime import disable_power_throttling
    from replay_converter import InstanceLock
    disable_power_throttling()
    RUN.mkdir(exist_ok=True)
    with InstanceLock(RUN/'service.lock'):
        try:
            status('selecting')
            plan=load(RUN/'plan.json')
            if plan:
                for name,value in (('training',args.training),('validation',args.validation),('seed',args.seed),('patience',args.patience),('max_epochs',args.max_epochs),('device',args.device),('lr_decay',args.lr_decay),('min_lr',args.min_lr)):
                    if plan.get(name,{'lr_decay':1.0,'min_lr':0.0}.get(name))!=value:raise ValueError('Existing frozen experiment settings differ: '+name)
            else:plan=freeze(args)
            guard(plan)
            # These writable paths are required by this installed Windows
            # ROCm/MIOpen build, matching the existing training coordinator.
            for variable,name in (('MIOPEN_USER_DB_PATH','user-db'),('MIOPEN_CUSTOM_CACHE_DIR','kernels')):
                cache=ROOT/'.cache/miopen'/name;cache.mkdir(parents=True,exist_ok=True)
                os.environ[variable]=str(cache)
            # The full search reruns for every new microbatch shape: 8 against 184 rows/s on the 6750 XT.
            os.environ['MIOPEN_FIND_MODE']='FAST'
            while not plan.get('checkpoint'):
                checkpoint_stop=RUN/'checkpoint-stop.json';checkpoint_stop.unlink(missing_ok=True)
                command=[sys.executable,'-B',ROOT/'scripts/train_events.py','train','--dataset',MANIFEST,'--output',RUN,
                    '--device',plan['device'],'--epochs',str(plan['max_epochs']),'--early-stopping-patience',str(plan['patience']),
                    '--seed',str(plan['seed']),'--threads','1','--batch-size','256','--buffer-mib','64',
                    '--pair-budget','8000000','--checkpoint-seconds','600','--min-free-ram-mib','1024',
                    '--max-private-mib','8192','--stop-file',checkpoint_stop,
                    '--lr-decay',str(plan.get('lr_decay',1.0)),'--min-lr',str(plan.get('min_lr',0.0))]
                for name in ('latest.npz','initial.npz'):
                    if (RUN/name).exists():command+=['--resume',RUN/name];break
                report_path=RUN/'report.json'
                previous_stamp=report_path.stat().st_mtime_ns if report_path.exists() else None
                result=phase(plan,'training',command,training=True)
                fresh=report_path.exists() and report_path.stat().st_mtime_ns!=previous_stamp
                report=load(report_path,{}) if fresh else {}
                if result['returncode']==0 and report.get('status') in ('completed','early_stopped'):
                    if report['training_matches']!=args.training or report['validation_matches']!=args.validation:
                        raise ValueError('Trainer used incorrect split counts')
                    best=Path(report['best_checkpoint'])
                    if not best.exists() or digest(best)!=report['best_checkpoint_sha256']:
                        raise ValueError('Best checkpoint identity check failed')
                    plan.update(checkpoint=str(best),checkpoint_sha256=digest(best),
                        best_epoch=report['early_stopping']['best_epoch'],best_validation_nll=report['early_stopping']['best_score'],
                        stop_reason=report['status'],training_epochs=report['cursor']['epoch'])
                    save(RUN/'plan.json',plan);break
                text=Path(result['log']).read_text(encoding='utf-8',errors='replace')[-12000:]
                if result['resource_interruption'] or report.get('status') in ('resource_limit','interrupted') or any(token in text for token in ('resource reserve','resource boundary','out of memory','OutOfMemoryError')):
                    status('waiting_to_resume_training',reason=text[-1200:]);time.sleep(15);continue
                raise RuntimeError('Training failed; inspect '+result['log'])
            import cycle_runtime as cycle
            cycle.HERE=RUN;cycle.CPU=cycle.GPU=Path(sys.executable)
            cycle.source_guard=lambda:guard(plan)
            cycle.status=lambda label,**fields:status(label,**fields)
            cycle.run_phase=lambda label,command,**options:phase(plan,label,command,minimum=int(2.25*1024**3))
            cycle.wait_for_resources=lambda *unused:wait_for_ram(plan,int(2.25*1024**3))
            # Reuse the gameplay profile's verified handshake; no second asset
            # environment or extra native process is needed for its catalogue.
            wait_for_ram(plan,int(2.25*1024**3))
            cycle.setup_play_runtime()
            ready=[json.loads(line[4:]) for line in (Path(plan['gameplay_work_root'])/'profile-preflight.log').read_text(encoding='utf-8').splitlines() if line.startswith('GAI ')]
            catalog=next(row for row in ready if row.get('event')=='ready')
            plan['action_catalog']=str(RUN/'action-catalog.json');save(Path(plan['action_catalog']),catalog);save(RUN/'plan.json',plan)
            while len(plan['gameplay'])<len(cycle.CASES):
                cycle.play_all(plan)
            save(RUN/'results.json',dict(status='completed',training_matches=args.training,validation_matches=args.validation,
                epochs=plan['training_epochs'],best_epoch=plan['best_epoch'],best_validation_nll=plan['best_validation_nll'],
                stop_reason=plan['stop_reason'],checkpoint=plan['checkpoint'],checkpoint_sha256=plan['checkpoint_sha256'],
                gameplay=[dict(case=g['case'],score=g['score']) for g in plan['gameplay']]))
            status('completed',results=str(RUN/'results.json'),best_epoch=plan['best_epoch'])
        except InterruptedError as error:status('stopped',reason=str(error))
        except BaseException as error:status('failed',error=repr(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--training',type=int,default=5000)
    parser.add_argument('--validation',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--max-epochs',type=int,default=0,help='Optional ceiling; 0 trains until validation plateau')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--lr-decay',type=float,default=1.0,help='Learning-rate factor after each epoch without validation improvement (1 disables)')
    parser.add_argument('--min-lr',type=float,default=0.0)
    parser.add_argument('--validation-from',help='Reuse the validation matches of an earlier dataset manifest')
    args=parser.parse_args()
    if min(args.training,args.validation,args.patience)<1 or args.max_epochs<0:parser.error('Positive splits/patience and nonnegative epoch ceiling required')
    if not 0<args.lr_decay<=1 or args.min_lr<0:parser.error('Learning-rate decay must be in (0, 1] and the floor nonnegative')
    if args.background:
        RUN.mkdir(exist_ok=True)
        executable,env=python_runtime()
        command=[executable,'-u','-B',str(Path(__file__).resolve()),*[s for s in sys.argv[1:] if s!='--background']]
        with (RUN/'runner.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        print(json.dumps(dict(started_pid=process.pid,status=str(RUN/'status.json'))))
    else:run(args)


if __name__=='__main__':main()
