"""Frozen prepared-data experiment: fresh split, early stopping, then Hard-AI play."""
from __future__ import annotations

import argparse
import collections
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
# The converter's pinned engine: selection takes its shards, and live play and validation run on it (one pin).
from replay_conversion import ENGINE
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


# Validation strata (user decision, 9 Nov 2026): equal thirds of short, medium and
# long games, and five rating bands by the match's mean pre-match Overall rating,
# each with its proportional share but at least VALIDATION_FLOOR games per length
# third, so the scarce high-rated games give a usable signal (~2% of them held out).
LENGTH_BANDS=((10,'short'),(20,'medium'),(None,'long'))           # game minutes, upper bounds
RATING_BANDS=((1100,'<1100'),(1400,'1100-1400'),(1700,'1400-1700'),(None,'1700+'))
VALIDATION_FLOOR=50


def band(value,bands):
    return next(name for bound,name in bands if bound is None or value<bound)


def validation_strata(metrics=None):
    """match key -> (length band, rating band) from the raw-replay cleaning scan (scripts/replay_cleaning.py).

    Length is measured from the replay's frames, never listing text. An unrated
    match (either player without an Overall rating) is its own rating band.
    """
    from replay_cleaning import METRICS
    strata={}
    with Path(metrics or METRICS).open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if 'error' in row:continue
            minutes=max(row['frames'],row['last_command_frame'])/row['rate']/60
            ratings=[(player.get('rating') or {}).get('overall') for player in row['players']]
            rating='unrated' if None in ratings else band(sum(ratings)/len(ratings),RATING_BANDS)
            strata[row['match_key']]=(band(minutes,LENGTH_BANDS),rating)
    return strata


def allocate(total,weights,capacity,floor=0):
    """Integer quotas summing to total: at least `floor` each (when they all fit), then by weight, within capacity."""
    cells=sorted(weights,key=str)
    quota={c:min(floor,capacity[c]) for c in cells}
    if sum(quota.values())>total:quota={c:0 for c in cells}
    while sum(quota.values())<total:
        open_=[c for c in cells if quota[c]<capacity[c]]
        if not open_:raise ValueError(f'Only {sum(capacity.values())} candidates for {total} stratified validation matches')
        free=total-sum(quota[c] for c in cells if c not in open_)
        weight=sum(weights[c] for c in open_) or len(open_)
        # One match to the open cell furthest below its weighted share; ties by name.
        cell=max(open_,key=lambda c:(free*(weights[c] or 1)/weight-quota[c],-cells.index(c)))
        quota[cell]+=1
    return quota


def stratified(rows,total,strata):
    """`total` rows by stratum quota, each stratum in the given (seeded) order."""
    cells={}
    for row in rows:
        cells.setdefault(strata[row['match_key']],[]).append(row)
    lengths={}
    for (length,rating),members in cells.items():
        lengths.setdefault(length,{})[rating]=len(members)
    per_length=allocate(total,{length:1 for length in lengths},{length:sum(sizes.values()) for length,sizes in lengths.items()})
    chosen=set()
    for length,sizes in lengths.items():
        for rating,count in allocate(per_length[length],sizes,sizes,VALIDATION_FLOOR).items():
            chosen.update(id(row) for row in cells[(length,rating)][:count])
    return [row for row in rows if id(row) in chosen]


def choose(candidates,training,validation,seed,protected_keys=(),protected_shas=(),validation_keys=None,strata=None):
    """Experiment roles are chosen here, independently of conversion hints.

    Validation is chosen first, so runs of different training sizes share it;
    validation_keys reuses an earlier run's exact validation matches, and
    strata (validation_strata) draws it by length and rating band instead of
    the plain seeded order. training=None takes every remaining compatible replay.
    """
    wanted=set(validation_keys or ())
    ordered=sorted(candidates,key=lambda r:(hashlib.sha256(f'{seed}|{r["replay_sha256"]}'.encode()).digest(),r['archive_sha256']))
    if wanted:ordered=[r for r in ordered if r['match_key'] in wanted]+[r for r in ordered if r['match_key'] not in wanted]
    unique=[];keys=set();shas=set()
    for row in ordered:
        key,sha=row['match_key'],row['replay_sha256']
        if key in protected_keys or sha in protected_shas or key in keys or sha in shas:continue
        keys.add(key);shas.add(sha);unique.append(row)
    if wanted:
        held=unique[:validation]
        if len(held)<validation or any(r['match_key'] not in wanted for r in held):
            raise ValueError('Reused validation matches are not all prepared and available')
    elif strata is not None:
        held=stratified(unique,validation,strata)
    else:
        held=unique[:validation]
    taken={id(r) for r in held}
    rest=[r for r in unique if id(r) not in taken]
    if training is None and not rest:
        raise ValueError(f'Need more than {validation} unique compatible prepared replays; found {len(unique)}')
    if training is not None and len(rest)<training:
        raise ValueError(f'Need {training+validation} unique compatible prepared replays; found {len(unique)}')
    return ([dict(r,split='validation') for r in held]
            +[dict(r,split='train') for r in (rest if training is None else rest[:training])])


def selection(args):
    """(dataset document, feature key) for args.training/validation/seed; shards stay in training-data/."""
    from replay_catalog import connect
    from event_store import feature_identity
    from event_dataset import EventDataset
    from generals_policy import NeuralPolicy
    policy=load(ROOT/'replay-runtime/policy.json')
    model=NeuralPolicy(policy,parameters=False)
    key=feature_identity(model)
    db=connect(CATALOG);db.row_factory=__import__('sqlite3').Row
    try:
        protected={r[0] for r in db.execute("SELECT match_key FROM replay_assignment WHERE split='test'")}
        protected.update(r['match_key'] for r in load(ROOT/'replays/test-reserve.json')['matches'])
        protected_shas={row[0] for row in db.execute('SELECT raw_sha256 FROM replay WHERE match_key IN ('+','.join('?' for _ in protected)+')',list(protected))} if protected else set()
        candidates=[]
        # Shards from the pinned engine and from the retail 1.04 engine (gamereplays.org archive,
        # scripts/retail_conversion.py; user decision 6 Nov 2026: one batch, rating-conditioned).
        # The Generals Online 062026 release plays on the pinned engine since the v6 re-pin.
        from retail_conversion import RETAIL_ENGINE_SHA
        engines=(digest(ENGINE),RETAIL_ENGINE_SHA)
        # fetchall first: the file checks below must not hold a read lock on the live catalogue
        # (journal_mode=delete, so a long read blocks the converter's commits).
        for row in db.execute('SELECT * FROM prepared_shard WHERE available=1 AND feature_key=? AND engine_sha256 IN (?,?)',(key,*engines)).fetchall():
            path=Path(row['storage_path']);path=path if path.is_absolute() else ROOT/path
            if path.is_file() and path.parent.resolve()==DATA.resolve() and path.stat().st_size==row['bytes']:
                candidates.append(dict(row))
        cleaning=None
        if getattr(args,'cleaning',True):
            # Raw-replay data cleaning (scripts/replay_cleaning.py): thresholds are
            # applied now, so any setting reuses one scan of the archive.
            from replay_cleaning import selection_filter
            dropped,scanned,cleaning=selection_filter(**getattr(args,'cleaning_options',{}))
            unscanned=sorted({r['match_key'] for r in candidates}-scanned)
            if unscanned:raise ValueError(f'{len(unscanned)} candidates lack cleaning metrics (e.g. {unscanned[0]}); run replay_cleaning.py scan')
            before=len(candidates);candidates=[r for r in candidates if r['match_key'] not in dropped]
            cleaning.update(candidates_before=before,candidates_after=len(candidates))
        listed=None
        if getattr(args,'matches',None):
            raw=load(args.matches)
            keys={m if isinstance(m,str) else m['match_key'] for m in (raw['matches'] if isinstance(raw,dict) else raw)}
            # A reused validation set may lie outside the list (all listed matches then train).
            if args.validation_from:keys|={e['match_key'] for e in load(args.validation_from)['entries'] if e['split']=='validation'}
            before=len(candidates);candidates=[r for r in candidates if r['match_key'] in keys]
            listed=dict(source=str(args.matches),sha256=digest(Path(args.matches)),listed=len(keys),
                        candidates_before=before,candidates_after=len(candidates),
                        matches_found=len({r['match_key'] for r in candidates}))
        rung=None
        if getattr(args,'min_rating',None) is not None:
            from replay_cleaning import china_match_ratings
            best=china_match_ratings()
            before=len(candidates)
            candidates=[r for r in candidates if best.get(r['match_key']) is not None and best[r['match_key']]>=args.min_rating]
            rung=dict(min_rating=args.min_rating,rule='highest China rating, estimates count',
                      candidates_before=before,candidates_after=len(candidates))
        vanilla=None
        if getattr(args,'vanilla_from',None):
            from replay_cleaning import vanilla_china_matches
            allowed={e['match_key'] for e in load(args.vanilla_from)['entries']}
            with_vanilla=vanilla_china_matches()
            before=len(candidates)
            candidates=[r for r in candidates if r['match_key'] not in with_vanilla or r['match_key'] in allowed]
            vanilla=dict(source=args.vanilla_from,candidates_before=before,candidates_after=len(candidates))
        reused=None
        if args.validation_from:
            reused={e['match_key'] for e in load(args.validation_from)['entries'] if e['split']=='validation'}
            if len(reused)!=args.validation:raise ValueError('Reused validation set has a different size')
        pool,forced,archive=archive_selection(db,candidates,args)
        training=None if args.training is None else args.training-len(forced)
        if training is not None and training<1:raise ValueError('Forced archive matches fill the whole training set')
        strata=validation_strata()
        selected=choose(pool,training,args.validation,args.seed,protected,protected_shas,reused,strata=strata)
        taken={r['match_key'] for r in selected}|{r['replay_sha256'] for r in selected}
        selected+=[dict(r,split='train') for r in forced
                   if r['match_key'] not in protected and r['replay_sha256'] not in protected_shas
                   and r['match_key'] not in taken and r['replay_sha256'] not in taken]
        doc=document(db,selected,policy,model)
        if rung:
            # The bundle's own per-shard rule (subset_manifest.match_rating) must agree with the cut.
            # A shard's player indices are the engine's; the ratings are keyed by the replay header's slot order.
            # They differ when the lobby had an empty slot between players (0.4% of the top 10% on 12 Nov), so
            # such a shard joins no rating, or the opponent's (pgo:3488269: China in header slot 2, shard player 3
            # = the GLA player's slot). Either way it is kept (the cut was met by the header rule) and counted;
            # more than 1% disagreeing would mean the two rules have drifted apart, not the known mismatch.
            from subset_manifest import match_rating
            below=sorted(e['match_key'] for e in doc['entries'] if (r:=match_rating(e)) is not None and r<args.min_rating)
            if len(below)>0.01*len(doc['entries']):raise ValueError(f'{len(below)} selected matches fall below --min-rating by the shard rule (e.g. {below[0]})')
            rung['shards_without_joined_rating']=sum(match_rating(e) is None for e in doc['entries'])
            rung['shards_joined_below_cut']=below
            # Per player (user, 17 Nov 2026: the top 20% of PLAYERS): in China-vs-China matches the weaker China player
            # can be below the floor; their episodes are skipped (event_dataset.chunks). A match whose every episode
            # would go is the known slot/player-index mismatch above (the header rule met the cut), so it is kept.
            excluded=mismatched=0
            for e in doc['entries']:
                players=[str(ep['player']) for ep in e['index']['episodes']]
                low=[p for p in players if (r:=e.get('ratings',{}).get(p)) is not None and (r['rating'] if isinstance(r,dict) else r)<args.min_rating]
                if len(low)==len(players):mismatched+=1;continue
                if low:e['excluded_players']=low;excluded+=len(low)
            rung.update(players_below_floor_excluded=excluded,matches_kept_as_index_mismatch=mismatched)
        held=collections.Counter('/'.join(strata[r['match_key']]) for r in selected if r['split']=='validation')
        doc['selection']=dict(seed=args.seed,training=args.training,validation=args.validation,
            split_policy='fresh_whole_match_and_raw_hash',feature_key=key,cleaning=cleaning,archive=archive,rung=rung,vanilla_from=vanilla,matches=listed,
            validation_strata=dict(length_minutes=LENGTH_BANDS,mean_rating=RATING_BANDS,floor=VALIDATION_FLOOR,
                                   reused=bool(reused),counts=dict(sorted(held.items()))))
        return doc,key
    finally:db.close()


# Archive replays (scripts/retail_conversion.py): the user's curated gamereplays.org batch
# and GenTool uploads, told apart by their source folder.
ARCHIVE_SOURCES={'curated':os.sep+'gamereplays'+os.sep,'gentool':os.sep+'gentool'+os.sep}
CURATED_HOLDOUT_SALT='curated-holdout-v1'


def archive_selection(db,candidates,args):
    """(pool chosen by seed, training rows forced in, provenance) for the archive options.

    archive='all' keeps every source in one pool. 'main' drops archive replays; 'curated'
    forces every curated match into training (less a fixed holdout) and fills the rest
    from the main store. GenTool uploads are only ever in 'all'. main_from restricts the
    main store to an earlier dataset's matches, so two arms share their seed order.
    """
    mode=getattr(args,'archive','all')
    if mode not in ('all','main','curated'):raise ValueError('Unknown archive mode '+str(mode))
    sources={}
    if mode=='all' and not getattr(args,'curated_holdout',0) and not getattr(args,'main_from',None):
        # One pool, no holdout: provenance is not needed, and the scan (all 1.4M conversion jobs on
        # 12 Nov) is the one long read that would block a running converter.
        return candidates,[],dict(mode=mode,curated_available=None,curated_train=0,curated_holdout=[],main_from=None)
    for sha,path in db.execute("SELECT replay_sha256,source_path FROM conversion_job WHERE instr(source_path,?)>0 OR instr(source_path,?)>0",
                               tuple(ARCHIVE_SOURCES.values())):
        sources[sha]=next(name for name,part in ARCHIVE_SOURCES.items() if part in path)
    curated=sorted((r for r in candidates if sources.get(r['replay_sha256'])=='curated'),
                   key=lambda r:hashlib.sha256(f'{CURATED_HOLDOUT_SALT}|{r["match_key"]}'.encode()).digest())
    # The holdout never depends on the seed: every arm and replicate leaves out the same games.
    count=getattr(args,'curated_holdout',0) or 0
    holdout=[r['match_key'] for r in curated[:count]]
    main=[r for r in candidates if r['replay_sha256'] not in sources]
    main_from=getattr(args,'main_from',None)
    if main_from:
        keys={e['match_key'] for e in load(main_from)['entries']}
        main=[r for r in main if r['match_key'] in keys]
    forced=[]
    if mode=='all':pool=[r for r in candidates if r['match_key'] not in set(holdout)]
    elif mode=='main':pool=main
    else:pool,forced=main,curated[count:]
    return pool,forced,dict(mode=mode,curated_available=len(curated),curated_train=len(forced),
                            curated_holdout=holdout,main_from=main_from)


def document(db,selected,policy,model):
    """Dataset document for selected prepared_shard rows (each with its 'split'); primary-key reads only."""
    from replay_catalog import document_bytes
    from enemy_side import match_factions
    entries=[];profile_cache={};deployment=None
    # Model rating inputs (generals_policy.rating_inputs), joined per match from the raw-replay scan.
    from replay_cleaning import player_ratings
    ratings=player_ratings()
    for row in selected:
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
        # Both players' identities; event_dataset.read gives each row the one that is not its own.
        factions=match_factions(entry['index']['source']['original_replay'])
        entry['faction_identities']=[list(model.encoder.identity.player_identity(f)) for f in factions] if factions else None
        entry['ratings']=ratings.get(row['match_key'],{})
        entries.append(entry)
    return dict(schema=1,kind='generals_event_dataset',complete=True,config=policy,deployment=deployment,
        capability_catalog_sha256=next(iter(profile_cache.values()))['capability_catalog_sha256'],entries=entries)


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
                    '--seed',str(plan['seed']),'--threads','1','--lanes','32','--unroll','64','--part-lanes','8','--cache-mib','256',
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


def add_archive_arguments(parser):
    parser.add_argument('--archive',choices=('all','main','curated'),default='all',
        help="all: one pool; main: no archive replays; curated: every curated match in training, the rest from the main store")
    parser.add_argument('--curated-holdout',type=int,default=0,help='Curated matches left out of training (fixed, seed-independent)')
    parser.add_argument('--main-from',help="Draw main-store matches only from this dataset manifest's matches")
    parser.add_argument('--min-rating',type=float,help='Rating ladder rung: only matches whose highest China rating '
        '(estimates count) is at least this, for training and validation. Shares of ALL replays (12 Nov): top 10%% = 1574, top 20%% = 1356')
    parser.add_argument('--vanilla-from',help="Matches with a vanilla China player only from this dataset manifest's matches "
        "(Tank/Nuke-only matches unrestricted): a faction-mix arm then gains no vanilla games its baseline lacked")
    parser.add_argument('--matches',help="Only these matches: a JSON list of match keys, or a top_matches.py ranking "
        "({'matches': [{'match_key': ...}, ...]})")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--training',type=int,default=5000)
    parser.add_argument('--validation',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--max-epochs',type=int,default=0,help='Optional ceiling; 0 trains until validation plateau')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--lr-decay',type=float,default=1.0,help='Learning-rate factor after two validation rounds in a row without improvement (1 disables)')
    parser.add_argument('--min-lr',type=float,default=0.0)
    parser.add_argument('--validation-from',help='Reuse the validation matches of an earlier dataset manifest')
    parser.add_argument('--no-cleaning',dest='cleaning',action='store_false',help='Select without the raw-replay cleaning filter')
    add_archive_arguments(parser)
    parser.add_argument('--cleaning-option',action='append',default=[],metavar='NAME=VALUE',
        help='Override a replay_cleaning threshold, e.g. checks=afk,early_end or early_end_minutes=4')
    args=parser.parse_args()
    from replay_cleaning import DEFAULTS as cleaning_defaults
    args.cleaning_options={}
    for item in args.cleaning_option:
        name,_,value=item.partition('=')
        if name not in cleaning_defaults:parser.error('Unknown cleaning option '+name)
        args.cleaning_options[name]=type(cleaning_defaults[name])(value)
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
