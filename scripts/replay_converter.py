"""Restartable background conversion with a live worker limit. Windows only."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes as w
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
import uuid

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'

from replay_conversion import (ROOT,WORK,CATALOG,DATA,connect,contract,inventory,recover,
    claim,update,counts,due,source_guard,worker_main,save,ACTIVE,remove_owned,refresh_collection)
from replay_catalog import record_failure,resolve_failures
from replay_training import AUDIT_LEVELS

CONTROL=ROOT/'replay-runtime/converter.json'
STATUS=WORK/'status.json'
DEFAULTS=dict(workers=8,paused=False,stop=False,stop_now=False,interrupt_excess=False,startup_ram_mib=2304,
    reserve_ram_mib=1024,max_private_mib=8192,reserve_disk_gib=80,timeout_seconds=2400,scan_seconds=300,
    audit_level='integrity')


def settings(path=None):
    value=json.loads(Path(path or CONTROL).read_text(encoding='utf-8-sig'))
    return validate_settings(value)


def validate_settings(value):
    if set(value)!=set(DEFAULTS):raise ValueError('Converter settings have missing or unknown fields')
    for name in ('paused','stop','stop_now','interrupt_excess'):
        if type(value[name]) is not bool:raise ValueError(name+' must be true/false')
    ranges=dict(workers=(0,64),startup_ram_mib=(128,1048576),reserve_ram_mib=(128,1048576),
        max_private_mib=(256,1048576),reserve_disk_gib=(80,1048576),timeout_seconds=(1,86400),scan_seconds=(10,86400))
    for name,(low,high) in ranges.items():
        if type(value[name]) is not int or not low<=value[name]<=high:
            raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    if value['startup_ram_mib']<value['reserve_ram_mib']:
        raise ValueError('Startup RAM must include the RAM reserve')
    if value['audit_level'] not in AUDIT_LEVELS:
        raise ValueError('audit_level must be one of '+', '.join(AUDIT_LEVELS))
    return value


class InstanceLock:
    def __init__(self,path):self.path=Path(path);self.file=None
    def __enter__(self):
        import msvcrt
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=self.path.open('a+b')
        if self.file.tell()==0:self.file.write(b'0');self.file.flush()
        self.file.seek(0)
        try:msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            self.file.close();self.file=None
            raise RuntimeError('A replay converter is already running')
        return self
    def __exit__(self,*unused):
        import msvcrt
        if self.file:
            self.file.seek(0);msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1);self.file.close()


def is_running():
    if os.name!='nt':return False
    try:
        with InstanceLock(WORK/'service.lock'):pass
        return False
    except RuntimeError:return True


class ProcessFamily:
    """Closing the parent's handle kills a worker and all native descendants.

    Handles are non-inheritable. This also covers an abruptly killed daemon;
    no stale PID from a previous service is ever used to kill a process.
    """
    def __init__(self,pid):
        class Basic(ctypes.Structure):
            _fields_=[('per_process',ctypes.c_longlong),('per_job',ctypes.c_longlong),('flags',w.DWORD),
                ('min_working',ctypes.c_size_t),('max_working',ctypes.c_size_t),('active',w.DWORD),
                ('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(name,ctypes.c_ulonglong) for name in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Extended(ctypes.Structure):
            _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
                      ('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        k=self.kernel
        k.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];k.CreateJobObjectW.restype=w.HANDLE
        k.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
        k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];k.OpenProcess.restype=w.HANDLE
        k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
        k.CloseHandle.argtypes=[w.HANDLE]
        self.handle=k.CreateJobObjectW(None,None)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
        info=Extended();info.basic.flags=0x2000
        process=None
        try:
            if not k.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            process=k.OpenProcess(0x101,False,pid)
            if not process or not k.AssignProcessToJobObject(self.handle,process):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:self.close();raise
        finally:
            if process:k.CloseHandle(process)
    def close(self):
        if self.handle:self.kernel.CloseHandle(self.handle);self.handle=None


def admission(resource,control):
    if resource.get('available_ram_bytes',0)<control['startup_ram_mib']*1024**2:return 'waiting_for_ram'
    if resource['free_disk_bytes']<control['reserve_disk_gib']*1024**3:return 'waiting_for_disk'
    if resource.get('aggregate_private_bytes',0)>=control['max_private_mib']*1024**2:return 'waiting_for_memory_budget'
    return None


def resize_plan(children,target):
    keep=set(sorted(children)[:target])
    retire=[slot for slot,child in children.items() if slot not in keep
            and child['job'] is None and child['ready'] and not child['stopping']]
    return keep,retire


def interrupt_slots(children,keep,enabled):
    return [slot for slot,child in children.items() if enabled and slot not in keep
            and not child.get('cancel_reason')]


def record_job_failure(db,identity,job,message,*,stage,reason,evidence=None):
    record_failure(db,replay_sha256=job['replay_sha256'],match_key=job['match_key'],stage=stage,reason=reason,
        engine_sha256=identity['engine_sha256'],pipeline=identity['pipeline'],message=message,evidence=evidence)


def retry_failed(db,pipeline,*,match_key=None,reason=None):
    """Close the selected replays' failures (inventory would otherwise skip them again) and re-queue them."""
    if reason:shas=[s for (s,) in db.execute('SELECT DISTINCT replay_sha256 FROM replay_failure WHERE reason=? AND resolved_utc IS NULL',(reason,))]
    else:shas=[s for (s,) in db.execute('SELECT replay_sha256 FROM conversion_job WHERE pipeline=? AND match_key=?',(pipeline,match_key))]
    reset=0
    for sha in shas:
        resolve_failures(db,sha)
        reset+=db.execute("""UPDATE conversion_job SET status='pending',stage='queued',attempts=0,retry_after=0,message=NULL
            WHERE pipeline=? AND replay_sha256=? AND status IN ('blocked','rejected')""",(pipeline,sha)).rowcount
    db.commit();return dict(replays=len(shas),reset=reset)


def logger():
    log=logging.getLogger('replay-converter');log.setLevel(logging.INFO)
    if not log.handlers:
        handler=RotatingFileHandler(WORK/'service.log',maxBytes=4*1024**2,backupCount=1,encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'));log.addHandler(handler)
    return log


def run(args):
    if os.name!='nt':raise RuntimeError('The native replay runtime requires Windows')
    WORK.mkdir(parents=True,exist_ok=True);DATA.mkdir(exist_ok=True)
    with InstanceLock(WORK/'service.lock'):
        log=logger();identity=contract();pipeline=identity['pipeline'];session=str(uuid.uuid4())
        control=settings();db=connect();children={};context=mp.get_context('spawn');outbox=context.Queue()
        save(STATUS,dict(status='discovering',pid=os.getpid(),session=session,pipeline=pipeline,updated=time.time()))
        log.info('Starting session %s; recovered %d jobs',session,recover(db,pipeline))
        if args.match_key:
            # Qualification mode adds only its requested recording; normal mode
            # always scans the entire collected corpus.
            identity['only_match_keys']=args.match_key
        if not args.match_key:refresh_collection(db)
        inventory(db,identity)
        from pipeline_resources import tree
        now=time.monotonic();next_scan=now+control['scan_seconds'];next_guard=now
        completed=0;fatal=None;reason='running';last_control=control

        def retire(slot,kill=False,cancel_reason=None):
            child=children[slot]
            if kill:
                child['cancel_reason']=cancel_reason
                child['family'].close()
            elif not child['stopping']:child['inbox'].put(None);child['stopping']=True

        try:
            while True:
                settings_error=None
                try:control=settings();last_control=control
                except (OSError,ValueError) as error:control=last_control;settings_error=str(error)
                if time.monotonic()>=next_guard:
                    try:source_guard(identity)
                    except (OSError,ValueError) as error:fatal=str(error)
                    next_guard=time.monotonic()+30
                while True:
                    try:event=outbox.get_nowait()
                    except queue.Empty:break
                    child=children.get(event['slot'])
                    if child is None or event.get('pid')!=child['process'].pid:continue
                    if event['event']=='ready':child['ready']=True
                    elif event['event']=='stage':
                        if child['job'] and event['sha']==child['job']['replay_sha256']:
                            stage=event['stage'];update(db,pipeline,event['sha'],stage=stage,
                                status=stage if stage in ACTIVE else 'running')
                    elif event['event']=='result':
                        result=event['result'];job=child['job']
                        if job is None or event['sha']!=job['replay_sha256']:raise RuntimeError('Worker result does not own its queue job')
                        fields={key:result[key] for key in ('status','stage','message') if key in result}
                        fields.update(owner=None,retry_after=time.time()+30 if result['status']=='pending' else 0)
                        if result['status']=='pending':fields['attempts']=job['attempts']
                        if result.get('failure'):record_job_failure(db,identity,job,result['message'],**result['failure'])
                        elif result['status']=='done':resolve_failures(db,job['replay_sha256'])
                        update(db,pipeline,event['sha'],**fields)
                        log.info('%s %s %s',job['match_key'],result['status'],result.get('message',''))
                        completed+=1;child['job']=None
                    elif event['event']=='fatal':
                        fatal='Worker initialization/execution failed: '+event['message'];log.error(fatal)
                # A dead worker never permanently owns a queue item.
                for slot,child in list(children.items()):
                    if child['process'].is_alive():continue
                    child['process'].join()
                    if child['job'] is None and child['process'].exitcode and not child.get('cancel_reason'):
                        fatal=fatal or f'Worker {slot} exited unexpectedly ({child["process"].exitcode})'
                    if child['job']:
                        job=child['job'];row=db.execute('SELECT attempts FROM conversion_job WHERE pipeline=? AND replay_sha256=?',(pipeline,job['replay_sha256'])).fetchone()
                        cancelled=child.get('cancel_reason')
                        state='pending' if cancelled or row[0]<3 else 'blocked'
                        fields=dict(status=state,owner=None,retry_after=time.time()+30,
                            message=cancelled or 'Worker exited before committing its result')
                        if cancelled:fields['attempts']=max(0,row[0]-1)
                        if state=='blocked':
                            record_job_failure(db,identity,job,fields['message'],stage='conversion',reason='worker_exited',
                                evidence=dict(exitcode=child['process'].exitcode,attempts=row[0]))
                        update(db,pipeline,job['replay_sha256'],**fields)
                    child['family'].close();child['inbox'].close();child['process'].close();del children[slot]
                    remove_owned(WORK/'workers'/str(slot))
                stopping=control['stop'] or control['stop_now'] or fatal is not None or (args.limit and completed>=args.limit)
                resource=tree()
                emergency=(resource.get('available_ram_bytes',0)<control['reserve_ram_mib']*1024**2
                    or resource.get('aggregate_private_bytes',0)>control['max_private_mib']*1024**2)
                if control['stop_now'] or emergency:
                    # Cancel one owned process family at a time under pressure.
                    victims=sorted(children,reverse=True) if control['stop_now'] else sorted(children,reverse=True)[:1]
                    for slot in victims:retire(slot,kill=True,cancel_reason='Requested immediate stop' if control['stop_now'] else 'RAM reserve reached')
                target=0 if stopping or control['paused'] or settings_error else control['workers']
                if args.match_key:target=min(target,1)
                keep,retiring=resize_plan(children,target)
                for slot in retiring:retire(slot)
                for slot in interrupt_slots(children,keep,control['interrupt_excess']):
                    retire(slot,kill=True,cancel_reason='Requested immediate worker reduction')
                reason='stopping' if stopping else 'paused' if control['paused'] or control['workers']==0 else admission(resource,control) or 'running'
                if settings_error:reason='invalid_settings'
                if fatal:reason='stopping_after_error'
                if not stopping and not emergency and not settings_error and target>0 and not admission(resource,control):
                    for slot in sorted(keep):
                        child=children[slot]
                        if child['ready'] and child['job'] is None and not child['stopping']:
                            job=claim(db,pipeline,session+':'+str(slot),args.match_key)
                            if job:
                                child['job']=job;child['inbox'].put((job,control.copy()))
                    ready_pending=due(db,pipeline,args.match_key)
                    active=sum(not c['stopping'] for c in children.values())
                    if ready_pending and active<target:
                        slot=next(i for i in range(64) if i not in children)
                        inbox=context.Queue(maxsize=2)
                        process=context.Process(target=worker_main,args=(slot,inbox,outbox,identity),name='replay-converter-'+str(slot))
                        process.start()
                        try:family=ProcessFamily(process.pid)
                        except BaseException:
                            process.terminate();process.join();raise
                        children[slot]=dict(process=process,family=family,inbox=inbox,ready=False,stopping=False,job=None)
                        inbox.put('boot')
                total=counts(db,pipeline,args.match_key)
                if not total.get('pending',0) and not any(total.get(k,0) for k in ACTIVE) and not stopping:
                    reason='caught_up'
                    for slot,c in children.items():
                        if c['ready'] and c['job'] is None:retire(slot)
                    if args.limit or args.match_key:stopping=True
                save(STATUS,dict(status=reason,pid=os.getpid(),session=session,pipeline=pipeline,feature_key=identity['feature_key'],
                    requested_workers=control['workers'],live_workers=len(children),busy_workers=sum(c['job'] is not None for c in children.values()),
                    counts=total,resources=resource,settings_error=settings_error,error=fatal,updated=time.time(),
                    workers=[dict(slot=s,pid=c['process'].pid,retiring=s not in keep or c['stopping'],
                        match_key=c['job']['match_key'] if c['job'] else None) for s,c in children.items()]))
                if stopping and not children:break
                if not stopping and time.monotonic()>=next_scan:
                    if not args.match_key:refresh_collection(db)
                    inventory(db,identity);next_scan=time.monotonic()+control['scan_seconds']
                time.sleep(2)
        except KeyboardInterrupt:
            fatal='Foreground converter interrupted; unfinished jobs remain resumable'
        finally:
            for child in children.values():child['family'].close()
            for child in children.values():
                child['process'].join(timeout=30);child['inbox'].close()
            recover(db,pipeline)
            save(STATUS,dict(status='failed' if fatal else 'stopped',pid=os.getpid(),pipeline=pipeline,
                counts=counts(db,pipeline,args.match_key),error=fatal,updated=time.time(),live_workers=0,busy_workers=0))
            db.close();outbox.close();outbox.cancel_join_thread()
        if fatal:raise RuntimeError(fatal)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('start');sub.add_parser('status')
    stop=sub.add_parser('stop');stop.add_argument('--now',action='store_true')
    change=sub.add_parser('set');change.add_argument('--workers',type=int)
    change.add_argument('--now',action='store_true',help='Interrupt excess workers instead of draining their current replay')
    state=change.add_mutually_exclusive_group();state.add_argument('--pause',action='store_true');state.add_argument('--resume',action='store_true')
    change.add_argument('--audit-level',choices=AUDIT_LEVELS,help='full adds the per-entity observation contract (engine qualification)')
    retry=sub.add_parser('retry',help='Re-queue failed replays and close their failure records')
    target=retry.add_mutually_exclusive_group(required=True)
    target.add_argument('--match-key');target.add_argument('--reason',help='Every open failure with this reason')
    failed=sub.add_parser('failures',help='Summarise recorded replay failures, or list one reason')
    failed.add_argument('--reason');failed.add_argument('--limit',type=int,default=20)
    failed.add_argument('--resolved',action='store_true',help='Include failures later fixed by a successful conversion')
    foreground=sub.add_parser('run');foreground.add_argument('--limit',type=int,default=0)
    foreground.add_argument('--match-key',action='append')
    args=parser.parse_args()
    if args.command=='run':return run(args)
    if args.command=='status':
        status=json.loads(STATUS.read_text()) if STATUS.exists() else dict(status='not_started')
        status.update(running=is_running(),settings=settings())
        print(json.dumps(status,indent=2));return
    if args.command=='failures':
        db=connect();open_only='' if args.resolved else ' AND resolved_utc IS NULL'
        if args.reason:
            rows=db.execute('SELECT match_key,stage,substr(engine_sha256,1,16),occurrences,last_utc,message FROM replay_failure WHERE reason=?'
                +open_only+' ORDER BY last_utc DESC LIMIT ?',(args.reason,args.limit)).fetchall()
            print(json.dumps([dict(zip(('match_key','stage','engine','occurrences','last_utc','message'),r)) for r in rows],indent=2))
        else:
            rows=db.execute('SELECT stage,reason,COUNT(DISTINCT replay_sha256) FROM replay_failure WHERE 1=1'+open_only
                +' GROUP BY stage,reason ORDER BY 3 DESC').fetchall()
            print(json.dumps([dict(stage=s,reason=r,replays=n) for s,r,n in rows],indent=2))
        db.close();return
    if args.command=='retry':
        if is_running():raise RuntimeError('Stop the converter before resetting failed jobs')
        identity=contract();db=connect()
        print(json.dumps(retry_failed(db,identity['pipeline'],match_key=args.match_key,reason=args.reason)));db.close();return
    if args.command=='start' and is_running():raise RuntimeError('Converter is already running; use set to resize it')
    control=settings()
    if args.command=='set':
        if args.now and args.workers is None and not args.pause:parser.error('--now needs --workers or --pause')
        control['interrupt_excess']=args.now
        if args.workers is not None:control['workers']=args.workers
        if args.pause:control['paused']=True
        if args.resume:control['paused']=False
        if args.audit_level:control['audit_level']=args.audit_level
    elif args.command=='stop':control.update(stop=True,stop_now=args.now)
    elif args.command=='start':control.update(stop=False,stop_now=False)
    # Validate before replacing the user-editable live settings file.
    validate_settings(control)
    save(CONTROL,control)
    if args.command=='start':
        WORK.mkdir(parents=True,exist_ok=True)
        with (WORK/'launcher.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'run'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        print(json.dumps(dict(started_pid=process.pid,settings=str(CONTROL),status=str(STATUS))))
    else:print(json.dumps(control,indent=2))


if __name__=='__main__':
    mp.freeze_support()
    main()
