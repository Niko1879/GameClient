"""Queue recovery, exclusion/reuse, live resizing and native process ownership."""
import copy
import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import sqlite3
import unittest
from unittest.mock import patch

import replay_conversion as core
import replay_converter as service


def family_child(inbox,outbox):
    inbox.get()
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],
        creationflags=subprocess.CREATE_NO_WINDOW)
    outbox.put(child.pid)
    child.wait()


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.path=self.root/'catalog.sqlite3';self.db=core.connect(self.path)
        self.db.execute('''CREATE TABLE replay(match_key TEXT PRIMARY KEY,raw_path TEXT,
            raw_sha256 TEXT,raw_status TEXT)''')
        self.identity=dict(pipeline='p',feature_key='f',engine_sha256='e')

    def tearDown(self):self.db.close();self.temp.cleanup()

    def raw(self,key,sha,path='216B3EEA/match_100.rep'):
        self.db.execute('INSERT INTO replay VALUES(?,?,?,?)',(key,path,sha,'downloaded'));self.db.commit()

    def inventory(self,reserve=None):return core.inventory(self.db,self.identity,reserve or {'matches':[]})

    def test_restart_recovers_only_unfinished_work_and_preserves_rejections(self):
        self.raw('pgo:100','a'*64);self.raw('pgo:101','b'*64)
        self.inventory()
        first=core.claim(self.db,'p','first')
        second=core.claim(self.db,'p','first')
        core.update(self.db,'p',second['replay_sha256'],status='rejected',message='crc_diverged')
        self.assertEqual(core.recover(self.db,'p'),1)
        replay=core.claim(self.db,'p','second')
        self.assertEqual(replay['replay_sha256'],first['replay_sha256'])
        self.assertIsNone(core.claim(self.db,'p','third'))
        self.inventory()
        self.assertEqual(core.counts(self.db,'p').get('rejected'),1)

    def record_failure(self,sha,stage,reason,engine='e',key='pgo:100'):
        service.record_job_failure(self.db,dict(self.identity,engine_sha256=engine),
            dict(replay_sha256=sha,match_key=key),reason+' message',stage=stage,reason=reason,evidence=dict(log_tail=['x']))

    def test_failures_are_recorded_with_reason_evidence_and_occurrences(self):
        self.record_failure('a'*64,'playback','crc_diverged');self.record_failure('a'*64,'playback','crc_diverged')
        self.record_failure('b'*64,'header','desync_or_disconnect',key='pgo:101')
        rows=self.db.execute('SELECT match_key,stage,reason,engine_sha256,occurrences,message,evidence_sha256 '
            'FROM replay_failure ORDER BY match_key').fetchall()
        self.assertEqual([r[:6] for r in rows],[('pgo:100','playback','crc_diverged','e',2,'crc_diverged message'),
            ('pgo:101','header','desync_or_disconnect','',1,'desync_or_disconnect message')])
        from replay_catalog import document_bytes
        self.assertEqual(json.loads(document_bytes(self.db,rows[0][6])),dict(log_tail=['x']))

    def shard(self,sha,key='pgo:100',name='existing.zip',engine='e'):
        from replay_catalog import store_document
        doc=store_document(self.db,b'{}','test','test')
        self.db.execute('INSERT OR IGNORE INTO prepared_profile VALUES(?,?,?)',('profile','f',doc))
        path=self.root/name;path.write_bytes(b'physical shard')
        self.db.execute('INSERT INTO prepared_shard VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (name,key,sha,'export',engine,'f','index',str(path),path.stat().st_size,1,'profile',doc))
        self.db.commit();return path

    def status(self):
        return dict(self.db.execute("SELECT substr(replay_sha256,1,1),status FROM conversion_job WHERE pipeline='p'"))

    def test_every_open_failure_is_skipped_until_retried(self):
        cases=(('a','header','desync_or_disconnect','old'),('b','playback','crc_diverged','e'),
               ('c','playback','crc_diverged','old'),('d','playback','timeout','e'),
               ('f','conversion','vocabulary','e'),('9','playback','map_mismatch','e'))
        for n,(sha,stage,reason,engine) in enumerate(cases):
            self.raw(f'pgo:{100+n}',sha*64);self.record_failure(sha*64,stage,reason,engine,key=f'pgo:{100+n}')
        from replay_catalog import resolve_failures
        resolve_failures(self.db,'9'*64)
        self.inventory()
        self.assertEqual(self.status(),{'a':'rejected','b':'rejected','c':'rejected','d':'rejected','f':'rejected','9':'pending'})
        self.assertIn('Known playback failure: crc_diverged',self.db.execute(
            "SELECT message FROM conversion_job WHERE replay_sha256=?",('b'*64,)).fetchone()[0])
        self.assertEqual(service.retry_failed(self.db,'p',reason='crc_diverged'),dict(replays=2,reset=2))
        self.inventory()
        self.assertEqual(self.status()['b'],'pending')
        service.retry_failed(self.db,'p',match_key='pgo:100');self.inventory()
        self.assertEqual(self.status()['a'],'pending')

    def test_queued_replays_follow_later_failures_and_existing_shards(self):
        self.raw('pgo:100','a'*64);self.raw('pgo:101','b'*64);self.raw('pgo:102','c'*64);self.inventory()
        self.record_failure('a'*64,'playback','crc_diverged');self.shard('b'*64,'pgo:101')
        self.inventory()
        self.assertEqual(self.status(),{'a':'rejected','b':'done','c':'pending'})
        self.assertIsNone(core.claim(self.db,'p','w',['pgo:100','pgo:101']))

    def test_another_engines_shard_does_not_count_as_converted(self):
        # Training selects by export engine, so a re-pinned engine re-exports.
        self.shard('a'*64,engine='old');self.raw('pgo:100','a'*64);self.inventory()
        self.assertEqual(self.status(),{'a':'pending'})
        self.assertIsNone(core.reusable(self.db,dict(match_key='pgo:100',replay_sha256='a'*64,split='train'),self.identity))

    def test_new_pipeline_marks_existing_shards_done_and_rechecks_disk(self):
        path=self.shard('a'*64);self.raw('pgo:100','a'*64);self.inventory()
        self.assertEqual(self.status(),{'a':'done'})
        self.assertIn('existing.zip',self.db.execute('SELECT shard_path FROM replay_status').fetchone()[0])
        path.unlink()
        self.assertEqual(self.inventory()['reopened'],1)
        self.assertEqual(self.status(),{'a':'pending'})
        self.assertEqual(tuple(self.db.execute('SELECT state,shard_path FROM replay_status').fetchone()),('unconverted',None))

    def test_two_connections_never_claim_the_same_recording(self):
        self.raw('pgo:100','a'*64);self.inventory()
        other=core.connect(self.path)
        try:
            self.assertIsNotNone(core.claim(self.db,'p','one'))
            self.assertIsNone(core.claim(other,'p','two'))
        finally:other.close()

    def test_duplicate_raw_bytes_and_test_alias_are_not_trainable(self):
        self.raw('pgo:100','a'*64);self.raw('pgo:101','a'*64)
        self.inventory({'matches':[{'match_key':'pgo:101'}]})
        self.assertEqual(core.counts(self.db,'p'),{'heldout':1})
        self.assertIsNone(core.claim(self.db,'p','worker'))

    def test_blacklisted_release_is_excluded_before_export(self):
        self.raw('pgo:100','a'*64,'FBA615FA/match_100.rep');self.inventory()
        self.assertEqual(core.counts(self.db,'p'),{'excluded':1})

    def test_parked_release_is_excluded_without_failures_and_requeued_when_unparked(self):
        self.raw('pgo:100','a'*64,'3DF43954/match_100.rep');self.raw('pgo:101','b'*64,'B9DB8815/match_101.rep')
        with patch.dict(core.PARKED_RELEASES,{},clear=True):self.inventory()
        self.assertEqual(self.status(),{'a':'pending','b':'pending'})
        with patch.dict(core.PARKED_RELEASES,{'3DF43954':'Parked for this test'},clear=True):self.inventory()
        self.assertEqual(self.status(),{'a':'excluded','b':'pending'})
        self.assertEqual(self.db.execute('SELECT count(*) FROM replay_failure').fetchone()[0],0)
        with patch.dict(core.PARKED_RELEASES,{},clear=True):self.inventory()
        self.assertEqual(self.status(),{'a':'pending','b':'pending'})

    def test_releases_recorded_before_simulation_fixes_use_their_replay_logic(self):
        self.assertEqual(core.replay_logic('3df43954'),'081326')
        self.assertEqual(core.replay_logic('216B3EEA'),'current')

    def test_rescan_does_not_resolve_or_reinsert_already_queued_source_paths(self):
        self.raw('pgo:100','a'*64);self.inventory()
        with patch.object(core,'resolve_recording',side_effect=AssertionError('Unnecessary file-path scan')):
            self.assertEqual(self.inventory(),dict(added=0,reopened=0))

    def test_missing_shard_reopens_a_success_instead_of_trusting_export_history(self):
        self.raw('pgo:100','a'*64);self.inventory()
        core.update(self.db,'p','a'*64,status='done')
        self.assertEqual(self.inventory()['reopened'],1)
        self.assertIsNotNone(core.claim(self.db,'p','again'))

    def test_extant_prepared_file_keeps_done_job_out_of_queue(self):
        from replay_catalog import store_document
        self.raw('pgo:100','a'*64);self.inventory()
        doc=store_document(self.db,b'{}','test','test')
        self.db.execute('INSERT INTO prepared_profile VALUES(?,?,?)',('profile','f',doc))
        shard=self.root/'existing.zip';shard.write_bytes(b'physical shard')
        self.db.execute('INSERT INTO prepared_shard VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            ('archive','pgo:100','a'*64,'export','e','f','index',str(shard),shard.stat().st_size,1,'profile',doc))
        self.db.commit();core.update(self.db,'p','a'*64,status='done')
        self.assertEqual(self.inventory()['reopened'],0)
        self.assertIsNone(core.claim(self.db,'p','worker'))

    def test_retry_delay_and_qualification_filter_are_respected(self):
        import time
        self.raw('pgo:100','a'*64);self.raw('pgo:101','b'*64);self.inventory()
        core.update(self.db,'p','a'*64,retry_after=time.time()+100)
        self.assertFalse(core.due(self.db,'p',['pgo:100']))
        self.assertIsNone(core.claim(self.db,'p','w',['pgo:100']))
        self.assertEqual(core.claim(self.db,'p','w',['pgo:101'])['match_key'],'pgo:101')

    def test_cleanup_never_deletes_sources_or_shared_training_directory(self):
        with self.assertRaises(ValueError):core.remove_owned(self.root,root=self.root)
        with self.assertRaises(ValueError):core.remove_owned(self.root.parent,root=self.root)
        owned=self.root/'owned';owned.mkdir();(owned/'raw.bin').write_bytes(b'trial')
        core.remove_owned(owned,root=self.root);self.assertFalse(owned.exists())


class ControlTests(unittest.TestCase):
    def test_handshake_catalogue_uses_measured_executable_hash(self):
        import hashlib
        with tempfile.TemporaryDirectory() as name:
            executable=Path(name)/'native.exe';executable.write_bytes(b'known local build')
            sha=hashlib.sha256(executable.read_bytes()).hexdigest()
            metadata=dict(event='ready',protocol_version=1,ticks_per_second=60)
            self.assertEqual(core.verified_catalog(metadata,executable,sha)['engine_exe_sha256'],sha)
            with self.assertRaises(ValueError):core.verified_catalog(metadata,executable,'incorrect')

    def test_downsize_drains_busy_workers_and_upsize_reuses_them(self):
        workers={i:dict(job={'id':i},ready=True,stopping=False) for i in range(8)}
        keep,retire=service.resize_plan(workers,2)
        self.assertEqual(keep,{0,1});self.assertEqual(retire,[])
        workers[5]['job']=None
        self.assertEqual(service.resize_plan(workers,2)[1],[5])
        self.assertEqual(service.resize_plan(workers,8),(set(range(8)),[]))

    def test_immediate_resize_cancels_only_excess_process_families(self):
        workers={i:dict(job={'id':i},ready=True,stopping=False) for i in range(4)}
        keep,_=service.resize_plan(workers,2)
        self.assertEqual(service.interrupt_slots(workers,keep,False),[])
        self.assertEqual(service.interrupt_slots(workers,keep,True),[2,3])
        workers[2]['cancel_reason']='already cancelled'
        self.assertEqual(service.interrupt_slots(workers,keep,True),[3])

    def test_invalid_settings_do_not_replace_previous_file(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'control.json';core.save(p,service.DEFAULTS)
            with patch.object(service,'CONTROL',p),patch.object(sys,'argv',['converter','set','--workers','-1']):
                with self.assertRaises(ValueError):service.main()
            self.assertEqual(service.settings(p)['workers'],8)

    def test_live_worker_change_and_pause_preserve_other_controls(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'control.json';core.save(p,service.DEFAULTS)
            with patch.object(service,'CONTROL',p),patch.object(sys,'argv',['converter','set','--workers','2','--pause']):service.main()
            value=service.settings(p)
            self.assertEqual(value['workers'],2);self.assertTrue(value['paused'])
            self.assertEqual(value['reserve_ram_mib'],1024)

    def test_audit_level_defaults_to_integrity_and_rejects_unknown_levels(self):
        self.assertEqual(service.DEFAULTS['audit_level'],'integrity')
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'control.json';core.save(p,service.DEFAULTS)
            with patch.object(service,'CONTROL',p),patch.object(sys,'argv',['converter','set','--audit-level','full']):service.main()
            self.assertEqual(service.settings(p)['audit_level'],'full')
        with self.assertRaisesRegex(ValueError,'audit_level'):
            service.validate_settings(dict(service.DEFAULTS,audit_level='none'))

    def test_resource_admission_is_separate_from_desired_worker_count(self):
        resource=dict(available_ram_bytes=3*1024**3,free_disk_bytes=100*1024**3,aggregate_private_bytes=0)
        self.assertIsNone(service.admission(resource,service.DEFAULTS))
        resource['available_ram_bytes']=1024**3
        self.assertEqual(service.admission(resource,service.DEFAULTS),'waiting_for_ram')


class CollectionTests(unittest.TestCase):
    def test_crawler_changes_are_imported_without_rewriting_history_or_source_db(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'harvest.sqlite3';c=sqlite3.connect(source)
            c.execute('CREATE TABLE matches(id INTEGER PRIMARY KEY,candidate TEXT,status TEXT,target TEXT,sha256 TEXT,error TEXT,updated TEXT)')
            c.execute('INSERT INTO matches VALUES(1,?,\'downloaded\',\'216B3EEA/match_1.rep\',?,NULL,\'20\')',('{}','a'*64));c.commit()
            db=core.connect(root/'catalog.sqlite3')
            try:
                self.assertEqual(core.refresh_collection(db,source),1)
                self.assertEqual(core.refresh_collection(db,source),0)
                # A new row can have an older timestamp: it must still appear.
                c.execute('INSERT INTO matches VALUES(2,?,\'downloaded\',\'216B3EEA/match_2.rep\',?,NULL,\'10\')',('{}','b'*64));c.commit()
                self.assertEqual(core.refresh_collection(db,source),1)
                c.execute("UPDATE matches SET status='error',updated='30' WHERE id=1");c.commit()
                self.assertEqual(core.refresh_collection(db,source),1)
                self.assertEqual(db.execute("SELECT raw_status FROM replay WHERE match_key='pgo:1'").fetchone()[0],'error')
                self.assertEqual(c.execute('SELECT COUNT(*) FROM matches').fetchone()[0],2)
            finally:db.close();c.close()


@unittest.skipUnless(os.name=='nt','Windows process-family ownership')
class OwnershipTests(unittest.TestCase):
    def test_closing_owner_handle_stops_worker_and_its_native_descendant(self):
        context=multiprocessing.get_context('spawn');inbox=context.Queue();outbox=context.Queue()
        process=context.Process(target=family_child,args=(inbox,outbox));process.start()
        family=None;handle=None
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.OpenProcess.restype=ctypes.c_void_p
        k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        k.CloseHandle.argtypes=[ctypes.c_void_p]
        try:
            family=service.ProcessFamily(process.pid);inbox.put('boot')
            pid=outbox.get(timeout=15);handle=k.OpenProcess(0x100000,False,pid)
            self.assertTrue(handle)
            family.close();process.join(timeout=10)
            self.assertFalse(process.is_alive())
            self.assertEqual(k.WaitForSingleObject(handle,10000),0)
        finally:
            if family:family.close()
            if handle:k.CloseHandle(handle)
            if process.is_alive():process.terminate();process.join()
            process.close();inbox.close();outbox.close()


if __name__=='__main__':unittest.main()
