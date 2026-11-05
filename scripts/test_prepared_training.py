import copy
import hashlib
import json
from pathlib import Path
import sys
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import run_prepared_training as runner
import train_events as trainer


class EarlyStoppingTests(unittest.TestCase):
    def test_plateau_retains_best_and_waits_three_epochs(self):
        state=trainer.EarlyStopping(3)
        scores=[3,2,2.1,2,2.3]
        for i,score in enumerate(scores,1):
            state.observe(i,score)
            self.assertEqual(state.stopped,i==5)
        self.assertEqual(state.best_epoch,2);self.assertEqual(state.best_score,2)

    def test_resume_preserves_patience_and_rejects_nonfinite_loss(self):
        state=trainer.EarlyStopping(3);state.observe(1,2);state.observe(2,3)
        resumed=trainer.EarlyStopping(3,state=state.state())
        resumed.observe(3,4);self.assertFalse(resumed.stopped)
        resumed.observe(4,5);self.assertTrue(resumed.stopped)
        with self.assertRaises(ValueError):resumed.observe(5,float('nan'))
        with self.assertRaises(ValueError):trainer.EarlyStopping(2,state=state.state())

    def test_best_copy_recovers_atomic_publication_gap_but_not_missing_old_weights(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);resume=root/'latest.npz';best=root/'best.npz'
            cursor=dict(epoch=2,position=[0,0],validations=0)
            metadata=dict(event_dataset_sha256='d',event_cursor=dict(cursor),
                          event_history=[dict(round=1),dict(round=2,cursor=dict(cursor))])
            np.savez(resume,metadata=np.asarray(json.dumps(dict(metadata=metadata))))
            state=trainer.EarlyStopping(3);state.observe(2,1)
            trainer.restore_best_checkpoint(resume,best,state,metadata)
            self.assertEqual(best.read_bytes(),resume.read_bytes())
            # Training after the best round moved the cursor: its weights are gone.
            best.unlink();metadata['event_cursor']['position']=[0,4]
            with self.assertRaises(ValueError):trainer.restore_best_checkpoint(resume,best,state,metadata)
            # So did a later round.
            metadata['event_cursor']=dict(cursor);metadata['event_history'].append(dict(round=3,cursor=dict(cursor)))
            with self.assertRaises(ValueError):trainer.restore_best_checkpoint(resume,best,state,metadata)

    def run_loop(self,**changes):
        class Dataset:
            config={'hidden':2};digest='dataset';key='feature';document={}
            entries=[dict(match_key='t',split='train',index={'source':{}}),dict(match_key='v',split='validation',index={'source':{}})]
            def __init__(self,*unused):pass
            def batches(self,split,**unused):yield [dict(split=split)],(0,1)
        class Policy:
            parameters={}
            def __init__(self,*unused):pass
        class Learner:
            def __init__(self,*unused,**options):
                self.step_count=0;self.scores=iter([3,2,2.1,2,2.3]);self.lr=.0003
                self.model=SimpleNamespace(device=SimpleNamespace(type='cpu'),stager=SimpleNamespace(stats={}),
                                           float_dtype=np.float32,device_type='cpu')
                self.fp16=False
            def set_lr(self,lr):self.lr=lr
            def run(self,prepared,train):
                if train:self.step_count+=1
                return dict(loss_sum=1 if train else next(self.scores),decisions=1)
            def save(self,path,metadata):
                with Path(path).open('wb') as f:np.savez(f,metadata=np.asarray(json.dumps(dict(metadata=metadata))),step=self.step_count)
        torch=SimpleNamespace(manual_seed=lambda seed:None,__version__='fake',version=SimpleNamespace(hip=None))
        resource=dict(available_ram_bytes=10*1024**3,free_disk_bytes=100*1024**3,private_bytes=0)
        with tempfile.TemporaryDirectory() as name,patch.dict(sys.modules,{'torch':torch}),\
                patch.object(trainer,'EventDataset',Dataset),patch.object(trainer,'NeuralPolicy',Policy),\
                patch.object(trainer,'EventLearner',Learner),patch.object(trainer,'feature_identity',return_value='feature'),\
                patch.object(trainer,'snapshot',return_value=resource),patch.object(trainer,'Adam',return_value=None),                patch.object(trainer,'PreparationSpec'),                patch.object(trainer,'prepared_batches',lambda data,split,*,spec,pair_budget,reader,**settings:data.batches(split,**settings)):
            args=SimpleNamespace(output=name,dataset='ignored',resume=None,device='cpu',epochs=0,
                max_updates=0,seed=123,lr=.0003,threads=1,batch_size=2,pair_budget=100,
                buffer_mib=1,checkpoint_seconds=600,min_free_ram_mib=1024,max_private_mib=8192,
                stop_file=None,early_stopping_patience=3,min_delta=0,prefetch_batches=0,**changes)
            report=trainer.train(args)
            with np.load(Path(name)/'best.npz') as saved:best=int(saved['step'])
            with np.load(Path(name)/'latest.npz') as saved:latest=int(saved['step'])
            return report,best,latest

    def test_training_loop_stops_on_plateau_and_best_weights_are_from_best_epoch(self):
        report,best,latest=self.run_loop()
        self.assertEqual(report['status'],'early_stopped');self.assertEqual(report['cursor']['epoch'],5)
        self.assertEqual(report['early_stopping']['best_epoch'],2)
        self.assertEqual((best,latest),(2,5))
        self.assertEqual({h['lr'] for h in report['history']},{.0003})

    def test_learning_rate_drops_after_each_non_improving_epoch_until_stopping(self):
        report,best,latest=self.run_loop(lr_decay=.5,min_lr=.0001)
        self.assertEqual(report['status'],'early_stopped');self.assertEqual(report['early_stopping']['best_epoch'],2)
        # Epochs 3 and 4 do not improve; the third bad epoch stops without another decay.
        self.assertEqual([h['lr'] for h in report['history']],[.0003,.0003,.0003,.00015,.0001])
        self.assertEqual(report['settings']['lr_schedule'],dict(decay=.5,min_lr=.0001))


class ValidationRoundTests(unittest.TestCase):
    """Mid-epoch validation over a fake 4-window dataset; validation loss depends only on the update count."""
    def run_training(self,root,scores=None,**changes):
        class Dataset:
            config={'hidden':2};digest='dataset';key='feature';document={}
            entries=[dict(match_key='t',split='train',index={'source':{}}),dict(match_key='v',split='validation',index={'source':{}})]
            def __init__(self,*unused):pass
            def windows(self,split,**unused):return [0,1,2,3]
            def batches(self,split,*,cursor=(0,0),**unused):
                if split=='validation':
                    yield [dict(split=split)],(0,1);yield [dict(split=split)],(0,2);return
                for window in range(cursor[0],4):
                    for batch in range(cursor[1] if window==cursor[0] else 0,1):
                        yield [dict(split=split)],(window,batch+1)
        losses=scores or {}
        class Policy:
            parameters={}
            def __init__(self,*unused):self.step=0
            @staticmethod
            def load(path):
                with np.load(path) as saved:
                    policy=Policy();policy.step=int(saved['step'])
                    return policy,None,json.loads(str(saved['metadata']))
        class Learner:
            def __init__(self,reference,*unused,**options):
                self.step_count=reference.step;self.lr=.0003;self.fp16=False
                self.model=SimpleNamespace(device=SimpleNamespace(type='cpu'),stager=SimpleNamespace(stats={}),
                                           float_dtype=np.float32,device_type='cpu')
            def set_lr(self,lr):self.lr=lr
            def restore_rng(self,metadata):pass
            def restore_scaler(self,metadata):pass
            def run(self,prepared,train):
                if train:
                    self.step_count+=1
                    return dict(loss_sum=1,decisions=1,updates=1)
                stop=changes.get('stop_during_validation')
                if stop and self.step_count==stop[0] and not Path(stop[1]).exists():
                    Path(stop[1]).write_text('{}')
                return dict(loss_sum=losses.get(self.step_count,10-self.step_count),decisions=1)
            def save(self,path,metadata):
                with Path(path).open('wb') as f:np.savez(f,metadata=np.asarray(json.dumps(dict(metadata=metadata))),step=self.step_count)
        torch=SimpleNamespace(manual_seed=lambda seed:None,__version__='fake',version=SimpleNamespace(hip=None))
        resource=dict(available_ram_bytes=10*1024**3,free_disk_bytes=100*1024**3,private_bytes=0)
        settings=dict(output=str(root),dataset='ignored',resume=None,device='cpu',epochs=2,
            max_updates=0,seed=123,lr=.0003,threads=1,batch_size=2,pair_budget=100,
            buffer_mib=1,checkpoint_seconds=600,min_free_ram_mib=1024,max_private_mib=8192,
            stop_file=None,early_stopping_patience=0,min_delta=0,prefetch_batches=0,validations_per_epoch=2)
        settings.update((k,v) for k,v in changes.items() if k!='stop_during_validation')
        with patch.dict(sys.modules,{'torch':torch}),patch('builtins.print'),\
                patch.object(trainer,'EventDataset',Dataset),patch.object(trainer,'NeuralPolicy',Policy),\
                patch.object(trainer,'EventLearner',Learner),patch.object(trainer,'feature_identity',return_value='feature'),\
                patch.object(trainer,'snapshot',return_value=resource),patch.object(trainer,'Adam',return_value=None),\
                patch.object(trainer,'PreparationSpec'),\
                patch.object(trainer,'prepared_batches',lambda data,split,*,spec,pair_budget,reader,**s:data.batches(split,**s)):
            return trainer.train(SimpleNamespace(**settings))

    @staticmethod
    def rounds(report):
        return [(h['round'],h['epoch'],h['epoch_fraction'],h['train']['decisions'],h['validation']['mean_nll'],h['cursor'])
                for h in report['history']]

    def test_rounds_split_each_epoch_at_window_boundaries(self):
        with tempfile.TemporaryDirectory() as name:
            report=self.run_training(Path(name))
        self.assertEqual(report['status'],'completed');self.assertEqual(report['optimizer_updates'],8)
        # Two updates per round; round k validates after 2k updates (loss 10-2k).
        self.assertEqual(self.rounds(report),[
            (1,1,.5,2,8,dict(epoch=0,position=[1,1],validations=1)),(2,1,1.0,2,6,dict(epoch=1,position=[0,0],validations=0)),
            (3,2,.5,2,4,dict(epoch=1,position=[1,1],validations=1)),(4,2,1.0,2,2,dict(epoch=2,position=[0,0],validations=0))])

    def test_resume_after_an_update_limit_or_inside_validation_matches_one_run(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);stop=root/'stop'
            full=self.run_training(root/'full')
            for label,changes in (('limit',dict(max_updates=3)),('validation',dict(stop_file=str(stop),
                                                                                       stop_during_validation=(2,stop)))):
                with self.subTest(label):
                    stopped=self.run_training(root/label,**changes)
                    self.assertIn(stopped['status'],('update_limit','interrupted'))
                    if label=='validation':
                        # Stopped before round 1 was recorded: it reruns before any further training.
                        self.assertEqual(stopped['cursor'],dict(epoch=0,position=[1,1],validations=0));stop.unlink()
                    resumed=self.run_training(root/label,resume=stopped['checkpoint'],
                                              **{k:v for k,v in changes.items() if k=='stop_file'})
                    self.assertEqual(self.rounds(resumed),self.rounds(full))
                    self.assertEqual(resumed['optimizer_updates'],full['optimizer_updates'])

    def test_patience_counts_rounds_and_best_is_the_best_round(self):
        with tempfile.TemporaryDirectory() as name:
            # Rounds validate after 2, 4, 6 ... updates; round 2 is best.
            report=self.run_training(Path(name),scores={2:5,4:4,6:4.5,8:4.6},epochs=0,early_stopping_patience=2)
            with np.load(Path(name)/'best.npz') as saved:best=int(saved['step'])
        self.assertEqual(report['status'],'early_stopped')
        self.assertEqual(report['early_stopping']['best_epoch'],2);self.assertEqual(best,4)
        self.assertEqual(len(report['history']),4);self.assertEqual(report['history'][-1]['epoch_fraction'],1.0)

    def test_one_round_per_epoch_needs_no_window_count(self):
        report,_,_=EarlyStoppingTests.run_loop(EarlyStoppingTests(),validations_per_epoch=1)
        self.assertEqual([h['epoch_fraction'] for h in report['history']],[1.0]*5)


class SplitTests(unittest.TestCase):
    def rows(self,n):
        return [dict(match_key=f'pgo:{i}',replay_sha256=f'{i:064x}',archive_sha256=f'{i+100:064x}',split='validation') for i in range(n)]

    def test_exact_5000_1000_split_is_fresh_disjoint_and_deterministic(self):
        rows=self.rows(6500)
        chosen=runner.choose(rows,5000,1000,123)
        self.assertEqual(sum(r['split']=='train' for r in chosen),5000)
        self.assertEqual(sum(r['split']=='validation' for r in chosen),1000)
        self.assertEqual(len({r['replay_sha256'] for r in chosen}),6000)
        self.assertEqual(chosen,runner.choose(list(reversed(rows)),5000,1000,123))
        for row in rows:row['split']='train'
        self.assertEqual(chosen,runner.choose(rows,5000,1000,123))

    def test_duplicates_and_protected_test_aliases_never_enter_both_splits(self):
        rows=self.rows(25);rows.append(dict(rows[3],match_key='alias',archive_sha256='another'))
        chosen=runner.choose(rows,10,10,123,protected_keys={'pgo:1'},protected_shas={rows[2]['replay_sha256']})
        self.assertNotIn('pgo:1',{r['match_key'] for r in chosen})
        self.assertNotIn(rows[2]['replay_sha256'],{r['replay_sha256'] for r in chosen})
        self.assertEqual(len({r['replay_sha256'] for r in chosen}),20)

    def test_insufficient_unique_replays_fails_before_training(self):
        with self.assertRaises(ValueError):runner.choose(self.rows(10),10,1,123)

    def test_validation_is_chosen_first_and_can_be_reused_exactly(self):
        rows=self.rows(40)
        small,large=runner.choose(rows,10,5,123),runner.choose(rows,30,5,123)
        validation={r['match_key'] for r in small if r['split']=='validation'}
        self.assertEqual(validation,{r['match_key'] for r in large if r['split']=='validation'})
        keep=sorted(validation)
        # A different seed would pick another validation set; reuse pins it.
        reused=runner.choose(rows,10,5,7,validation_keys=keep)
        self.assertEqual({r['match_key'] for r in reused if r['split']=='validation'},validation)
        missing=[r for r in rows if r['match_key']!=keep[0]]
        with self.assertRaisesRegex(ValueError,'Reused validation'):runner.choose(missing,10,5,123,validation_keys=keep)


@unittest.skipUnless(os.name=='nt','Windows native runtime and process ownership')
class RuntimeTests(unittest.TestCase):
    def test_owned_script_preserves_venv_and_arguments(self):
        with tempfile.TemporaryDirectory(prefix='training test ') as name:
            root=Path(name);script=root/'capture.py';result=root/'result.json'
            (root/'local_helper.py').write_text('VALUE=42')
            script.write_text('import sys,json,local_helper; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(dict(prefix=sys.prefix,args=sys.argv[2:],local=local_helper.VALUE)))')
            with (root/'child.log').open('w') as log:
                process,family=runner.spawn_owned_python([sys.executable,'-B',script,result,'argument with spaces'],log)
                try:self.assertEqual(process.wait(timeout=20),0)
                finally:family.close()
            data=json.loads(result.read_text())
            self.assertEqual(data['prefix'],sys.prefix)
            self.assertEqual(data['args'],['argument with spaces'])
            self.assertEqual(data['local'],42)

    def test_hard_ai_runtime_uses_the_selected_native_filename(self):
        import cycle_runtime as cycle
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);game=root/'gameplay';runtime=game/'runtime';profile=game/'profile'
            runtime.mkdir(parents=True);profile.mkdir()
            executable=runtime/'chosen.exe';executable.write_bytes(b'local native')
            maps={'[RANK] TD NoBugsCars ZH v1':'ed468fe5a5cb4e2145fa0b37bc5d2a1d4eec58b55fba58395deebe9f17dbd083',
                  'Tournament Desert':'db429b884bc593971feb9d545ada1f1d0aea114f2262b052fe46d81f9f034d01'}
            for map_name in maps:
                directory=profile/'Maps'/map_name;directory.mkdir(parents=True);(directory/(map_name+'.map')).write_bytes(b'map')
            runner.save(root/'plan.json',dict(native=dict(exe=str(executable),sha256='native'),gameplay_work_root=str(game)))
            def digest(p):return 'native' if Path(p).suffix=='.exe' else maps[Path(p).stem]
            with patch.object(cycle,'HERE',root),patch.object(cycle,'digest',side_effect=digest),patch.object(cycle,'preflight'):
                selected,_=cycle.setup_play_runtime()
            self.assertEqual(selected,executable)


if __name__=='__main__':unittest.main()
