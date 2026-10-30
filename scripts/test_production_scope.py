"""China production scope preserves foreign observations/control and causal labels."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from event_ingest import check_vocabulary
from event_policy import default_config
from event_dataset import EventDataset
from event_store import feature_identity, write_shard
from generals_identity import hierarchy_config, vanilla_china_production
from generals_policy import NeuralPolicy, Unrepresentable
from prepared_io import _write_json
from replay_events import decisions, encoded_events
from test_fixtures import SMALL, observation_fixture


class ProductionScopeTests(unittest.TestCase):
    def setUp(self):
        vocabulary, self.obs, self.terrain = observation_fixture()
        vocabulary['types'] += ['GLAArmsDealer','GLATankScorpion','AmericaVehicleDozer',
                                'AmericaPowerPlant','Nuke_ChinaCommandCenter']
        vocabulary['templates'] += ['GLATankScorpion','AmericaPowerPlant']
        vocabulary['menus'].update({'GLAArmsDealer|train':['GLATankScorpion'],
            'AmericaVehicleDozer|build':['AmericaPowerPlant'],
            'Nuke_ChinaCommandCenter|train':['ChinaVehicleDozer']})
        self.config = default_config(vocabulary)
        self.config.update(SMALL)
        self.wide = copy.deepcopy(self.config)
        self.wide.pop('production_scope')
        self.wide['templates'] = vocabulary['templates']
        self.wide['identity_hierarchy'] = hierarchy_config(self.wide['types'],self.wide['templates'])
        for number,kind in ((30,'GLAArmsDealer'),(31,'GLATankScorpion'),
                            (32,'AmericaVehicleDozer'),(33,'AmericaPowerPlant'),
                            (40,'Nuke_ChinaCommandCenter')):
            entity=copy.deepcopy(self.obs['own_entities'][0])
            entity.update(id=number,type=kind,position=[200+number,250,0])
            self.obs['own_entities'].append(entity)
        self.policy = NeuralPolicy(self.config,5)
        self.encoded = self.policy.encoder.encode(self.obs,self.terrain,0)

    def row(self,frame,actions):
        return dict(frame=frame,observation=copy.deepcopy(self.obs),actions=actions,unsupported=[],
            imitation_loss_mask=True,hold_ticks=6,terminal_sample=False)

    def test_default_outputs_are_china_only_but_object_facts_are_complete(self):
        self.assertEqual(self.config['production_scope'],'vanilla_china')
        self.assertEqual(self.config['templates'],['ChinaPowerPlant','ChinaVehicleDozer'])
        self.assertIn('GLATankScorpion',self.config['types'])
        self.assertEqual(self.config['menus'],self.wide['menus'])
        self.assertEqual(vanilla_china_production(self.wide),self.config)
        wide = NeuralPolicy(self.wide,5)
        encoded = wide.encoder.encode(self.obs,self.terrain,0)
        for key in ('types','numeric','extra_numeric','queued_types','scalars','map_inputs','player_identity'):
            np.testing.assert_array_equal(self.encoded[key],encoded[key])
        self.assertNotEqual(feature_identity(self.policy),feature_identity(wide))

    def test_foreign_production_is_blocked_while_control_and_sale_remain_legal(self):
        index = self.encoded['ids']
        self.assertTrue(self.policy.actor_mask(self.encoded,'train')[index[20]])
        self.assertFalse(self.policy.actor_mask(self.encoded,'train')[index[30]])
        self.assertFalse(self.policy.actor_mask(self.encoded,'train')[index[40]])
        self.assertTrue(self.policy.actor_mask(self.encoded,'build')[index[10]])
        self.assertFalse(self.policy.actor_mask(self.encoded,'build')[index[32]])
        for action in (dict(kind='train',producer=30,template='GLATankScorpion'),
                       dict(kind='train',producer=40,template='ChinaVehicleDozer'),
                       dict(kind='build',builder=32,template='AmericaPowerPlant',position=[300,400,0])):
            with self.assertRaises(Unrepresentable): self.policy.target(self.encoded,[action])
        for action in (dict(kind='move',units=[31],position=[400,500,0]),
                       dict(kind='attack',units=[31],target=99),
                       dict(kind='stop',units=[31]),dict(kind='sell',units=[30]),
                       dict(kind='rally',producer=30,position=[400,500,0])):
            self.assertEqual(self.policy.target(self.encoded,[action])['kind'],action['kind'])

    def test_chinese_producer_with_only_foreign_availability_is_not_selected(self):
        obs=copy.deepcopy(self.obs)
        obs['own_entities'][1]['trainable_templates']=['GLATankScorpion']
        encoded=self.policy.encoder.encode(obs,self.terrain,0)
        self.assertFalse(self.policy.actor_mask(encoded,'train').any())

    def test_foreign_production_ends_episode_and_censors_final_delay(self):
        stop=[dict(kind='stop',units=[10])]
        train=[dict(kind='train',producer=20,template='ChinaVehicleDozer')]
        for foreign in (dict(kind='train',producer=30,template='GLATankScorpion'),
                        dict(kind='train',producer=40,template='ChinaVehicleDozer'),
                        dict(kind='build',builder=32,template='AmericaPowerPlant',position=[300,400,0])):
            rows=[self.row(0,stop),self.row(30,train),self.row(60,[foreign]),self.row(90,stop)]
            counts=Counter(); events=list(decisions(rows,self.config,counts))
            self.assertEqual([event['source']['frame'] for event in events],[0,30])
            self.assertEqual(events[0]['delay_ticks'],30)
            self.assertIsNone(events[1]['delay_ticks'])
            self.assertEqual(counts['excluded:production_scope'],1)
            self.assertEqual(counts['truncated:production_scope'],1)
            self.assertEqual(counts['production_scope_cutoff_frame'],60)
            self.assertTrue(all(event['actions'] and event['actions'][0]['kind']!='sell' for event in events))

    def test_cutoff_precedes_multi_action_and_unsupported_filters(self):
        stop=dict(kind='stop',units=[10])
        foreign=dict(kind='train',producer=30,template='GLATankScorpion')
        for actions,unsupported in (([foreign],[]),([stop,foreign],[]),
                                    ([foreign],[dict(name='unsupported')])):
            row=self.row(10,actions); row['unsupported']=unsupported
            counts=Counter()
            events=list(decisions([self.row(0,[stop]),row,self.row(60,[stop])],self.config,counts))
            self.assertEqual([event['source']['frame'] for event in events],[0])
            self.assertIsNone(events[0]['delay_ticks'])
            self.assertEqual(counts['truncated:production_scope'],1)

    def test_first_event_cutoff_does_not_consume_or_encode_the_suffix(self):
        def rows():
            yield self.row(0,[dict(kind='train',producer=30,template='GLATankScorpion')])
            raise AssertionError('Consumed replay beyond its production cutoff')
        counts=Counter()
        self.assertEqual(list(encoded_events(self.policy,dict(rows=rows(),map=self.terrain,
            source_ticks_per_second=60,player=2),counts)),[])
        self.assertEqual(counts['production_scope_cutoff_frame'],0)

    def test_portable_shards_keep_scoped_targets_and_foreign_entities(self):
        rows=[self.row(0,[dict(kind='stop',units=[10])]),
              self.row(30,[dict(kind='train',producer=20,template='ChinaVehicleDozer')]),
              self.row(60,[dict(kind='train',producer=30,template='GLATankScorpion')]),
              self.row(90,[dict(kind='stop',units=[31])])]
        game=dict(rows=rows,map=self.terrain,source_ticks_per_second=60,player=2)
        other=dict(rows=[self.row(0,[dict(kind='stop',units=[31])]),
                         self.row(90,[dict(kind='stop',units=[31])])],
                   map=self.terrain,source_ticks_per_second=60,player=3)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=write_shard(root/'shards/a.zip',[game,other],self.policy,dict(match_key='test:scope'))
            self.assertEqual(result['index']['coverage']['excluded:production_scope:train'],1)
            self.assertEqual(result['index']['coverage']['truncated:production_scope'],1)
            self.assertNotIn('production_scope_cutoff_frame',result['index']['coverage'])
            first,second=result['index']['episodes']
            self.assertEqual(first['censored'],dict(reason='production_scope',frame=60))
            self.assertNotIn('censored',second)
            _write_json(root/'dataset.json',dict(schema=1,kind='generals_event_dataset',config=self.config,
                entries=[dict(result,path='shards/a.zip',match_key='test:scope',split='train')]))
            loaded=[row for batch,_ in EventDataset(root/'dataset.json').batches('train',seed=3,epoch=0) for row in batch]
            self.assertEqual(sorted(row['frame'] for row in loaded if row['player']==2),[0,30])
            self.assertEqual(sorted(row['frame'] for row in loaded if row['player']==3),[0,90])
            teacher=next(row for row in loaded if row['frame']==30)
            self.assertEqual(self.config['templates'][teacher['target']['template']],'ChinaVehicleDozer')
            self.assertNotIn('delay',teacher['target'])
            mask=teacher['encoded']['_training_legality']
            self.assertFalse(mask['actor'][self.encoded['ids'][30]])
            self.assertEqual(len(mask['template']),len(self.config['templates']))
            self.assertIn(self.config['types'].index('GLATankScorpion'),teacher['encoded']['types'])

    def test_vocabulary_check_ignores_only_out_of_scope_production(self):
        check_vocabulary(self.config,self.wide)
        observed=copy.deepcopy(self.wide)
        observed['templates'].append('ChinaMissingUnit')
        with self.assertRaisesRegex(ValueError,'ChinaMissingUnit'): check_vocabulary(self.config,observed)
        observed=copy.deepcopy(self.wide); observed['types'].append('UnknownEnemyUnit')
        with self.assertRaisesRegex(ValueError,'UnknownEnemyUnit'): check_vocabulary(self.config,observed)
        for changes in (dict(production_scope='unsupported'),
                        dict(templates=self.config['templates']+['GLATankScorpion'])):
            with self.assertRaises(ValueError): NeuralPolicy(dict(self.config,**changes))

    def test_cpu_training_uses_only_the_scoped_output_head(self):
        from train_events import EventLearner
        rows=[self.row(0,[dict(kind='stop',units=[10])]),
              self.row(30,[dict(kind='train',producer=20,template='ChinaVehicleDozer')]),
              self.row(60,[dict(kind='train',producer=30,template='GLATankScorpion')]),
              self.row(90,[dict(kind='stop',units=[31])])]
        events=list(encoded_events(self.policy,dict(rows=rows,map=self.terrain,
            source_ticks_per_second=60,player=2),Counter()))
        learner=EventLearner(self.policy,device='cpu',threads=1)
        result=learner.batch(events,train=True)
        self.assertEqual(learner.model.weights['template_b'].numel(),2)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))
        self.assertEqual([event['frame'] for event in events],[0,30])

if __name__ == '__main__': unittest.main()
