"""Accepted-only construction targets, unchanged clocks and explicit snap execution."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from event_policy import EventActor, placement_config
from event_store import feature_identity
from generals_env import placement_options
from generals_policy import NeuralPolicy
from nn_numpy import no_grad
from replay_actions import ReplayActionAdapter
from replay_events import decisions
from test_fixtures import fixture, raw_row
from test_replay_actions import command, sample
from train_events import EventLearner


class FastPlacementTests(unittest.TestCase):
    def test_rejected_request_is_not_a_target_and_preserves_selection(self):
        adapter = ReplayActionAdapter({'7':'ChinaPowerPlant'})
        rejected = command('MSG_DOZER_CONSTRUCT',7,[10,20,0],0)
        rejected['build_result'] = dict(accepted=False,status='rejected')
        decoded = adapter.decode(sample(command('MSG_CREATE_SELECTED_GROUP',True,1),rejected))
        self.assertEqual(decoded['actions'],[])
        self.assertEqual(decoded['unsupported'],[])
        self.assertFalse(decoded['imitation_loss_mask'])
        self.assertEqual(decoded['rejected_build_requests'],1)
        accepted = copy.deepcopy(rejected)
        accepted['build_result'] = dict(accepted=True,status='created',building_id=100)
        mixed = adapter.decode(sample(rejected,accepted))
        self.assertTrue(mixed['imitation_loss_mask'])
        self.assertEqual(len(mixed['actions']),1)
        self.assertEqual(mixed['actions'][0]['builder'],1)
        missing = copy.deepcopy(accepted); del missing['build_result']
        with self.assertRaisesRegex(ValueError,'construction outcome'):
            adapter.decode(sample(missing))

    def test_rejected_request_does_not_reset_or_censor_decision_delay(self):
        config,obs,_ = fixture()
        stop=[dict(kind='stop',units=[10])]
        rejected=raw_row(30,obs,ui=True)
        rejected['rejected_build_requests']=1
        counts=Counter()
        events=list(decisions([raw_row(0,obs,stop),rejected,raw_row(60,obs,stop)],config,counts))
        self.assertEqual([e['source']['frame'] for e in events],[0,60])
        self.assertEqual(events[0]['delay_ticks'],60)
        self.assertEqual(counts['rejected_build_requests'],1)

    def test_fast_training_needs_no_grid_and_torch_matches_reference(self):
        torch.set_num_threads(1)
        config,obs,terrain=fixture()
        config=placement_config(config,'fast')
        policy=NeuralPolicy(config,5)
        encoded=policy.encoder.encode(obs,terrain,0)
        actions=[dict(kind='build',builder=10,template='ChinaPowerPlant',position=[350,400,0],angle=0)]
        target=policy.target(encoded,actions)
        self.assertNotIn('placement_query',encoded)
        self.assertNotIn('build_mask',encoded)
        with no_grad():
            _,_,_,_,reference,_=policy.decide(encoded,None,teacher=target)
        learner=EventLearner(policy,device='cpu')
        with torch.no_grad():
            hidden,entities,spatial=learner.model.events([encoded])
            output=learner.model.teacher_heads([encoded],[target],hidden,entities,spatial)
        self.assertAlmostEqual(float(output['logp'][0]),float(reference.data),places=4)
        row=dict(encoded=encoded,target=target,frame=0,player=2,source_ticks_per_second=60)
        result=learner.batch([row],train=True)
        self.assertTrue(all(np.isfinite(v) for v in result.values() if isinstance(v,(float,int))))
        full=placement_config(config,'full_mask')
        self.assertNotEqual(feature_identity(policy),feature_identity(NeuralPolicy(full,5)))

    def test_execution_settings_are_shared_and_feedback_is_counted(self):
        config,obs,terrain=fixture()
        actor=EventActor(NeuralPolicy(placement_config(config,'fast'),5))
        actor.feedback([dict(accepted=True,placement_adjusted=True,placement_distance=4,
                             executed_action=dict(kind='build',position=[354,400,0]))])
        self.assertEqual((actor.counters['placement_adjusted'],actor.counters['placement_distance_total']),(1,4))
        self.assertEqual(placement_options([actor]),dict(build_snap_radius=32.0,build_snap_step=4.0))
        other=EventActor(NeuralPolicy(placement_config(config,'full_mask'),5))
        self.assertEqual(placement_options([other]),{})
        with self.assertRaisesRegex(ValueError,'different placement execution'):
            placement_options([actor,other])


if __name__ == '__main__': unittest.main()
