"""Learned builder/foundation pointers, native legality and training parity."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from event_store import feature_identity
from generals_policy import NeuralPolicy, Unrepresentable
from nn_numpy import no_grad
from replay_actions import ReplayActionAdapter
from replay_training import audit
from tensor_replay_store import project, encode_columns, decode_columns, TensorRows
from test_fixtures import entity, fixture, task
from test_replay_actions import command, sample
from test_replay_training import fixture as audit_fixture
from train_events import EventLearner
from training_legality import build_legality
from unit_tasks import COMMAND_FEATURES, COMMANDS


def fixture_resume():
    config, obs, terrain = fixture('full_mask')
    obs['own_entities'][0]['resume_targets'] = [30]
    obs['own_entities'].append(entity(30, 'ChinaPowerPlant', True, [200, 200, 0], last_seen_frame=6,
        max_health=400, construction_percent=25, builder_id=10, task=task('construct'),
        has_weapon=False, can_attack=False, has_atk_move_cmd=False, can_attack_move=False))
    obs['last_submission'] = {'available': True, 'kind': 'resume_construction', 'actors': [10],
        'target': {'status': 'current', 'id': 30}, 'result': 'submitted', 'age_seconds': .1}
    return config, obs, terrain


class ResumeConstructionTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture_resume()
        self.policy = NeuralPolicy(self.config, 5)

    def encode(self, obs=None):
        return self.policy.encoder.encode(self.obs if obs is None else obs, self.terrain, 6)

    def test_exact_builder_target_pairs_ignore_money_and_refuse_foreign_or_completed_targets(self):
        obs = copy.deepcopy(self.obs)
        obs['player']['cash'] = 0
        # No affordable build templates are needed: this foundation was paid for.
        obs['own_entities'][0]['buildable_templates'] = []
        obs['own_entities'][0]['resume_targets'] += [99, 20]
        encoded = self.encode(obs)
        pairs = self.policy.resume_matrix(encoded)
        self.assertEqual(np.argwhere(pairs).tolist(), [[encoded['ids'][10], encoded['ids'][30]]])
        target = self.policy.target(encoded, [{'kind': 'resume_construction', 'builder': 10, 'target': 30}])
        self.assertEqual(target['selection'], [encoded['ids'][10]])
        for action in ({'kind': 'resume_construction', 'builder': 20, 'target': 30},
                       {'kind': 'resume_construction', 'builder': 10, 'target': 99}):
            with self.assertRaises(Unrepresentable):
                self.policy.target(encoded, [action])

    def test_native_busy_or_unavailable_foundations_disable_resume_without_automatic_recovery(self):
        obs = copy.deepcopy(self.obs)
        obs['own_entities'][0]['resume_targets'] = []
        encoded = self.encode(obs)
        self.assertFalse(self.policy.verb_mask(encoded)[self.policy.kinds.index('resume_construction')])
        with self.assertRaises(Unrepresentable):
            self.policy.target(encoded, [{'kind': 'resume_construction', 'builder': 10, 'target': 30}])
        with no_grad():
            actions, *_ = self.policy.decide(self.encode(), None, teacher={'kind': 'wait', 'delay': 0})
        self.assertEqual(actions, [])

    def test_teacher_round_trip_chooses_one_builder_and_exact_existing_foundation(self):
        encoded = self.encode()
        action = {'kind': 'resume_construction', 'builder': 10, 'target': 30}
        teacher = {**self.policy.target(encoded, [action]), 'delay': 0}
        with no_grad():
            decoded, trace, *_ = self.policy.decide(encoded, None, teacher=teacher)
        self.assertEqual(decoded, [action])
        self.assertEqual(trace['kind'], 'resume_construction')

    def test_replay_preserves_unique_effective_builder_and_excludes_ambiguous_or_unavailable_orders(self):
        for candidates in ([1], [1, 2], []):
            row = sample(command('MSG_CREATE_SELECTED_GROUP', True, 1, 2), command('MSG_RESUME_CONSTRUCTION', 30))
            row['observation']['own_entities'].append({'id': 30})
            for unit in row['observation']['own_entities']:
                unit['resume_targets'] = [30] if unit['id'] in candidates else []
            result = ReplayActionAdapter({}).decode(row)
            self.assertEqual(result['imitation_loss_mask'], len(candidates) == 1)
            self.assertEqual(result['actions'], [{'kind': 'resume_construction', 'builder': 1, 'target': 30}]
                             if len(candidates) == 1 else [])

    def test_command_context_includes_resume_construction(self):
        encoded = self.encode()
        self.assertEqual(encoded['task_commands'].shape, (2, COMMAND_FEATURES))
        self.assertEqual(encoded['task_commands'][0, 1+COMMANDS.index('resume_construction')], 1)

    def test_raw_audit_requires_owned_legality_and_rejects_enemy_legality(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'replay.jsonl'
            rows = audit_fixture()
            for row in rows[2:4]:
                row['observation']['own_entities'] = [{'id': 1, 'owner': 3, 'own': True, 'visible': True,
                    'last_seen_frame': row['frame'], 'resume_targets': []}]
            def save():
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save()
            self.assertEqual(audit(path, level='full')['samples'], 2)
            rows[2]['observation']['known_nonowned_entities'][0]['resume_targets'] = []
            save()
            with self.assertRaisesRegex(ValueError, 'Private nonowned state'):
                audit(path, level='full')
            del rows[2]['observation']['known_nonowned_entities'][0]['resume_targets']
            del rows[2]['observation']['own_entities'][0]['resume_targets']
            save()
            with self.assertRaisesRegex(ValueError, 'Missing owned resume'):
                audit(path, level='full')

    def test_resume_target_head_portable_data_gradients_and_exact_training_continuation(self):
        second = copy.deepcopy(self.obs['own_entities'][-1])
        second.update(id=31, position=[280, 260, 0])
        self.obs['own_entities'].append(second)
        self.obs['own_entities'][0]['resume_targets'].append(31)
        rows = []
        for action in ([], [{'kind': 'resume_construction', 'builder': 10, 'target': 30}],
                       [{'kind': 'attack', 'units': [10], 'target': 99}]):
            encoded = self.encode()
            target = {**self.policy.target(encoded, action), 'delay': 5}
            rows.append({'encoded': encoded, 'target': target, 'hold_ticks': 6, 'scope': 'event'})
        key = feature_identity(self.policy)
        for row in rows:
            row['encoded']['_training_legality'] = build_legality(self.policy, row['encoded'], row['target'], key)
        packed = project(rows, {}, self.policy, key)
        arrays, metadata = decode_columns(encode_columns(packed.columns, packed.metadata))
        portable = list(TensorRows(arrays, metadata, {}, key))
        learner = EventLearner(self.policy, device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
        learner.model.legality_key = key
        for batch in (rows, portable):
            encoded = [row['encoded'] for row in batch]
            h, entities, spatial = learner.model.events(encoded)
            actual = learner.model.teacher_heads(encoded, [r['target'] for r in batch], h, entities, spatial)
            with no_grad():
                expected = [float(self.policy.decide(r['encoded'], None, teacher=r['target'])[4].data) for r in rows]
            np.testing.assert_allclose(actual['logp'].detach().cpu().numpy(), expected, rtol=4e-6, atol=3e-5)
        learner.batch(portable, train=True)
        for name in ('target_pointer_w', 'task_command_w'):
            self.assertGreater(float(learner.model.weights[name].grad.abs().sum().cpu()), 0)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)/'checkpoint.npz'
            learner.save(checkpoint, {})
            policy, optimizer, document = NeuralPolicy.load(checkpoint)
            other = EventLearner(policy, optimizer, device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
            other.restore_rng(document['metadata'])
            learner.batch(portable, train=True); other.batch(portable, train=True)
            for name in learner.model.weights:
                np.testing.assert_array_equal(learner.model.weights[name].detach().cpu().numpy(),
                    other.model.weights[name].detach().cpu().numpy())


if __name__ == '__main__':
    unittest.main()
