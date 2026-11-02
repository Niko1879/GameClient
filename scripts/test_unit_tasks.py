"""Individual task identity, causal context and CPU/Torch/portable agreement."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from collections import Counter

import numpy as np

from event_store import feature_identity
from generals_policy import NeuralPolicy
from nn_numpy import no_grad
from replay_events import encoded_events
from replay_sequences import normalize_demonstration
from tensor_replay_store import project, encode_columns, decode_columns, TensorRows
from test_fixtures import fixture, grid, raw_row, task
from train_events import EventLearner
from training_legality import build_legality
from unit_tasks import TASKS, RELATIONS


def fixture_tasks():
    """Unit tasks with exact native placement grids."""
    return fixture('full_mask')


class UnitTaskTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture_tasks()
        self.policy = NeuralPolicy(self.config, 5)

    def encode(self, obs=None, **kwargs):
        return self.policy.encoder.encode(self.obs if obs is None else obs, self.terrain, 6,
                                          home_position=kwargs.pop('home_position', [100, 100]), **kwargs)

    def test_policy_requires_observation_version_and_owned_task_data(self):
        for remove in ('observation_policy_version', 'task'):
            obs = copy.deepcopy(self.obs)
            if remove == 'task':
                del obs['own_entities'][0]['task']
            else:
                del obs[remove]
            with self.assertRaises(ValueError):
                self.encode(obs)
        obs = copy.deepcopy(self.obs)
        obs['own_entities'][0]['task']['kind'] = 'raw-engine-state-42'
        with self.assertRaises(ValueError):
            self.encode(obs)

    def test_idle_move_attack_build_and_collection_are_distinct(self):
        values = []
        for kind in ('idle', 'move', 'attack', 'construct', 'collect', 'return_supplies'):
            obs = copy.deepcopy(self.obs)
            obs['own_entities'][0]['task'] = task(kind)
            encoded = self.encode(obs)
            vector = encoded['task_numeric'][encoded['ids'][10]]
            self.assertEqual(vector[TASKS.index(kind)], 1)
            values.append(vector.tobytes())
        self.assertEqual(len(set(values)), 6)

    def test_relationships_resolve_exact_observation_rows(self):
        obs = copy.deepcopy(self.obs)
        obs['own_entities'][0]['task'] = task('attack', target={'status': 'current', 'id': 99},
            combat_target={'status': 'current', 'id': 99})
        encoded = self.encode(obs)
        for actor, role, target in ((10, 'target', 99), (10, 'combat_target', 99), (20, 'builder', 10)):
            row, column = encoded['ids'][actor], RELATIONS.index(role)
            self.assertTrue(encoded['task_relation_valid'][row, column])
            self.assertEqual(encoded['task_relations'][row, column], encoded['ids'][target])

    def test_id_renaming_and_input_order_do_not_change_neural_features(self):
        original = self.encode()
        obs = copy.deepcopy(self.obs)
        rename = {10: 440, 20: 7, 99: 10101, 100: 22}
        for key in ('own_entities', 'known_nonowned_entities'):
            for entity in obs[key]:
                entity['id'] = rename[entity['id']]
                for field in ('builder_id', 'producer_id'):
                    if field in entity:
                        entity[field] = rename[entity[field]]
            obs[key].reverse()
        obs['last_submission']['actors'] = [rename[i] for i in obs['last_submission']['actors']]
        changed = self.encode(obs)
        for name in ('task_numeric', 'task_relations', 'task_relation_valid', 'task_relation_features', 'task_commands'):
            np.testing.assert_array_equal(original[name], changed[name])
        with no_grad():
            for a, b in zip(self.policy._trunk(original), self.policy._trunk(changed)):
                np.testing.assert_array_equal(a.data, b.data)

    def test_hidden_target_does_not_resolve_or_import_nonowned_orders(self):
        obs = copy.deepcopy(self.obs)
        obs['own_entities'][0]['task'] = task('attack', target={'status': 'current', 'id': 100})
        first = self.encode(obs)
        for entity in obs['known_nonowned_entities']:
            entity['task'] = task('construct', destination=[999, 999, 0])
        second = self.encode(obs)
        i, role = first['ids'][10], RELATIONS.index('target')
        self.assertFalse(first['task_relation_valid'][i, role])
        np.testing.assert_array_equal(first['task_relation_features'][i, role, :3], [0, 0, 1])
        np.testing.assert_array_equal(first['task_numeric'], second['task_numeric'])
        np.testing.assert_array_equal(first['task_relation_features'], second['task_relation_features'])

    def test_destinations_and_relationships_rotate_with_spawn(self):
        original = self.encode()
        obs = copy.deepcopy(self.obs)
        for key in ('own_entities', 'known_nonowned_entities'):
            for entity in obs[key]:
                entity['position'][:2] = (1000-np.asarray(entity['position'][:2])).tolist()
                entity['facing'] = np.pi
                if entity.get('task', {}).get('destination') is not None:
                    entity['task']['destination'][:2] = (1000-np.asarray(entity['task']['destination'][:2])).tolist()
        obs['last_submission']['position'][:2] = (1000-np.asarray(obs['last_submission']['position'][:2])).tolist()
        rotated = self.encode(obs, home_position=[900, 900])
        for name in ('numeric', 'task_numeric', 'task_relation_features', 'task_commands'):
            np.testing.assert_allclose(original[name], rotated[name], atol=1e-15)

    def test_unseen_validation_receipts_start_neutral_and_preserve_actual_task(self):
        before = self.encode()
        obs = copy.deepcopy(self.obs)
        obs['last_validation'] = {'available': True, 'kind': 'build', 'actors': [10],
            'target': {'status': 'current', 'id': 99}, 'position': [900, 600, 0],
            'template': 'ChinaPowerPlant', 'result': 'rejected', 'reason': 'placement', 'age_seconds': .02}
        after = self.encode(obs)
        self.assertEqual(after['task_numeric'][after['ids'][10], TASKS.index('move')], 1)
        with no_grad():
            for a, b in zip(self.policy._trunk(before), self.policy._trunk(after)):
                np.testing.assert_array_equal(a.data, b.data)

    def test_future_teacher_command_does_not_change_earlier_inputs(self):
        def convert(kind):
            rows = [raw_row(0, self.obs), raw_row(60, self.obs,
                [{'kind': kind, 'units': [10], **({'target': 99} if kind == 'attack' else {})}])]
            return list(encoded_events(self.policy, {'rows': rows, 'map': self.terrain,
                'player': 2, 'source_ticks_per_second': 60}, Counter()))[0]
        left, right = convert('stop'), convert('attack')
        for name in ('task_numeric', 'task_relations', 'task_commands'):
            np.testing.assert_array_equal(left['encoded'][name], right['encoded'][name])

    def test_task_ages_are_seconds_at_both_source_rates(self):
        for rate, frame in ((30, 3), (60, 6)):
            row = normalize_demonstration(raw_row(frame, copy.deepcopy(self.obs)), rate)
            encoded = self.encode(row['observation'], source_ticks_per_second=rate)
            np.testing.assert_array_equal(encoded['task_commands'], self.encode()['task_commands'])

    def test_replay_pretraining_keeps_unseen_receipt_inputs_neutral(self):
        before = self.encode()
        learner = EventLearner(self.policy)
        rows = [{'encoded': before, 'target': {**self.policy.target(before, [{'kind': 'stop', 'units': [10]}]), 'delay': 5}}]
        learner.batch(rows, train=True)
        obs = copy.deepcopy(self.obs)
        obs['last_validation'] = {'available': True, 'kind': 'move', 'actors': [10],
            'position': [50, 50, 0], 'result': 'rejected', 'reason': 'placement', 'age_seconds': 0}
        after = self.encode(obs)
        first = learner.model.events([before])
        second = learner.model.events([after])
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.detach().cpu().numpy(), b.detach().cpu().numpy())

    def test_invalid_cargo_or_destination_is_rejected(self):
        for changes in ({'cargo': {'available': True, 'boxes': 9, 'capacity': 1}},
                        {'destination': [float('nan'), 0, 0]}):
            obs = copy.deepcopy(self.obs)
            obs['own_entities'][0]['task'].update(changes)
            with self.assertRaises(ValueError):
                self.encode(obs)

    def test_engine_debug_state_never_enters_inputs(self):
        obs = copy.deepcopy(self.obs)
        obs['last_submission'].update(kind='other', age_seconds=.25)
        left = self.encode(obs)
        obs['own_entities'][0]['ai_state'] = 'unrelated-engine-debug-state'
        right = self.encode(obs)
        for name in ('scalars', 'numeric', 'extra_numeric', 'task_numeric', 'task_commands'):
            np.testing.assert_array_equal(left[name], right[name])

    def test_empty_owned_and_target_tables_keep_defeat_inputs_defined(self):
        obs = copy.deepcopy(self.obs)
        obs['own_entities'], obs['known_nonowned_entities'] = [], []
        obs['player']['dead'] = True
        encoded = self.encode(obs)
        self.assertEqual(len(encoded['task_numeric']), 1)
        self.assertFalse(encoded['task_relation_valid'].any())
        with no_grad():
            actions, trace, *_ = self.policy.decide(encoded, None, deterministic=True)
        self.assertEqual(actions, [])
        self.assertEqual(trace['kind'], 'wait')

    def test_every_head_portable_columns_gradients_and_resume(self):
        obs = copy.deepcopy(self.obs)
        obs['last_validation'] = {'available': True, 'kind': 'attack', 'actors': [10],
            'target': {'status': 'current', 'id': 99}, 'result': 'rejected',
            'reason': 'unavailable_target', 'age_seconds': .05}
        obs['build_placements'] = [grid(np.ones((16, 16), bool), angle=0, frame=6)]
        actions = [[], [{'kind': 'stop', 'units': [10]}],
            [{'kind': 'move', 'units': [10], 'position': [300, 400, 0]}],
            [{'kind': 'attack', 'units': [10], 'target': 99}],
            [{'kind': 'train', 'producer': 20, 'template': 'ChinaVehicleDozer'}],
            [{'kind': 'build', 'builder': 10, 'template': 'ChinaPowerPlant', 'position': [300, 400, 0], 'angle': 0}]]
        rows = []
        for action in actions:
            encoded = self.encode(obs)
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
            output = learner.model.teacher_heads(encoded, [row['target'] for row in batch], h, entities, spatial)
            with no_grad():
                expected = [float(self.policy.decide(row['encoded'], None, teacher=row['target'])[4].data) for row in rows]
            np.testing.assert_allclose(output['logp'].detach().cpu().numpy(), expected, rtol=4e-6, atol=3e-5)
        learner.batch(portable, train=True)
        for name in ('task_w', 'task_builder_w', 'task_command_w', 'task_validation_w', 'task_validation_target_w'):
            self.assertGreater(float(learner.model.weights[name].grad.abs().sum().cpu()), 0, name)
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

    def test_compact_owned_task_rows_match_reference_without_owned_entities(self):
        # Torch stages task inputs for owned rows only. A defeated player has
        # none; alone and padded beside a live player it must match the reference.
        dead = copy.deepcopy(self.obs)
        dead['own_entities'] = []
        dead['player']['dead'] = True
        key = feature_identity(self.policy)
        rows = []
        for obs in (dead, self.obs):
            encoded = self.encode(obs)
            target = {**self.policy.target(encoded, []), 'delay': 5}
            encoded['_training_legality'] = build_legality(self.policy, encoded, target, key)
            rows.append({'encoded': encoded, 'target': target, 'hold_ticks': 6, 'scope': 'event'})
        self.assertFalse(rows[0]['encoded']['own'].any())
        learner = EventLearner(self.policy, device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
        learner.model.legality_key = key
        for batch in (rows[:1], rows):
            encoded = [row['encoded'] for row in batch]
            output = learner.model.teacher_heads(encoded, [row['target'] for row in batch],
                                                 *learner.model.events(encoded))
            with no_grad():
                expected = [float(self.policy.decide(row['encoded'], None, teacher=row['target'])[4].data) for row in batch]
            np.testing.assert_allclose(output['logp'].detach().cpu().numpy(), expected, rtol=4e-6, atol=3e-5)
        learner.batch(rows[:1], train=True)
        gradient = learner.model.weights['task_builder_w'].grad
        self.assertIsNotNone(gradient)
        self.assertEqual(float(gradient.abs().sum().cpu()), 0)


if __name__ == '__main__':
    unittest.main()
