"""Placement prefix identity, exact masks, rotated coordinates and train/play agreement."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from build_placement import decode_grid, mask_for, categorical_masks
from event_dataset import EventDataset
from event_store import write_shard
from generals_policy import NeuralPolicy, Unrepresentable
from nn_numpy import no_grad
from test_fixtures import fixture, grid, raw_row
from train_events import EventLearner


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture('full_mask')
        self.policy = NeuralPolicy(self.config, 5)
        self.mask = np.zeros((16, 16), bool)
        self.mask[4:7, 4:9] = True
        self.mask[10:13, 10:13] = True

    def example(self, *, angle=3, position=(350, 350, 0), rotate=False, frame=0):
        obs = copy.deepcopy(self.obs)
        obs['build_placements'] = [grid(self.mask, angle=angle, frame=frame)]
        encoded = self.policy.encoder.encode(obs, self.terrain, frame,
            home_position=[900, 900] if rotate else [100, 100])
        target = self.policy.target(encoded, [{'kind': 'build', 'builder': 10,
            'template': 'ChinaPowerPlant', 'position': list(position), 'angle': angle*np.pi/8}])
        target['delay'] = 5
        return {'encoded': encoded, 'target': target}

    def test_grid_rejects_stale_wrong_prefix_and_corruption(self):
        record = grid(self.mask)
        expected = dict(bins=16, builder=10, template='ChinaPowerPlant', angle=0, frame=0)
        np.testing.assert_array_equal(decode_grid(record, **expected), self.mask)
        for field, value in (('frame', 1), ('builder', 11), ('angle_index', 2),
                             ('bins', 32), ('mask_hex', '00'), ('legal_sites', 999)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                decode_grid({**record, field: value}, **expected)

    def test_rotated_and_unrotated_teacher_decode_the_same_world_site(self):
        for rotate in (False, True):
            row = self.example(rotate=rotate)
            with no_grad():
                actions, trace, *_ = self.policy.decide(row['encoded'], None, teacher=row['target'])
            self.assertEqual(actions[0]['position'][:2], [343.75, 343.75])
            self.assertAlmostEqual(actions[0]['angle'], 3*np.pi/8)
            mask = row['encoded']['build_mask']
            cell, offset = categorical_masks(mask, self.config['map_side'], trace['cell'])
            self.assertTrue(cell[trace['cell']]); self.assertTrue(offset[trace['offset']])

    def test_masked_quantized_teacher_is_excluded_instead_of_relocated(self):
        with self.assertRaisesRegex(Unrepresentable, 'Quantized build placement'):
            self.example(position=(100, 100, 0))

    def test_exact_grid_boundary_decodes_identically_from_both_spawn_orientations(self):
        self.mask[8, 8] = True
        for rotate in (False, True):
            row = self.example(position=(500, 500, 0), rotate=rotate)
            with no_grad():
                actions, trace, *_ = self.policy.decide(row['encoded'], None, teacher=row['target'])
            self.assertEqual(actions[0]['position'][:2], [531.25, 531.25])

    def test_live_sampling_uses_only_exact_legal_sites_and_queries_once(self):
        encoded = self.policy.encoder.encode(self.obs, self.terrain, 0)
        query = Mock(side_effect=lambda builder, name, angle, bins: grid(self.mask, angle=angle))
        encoded['placement_query'] = query
        self.policy.parameters['verb_b'].data[:] = -100
        self.policy.parameters['verb_b'].data[self.policy.kinds.index('build')] = 100
        with no_grad():
            actions, trace, *_ = self.policy.decide(encoded, None, deterministic=True)
        query.assert_called_once()
        self.assertTrue(self.mask[trace['y'], trace['x']])
        expected = (np.asarray([trace['x'], trace['y']])+.5)*1000/16
        np.testing.assert_array_equal(actions[0]['position'][:2], expected)

    def test_empty_preview_defers_without_selecting_a_fallback_site(self):
        encoded = self.policy.encoder.encode(self.obs, self.terrain, 0)
        encoded['placement_query'] = lambda builder, name, angle, bins: grid(np.zeros((16, 16), bool), angle=angle)
        self.policy.parameters['verb_b'].data[:] = -100
        self.policy.parameters['verb_b'].data[self.policy.kinds.index('build')] = 100
        with no_grad():
            actions, trace, *_ = self.policy.decide(encoded, None, deterministic=True)
        self.assertEqual(actions, [])
        self.assertTrue(trace['placement_unavailable'])
        self.assertEqual(trace['attempted_kind'], 'build')
        self.assertNotIn('cell', trace)

    def test_preview_diagnostics_never_enter_model_inputs(self):
        a = self.example()
        obs = copy.deepcopy(self.obs)
        record = grid(self.mask, angle=3)
        record.update(seconds=999, round_trip_audit={'original': 999}, path_check='changed diagnostic')
        obs['build_placements'] = [record]
        encoded = self.policy.encoder.encode(obs, self.terrain, 0)
        mask_for(self.config, encoded, a['target']['selection'][0], a['target']['template'], 3)
        for field in ('map_visibility', 'scalars', 'numeric', 'build_mask', 'build_features'):
            np.testing.assert_array_equal(a['encoded'][field], encoded[field])

    def test_teacher_placement_information_cannot_change_earlier_heads(self):
        a = self.example(angle=3)
        self.mask[:] = True
        b = self.example(angle=7)
        with no_grad():
            left = self.policy._trunk(a['encoded'])
            right = self.policy._trunk(b['encoded'])
        # Conditional legality is used only after choosing builder/template/angle.
        self.assertFalse(np.array_equal(a['encoded']['build_features'], b['encoded']['build_features']))
        for first, second in zip(left, right):
            np.testing.assert_array_equal(first.data, second.data)

    def test_placement_frame_remains_native_for_a_30_hz_source(self):
        obs = copy.deepcopy(self.obs)
        obs['build_placements'] = [grid(self.mask, frame=3)]
        encoded = self.policy.encoder.encode(obs, self.terrain, 6, source_ticks_per_second=30)
        template = self.config['templates'].index('ChinaPowerPlant')
        np.testing.assert_array_equal(mask_for(self.config, encoded, encoded['ids'][10], template, 0), self.mask)

    def test_numpy_torch_and_portable_columns_agree_on_every_head(self):
        rows = [self.example(), self.example(rotate=True)]
        for row in rows:
            row['hold_ticks'] = 6
            row['scope'] = 'event'
        from training_legality import build_legality
        from event_store import feature_identity
        from tensor_replay_store import project, encode_columns, decode_columns, TensorRows
        key = feature_identity(self.policy)
        for row in rows:
            row['encoded']['_training_legality'] = build_legality(self.policy, row['encoded'], row['target'], key)
        columns = project(rows, {}, self.policy, key)
        arrays, metadata = decode_columns(encode_columns(columns.columns, columns.metadata))
        portable = TensorRows(arrays, metadata, {}, key)
        learner = EventLearner(self.policy, device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
        learner.model.legality_key = key
        for batch in (rows, list(portable)):
            encoded = [r['encoded'] for r in batch]
            h, entities, spatial = learner.model.events(encoded)
            output = learner.model.teacher_heads(encoded, [r['target'] for r in batch], h, entities, spatial)
            with no_grad():
                expected = [float(self.policy.decide(r['encoded'], None, teacher=r['target'])[4].data) for r in rows]
            np.testing.assert_allclose(output['logp'].detach().cpu().numpy(), expected, rtol=3e-6, atol=2e-5)
        learner.batch(list(portable), train=True)
        self.assertGreater(float(learner.model.weights['build_map_w'].grad.abs().sum().cpu()), 0)
        self.assertGreater(float(learner.model.weights['build_angle_embedding'].grad.abs().sum().cpu()), 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.npz'
            learner.save(path, {})
            restored, optimizer, document = NeuralPolicy.load(path)
            other = EventLearner(restored, optimizer, device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
            other.restore_rng(document['metadata'])
            learner.batch(list(portable), train=True); other.batch(list(portable), train=True)
            for name in learner.model.weights:
                np.testing.assert_array_equal(learner.model.weights[name].detach().cpu().numpy(),
                                              other.model.weights[name].detach().cpu().numpy())


if __name__ == '__main__':
    unittest.main()
