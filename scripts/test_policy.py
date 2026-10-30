"""Event policy: autodiff primitives, encoder invariants, masks, sampling and Torch/NumPy agreement."""
import copy
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from event_policy import EventActor
from generals_policy import KINDS, NeuralPolicy
from nn_numpy import Tensor, categorical, conv2d_same, scatter_mean, no_grad
from tensor_replay_store import Column, decode_columns, encode_columns
from test_fixtures import entity, fixture, grid
from torch_policy_v3 import TorchPolicyV3

ACTIONS = [[],
    [{'kind': 'move', 'units': [10], 'position': [200, 300, 0]}],
    [{'kind': 'move', 'units': [10, 30, 40], 'position': [300, 400, 0]}],
    [{'kind': 'attack_move', 'units': [10, 30], 'position': [350, 450, 0]}],
    [{'kind': 'attack', 'units': [10], 'target': 99}],
    [{'kind': 'stop', 'units': [10]}],
    [{'kind': 'sell', 'units': [20]}],
    [{'kind': 'build', 'builder': 10, 'template': 'ChinaPowerPlant', 'position': [230, 350, 0], 'angle': .5}],
    [{'kind': 'train', 'producer': 20, 'template': 'ChinaVehicleDozer'}],
    [{'kind': 'rally', 'producer': 20, 'position': [200, 300, 0]}]]


def finite_difference(test, loss, parameters, places):
    loss().backward()
    for parameter in parameters:
        analytic = parameter.grad.copy()
        for index in np.ndindex(parameter.data.shape):
            old = parameter.data[index]
            parameter.data[index] = old+1e-6; high = float(loss().data)
            parameter.data[index] = old-1e-6; low = float(loss().data)
            parameter.data[index] = old
            test.assertAlmostEqual(analytic[index], (high-low)/2e-6, places=places)


def examples(policy, observation, terrain):
    """One decision per supported kind, with extra mobile units for group orders."""
    extra = copy.deepcopy(observation)
    for identifier in (30, 40):
        unit = copy.deepcopy(observation['own_entities'][0]); unit['id'] = identifier
        unit['position'][0] += identifier
        extra['own_entities'].append(unit)
    empty = copy.deepcopy(observation); empty['own_entities'] = empty['known_nonowned_entities'] = []
    rows = []
    for i, action in enumerate(ACTIONS + [[]]):
        source = empty if i == len(ACTIONS) else (extra if i in (2, 3) else observation)
        encoded = policy.encoder.encode(source, terrain, 6)
        target = policy.target(encoded, action)
        if i % 2:
            target['delay'] = 2
        rows.append({'encoded': encoded, 'target': target})
    return rows


class PrimitiveTests(unittest.TestCase):
    def test_primitives_match_finite_differences_with_broadcast_and_repeated_embeddings(self):
        a, b = Tensor([[.2, -.4], [.6, .1]], requires_grad=True), Tensor([.1, .3], requires_grad=True)
        def loss():
            y = (a @ a + b).tanh()
            logits = y[0] + y[[1, 1]].mean(axis=0)
            _, logp, ent = categorical(logits, [True, True], choice=1)
            return -logp - ent * .1
        finite_difference(self, loss, (a, b), 6)

    def test_spatial_primitives_have_correct_derivatives(self):
        value = Tensor(np.arange(12).reshape(4, 3)/10, requires_grad=True)
        kernel = Tensor(np.linspace(-.3, .4, 54).reshape(27, 2), requires_grad=True)
        def loss():
            x = conv2d_same(value, kernel, 2).layer_norm().sigmoid()
            pooled = scatter_mean(x, [0, 0, 1, 2], 3)
            return (pooled.softmax(axis=1)*pooled).sum()
        finite_difference(self, loss, (value, kernel), 5)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture()
        self.policy = NeuralPolicy(self.config, 5)
        self.encoded = self.policy.encoder.encode(self.obs, self.terrain, 6)

    def test_entity_grid_cells_preserve_boundaries(self):
        policy = NeuralPolicy({**self.config, 'map_side': 16})
        observation = copy.deepcopy(self.obs)
        prototype = observation['own_entities'][0]
        positions = [-1., 0., np.nextafter(62.5, 0), 62.5, np.nextafter(62.5, 1000), 999.99, 1000., 1001.]
        observation['own_entities'] = [{**copy.deepcopy(prototype), 'id': 100+i, 'position': [float(x), 0., 0.]}
                                       for i, x in enumerate(positions)]
        observation['known_nonowned_entities'] = []
        unchanged = copy.deepcopy(observation)
        encoded = policy.encoder.encode(observation, self.terrain, 6)
        self.assertFalse(encoded['rotated'])
        for i, cell in enumerate((0, 0, 0, 1, 1, 15, 15, 15)):
            self.assertEqual(encoded['cells'][encoded['ids'][100+i]], cell)
        self.assertEqual(observation, unchanged)

    def test_empty_entity_grid_preserves_zero_sentinel(self):
        observation = copy.deepcopy(self.obs)
        observation['own_entities'] = observation['known_nonowned_entities'] = []
        encoded = self.policy.encoder.encode(observation, self.terrain, 6)
        np.testing.assert_array_equal(encoded['numeric'], np.zeros((1, 14)))
        np.testing.assert_array_equal(encoded['cells'], [0])
        self.assertFalse(encoded['map_inputs'][:, 5:9].any())
        self.assertEqual(self.policy.decide(encoded, teacher={'kind': 'wait'})[1]['kind'], 'wait')

    def test_entity_ids_and_order_do_not_enter_network(self):
        other = copy.deepcopy(self.obs)
        for e in other['own_entities']+other['known_nonowned_entities']:
            e['id'] += 1000
        for e in other['own_entities']:
            e['task']['destination'] = e['task']['destination']
        other['last_submission']['actors'] = [1010]
        other['own_entities'][1]['builder_id'] = 1010
        other['own_entities'].reverse(); other['known_nonowned_entities'].reverse()
        encoded = self.policy.encoder.encode(other, self.terrain, 6)
        a = self.policy.decide(self.encoded, teacher={'kind': 'wait'})
        b = self.policy.decide(encoded, teacher={'kind': 'wait'})
        np.testing.assert_array_equal(a[2].data, b[2].data)
        self.assertEqual(float(a[4].data), float(b[4].data))

    def test_opposite_spawns_share_canonical_features_and_commands_rotate_back(self):
        other = copy.deepcopy(self.obs)
        for e in other['own_entities']+other['known_nonowned_entities']:
            e['position'][:2] = [1000-x for x in e['position'][:2]]
        other['own_entities'][0]['task']['destination'] = [700, 600, 0]
        other['last_submission']['position'] = [700, 600, 0]
        original = self.policy.target(self.encoded, ACTIONS[7])
        rotated = self.policy.encoder.encode(other, self.terrain, 6)
        self.assertTrue(rotated['rotated'] and not self.encoded['rotated'])
        transformed = self.policy.target(rotated, [{**ACTIONS[7][0], 'position': [770, 650, 0], 'angle': .5+np.pi}])
        self.assertEqual(original, transformed)
        for name in ('numeric', 'map_inputs', 'scalars', 'task_numeric', 'task_relation_features'):
            np.testing.assert_allclose(self.encoded[name], rotated[name], atol=1e-12, err_msg=name)
        a = self.policy.decide(self.encoded, teacher=original)[0][0]
        b = self.policy.decide(rotated, teacher=transformed)[0][0]
        np.testing.assert_allclose(np.asarray(a['position'][:2])+b['position'][:2], [1000, 1000])
        self.assertAlmostEqual((b['angle']-a['angle'])%(2*np.pi), np.pi)

    def test_home_orientation_is_pinned_after_initial_observation(self):
        actor = EventActor(self.policy)
        actor.act(self.obs, self.terrain, 6)
        first = actor.last['encoded']['rotated']
        moved = copy.deepcopy(self.obs)
        for e in moved['own_entities']:
            e['position'][:2] = [900, 900]
        actor.next_decision_frame = 0
        actor.act(moved, self.terrain, 12)
        self.assertEqual(actor.last['encoded']['rotated'], first)

    def test_categorical_shroud_public_supplies_and_hidden_sightings(self):
        obs, terrain = copy.deepcopy(self.obs), copy.deepcopy(self.terrain)
        obs['visibility_grid'] = [0]*1024
        terrain['supply_sources'] = [{'id': 123, 'type': 'SupplyDock', 'position': [500, 500, 0]}]
        encoded = self.policy.encoder.encode(obs, terrain, 600)
        self.assertTrue((encoded['map_inputs'][:, 2] == 1).all())
        self.assertTrue((encoded['map_inputs'][:, 3:5] == 0).all())
        self.assertGreater(encoded['map_inputs'][:, 9].sum(), 0)
        self.assertFalse(encoded['visible_target'][encoded['ids'][100]])
        self.assertGreater(encoded['numeric'][encoded['ids'][100], 8], 0)

    def test_masks_use_owner_capabilities_mobility_and_structures(self):
        own = self.encoded['ids']
        self.assertTrue(self.policy.actor_mask(self.encoded, 'move')[own[10]])
        self.assertFalse(self.policy.actor_mask(self.encoded, 'move')[own[20]])
        self.assertTrue(self.policy.actor_mask(self.encoded, 'sell')[own[20]])
        self.assertFalse(self.policy.actor_mask(self.encoded, 'sell')[own[10]])
        disarmed = copy.deepcopy(self.obs)
        disarmed['own_entities'][0].update(can_attack=False, can_attack_move=False)
        encoded = self.policy.encoder.encode(disarmed, self.terrain, 6)
        self.assertFalse(self.policy.actor_mask(encoded, 'attack').any())
        self.assertFalse(self.policy.verb_mask(encoded)[KINDS.index('attack_move')])

    def test_hidden_sighting_cannot_be_an_attack_target(self):
        with self.assertRaisesRegex(ValueError, 'Target outside'):
            self.policy.target(self.encoded, [{'kind': 'attack', 'units': [10], 'target': 100}])

    def test_sampling_is_reproducible_and_inference_needs_no_graph(self):
        first = self.policy.decide(self.encoded, rng=np.random.default_rng(3))
        second = self.policy.decide(self.encoded, rng=np.random.default_rng(3))
        self.assertEqual(first[:2], second[:2])
        with no_grad():
            third = self.policy.decide(self.encoded, rng=np.random.default_rng(3))
        self.assertEqual(first[:2], third[:2])
        self.assertAlmostEqual(float(first[4].data), float(third[4].data))

    def test_checkpoint_roundtrip_includes_optimizer_without_pickle(self):
        from nn_numpy import Adam
        optimizer = Adam(self.policy.parameters, 3e-4)
        optimizer.step_count = 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'
            self.policy.save(path, optimizer, {'event_optimizer_steps': {'verb_w': 7}})
            loaded, restored, document = NeuralPolicy.load(path)
        self.assertEqual(loaded.weight_hash(), self.policy.weight_hash())
        self.assertEqual(restored.step_count, 7)
        self.assertEqual(restored.parameter_steps, {'verb_w': 7})


class AgreementTests(unittest.TestCase):
    """The deployed NumPy policy and the PyTorch learner compute the same likelihoods and gradients."""
    def check(self, placement_mode):
        config, obs, terrain = fixture(placement_mode)
        policy = NeuralPolicy(config, 5)
        if placement_mode == 'full_mask':
            obs['build_placements'] = [grid(np.ones((16, 16), bool), angle=angle, frame=6) for angle in range(16)]
        rows = examples(policy, obs, terrain)
        model = TorchPolicyV3(policy, dtype=torch.float64)
        encoded = [row['encoded'] for row in rows]
        output = model.teacher_heads(encoded, [row['target'] for row in rows], *model.events(encoded))
        expected = []
        for parameter in policy.parameters.values():
            parameter.grad = None
        for row in rows:
            logp = policy.decide(row['encoded'], teacher=row['target'])[4]
            expected.append(float(logp.data))
            logp.backward()
        np.testing.assert_allclose(output['logp'].detach().numpy(), expected, atol=1e-9, rtol=1e-9)
        output['logp'].sum().backward()
        for name, parameter in model.weights.items():
            reference = policy.parameters[name].grad
            actual = parameter.grad.numpy() if parameter.grad is not None else np.zeros_like(policy.parameters[name].data)
            reference = np.zeros_like(actual) if reference is None else reference
            np.testing.assert_allclose(actual, reference, atol=3e-9, rtol=3e-8, err_msg=name)

    def test_fast_placement_likelihoods_and_gradients_match(self):
        self.check('fast')

    def test_full_mask_placement_likelihoods_and_gradients_match(self):
        self.check('full_mask')


class StorageTests(unittest.TestCase):
    def test_ragged_bits_dtypes_shapes_orders_missing_and_deduplication(self):
        arrays = [np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
                  None, np.array([[0., -0., np.nan, np.inf]]),
                  np.empty((0, 4)), np.arange(12, dtype=np.float64).reshape(3, 4)]
        columns = {'numbers': Column.build(arrays),
                   'bits': Column.build([np.array([True, False, True])]*5)}
        actual, _ = decode_columns(encode_columns(columns, [{}]*5))
        for i, expected in enumerate(arrays):
            result = actual['numbers'].at(i)
            if expected is None:
                self.assertIsNone(result)
            else:
                self.assertEqual((result.dtype, result.shape, result.tobytes()),
                                 (expected.dtype, expected.shape, expected.tobytes()))
                self.assertFalse(result.flags.writeable)
        self.assertTrue(actual['numbers'].at(0).flags.f_contiguous)
        self.assertEqual(len(actual['bits'].shapes), 1)

    def test_dtype_staging_keeps_values_and_source_lifetime(self):
        config, _, _ = fixture()
        model = TorchPolicyV3(NeuralPolicy(config, 5), dtype=torch.float32,
                              device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
        source = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
        expected = source.copy()
        result = model.stager.stage({'f64': (source, torch.float64), 'f32': (source, torch.float32),
                                     'ids': (source.astype(np.int64), torch.int64), 'mask': (source > 8, torch.bool)})
        source[:] = -999
        np.testing.assert_array_equal(result['f64'].cpu(), expected)
        np.testing.assert_array_equal(result['f32'].cpu(), expected.astype(np.float32))
        np.testing.assert_array_equal(result['ids'].cpu(), expected.astype(np.int64))
        np.testing.assert_array_equal(result['mask'].cpu(), expected > 8)


if __name__ == '__main__':
    unittest.main()
