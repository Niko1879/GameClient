"""The model's enemy side input and its hidden decision clocks, in NumPy and PyTorch."""
import copy
import os
import unittest

import numpy as np

from enemy_side import NATIVE, _other
from event_policy import EventActor
from event_store import feature_identity
from generals_identity import PLAYER_FACTIONS
from generals_policy import HIDDEN_SCALARS, NeuralPolicy, visible_scalars
from harvest_china_replays import TEMPLATES
from nn_numpy import no_grad
from test_fixtures import fixture
from train_events import EventLearner
from training_legality import build_legality


class EnemyAndClockTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture()
        self.policy = NeuralPolicy(self.config, 5)
        self.encoded = self.policy.encoder.encode(self.obs, self.terrain, 300, previous_frame=240)
        self.target = {**self.policy.target(self.encoded, [{'kind': 'stop', 'units': [10]}]), 'delay': 30}

    def logp(self, encoded):
        with no_grad():
            return float(self.policy.decide(encoded, None, teacher=self.target)[4].data)

    def test_clocks_stay_stored_but_never_reach_the_model(self):
        self.assertGreater(self.encoded['scalars'][9], 0)
        for index in HIDDEN_SCALARS:
            scalars = self.encoded['scalars'].copy()
            scalars[index] += 3
            self.assertEqual(self.logp({**self.encoded, 'scalars': scalars}), self.logp(self.encoded))
        scalars = self.encoded['scalars'].copy()
        scalars[0] += 3
        self.assertNotEqual(self.logp({**self.encoded, 'scalars': scalars}), self.logp(self.encoded))
        from unit_tasks import COMMANDS
        COMMAND_AGE = 1+len(COMMANDS)
        commands = self.encoded['task_commands'].copy()
        self.assertGreater(commands[0, COMMAND_AGE], 0, 'The fixture receipt has an age')
        commands[:, COMMAND_AGE] += 3
        self.assertEqual(self.logp({**self.encoded, 'task_commands': commands}), self.logp(self.encoded))
        batch = visible_scalars(np.stack([self.encoded['scalars']]*2))
        self.assertFalse(batch[:, HIDDEN_SCALARS].any())
        self.assertTrue(self.encoded['scalars'][9], 'The stored row must stay unchanged')

    def test_enemy_side_changes_decisions_and_defaults_to_unknown(self):
        identity = self.policy.encoder.identity
        usa = {**self.encoded, 'enemy_identity': identity.player_identity('FactionAmerica')}
        gla = {**self.encoded, 'enemy_identity': identity.player_identity('FactionGLA')}
        self.assertNotEqual(self.logp(usa), self.logp(gla))
        self.assertEqual(self.logp(self.encoded), self.logp({**self.encoded, 'enemy_identity': (0, 0)}))

    def test_numpy_and_torch_agree_and_only_visible_inputs_learn(self):
        key = feature_identity(self.policy)
        identity = self.policy.encoder.identity
        self.encoded['enemy_identity'] = identity.player_identity('FactionGLAToxinGeneral')
        self.encoded['_training_legality'] = build_legality(self.policy, self.encoded, self.target, key)
        row = {'encoded': self.encoded, 'target': self.target, 'hold_ticks': 6, 'scope': 'event'}
        learner = EventLearner(self.policy, device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
        learner.model.legality_key = key
        h, entities, spatial = learner.model.events([self.encoded])
        output = learner.model.teacher_heads([self.encoded], [self.target], h, entities, spatial)
        np.testing.assert_allclose(output['logp'].detach().cpu().numpy(), [self.logp(self.encoded)], rtol=3e-6, atol=2e-5)
        learner.batch([row], train=True)
        self.assertGreater(float(learner.model.weights['enemy_identity_w'].grad.abs().sum().cpu()), 0)
        gradient = learner.model.weights['scalar_w'].grad.abs().sum(dim=1).cpu().numpy()
        self.assertFalse(gradient[list(HIDDEN_SCALARS)].any())
        self.assertGreater(gradient[0], 0)

    def test_actor_passes_the_lobby_side(self):
        actor = EventActor(self.policy, 1)
        actor.configure_enemy('FactionGLA')
        actor.act(self.obs, self.terrain, 0)
        self.assertEqual(actor.last['encoded']['enemy_identity'], self.policy.encoder.identity.player_identity('FactionGLA'))
        self.assertNotEqual(actor.enemy_identity, (0, 0))
        actor.configure_enemy('random')
        self.assertEqual(actor.enemy_identity, (0, 0))
        actor.configure_enemy('FactionGLA')
        actor.reset()
        self.assertEqual(actor.enemy_identity, (0, 0))

    def test_positions_become_bits_and_scatter_pixels(self):
        from generals_policy import location_order, position_bits, scatter_weights
        bits = position_bits(np.array([[0, 1], [.5, 255.5/256]]))
        self.assertEqual(bits[0].tolist(), [0]*8+[1]*8)
        self.assertEqual(bits[1].tolist(), [1]+[0]*7+[1]*8)
        weights = scatter_weights(np.array([3]*8+[1]), np.ones(9, dtype=bool), 4)
        self.assertEqual(weights.tolist(), [.5]*8+[1])
        order = location_order(self.config).reshape(-1)
        self.assertEqual(sorted(order.tolist()), list(range(self.config['position_bins']**2)))

    def test_weapon_state_becomes_cooldown_bins_and_an_alert(self):
        from generals_policy import COOLDOWN_COLUMN, ENTITY_FEATURES, UNDER_ATTACK_COLUMN, derived_inputs, map_planes
        observation = copy.deepcopy(self.obs)
        dozer, center = observation['own_entities']
        dozer.update(weapon_cooldown_ticks=27, ticks_since_damage=119)
        center.update(weapon_cooldown_ticks=2_147_459_520, ticks_since_damage=121)
        encoded = self.policy.encoder.encode(observation, self.terrain, 300)
        first, second = encoded['ids'][dozer['id']], encoded['ids'][center['id']]
        numeric = encoded['numeric']
        self.assertEqual(numeric.shape[1], ENTITY_FEATURES)
        self.assertAlmostEqual(numeric[first, COOLDOWN_COLUMN], .45)
        self.assertEqual(numeric[second, COOLDOWN_COLUMN], 10, 'A weapon that cannot fire stays bounded')
        self.assertEqual(numeric[first, UNDER_ATTACK_COLUMN], 1)
        self.assertEqual(numeric[second, UNDER_ATTACK_COLUMN], 0)
        bins = derived_inputs(numeric)[:, 16:]
        self.assertEqual(int(bins[first].argmax()), 10)
        self.assertEqual(int(bins[second].argmax()), 31)
        side = self.terrain['grid_size']
        alert = map_planes(encoded)[12]
        self.assertEqual(alert.reshape(-1)[encoded['cells'][first]], 1)
        self.assertEqual(alert.sum(), 1)
        self.assertEqual(alert.shape, (side, side))
        at_half_rate = self.policy.encoder.encode(observation, self.terrain, 300, source_ticks_per_second=30)
        self.assertAlmostEqual(at_half_rate['numeric'][first, COOLDOWN_COLUMN], .9)
        self.assertEqual(at_half_rate['numeric'][first, UNDER_ATTACK_COLUMN], 0, '119 ticks at 30 Hz is nearly 4 seconds')

    def test_motion_facing_and_veterancy_for_seen_units(self):
        observation = copy.deepcopy(self.obs)
        seen, hidden = observation['known_nonowned_entities']
        for row in (seen, hidden):
            row.update(velocity_x=.5, velocity_y=-1., facing=np.pi/2, veterancy=3)
        for rotated in (False, True):
            home = [900, 900] if rotated else [100, 100]
            encoded = self.policy.encoder.encode(observation, self.terrain, 300, home_position=home)
            self.assertEqual(encoded['rotated'], rotated)
            sign = -1 if rotated else 1
            row = encoded['numeric'][encoded['ids'][seen['id']]]
            np.testing.assert_allclose(row[16:21], [sign*.3, sign*-.6, sign*1, 0, 1], atol=1e-12)
            row = encoded['numeric'][encoded['ids'][hidden['id']]]
            np.testing.assert_allclose(row[16:21], [0, 0, sign*1, 0, 1], atol=1e-12, err_msg='Hidden motion is unknown')

    def test_location_is_one_distribution_over_every_bin(self):
        with no_grad():
            trunk = self.policy._trunk(self.encoded)
            _, logits = self.policy._location(trunk[2], trunk[0])
        n, fine = self.config['map_side'], self.config['position_bins']//self.config['map_side']
        self.assertEqual(logits.data.shape, (n*n, fine*fine))
        cell = logits.logsumexp(axis=1).data
        total = np.exp(cell-cell.max()).sum()*np.exp(cell.max())
        np.testing.assert_allclose(total, np.exp(logits.data).sum())

    def test_replay_sides_name_every_native_faction(self):
        self.assertEqual(set(NATIVE.values()), set(PLAYER_FACTIONS))
        self.assertEqual(set(TEMPLATES.values())-{'Random'}, set(NATIVE))
        self.assertEqual(_other(['China', 'GLA Toxin']), 'FactionGLAToxinGeneral')
        self.assertEqual(_other(['China', 'China']), 'FactionChina')
        self.assertIsNone(_other(['China', 'Random']))
        self.assertIsNone(_other(['Random', 'Random']))
        self.assertIsNone(_other(()))


if __name__ == '__main__':
    unittest.main()
