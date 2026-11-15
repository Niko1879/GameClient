"""Observation v11 current perception (15 Nov 2026): projectiles, ground hazards, launch-sound cues,
public superweapon timers and the firing/poisoned/burning unit flags, in NumPy and PyTorch."""
import copy
import os
import unittest

import numpy as np

from event_store import feature_identity
from generals_policy import (BURNING_COLUMN, EFFECT_FEATURES, ENTITY_FEATURES, FIRING_COLUMN, POISONED_COLUMN,
                             SCALAR_FEATURES, SUPERWEAPON_SECONDS, NeuralPolicy)
from nn_numpy import no_grad
from test_fixtures import fixture
from train_events import EventLearner
from training_legality import build_legality


def perceived(observation):
    """The fixture with one of each v11 signal."""
    observation = copy.deepcopy(observation)
    observation['own_entities'][0].update(firing=True)
    observation['known_nonowned_entities'][0].update(burning=True, poisoned=True)
    observation['effects'] = [
        {'id': 501, 'type': 'Enemy', 'owner': 3, 'kind': 'projectile', 'position': [400, 420, 30],
         'velocity': [1.5, -2.0, -0.25], 'radius': 2.0},
        {'id': 502, 'type': 'ChinaPowerPlant', 'owner': 2, 'kind': 'hazard', 'position': [150, 160, 0],
         'velocity': [0, 0, 0], 'radius': 40.0}]
    observation['audio_cues'] = [
        {'power': 'SuperweaponArtilleryBarrage', 'owner': 3, 'ticks_since': 30, 'position': [700, 650, 0]},
        {'power': 'SuperweaponCarpetBomb', 'owner': 3, 'ticks_since': 150, 'position': None}]
    observation['superweapons'] = [
        {'owner': 2, 'power': 'SuperweaponNeutronMissile', 'ready_seconds': 120, 'ready': False},
        {'owner': 3, 'power': 'SuperweaponScudStorm', 'ready_seconds': 0, 'ready': True}]
    return observation


class CurrentPerceptionTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture()
        self.policy = NeuralPolicy(self.config, 7)
        self.plain = self.policy.encoder.encode(self.obs, self.terrain, 300)
        self.encoded = self.policy.encoder.encode(perceived(self.obs), self.terrain, 300)
        self.target = {**self.policy.target(self.encoded, [{'kind': 'stop', 'units': [10]}]), 'delay': 30}

    def logp(self, encoded):
        with no_grad():
            return float(self.policy.decide(encoded, None, teacher=self.target)[4].data)

    def test_encoding_of_every_signal(self):
        e = self.encoded
        self.assertEqual(e['numeric'].shape[1], ENTITY_FEATURES)
        dozer, enemy = e['ids'][10], e['ids'][99]
        self.assertEqual(e['numeric'][dozer, FIRING_COLUMN], 1)
        self.assertEqual(e['numeric'][enemy, [POISONED_COLUMN, BURNING_COLUMN]].tolist(), [1, 1])
        self.assertEqual(len(e['scalars']), SCALAR_FEATURES)
        np.testing.assert_allclose(e['scalars'][-6:], [1, 120/SUPERWEAPON_SECONDS, 0, 1, 0, 1])
        self.assertFalse(self.plain['scalars'][-6:].any())
        # Effects first (hazard before projectile by kind), then cues by power name.
        self.assertEqual(e['effect_numeric'].shape, (4, EFFECT_FEATURES))
        hazard, projectile, barrage, carpet = e['effect_numeric']
        self.assertEqual(hazard[[8, 9, 10, 11, 12]].tolist(), [1, 0, 0, 1, 0])
        self.assertEqual(projectile[[8, 9, 10, 11, 12]].tolist(), [0, 1, 1, 0, 0])
        self.assertEqual(projectile[3], 1)
        types = self.config['types']
        self.assertEqual([types[t] for t in e['effect_types'][:2]], ['ChinaPowerPlant', 'Enemy'])
        powers = self.config['powers']
        self.assertEqual([powers[p-1] for p in e['effect_powers'][2:]], ['SuperweaponArtilleryBarrage', 'SuperweaponCarpetBomb'])
        self.assertEqual(barrage[[3, 9, 12]].tolist(), [1, 1, 1])
        self.assertAlmostEqual(barrage[13], 30/60/5)
        self.assertEqual(carpet[[0, 1, 2, 3]].tolist(), [0, 0, 0, 0], 'A cue heard far away has no position')
        self.assertEqual(self.plain['effect_numeric'].shape, (0, EFFECT_FEATURES))

    def test_effects_inform_decisions_but_are_never_actors_or_targets(self):
        self.assertEqual(len(self.encoded['entities']), len(self.plain['entities']))
        self.assertEqual(set(self.encoded['ids']), set(self.plain['ids']))
        self.assertNotIn(501, self.encoded['ids'])
        self.assertNotEqual(self.logp(self.encoded), self.logp({**self.encoded, **{k: self.plain[k] for k in
                            ('effect_numeric', 'effect_types', 'effect_powers')}}))

    def test_numpy_and_torch_agree_with_padded_effects_and_effect_weights_learn(self):
        key = feature_identity(self.policy)
        rows, targets = [], []
        for encoded in (self.encoded, self.plain):
            target = {**self.policy.target(encoded, [{'kind': 'stop', 'units': [10]}]), 'delay': 30}
            encoded['_training_legality'] = build_legality(self.policy, encoded, target, key)
            rows.append({'encoded': encoded, 'target': target, 'hold_ticks': 6, 'scope': 'event'})
            targets.append(target)
        learner = EventLearner(self.policy, device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
        learner.model.legality_key = key
        h, entities, spatial = learner.model.events([self.encoded, self.plain])
        output = learner.model.teacher_heads([self.encoded, self.plain], targets, h, entities, spatial)
        with no_grad():
            expected = [float(self.policy.decide(r['encoded'], None, teacher=t)[4].data) for r, t in zip(rows, targets)]
        np.testing.assert_allclose(output['logp'].detach().cpu().numpy(), expected, rtol=3e-6, atol=2e-5)
        learner.batch(rows, train=True)
        for name in ('effect_w', 'effect_type_w', 'effect_power_w', 'power_embedding'):
            self.assertGreater(float(learner.model.weights[name].grad.abs().sum().cpu()), 0, name)

    def test_mirrored_maps_mirror_effect_motion(self):
        observation = perceived(self.obs)
        for row in observation['own_entities']:
            row['position'] = [1000-row['position'][0], 1000-row['position'][1], row['position'][2]]
        mirrored = self.policy.encoder.encode(observation, self.terrain, 300)
        self.assertNotEqual(mirrored['rotated'], self.encoded['rotated'])
        projectile = 1
        a, b = self.encoded['effect_numeric'][projectile], mirrored['effect_numeric'][projectile]
        np.testing.assert_allclose(a[4:6], -b[4:6])
        np.testing.assert_allclose(a[6], b[6])


if __name__ == '__main__':
    unittest.main()
