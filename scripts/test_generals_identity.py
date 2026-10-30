import copy
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from generals_identity import (BASE_FACTIONS, GENERALS, PLAYER_FACTIONS, IdentityHierarchy,
                               hierarchy_config, template_identity)
from generals_policy import NeuralPolicy
from nn_numpy import Adam
from test_fixtures import fixture


def identity_fixture():
    # General-variant production needs the unscoped production toggle.
    config, observation, terrain = fixture()
    config.pop('production_scope')
    config['types'] += ['AmericaInfantryRanger', 'Lazr_AmericaInfantryRanger',
                        'SupW_AmericaInfantryRanger', 'Nuke_ChinaVehicleDozer',
                        'Nuke_ChinaCommandCenter', 'ChinaBarracks']
    config['templates'] += ['Nuke_ChinaVehicleDozer', 'ChinaBarracks']
    config['menus']['ChinaVehicleDozer|build'].append('ChinaBarracks')
    config['menus']['Nuke_ChinaCommandCenter|train'] = ['Nuke_ChinaVehicleDozer']
    config['identity_hierarchy'] = hierarchy_config(config['types'], config['templates'])
    observation['known_nonowned_entities'][0]['type'] = 'Lazr_AmericaInfantryRanger'
    return config, observation, terrain


class IdentityHierarchyTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = identity_fixture()
        self.policy = NeuralPolicy(self.config, 5)
        self.identity = self.policy.encoder.identity
        self.encoded = self.policy.encoder.encode(self.obs, self.terrain, 6)

    def test_all_playable_generals_have_base_and_subfaction(self):
        expected = {
            'AirF_AmericaInfantryRanger': ('AmericaInfantryRanger', 'USA', 'air_force'),
            'Lazr_AmericaInfantryRanger': ('AmericaInfantryRanger', 'USA', 'laser'),
            'SupW_AmericaInfantryRanger': ('AmericaInfantryRanger', 'USA', 'superweapon'),
            'Tank_ChinaVehicleDozer': ('ChinaVehicleDozer', 'China', 'tank'),
            'Infa_ChinaVehicleDozer': ('ChinaVehicleDozer', 'China', 'infantry'),
            'Nuke_ChinaVehicleDozer': ('ChinaVehicleDozer', 'China', 'nuke'),
            'Chem_GLAInfantryWorker': ('GLAInfantryWorker', 'GLA', 'toxin'),
            'Demo_GLAInfantryWorker': ('GLAInfantryWorker', 'GLA', 'demolition'),
            'Slth_GLAInfantryWorker': ('GLAInfantryWorker', 'GLA', 'stealth'),
            'AmericaInfantryRanger': ('AmericaInfantryRanger', 'USA', 'vanilla'),
            'ChinaVehicleDozer': ('ChinaVehicleDozer', 'China', 'vanilla'),
            'GLAInfantryWorker': ('GLAInfantryWorker', 'GLA', 'vanilla'),
        }
        for name, result in expected.items():
            self.assertEqual(template_identity(name), result)
        for native, (base, general) in PLAYER_FACTIONS.items():
            self.assertEqual(self.identity.player_identity(native),
                             (BASE_FACTIONS.index(base), GENERALS.index(general)))
        self.assertEqual(self.identity.player_identity('FactionBossGeneral'), (0, 0))

    def test_major_defenses_are_distinct_and_unknown_prefixes_are_preserved(self):
        names = ['AmericaPatriotBattery', 'SupW_AmericaPatriotBattery',
                 'Lazr_AmericaPatriotBattery', 'GLATunnelNetwork', 'Chem_GLATunnelNetwork']
        self.assertEqual(len({template_identity(name)[0] for name in names}), len(names))
        for name in ('SupplyDock', 'TechOilDerrick', 'Future_AmericaInfantryRanger',
                     'Boss_PatriotBattery', 'GC_Chem_GLATunnelNetwork'):
            self.assertEqual(template_identity(name), (name, '<unknown>', '<unknown>'))

    def test_training_one_ranger_transfers_to_other_variants(self):
        names = ['AmericaInfantryRanger', 'Lazr_AmericaInfantryRanger', 'SupW_AmericaInfantryRanger']
        objects = np.asarray([self.identity.objects[name] for name in names])
        before = self.policy.object_embedding(objects).data.copy()
        np.testing.assert_array_equal(before[0], before[1])
        np.testing.assert_array_equal(before[0], before[2])
        _, _, _, _, lp, _ = self.policy.decide(self.encoded, teacher={'kind': 'wait'})
        (-lp).backward()
        p = self.policy.parameters
        family = self.identity.families[objects[1]]
        self.assertGreater(np.linalg.norm(p['family_embedding'].grad[family]), 0)
        self.assertGreater(np.linalg.norm(p['variant_embedding'].grad[objects[1]]), 0)
        np.testing.assert_array_equal(p['variant_embedding'].grad[objects[[0, 2]]], 0)
        Adam(p, .001).step()
        after = self.policy.object_embedding(objects).data
        self.assertFalse(np.array_equal(before[0], after[0]))
        self.assertFalse(np.array_equal(before[2], after[2]))
        np.testing.assert_array_equal(p['variant_embedding'].data[objects[[0, 2]]], 0)
        self.assertFalse(np.array_equal(after[0], after[1]))

    def test_captured_equipment_does_not_relabel_its_owner(self):
        observation = copy.deepcopy(self.obs)
        captured = copy.deepcopy(observation['known_nonowned_entities'][0])
        captured.update(id=55, own=True, owner=observation['player']['index'], construction_percent=-1,
                        production_queue=[], task=copy.deepcopy(observation['own_entities'][0]['task']),
                        can_attack=True, can_attack_move=True, dock_targets=[], resume_targets=[],
                        buildable_templates=[], trainable_templates=[])
        observation['own_entities'].append(captured)
        encoded = self.policy.encoder.encode(observation, self.terrain, 6)
        obj = self.identity.type_objects[encoded['types'][encoded['ids'][55]]]
        self.assertEqual(self.identity.base_factions[obj], BASE_FACTIONS.index('USA'))
        self.assertEqual(self.identity.generals[obj], GENERALS.index('laser'))
        self.assertEqual(encoded['player_identity'], (BASE_FACTIONS.index('China'), GENERALS.index('vanilla')))
        observation['player']['faction'] = 'FactionChinaNukeGeneral'
        with self.assertRaisesRegex(ValueError, 'vanilla China'):
            self.policy.encoder.encode(observation, self.terrain, 6)

    def test_shared_production_features_keep_exact_legality_and_dispatch(self):
        action = {'kind': 'train', 'producer': 20, 'template': 'ChinaVehicleDozer'}
        teacher = self.policy.target(self.encoded, [action])
        result = self.policy.decide(self.encoded, teacher=teacher)
        self.assertEqual(result[0][0]['template'], 'ChinaVehicleDozer')
        invalid = {**teacher, 'template': self.config['templates'].index('Nuke_ChinaVehicleDozer')}
        with self.assertRaisesRegex(ValueError, 'categorical mask'):
            self.policy.decide(self.encoded, teacher=invalid)
        observation = copy.deepcopy(self.obs)
        observation['own_entities'][1].update(type='Nuke_ChinaCommandCenter', trainable_templates=['Nuke_ChinaVehicleDozer'])
        encoded = self.policy.encoder.encode(observation, self.terrain, 6)
        target = self.policy.target(encoded, [{**action, 'template': 'Nuke_ChinaVehicleDozer'}])
        emitted = self.policy.decide(encoded, teacher=target)[0][0]
        self.assertEqual(emitted['template'], 'Nuke_ChinaVehicleDozer')
        # The queued product has exactly the same representation as a sighting
        # or output choice of that product, not an unrelated queue vocabulary.
        observation['own_entities'][1]['production_queue'] = [
            {'remaining_quantity': 1, 'progress': 20, 'unit_type': 'Nuke_ChinaVehicleDozer'}]
        queued = self.policy.encoder.encode(observation, self.terrain, 6)
        index = queued['queued_types'][queued['ids'][20]]
        np.testing.assert_array_equal(self.policy.type_embedding(np.asarray([index])).data[0],
            self.policy.object_embedding(np.asarray([self.identity.objects['Nuke_ChinaVehicleDozer']])).data[0])

    def test_hierarchy_and_production_gradients_match_finite_differences(self):
        target = self.policy.target(self.encoded, [{'kind': 'build', 'builder': 10,
                      'template': 'ChinaPowerPlant', 'position': [230, 350, 0]}])
        def loss():
            result = self.policy.decide(self.encoded, teacher={**target, 'delay': 3})
            return -result[4]+result[3]*result[3]*.1
        loss().backward()
        obj = self.identity.objects['ChinaVehicleDozer']
        checks = [('family_embedding', (self.identity.families[obj], 0)),
                  ('variant_embedding', (obj, 1)),
                  ('base_faction_embedding', (BASE_FACTIONS.index('China'), 0)),
                  ('general_embedding', (GENERALS.index('vanilla'), 1)),
                  ('player_identity_w', (0, 1)), ('template_query_w', (0, 1)),
                  ('template_context_w', (0, 1)), ('template_b', (0,))]
        for name, index in checks:
            parameter = self.policy.parameters[name]
            analytic, original = parameter.grad[index], parameter.data[index]
            parameter.data[index] = original+1e-6; hi = float(loss().data)
            parameter.data[index] = original-1e-6; lo = float(loss().data)
            parameter.data[index] = original
            self.assertAlmostEqual(analytic, (hi-lo)/2e-6, places=5, msg=name)

    def test_training_shrinks_unobserved_variants_toward_shared_family(self):
        from train_events import EventLearner
        policy = NeuralPolicy(self.config, 5)
        obj = policy.encoder.identity.objects['Nuke_ChinaVehicleDozer']
        policy.parameters['variant_embedding'].data[obj] = .25
        learner = EventLearner(policy)
        regularization = float(learner.model.regularization_loss())
        self.assertGreater(regularization, 0)
        learner.batch([{'encoded': self.encoded, 'target': {'kind': 'wait', 'delay': 5}}], train=True)
        updated = learner.model.weights['variant_embedding'].detach().numpy()[obj]
        self.assertTrue((updated < .25).all())

    def test_checkpoint_pins_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'shared.npz'
            self.policy.save(path)
            with patch('generals_identity.FAMILY_OVERRIDES', {}):
                loaded = NeuralPolicy.load(path)[0]
            self.assertEqual(loaded.config['identity_hierarchy'], self.config['identity_hierarchy'])
            self.assertEqual(loaded.identity_hash(), self.policy.identity_hash())

    def test_invalid_or_misaligned_mappings_are_rejected(self):
        config = copy.deepcopy(self.config)
        config['identity_hierarchy']['type_objects'][1] = 0
        with self.assertRaisesRegex(ValueError, 'exact template vocabulary'):
            IdentityHierarchy(config)
        config = copy.deepcopy(self.config)
        config['identity_hierarchy']['object_generals'][1] = len(GENERALS)
        with self.assertRaisesRegex(ValueError, 'indices'):
            IdentityHierarchy(config)
        config = copy.deepcopy(self.config); config['variant_l2'] = -1
        with self.assertRaisesRegex(ValueError, 'regularization'):
            NeuralPolicy(config)


if __name__ == '__main__':
    unittest.main()
