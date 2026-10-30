"""Upgrades, abilities, garrison and promotions: native legality, labels and training parity."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from event_store import feature_identity
from generals_policy import KINDS, NeuralPolicy, Unrepresentable
from nn_numpy import no_grad
from replay_actions import ReplayActionAdapter
from replay_training import audit
from tensor_replay_store import TensorRows, decode_columns, encode_columns, project
from test_fixtures import entity, fixture
from test_replay_actions import command, sample
from test_replay_training import fixture as audit_fixture
from train_events import EventLearner
from training_legality import build_legality
from training_preparation import prepare_update


def fixture_commands():
    config, obs, terrain = fixture()
    dozer, center = obs['own_entities']
    dozer.update(repair_targets=[40], commands=['Command_Capture', 'Command_ArtilleryBarrage'], command_targets=[{'command': 'Command_Capture', 'targets': [99, 101]}],
                 enter_targets=[40])
    center.update(production_queue=[{'production_id': 3, 'kind': 1, 'progress': 20.0, 'remaining_quantity': 1,
                                      'unit_type': 'ChinaVehicleDozer'}],
                  commands=['Command_UpgradeChinaRadar', 'Command_ArtilleryBarrage'],
                  upgrades=['Upgrade_InfantryCaptureBuilding'],
                  command_cooldowns=[{'command': 'Command_EMPPulse', 'ready_percent': 40.0}])
    obs['own_entities'] += [
        entity(11, 'ChinaVehicleDozer', True, [140, 100, 0], enter_targets=[40], repair_depots=[20]),
        entity(50, 'ChinaPowerPlant', True, [500, 200, 0], construction_percent=30, can_guard=False),
        entity(40, 'ChinaBunker', True, [300, 300, 0], commands=['Command_ChinookUnload'], occupants=1, capacity=5,
               can_guard=False),
        entity(41, 'ChinaInfantryRedguard', True, [300, 300, 0], container_id=40)]
    obs['known_nonowned_entities'].append(entity(101, 'Enemy', False, [700, 820, 0]))
    obs['player'].update(purchasable_sciences=['SCIENCE_ArtilleryBarrage1', 'SCIENCE_Frenzy1'], upgrades=['Upgrade_ChinaRadar'])
    return config, obs, terrain


ACTIONS = ([{'kind': 'upgrade', 'producer': 20, 'command': 'Command_UpgradeChinaRadar'}],
           [{'kind': 'science', 'science': 'SCIENCE_ArtilleryBarrage1'}],
           [{'kind': 'ability', 'units': [10], 'command': 'Command_Capture', 'target': 99}],
           [{'kind': 'ability', 'units': [20], 'command': 'Command_ArtilleryBarrage', 'position': [512.0, 640.0, 0]}],
           [{'kind': 'enter', 'units': [10, 11], 'target': 40}],
           [{'kind': 'exit', 'units': [41]}],
           [{'kind': 'evacuate', 'units': [40]}],
           [{'kind': 'guard', 'units': [10, 11], 'position': [400.0, 300.0, 0]}],
           [{'kind': 'guard_air', 'units': [41], 'position': [320.0, 600.0, 0]}],
           [{'kind': 'cancel_production', 'producer': 20, 'slot': 0}],
           [{'kind': 'cancel_construction', 'units': [50]}],
           [{'kind': 'scatter', 'units': [10, 11]}],
           [{'kind': 'add_waypoint', 'units': [10], 'position': [600.0, 600.0, 0]}],
           [{'kind': 'force_attack', 'units': [10], 'target': 99}],
           [{'kind': 'force_attack_ground', 'units': [10, 11], 'position': [820.0, 790.0, 0]}],
           [{'kind': 'repair', 'units': [10], 'target': 40}],
           [{'kind': 'get_repaired', 'units': [11], 'target': 20}])


class CommandActionTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture_commands()
        self.policy = NeuralPolicy(self.config, 5)

    def encode(self, obs=None):
        return self.policy.encoder.encode(self.obs if obs is None else obs, self.terrain, 6)

    def test_scope_keeps_china_menus_and_shortcuts_only(self):
        self.assertEqual(self.config['commands'],
                         ['Command_ArtilleryBarrage', 'Command_Capture', 'Command_UpgradeChinaRadar'])
        self.assertEqual(self.config['sciences'], ['SCIENCE_ArtilleryBarrage1', 'SCIENCE_Frenzy1'])

    def test_every_new_kind_is_legal_and_round_trips_through_its_teacher(self):
        encoded = self.encode()
        verbs = self.policy.verb_mask(encoded)
        for kind in ('upgrade', 'science', 'ability', 'enter', 'exit', 'evacuate', 'guard', 'guard_air', 'cancel_production',
                     'cancel_construction', 'scatter', 'add_waypoint', 'force_attack', 'force_attack_ground',
                     'repair', 'get_repaired'):
            self.assertTrue(verbs[KINDS.index(kind)], kind)
        for actions in ACTIONS:
            teacher = {**self.policy.target(encoded, actions), 'delay': 3}
            with no_grad():
                decoded, trace, *_ = self.policy.decide(encoded, None, teacher=teacher)
            expected = copy.deepcopy(actions)
            if 'position' in expected[0]:
                # Positions decode at the centre of their quantized cell.
                self.assertEqual(len(decoded[0]['position']), 3)
                for a, b in zip(decoded[0]['position'][:2], expected[0]['position'][:2]):
                    self.assertLess(abs(a-b), 1000/self.config['position_bins'])
                decoded[0].pop('position'); expected[0].pop('position')
            self.assertEqual(decoded, expected)
            self.assertEqual(trace['kind'], actions[0]['kind'])

    def test_unavailable_or_foreign_commands_and_targets_are_unrepresentable(self):
        encoded = self.encode()
        for action in ({'kind': 'upgrade', 'producer': 10, 'command': 'Command_UpgradeChinaRadar'},
                       {'kind': 'ability', 'units': [10], 'command': 'Command_Capture', 'target': 100},
                       {'kind': 'ability', 'units': [10, 11], 'command': 'Command_Capture', 'target': 99},
                       {'kind': 'ability', 'units': [20], 'command': 'Command_NotInScope', 'position': [1, 1, 0]},
                       {'kind': 'enter', 'units': [20], 'target': 40},
                       {'kind': 'exit', 'units': [10]},
                       {'kind': 'evacuate', 'units': [20]},
                       {'kind': 'guard', 'units': [20], 'position': [400, 300, 0]},
                       {'kind': 'guard', 'units': [99], 'position': [400, 300, 0]},
                       {'kind': 'cancel_production', 'producer': 20, 'slot': 1},
                       {'kind': 'cancel_construction', 'units': [20]},
                       {'kind': 'force_attack', 'units': [10], 'target': 100},
                       {'kind': 'repair', 'units': [10], 'target': 20},
                       {'kind': 'get_repaired', 'units': [10], 'target': 20}):
            with self.assertRaises(Unrepresentable, msg=str(action)):
                self.policy.target(encoded, [action])
        obs = copy.deepcopy(self.obs)
        obs['player']['purchasable_sciences'] = []
        encoded = self.encode(obs)
        self.assertFalse(self.policy.verb_mask(encoded)[KINDS.index('science')])
        with self.assertRaises(Unrepresentable):
            self.policy.target(encoded, ACTIONS[1])

    def test_group_ability_needs_a_command_shared_by_every_member(self):
        encoded = self.encode()
        first = encoded['ids'][10]
        mask = self.policy.selection_mask(encoded, 'ability', [first])
        self.assertFalse(mask[encoded['ids'][11]])
        # The command center shares the barrage, so that group is dispatchable.
        self.assertTrue(mask[encoded['ids'][20]])
        self.assertEqual(np.flatnonzero(self.policy.command_mask(encoded, 'ability', [first, encoded['ids'][20]])).tolist(),
                         [self.config['commands'].index('Command_ArtilleryBarrage')])

    def test_owned_state_features_and_player_flags(self):
        encoded = self.encode()
        center = encoded['ids'][20]
        self.assertEqual(encoded['entity_upgrades'][center].tolist()[:1],
                         [self.config['upgrades'].index('Upgrade_InfantryCaptureBuilding')+1])
        self.assertEqual(encoded['player_flags'].tolist(), [1, 0, 0, 0, 1, 1])
        self.assertEqual(encoded['extra_numeric'].shape[1], 14)

    def test_replay_adapter_uses_engine_resolution_and_masks_unresolved_commands(self):
        resolved = command('MSG_DO_SPECIAL_POWER_AT_OBJECT', 5, 2, 0, 0)
        resolved['command_result'] = {'resolved': True, 'action': {'kind': 'ability', 'units': [1],
                                                                   'command': 'Command_Capture', 'target': 2}}
        result = ReplayActionAdapter({}).decode(sample(resolved))
        self.assertEqual(result['actions'], [resolved['command_result']['action']])
        self.assertTrue(result['imitation_loss_mask'])
        unavailable = command('MSG_QUEUE_UPGRADE', 1, 77)
        unavailable['command_result'] = {'resolved': False, 'reason': 'Command is not available to the actor'}
        result = ReplayActionAdapter({}).decode(sample(unavailable))
        self.assertEqual(result['actions'], [])
        self.assertFalse(result['imitation_loss_mask'])
        foreign = copy.deepcopy(resolved)
        foreign['command_result']['action']['units'] = [9]
        self.assertFalse(ReplayActionAdapter({}).decode(sample(foreign))['imitation_loss_mask'])

    def test_settings_messages_neither_mask_nor_become_actions(self):
        setting = ReplayActionAdapter({}).decode(sample(command('MSG_ENABLE_RETALIATION_MODE', True)))
        self.assertEqual((setting['actions'], setting['unsupported'], setting['imitation_loss_mask']), ([], [], False))
        disarm = command('MSG_DO_WEAPON_AT_LOCATION', 0, [5, 6, 0], 1, 0)
        disarm['command_result'] = {'resolved': True, 'action': {'kind': 'ability', 'units': [1],
            'command': 'Command_DisarmMinesAtPosition', 'position': [5, 6, 0]}}
        result = ReplayActionAdapter({}).decode(sample(command('MSG_SET_MINE_CLEARING_DETAIL'), disarm))
        self.assertEqual(result['actions'], [disarm['command_result']['action']])
        self.assertTrue(result['imitation_loss_mask'])

    def test_audit_requires_a_valid_command_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'replay.jsonl'
            rows = audit_fixture()
            action = command('MSG_PURCHASE_SCIENCE', 12)
            action.update(type_id=1044, command_result={'resolved': True, 'action': {'kind': 'science', 'science': 'X'}})
            rows[2]['actions'] = [action]
            rows[2]['decision'] = 'commands'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            audit(path)
            action['command_result'] = {'resolved': True, 'action': {'kind': 'unknown'}}
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'command resolution'):
                audit(path)

    def test_new_heads_match_the_reference_through_stored_columns_and_receive_gradients(self):
        rows = []
        for actions in ([], *ACTIONS):
            encoded = self.encode()
            target = {**self.policy.target(encoded, actions), 'delay': 5}
            rows.append({'encoded': encoded, 'target': target, 'hold_ticks': 6, 'scope': 'event'})
        key = feature_identity(self.policy)
        for row in rows:
            row['encoded']['_training_legality'] = build_legality(self.policy, row['encoded'], row['target'], key)
        packed = project(rows, {}, self.policy, key)
        arrays, metadata = decode_columns(encode_columns(packed.columns, packed.metadata))
        portable = list(TensorRows(arrays, metadata, {}, key))
        learner = EventLearner(self.policy, device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
        learner.model.legality_key = key
        with no_grad():
            expected = [float(self.policy.decide(r['encoded'], None, teacher=r['target'])[4].data) for r in rows]
        for batch in (rows, portable):
            encoded = [row['encoded'] for row in batch]
            h, entities, spatial = learner.model.events(encoded)
            actual = learner.model.teacher_heads(encoded, [r['target'] for r in batch], h, entities, spatial)
            np.testing.assert_allclose(actual['logp'].detach().cpu().numpy(), expected, rtol=4e-6, atol=3e-5)
        # Science selects no units: a microbatch whose only commands are
        # promotions has no unit tokens and must still prepare and train.
        science = [rows[0], rows[2]]
        with no_grad():
            expected = [float(self.policy.decide(r['encoded'], None, teacher=r['target'])[4].data) for r in science]
        prepared = prepare_update(learner.model, science, learner.pair_budget)
        self.assertEqual([part['plan']['unit_tokens'] for part in prepared['microbatches']], [0])
        part = prepared['microbatches'][0]
        h, entities, spatial = learner.model.events(prepared=part)
        actual = learner.model.teacher_heads(None, [r['target'] for r in part['rows']], h, entities, spatial,
                                             plan=part['plan'])
        by_kind = {r['target']['kind']: value for r, value in zip(science, expected)}
        np.testing.assert_allclose(actual['logp'].detach().cpu().numpy(),
                                   [by_kind[row['target']['kind']] for row in part['rows']], rtol=4e-6, atol=3e-5)
        self.assertEqual(learner.run(prepared, train=True)['updates'], 1)
        learner.batch(portable, train=True)
        for name in ('command_embedding', 'command_query_w', 'science_embedding', 'science_query_w',
                     'upgrade_embedding', 'player_flags_w', 'target_pointer_w'):
            self.assertGreater(float(learner.model.weights[name].grad.abs().sum().cpu()), 0, name)


if __name__ == '__main__':
    unittest.main()
