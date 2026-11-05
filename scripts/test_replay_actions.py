import copy
import unittest
from unittest.mock import patch

from replay_actions import ReplayActionAdapter, iter_demonstrations


def command(name, *values):
    row = {'name': name, 'arguments': [{'value': v} for v in values]}
    if name == 'MSG_DOZER_CONSTRUCT':
        # Native exports record the engine's construction outcome for every request.
        row['build_result'] = {'accepted': True, 'status': 'created'}
    return row


def sample(*commands):
    return {'actions': list(commands), 'observation': {'own_entities': [
        {'id': 1}, {'id': 2, 'trainable_templates': ['ChinaInfantryRedguard']}]}}


class ReplayAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ReplayActionAdapter({'7': 'ChinaPowerPlant', '8': 'ChinaInfantryRedguard'})

    def test_selection_and_build_become_explicit_action(self):
        result = self.adapter.decode(sample(command('MSG_CREATE_SELECTED_GROUP', True, 1),
                                            command('MSG_DOZER_CONSTRUCT', 7, [10, 20, 0], 0.5)))
        self.assertEqual(result['actions'], [{'kind': 'build', 'builder': 1, 'template': 'ChinaPowerPlant',
                                             'position': [10, 20, 0], 'angle': 0.5}])
        self.assertTrue(result['imitation_loss_mask'])

    def test_hotkey_reconstructs_selection(self):
        self.adapter.decode(sample(command('MSG_CREATE_SELECTED_GROUP', True, 2), command('MSG_CREATE_TEAM1', 1)))
        self.adapter.decode(sample(command('MSG_DESTROY_SELECTED_GROUP')))
        result = self.adapter.decode(sample(command('MSG_SELECT_TEAM1'), command('MSG_DO_MOVETO', [30, 40, 0])))
        self.assertEqual(result['actions'][0]['units'], [1])

    def test_team_reassignment_removes_old_membership(self):
        self.adapter.decode(sample(command('MSG_CREATE_TEAM1', 1, 2), command('MSG_CREATE_TEAM2', 1)))
        self.assertEqual(self.adapter.teams, {'1': [2], '2': [1]})

    def test_attack_at_hidden_sighting_is_not_a_legal_training_target(self):
        row = sample(command('MSG_CREATE_SELECTED_GROUP', True, 1), command('MSG_DO_ATTACK_OBJECT', 99))
        row['observation']['known_nonowned_entities'] = [{'id': 99, 'visible': False}]
        self.assertFalse(self.adapter.decode(row)['imitation_loss_mask'])

    def test_add_and_remove_selection(self):
        result = self.adapter.decode(sample(command('MSG_CREATE_SELECTED_GROUP', True, 1),
                                            command('MSG_CREATE_SELECTED_GROUP', False, 2),
                                            command('MSG_REMOVE_FROM_SELECTED_GROUP', 1), command('MSG_DO_STOP')))
        self.assertEqual(result['actions'], [{'kind': 'stop', 'units': [2]}])

    def test_wait_is_distinct_from_selection_only_and_unsupported(self):
        self.assertTrue(self.adapter.decode(sample())['imitation_loss_mask'])
        self.assertFalse(self.adapter.decode(sample(command('MSG_CREATE_SELECTED_GROUP', True, 1)))['imitation_loss_mask'])
        result = self.adapter.decode(sample(command('MSG_QUEUE_UPGRADE', 123, 1)))
        self.assertFalse(result['imitation_loss_mask'])
        self.assertEqual(result['unsupported'][0]['name'], 'MSG_QUEUE_UPGRADE')

    def test_partial_supported_batch_is_masked(self):
        result = self.adapter.decode(sample(command('MSG_CREATE_SELECTED_GROUP', True, 1),
                                            command('MSG_DO_STOP'), command('MSG_ENTER', 99)))
        self.assertEqual(result['actions'], [{'kind': 'stop', 'units': [1]}])
        self.assertFalse(result['imitation_loss_mask'])

    def test_template_names_replace_version_specific_ids(self):
        result = self.adapter.decode(sample(command('MSG_CREATE_SELECTED_GROUP', True, 2),
                                            command('MSG_QUEUE_UNIT_CREATE', 8, 555)))
        self.assertEqual(result['actions'], [{'kind': 'train', 'producer': 2, 'template': 'ChinaInfantryRedguard'}])

    def test_train_the_engine_refuses_is_ui_noise(self):
        # A full queue or no money: the engine ignores the order, as it does a rejected build.
        refused = sample(command('MSG_CREATE_SELECTED_GROUP', True, 2), command('MSG_QUEUE_UNIT_CREATE', 8, 555))
        refused['observation']['own_entities'][1]['trainable_templates'] = []
        result = self.adapter.decode(refused)
        self.assertEqual((result['actions'], result['unsupported']), ([], []))
        self.assertEqual(result['rejected_train_requests'], 1)
        self.assertFalse(result['imitation_loss_mask'])

    def test_missing_selection_and_template_are_masked(self):
        result = self.adapter.decode(sample(command('MSG_DO_MOVETO', [1, 2, 0])))
        self.assertFalse(result['imitation_loss_mask'])
        result = self.adapter.decode(sample(command('MSG_CREATE_SELECTED_GROUP', True, 1),
                                            command('MSG_DOZER_CONSTRUCT', 999, [1, 2, 0], 0)))
        self.assertFalse(result['imitation_loss_mask'])

    def test_game_starts_with_command_center_selected(self):
        # Replays carry no selection message for the opening dozer queue.
        opening = {'actions': [command('MSG_QUEUE_UNIT_CREATE', 9, 1)],
                   'observation': {'own_entities': [{'id': 3, 'type': 'ChinaVehicleDozer'},
                                                    {'id': 4, 'type': 'ChinaCommandCenter',
                                                     'trainable_templates': ['ChinaVehicleDozer']}]}}
        adapter = ReplayActionAdapter({'9': 'ChinaVehicleDozer'})
        result = adapter.decode(opening)
        self.assertEqual(result['actions'], [{'kind': 'train', 'producer': 4, 'template': 'ChinaVehicleDozer'}])
        self.assertTrue(result['imitation_loss_mask'])
        # Later selection messages replace it as usual.
        result = adapter.decode(dict(opening, actions=[command('MSG_CREATE_SELECTED_GROUP', True, 3),
                                                       command('MSG_DO_MOVETO', [1, 2, 0])]))
        self.assertEqual(result['actions'], [{'kind': 'move', 'units': [3], 'position': [1, 2, 0]}])

    def test_stream_reuses_each_players_selection_and_adapter(self):
        def row(player, frame, *commands):
            return dict(sample(*commands), record='sample', player_index=player, frame=frame)

        records = [
            {'record': 'header', 'template_catalog': {'7': 'ChinaPowerPlant'}},
            row(0, 0, command('MSG_CREATE_SELECTED_GROUP', True, 1)),
            row(1, 1, command('MSG_CREATE_SELECTED_GROUP', True, 2)),
            row(0, 3, command('MSG_DO_MOVETO', [30, 40, 0])),
            row(1, 4, command('MSG_DO_STOP')),
            row(0, 8),
            {'record': 'footer'},
        ]
        with patch('replay_actions.records', return_value=iter(records)), patch(
                'replay_actions.ReplayActionAdapter', wraps=ReplayActionAdapter) as constructor:
            examples = list(iter_demonstrations('unused', validate=False))
        self.assertEqual(constructor.call_count, 2)
        moves = [e for e in examples if e['actions']]
        self.assertEqual(moves[0]['actions'], [{'kind': 'move', 'units': [1], 'position': [30, 40, 0]}])
        self.assertEqual(moves[0]['hold_ticks'], 5)
        self.assertTrue(moves[0]['imitation_loss_mask'])
        self.assertEqual(moves[1]['actions'], [{'kind': 'stop', 'units': [2]}])
        self.assertTrue(moves[1]['terminal_sample'])
        self.assertIsNone(moves[1]['hold_ticks'])
        self.assertFalse(moves[1]['imitation_loss_mask'])
        self.assertTrue(moves[1]['terminal_loss_mask'])
        self.assertEqual([(e['player_index'], e['hold_ticks']) for e in examples[:2]], [(0, 3), (1, 3)])
        self.assertFalse(examples[0]['imitation_loss_mask'])
        self.assertTrue(examples[-1]['terminal_loss_mask'])


if __name__ == '__main__':
    unittest.main()
