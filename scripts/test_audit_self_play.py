import copy
import json
from pathlib import Path
import tempfile
import unittest

from audit_self_play import audit_rollout, placement_candidates


def fixture():
    observations = {str(p): {'player': {'index': p, 'dead': False}, 'own_entities': [],
                             'known_nonowned_entities': []} for p in (2, 3)}
    successor = copy.deepcopy(observations); successor['3']['player']['dead'] = True
    return [
        {'record': 'header', 'source': 'self_play', 'players': [2, 3], 'controlled_players': [3],
         'execution_settings': {'build_snap_radius': 32.0, 'build_snap_step': 4.0}},
        {'record': 'transition', 'frame': 1, 'next_frame': 7, 'observations': observations,
         'next_observations': successor, 'actions': {}, 'action_feedback': [],
         'rewards': {'2': 1, '3': -1}, 'terminated': True, 'truncated': False},
        {'record': 'footer', 'completed': True, 'steps': 1, 'frame': 7,
         'rewards': {'2': 1, '3': -1}, 'terminated': True, 'truncated': False}]


class RolloutOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.path = self.root/'episode.jsonl'

    def write(self, rows):
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')

    def test_placement_resolution_keeps_the_original_order_and_only_changes_position(self):
        rows = fixture()
        action = {'kind': 'build', 'builder': 10, 'template': 'ChinaPowerPlant', 'position': [100,100,0]}
        feedback = {'player': 3, 'action': action, 'accepted': True, 'placement_attempts': 3,
                    'placement_adjusted': True, 'executed_action': {**action, 'position': [120,100,0]}}
        rows[1]['actions'] = {'3': [action]}
        rows[1]['action_feedback'] = [feedback]
        self.write(rows)
        self.assertEqual(audit_rollout(self.path)['accepted_actions']['build'], 1)
        feedback['executed_action']['builder'] = 20
        self.write(rows)
        with self.assertRaisesRegex(ValueError, 'nonposition arguments'):
            audit_rollout(self.path)

    def test_placement_retry_limit_follows_the_recorded_snap_disc(self):
        self.assertEqual(placement_candidates({}), 1)
        self.assertEqual(placement_candidates({'build_snap_radius': 32.0, 'build_snap_step': 4.0}), 197)
        rows = fixture()
        action = {'kind': 'build', 'builder': 10, 'template': 'ChinaSupplyCenter', 'position': [100,100,0]}
        feedback = {'player': 3, 'action': action, 'accepted': False, 'placement_attempts': 197,
                    'reason': 'Illegal building location'}
        rows[1]['actions'] = {'3': [action]}
        rows[1]['action_feedback'] = [feedback]
        self.write(rows)
        self.assertEqual(audit_rollout(self.path)['rejected_actions'], {'Illegal building location': 1})
        feedback['placement_attempts'] = 198
        self.write(rows)
        with self.assertRaisesRegex(ValueError, 'retry count'):
            audit_rollout(self.path)
        rows[0]['execution_settings'] = {}
        feedback['placement_attempts'] = 2
        self.write(rows)
        with self.assertRaisesRegex(ValueError, 'retry count'):
            audit_rollout(self.path)

    def test_owned_combat_capabilities_must_not_leak_into_enemy_sightings(self):
        rows = fixture()
        rows[1]['observations']['2']['known_nonowned_entities'] = [{'can_attack': True}]
        self.write(rows)
        with self.assertRaisesRegex(ValueError, 'Private enemy state'):
            audit_rollout(self.path)

    def test_defeated_second_side_is_a_loss(self):
        rows = fixture(); self.write(rows)
        result = audit_rollout(self.path)
        self.assertEqual(result['rewards'], {'2': 1, '3': -1})
        self.assertEqual(result['controlled_players'], [3])

    def test_inverted_rewards_are_rejected_even_when_zero_sum(self):
        rows = fixture(); rows[1]['rewards'] = {'2': -1, '3': 1}; self.write(rows)
        with self.assertRaisesRegex(ValueError, 'defeat state'): audit_rollout(self.path)

    def test_footer_cannot_change_the_final_outcome(self):
        rows = fixture(); rows[-1]['rewards'] = {'2': -1, '3': 1}; self.write(rows)
        with self.assertRaisesRegex(ValueError, 'Footer disagrees'): audit_rollout(self.path)


if __name__ == '__main__':
    unittest.main()
