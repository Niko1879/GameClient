"""Observation contract, live target masks and frozen replay sightings."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from generals_policy import NeuralPolicy
from replay_training import audit
from test_fixtures import fixture
from test_replay_training import fixture as replay_fixture


class CombatVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture()
        self.policy = NeuralPolicy(self.config, 5)

    def test_current_observation_version_is_required(self):
        for version in (None, 5):
            obs = {**self.obs, 'observation_policy_version': version}
            with self.assertRaisesRegex(ValueError, 'observation version'):
                self.policy.encoder.encode(obs, self.terrain, 6)
        self.assertIn(99, self.policy.encoder.encode(self.obs, self.terrain, 6)['ids'])

    def test_current_foundation_or_revealed_unit_is_targetable_but_memory_is_not(self):
        # Native visibility supplies current entities. A foundation or revealed
        # stealth unit needs no special policy target head or inferred detector.
        obs = copy.deepcopy(self.obs)
        target = next(e for e in obs['known_nonowned_entities'] if e['id'] == 99)
        current = self.policy.encoder.encode(obs, self.terrain, 6)
        self.assertTrue(self.policy.target_mask(current, 'attack', [current['ids'][10]])[current['ids'][99]])
        target['visible'] = False
        remembered = self.policy.encoder.encode(obs, self.terrain, 6)
        self.assertFalse(self.policy.target_mask(remembered, 'attack', [remembered['ids'][10]])[remembered['ids'][99]])
        with self.assertRaises(ValueError):
            self.policy.target(remembered, [{'kind': 'attack', 'units': [10], 'target': 99}])

    def test_replay_keeps_hidden_sightings_frozen_and_requires_matching_sample_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'replay.jsonl'
            rows = replay_fixture()
            def save():
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            save()
            self.assertEqual(audit(path, level='full')["observation_policy_version"], 9)
            rows[3]['observation']['known_nonowned_entities'][0]['position'][0] += 10
            save()
            with self.assertRaisesRegex(ValueError, 'Hidden sighting changed'):
                audit(path, level='full')
            rows[3]['observation'] = copy.deepcopy(rows[2]['observation'])
            rows[3]['observation']['observation_policy_version'] = 5
            save()
            with self.assertRaisesRegex(ValueError, 'Observation version mismatch'):
                audit(path)


if __name__ == '__main__':
    unittest.main()
