"""Observation lifetime: temporary tracks, ordinary fog memory and neutral scenery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from replay_training import audit
from test_replay_training import fixture


class SightingRetirementTests(unittest.TestCase):
    def rows(self):
        rows = fixture()
        rows[0]['observation_policy']['visible_only_templates'] = ['CargoPlane']
        return rows

    def audit(self, rows):
        temporary = tempfile.TemporaryDirectory()
        # assertRaises clears exception-held reader frames before test cleanup.
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)/'export.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return audit(path, level='full')

    def test_ordinary_hidden_sightings_stay_frozen(self):
        rows = self.rows()
        self.assertEqual(self.audit(rows)['observation_policy_version'], 7)
        rows[3]['observation']['known_nonowned_entities'][0]['health'] += 1
        with self.assertRaisesRegex(ValueError, 'Hidden sighting changed'):
            self.audit(rows)

    def test_visible_temporary_target_allowed_but_hidden_memory_rejected(self):
        rows = self.rows()
        for row in rows[2:-1]:
            row['observation']['known_nonowned_entities'][0]['type'] = 'CargoPlane'
        with self.assertRaisesRegex(ValueError, 'Hidden temporary sighting retained'):
            self.audit(rows)
        rows[3]['observation']['known_nonowned_entities'] = []
        self.assertEqual(self.audit(rows)['samples'], 2)

    def test_retired_track_can_return_only_after_new_visible_evidence(self):
        rows = self.rows()
        returned = copy.deepcopy(rows[3])
        rows[3]['observation']['known_nonowned_entities'] = []
        returned.update(frame=18, time_seconds=.3)
        rows.insert(-1, returned)
        rows[-1].update(samples=3, wait_samples=2, final_frame=18)
        with self.assertRaisesRegex(ValueError, 'Hidden sighting changed'):
            self.audit(rows)
        target = returned['observation']['known_nonowned_entities'][0]
        target.update(visible=True, last_seen_frame=18)
        self.assertEqual(self.audit(rows)['samples'], 3)

    def test_runtime_and_sample_versions_must_match(self):
        rows = self.rows()
        rows[3]['observation']['observation_policy_version'] = 5
        with self.assertRaisesRegex(ValueError, 'Observation version mismatch'):
            self.audit(rows)
        rows = self.rows()
        del rows[0]['observation_policy']['visible_only_templates']
        with self.assertRaisesRegex(ValueError, 'visible-only template contract'):
            self.audit(rows)

    def test_neutral_scenery_allowed_only_under_playable_control(self):
        rows = self.rows()
        entity = rows[2]['observation']['known_nonowned_entities'][0]
        rows[0]['observation_policy'].update(neutral_excluded_templates=['Barrel'], playable_players=[entity['owner']])
        for row in rows[2:-1]:
            row['observation']['known_nonowned_entities'][0]['type'] = 'Barrel'
        self.assertEqual(self.audit(rows)['observation_policy_version'], 7)
        rows[0]['observation_policy']['playable_players'] = [rows[2]['player_index']]
        with self.assertRaisesRegex(ValueError, 'Neutral scenery entity entered observation'):
            self.audit(rows)
        del rows[0]['observation_policy']['playable_players']
        with self.assertRaisesRegex(ValueError, 'Missing neutral entity contract'):
            self.audit(rows)


if __name__ == '__main__':
    unittest.main()
