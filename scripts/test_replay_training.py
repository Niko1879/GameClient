import copy
import json
from pathlib import Path
import tempfile
import struct
import unittest
from collections import Counter
from unittest.mock import patch

from replay_training import audit, iter_samples, replay_stream_summary, verified_crc_frame
from audit_replay_commands import reconcile


def fixture():
    header = {"record": "header", "schema_version": 1, "controlled_faction": "FactionChina", "ticks_per_second": 60,
              "placement_mode": "fast", "quiet_capture": "all",
              "observation_policy": {"version": 9, "excluded_templates": [], "visible_only_templates": [],
                                     "neutral_excluded_templates": [], "playable_players": [2, 3]}}
    terrain = {"record": "map", "grid_size": 1, "height": [0], "cliff": [False]}
    sighting = {"id": 10, "owner": 2, "own": False, "visible": True,
                "last_seen_frame": 6, "position": [10, 20, 0], "health": 100}
    sample = {"record": "sample", "frame": 6, "player_index": 3, "delta_ticks": 0, "time_seconds": 0.1,
              "observation": {"player": {"index": 3, "faction": "FactionChina"}, "own_entities": [],
                              "known_nonowned_entities": [sighting], "visibility_grid": [0],
                              "observation_policy_version": 9},
              "decision": "commands", "actions": [{"name": "MSG_DO_MOVETO", "arguments": []}]}
    wait = copy.deepcopy(sample)
    wait.update(frame=12, delta_ticks=6, time_seconds=0.2, decision="wait", actions=[])
    wait["observation"]["known_nonowned_entities"][0]["visible"] = False
    footer = {"record": "footer", "valid": True, "playback_completed": True, "crc_mismatch": False,
              "samples": 2, "commands": 1, "wait_samples": 1, "final_frame": 12,
              "recorded_crc_messages_seen": 1}
    return [header, terrain, sample, wait, footer]


class FileDigestTests(unittest.TestCase):
    def test_a_settled_file_is_hashed_once_and_any_change_is_seen(self):
        import hashlib
        import os
        from unittest import mock
        import replay_training
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'file.bin'
            path.write_bytes(b'first')
            old = os.stat(path).st_mtime-60
            os.utime(path, (old, old))
            digest = replay_training.file_sha256(path)
            self.assertEqual(digest, hashlib.sha256(b'first').hexdigest())
            with mock.patch.object(Path, 'open', side_effect=AssertionError('hashed again')):
                self.assertEqual(replay_training.file_sha256(path), digest)
            # Same size, later time: a rewrite.
            path.write_bytes(b'other')
            os.utime(path, (old+5, old+5))
            self.assertEqual(replay_training.file_sha256(path), hashlib.sha256(b'other').hexdigest())
            # Same size within the timestamp's granularity is not trusted while recent.
            fresh = Path(directory)/'fresh.bin'
            fresh.write_bytes(b'aaaaa')
            first = replay_training.file_sha256(fresh)
            stat = os.stat(fresh)
            fresh.write_bytes(b'bbbbb')
            os.utime(fresh, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertNotEqual(replay_training.file_sha256(fresh), first)


class TrainingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.jsonl"

    def write(self, rows):
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def test_valid_wait_and_remembered_sighting(self):
        self.write(fixture())
        self.assertEqual(audit(self.path, level='full')["wait_samples"], 1)
        self.assertEqual([s["frame"] for s in iter_samples(self.path)], [6, 12])

    def test_light_samples_only_for_scheduled_quiet_rows(self):
        rows = fixture()
        light = rows[3]
        del light["observation"]; light["light"] = True
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "Invalid light sample"):
            audit(self.path)  # quiet_capture all
        rows[0]["quiet_capture"] = "scheduled"
        self.write(rows)
        self.assertEqual(audit(self.path)["quiet_capture"], "scheduled")
        with self.assertRaisesRegex(ValueError, "full audit requires"):
            audit(self.path, level="full")
        commanded = copy.deepcopy(rows)
        commanded[2]["light"] = True; del commanded[2]["observation"]
        self.write(commanded)  # a command row, and the player's first row
        with self.assertRaisesRegex(ValueError, "Invalid light sample"):
            audit(self.path)

    def test_integrity_level_skips_only_the_entity_contract(self):
        rows = fixture()
        rows[3]["observation"]["known_nonowned_entities"][0]["health"] = 50
        self.write(rows)
        self.assertEqual(audit(self.path)['audit_level'], 'integrity')
        with self.assertRaisesRegex(ValueError, "Hidden sighting"):
            audit(self.path, level='full')
        rows[-1]["samples"] = 3
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            audit(self.path)
        with self.assertRaisesRegex(ValueError, "Unknown audit level"):
            audit(self.path, level='none')

    def test_metadata_consumer_keeps_complete_audit_and_command_counts(self):
        rows = fixture()
        rows[2]['actions'][0]['type_id'] = 1001
        self.write(rows)
        visited, captured = [], Counter()
        expected = audit(self.path)
        actual = audit(self.path, command_counter=captured,
                       record_consumer=lambda row: visited.append(copy.deepcopy(row)))
        self.assertEqual(actual, expected)
        self.assertEqual(visited, rows)
        self.assertEqual(captured, Counter({(6, 1001, 3): 1}))
        rows[-1]['valid'] = False
        self.write(rows)
        with self.assertRaisesRegex(ValueError, 'Playback failed'):
            audit(self.path, record_consumer=lambda row: None)

    def test_reconciliation_reuses_complete_audit_without_rescanning(self):
        rows = fixture()
        rows[2]['actions'][0]['type_id'] = 1001
        self.write(rows)
        captured = Counter()
        summary = audit(self.path, command_counter=captured)
        self.assertEqual(captured, Counter({(6, 1001, 3): 1}))
        with patch('audit_replay_commands.message_names', return_value={1001: 'MSG_DO_MOVETO'}), patch(
                'audit_replay_commands.binary_commands', return_value=iter([(6, 1001, 3)])), patch(
                'audit_replay_commands.records', side_effect=AssertionError('Unexpected second JSON scan')):
            result = reconcile('unused.rep', self.path, audit_summary=summary, captured_commands=captured)
        self.assertTrue(result['all_controlled_commands_captured'])
        self.assertEqual(result['captured_commands'], 1)

    def test_reconciliation_still_rejects_missing_duplicate_and_wrong_frame(self):
        rows = fixture()
        rows[2]['actions'][0]['type_id'] = 1001
        self.write(rows)
        summary = audit(self.path)
        for captured in (Counter(), Counter({(6, 1001, 3): 2}), Counter({(7, 1001, 3): 1})):
            with self.subTest(captured=captured), patch('audit_replay_commands.message_names',
                    return_value={1001: 'MSG_DO_MOVETO'}), patch('audit_replay_commands.binary_commands',
                    return_value=iter([(6, 1001, 3)])):
                with self.assertRaises(ValueError):
                    reconcile('unused.rep', self.path, audit_summary=summary, captured_commands=captured)

    def test_standalone_and_historical_audit_reconciliation_agree(self):
        rows = fixture()
        rows[2]['actions'][0]['type_id'] = 1001
        self.write(rows)
        results = []
        for summary in (None, audit(self.path)):
            with patch('audit_replay_commands.message_names', return_value={1001: 'MSG_DO_MOVETO'}), patch(
                    'audit_replay_commands.binary_commands', return_value=iter([(6, 1001, 3)])):
                results.append(reconcile('unused.rep', self.path, audit_summary=summary))
        self.assertEqual(results[0], results[1])

    def test_finite_decoder_rejects_nested_constants_and_overflow(self):
        for token in ('NaN', 'Infinity', '-Infinity', '1e999', '-1e999'):
            with self.subTest(token=token):
                self.write(fixture())
                raw = self.path.read_text().replace('"record": "header"',
                    '"record": "header", "extra": {"nested": [' + token + ']}', 1)
                self.path.write_text(raw)
                with self.assertRaisesRegex(ValueError, 'Nonfinite feature'):
                    audit(self.path)

    def test_finite_decoder_preserves_numeric_strings_and_finite_extremes(self):
        rows = fixture()
        rows[0]['extra'] = ['NaN', 'Infinity', 1e308, -0.0, 10**100]
        self.write(rows)
        self.assertEqual(audit(self.path)['samples'], 2)

    def test_filtered_observations_preserve_frozen_real_sightings(self):
        rows = fixture()
        rows[0]['observation_policy']['excluded_templates'] = ['ModdedFlameEffect']
        for row in rows[2:4]:
            row['observation']['known_nonowned_entities'][0]['type'] = 'ChinaVehicleDozer'
        self.write(rows)
        self.assertEqual(audit(self.path, level='full')['observation_policy_version'], 9)

    def test_declared_effect_filter_is_enforced_for_owned_and_remembered_objects(self):
        for owned in (False, True):
            with self.subTest(owned=owned):
                rows = fixture()
                rows[0]['observation_policy']['excluded_templates'] = ['ModdedFlameEffect']
                entity = rows[2]['observation']['known_nonowned_entities'][0]
                entity['type'] = 'ModdedFlameEffect'
                if owned:
                    entity.update(own=True, owner=3, resume_targets=[])
                    rows[2]['observation']['own_entities'] = [entity]
                    rows[2]['observation']['known_nonowned_entities'] = []
                self.write(rows)
                with self.assertRaisesRegex(ValueError, 'Excluded transient'):
                    audit(self.path, level='full')

    def test_foliage_filter_preserves_interactable_neutral_buildings(self):
        rows = fixture()
        rows[0]['observation_policy']['excluded_templates'] = ['TreePalm2', 'Shrub']
        for row in rows[2:4]:
            row['observation']['known_nonowned_entities'][0].update(type='CivilianBuilding', owner=1)
        self.write(rows)
        self.assertEqual(audit(self.path, level='full')['observation_policy_version'], 9)

    def test_foliage_cannot_enter_owned_or_remembered_entity_lists(self):
        for owned in (False, True):
            with self.subTest(owned=owned):
                rows = fixture()
                rows[0]['observation_policy']['excluded_templates'] = ['TreePalm2']
                entity = rows[2]['observation']['known_nonowned_entities'][0]
                entity.update(type='TreePalm2', own=owned, owner=3 if owned else 1, resume_targets=[])
                if owned:
                    rows[2]['observation']['own_entities'] = [entity]
                    rows[2]['observation']['known_nonowned_entities'] = []
                self.write(rows)
                with self.assertRaisesRegex(ValueError, 'Excluded transient'):
                    audit(self.path, level='full')

    def test_hidden_position_and_health_cannot_refresh(self):
        for field, value in (("position", [11, 20, 0]), ("health", 50), ("last_seen_frame", 12)):
            rows = fixture()
            rows[3]["observation"]["known_nonowned_entities"][0][field] = value
            self.write(rows)
            with self.assertRaisesRegex(ValueError, "Hidden sighting"):
                audit(self.path, level='full')

    def test_unseen_hidden_object_rejected(self):
        rows = fixture()
        rows[2]["observation"]["known_nonowned_entities"] = []
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "never seen"):
            audit(self.path, level='full')

    def test_private_enemy_queue_rejected(self):
        for field in ('production_queue', 'buildable_templates', 'trainable_templates'):
            with self.subTest(field=field):
                rows = fixture()
                rows[2]["observation"]["known_nonowned_entities"][0][field] = []
                self.write(rows)
                with self.assertRaisesRegex(ValueError, "Private nonowned"):
                    audit(self.path, level='full')

    def test_incomplete_file_yields_no_examples(self):
        self.write(fixture()[:-1])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            next(iter_samples(self.path))

    def test_divergence_and_missing_crc_coverage_rejected(self):
        for key, value in (("crc_mismatch", True), ("playback_completed", False),
                           ("valid", False), ("recorded_crc_messages_seen", 0)):
            rows = fixture()
            rows[-1][key] = value
            self.write(rows)
            with self.assertRaises(ValueError):
                audit(self.path)

    def test_verified_frame_is_the_crc_before_the_failed_comparison(self):
        native = dict(crc_comparisons=73, final_frame=7310, ticks_per_second=60)
        header = dict(crc_interval='100')
        self.assertEqual(verified_crc_frame(native, header, ''), 7200)
        # Exactly the minimum is kept; a comparison earlier is pointless.
        self.assertEqual(verified_crc_frame(dict(native, crc_comparisons=37), header, ''), 3600)
        self.assertIsNone(verified_crc_frame(dict(native, crc_comparisons=36), header, ''))
        # A comparison later than its playback, no interval or a broken capture proves nothing.
        self.assertIsNone(verified_crc_frame(dict(native, crc_comparisons=74), header, ''))
        self.assertIsNone(verified_crc_frame(native, dict(crc_interval=None), ''))
        self.assertIsNone(verified_crc_frame(native, header, 'Training export failed: x\n'))

    def test_trimmed_footer_needs_divergence_and_its_verified_frame(self):
        rows = fixture()
        rows[-1].update(playback_completed=False, crc_mismatch=True, verified_through_frame=12,
                        recorded_crc_messages_compared=1, replay_stream_eof=False, last_recorded_command_frame=99)
        self.write(rows)
        self.assertEqual(audit(self.path)['verified_through_frame'], 12)
        for key, value in (('crc_mismatch', False), ('valid', False), ('verified_through_frame', 6),
                           ('recorded_crc_messages_compared', 0)):
            with self.subTest(field=key):
                changed = copy.deepcopy(rows)
                changed[-1][key] = value
                self.write(changed)
                with self.assertRaises(ValueError):
                    audit(self.path)

    def test_reconciliation_of_a_trimmed_export_stops_at_its_verified_frame(self):
        rows = fixture()
        rows[2]['actions'][0]['type_id'] = 1001
        rows[-1].update(playback_completed=False, crc_mismatch=True, verified_through_frame=12)
        self.write(rows)
        summary = audit(self.path)
        with patch('audit_replay_commands.message_names', return_value={1001: 'MSG_DO_MOVETO'}):
            with patch('audit_replay_commands.binary_commands', return_value=iter([(6, 1001, 3), (13, 1001, 3)])):
                proof = reconcile('unused.rep', self.path, audit_summary=summary, captured_commands=Counter({(6, 1001, 3): 1}))
            self.assertTrue(proof['all_controlled_commands_captured'])
            # A command inside the verified range must still be there.
            with patch('audit_replay_commands.binary_commands', return_value=iter([(6, 1001, 3), (12, 1001, 3)])):
                with self.assertRaises(ValueError):
                    reconcile('unused.rep', self.path, audit_summary=summary, captured_commands=Counter({(6, 1001, 3): 1}))

    def test_explicit_crc_and_stream_completion_evidence(self):
        rows = fixture()
        rows[-1].update(recorded_crc_messages_compared=1, replay_stream_eof=True, last_recorded_command_frame=12)
        self.write(rows)
        self.assertEqual(audit(self.path)["recorded_crc_messages_compared"], 1)
        for key, value in (("recorded_crc_messages_compared", 0), ("recorded_crc_messages_compared", 2),
                           ("replay_stream_eof", False), ("last_recorded_command_frame", 13)):
            with self.subTest(field=key):
                changed = copy.deepcopy(rows)
                changed[-1][key] = value
                self.write(changed)
                with self.assertRaises(ValueError):
                    audit(self.path)

    def test_stream_end_is_independent_of_header_duration_and_checks_truncation(self):
        from test_harvest_china_replays import replay
        source = Path(self.temp.name) / "old.rep"
        data = replay()[:-13] + struct.pack("<IiiB", 6, 1001, 2, 0) + struct.pack("<IiiB", 12, 1001, 2, 0)
        source.write_bytes(data)
        self.assertEqual(replay_stream_summary(source), {"messages": 2, "last_command_frame": 12})
        source.write_bytes(data[:-1])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            replay_stream_summary(source)
        source.write_bytes(replay()[:-13] + struct.pack("<IiiB", 12, 1001, 2, 0) + struct.pack("<IiiB", 6, 1001, 2, 0))
        with self.assertRaisesRegex(ValueError, "monotonic"):
            replay_stream_summary(source)

    def test_time_order_and_footer_counts_rejected(self):
        for key, value in (("frame", 6), ("delta_ticks", 1), ("time_seconds", 1.0)):
            rows = fixture()
            rows[3][key] = value
            self.write(rows)
            with self.assertRaises(ValueError):
                audit(self.path)
        rows = fixture()
        rows[-1]["samples"] = 3
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            audit(self.path)

    def test_observation_faction_and_shape_rejected(self):
        rows = fixture()
        rows[2]["observation"]["player"]["faction"] = "FactionChinaTankGeneral"
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "Non-vanilla"):
            audit(self.path)
        rows = fixture()
        rows[2]["observation"]["visibility_grid"] = []
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "visibility shape"):
            audit(self.path)


if __name__ == "__main__":
    unittest.main()
