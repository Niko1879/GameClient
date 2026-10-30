"""Audit reuse must preserve inputs, reject drift and retain independent proofs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import replay_metadata
from replay_loading import load_replays
from replay_storage import manifest_path
from replay_training import audit, file_sha256
from test_replay_metadata import fixture


class MetadataReceiptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root/'match_42.rep.training.jsonl'
        self.cache = self.root/'receipts'
        patched = patch('replay_metadata.CACHE_DIRECTORY', self.cache)
        patched.start(); self.addCleanup(patched.stop)
        self.write(fixture(60))

    def write(self, rows):
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')

    def load(self, **kwargs):
        return load_replays([self.path], **kwargs)

    def cache_path(self):
        return self.cache/(file_sha256(self.path)+'.json')

    def add_manifest(self):
        raw = self.root/'match_42.rep'
        raw.write_bytes(b'independently reconciled fixture')
        summary = audit(self.path)
        proof = dict(all_controlled_commands_captured=True, controlled_players=summary['players'],
                     captured_commands=summary['commands'], original_commands_by_player={})
        document = dict(schema_version=1, trajectory_sha256=file_sha256(self.path),
            replay=str(raw), replay_sha256=file_sha256(raw), engine_sha256='engine',
            replay_header={'exe_crc': 'EXE', 'ini_crc': 'INI', 'map_crc': 'MAP'},
            harvest_metadata={'match_id': 42}, audit=summary, command_reconciliation=proof)
        manifest_path(self.path).write_text(json.dumps(document))
        return raw, document, proof

    def test_hit_preserves_vocabulary_sources_order_rows_and_terminal_mask(self):
        expected = self.load()
        with patch('replay_loading.audit', side_effect=AssertionError('No re-audit')):
            actual = self.load()
        self.assertEqual(actual[1:], expected[1:])
        self.assertEqual([g['player'] for g in actual[0]], [3, 2])
        for a, b in zip(actual[0], expected[0]):
            self.assertEqual(list(a['rows']), list(b['rows']))
        self.assertEqual(actual[1]['menus'], {'ChinaVehicleDozer|build': ['ChinaPowerPlant']})

    def test_damaged_stale_and_wrong_source_receipts_fall_back_to_full_audit(self):
        expected = self.load()
        path, original = self.cache_path(), self.cache_path().read_bytes()
        variants = [None]
        for field, value in (('schema_version', 99), ('source_sha256', 'other'),
                             ('implementation', {}), ('provenance_sha256', 'other'), ('storage_verified', False)):
            document = json.loads(original)
            document[field] = value
            variants.append(document)
        changed = json.loads(original)
        changed['metadata']['templates'].append('UnobservedTemplate')
        variants.append(changed)
        for document in variants:
            with self.subTest(document=document):
                path.write_text('broken json' if document is None else json.dumps(document))
                with patch('replay_loading.audit', wraps=audit) as checked:
                    actual = self.load()
                checked.assert_called_once()
                self.assertEqual(actual[1:], expected[1:])

    def test_actual_implementation_change_misses_the_old_receipt(self):
        expected = self.load()
        changed = {**replay_metadata.implementation_identity(), 'replay_actions.py': 'new implementation'}
        with patch('replay_loading.implementation_identity', return_value=changed), patch(
                'replay_metadata.implementation_identity', return_value=changed), patch(
                'replay_loading.audit', wraps=audit) as checked:
            actual = self.load()
        checked.assert_called_once()
        self.assertEqual(actual[1:], expected[1:])

    def test_changed_source_bytes_cannot_use_a_prior_valid_receipt(self):
        self.load()
        rows = fixture(60); rows[-1]['replay_stream_eof'] = False
        self.write(rows)
        with patch('replay_loading.audit', wraps=audit) as checked:
            with self.assertRaisesRegex(ValueError, 'clean EOF'):
                self.load()
        checked.assert_called_once()

    def test_invalid_incomplete_nonfinite_or_corrupt_observations_publish_no_cache(self):
        for problem in ('incomplete', 'footer', 'visibility', 'nonfinite'):
            with self.subTest(problem=problem):
                rows = fixture(60)
                if problem == 'incomplete':
                    rows.pop()
                elif problem == 'footer':
                    rows[-1]['samples'] += 1
                elif problem == 'visibility':
                    rows[2]['observation']['visibility_grid'][0] = 3
                else:
                    rows[0]['extra'] = float('inf')
                self.write(rows)
                with self.assertRaises(ValueError):
                    self.load()
                self.assertFalse(self.cache.exists())

    def test_hit_still_checks_source_bytes_before_returning(self):
        self.load()
        original = replay_metadata.find_receipt
        def drift(*args, **kwargs):
            result = original(*args, **kwargs)
            with self.path.open('a') as stream:
                stream.write('\n')
            return result
        with patch('replay_loading.find_receipt', side_effect=drift):
            with self.assertRaisesRegex(ValueError, 'changed while loading'):
                self.load()

    def test_storage_and_reconciliation_are_reused_only_with_bound_bytes(self):
        raw, document, proof = self.add_manifest()
        with patch('replay_loading.reconcile', return_value=proof) as reconciled:
            expected = self.load()
        reconciled.assert_called_once()
        with patch('replay_loading.audit', side_effect=AssertionError('No re-audit')), patch(
                'replay_loading.reconcile', side_effect=AssertionError('No re-reconciliation')), patch(
                'replay_loading.verify_storage', side_effect=AssertionError('No storage rescan')):
            actual = self.load()
        self.assertEqual(actual[1:], expected[1:])
        raw.write_bytes(b'changed replay')
        with self.assertRaisesRegex(ValueError, 'Original replay no longer matches'):
            self.load()

    def test_export_manifest_receipt_works_without_a_disk_cache(self):
        _, document, proof = self.add_manifest()
        with patch('replay_loading.reconcile', return_value=proof):
            expected = self.load()
        document[replay_metadata.RECEIPT_FIELD] = json.loads(self.cache_path().read_text())
        manifest_path(self.path).write_text(json.dumps(document))
        self.cache_path().unlink()
        with patch('replay_loading.audit', side_effect=AssertionError('No export re-audit')), patch(
                'replay_loading.reconcile', side_effect=AssertionError('No repeated command scan')):
            actual = self.load()
        self.assertEqual(actual[1:], expected[1:])
        self.assertFalse(self.cache_path().exists())

    def test_runtime_map_data_or_replay_manifest_identity_change_misses(self):
        _, document, proof = self.add_manifest()
        with patch('replay_loading.reconcile', return_value=proof):
            expected = self.load()
            for field, value in (('engine_sha256', 'another engine'),
                    ('replay_header', {**document['replay_header'], 'map_crc': 'another map'}),
                    ('data_pack_override', {'sha256': 'another data pack'}),
                    ('replay', str(self.root/'unavailable-copy.rep'))):
                with self.subTest(field=field):
                    changed = {**document, field:value}
                    manifest_path(self.path).write_text(json.dumps(changed))
                    with patch('replay_loading.audit', wraps=audit) as checked:
                        actual = self.load()
                    checked.assert_called_once()
                    self.assertEqual(actual[1], expected[1])

    def test_drift_during_validation_publishes_no_receipt(self):
        _, _, proof = self.add_manifest()
        def changed(*unused):
            with self.path.open('a') as stream:
                stream.write('\n')
        with patch('replay_loading.verify_storage', side_effect=changed), patch(
                'replay_loading.reconcile', return_value=proof):
            with self.assertRaisesRegex(ValueError, 'changed while loading'):
                self.load()
        self.assertFalse(self.cache.exists())

    def test_provenance_drift_during_audit_publishes_no_receipt(self):
        _, document, proof = self.add_manifest()
        def changed(*unused):
            manifest_path(self.path).write_text(json.dumps({**document, 'engine_sha256': 'changed'}))
        with patch('replay_loading.verify_storage', side_effect=changed), patch(
                'replay_loading.reconcile', return_value=proof):
            with self.assertRaisesRegex(ValueError, 'provenance changed while loading'):
                self.load()
        self.assertFalse(self.cache.exists())

    def test_unwritable_cache_and_explicitly_disabled_cache_keep_full_audit_path(self):
        with patch('replay_metadata.tempfile.NamedTemporaryFile', side_effect=PermissionError('read only')):
            expected = self.load()
        self.assertFalse(list(self.cache.glob('*.json')))
        self.load()
        before = self.cache_path().read_bytes()
        with patch('replay_loading.audit', wraps=audit) as checked:
            actual = self.load(use_metadata_cache=False)
        checked.assert_called_once()
        self.assertEqual(actual[1:], expected[1:])
        self.assertEqual(self.cache_path().read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
