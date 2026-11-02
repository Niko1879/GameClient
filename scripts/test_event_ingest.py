"""Batch conversion and verified native-payload retirement."""
import copy
from contextlib import closing, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import event_ingest as ingest
from event_dataset import EventDataset
from prepared_io import _digest, _write_json
from replay_storage import manifest_path
from test_fixtures import fixture, raw_row


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, obs, terrain = fixture()
        _write_json(self.root/'config.json', self.config)
        _write_json(self.root/'catalog.json', {})
        self.games, self.proofs, self.entries = {}, {}, []
        for i in range(3):
            raw, export = self.root/f'{i}.rep', self.root/f'{i}.training.jsonl'
            raw.write_bytes(b'original'+bytes([i])); export.write_bytes(b'export'+bytes([i]))
            match = f'test:{i}'
            proof = dict(path=str(export), original_replay=str(raw), sha256=_digest(export),
                replay_sha256=_digest(raw), match_key=match, audit=dict(observation_policy_version=8, placement_mode='fast'))
            self.proofs[str(export)] = proof
            self.games[str(export)] = dict(path=str(export), player=2, map=terrain, source_ticks_per_second=60,
                rows=[raw_row(0, obs), raw_row(30, obs, [{'kind':'stop','units':[10]}]), raw_row(60, obs, terminal=True)])
            self.entries.append(dict(trajectory=str(export), sha256=proof['sha256'],
                replay_sha256=proof['replay_sha256'], match_key=match, split='train'))
            _write_json(manifest_path(export), dict(trajectory_sha256=proof['sha256']))
        self.native = self.root/'native.json'
        _write_json(self.native, dict(entries=self.entries))
        self.addCleanup(patch.stopall)
        patch('event_ingest.verify_entry', side_effect=self.verify).start()
        patch('replay_loading.load_replays', side_effect=self.load).start()
        patch('event_ingest._preparation_guard').start()

    def verify(self, entry, **kwargs):
        if _digest(Path(entry['trajectory'])) != entry['sha256']:
            raise ValueError('Native payload changed')
        return self.proofs[entry['trajectory']]

    def load(self, paths, *args, **kwargs):
        return ([self.games[str(p)] for p in paths], self.config, [self.proofs[str(p)] for p in paths])

    def prepare(self, workers=1):
        with redirect_stdout(io.StringIO()):
            return ingest.prepare_batch(self.native, self.root/'events', self.root/'config.json',
                action_catalog=self.root/'catalog.json', workers=workers, minimum_ram=1)

    def test_flat_named_manifests_share_one_directory_and_resume(self):
        output = self.root/'flat-data'
        options = dict(action_catalog=self.root/'catalog.json', workers=1,
                       minimum_ram=1, flat=True, catalog=self.root/'catalog.sqlite3')
        with redirect_stdout(io.StringIO()):
            first = ingest.prepare_batch(self.native, output, self.root/'config.json',
                                         manifest_name='batch-a.json', **options)
            second = ingest.prepare_batch(self.native, output, self.root/'config.json',
                                          manifest_name='batch-b.json', **options)
        self.assertFalse(any(p.is_dir() for p in output.iterdir()))
        a, b = EventDataset(first), EventDataset(second)
        self.assertEqual([e['path'] for e in a.entries], [e['path'] for e in b.entries])
        self.assertTrue(all(Path(e['path']).parent == Path('.') for e in a.entries))
        self.assertTrue(ingest.retire_exports(self.native, first, self.root)['complete'])
        import sqlite3
        with closing(sqlite3.connect(options['catalog'])) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM prepared_shard').fetchone()[0],3)
        with patch('replay_loading.load_replays') as load:
            self.assertEqual(ingest.prepare_batch(self.native, output, self.root/'config.json',
                                                  manifest_name='batch-a.json', **options), first)
            load.assert_not_called()

    def test_exports_from_another_observation_version_are_rejected(self):
        next(iter(self.proofs.values()))['audit']['observation_policy_version'] = 5
        with self.assertRaisesRegex(ValueError, 'different observation version'):
            self.prepare()
        next(iter(self.proofs.values())).update(audit=dict(observation_policy_version=8, placement_mode='fast'))
        self.config.update(placement_mode='full_mask', build_snap_radius=0.0)
        _write_json(self.root/'config.json', self.config)
        with self.assertRaisesRegex(ValueError, 'full-mask exports'):
            self.prepare()

    def test_pool_conversion_retirement_and_resume_without_native_payloads(self):
        events = self.prepare(4)
        before = list(EventDataset(events).batches('train', seed=3, epoch=0))
        journal = ingest.retire_exports(self.native, events, self.root)
        self.assertTrue(journal['complete'])
        self.assertFalse(list(self.root.glob('*.training.jsonl')))
        self.assertEqual(len(list(self.root.glob('*.rep'))), 3)
        self.assertEqual(len(list(self.root.glob('*.training.manifest.json'))), 3)
        with patch('replay_loading.load_replays') as load:
            self.assertEqual(self.prepare(), events)
            load.assert_not_called()
        self.assertEqual(ingest.retire_exports(self.native, events, self.root), journal)
        self.assertEqual(sum(len(rows) for rows, _ in before),
                         sum(len(rows) for rows, _ in EventDataset(events).batches('train', seed=3, epoch=0)))

    def test_binary_payloads_retire_with_the_same_source_and_shard_proofs(self):
        for entry in self.entries:
            old=Path(entry['trajectory']);new=old.with_name(old.name.split('.')[0]+'.training.bin')
            old.rename(new)
            proof=self.proofs.pop(str(old));game=self.games.pop(str(old))
            proof['path']=game['path']=str(new)
            self.proofs[str(new)]=proof;self.games[str(new)]=game
            entry['trajectory']=str(new)
        _write_json(self.native,dict(entries=self.entries))
        events=self.prepare()
        journal=ingest.retire_exports(self.native,events,self.root)
        self.assertTrue(journal['complete'])
        self.assertFalse(list(self.root.glob('*.training.bin')))
        self.assertEqual(len(list(self.root.glob('*.rep'))),3)
        self.assertEqual(len(list(self.root.glob('*.training.manifest.json'))),3)
        self.assertEqual(ingest.retire_exports(self.native,events,self.root),journal)

    def test_corrupt_last_native_or_shard_preserves_all_exports(self):
        events = self.prepare()
        last = Path(self.entries[-1]['trajectory'])
        last.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Native payload changed'):
            ingest.retire_exports(self.native, events, self.root)
        self.assertTrue(all(Path(e['trajectory']).exists() for e in self.entries))
        last.write_bytes(b'export'+bytes([2]))
        shard = EventDataset(events).shard_path(EventDataset(events).entries[-1])
        with shard.open('ab') as stream: stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'shard changed'):
            ingest.retire_exports(self.native, events, self.root)
        self.assertTrue(all(Path(e['trajectory']).exists() for e in self.entries))

    def test_unowned_target_or_duplicate_or_changed_raw_is_rejected(self):
        events = self.prepare()
        with self.assertRaisesRegex(ValueError, 'owned native'):
            ingest.retire_exports(self.native, events, self.root/'elsewhere')
        _write_json(self.native, dict(entries=[*self.entries, self.entries[0]]))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            ingest.retire_exports(self.native, events, self.root)
        _write_json(self.native, dict(entries=self.entries))
        (self.root/'2.rep').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Original replay changed'):
            ingest.retire_exports(self.native, events, self.root)
        self.assertTrue(all(Path(e['trajectory']).exists() for e in self.entries))

    def test_partial_retirement_is_recoverable_only_with_prior_journal(self):
        events = self.prepare()
        unlink = Path.unlink
        count = 0
        def interrupted(path, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 2: raise OSError('interrupted deletion')
            return unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', interrupted), self.assertRaisesRegex(OSError, 'interrupted deletion'):
            ingest.retire_exports(self.native, events, self.root)
        self.assertFalse(json.loads((self.root/'retired-exports.json').read_text())['complete'])
        self.assertTrue(ingest.retire_exports(self.native, events, self.root)['complete'])

    def test_uncovered_training_vocabulary_stops_before_any_export_deletion(self):
        observed = copy.deepcopy(self.config)
        observed['types'].append('NewTrainingType')
        with patch('replay_loading.load_replays', return_value=(list(self.games.values()), observed, list(self.proofs.values()))):
            with self.assertRaisesRegex(ValueError, 'vocabulary'):
                self.prepare()
        self.assertTrue(all(Path(e['trajectory']).exists() for e in self.entries))


if __name__ == '__main__': unittest.main()
