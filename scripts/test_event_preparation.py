"""Production pool equivalence, frozen vocabulary, interruption and resumption."""
import copy
from contextlib import redirect_stdout
import io
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import event_ingest
import event_store
from prepared_io import _digest, _write_json
from test_fixtures import fixture, raw_row


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config, self.obs, self.terrain = fixture()
        self.paths, self.games, self.sources, entries = [], [], [], []
        for index in range(6):
            path = self.root/f'{index}.jsonl'
            path.write_text(str(index)*max(index, 1))
            match = f'test:{index}'
            self.paths.append(path)
            self.sources.append(dict(path=str(path), sha256=_digest(path), match_key=match,
                audit=dict(observation_policy_version=7, placement_mode='fast')))
            rows = [raw_row(0, self.obs), raw_row(30, self.obs, [{'kind': 'stop', 'units': [10]}]),
                    raw_row(60, self.obs, terminal=True)]
            self.games.append(dict(path=str(path), player=2, map=self.terrain, rows=rows, source_ticks_per_second=60))
            entries.append(dict(trajectory=str(path), sha256=_digest(path), match_key=match, split='train'))
        self.input = self.root/'input.json'
        _write_json(self.input, dict(entries=entries))
        _write_json(self.root/'config.json', self.config)
        _write_json(self.root/'catalog.json', {})
        self.addCleanup(patch.stopall)
        patch('event_ingest.verify_entry', return_value=dict(audit=dict(observation_policy_version=7,
                                                                        placement_mode='fast'))).start()
        patch('replay_loading.load_replays', side_effect=self.load).start()

    def load(self, paths, *args, **kwargs):
        selected = set(map(str, paths))
        return ([game for game in self.games if game['path'] in selected], self.config,
                [source for source in self.sources if source['path'] in selected])

    def prepare(self, name, workers):
        with redirect_stdout(io.StringIO()):
            path = event_ingest.prepare_batch(self.input, self.root/name, self.root/'config.json',
                                              action_catalog=self.root/'catalog.json', workers=workers,
                                              minimum_ram=1024**2)
        return json.loads(path.read_text())

    def members(self, name, document):
        result = []
        for entry in document['entries']:
            with zipfile.ZipFile(self.root/name/entry['path']) as archive:
                result.append({member: archive.read(member) for member in archive.namelist()})
        return result

    def test_spawn_pool_equals_serial_and_preserves_input_order_and_config(self):
        serial = self.prepare('serial', 1)
        pooled = self.prepare('pool', 4)
        self.assertEqual(self.members('serial', serial), self.members('pool', pooled))
        self.assertEqual(pooled['config'], serial['config'])
        self.assertEqual([entry['match_key'] for entry in pooled['entries']], [f'test:{i}' for i in range(6)])
        self.assertEqual(pooled['preparation_workers'], 4)
        with patch('event_store.ProcessPoolExecutor') as pool, patch('replay_loading.load_replays') as load:
            self.assertEqual(self.prepare('pool', 4), pooled)
            pool.assert_not_called()
            load.assert_not_called()

    def test_interrupted_pool_retains_shards_and_serial_resume_uses_them(self):
        def interrupted(jobs, config, output, minimum_ram, workers, publish):
            prepared = event_store._encode_preparation(jobs[2], event_store.NeuralPolicy(config), lambda: None)
            publish(jobs[2], prepared, 123)
            raise RuntimeError('Event preparation stopped at its RAM/disk resource boundary; rerun to resume')
        with patch('event_ingest._prepare_pool', side_effect=interrupted), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'resource boundary'):
                self.prepare('resume', 4)
        partial = json.loads((self.root/'resume/dataset.json').read_text())
        self.assertFalse(partial['complete'])
        self.assertEqual([entry['match_key'] for entry in partial['entries']], ['test:2'])
        encode = event_store._encode_preparation
        with patch('event_ingest._encode_preparation', wraps=encode) as calls:
            resumed = self.prepare('resume', 1)
        self.assertEqual(calls.call_count, 5)
        serial = self.prepare('reference', 1)
        self.assertEqual(self.members('resume', resumed), self.members('reference', serial))

    def test_spawn_worker_error_stops_pool_and_cleans_partial_shards(self):
        before = {process.pid for process in multiprocessing.active_children()}
        self.games[0] = copy.deepcopy(self.games[0])
        self.games[0]['rows'][0]['observation']['own_entities'][0].pop('task')
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Missing owned-unit task snapshot'):
            self.prepare('failure', 4)
        manifest = self.root/'failure/dataset.json'
        if manifest.exists():
            self.assertFalse(json.loads(manifest.read_text())['complete'])
        self.assertEqual({process.pid for process in multiprocessing.active_children()}, before)
        self.assertFalse(list((self.root/'failure').rglob('*.part')))

    def test_worker_decoder_checks_resources_and_cancellation_without_changing_decode(self):
        stopped = multiprocessing.get_context('spawn').Event()
        decoder = event_store._PreparationDecoder(json.JSONDecoder(), self.root, 4*1024**3, stopped)
        ample = dict(free_disk_bytes=100*1024**3, available_ram_bytes=6*1024**3)
        with patch('event_resources.snapshot', return_value=ample) as check:
            for i in range(129):
                self.assertEqual(decoder.decode('{"value": 7}'), {'value': 7})
            self.assertEqual(check.call_count, 2)
            stopped.set()
            decoder.count = 256
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                decoder.decode('{}')
        stopped.clear()
        decoder.count = 384
        with patch('event_resources.snapshot', return_value={**ample, 'available_ram_bytes': 3*1024**3}):
            with self.assertRaisesRegex(RuntimeError, 'RAM/disk resource boundary'):
                decoder.decode('{}')

    def test_corrupt_cached_shard_is_rejected_before_pool_launch(self):
        def interrupted(jobs, config, output, minimum_ram, workers, publish):
            prepared = event_store._encode_preparation(jobs[0], event_store.NeuralPolicy(config), lambda: None)
            publish(jobs[0], prepared)
            raise RuntimeError('interrupt')
        with patch('event_ingest._prepare_pool', side_effect=interrupted), self.assertRaisesRegex(RuntimeError, 'interrupt'):
            self.prepare('corrupt', 4)
        shard = next((self.root/'corrupt/shards').glob('*.zip'))
        with shard.open('ab') as stream:
            stream.write(b'corruption')
        with patch('event_store.ProcessPoolExecutor') as pool, self.assertRaisesRegex(ValueError, 'shard changed'):
            self.prepare('corrupt', 4)
        pool.assert_not_called()

    def test_worker_count_validation(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, 'worker count'):
                self.prepare('invalid', workers)

    def test_validation_loading_uses_the_frozen_training_policy(self):
        policy = event_store.NeuralPolicy(self.config)
        destination = self.root/'validation.zip'
        job = dict(path=str(self.paths[0]), destination=str(destination), games=None, proof=None,
                   sha256=self.sources[0]['sha256'], match_key='test:0')
        heldout_vocabulary = {**self.config, 'types': [*self.config['types'], 'HeldoutOnlyType']}
        with patch('replay_loading.load_replays', return_value=([self.games[0]], heldout_vocabulary,
                                                               [self.sources[0]])) as load:
            prepared = event_store._encode_preparation(job, policy, lambda: None)
        load.assert_called_once_with([self.paths[0]])
        self.assertNotIn('HeldoutOnlyType', policy.config['types'])
        self.assertEqual(prepared['index']['legality_key'], event_store.feature_identity(policy))

    def test_pool_restores_coordinator_thread_environment_on_failure(self):
        import os
        environment = {name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
        with patch('event_store.ProcessPoolExecutor', side_effect=RuntimeError('pool creation failed')):
            with self.assertRaisesRegex(RuntimeError, 'pool creation failed'):
                event_store._prepare_pool([], self.config, self.root, 4*1024**3, 4, lambda *args: None)
        self.assertEqual({name: os.environ.get(name) for name in environment}, environment)


if __name__ == '__main__':
    unittest.main()
