"""Ordering, error propagation and bounded shutdown of the CPU readers."""
from pathlib import Path
import tempfile
from threading import Event
import unittest

import numpy as np

from event_training_batches import PreparedReader, PrefetchedBatches, prepared
from training_preparation import PreparationSpec, prepare_update


def assert_same(test, left, right, path='update'):
    """Exact structural equality, including array dtypes."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        test.assertIsInstance(left, np.ndarray, path); test.assertIsInstance(right, np.ndarray, path)
        test.assertEqual(left.dtype, right.dtype, path)
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        test.assertEqual(sorted(left), sorted(right), path)
        for key in left:
            assert_same(test, left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(test, a, b, f'{path}[{index}]')
    else:
        test.assertEqual(left, right, path)


class PrefetchTests(unittest.TestCase):
    def test_preserves_every_example_and_exact_resume_cursor(self):
        expected = [([i, i+1], (i//2, 1)) for i in range(0, 20, 2)]
        with PrefetchedBatches(iter(expected)) as stream:
            self.assertEqual(list(stream), expected)

    def test_decode_error_reaches_consumer_after_prior_batch(self):
        def source():
            yield ['good'], (1, 2)
            raise ValueError('corrupt training chunk')
        with PrefetchedBatches(source()) as stream:
            self.assertEqual(next(stream), (['good'], (1, 2)))
            with self.assertRaisesRegex(ValueError, 'corrupt training chunk'):
                next(stream)

    def test_early_exit_closes_producer_without_reading_unbounded_data(self):
        closed, entered_second = Event(), Event()
        advanced = []
        def source():
            try:
                for i in range(100000):
                    advanced.append(i)
                    if i == 1:
                        entered_second.set()
                    yield i
            finally:
                closed.set()
        with PrefetchedBatches(source()) as stream:
            self.assertEqual(next(stream), 0)
            self.assertTrue(entered_second.wait(5))
        self.assertTrue(closed.is_set())
        self.assertLessEqual(len(advanced), 2)
        self.assertFalse(stream.thread.is_alive())



class PreparedReaderTests(unittest.TestCase):
    def setUp(self):
        from test_event_policy import EventTests
        self.temporary = tempfile.TemporaryDirectory()
        fixture = EventTests('setUp'); fixture.setUp()
        self.dataset = fixture.shard_dataset(Path(self.temporary.name), validation=True)
        self.spec = PreparationSpec(self.dataset.config, self.dataset.key, device_type='cpu')
        self.settings = dict(seed=3, epoch=0, batch_size=3)

    def tearDown(self):
        self.temporary.cleanup()

    def test_reader_process_matches_inline_preparation_and_cursors(self):
        inline = list(prepared(self.dataset, 'train', spec=self.spec, pair_budget=4000, **self.settings))
        self.assertGreater(len(inline), 2)
        with PreparedReader(self.dataset, self.spec, 4000, depth=2) as reader:
            assert_same(self, list(prepared(self.dataset, 'train', spec=self.spec, pair_budget=4000,
                                            reader=reader, **self.settings)), inline)
            # An abandoned stream leaves nothing behind for the next one.
            stream = reader.stream('train', **self.settings)
            assert_same(self, next(stream), inline[0])
            stream.close()
            resumed = list(reader.stream('train', **self.settings, cursor=inline[0][1]))
            assert_same(self, resumed, inline[1:])
            validation = list(reader.stream('validation', **self.settings))
            assert_same(self, validation, list(prepared(self.dataset, 'validation', spec=self.spec,
                                                        pair_budget=4000, **self.settings)))
        self.assertEqual(reader.readers, [])

    def test_shared_memory_slots_round_trip_exact_independent_arrays(self):
        from multiprocessing import shared_memory
        from queue import Queue
        from event_training_batches import _Slots, _pack, _unpack
        update = prepare_update(self.spec, next(iter(self.dataset.batches('train', **self.settings)))[0], 4000)
        arrays = []
        skeleton = _pack(update, arrays)
        self.assertTrue(arrays)  # The fixture's updates do use the shared path.
        big = np.arange(3 << 20, dtype=np.float32).reshape(-1, 3)[:, 1:]  # noncontiguous, > one slot
        value = {'update': update, 'big': (big, 'float'), 'mask': np.arange(5000) % 3 == 0, 'small': np.ones(3)}
        slots = _Slots(Queue(), 1)
        try:
            for round in range(2):
                arrays = []
                skeleton = _pack(value, arrays)
                name, size = slots.write(slots.free.get(), arrays)
                block = shared_memory.SharedMemory(name=name, track=False)
                private = np.empty(size, dtype=np.uint8); private[:] = np.frombuffer(block.buf, np.uint8, size); block.close()
                copy = _unpack(skeleton, private)
                assert_same(self, [(copy['update'], None)], [(update, None)])
                np.testing.assert_array_equal(copy['big'][0], big)
                self.assertEqual(copy['big'][1], 'float')
                np.testing.assert_array_equal(copy['mask'], value['mask'])
                self.assertTrue(copy['big'][0].flags.writeable)
                slots.free.put(0)
        finally:
            slots.close()

    def test_parallel_readers_keep_the_sampler_order_exactly(self):
        from event_store import write_shard
        from prepared_io import _write_json
        from test_event_policy import EventTests
        from test_fixtures import raw_row
        fixture = EventTests('setUp'); fixture.setUp()
        root = Path(self.temporary.name)/'many'
        rows = [raw_row(0, fixture.obs)]+[raw_row(f, fixture.obs, [{'kind': 'stop', 'units': [10]}])
                                          for f in range(30, 331, 30)]
        game = {'rows': rows, 'map': fixture.terrain, 'source_ticks_per_second': 60, 'player': 2}
        entries = []
        for number in range(5):
            result = write_shard(root/f'shards/{number}.zip', [game], fixture.policy, {'match_key': f'test:{number}'})
            entries.append({**result, 'path': f'shards/{number}.zip', 'match_key': f'test:{number}', 'split': 'train'})
        _write_json(root/'dataset.json', {'kind': 'generals_event_dataset', 'schema': 1, 'config': fixture.config,
                                          'entries': entries})
        from event_dataset import EventDataset
        dataset = EventDataset(root/'dataset.json')
        settings = dict(seed=3, epoch=1, batch_size=4, buffer_bytes=1)  # one chunk per window
        self.assertEqual(len(dataset.windows('train', seed=3, epoch=1, buffer_bytes=1)), 5)
        inline = list(prepared(dataset, 'train', spec=self.spec, pair_budget=4000, **settings))
        resume = inline[4][1]
        for processes in (2, 3):
            with self.subTest(processes=processes), PreparedReader(dataset, self.spec, 4000, 1, processes) as reader:
                assert_same(self, list(reader.stream('train', **settings)), inline)
                assert_same(self, list(reader.stream('train', **settings, cursor=resume)), inline[5:])

    def test_reader_failure_reaches_consumer(self):
        with self.dataset.shard_path(self.dataset.entries[0]).open('ab') as stream:
            stream.write(b'corruption')
        with PreparedReader(self.dataset, self.spec, 4000) as reader:
            with self.assertRaisesRegex(RuntimeError, 'integrity mismatch'):
                list(reader.stream('train', **self.settings))


if __name__ == '__main__':
    unittest.main()
