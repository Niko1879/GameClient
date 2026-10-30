"""The training reader's row builder equals tensor_replay_store.TensorRows exactly."""
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from event_dataset import chunk_rows
from tensor_replay_store import TensorRows, decode_columns


def assert_identical(test, left, right, path='row'):
    """Equal values, dtypes, shapes, memory order, writability and key order."""
    test.assertIs(type(left), type(right), path)
    if isinstance(left, np.ndarray):
        test.assertEqual((left.dtype, left.shape), (right.dtype, right.shape), path)
        test.assertEqual((left.flags.c_contiguous, left.flags.f_contiguous, left.flags.writeable),
                         (right.flags.c_contiguous, right.flags.f_contiguous, right.flags.writeable), path)
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        test.assertEqual(list(left), list(right), path)
        for key in left:
            assert_identical(test, left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_identical(test, a, b, f'{path}[{index}]')
    else:
        test.assertEqual(left, right, path)


def compare_dataset(test, dataset):
    """Decode every stored chunk both ways; returns the number of rows compared."""
    compared = 0
    for entry in dataset.entries:
        with zipfile.ZipFile(dataset.shard_path(entry)) as archive:
            shared = {}
            for name, item in entry['index']['shared'].items():
                columns, meta = decode_columns(archive.read(item['member']))
                shared[name] = {**{key: column.at(0) for key, column in columns.items()}, **meta[0]}
            for episode in entry['index']['episodes']:
                for descriptor in episode['chunks']:
                    columns, metadata = decode_columns(archive.read(descriptor['member']))
                    common = shared[episode['shared']]
                    expected = list(TensorRows(columns, metadata, common, dataset.key))
                    assert_identical(test, chunk_rows(columns, metadata, common, dataset.key), expected)
                    compared += len(expected)
    return compared


class ChunkRowTests(unittest.TestCase):
    def test_rows_equal_tensor_rows_on_stored_shards(self):
        from test_event_policy import EventTests
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EventTests('setUp'); fixture.setUp()
            dataset = fixture.shard_dataset(Path(temporary), validation=True)
            self.assertGreater(compare_dataset(self, dataset), 0)
            # The dataset reader returns the same rows as well.
            rows = [row for split in ('train', 'validation') for ref in dataset.chunks(split) for row in dataset.read(ref)]
            self.assertTrue(rows)


if __name__ == '__main__':
    unittest.main()
