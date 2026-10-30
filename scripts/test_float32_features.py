"""Float32 prepared features preserve targets, masks, indices and ragged layout."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from event_dataset import EventDataset
from event_store import feature_identity, write_shard
from generals_policy import NeuralPolicy
from prepared_io import _write_json
from tensor_replay_store import Column, decode_columns, encode_columns
from test_fixtures import fixture, raw_row


class Float32FeatureTests(unittest.TestCase):
    def test_column_narrowing_keeps_ragged_layout_and_exact_discrete_values(self):
        values = [np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)/7), None,
                  np.empty((0, 4)), np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)/7)]
        original = [None if v is None else v.tobytes() for v in values]
        columns = dict(features=Column.build(values, float_dtype='float32'),
                       ids=Column.build([np.array([2**53+1], dtype=np.int64)]*4, float_dtype='float32'),
                       masks=Column.build([np.array([True, False, True])]*4, float_dtype='float32'))
        encoded = encode_columns(columns, [{'target': {'kind': 'wait', 'delay': i}} for i in range(4)],
                                 float_dtype='float32')
        restored, metadata = decode_columns(encoded)
        self.assertEqual(restored['features'].data.dtype, np.float32)
        self.assertEqual(restored['ids'].data.dtype, np.int64)
        self.assertEqual(restored['masks'].data.dtype, np.bool_)
        self.assertEqual(restored['ids'].at(0)[0], 2**53+1)
        self.assertEqual(len(restored['features'].shapes), 2)
        self.assertEqual(metadata[3]['target']['delay'], 3)
        for i, value in enumerate(values):
            if value is None:
                self.assertIsNone(restored['features'].at(i))
            else:
                np.testing.assert_array_equal(restored['features'].at(i), value.astype(np.float32))
                self.assertEqual(value.tobytes(), original[i])
                self.assertFalse(restored['features'].at(i).flags.writeable)
        self.assertTrue(restored['features'].at(0).flags.f_contiguous)

    def test_unsupported_precision_and_overflow_are_rejected(self):
        for operation in (lambda: Column.build([np.ones(2)], float_dtype='float16'),
                          lambda: encode_columns({}, [], float_dtype='float16')):
            with self.assertRaises(ValueError):
                operation()
        with self.assertRaises(FloatingPointError):
            Column.build([np.array([1e300])], float_dtype='float32')

    def test_fresh_shard_stores_only_float32_features_and_exact_labels(self):
        config, obs, terrain = fixture()
        policy = NeuralPolicy(config)
        rows = [raw_row(0, obs), raw_row(30, obs, [{'kind': 'stop', 'units': [10]}]), raw_row(60, obs, terminal=True)]
        game = dict(player=2, map=terrain, rows=rows, source_ticks_per_second=60)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'shard.zip'
            prepared = write_shard(path, [game], policy, dict(match_key='fixture', sha256='source'))
            self.assertEqual(prepared['index']['legality_key'], feature_identity(policy))
            self.assertEqual(prepared['index']['float_dtype'], 'float32')
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith('.bin'):
                        columns, _ = decode_columns(archive.read(name))
                        floating = [c.data for c in columns.values() if c.data.dtype.kind == 'f']
                        self.assertTrue(floating and all(a.dtype == np.float32 for a in floating))
            entry = dict(prepared, path=path.name, match_key='fixture', split='train')
            _write_json(root/'dataset.json', dict(schema=1, kind='generals_event_dataset', complete=True,
                                                  config=config, entries=[entry]))
            dataset = EventDataset(root/'dataset.json')
            decoded = [row for ref in dataset.chunks('train') for row in dataset.read(ref)]
            self.assertEqual([(row['frame'], row['target']['kind'], row['target'].get('delay')) for row in decoded],
                             [(0, 'wait', 29), (30, 'stop', None)])


if __name__ == '__main__':
    unittest.main()
