"""Chunk accounting counts stored feature arrays, not transient Python containers."""
import unittest

import numpy as np

from prepared_io import _array_bytes


class RowSizeTests(unittest.TestCase):
    def test_counts_only_encoded_arrays(self):
        encoded = {'numeric': np.zeros((5, 14), dtype=np.float64), 'own': np.zeros(5, dtype=bool),
                   'entities': [{'id': i, 'type': 'ChinaTank'} for i in range(5)], 'ids': {i: i for i in range(5)},
                   'rotated': True, 'frame': 120}
        self.assertEqual(_array_bytes(encoded), 5*14*8+5)

    def test_empty_row_has_no_array_bytes(self):
        self.assertEqual(_array_bytes({'entities': [], 'rotated': False}), 0)


if __name__ == '__main__':
    unittest.main()
