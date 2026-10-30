"""Full-epoch profiling must not lose its warm-up prefix or hide short reads."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import profile_event_training as profile


class Dataset:
    digest = 'fixture'
    config = {}

    def __init__(self, expected=5):
        self.entries = [dict(split='train', index=dict(episodes=[dict(length=expected)]))]
        self.verified, self.shared, self.calls = {}, {}, 0

    def batches(self, *args, **kwargs):
        self.calls += 1
        for index, size in enumerate((2, 1, 2)):
            self.verified[0] = True
            yield [dict(encoded=dict(numeric=[0]*(index+1))) for _ in range(size)], (0, index+1)


class EpochProfileTests(unittest.TestCase):
    def run_profile(self, dataset, *, full=True, prefetch=1):
        with tempfile.TemporaryDirectory() as temporary:
            args = ['profile_event_training.py', '--dataset', 'fixture.json', '--output', temporary,
                    '--label', 'fixture', '--loader-only', '--warmup', '1', '--batches', '1',
                    '--prefetch', str(prefetch)]
            if full:
                args.append('--full-epoch')
            with (patch.object(profile.sys, 'argv', args),
                  patch.object(profile, 'EventDataset', return_value=dataset),
                  patch.object(profile, 'snapshot', return_value=dict(free_disk_bytes=100*1024**3)),
                  patch.object(profile, 'within_budget', return_value=True), redirect_stdout(io.StringIO())):
                profile.main()
            return json.loads((Path(temporary)/'fixture.json').read_bytes())

    def test_full_epoch_restarts_after_warmup(self):
        for prefetch in (0, 1):
            with self.subTest(prefetch=prefetch):
                dataset = Dataset()
                result = self.run_profile(dataset, prefetch=prefetch)
                self.assertTrue(result['full_epoch_complete'])
                self.assertEqual(result['counts']['decisions'], 5)
                self.assertEqual(result['expected_examples'], 5)
                self.assertEqual(result['verified_shards'], 1)
                self.assertEqual(len(result['batches']), 3)
                self.assertEqual(dataset.calls, 2)

    def test_bounded_profile_keeps_existing_prefix_behavior(self):
        dataset = Dataset()
        result = self.run_profile(dataset, full=False)
        self.assertEqual(result['counts']['decisions'], 1)
        self.assertEqual(len(result['batches']), 1)
        self.assertEqual(dataset.calls, 1)
        self.assertNotIn('full_epoch_complete', result)

    def test_full_epoch_rejects_missing_examples(self):
        with self.assertRaisesRegex(ValueError, 'omitted or duplicated'):
            self.run_profile(Dataset(expected=6))


if __name__ == '__main__':
    unittest.main()
