import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from event_dataset import EventDataset
from event_store import write_shard
from generals_policy import NeuralPolicy
from prepared_io import _write_json
from test_fixtures import fixture, raw_row
from train_events import EventLearner
import learner_bench


class LearnerBenchTests(unittest.TestCase):
    def dataset(self, root):
        config, obs, terrain = fixture()
        policy = NeuralPolicy(config, 5)
        rows = [raw_row(0, obs)]+[raw_row(f, obs, [{'kind': 'stop', 'units': [10]}]) for f in range(30, 331, 30)]
        game = {'rows': rows, 'map': terrain, 'source_ticks_per_second': 60, 'player': 2}
        entries = []
        for index, split in enumerate(('train', 'validation')):
            name = f'shards/{index}.zip'
            result = write_shard(root/name, [game], policy, {'match_key': f'test:{index}', 'path': 'absent'})
            entries.append({**result, 'path': name, 'match_key': f'test:{index}', 'split': split})
        _write_json(root/'dataset.json', {'kind': 'generals_event_dataset', 'schema': 1, 'config': config,
                                          'entries': entries})
        return policy

    def arguments(self, root, **changes):
        settings = dict(dataset=str(root/'dataset.json'), code=None, label=None, output=str(root/'run.json'),
                        device='cpu', precision='fp32', compile=False, compile_heads=False, warmup=1, updates=2,
                        widths=None, width_scale=1.0,
                        profile=0, batch_size=3, pair_budget=8_000_000, buffer_mib=64, prefetch=0, readers=1, seed=3)
        return SimpleNamespace(**{**settings, **changes})

    def test_run_times_real_updates_and_summary_compares_losses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.dataset(root)
            with contextlib.redirect_stdout(io.StringIO()):
                first = learner_bench.run(self.arguments(root, label='baseline-eager'))
                second = learner_bench.run(self.arguments(root, label='current-eager', output=str(root/'b.json'),
                                                          profile=1))
            self.assertEqual(first['updates'], 2)
            self.assertGreater(first['decisions_per_second'], 0)
            self.assertIn('train_events.py', first['code_sha256'])
            # Same seed and data: the same code reproduces every update's loss.
            self.assertEqual(first['update_losses'], second['update_losses'][:3])
            self.assertIsNotNone(second['kernels'])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                learner_bench.summary(SimpleNamespace(reports=[str(root/'run.json'), str(root/'b.json')]))
            self.assertIn('first 2 updates differ by at most 0.00e+00', out.getvalue())

    def test_gpu_sampler_parses_samples_and_names_the_bound(self):
        import sys
        script = "import time\nfor v in (40, 'N/A', 60, 50):\n    print(v, flush=True)\ntime.sleep(30)\n"
        sampler = learner_bench.GpuSampler([sys.executable, '-c', script])
        for _ in range(100):
            if len(sampler.samples) == 3:
                break
            import time
            time.sleep(.05)
        busy = sampler.stop()
        self.assertEqual((busy['samples'], busy['mean'], busy['p50']), (3, 50.0, 50.0))
        self.assertTrue(busy['cpu_bound'])
        self.assertIsNone(learner_bench.GpuSampler(['no-such-nvidia-smi-binary']).stop())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {'label': 'current-eager', 'updates': 2, 'decisions_per_second': 10., 'updates_per_second': 1.,
                      'reader_wait_share': 0., 'max_gpu_mib': 1., 'mean_loss': 1., 'kernels': None,
                      'update_losses': [1., 1.], 'gpu_busy': busy}
            (root/'a.json').write_text(json.dumps(report), encoding='utf-8')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                learner_bench.summary(SimpleNamespace(reports=[str(root/'a.json')]))
            self.assertIn('GPU busy 50% over 3 samples -> CPU-bound', out.getvalue())

    def test_width_sweep_scales_the_trunk_in_multiples_of_64(self):
        config = {'hidden': 1408, 'entity_hidden': 256, 'map_hidden': 128, 'scalar_hidden': 256}
        self.assertEqual(learner_bench.model_widths(config, scale=1.5),
                         {'hidden': 2112, 'entity_hidden': 384, 'map_hidden': 192, 'scalar_hidden': 384})
        self.assertEqual(learner_bench.model_widths(config), {})
        import cloud_bundle
        self.assertEqual(learner_bench.model_widths(config, 'wide'), cloud_bundle.WIDTHS['wide'])
        with self.assertRaises(ValueError):
            learner_bench.model_widths(config, 'wide', 2.0)
        with self.assertRaises(ValueError):
            learner_bench.model_widths(config, '{"map_side": 32}')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.dataset(root)
            with contextlib.redirect_stdout(io.StringIO()):
                plain = learner_bench.run(self.arguments(root, widths=None, width_scale=1.0))
                wide = learner_bench.run(self.arguments(root, widths=None, width_scale=1.5,
                                                        output=str(root/'wide.json')))
            self.assertEqual(plain['widths'], {})
            self.assertGreater(wide['parameters'], plain['parameters'])
            self.assertEqual(wide['updates'], 2)

    def test_run_imports_the_learner_from_another_source_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.dataset(root)
            source = Path(learner_bench.__file__).parent
            with contextlib.redirect_stdout(io.StringIO()):
                report = learner_bench.run(self.arguments(root, code=str(source)))
            self.assertEqual(Path(report['code']), source.resolve())

    def test_splits_keep_the_checkpoint_dataset_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = self.dataset(root)
            dataset = EventDataset(root/'dataset.json')
            learner = EventLearner(policy)
            learner.save(root/'best.npz', {'event_dataset_sha256': dataset.digest, 'event_match_partitions':
                                           {e['match_key']: e['split'] for e in dataset.entries}})
            with contextlib.redirect_stdout(io.StringIO()):
                learner_bench.splits(SimpleNamespace(checkpoint=str(root/'best.npz'), output=str(root/'split.json')))
            document = json.loads((root/'split.json').read_text(encoding='utf-8'))
            self.assertEqual(document['entries'], [{'match_key': 'test:0', 'split': 'train'},
                                                   {'match_key': 'test:1', 'split': 'validation'}])
            self.assertEqual(document['event_dataset_sha256'], dataset.digest)


if __name__ == '__main__':
    unittest.main()
