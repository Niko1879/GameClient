"""Data-parallel ranks (train_events --world-size) compute the single-process run.

CPU gloo ranks on this host; the CUDA/NCCL path differs only in backend.
"""
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from event_training_batches import assemble, lane_parts, prepare_part
from generals_policy import NeuralPolicy
from train_events import EventLearner, global_lanes, launch, rank_lanes, train
from training_preparation import PreparationSpec


def settings(data, output, **changes):
    return {**dict(dataset=str(data.path), output=str(output), resume=None, device='cpu', epochs=2, max_updates=0,
                   seed=123, lr=3e-4, threads=1, lanes=6, unroll=2, part_lanes=1, pair_budget=2_000_000, cache_mib=1,
                   checkpoint_seconds=600, min_free_ram_mib=256, max_private_mib=16384, min_free_disk_gib=1,
                   early_stopping_patience=0, min_delta=0.0, validations_per_epoch=2, prefetch_batches=0,
                   reader_processes=1), **changes}


def weights(path):
    policy, optimizer, document = NeuralPolicy.load(path)
    return policy, optimizer, document['metadata']


class DistributedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_event_policy import EventTests
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        fixture = EventTests('setUp'); fixture.setUp()
        cls.data = fixture.shard_dataset(cls.root/'data', validation=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def single(self, name, **changes):
        with redirect_stdout(io.StringIO()):
            report = train(SimpleNamespace(**settings(self.data, self.root/name, **changes)))
        return report, *weights(report['checkpoint'])

    def ranks(self, name, world, **changes):
        args = SimpleNamespace(**settings(self.data, self.root/name, **changes))
        report = launch(args, world)
        return report, *weights(report['checkpoint'])

    def assert_close_runs(self, left, right):
        (a_report, a, a_opt, _), (b_report, b, b_opt, _) = left, right
        self.assertEqual(a_report['optimizer_updates'], b_report['optimizer_updates'])
        self.assertEqual(a_opt.parameter_steps, b_opt.parameter_steps)  # Same parameters touched per update.
        for name in a.parameters:
            np.testing.assert_allclose(b.parameters[name].data, a.parameters[name].data, rtol=1e-4, atol=1e-6,
                                       err_msg=name)
        for one, other in zip(a_report['history'], b_report['history']):
            for split in ('train', 'validation'):
                for key in ('decisions', 'correct', 'commands', 'defers', 'updates', 'microbatches'):
                    if key in one[split] and not (key == 'microbatches'):
                        self.assertEqual(one[split][key], other[split][key], f'{split}.{key}')
                self.assertAlmostEqual(one[split]['mean_nll'], other[split]['mean_nll'], places=4)
        self.assertEqual(len(a_report['history']), len(b_report['history']))

    def test_two_and_three_ranks_match_one_process_on_the_same_global_batches(self):
        reference = self.single('one')
        self.assertGreater(reference[0]['optimizer_updates'], 2)
        self.assert_close_runs(reference, self.ranks('two', 2))
        # Three ranks and a reader process per rank: two of the six lanes each.
        self.assert_close_runs(reference, self.ranks('three', 3, prefetch_batches=1))

    def test_a_rank_with_an_empty_share_still_joins_the_update(self):
        # One training episode on two lanes of 11 steps: lane 1 has no episode, so rank 1's part is empty.
        self.assertEqual(len(self.data.episodes('train')), 1)
        self.assert_close_runs(self.single('one-11', lanes=2, unroll=11, epochs=1),
                               self.ranks('two-11', 2, lanes=2, unroll=11, epochs=1))

    def test_ranked_resume_matches_an_uninterrupted_ranked_run(self):
        full = self.ranks('full', 2)
        partial = self.ranks('resumed', 2, max_updates=2)
        self.assertEqual(partial[0]['status'], 'update_limit')
        resumed = self.ranks('resumed', 2, resume=partial[0]['checkpoint'])
        self.assertEqual(resumed[0]['status'], 'completed')
        self.assertEqual(full[0]['optimizer_updates'], resumed[0]['optimizer_updates'])
        self.assertEqual([h['train'] for h in full[0]['history']], [h['train'] for h in resumed[0]['history']])
        for name in full[1].parameters:
            np.testing.assert_array_equal(full[1].parameters[name].data, resumed[1].parameters[name].data)
            np.testing.assert_array_equal(full[2].m[name], resumed[2].m[name])

    def test_resume_refuses_a_different_world_size(self):
        partial = self.ranks('world-change', 2, max_updates=1)
        with self.assertRaisesRegex(ValueError, 'identical data, sampler'), redirect_stdout(io.StringIO()):
            train(SimpleNamespace(**settings(self.data, self.root/'world-change', resume=partial[0]['checkpoint'])))

    def test_a_failing_rank_stops_the_others(self):
        with self.assertRaisesRegex(RuntimeError, 'exited with code'):
            launch(SimpleNamespace(**settings(self.data, self.root/'broken', lanes=1)), 2)

    def test_nccl_ranks_match_one_gpu_strict_and_with_the_compiled_encoder(self):
        import torch
        if torch.version.hip is not None or torch.cuda.device_count() < 2:
            self.skipTest('needs two NVIDIA GPUs (the cloud instance)')
        self.assert_close_runs(self.single('gpu-one', device='cuda'), self.ranks('gpu-two', 2, device='cuda'))
        # BF16 + compile: dynamic shapes per rank, so only the trajectory is compared, loosely.
        fast = dict(device='cuda', precision='bf16', compile=True, prefetch_batches=1)
        one, two = self.single('fast-one', **fast), self.ranks('fast-two', 2, **fast)
        self.assertEqual(one[0]['optimizer_updates'], two[0]['optimizer_updates'])
        for left, right in zip(one[0]['history'], two[0]['history']):
            self.assertAlmostEqual(left['validation']['mean_nll'], right['validation']['mean_nll'], delta=.02)

    def test_shares_partition_each_update_and_scale_by_the_global_rows(self):
        spec = PreparationSpec(self.data.config, self.data.key, device_type='cpu')
        options = dict(seed=123, epoch=0, lanes=3, steps=5, part_lanes=1)

        def update(share):
            parts = [prepare_part(spec, self.data, 'train', item, 2_000_000, 1 << 20)
                     for item in lane_parts(self.data, 'train', share=share, **options) if item[0] == 0]
            return assemble(parts, options['steps'])
        whole = update(None)
        shares = [update((rank, 3)) for rank in range(3)]
        self.assertEqual([share['lanes'] for share in shares], [1, 1, 1])
        self.assertEqual({share['global_rows'] for share in shares}, {whole['rows']})
        self.assertEqual(sum(share['rows'] for share in shares), whole['rows'])
        frames = sorted(row['frame'] for share in shares for part in share['microbatches'] for row in part['rows'])
        self.assertEqual(frames, sorted(row['frame'] for part in whole['microbatches'] for row in part['rows']))
        # A rank's lanes are numbered locally; its resets are its parts' rows.
        np.testing.assert_array_equal(np.concatenate([share['resets'] for share in shares]), whole['resets'])
        # An empty share is a valid update input: it contributes nothing, but takes part.
        empty = [share for share in shares if not share['rows']]
        learner = EventLearner(NeuralPolicy(self.data.config, 5), device='cpu')
        for share in empty:
            self.assertEqual(learner.run(share, train=False, split='validation')['decisions'], 0)
        self.assertEqual(learner.run(whole, train=True)['decisions'], whole['rows'])

    def test_lane_states_gather_to_global_order_and_split_back(self):
        lanes = {'lane_h0': np.arange(12.)[:, None], 'lane_c0': -np.arange(12.)[:, None]}
        shares = [rank_lanes(lanes, 2, rank, 3) for rank in range(3)]
        np.testing.assert_array_equal(shares[1]['lane_h0'][:, 0], [2, 3, 8, 9])
        merged = global_lanes(shares, 2)
        for name in lanes:
            np.testing.assert_array_equal(merged[name], lanes[name])


if __name__ == '__main__':
    unittest.main()
