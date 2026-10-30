"""The compiled encoder (--compile) trains like the eager learner; NVIDIA CUDA with Triton only."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from event_training_batches import prepared
from generals_policy import NeuralPolicy
from train_events import EventLearner
from training_preparation import PreparationSpec

CUDA = (os.environ.get('TEST_TORCH_DEVICE') == 'cuda' and torch.cuda.is_available() and torch.version.hip is None
        and importlib.util.find_spec('triton') is not None)


class CompiledLearnerTests(unittest.TestCase):
    def test_compile_is_refused_without_nvidia_triton(self):
        if CUDA:
            self.skipTest('compile is available here')
        policy = NeuralPolicy({**self.config(), 'hidden': 16}, 1)
        with self.assertRaisesRegex(ValueError, 'NVIDIA CUDA'):
            EventLearner(policy, device='cpu', compile_encoder=True)

    @unittest.skipUnless(CUDA, 'needs TEST_TORCH_DEVICE=cuda on NVIDIA with Triton')
    def test_compiled_encoder_matches_eager_updates(self):
        from test_event_policy import EventTests
        fixture = EventTests('setUp'); fixture.setUp()
        with tempfile.TemporaryDirectory() as temporary:
            dataset = fixture.shard_dataset(Path(temporary), validation=True)
            spec = PreparationSpec(dataset.config, dataset.key)
            updates = [u for u, _ in prepared(dataset, 'train', spec=spec, pair_budget=4000, seed=3, epoch=0,
                                               batch_size=3)]
            self.assertGreater(len(updates), 2)
            for precision in ('fp32', 'bf16'):
                learners = [EventLearner(NeuralPolicy(dataset.config, 5), device='cuda', precision=precision,
                                         compile_encoder=compiled) for compiled in (False, True)]
                for update in updates:
                    eager, compiled = (learner.run(update, train=True) for learner in learners)
                    self.assertEqual((eager['decisions'], eager['microbatches']),
                                     (compiled['decisions'], compiled['microbatches']))
                    np.testing.assert_allclose(compiled['loss_sum'], eager['loss_sum'], rtol=2e-3 if precision == 'bf16' else 1e-4)
                # Adam normalises near-zero gradients, so float-order noise can
                # move such weights by up to ~lr (3e-4) per step; losses are the
                # tight check.
                for left, right in zip(*(learner.model.parameters() for learner in learners)):
                    np.testing.assert_allclose(right.detach().cpu().numpy(), left.detach().cpu().numpy(),
                                               atol=2e-3 if precision == 'bf16' else 5e-4)
                validation = [learner.run(updates[0], train=False) for learner in learners]
                np.testing.assert_allclose(validation[1]['loss_sum'], validation[0]['loss_sum'], rtol=2e-3)
                learners[1].save(Path(temporary)/f'{precision}.npz', {})  # weights still save normally

    @staticmethod
    def config():
        from test_fixtures import fixture
        return fixture()[0]


if __name__ == '__main__':
    unittest.main()
