"""Updates uploaded ahead on a background thread train exactly as uploads on the main thread."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from event_training_batches import PreparedReader, prepared
from generals_policy import NeuralPolicy
from train_events import EventLearner
from training_preparation import PreparationSpec

DEVICE = os.environ.get('TEST_TORCH_DEVICE', 'cpu')


class StagedUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_event_policy import EventTests
        cls.temporary = tempfile.TemporaryDirectory()
        fixture = EventTests('setUp'); fixture.setUp()
        cls.data = fixture.shard_dataset(Path(cls.temporary.name), validation=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def updates(self, learner, reader=None):
        spec = PreparationSpec(self.data.config, self.data.key, float_dtype=learner.model.float_dtype,
                               device_type=learner.model.device_type)
        return prepared(self.data, 'train', spec=spec, pair_budget=2_000_000, reader=reader,
                        seed=3, epoch=0, batch_size=3)

    def train(self, staged, reader=False):
        learner = EventLearner(NeuralPolicy(self.data.config, 5), device=DEVICE)
        spec = PreparationSpec(self.data.config, self.data.key, float_dtype=learner.model.float_dtype,
                               device_type=learner.model.device_type)
        source = PreparedReader(self.data, spec, 2_000_000, 2) if reader else None
        try:
            stream = self.updates(learner, source)
            stream = learner.staged(stream) if staged else stream
            totals = [learner.run(update, train=True) for update, _ in stream]
            if staged:
                stream.close()
            validation = [learner.run(update, train=False) for update, _ in self.updates(learner)]
        finally:
            if source is not None:
                source.close()
        learner.save(Path(self.temporary.name)/f'staged-{staged}-{reader}.npz', {})
        weights = {name: value.detach().cpu().numpy().copy() for name, value in learner.model.weights.items()}
        return totals, validation, weights

    def test_staged_training_is_bitwise_the_main_thread_training(self):
        plain = self.train(False)
        for reader in (False, True):
            with self.subTest(reader=reader):
                staged = self.train(True, reader)
                self.assertGreater(len(staged[0]), 2)
                self.assertEqual(staged[0], plain[0])
                self.assertEqual(staged[1], plain[1])
                for name in plain[2]:
                    np.testing.assert_array_equal(staged[2][name], plain[2][name], err_msg=name)

    def test_errors_arrive_after_earlier_updates_and_close_stops_the_thread(self):
        learner = EventLearner(NeuralPolicy(self.data.config, 5), device=DEVICE)
        good = list(self.updates(learner))

        def source():
            yield good[0]
            raise ValueError('corrupt training chunk')
        stream = learner.staged(source())
        self.assertIs(next(stream)[0], good[0][0])
        self.assertIn('staged_values', good[0][0]['microbatches'][0])
        with self.assertRaisesRegex(ValueError, 'corrupt training chunk'):
            next(stream)
        stream.close()
        self.assertFalse(stream.thread.is_alive())

        closed = []
        def endless():
            try:
                while True:
                    yield list(self.updates(learner))[0]
            finally:
                closed.append(True)
        stream = learner.staged(endless())
        next(stream)
        stream.close()
        self.assertFalse(stream.thread.is_alive())
        self.assertEqual(closed, [True])


if __name__ == '__main__':
    unittest.main()
