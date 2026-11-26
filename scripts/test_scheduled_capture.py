"""Scheduled quiet capture: no quiet observations are kept, and preparation needs none (orders only, 26 Nov)."""
from collections import Counter
import copy
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest

import numpy as np

from generals_policy import NeuralPolicy
from replay_events import encoded_events
from replay_sequences import normalize_demonstration
from scheduled_capture import CaptureController, GrowingFile
from test_fixtures import RECORDING, fixture, raw_row

CADENCE = 6


def scenario(obs, seed, length=2400, cadence=CADENCE):
    """As the engine samples: every command frame, plus each command-free cadence frame.

    Busy and silent stretches alternate.
    """
    rng = random.Random(seed)
    rows, busy = [], True
    for frame in range(length+1):
        if frame % 300 == 0:
            busy = rng.random() < .6
        draw = rng.random()
        if frame and draw < (.08 if busy else .003):
            kind = rng.random()
            if kind < .7:
                rows.append(raw_row(frame, obs, [{'kind': 'stop', 'units': [10]}]))
            elif kind < .9:
                rows.append(raw_row(frame, obs, ui=True))
            else:
                rows.append(raw_row(frame, obs, unsupported=True))
        elif frame % cadence == 0:
            rows.append(raw_row(frame, obs))
    rows[-1]['terminal_sample'], rows[-1]['hold_ticks'] = True, None
    return rows


def commanded(row):
    return bool(row['actions'] or row['unsupported'] or not row['imitation_loss_mask'])


def simulated_export(policy, rows, rate=60):
    """The engine's capture loop (TrainingExporter quietObservationWanted) with the controller answering none:
    only the first row and command rows keep their observation."""
    sparse = []
    for index, row in enumerate(rows):
        observed = index == 0 or commanded(row)
        kept = dict(row)
        if not observed:
            del kept['observation']
        sparse.append(kept)
    return sparse


class ScheduledCaptureTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture()
        self.config['snapshot_rate'] = .5
        self.policy = NeuralPolicy(self.config, 5)

    def events(self, rows):
        game = {'rows': rows, 'map': self.terrain, 'source_ticks_per_second': 60, 'player': 2, 'replay_sha256': RECORDING}
        return list(encoded_events(self.policy, game, Counter()))

    def assert_same_events(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertEqual((a['frame'], a['target']), (b['frame'], b['target']))
            self.assertEqual(sorted(a['encoded']), sorted(b['encoded']))
            for name, value in a['encoded'].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, b['encoded'][name], err_msg=name)

    def test_sparse_rows_prepare_exactly_like_every_row(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                rows = scenario(self.obs, seed)
                sparse = simulated_export(self.policy, rows)
                self.assertFalse(any(e['target']['kind'] == 'wait' for e in self.events(rows)))
                omitted = sum('observation' not in row for row in sparse)
                self.assertGreater(omitted, len(rows)//3)
                self.assert_same_events(self.events(rows), self.events(sparse))

    def test_thirty_hertz_sparse_rows_prepare_exactly_like_every_row(self):
        # Retail 1.04 runs at 30 Hz; preparation normalizes its rows to 60 Hz frames.
        for seed in range(4):
            with self.subTest(seed=seed):
                rows = scenario(self.obs, seed, length=1200, cadence=CADENCE//2)
                sparse = simulated_export(self.policy, rows, rate=30)
                omitted = sum('observation' not in row for row in sparse)
                self.assertGreater(omitted, len(rows)//3)
                self.assert_same_events(self.events([normalize_demonstration(r, 30) for r in rows]),
                                        self.events([normalize_demonstration(r, 30) for r in sparse]))

    def test_controller_answers_no_quiet_frames_and_still_checks_requests(self):
        controller = object.__new__(CaptureController)
        controller.config, controller.replay_sha256 = self.config, RECORDING
        controller.commands, controller.cadence = {2: {12, 13}}, 6
        controller.header, controller.error = {'ticks_per_second': 60}, None
        controller.last_frame, controller.ended, controller.dispatched = {2: 0}, set(), 0
        controller.ready, controller.asking = threading.Condition(), False
        with tempfile.TemporaryDirectory() as directory:
            controller.file = GrowingFile(Path(directory)/'absent.part', threading.Event())
            self.assertEqual(controller.directive(2, 6), [])
            for wrong in (12, 7, 0):  # A command frame, off the grid, already written.
                with self.assertRaises(ValueError):
                    controller.directive(2, wrong)

    def test_omitting_a_selected_command_row_fails_loudly(self):
        rows = scenario(self.obs, 1)
        selected = sorted({e['frame'] for e in self.events(rows)})[1]
        broken = [({k: v for k, v in row.items() if k != 'observation'} if row['frame'] == selected and i else row)
                  for i, row in enumerate(rows)]
        with self.assertRaisesRegex(ValueError, 'omitted a selected observation'):
            self.events(broken)

    def test_masks_match_the_pair_matrices(self):
        obs = copy.deepcopy(self.obs)
        obs['own_entities'][0].update(dock_targets=[20, 99, 100, 10, 404], enter_targets=[20, 100], repair_targets=[99, 100],
                                      repair_depots=[100, 10], resume_targets=[10, 404])
        obs['own_entities'][1].update(dock_targets=[10], enter_targets=[10], construction_percent=50)
        policy = NeuralPolicy(self.config, parameters=False)
        encoded = policy.encoder.encode(obs, self.terrain, 60)
        for kind, matrix in (('dock', policy.dock_matrix(encoded)), ('enter', policy.enter_matrix(encoded)),
                             ('repair', policy.pair_matrix(encoded, 'repair')),
                             ('get_repaired', policy.pair_matrix(encoded, 'get_repaired')),
                             ('resume_construction', policy.resume_matrix(encoded))):
            np.testing.assert_array_equal(policy.actor_mask(encoded, kind), encoded['own'] & matrix.any(axis=1), err_msg=kind)

    def test_growing_file_reads_across_writes_and_its_read_ahead(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'export.part'
            finished = threading.Event()
            reader = GrowingFile(path, finished)
            reader.READ_AHEAD = 7
            payload = bytes(range(256))*4
            with path.open('wb') as stream:
                stream.write(payload[:100]); stream.flush()
                self.assertEqual(reader.read(10), payload[:10])
                self.assertEqual(reader.read(90), payload[10:100])
                self.assertEqual(reader.tell(), 100)
                result = []
                thread = threading.Thread(target=lambda: result.append(reader.read(500)))
                thread.start()
                stream.write(payload[100:400]); stream.flush()
                time.sleep(.1)
                self.assertTrue(thread.is_alive())
                stream.write(payload[400:]); stream.flush()
                reader.more.set()
                thread.join(10)
                self.assertEqual(result, [payload[100:600]])
            finished.set()
            self.assertEqual(reader.read(10000), payload[600:])
            self.assertEqual(reader.read(10), b'')
            self.assertEqual(reader.tell(), len(payload))
            reader.close()


if __name__ == '__main__':
    unittest.main()
