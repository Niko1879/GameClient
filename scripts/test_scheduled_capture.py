"""Scheduled quiet capture keeps every observation the event schedule selects."""
from collections import Counter
import copy
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest

import numpy as np

from generals_policy import KINDS, NeuralPolicy, Unrepresentable
from replay_events import decisions, encoded_events
from scheduled_capture import GrowingFile, PlayerSchedule, wanted_frames
from test_fixtures import entity, fixture, raw_row

CADENCE = 6


def scenario(obs, seed, length=2400):
    """As the engine samples: every command frame, plus each command-free cadence frame.

    Busy and silent stretches alternate, so silences exceed the scheduling cap.
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
        elif frame % CADENCE == 0:
            rows.append(raw_row(frame, obs))
    rows[-1]['terminal_sample'], rows[-1]['hold_ticks'] = True, None
    return rows


def commanded(row):
    return bool(row['actions'] or row['unsupported'] or not row['imitation_loss_mask'])


def simulated_export(policy, terrain, rows, every_row=False):
    """The engine's capture loop: quiet rows keep observations only on request.

    every_row is the reference: ask again after every observed row. Returns the
    rows and the number of questions asked.
    """
    quiet_frames = {r['frame'] for r in rows if not commanded(r)}
    schedule = PlayerSchedule(2, policy, {'map': terrain, 'source_ticks_per_second': 60, 'player': 2})
    schedule.start()
    sparse, current, wanted, earliest, asked = [], False, [], 0, 0
    for index, row in enumerate(rows):
        observed = index == 0 or commanded(row)
        if commanded(row):
            current = False
        if not observed:
            if current and wanted and row['frame'] > wanted[-1]:
                current = False
            if not current and (every_row or row['frame'] >= earliest):
                schedule.settled(30)
                wanted = wanted_frames(row['frame'], schedule.last_decision, schedule.done or schedule.finished,
                                       policy.config['event_max_delay'], CADENCE,
                                       lambda f: f in quiet_frames or f > rows[-1]['frame'])
                earliest = wanted[0] if wanted else float('inf')
                current, asked = True, asked+1
            observed = current and row['frame'] in wanted
            if every_row and observed:
                current = False
        kept = dict(row)
        if not observed:
            del kept['observation']
        sparse.append(kept)
        schedule.put(kept)
    schedule.close(); schedule.join(30)
    if schedule.error is not None:
        raise schedule.error
    return sparse, asked


class ScheduledCaptureTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture()
        self.policy = NeuralPolicy(self.config, 5)

    def events(self, rows):
        game = {'rows': rows, 'map': self.terrain, 'source_ticks_per_second': 60, 'player': 2}
        return list(encoded_events(self.policy, game, Counter()))

    def assert_same_events(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertEqual((a['frame'], a['hold_ticks'], a['target']), (b['frame'], b['hold_ticks'], b['target']))
            self.assertEqual(sorted(a['encoded']), sorted(b['encoded']))
            for name, value in a['encoded'].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, b['encoded'][name], err_msg=name)

    def test_sparse_rows_prepare_exactly_like_every_row(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                rows = scenario(self.obs, seed)
                sparse, _ = simulated_export(self.policy, self.terrain, rows)
                omitted = sum('observation' not in row for row in sparse)
                self.assertGreater(omitted, len(rows)//3)
                self.assert_same_events(self.events(rows), self.events(sparse))

    def test_asking_only_when_it_matters_captures_the_same_rows(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                rows = scenario(self.obs, seed)
                lazy, lazy_asked = simulated_export(self.policy, self.terrain, rows)
                eager, eager_asked = simulated_export(self.policy, self.terrain, rows, every_row=True)
                self.assertEqual(['observation' in row for row in lazy], ['observation' in row for row in eager])
                self.assertLess(lazy_asked, eager_asked)

    def test_omitting_a_selected_quiet_row_fails_loudly(self):
        rows = scenario(self.obs, 1)
        selected = {e['frame'] for e in self.events(rows) if e['target']['kind'] == 'wait'}
        broken = [({k: v for k, v in row.items() if k != 'observation'}
                   if row['frame'] in selected and not commanded(row) and i else row) for i, row in enumerate(rows)]
        with self.assertRaisesRegex(ValueError, 'omitted a selected observation'):
            self.events(broken)

    def test_observe_reports_last_decision_after_each_row(self):
        rows = [raw_row(0, self.obs), raw_row(30, self.obs, [{'kind': 'stop', 'units': [10]}]), raw_row(36, self.obs)]
        seen = []
        list(decisions(rows, self.config, Counter(), observe=lambda last, done: seen.append(last)))
        self.assertEqual(seen, [0, 30, 30])

    def test_schedule_inputs_accept_exactly_what_the_encoder_accepts(self):
        obs = copy.deepcopy(self.obs)
        obs['own_entities'][0].update(commands=['Command_Capture'],
            command_targets=[{'command': 'Command_Capture', 'targets': [99]}], dock_targets=[20], enter_targets=[20],
            repair_targets=[20], resume_targets=[20])
        obs['own_entities'].append(entity(30, 'ChinaVehicleDozer', True, [900, 900, 0], health=0))
        obs['player']['purchasable_sciences'] = ['SCIENCE_Frenzy1']
        actions = []
        for kind in KINDS[1:]:
            for units in ([10], [20], [10, 20], [30], [404]):
                actions.append({'kind': kind, 'units': units, 'position': [500, 500, 0], 'target': 99})
                actions.append({'kind': kind, 'units': units, 'position': [5000, 500, 0], 'target': 20})
            for actor in (10, 20, 30):
                actions.append({'kind': kind, 'builder': actor, 'producer': actor, 'template': 'ChinaPowerPlant',
                                'position': [500, 500, 0], 'angle': 0, 'target': 20, 'slot': 0,
                                'command': 'Command_Capture', 'units': [actor], 'science': 'SCIENCE_Frenzy1'})
                actions.append({'kind': kind, 'builder': actor, 'producer': actor, 'template': 'ChinaVehicleDozer',
                                'position': [500, 500, 0], 'target': 100, 'slot': 3,
                                'command': 'Command_UpgradeChinaRadar', 'units': [actor], 'science': 'SCIENCE_ArtilleryBarrage1'})
        policy = NeuralPolicy(self.config, parameters=False)
        accepted = 0
        # A home on the far side mirrors the full encoding; acceptance must not change.
        for home in ([100, 100], [900, 900]):
            for action in actions:
                full = policy.encoder.encode(obs, self.terrain, 60, home_position=home)
                light = policy.encoder.schedule_inputs(obs, self.terrain)
                outcomes = []
                for encoded in (full, light):
                    try:
                        policy.target(encoded, [action])
                        outcomes.append(True)
                    except (Unrepresentable, KeyError):
                        outcomes.append(False)
                self.assertEqual(outcomes[0], outcomes[1], action)
                accepted += outcomes[0]
        self.assertGreater(accepted, 10)
        self.assertLess(accepted, len(actions)*2-10)

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

    def test_wanted_frames_brackets_the_cap(self):
        quiet = lambda f: f not in (114, 120)
        # From decision 7: last quiet before 127 is 126; first at/after 127 is 132.
        self.assertEqual(wanted_frames(12, 7, False, 120, 6, quiet), [126, 132])
        # Commands occupy 114 and 120: the last quiet before 120 is 108; then 126.
        self.assertEqual(wanted_frames(12, 0, False, 120, 6, quiet), [108, 126])
        self.assertEqual(wanted_frames(30, None, False, 120, 6, quiet), [30])
        self.assertEqual(wanted_frames(30, 0, True, 120, 6, quiet), [])
        # Past the cap already: this very frame.
        self.assertEqual(wanted_frames(300, 0, False, 120, 6, quiet), [300])


if __name__ == '__main__':
    unittest.main()
