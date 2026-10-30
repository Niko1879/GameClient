"""Scheduled quiet capture keeps every observation the event schedule selects."""
from collections import Counter
import random
import unittest

import numpy as np

from generals_policy import NeuralPolicy
from replay_events import decisions, encoded_events
from scheduled_capture import PlayerSchedule, wanted_frames
from test_fixtures import fixture, raw_row

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


def simulated_export(policy, terrain, rows):
    """The engine's capture loop: quiet rows keep observations only on request."""
    quiet_frames = {r['frame'] for r in rows if not commanded(r)}
    schedule = PlayerSchedule(2, policy, {'map': terrain, 'source_ticks_per_second': 60, 'player': 2})
    schedule.start()
    sparse, stale, wanted = [], True, []
    for index, row in enumerate(rows):
        observed = index == 0 or commanded(row)
        if not observed:
            if stale:
                schedule.settled(30)
                wanted = wanted_frames(row['frame'], schedule.last_decision, schedule.done or schedule.finished,
                                       policy.config['event_max_delay'], CADENCE,
                                       lambda f: f in quiet_frames or f > rows[-1]['frame'])
                stale = False
            observed = row['frame'] in wanted
        stale = stale or observed
        kept = dict(row)
        if not observed:
            del kept['observation']
        sparse.append(kept)
        schedule.put(kept)
    schedule.close(); schedule.join(30)
    if schedule.error is not None:
        raise schedule.error
    return sparse


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
                sparse = simulated_export(self.policy, self.terrain, rows)
                omitted = sum('observation' not in row for row in sparse)
                self.assertGreater(omitted, len(rows)//3)
                self.assert_same_events(self.events(rows), self.events(sparse))

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
