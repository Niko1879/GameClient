"""No-op rows in existing shards (notes/wait_snapshots.md): thinning, live looks, the pilot rules.

New exports hold no no-op rows (26 Nov feature review); their drawing and the hidden clocks are gone."""
import copy
import json
import math
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from event_dataset import APM_MIN_COMMANDS, EventDataset, apm_blanked, episode_apm, thinned_rows
from event_policy import EventActor
from event_store import write_shard
from generals_policy import KINDS, SCALAR_FEATURES, NeuralPolicy, apm_inputs, scalar_input_slices
from nn_numpy import Tensor, no_grad
from prepared_io import _write_json
from replay_events import decisions, encoded_events
from subset_manifest import subset
from test_fixtures import RECORDING, fixture, raw_row

STOP = [{'kind': 'stop', 'units': [10]}]


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture()
        self.policy = NeuralPolicy(self.config, 5)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, rows, rate=60, **document):
        game = {'rows': rows, 'map': self.terrain, 'source_ticks_per_second': rate, 'player': 2, 'replay_sha256': RECORDING}
        result = write_shard(self.root/'a.zip', [game], self.policy, {'match_key': 'test:1', 'path': 'absent'})
        _write_json(self.root/'dataset.json', {'kind': 'generals_event_dataset', 'schema': 1, 'config': self.config,
            'entries': [{**result, 'path': 'a.zip', 'match_key': 'test:1', 'split': 'train'}], **document})
        return EventDataset(self.root/'dataset.json')

    def rows(self, count=400):
        return [raw_row(6*i, self.obs, STOP if i % 3 == 1 else ()) for i in range(count)]

    def test_thinning_is_independent_of_what_the_rows_hold(self):
        descriptor = lambda bits: {'rows': 64, 'start': 128, 'noops': np.packbits(bits).tobytes().hex()}
        every = thinned_rows('m', 2, descriptor(np.ones(64, bool)), .5)
        mixed = np.arange(64) % 2 == 0
        np.testing.assert_array_equal(thinned_rows('m', 2, descriptor(mixed), .5)[mixed], every[mixed])
        self.assertIsNone(thinned_rows('m', 2, descriptor(mixed), 1.))

    def test_a_dataset_can_only_thin(self):
        with self.assertRaisesRegex(ValueError, 'only thin'):
            self.dataset(self.rows(10), snapshot_rate=11)
        document = {'config': self.config, 'entries': []}
        self.assertEqual(subset(document, snapshot_rate=.25)['snapshot_rate'], .25)
        self.assertNotIn('snapshot_rate', subset(document))

    def test_apm_is_the_players_own_pace_and_blanked_by_hash(self):
        data = self.dataset(self.rows(600))
        episode = data.entries[0]['index']['episodes'][0]
        self.assertEqual((episode['commands'], episode['ticks']), (200, 599*6))
        expected = 200*3600/(599*6)
        for blank in (False, True):
            with patch('event_dataset.apm_blanked', return_value=blank):
                row = EventDataset(self.root/'dataset.json').episode_rows('train', 0, 0, 1)[0]
            self.assertAlmostEqual(row['apm'], expected)  # Reporting keeps the true pace.
            np.testing.assert_allclose(row['encoded']['apm'], apm_inputs(None if blank else expected))
        self.assertIsNone(episode_apm({'commands': APM_MIN_COMMANDS-1, 'ticks': 10_000}))
        self.assertAlmostEqual(np.mean([apm_blanked(f'm{i}', 2) for i in range(4000)]), .15, delta=.02)

    def test_thirty_hertz_rows_use_canonical_seconds(self):
        # Stored frames are canonical 60 Hz: the opener window closes at 240 s for both rates.
        rows = [raw_row(f, self.obs, STOP) for f in (0, 239*60, 241*60)]
        for rate in (30, 60):
            data = self.dataset(rows, rate)
            document = json.loads((self.root/'dataset.json').read_text())
            document['entries'][0]['openers'] = {'2': {'templates': [0]*6}}
            _write_json(self.root/'dataset.json', document)
            with patch('event_dataset.blanked', return_value=False):
                stored = EventDataset(self.root/'dataset.json').episode_rows('train', 0, 0, 10)
            self.assertEqual([r['opener'] for r in stored if r['target']['kind'] != 'wait'], ['given', 'given', 'after'])


class PilotRuleTests(unittest.TestCase):
    """noop_pilot.report on simulated players: a constant-rate sample passes, each planted leak fails its own rule."""
    def rows(self, seed=1, episodes=60, minutes=12):
        rng = np.random.default_rng(seed)
        values = {name: [] for name in ('match', 'frame', 'noop', 'since', 'situation', 'apm', 'p', 'odds')}
        for episode in range(episodes):
            apm, situation, last = rng.uniform(30, 80), 2, -np.inf
            for frame in range(0, minutes*3600, 6):
                if rng.random() < .01:
                    situation = int(rng.integers(0, 3))
                hazard = apm/60*(1.4, 1., .6)[situation]  # Commands per second.
                draw, offset = rng.random(), int(rng.integers(0, 6))
                acts = draw < hazard*.1
                if not acts and rng.random() >= .05:
                    continue
                at = frame+offset if acts else frame
                odds = hazard/.5
                for name, value in (('match', f'm{episode}'), ('frame', at), ('noop', not acts), ('since', (at-last)/60),
                                    ('situation', situation), ('apm', apm), ('p', 1/(1+odds)), ('odds', odds)):
                    values[name].append(value)
                last = at if acts else last
        rows = {name: np.asarray(value) for name, value in values.items()}
        logp = np.log(rows.pop('p'))
        odds = rows.pop('odds')
        rows.update(memory=logp, reset=logp.copy())
        for apm in (40, 80):
            rows[f'apm{apm}'] = -np.log1p(odds*apm/rows['apm'])
        return rows

    def test_a_calibrated_leak_free_model_passes_every_rule(self):
        from noop_pilot import report
        result = report(self.rows())
        self.assertTrue(result['passed'], result)
        self.assertEqual(len(result['player_pace']), 3)
        self.assertAlmostEqual(result['apm_input']['rate_ratio_80_over_40'], 2, places=6)
        self.assertFalse(report({**self.rows(), 'apm40': self.rows()['memory'], 'apm80': self.rows()['memory']})['apm_input']['works'])

    def test_each_planted_fault_fails_its_rule(self):
        from noop_pilot import report
        rows = self.rows()
        sharper = {**rows, 'memory': np.where(rows['noop'], -.02, rows['memory']-1.5)}  # Memory knows the row type.
        self.assertFalse(report(sharper)['rule1_memory_leak']['passed'])
        leaky = {**rows, 'memory': np.where(rows['frame'] % 6 == 0, rows['memory']+.1, rows['memory'])}
        self.assertFalse(report(leaky)['rule2_grid_leak']['passed'])
        constant = np.full(len(rows['noop']), np.log(rows['noop'].mean()))
        self.assertFalse(report({**rows, 'memory': constant})['rule3_signal']['passed'])
        self.assertFalse(report({**rows, 'memory': constant})['rule4_calibration']['passed'])
        # Exaggerated or flattened paces, right on average: only the extremes show it.
        p = np.exp(rows['memory'])
        logit = np.log(p/(1-p))
        for slope in (.4, 2.):
            shifts = np.linspace(-4, 4, 801)
            means = [(1/(1+np.exp(-(slope*logit+s)))).mean() for s in shifts]
            fitted = slope*logit+shifts[int(np.argmin(np.abs(np.array(means)-rows['noop'].mean())))]
            self.assertFalse(report({**rows, 'memory': -np.logaddexp(0, -fitted)})['rule5_extremes']['passed'], slope)

    def test_scoring_a_dataset_matches_the_reference_policy(self):
        from noop_pilot import score
        from test_event_policy import EventTests
        case = EventTests('setUp'); case.setUp()
        with tempfile.TemporaryDirectory() as temporary:
            data = case.shard_dataset(Path(temporary), validation=True)
            got = score(data, case.policy, 'cpu', rows_at_once=4)
            rows = data.episode_rows('validation', 0, 0, 100)
        state, fresh = None, []
        for index, row in enumerate(rows):
            mask = case.policy.action_mask(row['encoded'], row['encoded']['_training_legality']['verb'])
            def noop(state):
                from generals_policy import cap_logits
                from nn_numpy import categorical
                verb = row['encoded']['_training_legality']['verb']
                with no_grad(), patch.object(case.policy, 'verb_mask', return_value=verb):
                    trunk = case.policy._trunk(self.entities(row['encoded']), state)
                    return float(categorical(cap_logits(case.policy._action_logits(trunk), mask), mask, choice=0)[1].data), trunk['state']
            logp, state = noop(state)
            self.assertAlmostEqual(got['memory'][index], logp, places=4)
            self.assertAlmostEqual(got['reset'][index], noop(None)[0], places=4)
        self.assertEqual(got['noop'].tolist(), [row['target']['kind'] == 'wait' for row in rows])

    @staticmethod
    def entities(encoded):
        # Stored rows keep no entity list; the reference network reads only its length and owners.
        owners = encoded['_training_legality']['owners']
        return {**encoded, 'entities': [{'owner': int(owner)} for owner in owners[:encoded['entity_count']]]}


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture()
        self.policy = NeuralPolicy(self.config, 5)
        self.policy.snapshot_rate = .5
        self.encoded = self.policy.encoder.encode(self.obs, self.terrain, 0)

    def test_a_look_acts_with_the_orders_expected_in_its_time(self):
        with no_grad():
            teacher = self.policy.decide(self.encoded, teacher={'kind': 'wait'})
            noop = math.exp(float(teacher[4].data))
            odds, rng = (1-noop)/noop, np.random.default_rng(1)
            # Up to certainty, at one order per look; without a look time, the training softmax itself.
            for snapshots, expected in ((None, 1-noop), (.5/odds, .5), (.05/odds, .05), (.999/odds, .999), (3/odds, 1.)):
                looks = [self.policy.decide(self.encoded, rng=rng, snapshots=snapshots) for _ in range(40)]
                for actions, trace, *_, logp, entropy in looks:
                    self.assertAlmostEqual(trace['act_probability'], expected, places=5)
                    self.assertEqual(bool(actions), trace['kind'] != 'wait')
                    self.assertTrue(np.isfinite(logp.data) and np.isfinite(entropy.data))
                    if not actions:
                        self.assertAlmostEqual(float(logp.data), math.log(1-expected), places=4)

    def test_the_complement_and_its_gradient(self):
        from generals_policy import complement
        for value in (-1e-3, -.5, -8.):
            logp = Tensor(np.array([value]), requires_grad=True)
            result = complement(logp)
            self.assertAlmostEqual(float(result.data[0]), math.log(1-math.exp(value)), places=9)
            result.sum().backward()
            step = 1e-7
            slope = (math.log(1-math.exp(value+step))-math.log(1-math.exp(value-step)))/(2*step)
            self.assertAlmostEqual(float(logp.grad[0])/slope, 1, places=4)

    def test_deterministic_play_still_samples_when_to_act(self):
        rng = np.random.default_rng(3)
        with no_grad():
            kinds = [self.policy.decide(self.encoded, rng=rng, deterministic=True, snapshots=.02)[1]['kind'] for _ in range(200)]
        self.assertIn('wait', kinds)
        self.assertEqual(len(set(kinds)-{'wait'}), 1)  # What to do is the argmax.

    def actor(self, *, act, look_ms=200):
        """An actor whose policy acts when act(frame); records each look's expected snapshots and memory."""
        actor, looks = EventActor(self.policy, 7), []
        actor.configure_look(look_ms)
        def decide(encoded, memory, *, rng, deterministic, snapshots):
            looks.append((snapshots, memory))
            acts = act(len(looks))
            return (copy.deepcopy(STOP) if acts else [], {'kind': 'stop' if acts else 'wait'}, len(looks),
                    Tensor(0), Tensor(0), Tensor(0))
        return actor, looks, decide

    def run_looks(self, actor, decide, frames):
        with patch.object(self.policy, 'decide', side_effect=decide), \
                patch.object(self.policy.encoder, 'encode', return_value={}):
            for frame in frames:
                actor.act(self.obs, self.terrain, frame)

    def test_the_expected_snapshots_follow_the_time_since_the_last_look(self):
        actor, looks, decide = self.actor(act=lambda n: False)
        self.run_looks(actor, decide, [0, 6, 12, 24, 48])  # 6 is skipped: not due yet.
        self.assertEqual([round(snapshots, 6) for snapshots, _ in looks], [.1, .1, .1, .2])

    def test_memory_commits_at_orders_and_at_the_training_rate_otherwise(self):
        actor, looks, decide = self.actor(act=lambda n: True)
        self.run_looks(actor, decide, range(0, 120, 12))
        self.assertEqual([memory for _, memory in looks], [None, *range(1, 10)])
        for look_ms in (100, 200):
            actor, looks, decide = self.actor(act=lambda n: False, look_ms=look_ms)
            ticks = round(look_ms*.06)
            self.run_looks(actor, decide, range(0, ticks*8000, ticks))
            self.assertEqual(len(looks), 8000)
            # 0.5 commits per second of game time, whatever the look interval.
            self.assertAlmostEqual(actor.counters['memory_commits']/(8000*look_ms/1000), .5, delta=.05)

    def test_the_pace_does_not_depend_on_the_look_interval(self):
        # One action rate h a second, as training would show it: P(act)/P(no-op) = h/r. Through the real decide().
        for hazard, looks in ((.2, (50, 100, 200)), (.85, (50, 100, 200)), (2., (50, 100, 200)), (4., (100, 200)), (20., (200,))):
            logits = np.full(len(self.policy.actions), -60.)
            logits[self.policy.action_index[(0, -1)]] = 0
            mask = self.policy.action_mask(self.encoded) & self.policy.exact_action_mask(self.encoded)
            stop = next(i for i in np.flatnonzero(mask) if self.policy.actions[i, 0] == KINDS.index('stop'))
            logits[stop] = math.log(hazard/self.policy.snapshot_rate)
            for look_ms in looks:
                with self.subTest(hazard=hazard, look_ms=look_ms):
                    actor, minutes = EventActor(self.policy, 9), 12
                    actor.configure_look(look_ms)
                    with patch.object(self.policy, '_trunk', return_value={'state': None, 'value': Tensor(0)}), \
                            patch.object(self.policy, '_action_logits', return_value=Tensor(logits)), \
                            patch.object(self.policy, 'action_mask', return_value=mask), \
                            patch.object(self.policy, 'exact_action_mask', return_value=mask), \
                            patch.object(self.policy, '_unit_begin', side_effect=lambda *_: None), \
                            patch.object(self.policy, '_unit_scores', side_effect=lambda *_: (Tensor(np.zeros(len(self.encoded['own'])+1)), None)), \
                            patch.object(self.policy, '_unit_add', side_effect=lambda *_: None), \
                            patch.object(self.policy, '_selection_summary', return_value=None), \
                            patch.object(actor, '_encode', return_value=self.encoded):
                        for frame in range(0, 3600*minutes, 3):
                            actor.act(self.obs, self.terrain, frame)
                    # 20/s cannot be reached: one order per look.
                    expected = min(hazard*60, 60_000/look_ms)
                    # Four standard deviations of the count, at least 5%.
                    self.assertAlmostEqual(actor.counters['commands']/minutes/expected, 1,
                                           delta=max(.05, 4/math.sqrt(expected*minutes)))

    def test_the_apm_input_reaches_the_network_and_the_actor_sets_it(self):
        self.assertFalse(apm_inputs(None).any())
        np.testing.assert_allclose(apm_inputs(100), [math.log(2), 1])
        for wrong in (0, -1, 601):
            with self.assertRaises(ValueError):
                apm_inputs(wrong)
        rows = scalar_input_slices(self.config)['apm']
        self.assertEqual(rows.stop-rows.start, 2)
        with no_grad():
            values = [self.policy._scalars({**self.encoded, 'apm': apm_inputs(apm)}).data for apm in (None, 40, 80)]
        self.assertFalse(np.array_equal(values[1], values[2]))
        self.assertFalse(np.array_equal(values[0], values[1]))
        actor = EventActor(self.policy)
        actor.configure_apm(75)
        actor.reset()
        with patch.object(self.policy, 'decide', return_value=([], {'kind': 'wait'}, None, Tensor(0), Tensor(0), Tensor(0))) as decide:
            actor.act(self.obs, self.terrain, 0)
        np.testing.assert_allclose(decide.call_args[0][0]['apm'], apm_inputs(75))

    def test_a_checkpoint_carries_its_training_rows_snapshot_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'a.npz'
            self.policy.save(path, metadata={'event_snapshot_rate': .25})
            self.assertEqual(NeuralPolicy.load(path)[0].snapshot_rate, .25)
            self.policy.save(path)
            self.assertEqual(NeuralPolicy.load(path)[0].snapshot_rate, self.config['snapshot_rate'])


if __name__ == '__main__':
    unittest.main()
