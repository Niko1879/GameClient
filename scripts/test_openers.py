"""Strategy input z (scripts/openers.py): extraction from shard headers, the window, blanking, live play."""
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import openers
from event_dataset import EventDataset
from event_policy import EventActor
from event_store import feature_identity, write_shard
from generals_policy import OPENER_BUILDINGS, OPENER_WINDOW_SECONDS, NeuralPolicy, opener_inputs, scalar_input_slices
from nn_numpy import Tensor, no_grad
from prepared_io import _write_json
from test_fixtures import RECORDING, fixture, raw_row
from train_events import EventLearner
from training_legality import build_legality

ROOT = Path(__file__).resolve().parents[1]


def build(template, frame, tps=60):
    return {'target': {'kind': 'build', 'template': template}, 'frame': frame, 'source_ticks_per_second': tps}


def chunk(rows):
    header = json.dumps({'rows': [{'row': row} for row in rows]}).encode()
    return struct.pack('<Q', len(header))+header


class OpenerExtractionTests(unittest.TestCase):
    def test_first_six_builds_with_reclicks_counted_once(self):
        rows = [build(12, 0), build(15, 600), build(15, 660), build(15, 1200), {'target': {'kind': 'wait'}, 'frame': 1300},
                build(1, 1500), build(28, 2000), build(28, 2400), build(0, 3000), build(2, 3600)]
        orders = openers.build_orders(rows)
        self.assertEqual([t for t, _ in orders], [12, 15, 15, 1, 28, 28])  # 660 is a re-click of 600
        self.assertEqual(orders[-1][1], 40.0)

    def test_retail_rows_count_seconds_in_stored_canonical_frames(self):
        # A 30 Hz recording is stored at 60 ticks/s too: 120 ticks apart is 2 s, a re-click; 240 apart is 4 s.
        orders = openers.build_orders([build(15, 0, 30), build(15, 120, 30), build(15, 360, 30)])
        self.assertEqual([round(s) for _, s in orders], [0, 6])

    def test_shard_headers_across_chunks_and_short_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'a.zip'
            first = [dict(build(t, 600*i), player=2) for i, t in enumerate((12, 15, 15))]
            # The second chunk starts with a re-click of the last supply centre.
            second = [dict(build(15, 1250), player=2)]+[dict(build(t, 2000+600*i), player=2) for i, t in enumerate((1, 28, 28, 0))]
            short = [dict(build(12, 0), player=3)]
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('c/0', chunk(first)); archive.writestr('c/1', chunk(second)); archive.writestr('c/2', chunk(short))
            index = {'episodes': [{'player': 2, 'chunks': [{'member': 'c/1', 'start': 3}, {'member': 'c/0', 'start': 0}]},
                                  {'player': 3, 'chunks': [{'member': 'c/2', 'start': 0}]}]}
            result = openers.shard_openers(path, index)
        self.assertEqual(result['2']['templates'], [12, 15, 15, 1, 28, 28])
        self.assertIsNone(result['3'], 'under six buildings in the whole game: no opener')

    def test_inputs_window_and_validation(self):
        config = {'templates': ['A', 'B', 'C']}
        values = opener_inputs(config, [0, 0, 0, 1, 1, 2], seconds=10)
        np.testing.assert_allclose(values, [3/6, 2/6, 1/6, 1])
        self.assertFalse(opener_inputs(config, [0]*6, seconds=OPENER_WINDOW_SECONDS).any())
        self.assertFalse(opener_inputs(config, None).any())
        with self.assertRaises(ValueError):
            opener_inputs(config, [0]*5)

    def test_blanking_is_per_player_game_and_about_half(self):
        self.assertEqual(openers.blanked('pgo:1', 2), openers.blanked('pgo:1', 2))
        share = np.mean([openers.blanked(f'pgo:{i}', 2) for i in range(4000)])
        self.assertAlmostEqual(share, openers.OPENER_BLANK, delta=.03)

    def test_named_openers_resolve_against_the_policy(self):
        config = json.loads((ROOT/'replay-runtime/policy.json').read_text())
        for name in openers.NAMED:
            templates = openers.named_templates(config, name)
            self.assertEqual(len(templates), OPENER_BUILDINGS, name)
        self.assertIsNone(openers.named_templates(config, 'none'))
        with self.assertRaises(ValueError):
            openers.named_templates(config, 'four_rax')


class OpenerModelTests(unittest.TestCase):
    def setUp(self):
        self.config, self.obs, self.terrain = fixture()
        self.policy = NeuralPolicy(self.config, 7)
        self.plan = [0, 0, 1, 1, 1, 1]  # fixture templates: power plant, dozer

    def test_dataset_rows_carry_z_only_in_the_window_and_when_not_blanked(self):
        rows = [raw_row(f, self.obs, [{'kind': 'stop', 'units': [10]}]) for f in range(30, 331, 30)]
        game = {'rows': rows, 'map': self.terrain, 'source_ticks_per_second': 60, 'player': 2, 'replay_sha256': RECORDING}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = write_shard(root/'a.zip', [game], self.policy, {'match_key': 'test:1', 'path': 'absent'})
            entries = []
            for match in ('test:shown', 'test:hidden'):
                entries.append({**result, 'path': 'a.zip', 'match_key': match, 'split': 'train',
                                'openers': {'2': {'templates': self.plan, 'seconds': [0]*6}}})
            _write_json(root/'dataset.json', {'kind': 'generals_event_dataset', 'schema': 1, 'config': self.config,
                                              'entries': entries})
            with patch('event_dataset.blanked', side_effect=lambda match, player: match == 'test:hidden'), \
                    patch('event_dataset.OPENER_WINDOW_SECONDS', 3):  # frames 30..330 at 60/s: 0.5..5.5 s
                dataset = EventDataset(root/'dataset.json')
                stored = {}
                for reference in dataset.chunks('train'):
                    for row in dataset.read(reference):
                        stored.setdefault(row['match_key'], []).append(row)
        shown = stored['test:shown']
        self.assertEqual([r['opener'] for r in shown], ['given']*5+['after']*6)
        np.testing.assert_allclose(shown[0]['encoded']['opener'], [2/6, 4/6, 1])
        self.assertFalse(shown[-1]['encoded']['opener'].any())
        self.assertEqual({r['opener'] for r in stored['test:hidden']}, {'blanked', 'after'})
        self.assertFalse(any(r['encoded']['opener'].any() for r in stored['test:hidden']))

    def test_numpy_and_torch_agree_and_the_opener_weight_learns(self):
        key = feature_identity(self.policy)
        encoded = self.policy.encoder.encode(self.obs, self.terrain, 300)
        given = {**encoded, 'opener': opener_inputs(self.config, self.plan)}
        rows, targets = [], []
        for item in (given, encoded):
            target = self.policy.target(item, [{'kind': 'stop', 'units': [10]}])
            item['_training_legality'] = build_legality(self.policy, item, target, key)
            rows.append({'encoded': item, 'target': target, 'scope': 'event'})
            targets.append(target)
        learner = EventLearner(self.policy, device=os.environ.get('TEST_TORCH_DEVICE', 'cpu'))
        learner.model.legality_key = key
        output = learner.model.teacher_heads([given, encoded], targets, *learner.model.events([given, encoded]))
        with no_grad():
            expected = [float(self.policy.decide(r['encoded'], None, teacher=t)[4].data) for r, t in zip(rows, targets)]
        np.testing.assert_allclose(output['logp'].detach().cpu().numpy(), expected, rtol=3e-6, atol=2e-5)
        self.assertNotAlmostEqual(expected[0], expected[1], places=6)
        learner.batch(rows, train=True)
        # AlphaGenerals: the opener is a row block of scalar_in_w (scalar_input_slices).
        rows = scalar_input_slices(self.policy.config)['opener']
        self.assertGreater(float(learner.model.weights['scalar_in_w'].grad[rows].abs().sum().cpu()), 0)

    def test_actor_shows_its_opener_only_in_the_window_and_keeps_it_across_games(self):
        actor = EventActor(self.policy)
        trace = ([], {'kind': 'wait'}, None, Tensor(0), Tensor(0), Tensor(0))
        with patch.object(self.policy, 'decide', return_value=trace) as decide:
            actor.act(self.obs, self.terrain, 0)
        self.assertFalse(decide.call_args[0][0]['opener'].any(), 'no opener unless one is configured')
        actor.configure_opener(self.plan)
        actor.reset()
        late = OPENER_WINDOW_SECONDS*60
        with patch.object(self.policy, 'decide', return_value=trace) as decide:
            actor.act(self.obs, self.terrain, 0)
            np.testing.assert_allclose(decide.call_args[0][0]['opener'], opener_inputs(self.config, self.plan))
            actor.next_decision_frame = 0
            actor.act(self.obs, self.terrain, late)
            self.assertFalse(decide.call_args[0][0]['opener'].any())
        with self.assertRaises(ValueError):
            actor.configure_opener([0])


if __name__ == '__main__':
    unittest.main()
