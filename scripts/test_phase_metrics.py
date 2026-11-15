"""Validation is reported by game phase as well as overall."""
from collections import Counter
import unittest

from train_events import phase, rating_band, summarize


class PhaseMetricTests(unittest.TestCase):
    def test_source_frames_map_to_game_minute_buckets(self):
        self.assertEqual(phase({'frame': 0, 'source_ticks_per_second': 60}), '00-05')
        self.assertEqual(phase({'frame': 60*60*12, 'source_ticks_per_second': 60}), '10-15')
        self.assertEqual(phase({'frame': 30*60*12, 'source_ticks_per_second': 30}), '10-15')
        self.assertEqual(phase({'frame': 60*60*45, 'source_ticks_per_second': 60}), '30+')

    def test_summary_reports_each_phase_and_keeps_overall_metrics(self):
        counts = Counter({'decisions': 3, 'correct': 2, 'loss_sum': 6.0, 'commands': 1, 'commands_correct': 1,
                          'minute:00-05:decisions': 2, 'minute:00-05:loss_sum': 2.0, 'minute:00-05:correct': 2,
                          'minute:10-15:decisions': 1, 'minute:10-15:loss_sum': 4.0, 'minute:10-15:correct': 0})
        summary = summarize(counts)
        self.assertEqual(summary['mean_nll'], 2.0)
        self.assertEqual(summary['by_minute'], {'00-05': {'decisions': 2, 'mean_nll': 1.0, 'decision_accuracy': 1.0},
                                                '10-15': {'decisions': 1, 'mean_nll': 4.0, 'decision_accuracy': 0.0}})
        self.assertFalse(any(key.startswith('minute:') for key in summary))


    def test_ratings_map_to_the_validation_bands(self):
        self.assertEqual([rating_band(r) for r in (None, 900, 1100, 1399, 1400, 1699.5, 1700, 2400)],
                         ['unrated', '<1100', '1100-1400', '1100-1400', '1400-1700', '1400-1700', '1700+', '1700+'])

    def test_learner_reports_each_acting_players_rating_band(self):
        import tempfile
        from pathlib import Path
        from generals_policy import NeuralPolicy
        from test_event_policy import EventTests
        from train_events import EventLearner
        from training_preparation import PreparationSpec, prepare_update
        fixture = EventTests('setUp'); fixture.setUp()
        with tempfile.TemporaryDirectory() as temporary:
            data = fixture.shard_dataset(Path(temporary))
            data.entries[0]['ratings'] = {'2': 1750, '3': 1200}  # The fixture's rows are player 2's.
            spec = PreparationSpec(data.config, data.key, device_type='cpu')
            learner = EventLearner(NeuralPolicy(data.config, 5))
            counts = Counter()
            for rows, _ in data.batches('train', seed=3, epoch=0, batch_size=4):
                self.assertTrue(all(row['rating'] == 1750 for row in rows))
                counts.update(learner.run(prepare_update(spec, rows, 2_000_000), train=False))
        summary = summarize(counts)
        self.assertEqual(list(summary['by_rating']), ['1700+'])
        self.assertEqual(summary['by_rating']['1700+']['decisions'], summary['decisions'])
        self.assertAlmostEqual(summary['by_rating']['1700+']['mean_nll'], summary['mean_nll'], places=6)  # float32 sums
        self.assertFalse(any(key.startswith('rating:') for key in summary))

    def test_validation_reports_distance_to_click_and_training_skips_it(self):
        from generals_policy import NeuralPolicy
        from test_fixtures import fixture
        from test_policy import examples
        from train_events import EventLearner
        from training_preparation import PreparationSpec, prepare_update
        config, obs, terrain = fixture('fast')
        policy = NeuralPolicy(config, 5)
        rows = [{**row, 'frame': None} for row in examples(policy, obs, terrain)]
        spec = PreparationSpec(config, 'test', device_type='cpu', reference=policy)
        learner = EventLearner(policy)
        validation, training = Counter(), Counter()
        positions = Counter(row['target']['kind'] for row in rows if 'x' in row['target'])
        for batch in (rows[:6], rows[6:]):
            validation.update(learner.run(prepare_update(spec, batch, 2_000_000), train=False))
            training.update(learner.run(prepare_update(spec, batch, 2_000_000), train=True))
        self.assertTrue(positions, 'the fixture should have position orders')
        summary = summarize(validation)
        self.assertEqual({kind: v['orders'] for kind, v in summary['position_error']['by_kind'].items()}, dict(positions))
        for value in summary['position_error']['by_kind'].values():
            self.assertGreaterEqual(value['mean_expected_pixels'], 0)
            self.assertTrue(0 <= value['mode_within_3'] <= value['mode_within_10'] <= 1)
        self.assertFalse(any(key.startswith(('position:', 'position_length:')) for key in summary))
        self.assertNotIn('position_error', summarize(training))


if __name__ == '__main__':
    unittest.main()
