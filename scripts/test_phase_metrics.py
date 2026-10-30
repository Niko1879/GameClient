"""Validation is reported by game phase as well as overall."""
from collections import Counter
import unittest

from train_events import phase, summarize


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


if __name__ == '__main__':
    unittest.main()
