"""Experiment manifests and the 1400+ validation selection score."""
import unittest

from subset_manifest import match_rating, subset, top_fraction_rating
from train_events import selection_score


def rated(value, estimated=False):
    return {'rating': value, 'estimated': estimated}


def entry(key, split, ratings, players=(1,)):
    """`players`: the China players whose episodes the shard holds; `ratings` covers both sides."""
    return {'match_key': key, 'split': split, 'ratings': ratings,
            'index': {'episodes': [{'player': player} for player in players]}}


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.document = {'config': {'widths': {'core': 8}, 'map_side': 16}, 'selection': {'seed': 1}, 'entries': [
            entry('v-low', 'validation', {'1': rated(900), '2': rated(1000)}),
            entry('t-low', 'train', {'1': rated(900), '2': rated(2000)}),          # A strong opponent does not count.
            entry('t-high', 'train', {'1': rated(1500), '2': rated(1300)}),
            entry('t-mirror', 'train', {'1': rated(1000), '2': rated(1450, True)}, players=(1, 2)),
            entry('t-unrated', 'train', {}),
            entry('t-curated', 'train', {'1': rated(1100)})]}

    def test_match_rating_is_the_best_china_player_and_estimates_count(self):
        ratings = [match_rating(e) for e in self.document['entries']]
        self.assertEqual(ratings, [900, 900, 1500, 1450, None, 1100])

    def test_keeps_all_validation_and_filters_training(self):
        result = subset(self.document, min_rating=1400, include=['t-curated', 'v-low'])
        self.assertEqual([e['match_key'] for e in result['entries']], ['v-low', 't-high', 't-mirror', 't-curated'])
        self.assertEqual(result['selection']['subset']['counts'], {'validation': 1, 'rated': 2, 'included': 1})
        self.assertEqual(result['selection']['seed'], 1)

    def test_top_fractions_are_nested_cuts_of_the_training_matches(self):
        # Rated training matches, best first: 1500, 1450, 1100, 900.
        self.assertEqual([top_fraction_rating(self.document, f) for f in (.25, .5, .75, 1)], [1500, 1450, 1100, 900])
        kept = [{e['match_key'] for e in subset(self.document, top_fraction=f)['entries'] if e['split'] == 'train'}
                for f in (.25, .5, 1)]
        self.assertEqual(kept[0], {'t-high'})
        self.assertTrue(kept[0] < kept[1] < kept[2])
        self.assertNotIn('t-unrated', kept[2])
        self.assertEqual(subset(self.document, top_fraction=.5)['selection']['subset']['min_rating'], 1450)
        for options in (dict(top_fraction=0), dict(top_fraction=.5, min_rating=1400)):
            with self.assertRaises(ValueError):
                subset(self.document, **options)

    def test_widths_change_only_model_widths(self):
        # Only model settings (event_store.model_only): the network's 'widths' would change the feature key.
        result = subset(self.document, widths={'map_hidden': 64})
        self.assertEqual(result['config'], {'widths': {'core': 8}, 'map_side': 16, 'map_hidden': 64})
        self.assertEqual(len(result['entries']), 6)
        for bad in ({'map_side': 8}, {'widths': {'core': 32}}):
            with self.assertRaises(ValueError):
                subset(self.document, widths=bad)


class SelectionScoreTests(unittest.TestCase):
    def test_1400_plus_is_the_decision_weighted_mean_of_its_bands(self):
        validation = {'decisions': 10, 'mean_nll': 3.0, 'by_rating': {
            '<1100': {'decisions': 6, 'mean_nll': 2.0},
            '1400-1700': {'decisions': 3, 'mean_nll': 4.0},
            '1700+': {'decisions': 1, 'mean_nll': 8.0}}}
        self.assertEqual(selection_score(validation), 3.0)
        self.assertEqual(selection_score(validation, '1400+'), 5.0)
        self.assertIsNone(selection_score({'decisions': 6, 'mean_nll': 2.0,
                                           'by_rating': {'<1100': {'decisions': 6, 'mean_nll': 2.0}}}, '1400+'))
        self.assertIsNone(selection_score({'decisions': 0, 'mean_nll': 0.0, 'by_rating': {}}))


if __name__ == '__main__':
    unittest.main()
