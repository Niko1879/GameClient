"""game_stats.SideStats: per-player head-to-head stats gathered step by step."""
import unittest

from game_stats import SideStats


def observation(cash, *entities):
    return {'player': {'cash': cash}, 'own_entities': [dict(id=i, type=t, **extra) for i, t, extra in entities]}


class SideStatsTests(unittest.TestCase):
    def test_counts_orders_income_dozers_production_and_the_command_center(self):
        stats = SideStats(2)
        start = observation(1000, (1, 'ChinaCommandCenter', {}), (2, 'ChinaVehicleDozer', {}))
        built = observation(1500, (1, 'ChinaCommandCenter', {}), (2, 'ChinaVehicleDozer', {}),
                            (3, 'ChinaVehicleDozer', {'producer_id': 1}))
        sold = observation(4000, (2, 'ChinaVehicleDozer', {}), (3, 'ChinaVehicleDozer', {'producer_id': 1}))
        sell = {'player': 2, 'accepted': True, 'action': {'kind': 'sell'}}
        stats.update({'2': start}, {'2': built}, 3600, [], {2: [{'kind': 'train'}, {'kind': 'rally'}]})
        # A trajectory read back from JSON keys the actions by the player's string.
        stats.update({'2': built}, {'2': sold}, 7200, [sell], {'2': [{'kind': 'sell'}]})
        result = stats.result()
        self.assertEqual(result['orders'], {'train': 1, 'rally': 1, 'sell': 1})
        self.assertEqual(result['income_proxy'], 500)  # The sale's cash is not income.
        self.assertEqual((result['dozers_ever'], result['max_dozers'], result['units_produced']), (2, 2, 1))
        self.assertEqual((result['no_cc_from_min'], result['no_dozer_from_min']), (2.0, None))


if __name__ == '__main__':
    unittest.main()
