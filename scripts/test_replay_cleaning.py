import argparse
import unittest

from replay_cleaning import Player, game_flags, listing_ratings, player_flags

DOZER, TRUCK = 444, 449
NAMES = {DOZER: 'ChinaVehicleDozer', TRUCK: 'ChinaVehicleSupplyTruck'}
OPTS = argparse.Namespace(afk_max_orders=10, early_end_minutes=5, opening_seconds=60, max_opening_dozers=3,
                          truck_minutes=6, max_trucks_per_center=1, min_china_rating=0)


def play(commands):
    player = Player()
    for frame, name, args in commands:
        player.handle(frame, name, args, set(NAMES))
    return player


def row(player, enemy='USA', minutes=20):
    china = {'slot': 0, 'player_index': 2, 'faction': 'China', 'rating': {'overall': 1200}, 'orders': player.orders,
             'self_destruct': player.self_destruct, 'opening_queue': player.opening_queue,
             'queues': [q+[q[3] in player.cancelled] for q in player.queues]}
    other = {'slot': 1, 'player_index': 3, 'faction': enemy, 'orders': 500, 'self_destruct': None}
    return {'rate': 60, 'frames': minutes*3600, 'last_command_frame': 0, 'players': [china, other]}


class CleaningTest(unittest.TestCase):
    def test_opening_queue_comes_from_the_implicit_command_center(self):
        player = play([(31, 'MSG_QUEUE_UNIT_CREATE', [DOZER, 1]),
                       (60, 'MSG_CREATE_SELECTED_GROUP', [1, 500]),
                       (90, 'MSG_QUEUE_UNIT_CREATE', [TRUCK, 2])])
        self.assertEqual(player.opening_queue, DOZER)
        self.assertEqual([q[2] for q in player.queues], ['start_cc', '500'])
        self.assertEqual(player.orders, 2)  # selections are not orders

    def test_cancelled_dozers_are_netted_out(self):
        queues = [(31, 'MSG_QUEUE_UNIT_CREATE', [DOZER, i]) for i in range(1, 6)]
        self.assertIn('opening_dozers:5', player_flags(row(play(queues)), row(play(queues))['players'][0],
                                                       OPTS, NAMES))
        cancelled = play(queues+[(40, 'MSG_CANCEL_UNIT_CREATE', [4]), (41, 'MSG_CANCEL_UNIT_CREATE', [5])])
        game = row(cancelled)
        self.assertEqual(player_flags(game, game['players'][0], OPTS, NAMES), [])

    def test_trucks_count_per_center_and_gla_is_exempt(self):
        commands = [(60, 'MSG_CREATE_SELECTED_GROUP', [1, 700]),
                    (100, 'MSG_QUEUE_UNIT_CREATE', [TRUCK, 1]), (110, 'MSG_QUEUE_UNIT_CREATE', [TRUCK, 2]),
                    (120, 'MSG_CREATE_TEAM1', []), (130, 'MSG_CREATE_SELECTED_GROUP', [1, 701]),
                    (140, 'MSG_QUEUE_UNIT_CREATE', [TRUCK, 3]), (150, 'MSG_SELECT_TEAM1', []),
                    (160, 'MSG_QUEUE_UNIT_CREATE', [TRUCK, 4])]
        player = play(commands)
        self.assertEqual([q[2] for q in player.queues], ['700', '700', '701', '700'])
        usa, gla = row(player), row(player, enemy='GLA Toxin')
        self.assertEqual(player_flags(usa, usa['players'][0], OPTS, NAMES), ['extra_trucks:3'])
        self.assertEqual(player_flags(gla, gla['players'][0], OPTS, NAMES), [])

    def test_afk_and_early_end(self):
        quitter = play([(100, 'MSG_DO_MOVETO', [])]*3+[(7000, 'MSG_SELF_DESTRUCT', [1])])
        self.assertEqual(game_flags(row(quitter, minutes=2), OPTS), ['afk:slot0:4', 'early_end:2.0min:slot0'])
        self.assertEqual(game_flags(row(play([(1, 'MSG_DO_MOVETO', [])]*11)), OPTS), [])

    def test_listing_ratings_follow_listing_sides(self):
        text = ('21 China Tank karam -28 Seasonal: 935 → 908 -27 Overall: 921 → 893 -28 VS China Abdullah awad '
                '+28 Seasonal: 913 → 940 +27 Overall: 886 → 914 +28 [RANK] Mountain Mayhem v2 1v1 1:37')
        self.assertEqual(listing_ratings(text, ['China Tank', 'China']),
                         [{'seasonal': 935, 'overall': 921}, {'seasonal': 913, 'overall': 886}])
        self.assertEqual(listing_ratings('94 China mGic^ VS GLA Toxin YOUSIF Twilight Flame 1v1 6:50',
                                         ['China', 'GLA Toxin']), [{}, {}])
        # Stats tracking disabled: no ladder figures can be read, so the side is unrated.
        self.assertIsNone(listing_ratings('53 Stats tracking was disabled for this match. It does not count '
                                          'towards player stats or leagues. GLA Stealth BigBoss VS China Medonho '
                                          '[RANK] TD Classic NoCars ZH v1 1v1 9:12', ['GLA Stealth', 'China']))

    def test_only_rated_ladder_matches_are_selected_by_default(self):
        import replay_cleaning
        self.assertIn('unrated', replay_cleaning.DEFAULTS['checks'].split(','))
        game = row(play([(1, 'MSG_DO_MOVETO', [])]*11))
        self.assertEqual(player_flags(game, game['players'][0], OPTS, NAMES), [])
        del game['players'][0]['rating']
        self.assertEqual(player_flags(game, game['players'][0], OPTS, NAMES), ['unrated'])


if __name__ == '__main__':
    unittest.main()
