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

    def test_unrated_games_are_kept_and_joke_maps_dropped_by_default(self):
        # User decision, 7 Nov 2026: unrated games train with the rated flag off; the map rule
        # (official, ranked-pack or rated-ladder CRC) keeps out joke/4fun maps instead.
        import replay_cleaning
        checks = replay_cleaning.DEFAULTS['checks'].split(',')
        self.assertIn('map', checks)
        self.assertNotIn('unrated', checks)
        game = row(play([(1, 'MSG_DO_MOVETO', [])]*11))
        self.assertEqual(player_flags(game, game['players'][0], OPTS, NAMES), [])
        del game['players'][0]['rating']
        self.assertEqual(player_flags(game, game['players'][0], OPTS, NAMES), ['unrated'])  # reported
        ladder = {'0ABC1234'}
        for path, crc, kept in (('maps/tournament desert', 'FFFFFFFF', True),
                                ('userdata/maps/[rank] vendetta zh v1', '0ABC1234', True),
                                ('userdata/maps/renamed vendetta', '0ABC1234', True),
                                ('userdata/maps/[rank] vendetta zh v1', '0ABC1235', False),
                                ('userdata/maps/!1v1 cxn', '12345678', False)):
            game.update(map=path, map_crc=crc)
            self.assertEqual(game_flags(game, OPTS, ladder), [] if kept else ['map:'+crc], path)
        self.assertIn('cash', checks)
        game.update(map='maps/tournament desert', starting_cash=1000000)
        self.assertEqual(game_flags(game, OPTS, ladder), ['cash:1000000'])
        game['starting_cash'] = 10000
        self.assertEqual(game_flags(game, OPTS, ladder), [])

    def test_only_verified_two_slot_maps_are_kept(self):
        # User decision, 10 Nov 2026: a competitive 1v1 is played on a 2-slot map.
        import replay_cleaning
        self.assertIn('slots', replay_cleaning.DEFAULTS['checks'].split(','))
        table = {'by_crc': {'0ABC1234': 2, '0ABC9999': 8}, 'official': {'tournament desert': 2, 'twilight flame': 8}}
        game = row(play([(1, 'MSG_DO_MOVETO', [])]*11))
        for path, crc, flag in (('userdata/maps/[rank] vendetta zh v1', '0ABC1234', None),
                                ('userdata/maps/[rank] [nmc 2x2] middle fight', '0ABC9999', 'slots:0ABC9999:8'),
                                ('maps/tournament desert', 'FFFFFFFF', None),  # patched official: by name
                                ('maps/twilight flame', 'FFFFFFFE', 'slots:FFFFFFFE:8'),
                                ('userdata/maps/[rank] unknown', '12345678', 'slots:12345678:unverified')):
            game.update(map=path, map_crc=crc)
            self.assertEqual(game_flags(game, OPTS, None, table), [flag] if flag else [], path)

    def test_map_slots_reads_refpack_maps_and_map_cache(self):
        import tempfile
        from pathlib import Path
        import map_slots
        # Literal 'abcd', then an overlapping copy of 8 bytes from distance 4, then the end code.
        packed = bytes([0x10, 0xFB, 0, 0, 12, 0xE0])+b'abcd'+bytes([0x14, 3, 0xFC])
        self.assertEqual(map_slots.refpack(packed), b'abcdabcdabcd')
        with tempfile.TemporaryDirectory() as folder:
            body = b'..Player_1_Start..Player_2_Start..Player_4_Start..'  # 3 missing: two consecutive
            path = Path(folder)/'a.map'
            path.write_bytes(b'EAR\0'+len(body).to_bytes(4, 'little')+bytes([0x10, 0xFB])+len(body).to_bytes(3, 'big')
                             + b''.join(bytes([0xE0+((len(c)-4) >> 2)])+c for c in (body[:48],))
                             + bytes([0xFC+len(body)-48])+body[48:])
            self.assertEqual(map_slots.start_count(path), 2)
            cache = Path(folder)/'MapCache.ini'
            cache.write_text('MapCache maps_5calpine_20assault_5calpine_20assault_2emap\n  fileCRC = 3735928559\n'
                             '  isOfficial = yes\n  numPlayers = 2\nEND\n', encoding='latin-1')
            self.assertEqual(map_slots.map_cache(cache), [('maps/alpine assault/alpine assault.map', 'DEADBEEF', 2, True)])
            table = map_slots.build(caches=[cache], retail_maps=Path(folder)/'none')
            self.assertEqual(table['official'], {'alpine assault': 2})

    def test_map_identity_strips_the_header_prefix_and_pads_the_crc(self):
        from replay_cleaning import map_identity
        self.assertEqual(map_identity({'map': '0Buserdata\\maps\\[rank] td nobugscars zh v1', 'map_crc': 'abc'}),
                         ('userdata/maps/[rank] td nobugscars zh v1', '00000ABC'))
        self.assertEqual(map_identity({'map': '0fmaps/triple combat', 'map_crc': '1'})[0], 'maps/triple combat')


class RatingRuleTest(unittest.TestCase):
    def test_rating_rule(self):
        from replay_cleaning import ESTIMATE_FIELD, rating_of
        row = {'rating': None, 'name_rating': None, 'identity': 'gamereplays:ExCaL', 'style_estimate': 1580,
               'joint_estimate': 2200}
        est = row[ESTIMATE_FIELD]
        self.assertEqual(rating_of(row), (est, True))
        self.assertEqual(rating_of({**row, 'identity': 'go:x', 'name_rating': 1700}), (1700, False))
        self.assertEqual(rating_of({**row, 'identity': 'gamereplays:x', 'name_rating': 1700}), (est, True))
        self.assertEqual(rating_of({**row, 'rating': 1400}), (1400, False))


if __name__ == '__main__':
    unittest.main()
