"""One audited read yields per-player views, vocabulary and proofs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from replay_loading import load_replays
from replay_training import audit, records


def command(name, *arguments):
    row = {'name': name, 'type_id': 1000, 'arguments': [{'value': arg} for arg in arguments]}
    if name == 'MSG_DOZER_CONSTRUCT':
        row['build_result'] = {'accepted': True, 'status': 'created'}
    return row


def fixture(rate):
    header = dict(record='header', schema_version=1, controlled_faction='FactionChina',
        ticks_per_second=rate, template_catalog={'7': 'ChinaPowerPlant', '8': 'ChinaInfantryRedguard'},
        engine_git_sha='fixture', engine_git_dirty=True, engine_exe_crc=1, engine_ini_crc=2, placement_mode='fast', quiet_capture='all',
        observation_policy=dict(version=8, excluded_templates=[], visible_only_templates=[],
                                neutral_excluded_templates=[], playable_players=[2, 3, 4]))
    terrain = dict(record='map', grid_size=1, height=[0], cliff=[False])
    rows, previous = [], {}
    def sample(player, frame, *actions):
        observation = dict(player={'index': player, 'faction': 'FactionChina'}, visibility_grid=[0],
            own_entities=[dict(id=player, type='ChinaVehicleDozer', owner=player, own=True,
                visible=True, last_seen_frame=frame, resume_targets=[])], known_nonowned_entities=[],
            observation_policy_version=8)
        delta = frame-previous[player] if player in previous else 0
        previous[player] = frame
        rows.append(dict(record='sample', player_index=player, frame=frame, delta_ticks=delta,
            time_seconds=frame/rate, decision='commands' if actions else 'wait',
            observation=observation, actions=list(actions)))
    # Player 2 appears first but has only one observation. Reference delivery
    # discovers player 3 first; the terminal train order must not enter menus.
    sample(2, 1, command('MSG_CREATE_SELECTED_GROUP', True, 2), command('MSG_QUEUE_UNIT_CREATE', 8, 99))
    sample(3, 2, command('MSG_CREATE_SELECTED_GROUP', True, 3), command('MSG_CREATE_TEAM1', 3))
    sample(3, 3, command('MSG_DOZER_CONSTRUCT', 7, [10, 20, 0], 0))
    sample(3, 4, command('MSG_DESTROY_SELECTED_GROUP'))
    sample(3, 5, command('MSG_SELECT_TEAM1'), command('MSG_DO_MOVETO', [20, 30, 0]))
    sample(3, 6, command('MSG_QUEUE_UPGRADE', 123))
    sample(3, 7)
    footer = dict(record='footer', valid=True, playback_completed=True, crc_mismatch=False,
        samples=len(rows), commands=sum(len(r['actions']) for r in rows),
        wait_samples=sum(not r['actions'] for r in rows), final_frame=7,
        recorded_crc_messages_seen=1, recorded_crc_messages_compared=1,
        replay_stream_eof=True, last_recorded_command_frame=6)
    return [header, terrain, *rows, footer]


class ReplayMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/'replay.jsonl'
        cache = patch('replay_metadata.CACHE_DIRECTORY', Path(temporary.name)/'metadata-cache')
        cache.start()
        self.addCleanup(cache.stop)

    def write(self, rows):
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')

    def test_players_hotkeys_terminal_labels_and_rates(self):
        for rate in (30, 60):
            with self.subTest(rate=rate):
                self.write(fixture(rate))
                before = self.path.read_bytes()
                games, vocabulary, sources = load_replays([self.path], use_metadata_cache=False)
                # Player 3 is discovered first; the terminal train order must not enter menus.
                self.assertEqual([g['player'] for g in games], [3, 2])
                self.assertEqual(vocabulary['menus'], {'ChinaVehicleDozer|build': ['ChinaPowerPlant']})
                self.assertEqual([len(g['rows']) for g in games], [6, 1])
                self.assertEqual([r['frame'] for r in games[0]['rows']], [f*60//rate for f in range(2, 8)])
                self.assertEqual(sources[0]['audit']['samples'], 7)
                self.assertEqual(self.path.read_bytes(), before)

    def test_metadata_uses_one_finite_audit_read(self):
        self.write(fixture(60))
        with patch('replay_training.records', wraps=records) as decoding:
            games, _, _ = load_replays([self.path], use_metadata_cache=False)
        decoding.assert_called_once_with(self.path, check_finite=True)
        self.assertEqual(sum(len(g['rows']) for g in games), 7)

    def test_invalid_completion_and_corrupt_observation_still_fail_metadata_loading(self):
        for error in ('footer', 'visibility'):
            with self.subTest(error=error):
                rows = fixture(60)
                if error == 'footer':
                    rows[-1]['replay_stream_eof'] = False
                else:
                    rows[2]['observation']['visibility_grid'][0] = 3
                self.write(rows)
                with self.assertRaises(ValueError):
                    load_replays([self.path], use_metadata_cache=False)

    def test_source_change_during_combined_audit_is_rejected(self):
        self.write(fixture(60))
        def changed(*args, **kwargs):
            result = audit(*args, **kwargs)
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write('\n')
            return result
        with patch('replay_loading.audit', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'changed while loading'):
                load_replays([self.path], use_metadata_cache=False)

    def test_cached_receipt_skips_a_second_audit(self):
        self.write(fixture(60))
        games, vocabulary, sources = load_replays([self.path])
        with patch('replay_loading.audit', side_effect=AssertionError('Unexpected cached re-audit')):
            reused, reused_vocabulary, reused_sources = load_replays([self.path])
        self.assertEqual((reused_vocabulary, reused_sources), (vocabulary, sources))
        self.assertEqual([(g['player'], len(g['rows'])) for g in reused],
                         [(g['player'], len(g['rows'])) for g in games])


if __name__ == '__main__':
    unittest.main()
