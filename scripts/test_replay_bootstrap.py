from pathlib import Path
import tempfile
import unittest

from harvest_china_replays import parse_replay
from replay_bootstrap import check_live_provenance, match_key, partition, verify_entry
from test_harvest_china_replays import replay


class BootstrapTests(unittest.TestCase):
    def test_match_identity_groups_other_uploads_and_ignores_recording_time(self):
        header = parse_replay(replay())
        self.assertEqual(match_key(header, {'match_id': 42}), match_key(header, replay_path='match_42_upload_2.rep'))
        header['game_options'] += ';SD=71'
        other = dict(header, start_time=99, end_time=100, local_player=1)
        self.assertEqual(match_key(header), match_key(other))
        other['game_options'] = header['game_options'].replace('SD=71', 'SD=72')
        self.assertNotEqual(match_key(header), match_key(other))
        with self.assertRaisesRegex(ValueError, 'match ID'):
            match_key(parse_replay(replay()))

    def test_partition_is_stable_when_archive_grows(self):
        before = {str(i): partition('pgo:'+str(i)) for i in range(100)}
        after = {str(i): partition('pgo:'+str(i)) for i in range(1000)}
        self.assertEqual(before, {k: after[k] for k in before})
        self.assertEqual(set(before.values()), {'train', 'validation'})

    def test_transfer_binds_live_runtime_instead_of_disabling_checks(self):
        live = {'engine_exe_crc': 11, 'engine_ini_crc': 12, 'ticks_per_second': 60}
        target = {'engine_sha256': 'target', 'engine_ini_crc': 12, 'ticks_per_second': 60}
        check_live_provenance(live, 'target', target)
        for metadata, digest in [(live, 'wrong'), (dict(live, engine_ini_crc=22), 'target'),
                                  (dict(live, ticks_per_second=30), 'target')]:
            with self.assertRaises(ValueError):
                check_live_provenance(metadata, digest, target)

    def test_changed_trajectory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trajectory'; path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'trajectory changed'):
                verify_entry({'trajectory': str(path), 'sha256': 'old'})


if __name__ == '__main__':
    unittest.main()
