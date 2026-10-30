import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from replay_revision_survey import classify, groups, wilson, probe
from test_harvest_china_replays import replay


class SurveyTests(unittest.TestCase):
    def test_blacklisted_release_probe_records_exclusion_without_launching_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root/'archive'
            for crc in (0xFBA615FA, 0x1F44693B):
                folder = archive/f'{crc:08X}'; folder.mkdir(parents=True)
                (folder/'match_42.rep').write_bytes(replay(crc=crc))
            exe = root/'fake.exe'; exe.write_bytes(b'not executed')
            args = SimpleNamespace(output=str(root/'output'), exe=str(exe), user_data=str(root/'profile'),
                data_root=str(root/'packs'), delay=1, archives=[archive], match_ids=None,
                min_frames=1, seed=1, per_group=1, timeout=1)
            with patch('replay_revision_survey.subprocess.run') as native:
                probe(args)
                native.assert_not_called()
            report = json.loads((root/'output/survey.json').read_text())
            self.assertEqual(report['cases'], [])
            self.assertEqual(len(report['excluded']), 2)

    def test_crc_padding_is_numeric_but_missing_or_invalid_checksums_fail_closed(self):
        evidence = dict(engine_ini_crc='0081FB56', loaded_map_crc='0079C463', crc_comparisons=20,
                        crc_mismatch=False, completed=True, stream_eof=True,
                        last_command_frame=1000, final_frame=1001)
        header, stream = dict(ini_crc='81fb56', map_crc='79c463'), dict(last_command_frame=1000)
        self.assertEqual(classify(evidence, header, stream, 0), 'passed')
        for value in (None, '', '0x79C463', ' 79C463', '0079C463\n', '00079C463', 'Z79C463'):
            with self.subTest(value=value):
                self.assertEqual(classify(evidence, dict(header, map_crc=value), stream, 0), 'map_mismatch')
                self.assertEqual(classify(dict(evidence, loaded_map_crc=value), header, stream, 0), 'map_mismatch')
        evidence['crc_mismatch'] = True
        self.assertEqual(classify(evidence, header, stream, 1), 'crc_diverged')
        self.assertEqual(classify(dict(evidence, crc_comparisons=0), header, stream, 0), 'no_crc_evidence')

    def test_setup_mismatches_and_missing_checks_are_not_crc_divergence(self):
        evidence = dict(engine_ini_crc='81FB5632', loaded_map_crc='40D65C33', crc_comparisons=20,
                        crc_mismatch=False, completed=True, stream_eof=True, last_command_frame=1000, final_frame=1001)
        header, stream = dict(ini_crc='81FB5632', map_crc='40D65C33'), dict(last_command_frame=1000)
        self.assertEqual(classify(evidence, header, stream, 0), 'passed')
        evidence['crc_mismatch'] = True
        self.assertEqual(classify(evidence, header, stream, 1), 'crc_diverged')
        self.assertEqual(classify(evidence, dict(header, ini_crc='12345678'), stream, 1), 'data_mismatch')
        self.assertEqual(classify(evidence, dict(header, map_crc='12345678'), stream, 1), 'map_mismatch')
        evidence['crc_comparisons'] = 0
        self.assertEqual(classify(evidence, header, stream, 1), 'no_crc_evidence')
        self.assertEqual(classify(None, header, stream, 1), 'engine_error')
        self.assertEqual(classify(None, header, stream, 1, 'Cannot open replay'), 'replay_rejected')

    def test_rate_denominator_excludes_setup_failures_and_zero_failures_is_not_certainty(self):
        result = groups([dict(replay_exe_crc='A', replay_ini_crc='B', status=status)
                         for status in ['passed', 'passed', 'crc_diverged', 'map_mismatch', 'timeout']])['A/B']
        self.assertEqual(result['resolved'], 3)
        self.assertAlmostEqual(result['divergence_rate_among_resolved'], 1/3)
        self.assertIsNone(wilson(0, 0))
        self.assertGreater(wilson(0, 8)[1], .3)


if __name__ == '__main__':
    unittest.main()
