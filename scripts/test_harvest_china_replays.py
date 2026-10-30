import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("harvest", Path(__file__).with_name("harvest_china_replays.py"))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def replay(templates=(3, 4), crc=0x216B3EEA, desync=False):
    wide = lambda text: text.encode("utf-16le") + b"\0\0"
    slots = ":".join(f"HPlayer{i},0,0,TT,-1,{t},-1,-1,1" for i, t in enumerate(templates)) + ":" + "X:" * (8 - len(templates))
    options = f"M=00userdata/maps/test;MC=12345678;MS=123;SC=10000;C=100;S={slots};"
    return (b"GENREP" + struct.pack("<iiI", 1, 2, 1000)
            + bytes([desync, 0]) + bytes(8) + wide("Last Replay") + bytes(16)
            + wide("Version 1.04") + wide("Oct 2 2026")
            + struct.pack("<III", 65540, crc, 0x81FB5632) + options.encode() + b"\0"
            + b"0\0" + struct.pack("<4i", 1, 5, 0, 0) + bytes(13))


class FakeClient:
    def __init__(self, data=None):
        self.data = data if data is not None else replay()
        self.calls = []

    def request(self, url):
        self.calls.append(url)
        if "/zh/match/" in url:
            return (b"<span>Game Version:</span><span>216B3EEA</span>"
                    b'<a href="https://strata.gamereplays.org/storage/versions/ini/patch_81FB5632.big">Patch</a>')
        return self.data


class HarvestTests(unittest.TestCase):
    def candidate(self, factions=None):
        return {"match_id": 42, "url": h.BASE + "/zh/match/42",
                "replay_url": "https://matchdata.playgenerals.online/replays/match_42.rep",
                "factions": factions or ["China", "GLA"], "text": "example"}

    def test_rows_distinguish_vanilla_and_generals_and_decode_attributes(self):
        parser = h.parse_page('''<tr wire:key="match-1"><td>
          <img alt="China Tank"><img alt="China">
          <a href="https://strata.gamereplays.org/zh/match/42">match</a>
          <a href="https://matchdata.playgenerals.online/replays/x.rep?a=1&amp;b=2">download</a>
          <script>not metadata</script></td></tr>''')
        self.assertEqual(parser.rows[0]["factions"], ["China Tank", "China"])
        self.assertEqual(parser.rows[0]["replay_url"],
                         "https://matchdata.playgenerals.online/replays/x.rep?a=1&b=2")
        self.assertNotIn("not metadata", parser.rows[0]["text"])

    def test_match_page_supplies_the_replay_link_missing_from_listings(self):
        link = "https://matchdata.playgenerals.online/replays/2026/10/29/match_42/user_a/match_42_user_a_replay.rep"
        class DetailLinkClient(FakeClient):
            def request(self, url):
                if "/zh/match/" in url:
                    self.calls.append(url)
                    page = super().request(url)
                    self.calls.pop()
                    return page + (f'<a href="{link}?x=1&amp;y=2">Replay</a><a href="{link}?x=1&amp;y=2">again</a>'
                                   '<img src="https://matchdata.playgenerals.online/screenshots/s.jpg">').encode()
                return super().request(url)
        self.assertEqual(h.replay_links(f'<a href="{link}?x=1&amp;y=2">r</a><a href="https://example.org/a.rep">x</a>'),
                         [link + "?x=1&y=2"])
        with tempfile.TemporaryDirectory() as directory:
            archive = h.Archive(Path(directory))
            try:
                row = {**self.candidate(), "replay_url": None}
                archive.discover(row)
                client = DetailLinkClient()
                self.assertEqual(archive.harvest(client, row), "downloaded")
                self.assertEqual(client.calls[-1], link + "?x=1&y=2")
                metadata = json.loads((Path(directory) / "216B3EEA/match_42.json").read_text())
                self.assertEqual(metadata["replay_url"], link + "?x=1&y=2")
                # Neither the listing nor the match page: deferred, never guessed.
                with self.assertRaisesRegex(ValueError, "no replay link"):
                    archive.harvest(FakeClient(), {**row, "match_id": 43, "url": h.BASE + "/zh/match/43"})
            finally:
                archive.db.close()

    def test_rediscovery_without_a_link_keeps_the_known_link(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = h.Archive(Path(directory))
            try:
                row = self.candidate(); archive.discover(row)
                archive.db.execute("UPDATE matches SET status='error',retry_after='2999-01-01' WHERE id=42")
                archive.discover({**row, "replay_url": None})
                status, candidate = archive.db.execute("SELECT status,candidate FROM matches WHERE id=42").fetchone()
                self.assertEqual(status, "error")  # backoff kept
                self.assertEqual(json.loads(candidate)["replay_url"], row["replay_url"])
            finally:
                archive.db.close()

    def test_header_preserves_unresolved_random(self):
        header = h.parse_replay(replay((-1, 3)))
        self.assertEqual(header["players"][0]["header_faction"], "Random")
        self.assertEqual(header["players"][1]["header_faction"], "China")
        self.assertEqual(header["exe_crc"], "216B3EEA")
        self.assertEqual(header["ini_crc"], "81FB5632")

    def test_player_name_delimiters_do_not_create_extra_slots(self):
        data = replay().replace(b"Player0", b"AndyDavo: Twitch.TV/AndyDavo,streamer;live")
        header = h.parse_replay(data)
        self.assertEqual(len(header["players"]), 2)
        self.assertEqual(header["players"][0]["name"], "AndyDavo: Twitch.TV/AndyDavo,streamer;live")
        self.assertEqual(header["players"][1]["slot"], 1)

    def test_html_download_and_truncated_header_rejected(self):
        for data in (b"<html>challenge</html>", replay()[:140], replay()[:-20]):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                h.parse_replay(data)

    def test_non_1v1_rejected(self):
        with self.assertRaisesRegex(ValueError, "active humans"):
            h.parse_replay(replay((3, 4, 8)))

    def test_versions_are_required_not_guessed_from_release_names(self):
        self.assertEqual(h.advertised_version("<span>Game Version:</span><b>216b3eea</b>")[0], "216B3EEA")
        with self.assertRaises(ValueError):
            h.advertised_version("Version 100126_QFE6")

    def test_historical_named_release_uses_header_crc_and_preserves_label(self):
        class HistoricalClient(FakeClient):
            def request(self, url):
                if "/zh/match/" in url:
                    self.calls.append(url)
                    return b"<span>Game Version:</span><b>082826_QFE1</b><span>Data Pack:</span>"
                return super().request(url)
        with tempfile.TemporaryDirectory() as directory:
            archive = h.Archive(Path(directory))
            try:
                row = self.candidate(); archive.discover(row)
                client = HistoricalClient(replay(crc=0xB9DB8815))
                self.assertEqual(archive.harvest(client, row), "downloaded")
                sidecar = Path(directory) / "B9DB8815/match_42.json"
                metadata = json.loads(sidecar.read_text())
                self.assertEqual(metadata["advertised_release_label"], "082826_QFE1")
                self.assertIsNone(metadata["advertised_game_version"])
                self.assertEqual(metadata["replay_game_version"], "B9DB8815")
                self.assertEqual(archive.harvest(client, row), "existing")
                self.assertEqual(len(client.calls), 2)
                self.assertIsNone(metadata["source_commit"])
            finally:
                archive.db.close()

    def test_missing_site_version_does_not_discard_a_valid_replay_header(self):
        class MissingVersionClient(FakeClient):
            def request(self, url):
                return b"<html>Old match details</html>" if "/zh/match/" in url else super().request(url)
        with tempfile.TemporaryDirectory() as directory:
            archive = h.Archive(Path(directory))
            try:
                row = self.candidate(); archive.discover(row)
                self.assertEqual(archive.harvest(MissingVersionClient(), row), "downloaded")
                metadata = json.loads((Path(directory) / "216B3EEA/match_42.json").read_text())
                self.assertIsNone(metadata["advertised_release_label"])
                self.assertIsNone(metadata["advertised_game_version"])
                self.assertTrue(archive.valid_existing(42))
            finally:
                archive.db.close()

    def test_download_version_folder_sidecar_and_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = h.Archive(Path(directory))
            try:
                client = FakeClient()
                row = self.candidate()
                archive.discover(row)
                archive.discover(row)
                self.assertEqual(archive.harvest(client, row), "downloaded")
                target = Path(directory) / "216B3EEA/match_42.rep"
                self.assertEqual(target.read_bytes(), replay())
                metadata = json.loads(target.with_suffix(".json").read_text())
                self.assertEqual(metadata["simulation_validation"], "not_run")
                self.assertEqual(metadata["replay_header"]["players"][0]["template_id"], 3)
                self.assertEqual(archive.harvest(client, row), "existing")
                self.assertEqual(len(client.calls), 2)
                self.assertEqual(archive.db.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 1)
                target.write_bytes(b"corrupt")
                self.assertFalse(archive.valid_existing(42))
                self.assertEqual(archive.harvest(client, row), "downloaded")
            finally:
                archive.db.close()

    def test_mismatching_advertised_checksum_never_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = h.Archive(Path(directory))
            try:
                row = self.candidate()
                archive.discover(row)
                with self.assertRaisesRegex(ValueError, "Advertised CRC"):
                    archive.harvest(FakeClient(replay(crc=1)), row)
                self.assertFalse(list(Path(directory).rglob("*.rep")))
            finally:
                archive.db.close()

    def test_blacklisted_release_is_excluded_from_new_collection(self):
        for crc in (0xFBA615FA, 0x1F44693B):
            with self.subTest(crc=crc), tempfile.TemporaryDirectory() as directory:
                archive = h.Archive(Path(directory))
                try:
                    row = self.candidate(); archive.discover(row)
                    with patch.object(h, 'version_metadata', return_value={
                            'crc': f'{crc:08X}', 'label': '092226_QFE1', 'data_pack_url': None}):
                        self.assertEqual(archive.harvest(FakeClient(replay(crc=crc)), row), 'excluded')
                    self.assertFalse(list(Path(directory).rglob('*.rep')))
                    status, error = archive.db.execute('SELECT status, error FROM matches').fetchone()
                    self.assertEqual(status, 'excluded')
                    self.assertIn('Blacklisted release 092226_QFE1', error)
                finally:
                    archive.db.close()

    def test_fixed_wrong_faction_rejected_but_random_uses_site_label(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = h.Archive(Path(directory))
            try:
                row = self.candidate()
                archive.discover(row)
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    archive.harvest(FakeClient(replay((8, 4))), row)
                self.assertEqual(archive.harvest(FakeClient(replay((-1, -1))), row), "downloaded")
            finally:
                archive.db.close()

    def test_desynchronized_match_not_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = h.Archive(Path(directory))
            try:
                row = self.candidate()
                archive.discover(row)
                self.assertEqual(archive.harvest(FakeClient(replay(desync=True)), row), "rejected")
                self.assertFalse(list(Path(directory).rglob("*.rep")))
            finally:
                archive.db.close()

    def test_checkpoint_and_pending_survive_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = h.Archive(Path(directory))
            archive.discover(self.candidate())
            archive.checkpoint(15)
            archive.db.close()
            archive = h.Archive(Path(directory))
            try:
                self.assertEqual(archive.resume_page(), 13)
                self.assertEqual(archive.db.execute("SELECT status FROM matches").fetchone()[0], "pending")
            finally:
                archive.db.close()

    def test_map_filter_ignores_virtual_path_but_preserves_revision_and_crc(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = h.Archive(Path(directory), map_name='test', map_crc='12345678')
            try:
                row = self.candidate(); archive.discover(row)
                self.assertEqual(archive.harvest(FakeClient(), row), 'downloaded')
            finally:
                archive.db.close()
            with self.assertRaisesRegex(ValueError, 'filter changed'):
                h.Archive(Path(directory), map_name='test v2', map_crc='12345678')
        for name, crc in [('test v2', '12345678'), ('test', '99999999')]:
            with tempfile.TemporaryDirectory() as directory:
                archive = h.Archive(Path(directory), map_name=name, map_crc=crc)
                try:
                    row = self.candidate(); archive.discover(row)
                    self.assertEqual(archive.harvest(FakeClient(), row), 'excluded')
                    self.assertFalse(list(Path(directory).rglob('*.rep')))
                finally:
                    archive.db.close()

    def test_map_filter_accepts_unpadded_replay_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = h.Archive(Path(directory), map_name='test', map_crc='0079C463')
            try:
                row = self.candidate(); archive.discover(row)
                client = FakeClient(replay().replace(b'MC=12345678', b'MC=79c463'))
                self.assertEqual(archive.harvest(client, row), 'downloaded')
            finally:
                archive.db.close()

    def test_map_listing_prefilter_accepts_rankless_title_and_header_check_is_authoritative(self):
        row = self.candidate(); row['text'] = 'China VS GLA TD NoBugsCars ZH v1 1v1'
        self.assertTrue(h.listing_matches_map(row, '[RANK] TD NoBugsCars ZH v1'))
        self.assertFalse(h.listing_matches_map(row, '[RANK] TD NoBugsCars ZH v2'))
        self.assertEqual(h.map_identity('0buserdata\\maps\\[RANK] TD NoBugsCars ZH v1'),
                         h.map_identity('00maps/[rank] td nobugscars zh v1'))


if __name__ == "__main__":
    unittest.main()
