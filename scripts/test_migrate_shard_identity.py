import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import migrate_shard_identity as migration
from prepared_io import _json_bytes
from replay_catalog import connect, document_bytes, register_prepared

OLD = dict(feature_key='old-key', engine_sha256='old-engine')
NEW = dict(feature_key='new-key', engine_sha256='new-engine', qualification='sample evidence')


class MigrateShardIdentity(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.data = self.root/'training-data'
        self.data.mkdir()
        self.catalog = self.root/'catalog.sqlite3'
        db = connect(self.catalog)
        for number in range(3):
            self.add_shard(db, number)
        db.commit()
        db.close()

    def add_shard(self, db, number):
        index = dict(schema=1, kind='generals_event_shard', legality_key=OLD['feature_key'], float_dtype='float32',
                     episodes=[dict(player=number, length=4, shared='0', chunks=[])], shared={}, coverage={},
                     source=dict(match_key=f'pgo:{number}', replay_sha256=f'raw{number}', sha256=f'export{number}',
                                 engine_sha256=OLD['engine_sha256'], engine=dict(engine_exe_crc=7)))
        path = self.data/f'{number}.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr('chunks/0000/000000.bin', bytes([number])*5000)
            archive.writestr('shared/0000.bin', b'shared'*100)
            archive.writestr('index.json', _json_bytes(index))
        entry = dict(archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                     index_sha256=hashlib.sha256(_json_bytes(index)).hexdigest(), index=index, match_key=f'pgo:{number}', split='train')
        doc = dict(config=dict(a=1), deployment=dict(engine_sha256=OLD['engine_sha256'], engine_ini_crc=5, ticks_per_second=60),
                   capability_catalog_sha256='catalog')
        register_prepared(db, entry, doc, path)
        db.execute('UPDATE prepared_shard SET storage_path=? WHERE archive_sha256=?',
                   (f'training-data/{number}.zip', entry['archive_sha256']))

    def members(self, path):
        with zipfile.ZipFile(path) as archive:
            return {name: archive.read(name) for name in archive.namelist() if name != 'index.json'}

    def run_migration(self):
        return migration.migrate(self.catalog, self.root, OLD['feature_key'], OLD['engine_sha256'], NEW, 2, lambda text: None)

    def test_only_the_index_changes_and_the_old_identity_is_kept(self):
        before = {path.name: (self.members(path), path.read_bytes()) for path in self.data.glob('*.zip')}
        self.assertEqual(self.run_migration(), (3, 3))
        for path in self.data.glob('*.zip'):
            members, original = before[path.name]
            self.assertEqual(self.members(path), members)
            with zipfile.ZipFile(path) as archive:
                start = archive.getinfo('index.json').header_offset
                self.assertEqual(path.read_bytes()[:start], original[:start])
                self.assertIsNone(archive.testzip())
                index = json.loads(archive.read('index.json'))
            self.assertEqual(index['legality_key'], NEW['feature_key'])
            self.assertEqual(index['source']['engine_sha256'], NEW['engine_sha256'])
            self.assertEqual(index['source']['engine'], dict(engine_exe_crc=7))
            self.assertEqual(index['source']['migrated_from'], dict(legality_key=OLD['feature_key'],
                engine_sha256=OLD['engine_sha256'], qualification=NEW['qualification']))
        self.assertEqual(list(self.data.glob('*'+migration.SUFFIX)), [])

    def test_catalogue_describes_the_new_files(self):
        self.run_migration()
        db = connect(self.catalog)
        rows = db.execute('SELECT archive_sha256,engine_sha256,feature_key,index_sha256,storage_path,bytes,profile_id,entry_document_sha256 FROM prepared_shard').fetchall()
        self.assertEqual(len(rows), 3)
        for sha, engine, key, index_sha, stored, size, profile_id, entry_sha in rows:
            path = self.root/stored
            self.assertEqual((engine, key), (NEW['engine_sha256'], NEW['feature_key']))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sha)
            self.assertEqual(path.stat().st_size, size)
            entry = json.loads(document_bytes(db, entry_sha))
            self.assertEqual((entry['archive_sha256'], entry['bytes'], entry['index_sha256']), (sha, size, index_sha))
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(hashlib.sha256(archive.read('index.json')).hexdigest(), index_sha)
                self.assertEqual(json.loads(archive.read('index.json')), entry['index'])
            document = db.execute('SELECT document_sha256,feature_key FROM prepared_profile WHERE profile_id=?', (profile_id,)).fetchone()
            self.assertEqual(document[1], NEW['feature_key'])
            self.assertEqual(json.loads(document_bytes(db, document[0]))['deployment'],
                             dict(engine_sha256=NEW['engine_sha256'], engine_ini_crc=5, ticks_per_second=60))
        db.close()

    def test_a_changed_original_stops_before_anything_is_replaced(self):
        target = self.data/'1.zip'
        target.write_bytes(target.read_bytes()+b'x')
        untouched = (self.data/'0.zip').read_bytes()
        with self.assertRaisesRegex(ValueError, 'differs from its catalogue'):
            self.run_migration()
        self.assertEqual((self.data/'0.zip').read_bytes(), untouched)
        db = connect(self.catalog)
        self.assertEqual(db.execute('SELECT DISTINCT feature_key FROM prepared_shard').fetchall(), [(OLD['feature_key'],)])
        db.close()

    def test_rerun_discards_unfinished_replacements(self):
        leftover = self.data/('0.zip'+migration.SUFFIX)
        leftover.write_bytes(b'partial')
        self.assertEqual(self.run_migration(), (3, 3))
        self.assertFalse(leftover.exists())

    def test_rerun_finishes_a_committed_migration(self):
        self.run_migration()
        db = connect(self.catalog)
        sha = db.execute("SELECT archive_sha256 FROM prepared_shard WHERE storage_path='training-data/0.zip'").fetchone()[0]
        db.close()
        # The state after the commit and before the swap: the original name holds other bytes.
        replacement = self.data/('0.zip'+migration.SUFFIX)
        shutil.copy2(self.data/'0.zip', replacement)
        (self.data/'0.zip').write_bytes(b'original')
        self.assertEqual(self.run_migration(), (0, 3))
        self.assertEqual(hashlib.sha256((self.data/'0.zip').read_bytes()).hexdigest(), sha)
        self.assertFalse(replacement.exists())


if __name__ == '__main__':
    unittest.main()
