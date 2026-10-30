"""Catalogue durability, complete outcomes and reusable data without dataset files."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import replay_catalog as catalog
import sync_replay_catalog as sync


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/'catalog.sqlite3'
        self.db = sync.connect(self.path)
        self.addCleanup(self.db.close)

    def manifest(self, key='feature-a', number=1, split='train'):
        source = dict(match_key=f'pgo:{number}',replay_sha256='raw',sha256='native-export',
                      engine_sha256='engine',audit=dict(replay_stream_eof=True,recorded_crc_messages_compared=3),
                      command_reconciliation=dict(all_controlled_commands_captured=True))
        index = dict(legality_key=key,source=source,episodes=[])
        shard = self.root/f'{key}-{number}.zip'
        with zipfile.ZipFile(shard,'w') as z:
            z.writestr('index.json',catalog.canonical(index))
            z.writestr('payload.bin',b'unchanged examples')
        entry = dict(match_key=source['match_key'],split=split,path=shard.name,index=index,
                     archive_sha256=catalog.digest(shard.read_bytes()),index_sha256=catalog.digest(catalog.canonical(index)))
        path = self.root/f'{key}-{number}.json'
        doc = dict(kind='generals_event_dataset',schema=1,complete=True,
                   config=dict(architecture='generals-relational-v4-event',feature_marker=key),entries=[entry])
        path.write_bytes(catalog.canonical(doc))
        return path,doc

    def test_documents_and_prepared_records_survive_source_deletion(self):
        manifest,doc = self.manifest()
        raw = manifest.read_bytes()
        sha = catalog.import_prepared_manifest(self.db,manifest)
        catalog.import_prepared_manifest(self.db,manifest)
        self.db.commit()
        manifest.unlink()
        self.assertEqual(catalog.document_bytes(self.db,sha),raw)
        self.assertEqual(self.db.execute('SELECT count(*) FROM prepared_shard').fetchone()[0],1)
        self.assertEqual(catalog.import_prepared_directory(self.db,self.root)['manifests'],0)
        rebuilt = catalog.build_dataset(self.db,'feature-a')
        self.assertEqual(rebuilt['entries'],doc['entries'])
        restored = catalog.restore_document(sha,self.root/'temporary-selection.json',self.path)
        self.assertEqual(restored.read_bytes(),raw)

    def test_reuse_checks_physical_shard_and_feature_identity_without_job_state(self):
        manifest,doc = self.manifest()
        catalog.import_prepared_manifest(self.db,manifest)
        self.assertEqual(len(catalog.reusable_prepared(self.db,'pgo:1','raw','feature-a')),1)
        self.assertEqual(catalog.reusable_prepared(self.db,'pgo:1','changed-raw','feature-a'),[])
        self.assertEqual(catalog.reusable_prepared(self.db,'pgo:1','raw','feature-b'),[])
        # Another engine's export of the same features does not qualify.
        self.assertEqual(len(catalog.reusable_prepared(self.db,'pgo:1','raw','feature-a','engine')),1)
        self.assertEqual(catalog.reusable_prepared(self.db,'pgo:1','raw','feature-a','other-engine'),[])
        manifest.unlink()
        self.assertEqual(len(catalog.build_dataset(self.db,'feature-a')['entries']),1)
        (self.root/doc['entries'][0]['path']).unlink()
        self.assertEqual(catalog.reusable_prepared(self.db,'pgo:1','raw','feature-a'),[])

    def test_failure_and_later_success_are_both_kept_and_import_is_idempotent(self):
        failure = dict(status='crc_diverged',replay='match_1.rep',backend='old-engine',
                       error='CRC failed at frame 301',validation=dict(crc_comparisons=3,crc_mismatch=True,completed=False))
        first = catalog.record_export(self.db,failure,engine_sha256='old-sha')
        self.assertEqual(catalog.record_export(self.db,failure,engine_sha256='old-sha'),first)
        manifest,_ = self.manifest()
        catalog.import_prepared_manifest(self.db,manifest)
        row = self.db.execute('SELECT exported,crc_diverged,prepared_shards FROM replay_processing WHERE match_key=?',('pgo:1',)).fetchone()
        self.assertEqual(row,(1,1,1))
        recorded=json.loads(catalog.document_bytes(self.db,first))
        self.assertEqual(recorded['result'],failure)
        self.assertEqual(recorded['engine_sha256'],'old-sha')

    def test_selection_keeps_feature_versions_separate_and_holds_out_validation(self):
        for key,number,split in [('feature-a',1,'train'),('feature-a',2,'validation'),('feature-b',1,'train')]:
            path,_ = self.manifest(key,number,split)
            catalog.import_prepared_manifest(self.db,path)
        selected = catalog.build_dataset(self.db,'feature-a',limit=1,seed=12)
        self.assertEqual([e['match_key'] for e in selected['entries']],['pgo:1'])
        self.assertEqual(selected['entries'][0]['index']['legality_key'],'feature-a')
        self.assertEqual(catalog.build_dataset(self.db,'feature-b')['config']['feature_marker'],'feature-b')
        self.assertEqual(catalog.build_dataset(self.db,'feature-a',split='validation')['entries'][0]['match_key'],'pgo:2')

    def test_test_reserve_is_held_out_of_training_selections(self):
        for number in (1, 2):
            path,_ = self.manifest(number=number)
            catalog.import_prepared_manifest(self.db,path)
        reserve = self.root/'replays/test-reserve.json'
        reserve.parent.mkdir()
        reserve.write_text(json.dumps(dict(matches=[dict(match_key='pgo:2')])))
        self.assertEqual(catalog.import_test_reserve(self.db,self.root),1)
        self.assertEqual([e['match_key'] for e in catalog.build_dataset(self.db,'feature-a')['entries']],['pgo:1'])
        self.assertEqual([e['match_key'] for e in catalog.build_dataset(self.db,'feature-a',split='test')['entries']],['pgo:2'])

    def test_corrupt_shards_and_catalogue_payloads_fail_closed(self):
        manifest,doc=self.manifest()
        shard=self.root/doc['entries'][0]['path']
        shard.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'):
            catalog.import_prepared_manifest(self.db,manifest)
        sha=catalog.store_document(self.db,b'original','source','fixture')
        self.db.execute('UPDATE catalog_document SET uncompressed_bytes=0 WHERE sha256=?',(sha,))
        with self.assertRaisesRegex(ValueError,'checksum'):
            catalog.document_bytes(self.db,sha)

    def test_missing_training_file_does_not_erase_export_or_metadata(self):
        manifest,doc=self.manifest()
        catalog.import_prepared_manifest(self.db,manifest)
        (self.root/doc['entries'][0]['path']).unlink()
        catalog.refresh_availability(self.db)
        self.assertEqual(self.db.execute('SELECT exported,prepared_shards FROM replay_processing').fetchone(),(1,0))
        self.assertEqual(self.db.execute('SELECT count(*) FROM prepared_shard').fetchone()[0],1)


if __name__=='__main__':
    unittest.main()
