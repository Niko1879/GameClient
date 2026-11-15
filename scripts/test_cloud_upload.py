"""Bundles carry no shards: the uploader streams them from the bundle's shard source."""
import json
from pathlib import Path
import tempfile
import unittest

import cloud_upload


class CloudUploadTests(unittest.TestCase):
    def bundle(self, root, shards=('b.zip', 'a.zip', 'c.zip')):
        source, bundle = root/'training-data', root/'bundle'
        source.mkdir(); (bundle/'code').mkdir(parents=True); (bundle/'data').mkdir()
        for name in shards:
            (source/name).write_bytes(name.encode()*10)
        (bundle/'code/train_events.py').write_text('code')
        (bundle/'train.sh').write_text('#!/bin/sh')
        (bundle/'bundle.json').write_text(json.dumps({'shard_source': str(source)}))
        (bundle/'data/dataset.json').write_text(json.dumps({'entries': [{'path': name} for name in shards]}))
        return bundle, source

    def test_manifest_part_comes_first_and_shard_parts_name_source_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle, source = self.bundle(Path(temporary))
            self.assertEqual(cloud_upload.shard_source(bundle), source)
            manifest, *shards = cloud_upload.parts(bundle, 2)
            self.assertEqual(manifest, ['bundle.json', 'train.sh', 'code/train_events.py', 'replay-runtime/science-orders.json', 'data/dataset.json'])
            self.assertEqual(shards, [['a.zip', 'b.zip'], ['c.zip']])
            self.assertTrue(all((source/name).is_file() for part in shards for name in part))
            self.assertFalse(any((bundle/'data'/name).exists() for part in shards for name in part))
            # Resume markers name the shards' place on the instance, data/<file>.
            self.assertEqual(cloud_upload.part_id(['a.zip']), cloud_upload.part_id(['a.zip']))
            self.assertNotEqual(cloud_upload.part_id(['a.zip']), cloud_upload.part_id(['b.zip']))

    def test_a_missing_shard_stops_before_anything_is_sent(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle, source = self.bundle(Path(temporary))
            (source/'b.zip').unlink()
            with self.assertRaisesRegex(SystemExit, 'b.zip'):
                cloud_upload.shard_source(bundle)


if __name__ == '__main__':
    unittest.main()
