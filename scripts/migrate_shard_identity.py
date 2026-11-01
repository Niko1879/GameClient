"""Carry prepared shards to a new engine and feature key once they are qualified as identical.

The key hashes source and shards name their exporting engine, so a re-pin otherwise retires
every shard. Run this only after re-exporting a random sample through the new pipeline and
finding it identical. Only index.json changes: the bytes of every other member are copied.
The replaced identity stays in each index under source.migrated_from.

Originals are untouched until every replacement is built and verified and the catalogue has
committed. Rerunning resumes from any interruption. Stop the converter first.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

from engine_runtime import disable_power_throttling
from prepared_io import _json_bytes
from replay_catalog import ROOT, DEFAULT_CATALOG, canonical, connect, document_bytes, profile_for, store_document

SUFFIX = '.migrating'


def file_digest(path):
    # Not prepared_io._digest: this must read the bytes, whatever their stat says.
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def migrated_index(index, identity):
    source = index['source']
    return {**index, 'legality_key': identity['feature_key'],
            'source': {**source, 'engine_sha256': identity['engine_sha256'],
                       'migrated_from': dict(legality_key=index['legality_key'], engine_sha256=source['engine_sha256'],
                                             qualification=identity['qualification'])}}


def rewrite_shard(path, temporary, index_bytes):
    """Write `path` with a new index.json, every earlier member copied byte for byte."""
    with zipfile.ZipFile(path) as source:
        infos = source.infolist()
    old = next(info for info in infos if info.filename == 'index.json')
    members = [info for info in infos if info is not old]
    if any(info.header_offset > old.header_offset for info in members):
        raise ValueError('index.json is not the last shard member: '+str(path))
    with open(path, 'rb') as stream:
        prefix = stream.read(old.header_offset)
    info = zipfile.ZipInfo('index.json', old.date_time)
    with zipfile.ZipFile(temporary, 'w') as output:
        output.fp.write(prefix)
        output.start_dir = output.fp.tell()
        for member in members:
            output.filelist.append(member)
            output.NameToInfo[member.filename] = member
        output.writestr(info, index_bytes, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
    with open(temporary, 'rb') as stream:
        if stream.read(old.header_offset) != prefix:
            raise ValueError('Rewritten shard members differ: '+str(path))
    with zipfile.ZipFile(temporary) as result:
        if ([(i.filename, i.CRC, i.file_size) for i in result.infolist() if i.filename != 'index.json']
                != [(i.filename, i.CRC, i.file_size) for i in members]):
            raise ValueError('Rewritten shard directory differs: '+str(path))
        if result.testzip() is not None or result.read('index.json') != index_bytes:
            raise ValueError('Rewritten shard failed verification: '+str(path))


def build(job):
    """Check one original against the catalogue and build its replacement beside it."""
    path = Path(job['path'])
    if file_digest(path) != job['archive_sha256']:
        raise ValueError('Shard differs from its catalogue record: '+str(path))
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Shard CRC failed: '+str(path))
        index = json.loads(archive.read('index.json'))
    if hashlib.sha256(_json_bytes(index)).hexdigest() != job['index_sha256']:
        raise ValueError('Shard index differs from its catalogue record: '+str(path))
    temporary = path.with_name(path.name+SUFFIX)
    rewrite_shard(path, temporary, _json_bytes(migrated_index(index, job['identity'])))
    return job['archive_sha256'], file_digest(temporary)


def verify(job):
    return job['archive_sha256'], file_digest(job['path']) == job['archive_sha256']


def _initialize():
    disable_power_throttling()


def replace(temporary, path):
    for attempt in range(150):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            # A scanner or indexer can hold a new file briefly.
            if attempt == 149:
                raise
            time.sleep(.2)


def finish_interrupted(db, data):
    """After the catalogue commits, its hashes name the replacements; anything else is unfinished."""
    for temporary in data.glob('*'+SUFFIX):
        if db.execute('SELECT 1 FROM prepared_shard WHERE archive_sha256=?', (file_digest(temporary),)).fetchone():
            replace(temporary, temporary.with_name(temporary.name[:-len(SUFFIX)]))
        else:
            temporary.unlink()


def migrate(catalog, root, source_key, source_engine, identity, workers, progress=print):
    root, data = Path(root), Path(root)/'training-data'
    db = connect(catalog)
    try:
        finish_interrupted(db, data)
        rows = db.execute('''SELECT archive_sha256,storage_path,index_sha256,profile_id,entry_document_sha256
            FROM prepared_shard WHERE available=1 AND feature_key=? AND engine_sha256=?''', (source_key, source_engine)).fetchall()
        jobs = [dict(archive_sha256=row[0], path=str(root/row[1]), index_sha256=row[2], identity=identity) for row in rows]
        progress(f'{len(jobs)} shards to migrate')
        built = {}
        with ProcessPoolExecutor(workers, initializer=_initialize) as pool:
            for old, new in pool.map(build, jobs, chunksize=8):
                built[old] = new
                if len(built) % 1000 == 0:
                    progress(f'built {len(built)}')
            if len(set(built.values())) != len(jobs):
                raise ValueError('Replacement shards are not unique')
            profiles = {}
            for sha, stored, index_sha, profile_id, entry_sha in rows:
                entry = json.loads(document_bytes(db, entry_sha))
                if entry['index_sha256'] != index_sha:
                    raise ValueError('Catalogue entry differs from its shard record: '+sha)
                index = migrated_index(entry['index'], identity)
                path = root/stored
                new_entry = dict(entry, archive_sha256=built[sha], bytes=path.with_name(path.name+SUFFIX).stat().st_size,
                                 index_sha256=hashlib.sha256(_json_bytes(index)).hexdigest(), index=index)
                new_sha = store_document(db, canonical(new_entry), 'shard-identity-migration', 'prepared-entry')
                if profile_id not in profiles:
                    profile = json.loads(document_bytes(db, db.execute('SELECT document_sha256 FROM prepared_profile WHERE profile_id=?', (profile_id,)).fetchone()[0]))
                    profile['deployment'] = dict(profile['deployment'], engine_sha256=identity['engine_sha256'])
                    profiles[profile_id] = profile_for(db, profile, identity['feature_key'], 'shard-identity-migration')
                db.execute('''UPDATE prepared_shard SET archive_sha256=?,engine_sha256=?,feature_key=?,index_sha256=?,
                    bytes=?,profile_id=?,entry_document_sha256=? WHERE archive_sha256=?''',
                    (built[sha], identity['engine_sha256'], identity['feature_key'], new_entry['index_sha256'],
                     new_entry['bytes'], profiles[profile_id], new_sha, sha))
            store_document(db, canonical(dict(kind='shard_identity_migration', source=dict(feature_key=source_key, engine_sha256=source_engine),
                identity=identity, shards=sorted(built.items()))), 'shard-identity-migration', 'shard-identity-migration')
            db.commit()
            progress('catalogue committed')
            for row in rows:
                path = root/row[1]
                replace(path.with_name(path.name+SUFFIX), path)
            progress('shards replaced')
            jobs = [dict(archive_sha256=sha, path=str(root/stored)) for sha, stored in db.execute(
                'SELECT archive_sha256,storage_path FROM prepared_shard WHERE available=1 AND feature_key=? AND engine_sha256=?',
                (identity['feature_key'], identity['engine_sha256'])).fetchall()]
            bad = [sha for sha, same in pool.map(verify, jobs, chunksize=32) if not same]
        if bad:
            raise ValueError(f'{len(bad)} shards differ from the catalogue after migration')
        progress(f'{len(jobs)} shards verified')
        return len(rows), len(jobs)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-key', required=True); parser.add_argument('--from-engine', required=True)
    parser.add_argument('--qualification', required=True, help='Evidence that a random sample came out identical')
    parser.add_argument('--catalog', default=str(DEFAULT_CATALOG)); parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    from replay_conversion import contract
    target = contract()
    identity = dict(feature_key=target['feature_key'], engine_sha256=target['engine_sha256'], qualification=args.qualification)
    disable_power_throttling()
    print(json.dumps(dict(migrated=migrate(args.catalog, args.root, args.from_key, args.from_engine, identity, args.workers,
                                           lambda text: print(text, flush=True))[0], **identity)))


if __name__ == '__main__':
    main()
