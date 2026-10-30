"""Prepare bounded export batches against a frozen vocabulary, then retire payloads.

The encoded feature contract stays in event_store; this wrapper adds batching,
catalogue registration and verified export retirement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import zipfile

from event_dataset import EventDataset
from event_store import (_check_preparation, _encode_preparation, _prepare_pool,
                         _preparation_guard, feature_identity)
from generals_policy import OBSERVATION_VERSION, NeuralPolicy
from prepared_io import _digest, _write_json
from replay_bootstrap import verify_entry
from replay_storage import manifest_path
from generals_identity import in_production_scope


def check_vocabulary(frozen, observed):
    missing = dict(types=sorted(set(observed['types'])-set(frozen['types'])),
        templates=sorted(name for name in set(observed['templates'])-set(frozen['templates'])
                         if in_production_scope(frozen, name)))
    missing['menus'] = {name: sorted(value for value in set(values)-set(frozen['menus'].get(name, []))
                                   if in_production_scope(frozen, value))
                        for name, values in observed['menus'].items()
                        if in_production_scope(frozen, name.rsplit('|', 1)[0])
                        and any(in_production_scope(frozen, value)
                                for value in set(values)-set(frozen['menus'].get(name, [])))}
    if any(missing.values()):
        raise ValueError('Frozen event vocabulary does not cover this batch: '+json.dumps(missing))


def prepare_batch(dataset, output, config_path, *, action_catalog, minimum_ram=1024**3, workers=4,
                  manifest_name='dataset.json', flat=False, catalog=None):
    from replay_loading import load_replays
    if type(workers) is not int or workers < 1:
        raise ValueError('Positive preparation worker count required')
    input_path, output = Path(dataset).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text())
    policy, document = NeuralPolicy(config, parameters=False), json.loads(input_path.read_text())
    key = feature_identity(policy)
    entries = document['entries']
    if not entries or len({e['match_key'] for e in entries}) != len(entries):
        raise ValueError('Expected unique nonempty whole-match batch')
    if Path(manifest_name).name != manifest_name or not manifest_name.endswith('.json'):
        raise ValueError('Manifest name must be a JSON filename without directories')
    destination = output/manifest_name
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous.get('complete'):
            if previous['input_manifest_sha256'] != _digest(input_path) or previous['config'] != config:
                raise ValueError('Prepared batch is immutable')
            verify_prepared(destination)
            if catalog:
                from replay_catalog import snapshot_manifest
                snapshot_manifest(destination,catalog)
            return destination
    jobs = []
    for ordinal, entry in enumerate(entries):
        _preparation_guard(output, minimum_ram)
        proof = verify_entry(entry, deep=False)
        if proof['audit'].get('observation_policy_version') != OBSERVATION_VERSION:
            raise ValueError('Export uses a different observation version')
        # Full-mask training needs exported placement grids; fast mode uses accepted sites.
        if config['placement_mode'] == 'full_mask' and proof['audit'].get('placement_mode') != 'full_mask':
            raise ValueError('Full-mask preparation requires full-mask exports')
        games, observed, sources = load_replays([entry['trajectory']])
        if entry['split'] == 'train':
            check_vocabulary(config, observed)
        path = Path(entry['trajectory']).resolve()
        name = hashlib.sha256((entry['sha256']+key).encode()).hexdigest()+'.zip'
        jobs.append(dict(ordinal=ordinal, path=str(path), sha256=entry['sha256'],
            destination=str((output if flat else output/'shards')/name), match_key=entry['match_key'], split=entry['split'],
            games=games, proof=sources[0]))
    result = dict(schema=1, kind='generals_event_dataset', config=config, complete=False,
        deployment=document.get('deployment'), input_manifest_sha256=_digest(input_path),
        capability_catalog_sha256=_digest(action_catalog), entries=[])
    completed, pending = {}, []

    def publish(job, prepared, pid=None):
        path = Path(job['destination'])
        _check_preparation(path, prepared, key, job['match_key'], job['sha256'])
        completed[job['ordinal']] = dict(prepared, path=path.relative_to(output).as_posix(),
            match_key=job['match_key'], split=job['split'])
        result['entries'] = [completed[i] for i in sorted(completed)]
        _write_json(destination, result)
        if catalog:
            from replay_catalog import connect as catalog_connect, register_prepared, store_document, canonical
            db = catalog_connect(catalog)
            try:
                register_prepared(db,completed[job['ordinal']],result,path,source_path=path.with_suffix('.json'),verified=True)
                store_document(db,canonical(prepared),path.with_suffix('.json'),'preparation-metadata')
                db.commit()
            finally:
                db.close()
        print(json.dumps(dict(prepared=job['match_key'], completed=len(completed), total=len(jobs))), flush=True)

    for job in jobs:
        path = Path(job['destination'])
        sidecar = path.with_suffix('.json')
        if path.exists() and sidecar.exists():
            publish(job, json.loads(sidecar.read_text()))
        else:
            pending.append(job)
    if len(pending) > 1 and workers > 1:
        _prepare_pool(pending, config, output, minimum_ram, min(workers, len(pending)), publish)
    else:
        for job in pending:
            publish(job, _encode_preparation(job, policy, lambda: _preparation_guard(output, minimum_ram)), os.getpid())
    result.update(complete=True, preparation_workers=min(workers, len(pending)))
    _write_json(destination, result)
    verify_prepared(destination)
    if catalog:
        from replay_catalog import snapshot_manifest
        snapshot_manifest(destination,catalog)
    return destination


def verify_prepared(dataset):
    data = EventDataset(dataset)
    expected = feature_identity(NeuralPolicy(data.config, parameters=False))
    if data.key != expected:
        raise ValueError('Prepared feature implementation changed')
    for entry in data.entries:
        _check_preparation(data.shard_path(entry), entry, data.key,
            entry['match_key'], entry['index']['source']['sha256'])
        with zipfile.ZipFile(data.shard_path(entry)) as archive:
            if archive.testzip() is not None:
                raise ValueError('Prepared shard CRC failed')
    return data


def retire_exports(native_dataset, event_dataset, owned_root):
    """Journal before deleting only verified trajectories inside an owned root.

    Raw .rep recordings, manifests, proof receipts, prepared shards and checkpoints
    stay intact. A partial retirement is resumable using the same immutable proof.
    """
    native_path = Path(native_dataset).resolve()
    native = json.loads(native_path.read_text())
    data = verify_prepared(event_dataset)
    prepared = {e['match_key']: e for e in data.entries}
    owned_root = Path(owned_root).resolve()
    journal_path = native_path.parent/'retired-exports.json'
    prior = json.loads(journal_path.read_text()) if journal_path.exists() else None
    identity = dict(native_dataset_sha256=_digest(native_path),
                    event_dataset_sha256=_digest(Path(event_dataset)), owned_root=str(owned_root))
    if prior and any(prior[k] != value for k, value in identity.items()):
        raise ValueError('Retirement proof changed')
    if (len(native['entries']) != len(prepared)
            or set(prepared) != {e['match_key'] for e in native['entries']}):
        raise ValueError('Prepared batch is incomplete; cannot retire exports')
    rows = []
    for entry in native['entries']:
        path = Path(entry['trajectory']).resolve()
        if not path.is_relative_to(owned_root) or not path.name.endswith(('.training.jsonl', '.training.bin')):
            raise ValueError('Retirement target is not an owned native training export')
        source = prepared[entry['match_key']]['index']['source']
        if (source['sha256'] != entry['sha256'] or source.get('replay_sha256') != entry['replay_sha256']
                or prepared[entry['match_key']]['split'] != entry['split']):
            raise ValueError('Prepared source identity/split differs from native export')
        manifest = json.loads(manifest_path(path).read_text())
        if manifest['trajectory_sha256'] != entry['sha256']:
            raise ValueError('Native export manifest changed')
        if _digest(Path(source['original_replay'])) != entry['replay_sha256']:
            raise ValueError('Original replay changed')
        if path.exists():
            verify_entry(entry, deep=False)
        elif prior is None or not any(r['path'] == str(path) and r['sha256'] == entry['sha256'] for r in prior['files']):
            raise ValueError('Missing native export without a retirement journal')
        rows.append(dict(path=str(path), sha256=entry['sha256'],
            bytes=path.stat().st_size if path.exists() else next(r['bytes'] for r in prior['files'] if r['path'] == str(path))))
    # All checks finish before the first deletion, and a crash leaves this proof.
    journal = dict(identity, files=rows, complete=False)
    _write_json(journal_path, journal)
    for row in rows:
        Path(row['path']).unlink(missing_ok=True)
    journal.update(complete=True, retired_bytes=sum(r['bytes'] for r in rows))
    _write_json(journal_path, journal)
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--action-catalog', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--min-free-ram-mib', type=int, default=1024)
    parser.add_argument('--retire-owned-root')
    parser.add_argument('--manifest-name', default='dataset.json')
    parser.add_argument('--flat', action='store_true')
    parser.add_argument('--catalog')
    args = parser.parse_args()
    result = prepare_batch(args.dataset, args.output, args.config, action_catalog=args.action_catalog,
        minimum_ram=args.min_free_ram_mib*1024**2, workers=args.workers,
        manifest_name=args.manifest_name, flat=args.flat, catalog=args.catalog)
    if args.retire_owned_root:
        print(json.dumps(retire_exports(args.dataset, result, args.retire_owned_root)), flush=True)
        if args.catalog:
            from replay_catalog import connect as catalog_connect, import_retirement
            db = catalog_connect(args.catalog)
            try:
                import_retirement(db,Path(args.dataset).parent/'retired-exports.json')
                db.commit()
            finally:
                db.close()


if __name__ == '__main__':
    main()
