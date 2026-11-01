"""Whole-match identity, native export proofs and historical data packages."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from harvest_china_replays import Client, atomic_write
from replay_revision_survey import classify
from replay_training import audit, file_sha256, require
from replay_storage import manifest_path, verify_storage

KIND = 'generals_bootstrap_dataset'


def save(path, value):
    atomic_write(Path(path), (json.dumps(value, indent=2, allow_nan=False)+'\n').encode('utf-8'))


def match_key(header, metadata=None, replay_path=None):
    metadata = metadata or {}
    number = metadata.get('match_id')
    if number is None and replay_path:
        found = re.fullmatch(r'match_(\d+)(?:_.*)?', Path(replay_path).stem)
        number = int(found[1]) if found else None
    if number is not None:
        return 'pgo:'+str(int(number))
    # Recording-local timestamps and player slot are deliberately absent.
    options = dict(x.split('=', 1) for x in header['game_options'].split(';') if '=' in x)
    identity = [header['exe_crc'], header['ini_crc'], header['map_crc'], options.get('SD'),
                options.get('SC'), header['players']]
    require(options.get('SD') is not None, 'No match ID or deterministic seed for whole-match splitting')
    return 'game:'+hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def partition(key, fraction=.2):
    require(0 < fraction < 1, 'Validation fraction must be between zero and one')
    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')/2**64
    return 'validation' if bucket < fraction else 'train'


def verify_entry(entry, *, deep=True):
    trajectory = Path(entry['trajectory'])
    require(file_sha256(trajectory) == entry['sha256'], 'Dataset trajectory changed')
    manifest = json.loads(manifest_path(trajectory).read_text(encoding='utf-8'))
    require(manifest['trajectory_sha256'] == entry['sha256'], 'Dataset entry differs from export manifest')
    require(manifest['replay_sha256'] == entry['replay_sha256'], 'Dataset replay identity differs from manifest')
    source = Path(manifest['replay'])
    require(source.is_file() and file_sha256(source) == entry['replay_sha256'], 'Dataset source replay changed/missing')
    native = manifest['native_validation']
    status = classify(native, manifest['replay_header'], manifest['replay_stream'], 0)
    require(status == 'passed' or (status == 'crc_diverged' and manifest['audit'].get('verified_through_frame') is not None),
            'Dataset lacks full native compatibility evidence')
    require(native['ticks_per_second'] == entry['ticks_per_second'], 'Dataset tick rate differs from native evidence')
    require(manifest['engine_sha256'] == entry['engine_sha256'], 'Dataset backend identity differs')
    require(manifest['command_reconciliation']['all_controlled_commands_captured'], 'Commands were lost during export')
    if deep:
        from replay_metadata import (AuditedReplayMetadata, implementation_identity,
            provenance_identity, find_receipt, make_receipt, cache_receipt)
        implementation = implementation_identity()
        provenance = provenance_identity(manifest)
        refreshed = None
        if find_receipt(entry['sha256'], manifest, implementation=implementation) is None:
            verify_storage(trajectory, manifest)
            metadata, captured = AuditedReplayMetadata(), Counter()
            summary = audit(trajectory, command_counter=captured, record_consumer=metadata)
            from audit_replay_commands import reconcile
            reconciliation = reconcile(source, trajectory, audit_summary=summary, captured_commands=captured)
            refreshed = make_receipt(entry['sha256'], manifest, summary, metadata.document(),
                reconciliation=reconciliation, implementation=implementation)
        require(file_sha256(trajectory) == entry['sha256'], 'Dataset trajectory changed during verification')
        require(file_sha256(source) == entry['replay_sha256'], 'Dataset source replay changed during verification')
        current = json.loads(manifest_path(trajectory).read_text(encoding='utf-8'))
        require(provenance_identity(current) == provenance, 'Dataset provenance changed during verification')
        require(implementation_identity() == implementation, 'Audit implementation changed during verification')
        if refreshed is not None:
            cache_receipt(refreshed)
    return manifest


def check_live_provenance(metadata, executable_sha256, deployment):
    """A checkpoint plays only on the engine, INI and tick rate its training data declared."""
    require(executable_sha256 == deployment['engine_sha256'], 'Live executable differs from declared deployment')
    require(metadata['engine_ini_crc'] == deployment['engine_ini_crc'], 'Live INI differs from declared deployment')
    require(metadata['ticks_per_second'] == deployment['ticks_per_second'], 'Live rate differs from declared deployment')


def data_pack(row, root, download=False):
    """The checksum-selected INI package that reproduces a historical game revision."""
    crc = row['header']['ini_crc']
    packs = sorted(root.glob(crc+'/*_'+crc+'.big'))
    if not packs and download:
        url = row['metadata'].get('advertised_data_pack_url', '')
        parsed = urlparse(url)
        require(parsed.scheme == 'https' and parsed.hostname == 'strata.gamereplays.org'
                and re.fullmatch(r'[A-Za-z0-9_]+_'+crc+r'\.big', Path(parsed.path).name),
                'No supported advertised data package for '+crc)
        blob = Client(3).request(url)
        require(blob[:4] in (b'BIGF', b'BIG4'), 'Historical data download is not a BIG archive')
        path = root/crc/Path(parsed.path).name
        atomic_write(path, blob); packs = [path]
    require(packs, 'Missing historical data package '+crc)
    return packs[0]
