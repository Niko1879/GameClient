"""Audited, reiterable per-player views of native training exports."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from audit_replay_commands import reconcile
from prepared_io import _digest
from replay_bootstrap import match_key
from replay_metadata import (AuditedReplayMetadata, cache_receipt, find_receipt, implementation_identity,
                             make_receipt, provenance_identity)
from replay_sequences import ReplayRows
from replay_storage import manifest_path, verify_storage
from replay_training import audit


def load_replays(paths, *, use_metadata_cache=True):
    """Audit each export once; return per-player row views, observed vocabulary and proofs.

    Rows are lazy ReplayRows; the full audit, command reconciliation and receipt
    cache keep repeated preparation cheap without trusting unverified payloads.
    """
    games, provenance, types, templates, menus = [], [], set(), set(), {}
    for path in paths:
        initial_digest = _digest(path)
        sidecar_path = manifest_path(path)
        manifest = json.loads(sidecar_path.read_text(encoding='utf-8')) if sidecar_path.is_file() else None
        if manifest is not None and manifest['trajectory_sha256'] != initial_digest:
            raise ValueError('Export no longer matches its provenance manifest')
        implementation = implementation_identity()
        starting_provenance = provenance_identity(manifest)
        receipt = find_receipt(initial_digest, manifest, implementation=implementation) if use_metadata_cache else None
        captured_commands = None
        if receipt is not None:
            summary, metadata = receipt['audit'], receipt['metadata']
        else:
            collector, captured_commands = AuditedReplayMetadata(), Counter()
            summary = audit(path, command_counter=captured_commands, record_consumer=collector)
            metadata = collector.document()
        header, terrain = metadata['header'], metadata['map']
        types.update(metadata['types']); templates.update(metadata['templates'])
        for key, values in metadata['menus'].items():
            menus.setdefault(key, set()).update(values)
        rate = header['ticks_per_second']
        if rate not in (30, 60) or header['controlled_faction'] != 'FactionChina':
            raise ValueError('Expected a vanilla China 30 or 60 Hz export')
        types.update(header['template_catalog'].values())
        for player, length in dict(metadata['players']).items():
            games.append({'path': str(Path(path).resolve()), 'player': player, 'map': terrain,
                          'rows': ReplayRows(path, player, length, rate, validated=True),
                          'source_ticks_per_second': rate})
        source = {'path': str(Path(path).resolve()), 'sha256': _digest(path), 'audit': summary,
                  'engine': {k: header[k] for k in ('engine_git_sha', 'engine_git_dirty',
                                                   'engine_exe_crc', 'engine_ini_crc', 'ticks_per_second')}}
        if source['sha256'] != initial_digest:
            raise ValueError('Replay export changed while loading')
        if manifest is not None:
            manifest = json.loads(sidecar_path.read_text(encoding='utf-8'))
            if provenance_identity(manifest) != starting_provenance or manifest['trajectory_sha256'] != source['sha256']:
                raise ValueError('Export provenance changed while loading')
            if receipt is None:
                verify_storage(path, manifest)
            source['engine_sha256'] = manifest['engine_sha256']
            source['replay_sha256'] = manifest['replay_sha256']
            source['replay_exe_crc'] = manifest['replay_header']['exe_crc']
            source['replay_ini_crc'] = manifest['replay_header']['ini_crc']
            source['match_key'] = match_key(manifest['replay_header'], manifest.get('harvest_metadata'), manifest['replay'])
            original = Path(manifest['replay'])
            if original.is_file():
                source['original_replay'] = str(original.resolve())
                if _digest(original) != source['replay_sha256']:
                    raise ValueError('Original replay no longer matches its manifest')
                source['command_reconciliation'] = (receipt['command_reconciliation'] if receipt is not None
                    and receipt['command_reconciliation'] is not None else
                    reconcile(original, path, audit_summary=summary, captured_commands=captured_commands))
        if _digest(path) != initial_digest:
            raise ValueError('Replay export changed while loading')
        final = json.loads(sidecar_path.read_text(encoding='utf-8')) if sidecar_path.is_file() else None
        if provenance_identity(final) != starting_provenance:
            raise ValueError('Export provenance changed while loading')
        if source.get('original_replay') and _digest(source['original_replay']) != source['replay_sha256']:
            raise ValueError('Original replay changed while loading')
        if implementation_identity() != implementation:
            raise ValueError('Replay audit implementation changed while loading')
        if receipt is None and use_metadata_cache:
            cache_receipt(make_receipt(initial_digest, manifest, summary, metadata,
                reconciliation=source.get('command_reconciliation'), implementation=implementation))
        provenance.append(source)
    vocabulary = {'types': sorted(types), 'templates': sorted(templates),
                  'menus': {k: sorted(v) for k, v in sorted(menus.items())}}
    return games, vocabulary, provenance
