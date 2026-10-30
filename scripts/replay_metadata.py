"""Reusable, source/code-bound receipts from complete replay content audits.

Receipts contain vocabulary metadata and validation results, never feature rows.
They are optional: stale/corrupt receipts fall back to the ordinary full audit.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from replay_actions import ReplayActionAdapter

RECEIPT_FIELD = 'audit_metadata_receipt'
CACHE_DIRECTORY = Path(__file__).resolve().parents[1]/'training-data/replay-metadata-cache'
SCHEMA = 1


class AuditedReplayMetadata:
    """Collect checked records in the reference adapter's delivery order.

    One pending sample per player preserves first-discovery order and excludes
    each player's terminal command from vocabulary labels. Do not publish the
    document until audit/footer/EOF and provenance checks have all succeeded.
    """
    def __init__(self, accept=None):
        self.accept = accept
        self.header = self.terrain = None
        self.adapters, self.pending, self.players = {}, {}, {}
        self.types, self.templates, self.menus = set(), set(), {}

    def _accept(self, row):
        player = row['player_index']
        self.players[player] = self.players.get(player, 0)+1
        observation = row.get('observation')
        if observation is None:  # a light quiet sample: no entities, no actions
            if self.accept is not None:
                self.accept(row)
            return
        for key in ('own_entities', 'known_nonowned_entities'):
            entities = observation[key]
            self.types.update(entities.field_values('type') if hasattr(entities, 'field_values')
                              else (e['type'] for e in entities))
        if row['imitation_loss_mask'] and row['actions']:
            owned = {e['id']: e for e in observation['own_entities']}
            for action in row['actions']:
                if action['kind'] in ('build', 'train'):
                    actor = action.get('builder', action.get('producer'))
                    key = owned[actor]['type']+'|'+action['kind']
                    self.menus.setdefault(key, set()).add(action['template'])
                    self.templates.add(action['template'])
        if self.accept is not None:
            self.accept(row)

    def __call__(self, record):
        kind = record['record']
        if kind == 'header':
            self.header = record
            if not record.get('template_catalog'):
                raise ValueError('Re-export with the shared encoder: this file lacks a template catalog')
        elif kind == 'map':
            self.terrain = record
        elif kind == 'sample':
            player = record['player_index']
            if player not in self.adapters:
                self.adapters[player] = ReplayActionAdapter(self.header['template_catalog'])
            example = {'player_index': player, **self.adapters[player].decode(record)}
            if 'observation' in record:
                example['observation'] = record['observation']
            if player in self.pending:
                self._accept(self.pending[player])
            self.pending[player] = example
        elif kind == 'footer':
            for example in self.pending.values():
                example['imitation_loss_mask'] = False
                self._accept(example)
            self.pending.clear()

    def document(self):
        if self.header is None or self.terrain is None or self.pending:
            raise ValueError('Replay metadata are not complete')
        return dict(header=self.header, map=self.terrain, players=list(map(list, self.players.items())),
            types=sorted(self.types), templates=sorted(self.templates),
            menus={key:sorted(values) for key, values in sorted(self.menus.items())})


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def implementation_identity():
    # Include the audit, adapter, metadata delivery/merging, provenance, binary
    # reconciliation and message numbering implementations. Any change misses.
    directory = Path(__file__).resolve().parent
    paths = {name:directory/name for name in ('replay_metadata.py', 'replay_training.py',
        'replay_actions.py', 'replay_storage.py', 'audit_replay_commands.py',
        'harvest_china_replays.py', 'replay_revision_survey.py', 'replay_bootstrap.py', 'binary_replay.py')}
    paths['MessageStream.h'] = directory.parent/'GeneralsMD/Code/GameEngine/Include/Common/MessageStream.h'
    return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def provenance_identity(manifest):
    # The enclosing manifest may carry this receipt; avoid a recursive hash.
    # All other fields bind replay/runtime/map/INI/data/storage/native proofs.
    return _hash(None if manifest is None else {k:v for k, v in manifest.items() if k != RECEIPT_FIELD})


def make_receipt(source_sha256, manifest, summary, metadata, *, reconciliation=None, implementation=None):
    current = implementation_identity()
    if implementation is not None and implementation != current:
        raise ValueError('Replay audit implementation changed while collecting metadata')
    body = dict(schema_version=SCHEMA, source_sha256=source_sha256,
        provenance_sha256=provenance_identity(manifest), implementation=current,
        audit=summary, metadata=metadata, storage_verified=True, command_reconciliation=reconciliation)
    return {**body, 'receipt_sha256': _hash(body)}


def _valid(receipt, digest, manifest, implementation):
    try:
        body = {key:value for key, value in receipt.items() if key != 'receipt_sha256'}
        if (receipt['schema_version'] != SCHEMA or receipt['source_sha256'] != digest
                or receipt['provenance_sha256'] != provenance_identity(manifest)
                or receipt['implementation'] != implementation or receipt['storage_verified'] is not True
                or receipt['receipt_sha256'] != _hash(body)):
            return False
        metadata, summary = receipt['metadata'], receipt['audit']
        players = dict(metadata['players'])
        if (len(players) != len(metadata['players']) or set(players) != set(summary['players'])
                or any(type(player) is not int or type(count) is not int or count < 1 for player, count in players.items())
                or sum(players.values()) != summary['samples'] or summary['commands'] < 1
                or metadata['header']['record'] != 'header' or metadata['map']['record'] != 'map'):
            return False
        # If a raw replay is available, preserve independent command proof.
        # A cache collected without that file cannot substitute for reconciling
        # a replay which has since become available.
        if manifest is not None and Path(manifest['replay']).is_file():
            proof = receipt['command_reconciliation']
            if (proof is None or proof['all_controlled_commands_captured'] is not True
                    or proof['controlled_players'] != summary['players']
                    or proof['captured_commands'] != summary['commands']):
                return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def find_receipt(digest, manifest, *, implementation=None):
    implementation = implementation if implementation is not None else implementation_identity()
    if manifest is not None:
        candidate = manifest.get(RECEIPT_FIELD)
        if candidate is not None and _valid(candidate, digest, manifest, implementation):
            return candidate
    path = CACHE_DIRECTORY/(digest+'.json')
    try:
        # A corrupt/oversized optional cache must not allocate unbounded RAM.
        # This limits receipt use, never observations, entities or vocabulary.
        if path.stat().st_size > 16*1024**2:
            return None
        candidate = json.loads(path.read_text(encoding='utf-8'))
        return candidate if _valid(candidate, digest, manifest, implementation) else None
    except (OSError, ValueError, TypeError):
        return None


def cache_receipt(receipt):
    """Atomic, bounded per-source cache; optional read-only environments still work."""
    temporary = None
    try:
        data = _bytes(receipt)
        if len(data) > 16*1024**2:
            return False
        CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=CACHE_DIRECTORY, prefix='.metadata-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, CACHE_DIRECTORY/(receipt['source_sha256']+'.json'))
        return True
    except OSError:
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
