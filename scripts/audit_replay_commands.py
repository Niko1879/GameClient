"""Independently reconcile an export with the original binary replay messages."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import struct

from harvest_china_replays import parse_replay
from replay_training import audit, records


def message_names():
    header = Path(__file__).resolve().parents[1] / 'GeneralsMD/Code/GameEngine/Include/Common/MessageStream.h'
    text = header.read_text()
    section = text.split('MSG_BEGIN_NETWORK_MESSAGES = 1000,', 1)[1].split('MSG_END_NETWORK_MESSAGES', 1)[0]
    section = re.sub(r'/\*.*?\*/|//[^\n]*', '', section, flags=re.S)
    names = {1000: 'MSG_BEGIN_NETWORK_MESSAGES'}
    for index, name in enumerate(re.findall(r'\bMSG_[A-Z0-9_]+\b', section), 1001):
        names[index] = name
    return names


def binary_commands(path):
    data = Path(path).read_bytes()
    header = parse_replay(data)
    position = header['commands_offset']
    widths = (4, 4, 1, 4, 4, 4, 12, 8, 16, 4, 2)
    while position < len(data):
        if len(data) - position < 13:
            raise ValueError('Truncated replay message header')
        frame, kind, player, ntypes = struct.unpack_from('<IiiB', data, position)
        position += 13
        if len(data) - position < 2 * ntypes:
            raise ValueError('Truncated argument descriptors')
        descriptors = [struct.unpack_from('<BB', data, position + 2 * i) for i in range(ntypes)]
        position += 2 * ntypes
        for dtype, count in descriptors:
            if dtype >= len(widths) or count == 0:
                raise ValueError('Unknown replay argument descriptor')
            position += widths[dtype] * count
        if position > len(data):
            raise ValueError('Truncated replay arguments')
        yield frame, kind, player


def reconcile(replay, export, *, audit_summary=None, captured_commands=None):
    # A loader which just audited the same fingerprint can reuse that result.
    # Standalone reconciliation retains its full validation pass.
    if audit_summary is None:
        captured_commands = Counter()
        audit_summary = audit(export, command_counter=captured_commands)
    names = message_names()
    # Long matches can contain hundreds of megabytes of observations. Only
    # command identities are needed here; retain neither the samples nor fog.
    players, captured = set(audit_summary['players']), Counter()
    if captured_commands is None:
        for row in records(export):
            if row['record'] == 'sample':
                captured.update((row['frame'], a['type_id'], row['player_index']) for a in row['actions'])
    else:
        # Internal callers just collected these identities during their complete
        # content audit and verify source hashes before/after consuming them.
        captured.update(captured_commands)
    if sum(captured.values()) != audit_summary['commands']:
        raise ValueError('Audited command count does not match reconciliation input')
    # A trimmed export accounts for the replay's commands up to its verified frame.
    through = audit_summary.get('verified_through_frame')
    original = [row for row in binary_commands(replay) if through is None or row[0] <= through]
    ignored = {'MSG_BEGIN_NETWORK_MESSAGES', 'MSG_LOGIC_CRC', 'MSG_SET_REPLAY_CAMERA'}
    expected = Counter((f, k, p) for f, k, p in original
                       if p in players and k in names and names[k] not in ignored)
    if captured != expected:
        raise ValueError(f'Command capture mismatch: missing {list((expected - captured).items())[:10]}, '
                         f'extra {list((captured - expected).items())[:10]}')
    by_player = {}
    for p in sorted({p for _, _, p in original}):
        by_player[str(p)] = dict(Counter(names.get(k, str(k)) for _, k, player in original
                                        if player == p and k in names and names[k] not in ignored))
    return {'all_controlled_commands_captured': True, 'controlled_players': sorted(players),
            'captured_commands': sum(captured.values()), 'original_commands_by_player': by_player}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay')
    parser.add_argument('export')
    args = parser.parse_args()
    print(json.dumps(reconcile(args.replay, args.export), indent=2))
