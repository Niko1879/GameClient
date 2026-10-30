"""Export/audit vanilla China demonstrations. Standard library only; no training yet."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from collections import Counter
from replay_storage import open_trajectory, manifest_path

# Mirrors generals_policy.OBSERVATION_VERSION; this module stays importable without NumPy.
OBSERVATION_VERSION = 7


def _finite_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError('Nonfinite feature')
    return value


def _nonfinite_constant(token):
    raise ValueError('Nonfinite feature')


_decoder = json.JSONDecoder()
_finite_decoder = json.JSONDecoder(parse_float=_finite_float, parse_constant=_nonfinite_constant)


def records(path, *, check_finite=False, lazy=False, player=None, validated=False):
    decoder = _finite_decoder if check_finite else _decoder
    if Path(path).suffix.lower() == '.bin':
        from binary_replay import records as binary_records
        yield from binary_records(path, decoder=decoder, check_finite=check_finite,
                                  lazy=lazy, player=player, validated=validated)
        return
    with open_trajectory(path) as stream:
        for line in stream:
            record = decoder.decode(line)
            if player is None or record.get('record') != 'sample' or record['player_index'] == player:
                yield record


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite feature")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


# Owner-only fields that must never appear on a nonowned entity.
PRIVATE_FIELDS = frozenset(("production_queue", "ai_state", "construction_percent", "buildable_templates",
                            "trainable_templates", "builder_id", "producer_id", "has_weapon", "can_attack",
                            "has_atk_move_cmd", "can_attack_move", "dock_targets", "resume_targets",
                            "commands", "command_cooldowns", "command_targets", "enter_targets", "container_id",
                            "occupants", "capacity", "upgrades", "can_guard", "repair_targets", "repair_depots"))
# Canonical actions the engine resolves at dispatch (upgrades, abilities, garrison, promotions).
RESOLVED_KINDS = frozenset(("upgrade", "science", "ability", "enter", "exit", "evacuate",
                            "guard", "guard_air", "cancel_production", "cancel_construction", "scatter",
                            "add_waypoint", "force_attack", "force_attack_ground", "repair", "get_repaired"))


def _plain(entities):
    # Binary entity tables decode each entity once; JSON already holds lists.
    plain = getattr(entities, 'plain', None)
    return plain() if plain is not None else entities


# integrity: every block, record order, timing, commands, footer, CRC and EOF.
# full: also the per-entity observation contract (no leaked owner-only state,
# hidden sightings frozen, exclusions honoured). Use it to qualify an engine.
AUDIT_LEVELS = ('integrity', 'full')


def audit(path, *, level='integrity', command_counter=None, record_consumer=None):
    """Reject incomplete/divergent trajectories and, at level full, check observation invariants.

    A read-only consumer may collect metadata from each checked record. Its
    results remain provisional until this function validates the entire file.
    """
    require(level in AUDIT_LEVELS, 'Unknown audit level')
    full = level == 'full'
    scheduled = False
    header = terrain = footer = None
    previous = {}
    memory = {}
    commands_by_name = Counter()
    excluded_templates = set()
    visible_only_templates = set()
    neutral_excluded_templates, playable_players = set(), set()
    samples = waits = command_count = 0
    # Validate every numeric token while decoding, including overflowing
    # exponents, instead of traversing the entire decoded tree a second time.
    checked = (records(path, check_finite=True, lazy=True) if Path(path).suffix.lower() == '.bin'
               else records(path, check_finite=True))
    for record in checked:
        kind = record["record"]
        require(footer is None, "Record after footer")
        if kind == "header":
            require(header is None and samples == 0 and terrain is None, "Duplicate/misordered header")
            require(record["schema_version"] == 1, "Unsupported schema")
            require(record["controlled_faction"] == "FactionChina", "Wrong training faction")
            require(record["ticks_per_second"] > 0, "Invalid tick rate")
            policy = record.get('observation_policy', {})
            require(policy.get('version') == OBSERVATION_VERSION, 'Unsupported observation version')
            excluded_templates = set(policy['excluded_templates'])
            names = policy.get('visible_only_templates')
            require(isinstance(names, list) and all(isinstance(name, str) for name in names),
                    'Missing visible-only template contract')
            visible_only_templates = set(names)
            require(len(visible_only_templates) == len(names)
                    and not visible_only_templates.intersection(excluded_templates),
                    'Invalid visible-only template contract')
            # Scenery clutter is omitted only when a non-playable side controls it.
            names, players = policy.get('neutral_excluded_templates'), policy.get('playable_players')
            require(isinstance(names, list) and all(isinstance(name, str) for name in names)
                    and isinstance(players, list) and all(type(p) is int for p in players),
                    'Missing neutral entity contract')
            neutral_excluded_templates, playable_players = set(names), set(players)
            require(record.get('quiet_capture') in ('all', 'scheduled'), 'Unknown quiet capture mode')
            scheduled = record['quiet_capture'] == 'scheduled'
            # Entity invariants compare consecutive observations of a player.
            require(not (full and scheduled), 'The full audit requires every quiet observation')
            require(len(neutral_excluded_templates) == len(names)
                    and not neutral_excluded_templates.intersection(excluded_templates),
                    'Invalid neutral entity contract')
            header = record
        elif kind == "map":
            require(header is not None and terrain is None and samples == 0, "Misordered map")
            cells = record["grid_size"] ** 2
            require(len(record["height"]) == cells and len(record["cliff"]) == cells, "Bad terrain shape")
            terrain = record
        elif kind == "sample":
            require(header is not None and terrain is not None, "Sample without header/map")
            for action in record['actions']:
                if action['name'] == 'MSG_DOZER_CONSTRUCT':
                    result = action.get('build_result', {})
                    require(type(result.get('accepted')) is bool, 'Missing engine construction outcome')
                    require(result.get('status') in ('created', 'rejected', 'not_dispatched')
                            and result['accepted'] == (result['status'] == 'created'), 'Invalid construction outcome')
                if 'command_result' in action:
                    result = action['command_result']
                    require(type(result.get('resolved')) is bool, 'Missing command resolution')
                    require(result['action']['kind'] in RESOLVED_KINDS if result['resolved']
                            else isinstance(result.get('reason'), str), 'Invalid command resolution')
            player, frame = record["player_index"], record["frame"]
            prior = previous.get(player)
            require(frame >= 0 and (prior is None or frame > prior), "Nonmonotonic player frames")
            require(record["delta_ticks"] == (0 if prior is None else frame - prior), "Wrong sample interval")
            require(abs(record["time_seconds"] - frame / header["ticks_per_second"]) < 1e-6, "Wrong sample time")
            previous[player] = frame
            light = record.get('light') is True
            # Scheduled capture omits only quiet observations, never a player's first.
            require(not light or (scheduled and not record['actions'] and prior is not None),
                    'Invalid light sample')
            observation = None if light else record["observation"]
            require(light or observation.get('observation_policy_version') == OBSERVATION_VERSION,
                    'Observation version mismatch')
            require(light or observation["player"]["index"] == player, "Player identity mismatch")
            require(light or observation["player"]["faction"] == "FactionChina", "Non-vanilla China sample")
            require(light or len(observation["visibility_grid"]) == terrain["grid_size"] ** 2, "Bad visibility shape")
            require(light or set(observation["visibility_grid"]) <= {0, 1, 2}, "Invalid visibility status")
            if full:
                _check_entities(observation, player, frame, memory, excluded_templates, visible_only_templates,
                                neutral_excluded_templates, playable_players)
            actions = record["actions"]
            require(record["decision"] == ("commands" if actions else "wait"), "Decision/action mismatch")
            for action in actions:
                require(action["name"] not in ("MSG_LOGIC_CRC", "MSG_SET_REPLAY_CAMERA"), "Nonpolicy message included")
                require(isinstance(action["arguments"], list), "Missing typed arguments")
                commands_by_name[action["name"]] += 1
                if command_counter is not None:
                    command_counter[(frame, action['type_id'], player)] += 1
            samples += 1
            waits += not actions
            command_count += len(actions)
        elif kind == "footer":
            footer = record
        else:
            raise ValueError(f"Unknown record {kind}")
        if record_consumer is not None:
            record_consumer(record)
    require(header is not None and terrain is not None and footer is not None, "Incomplete trajectory")
    require(footer["valid"] and footer["playback_completed"] and not footer["crc_mismatch"],
            "Playback failed or was not training eligible")
    require(footer["recorded_crc_messages_seen"] > 0, "No recorded CRC coverage")
    if "recorded_crc_messages_compared" in footer:
        require(0 < footer["recorded_crc_messages_compared"] <= footer["recorded_crc_messages_seen"],
                "No actual recorded CRC comparisons, or invalid comparison count")
        require(footer.get("replay_stream_eof") is True, "Replay command stream did not reach clean EOF")
        require(footer["final_frame"] >= footer["last_recorded_command_frame"], "Replay stopped before its last command")
    require(samples > 0 and command_count > 0, "Empty demonstration")
    require((samples, command_count, waits) == (footer["samples"], footer["commands"], footer["wait_samples"]),
            "Footer counts disagree")
    require(all(f <= footer["final_frame"] for f in previous.values()), "Sample past replay end")
    result = {"schema_version": 1, "samples": samples, "commands": command_count, "wait_samples": waits,
            "players": sorted(previous), "final_frame": footer["final_frame"],
            "recorded_crc_messages_seen": footer["recorded_crc_messages_seen"],
            "commands_by_name": dict(commands_by_name)}
    if "recorded_crc_messages_compared" in footer:
        result.update({key: footer[key] for key in ("recorded_crc_messages_compared", "replay_stream_eof",
                                                  "last_recorded_command_frame")})
    result['observation_policy_version'] = header['observation_policy']['version']
    for field in ('observed_sighting_removals', 'temporary_sightings_forgotten'):
        if field in footer:
            counts = footer[field]
            require(isinstance(counts, list) and all(type(n) is int and n >= 0 for n in counts),
                    'Invalid sighting retirement counters')
            result[field] = counts
    result['placement_mode'] = header['placement_mode']
    result['quiet_capture'] = header['quiet_capture']
    result['audit_level'] = level
    return result


def _check_entities(observation, player, frame, memory, excluded_templates, visible_only_templates,
                    neutral_excluded_templates, playable_players):
    # Owned checks read a few fields lazily; frozen sightings compare every field.
    own_entities = list(observation['own_entities'])
    known_entities = _plain(observation['known_nonowned_entities'])
    require(all(isinstance(e.get('resume_targets'), list) for e in own_entities),
            'Missing owned resume construction legality')
    require(all(e.get('type') not in excluded_templates for entities in
                (own_entities, known_entities) for e in entities),
            'Excluded transient/effect entity entered observation')
    require(all(e.get('type') not in neutral_excluded_templates or e['owner'] in playable_players
                for entities in (own_entities, known_entities) for e in entities),
            'Neutral scenery entity entered observation')
    identities = set()
    for entity in own_entities:
        require(entity["id"] not in identities, "Duplicate entity")
        identities.add(entity["id"])
        require(entity["own"] and entity["owner"] == player and entity["visible"], "Wrong owned entity")
        require(entity["last_seen_frame"] == frame, "Stale owned entity")
    player_memory = memory.setdefault(player, {})
    for entity in known_entities:
        require(entity["id"] not in identities, "Duplicate entity")
        identities.add(entity["id"])
        require(not entity["own"] and entity["owner"] != player, "Wrong nonowned entity")
        require(entity.keys().isdisjoint(PRIVATE_FIELDS), "Private nonowned state leaked")
        require(entity["last_seen_frame"] <= frame, "Future sighting")
        frozen = dict(entity)
        frozen.pop("visible", None)
        if entity["visible"]:
            require(entity["last_seen_frame"] == frame, "Stale visible entity")
        else:
            require(entity.get('type') not in visible_only_templates, 'Hidden temporary sighting retained')
            require(player_memory.get(entity["id"]) == frozen, "Hidden sighting changed or was never seen")
        player_memory[entity["id"]] = frozen
    # An absent track cannot later reappear hidden using obsolete
    # evidence. Visible observations may establish the ID afresh.
    memory[player] = {e['id']: player_memory[e['id']] for e in known_entities}


def iter_samples(path):
    """Validate the entire file before yielding any examples to a training consumer.

    Entity IDs and action arguments are replay-local. Keep samples sequential;
    split evaluation by match, never by frame. No outcome/reward is fabricated.
    """
    audit(path)
    for record in records(path):
        if record["record"] == "sample":
            yield record


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_stream_summary(replay):
    # Independent binary preflight prevents truncated arguments being mistaken
    # for a normal EOF by the legacy engine reader. Header duration is a hint.
    from audit_replay_commands import binary_commands
    previous, count = -1, 0
    for frame, _, _ in binary_commands(replay):
        require(previous <= frame < 2 ** 31, "Replay command frames are not monotonic or exceed engine range")
        previous, count = frame, count + 1
    require(count > 0, "Replay has no complete recorded commands")
    return {"messages": count, "last_command_frame": previous}


class ExportPlaybackError(ValueError):
    """A rejected native playback, with routing evidence from that same run."""
    def __init__(self, status, validation, returncode, log):
        super().__init__(f'Export playback {status}; engine exit {returncode}; see {log}')
        self.evidence = dict(status=status, validation=validation, returncode=returncode, log=str(log))


def export(args):
    exe, replay, root, user_data = map(lambda p: Path(p).resolve(),
                                    (args.exe, args.replay, args.output, args.user_data))
    require(exe.is_file() and replay.is_file(), "Executable/replay missing")
    require(user_data.is_dir(), "User-data directory missing")
    # Use the same parser as the harvesting pipeline, including Random headers.
    from harvest_china_replays import parse_replay, replay_release_exclusion
    replay_header = parse_replay(replay.read_bytes())
    exclusion = replay_release_exclusion(replay_header['exe_crc'])
    require(exclusion is None, exclusion)
    require(not replay_header["desync"], "Replay header marks a desync")
    replay_digest = file_sha256(replay)
    stream_summary = replay_stream_summary(replay)
    data_pack, data_pack_digest = None, None
    if getattr(args, "data_pack", None):
        data_pack = Path(args.data_pack).resolve()
        require(data_pack.is_file(), "Historical data package missing")
        with data_pack.open("rb") as stream:
            require(stream.read(4) in (b"BIGF", b"BIG4"), "Historical data package is not a BIG archive")
        data_pack_digest = file_sha256(data_pack)
    version = replay_header["exe_crc"]
    destination = root / version
    destination.mkdir(parents=True, exist_ok=True)
    require(args.storage in ('binary', 'jsonl'), 'Storage must be binary or jsonl')
    binary = args.storage == 'binary'
    trajectory = destination / (replay.name + (".training.bin" if binary else ".training.jsonl"))
    require(not trajectory.exists(), f"Export already exists: {trajectory}; choose a new output directory")
    require(not manifest_path(trajectory).exists(), 'Export/manifest already exists; choose a new output directory')
    incomplete = [Path(str(trajectory)+suffix) for suffix in ('.part', '.failed')]
    require(not any(path.exists() for path in incomplete),
            'Incomplete export already exists; choose a new output directory')
    replay_arg = os.path.relpath(replay, user_data / "Replays")
    command = [str(exe), "-headless", "-replay", replay_arg,
               "-exportTraining", str(destination), "-trainingInterval", str(args.interval_ms)]
    command.extend(['-trainingPlacement', getattr(args, 'placement_mode', 'fast')])
    if binary:
        command.extend(['-trainingFormat', args.storage])
    # An earlier release's game logic, for replays it recorded (engine Common/ReplayLogic.h).
    logic = getattr(args, 'replay_logic', 'current')
    require(logic in ('current', '081326'), 'Unknown replay logic')
    if logic != 'current':
        command.extend(['-replayLogic', logic])
    # scheduled: quiet observations only where the event schedule can use them
    # (scheduled_capture.py); preparation output is unchanged.
    quiet_capture = getattr(args, 'quiet_capture', 'all')
    require(quiet_capture in ('all', 'scheduled'), 'Quiet capture must be all or scheduled')
    controller = None
    if quiet_capture == 'scheduled':
        require(binary and getattr(args, 'capture_policy', None), 'Scheduled capture needs binary storage and a policy')
        command.extend(['-trainingQuietCapture', 'scheduled'])
        from scheduled_capture import CaptureController
        controller = CaptureController(args.capture_policy, replay, trajectory)
    if data_pack is not None:
        # Mount one historical override for this process. Do not combine it
        # with the user's current community patch or replace installed data.
        command.extend(["-disableCommunityDataPatch", "-mod", str(data_pack)])
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_path = destination / (replay.name + ".engine.log")
    from scheduled_capture import run_engine
    try:
        returncode = run_engine(command, cwd=exe.parent, log_path=log_path, timeout=args.timeout,
                                creationflags=creationflags, controller=controller)
    except subprocess.TimeoutExpired as error:
        for path in [trajectory, *incomplete]:
            path.unlink(missing_ok=True)
        raise ExportPlaybackError('timeout', None, None, log_path) from error
    # This is the compatibility playback too. Reject before hashing/auditing a
    # failed trajectory, and let routing continue using this run's evidence.
    native_text = log_path.read_text(encoding='utf-8', errors='replace')
    native_rows = [json.loads(line[len('ReplayValidation '):])
                   for line in native_text.splitlines()
                   if line.startswith('ReplayValidation ')]
    from replay_revision_survey import classify
    native = native_rows[-1] if native_rows else None
    status = classify(native, replay_header, stream_summary, returncode, native_text)
    if status == 'passed' and getattr(args, 'ticks_per_second', None) is not None:
        if native['ticks_per_second'] != args.ticks_per_second:
            status = 'rate_mismatch'
    if status != 'passed':
        # These paths did not exist before this invocation. Keep native logs
        # and structured evidence, rather than potentially huge rejected rows.
        for path in [trajectory, *incomplete]:
            path.unlink(missing_ok=True)
        raise ExportPlaybackError(status, native, returncode, log_path)
    trajectory_digest = file_sha256(trajectory)
    captured_commands = Counter()
    from replay_metadata import AuditedReplayMetadata, RECEIPT_FIELD, implementation_identity, make_receipt
    metadata = AuditedReplayMetadata()
    implementation = implementation_identity()
    summary = audit(trajectory, level=getattr(args, 'audit_level', 'integrity'),
                    command_counter=captured_commands, record_consumer=metadata)
    require(file_sha256(replay) == replay_digest, "Replay changed during export")
    if data_pack is not None:
        require(file_sha256(data_pack) == data_pack_digest, "Historical data package changed during export")
    require(summary["final_frame"] >= stream_summary["last_command_frame"], "Export missed the end of the recorded stream")
    if "last_recorded_command_frame" in summary:
        require(summary["last_recorded_command_frame"] == stream_summary["last_command_frame"],
                "Engine and independent parser disagree about the final recorded command")
    from audit_replay_commands import reconcile
    reconciliation = reconcile(replay, trajectory, audit_summary=summary, captured_commands=captured_commands)
    require(file_sha256(trajectory) == trajectory_digest, 'Export changed during auditing')
    manifest = {"schema_version": 1, "replay": str(replay), "replay_sha256": replay_digest,
                "replay_stream": stream_summary,
                "replay_header": replay_header, "engine": str(exe), "engine_sha256": file_sha256(exe),
                "engine_command": command, "compatibility_evidence": "completed engine recorded CRC checks",
                "trajectory": str(trajectory), "trajectory_sha256": trajectory_digest, "audit": summary,
                "native_validation": native, "command_reconciliation": reconciliation}
    if binary:
        manifest['storage'] = {'schema_version': 1, 'codec': 'gobs1',
                               'stored_bytes': trajectory.stat().st_size}
    if data_pack is not None:
        manifest["data_pack_override"] = {"path": str(data_pack), "sha256": data_pack_digest,
                                           "installed_community_patch_disabled": True}
    sidecar = replay.with_suffix(".json")
    if sidecar.is_file():
        manifest["harvest_metadata"] = json.loads(sidecar.read_text(encoding="utf-8"))
    # Publish only after the complete audit, independent binary reconciliation
    # and native map/CRC/EOF checks have all succeeded.
    manifest[RECEIPT_FIELD] = make_receipt(manifest['trajectory_sha256'], manifest, summary,
        metadata.document(), reconciliation=reconciliation, implementation=implementation)
    from harvest_china_replays import atomic_write
    atomic_write(manifest_path(trajectory), (json.dumps(manifest, indent=2)+'\n').encode('utf-8'))
    print(json.dumps({"trajectory": str(trajectory), **summary}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exp = commands.add_parser("export")
    exp.add_argument("--exe", required=True, help="Instrumented executable in an asset-populated runtime")
    exp.add_argument("--replay", required=True)
    exp.add_argument("--user-data", required=True, help="Engine Documents game user-data directory")
    exp.add_argument("--output", default="training-data/vanilla-china")
    exp.add_argument("--data-pack", help="Historical BIG package to mount instead of the installed community patch")
    exp.add_argument("--interval-ms", type=int, default=100)
    exp.add_argument('--placement-mode', choices=('fast','full_mask'), default='fast')
    exp.add_argument("--timeout", type=int, default=180)
    exp.add_argument('--storage', choices=('binary', 'jsonl'), default='binary',
                     help='binary: typed, block-compressed observations; jsonl: readable transport')
    exp.add_argument('--audit-level', choices=AUDIT_LEVELS, default='integrity',
                     help='full adds the per-entity observation contract; use it to qualify an engine')
    check = commands.add_parser("audit")
    check.add_argument("trajectory")
    check.add_argument('--level', choices=AUDIT_LEVELS, default='integrity')
    args = parser.parse_args()
    try:
        if args.command == "export":
            require(1 <= args.interval_ms <= 10000 and args.timeout > 0, "Invalid interval/timeout")
            export(args)
        else:
            print(json.dumps(audit(args.trajectory, level=args.level), indent=2))
    except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
