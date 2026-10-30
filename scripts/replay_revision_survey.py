"""Sample historical vChina matches and measure CRC compatibility without training exports."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

from harvest_china_replays import (Archive, ArchiveLock, Client, MatchesFeed, atomic_write,
                                   crc32_matches, parse_replay, replay_release_exclusion)
from replay_training import file_sha256, replay_stream_summary


def save(path, value):
    atomic_write(Path(path), (json.dumps(value, indent=2) + "\n").encode("utf-8"))


def sample(args):
    client, rng = Client(args.delay), random.Random(args.seed)
    report = {"seed": args.seed, "sampling": "Random candidates from bounded pagination windows; not a global random census",
              "windows": [], "errors": []}
    with ArchiveLock(args.archive):
        archive = Archive(Path(args.archive), include_desynced=True)
        try:
            feed = MatchesFeed(client)
            for anchor in args.anchors:
                captured = []
                for page in range(anchor, anchor + args.pages_per_anchor):
                    rows = feed.page(page)
                    eligible = [r for r in rows if r["replay_url"] and len(r["factions"]) == 2 and "China" in r["factions"]]
                    rng.shuffle(eligible)
                    for row in eligible:
                        if len(captured) >= args.matches_per_anchor:
                            break
                        archive.discover(row)
                        try:
                            archive.harvest(client, row)
                            captured.append(row["match_id"])
                            print(f"Sampled page {page}: match {row['match_id']}", flush=True)
                        except (ValueError, OSError, KeyError) as error:
                            archive.defer(row["match_id"], error)
                            report["errors"].append({"match_id": row["match_id"], "error": str(error)})
                    if len(captured) >= args.matches_per_anchor or not rows:
                        break
                report["windows"].append({"anchor_page": anchor, "match_ids": captured})
                save(Path(args.archive) / "sample-report.json", report)
        finally:
            archive.db.close()


def classify(evidence, header, stream, returncode, log_text=""):
    # Setup problems must not inflate the measured rate of engine divergence.
    if evidence is None:
        if "Cannot open replay" in log_text:
            return "replay_rejected"
        return "engine_error"
    if not crc32_matches(evidence.get("engine_ini_crc"), header.get("ini_crc")):
        return "data_mismatch"
    if not crc32_matches(evidence.get("loaded_map_crc"), header.get("map_crc")):
        return "map_mismatch"
    if evidence["crc_comparisons"] <= 0:
        return "no_crc_evidence"
    if evidence["crc_mismatch"]:
        return "crc_diverged"
    if (returncode == 0 and evidence["completed"] and evidence["stream_eof"]
            and evidence["last_command_frame"] == stream["last_command_frame"]
            and evidence["final_frame"] >= stream["last_command_frame"]):
        return "passed"
    return "incomplete"


def wilson(successes, total):
    if not total:
        return None
    z, p = 1.959963984540054, successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0, centre - radius), min(1, centre + radius)]


def groups(cases):
    result = defaultdict(lambda: defaultdict(int))
    for case in cases:
        key = case.get("replay_exe_crc", "unknown") + "/" + case.get("replay_ini_crc", "unknown")
        result[key][case["status"]] += 1
    for counts in result.values():
        resolved = counts["passed"] + counts["crc_diverged"]
        counts["resolved"] = resolved
        counts["divergence_rate_among_resolved"] = counts["crc_diverged"] / resolved if resolved else None
        counts["wilson_95_interval"] = wilson(counts["crc_diverged"], resolved)
    return dict(result)


def probe(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    exe, user_data = Path(args.exe).resolve(), Path(args.user_data).resolve()
    data_root = Path(args.data_root).resolve()
    client = Client(args.delay)
    candidates, seen, excluded = defaultdict(list), set(), []
    selected_ids = set(args.match_ids or [])
    for archive in args.archives:
        for replay in sorted(Path(archive).glob("*/*.rep")):
            match = re.fullmatch(r"match_(\d+)(?:_.*)?", replay.stem)
            if selected_ids and (not match or int(match[1]) not in selected_ids):
                continue
            header = parse_replay(replay.read_bytes())
            exclusion = replay_release_exclusion(header['exe_crc'])
            if exclusion:
                excluded.append({'replay': str(replay), 'reason': exclusion})
                continue
            digest = file_sha256(replay)
            if digest not in seen and not header["desync"] and header["frames"] >= args.min_frames:
                seen.add(digest)
                candidates[(header["exe_crc"], header["ini_crc"])].append((replay.resolve(), header, digest))
    rng = random.Random(args.seed)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "engine": str(exe),
              "engine_sha256": file_sha256(exe), "seed": args.seed, "cases": [], "excluded": excluded,
              "scope": "Bounded historical sample; pass validates only this replay on the tested engine/rate/data/map",
              "unknown_recorded_tick_rate": "Never infer simulation Hz from max_fps or wall-clock match duration"}
    save(root / "survey.json", report)
    for key, options in sorted(candidates.items()):
        rng.shuffle(options)
        for replay, header, digest in options[:args.per_group]:
            case = {"replay": str(replay), "replay_sha256": digest, "replay_exe_crc": key[0],
                    "replay_ini_crc": key[1], "recorded_tick_rate": None, "header": header}
            log_path = root / (replay.stem + ".engine.log")
            started = time.monotonic()
            try:
                stream = replay_stream_summary(replay)
                command = [str(exe), "-headless", "-replay", os.path.relpath(replay, user_data / "Replays")]
                sidecar = replay.with_suffix(".json")
                metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
                url = metadata.get("advertised_data_pack_url")
                packs = sorted(data_root.glob(f"{key[1]}/*_{key[1]}.big"))
                if url and not packs:
                    filename = url.rsplit("/", 1)[-1]
                    if not re.fullmatch(r"[A-Za-z0-9_]+_" + key[1] + r"\.big", filename):
                        raise ValueError("Historical package URL does not identify expected INI CRC")
                    pack = data_root / key[1] / filename
                    blob = client.request(url)
                    if blob[:4] not in (b"BIGF", b"BIG4"):
                        raise ValueError("Historical package is not a BIG archive")
                    atomic_write(pack, blob)
                    packs = [pack]
                if packs:
                    command += ["-disableCommunityDataPatch", "-mod", str(packs[0])]
                    case["data_pack"] = {"path": str(packs[0]), "sha256": file_sha256(packs[0])}
                case["engine_command"] = command
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(command, cwd=exe.parent, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=args.timeout, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                text = log_path.read_text(encoding="utf-8", errors="replace")
                summaries = [json.loads(line[len("ReplayValidation "):]) for line in text.splitlines()
                             if line.startswith("ReplayValidation ")]
                evidence = summaries[-1] if summaries else None
                case.update(status=classify(evidence, header, stream, completed.returncode, text),
                            evidence=evidence, returncode=completed.returncode, replay_stream=stream)
                mismatch = re.search(r"CRC Mismatch in Frame (\d+)", text)
                if mismatch:
                    case["first_mismatch_frame"] = int(mismatch.group(1))
                if file_sha256(replay) != digest:
                    case.update(status="source_changed", error="Replay changed while probing")
            except subprocess.TimeoutExpired:
                case.update(status="timeout", error="Full-playback budget exceeded")
            except (ValueError, OSError, KeyError) as error:
                case.update(status="setup_error", error=str(error))
            case.update(wall_seconds=time.monotonic() - started, engine_log=str(log_path))
            report["cases"].append(case)
            report["groups"] = groups(report["cases"])
            save(root / "survey.json", report)
            print(f"{key[0]}/{key[1]} {replay.name}: {case['status']} ({case['wall_seconds']:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample_parser = sub.add_parser("sample")
    sample_parser.add_argument("--archive", default="replays/revision-survey")
    sample_parser.add_argument("--anchors", type=int, nargs="+", default=[1000, 2500, 5000])
    sample_parser.add_argument("--pages-per-anchor", type=int, default=5)
    sample_parser.add_argument("--matches-per-anchor", type=int, default=8)
    probe_parser = sub.add_parser("probe")
    probe_parser.add_argument("--archives", nargs="+", default=["replays/revision-survey", "replays/revision-probe"])
    probe_parser.add_argument("--exe", required=True)
    probe_parser.add_argument("--user-data", required=True)
    probe_parser.add_argument("--data-root", default="replay-runtime/revision-data")
    probe_parser.add_argument("--output", default="replay-runtime/revision-survey")
    probe_parser.add_argument("--per-group", type=int, default=8)
    probe_parser.add_argument("--min-frames", type=int, default=3600)
    probe_parser.add_argument("--timeout", type=int, default=120)
    probe_parser.add_argument("--match-ids", type=int, nargs="+", help="Restrict diagnostic reruns to these match IDs")
    for command in (sample_parser, probe_parser):
        command.add_argument("--seed", type=int, default=20261011)
        command.add_argument("--delay", type=float, default=3)
    args = parser.parse_args()
    if args.delay < 1 or any(getattr(args, name, 1) <= 0 for name in ("pages_per_anchor", "matches_per_anchor", "per_group", "timeout")):
        parser.error("Request delay >=1 and positive sample/probe budgets required")
    (sample if args.command == "sample" else probe)(args)


if __name__ == "__main__":
    main()
