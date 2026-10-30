#!/usr/bin/env python3
"""Harvest public vanilla-China 1v1 replays, grouped by advertised executable CRC.

Python standard library only. Does not launch games or install dependencies.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import http.cookiejar
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sqlite3
import struct
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import build_opener, HTTPCookieProcessor, Request
from urllib.robotparser import RobotFileParser

BASE = "https://strata.gamereplays.org"
MATCHES = BASE + "/zh/matches"
REPLAY_HOST = "matchdata.playgenerals.online"
FACTIONS = {"USA", "China", "GLA", "USA Super Weapon", "USA Laser",
            "USA Airforce", "China Tank", "China Infantry", "China Nuke",
            "GLA Toxin", "GLA Demo", "GLA Stealth", "Random"}
TEMPLATES = {2: "USA", 3: "China", 4: "GLA", 5: "USA Super Weapon",
             6: "USA Laser", 7: "USA Airforce", 8: "China Tank",
             9: "China Infantry", 10: "China Nuke", 11: "GLA Toxin",
             12: "GLA Demo", 13: "GLA Stealth", -1: "Random"}
MAX_BYTES = 32 * 1024 * 1024
# User decision, 2026-10-21: park the September 23 release, which has no
# qualified exporter. Match executable CRCs, not ambiguous advertised labels.
BLACKLISTED_REPLAY_CRCS = frozenset({"FBA615FA", "1F44693B"})


def replay_release_exclusion(exe_crc):
    """Return the collection/training exclusion reason, leaving raw parsing intact."""
    crc = str(exe_crc).upper()
    if crc in BLACKLISTED_REPLAY_CRCS:
        return (f"Blacklisted release 092226_QFE1 ({crc}): no qualified exporter; "
                "excluded by user on 2026-10-21")
    return None


class ArchiveLock:
    """OS-owned lock: a crash releases it without stale PID-file guessing."""
    def __init__(self, root):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.stream = (root / "collector.lock").open("a+b")
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise ValueError(f"Another collector is using {root}") from error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stream.close()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def crc32_matches(left, right):
    """Compare unsigned 32-bit hex CRCs, accepting omitted leading zeroes only."""
    return (isinstance(left, str) and isinstance(right, str)
            and re.fullmatch(r"[0-9A-Fa-f]{1,8}", left) is not None
            and re.fullmatch(r"[0-9A-Fa-f]{1,8}", right) is not None
            and int(left, 16) == int(right, 16))


def map_identity(name):
    """Ignore client-specific virtual paths, retaining the exact map revision."""
    return " ".join((name or "").replace("\\", "/").rsplit("/", 1)[-1].casefold().split())


def listing_matches_map(row, name):
    # Listings can omit [RANK]. The downloaded header/CRC is authoritative.
    label = map_identity(name).removeprefix("[rank] ")
    return not name or label in " ".join(row["text"].casefold().split())


class PageParser(HTMLParser):
    """Read actual match rows and Livewire attributes without running JavaScript."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.snapshot = self.lazy = self.csrf = self.endpoint = None
        self.rows = []
        self.row = None
        self.ignore = 0
        self.text = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "wire:snapshot" in a:
            snapshot = json.loads(a["wire:snapshot"])
            if snapshot["memo"]["name"] == "matches.matches-page":
                self.snapshot = a["wire:snapshot"]
                lazy = re.search(r"__lazyLoad\('([^']+)'\)", a.get("x-intersect", ""))
                if lazy:
                    self.lazy = lazy.group(1)
        if "data-csrf" in a:
            self.csrf = a["data-csrf"]
            self.endpoint = a.get("data-update-uri")
        if tag in ("script", "style"):
            self.ignore += 1
        if tag == "tr" and a.get("wire:key", "").startswith("match-"):
            self.row = {"factions": [], "links": [], "text": []}
        if self.row is not None:
            if tag == "img" and a.get("alt") in FACTIONS:
                self.row["factions"].append(a["alt"])
            if tag == "a" and "href" in a:
                self.row["links"].append(a["href"])

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.ignore = max(0, self.ignore - 1)
        if tag == "tr" and self.row is not None:
            row = self.row
            self.row = None
            urls = [u for u in row["links"] if re.fullmatch(BASE + r"/zh/match/\d+", u)]
            replays = [u for u in row["links"] if urlparse(u).hostname == REPLAY_HOST
                       and urlparse(u).path.endswith(".rep")]
            if urls:
                row.update(match_id=int(urls[0].rsplit("/", 1)[1]), url=urls[0],
                           replay_url=replays[0] if replays else None,
                           text=" ".join(row["text"]))
                del row["links"]
                self.rows.append(row)

    def handle_data(self, data):
        if not self.ignore and data.strip():
            self.text.append(data.strip())
            if self.row is not None:
                self.row["text"].append(data.strip())


def parse_page(markup):
    parser = PageParser()
    parser.feed(markup)
    return parser


def parse_replay(data):
    """Parse the retail-compatible header using Recorder.cpp's field layout."""
    if not data.startswith(b"GENREP") or len(data) < 40:
        raise ValueError("Download is not a GENREP replay")
    p = 6

    def unpack(fmt):
        nonlocal p
        size = struct.calcsize(fmt)
        if p + size > len(data):
            raise ValueError("Truncated replay header")
        values = struct.unpack_from(fmt, data, p)
        p += size
        return values

    def string(wide=False):
        nonlocal p
        start = p
        step = 2 if wide else 1
        terminator = b"\0\0" if wide else b"\0"
        while p + step <= len(data) and p - start <= 4096:
            if data[p:p + step] == terminator:
                value = data[start:p].decode("utf-16le" if wide else "latin1")
                p += step
                return value
            p += step
        raise ValueError("Unterminated or oversized replay string")

    start, end, frames = unpack("<iiI")
    desync, quit_early = unpack("<BB")
    disconnections = list(unpack("<8B"))
    name = string(True)
    system_time = list(unpack("<8H"))
    version, build_time = string(True), string(True)
    version_number, exe_crc, ini_crc = unpack("<III")
    options = string()
    local_player = int(string())
    difficulty, game_mode, rank_points, max_fps = unpack("<4i")
    if not -1 <= local_player < 8 or not exe_crc or not ini_crc:
        raise ValueError("Invalid replay header fields")
    if len(data) < p + 13:
        raise ValueError("Replay has no command stream")
    # S is the final field. Player names may contain colons and commas, so
    # splitting the slot list on ':' corrupts otherwise valid human slots.
    slot_list = re.search(r"(?:^|;)S=(.*);$", options, re.S)
    if not slot_list:
        raise ValueError("Replay has no slot list")
    prefix = options[:slot_list.start()]
    fields = dict(x.split("=", 1) for x in prefix.split(";") if "=" in x)
    players = []
    slots = slot_list.group(1)
    cursor, slot = 0, 0
    human = re.compile(r"H(.*?),([0-9A-Fa-f]+),(\d+),([TF]{2}),(-?\d+),"
                       r"(-?\d+),(-?\d+),(-?\d+),(-?\d+):", re.S)
    while cursor < len(slots):
        if slot >= 8:
            raise ValueError("Replay has more than eight slots")
        if slots[cursor:cursor + 2] in ("X:", "O:"):
            cursor += 2
            slot += 1
            continue
        if slots[cursor] == "C":
            raise ValueError("Replay contains an AI slot; expected human 1v1")
        match = human.match(slots, cursor)
        if not match:
            raise ValueError("Malformed human replay slot")
        parts = match.groups()
        cursor = match.end()
        template = int(parts[5])
        if template in (-2, 1):  # observer slots, not playing participants
            slot += 1
            continue
        players.append({"slot": slot, "name": parts[0], "template_id": template,
                        "header_faction": TEMPLATES.get(template, "Unknown"),
                        "team": int(parts[7])})
        slot += 1
    if len(players) != 2:
        raise ValueError(f"Replay has {len(players)} active humans; expected 2")
    if players[0]["team"] >= 0 and players[0]["team"] == players[1]["team"]:
        raise ValueError("Replay players are on the same team")
    return {"start_time": start, "end_time": end, "frames": frames,
            "desync": bool(desync), "quit_early": bool(quit_early),
            "player_disconnections": disconnections, "name": name,
            "system_time": system_time, "version_string": version,
            "build_time": build_time, "version_number": version_number,
            "exe_crc": f"{exe_crc:08X}", "ini_crc": f"{ini_crc:08X}",
            "game_options": options, "players": players, "local_player": local_player,
            "difficulty": difficulty, "game_mode": game_mode, "max_fps": max_fps,
            "map": fields.get("M"), "map_crc": fields.get("MC"),
            "map_size": fields.get("MS"), "starting_cash": fields.get("SC"),
            "crc_interval": fields.get("C"), "commands_offset": p,
            "rank_points": rank_points}


class Client:
    def __init__(self, delay=2, retries=3):
        self.opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.delay, self.retries, self.last_request = delay, retries, 0
        self.robots = {}

    def request(self, url, payload=None, check_robots=True):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in (urlparse(BASE).hostname, REPLAY_HOST):
            raise ValueError(f"Unexpected download host: {url}")
        origin = f"https://{parsed.hostname}"
        if check_robots and origin not in self.robots:
            robot = RobotFileParser()
            try:
                raw = self.request(origin + "/robots.txt", check_robots=False)
                robot.parse(raw.decode("utf-8", errors="replace").splitlines())
            except HTTPError as error:
                if error.code != 404:
                    raise
                robot.parse(["User-agent: *", "Disallow:"])
            self.robots[origin] = robot
        if check_robots:
            robot = self.robots[origin]
            if not robot.can_fetch("GeneralsAIReplayCollector", url):
                raise ValueError(f"robots.txt disallows {url}")
            delay = max(self.delay, robot.crawl_delay("*") or 0)
        else:
            delay = self.delay
        for attempt in range(self.retries):
            time.sleep(max(0, delay - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            headers = {"User-Agent": "Mozilla/5.0 GeneralsAIReplayCollector/1.0"}
            body = None
            if payload is not None:
                body = json.dumps(payload).encode()
                headers.update({"Content-Type": "application/json", "X-Livewire": "",
                                "Referer": MATCHES})
            try:
                with self.opener.open(Request(url, data=body, headers=headers), timeout=60) as response:
                    final = urlparse(response.url)
                    if final.scheme != "https" or final.hostname not in (parsed.hostname,):
                        raise ValueError("Unexpected HTTP redirect")
                    data = response.read(MAX_BYTES + 1)
                    if len(data) > MAX_BYTES:
                        raise ValueError("Response exceeds 32 MiB")
                    return data
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt + 1 == self.retries:
                    raise
                retry = error.headers.get("Retry-After", "")
                time.sleep(min(60, int(retry)) if retry.isdigit() else 2 ** (attempt + 1))
            except (URLError, TimeoutError):
                if attempt + 1 == self.retries:
                    raise
                time.sleep(2 ** (attempt + 1))


class MatchesFeed:
    def __init__(self, client):
        self.client = client
        page = parse_page(client.request(MATCHES).decode("utf-8"))
        self.snapshot, self.csrf, self.endpoint = page.snapshot, page.csrf, page.endpoint
        if not all((self.snapshot, self.csrf, self.endpoint, page.lazy)):
            raise ValueError("Matches page loading format changed")
        self.update({}, [{"path": "", "method": "__lazyLoad", "params": [page.lazy]}])

    def update(self, updates, calls):
        raw = self.client.request(self.endpoint, {"_token": self.csrf, "components": [
            {"snapshot": self.snapshot, "updates": updates, "calls": calls}]})
        answer = json.loads(raw)
        component = answer["components"][0]
        self.snapshot = component["snapshot"]
        state = json.loads(self.snapshot)
        if state["memo"].get("errors"):
            raise ValueError(f"Feed rejected request: {state['memo']['errors']}")
        return parse_page(component["effects"]["html"])

    def page(self, number):
        page = self.update({"amount": 100, "filters.matchTypes": ["1v1"],
                            "filters.hasReplay": True},
                           [{"path": "", "method": "gotoPage", "params": [number]}])
        state = json.loads(self.snapshot)["data"]
        filters = state["filters"][0]
        if (state["amount"] != 100 or filters["matchTypes"][0] != ["1v1"]
                or filters["hasReplay"] is not True
                or state["paginators"][0].get("page") != number):
            raise ValueError("Server did not apply the expected 1v1/replay/page filters")
        return page.rows


def replay_links(markup):
    """Replay downloads on the replay host, in page order.

    Since October 2026 the Matches listing omits download links; each match's
    detail page carries them (one per uploading player).
    """
    links = []
    for href in re.findall(r'href="([^"]+)"', markup):
        url = html.unescape(href)
        if urlparse(url).hostname == REPLAY_HOST and urlparse(url).path.endswith(".rep") and url not in links:
            links.append(url)
    return links


def version_metadata(markup):
    page = parse_page(markup)
    text = " ".join(page.text)
    match = re.search(r"Game Version:\s*(\S+)", text)
    label = match.group(1) if match else None
    checksum = label.upper() if label and re.fullmatch(r"[0-9A-Fa-f]{8}", label) else None
    pack = re.search(r'href="([^"]*/storage/versions/ini/[^\"]+)"', markup)
    return {"label": label, "crc": checksum, "data_pack_url": pack.group(1) if pack else None}


def advertised_version(markup):
    """Legacy checksum-only API; release names never imply a checksum/commit."""
    version = version_metadata(markup)
    match = version["crc"]
    if not match:
        raise ValueError("Match detail has no advertised game-version CRC")
    return match, version["data_pack_url"]


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
    temporary.replace(path)


class Archive:
    def __init__(self, root, *, map_name=None, map_crc=None, include_desynced=False):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "harvest.sqlite3")
        self.db.execute("CREATE TABLE IF NOT EXISTS matches (id INTEGER PRIMARY KEY, "
                        "candidate TEXT NOT NULL, status TEXT NOT NULL, target TEXT, "
                        "sha256 TEXT, error TEXT, updated TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(matches)")}
        for name, definition in (("attempts", "INTEGER NOT NULL DEFAULT 0"), ("retry_after", "TEXT")):
            if name not in columns:
                self.db.execute(f"ALTER TABLE matches ADD COLUMN {name} {definition}")
        self.include_desynced = include_desynced
        self.map_name, self.map_crc = map_identity(map_name), (map_crc or "").upper()
        selection = json.dumps([self.map_name, self.map_crc])
        previous = self.db.execute("SELECT value FROM state WHERE key='map_selection'").fetchone()
        if previous and previous[0] != selection:
            self.db.close()
            raise ValueError("Map filter changed; use a separate archive directory/checkpoint")
        if not previous and (self.map_name or self.map_crc) and self.db.execute(
                "SELECT COUNT(*) FROM matches").fetchone()[0]:
            self.db.close()
            raise ValueError("Use a fresh archive directory for a filtered crawl")
        self.db.execute("INSERT OR REPLACE INTO state VALUES('map_selection',?)", (selection,))
        self.db.commit()

    def discover(self, row):
        previous = self.db.execute("SELECT candidate,status FROM matches WHERE id=?",
                                   (row["match_id"],)).fetchone()
        if previous:
            known = json.loads(previous[0]).get("replay_url")
            if row["replay_url"] is None and known:
                # Listings no longer carry links: keep the known one and its backoff.
                row = {**row, "replay_url": known}
            changed_url = known != row["replay_url"]
            self.db.execute("UPDATE matches SET candidate=? WHERE id=?", (json.dumps(row), row["match_id"]))
            if changed_url and previous[1] == "error":
                self.db.execute("UPDATE matches SET status='pending',retry_after=NULL WHERE id=?", (row["match_id"],))
        else:
            self.db.execute("INSERT INTO matches(id,candidate,status,updated) VALUES(?,?,?,?)",
                            (row["match_id"], json.dumps(row), "pending", utc_now()))
        self.db.commit()

    def finish(self, match_id, status, target=None, sha256=None, error=None):
        self.db.execute("UPDATE matches SET status=?,target=?,sha256=?,error=?,updated=?,retry_after=NULL WHERE id=?",
                        (status, str(target) if target else None, sha256, error, utc_now(), match_id))
        self.db.commit()

    def state(self, key, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_state(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, json.dumps(value)))
        self.db.commit()

    def defer(self, match_id, error):
        from datetime import timedelta
        attempts = self.db.execute("SELECT attempts FROM matches WHERE id=?", (match_id,)).fetchone()[0] + 1
        if isinstance(error, HTTPError) and error.code in (403, 404, 410):
            delay = min(86400, 21600 * 2 ** min(attempts - 1, 2))
        elif isinstance(error, (ValueError, KeyError, IndexError)):
            delay = 86400
        else:
            delay = min(21600, 300 * 2 ** min(attempts - 1, 7))
        retry = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
        self.db.execute("UPDATE matches SET status='error',error=?,updated=?,attempts=?,retry_after=? WHERE id=?",
                        (str(error), utc_now(), attempts, retry, match_id))
        self.db.commit()

    def queued(self, limit):
        # Reserve retry slots even while discovery continually adds fresh work.
        errors = list(self.db.execute(
            "SELECT candidate FROM matches WHERE status='error' AND "
            "(retry_after IS NULL OR retry_after<=?) ORDER BY updated,id LIMIT ?", (utc_now(), limit)))
        reserved = min(len(errors), max(1, limit // 5))
        pending = list(self.db.execute("SELECT candidate FROM matches WHERE status='pending' "
                                       "ORDER BY id DESC LIMIT ?", (limit - reserved,)))
        retries = errors[:limit - len(pending)]
        return [json.loads(r[0]) for r in retries + pending]

    def checkpoint(self, next_page):
        self.db.execute("INSERT OR REPLACE INTO state VALUES('next_page',?)", (str(next_page),))
        self.db.commit()

    def resume_page(self):
        value = self.db.execute("SELECT value FROM state WHERE key='next_page'").fetchone()
        return max(1, int(value[0]) - 2) if value else 1

    def valid_existing(self, match_id):
        row = self.db.execute("SELECT status,target,sha256 FROM matches WHERE id=?", (match_id,)).fetchone()
        if row and row[0] == "downloaded" and row[1]:
            path = self.root / row[1]
            if path.is_file() and path.with_suffix(".json").is_file():
                try:
                    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                    return (metadata["match_id"] == match_id and metadata["sha256"] == row[2]
                            and metadata.get("replay_game_version", metadata["advertised_game_version"]) == path.parent.name
                            and hashlib.sha256(path.read_bytes()).hexdigest() == row[2])
                except (ValueError, KeyError, OSError):
                    return False
        return False

    def harvest(self, client, row):
        match_id = row["match_id"]
        if self.valid_existing(match_id):
            return "existing"
        detail = client.request(row["url"]).decode("utf-8")
        version = version_metadata(detail)
        advertised, pack_url = version["crc"], version["data_pack_url"]
        # The match page is authoritative; a listing link is the fallback.
        replay_url = next(iter(replay_links(detail)), None) or row["replay_url"]
        if replay_url is None:
            raise ValueError("Match page has no replay link yet")
        data = client.request(replay_url)
        header = parse_replay(data)
        if advertised and header["exe_crc"] != advertised:
            raise ValueError(f"Advertised CRC {advertised} differs from replay {header['exe_crc']}")
        exclusion = replay_release_exclusion(header["exe_crc"])
        if exclusion:
            self.finish(match_id, "excluded", error=exclusion)
            return "excluded"
        if header["desync"] and not self.include_desynced:
            self.finish(match_id, "rejected", error="Replay header flags a desynchronized match")
            return "rejected"
        if ((self.map_name and map_identity(header["map"]) != self.map_name)
                or (self.map_crc and not crc32_matches(header["map_crc"], self.map_crc))):
            self.finish(match_id, "excluded", error="Replay map name/revision does not match the map filter")
            return "excluded"
        factions = row["factions"]
        if len(factions) != 2 or "China" not in factions:
            raise ValueError("Listing does not identify a vanilla-China 1v1")
        # Random remains unresolved in some headers. Do not invent a template ID.
        fixed = [p["header_faction"] for p in header["players"] if p["template_id"] != -1]
        available = list(factions)
        for faction in fixed:
            if faction not in available:
                raise ValueError(f"Header faction {faction} disagrees with listing {factions}")
            available.remove(faction)
        pack_crc = re.search(r"_([0-9A-Fa-f]{8})\.big$", pack_url or "")
        if pack_crc and pack_crc.group(1).upper() != header["ini_crc"]:
            raise ValueError("Advertised data-pack CRC differs from replay INI CRC")
        target = Path(header["exe_crc"]) / f"match_{match_id}.rep"
        sha256 = hashlib.sha256(data).hexdigest()
        metadata = {"schema_version": 2, "downloaded_utc": utc_now(),
                    "match_id": match_id, "match_url": row["url"],
                    "replay_url": replay_url, "advertised_game_version": advertised,
                    "advertised_release_label": version["label"], "replay_game_version": header["exe_crc"],
                    "version_classification_source": "advertised CRC confirmed by replay header" if advertised
                        else "replay header; site does not advertise an executable CRC",
                    "advertised_data_pack_url": pack_url, "listing_factions": factions,
                    "faction_classification_source": "Matches listing (exact China label)",
                    "listing_text": row["text"], "sha256": sha256, "bytes": len(data),
                    "replay_header": header, "simulation_validation": "not_run",
                    "source_commit": None}
        atomic_write(self.root / target, data)
        atomic_write((self.root / target).with_suffix(".json"),
                     (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        self.finish(match_id, "downloaded", target, sha256)
        return "downloaded"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("replays/vanilla-china"))
    parser.add_argument("--max-pages", type=int, default=10,
                        help="Pages this run; 0 scans until end (default: 10)")
    parser.add_argument("--max-replays", type=int, default=100,
                        help="New downloads this run; 0 is unlimited (default: 100)")
    parser.add_argument("--start-page", type=int, help="Override saved checkpoint; use 1 for fresh arrivals")
    parser.add_argument("--delay", type=float, default=2, help="Minimum seconds between requests")
    parser.add_argument("--map-name", help="Exact map basename, including revision; virtual path is ignored")
    parser.add_argument("--map-crc", help="Exact eight-digit map CRC; combine with --map-name")
    args = parser.parse_args(argv)
    if args.max_pages < 0 or args.max_replays < 0 or args.delay < 1 or (args.start_page or 1) < 1:
        parser.error("Limits must be nonnegative, start page positive, delay at least 1 second")
    if args.map_crc and not re.fullmatch(r"[0-9A-Fa-f]{8}", args.map_crc):
        parser.error("Map CRC must be eight hexadecimal digits")
    try:
        lock = ArchiveLock(args.output)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    try:
        archive = Archive(args.output, map_name=args.map_name, map_crc=args.map_crc)
    except Exception:
        lock.__exit__()
        raise
    client = Client(args.delay)
    counts = {"downloaded": 0, "existing": 0, "rejected": 0, "excluded": 0, "errors": 0, "pages": 0}

    def full():
        return args.max_replays > 0 and counts["downloaded"] >= args.max_replays

    def process(row):
        try:
            status = archive.harvest(client, row)
            counts[status] += 1
            print(f"{status}: match {row['match_id']}", flush=True)
        except (ValueError, OSError, KeyError, IndexError) as error:
            archive.finish(row["match_id"], "error", error=str(error))
            counts["errors"] += 1
            print(f"ERROR match {row['match_id']}: {error}", file=sys.stderr, flush=True)

    try:
        # Interrupted downloads and failures are retried even after page advancement.
        pending = list(archive.db.execute("SELECT candidate FROM matches WHERE status IN ('pending','error')"))
        for (candidate,) in pending:
            if full():
                break
            process(json.loads(candidate))
        if not full():
            feed = MatchesFeed(client)
            page = args.start_page or archive.resume_page()
            prior_ids = None
            while not full() and (not args.max_pages or counts["pages"] < args.max_pages):
                rows = feed.page(page)
                ids = tuple(r["match_id"] for r in rows)
                if not rows:
                    print("End of match history", flush=True)
                    break
                if ids == prior_ids:
                    raise ValueError("Feed repeated a page; stopping rather than skipping history")
                prior_ids = ids
                # The feed is filtered to matches with replays; links come from match pages.
                candidates = [r for r in rows if len(r["factions"]) == 2
                              and "China" in r["factions"] and listing_matches_map(r, args.map_name)]
                for row in candidates:
                    archive.discover(row)
                print(f"Page {page}: {len(rows)} matches, {len(candidates)} vanilla-China candidates", flush=True)
                for row in candidates:
                    if full():
                        break
                    process(row)
                counts["pages"] += 1
                archive.checkpoint(page + 1)
                page += 1
        counts["archive_total"] = archive.db.execute(
            "SELECT COUNT(*) FROM matches WHERE status='downloaded'").fetchone()[0]
        summary = {"finished_utc": utc_now(), "map_name": args.map_name, "map_crc": args.map_crc, **counts}
        atomic_write(archive.root / "last-run.json", (json.dumps(summary, indent=2) + "\n").encode())
        print(json.dumps(summary), flush=True)
        return 1 if counts["errors"] else 0
    except KeyboardInterrupt:
        print("Interrupted; saved candidates/checkpoint will resume on the next run.", file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, IndexError) as error:
        print(f"Harvest stopped: {error}", file=sys.stderr)
        return 1
    finally:
        archive.db.close()
        lock.__exit__()


if __name__ == "__main__":
    sys.exit(main())
