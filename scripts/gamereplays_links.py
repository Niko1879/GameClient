"""Enumerate Zero Hour replay download links on gamereplays.org.

Walks three sections: events (each event's own replay list), expert replays
and the member replay index. Every listing row is recorded in a CSV with its
download link and both sides' army labels; `wanted` marks rows where a player
is one of our China factions (Vanilla China, China Tank, China Nuke) or picked
Random Army (possibly China once resolved from the seed).

Listing pages are cached under .cache/gamereplays-listing/ so a restart does
not refetch them. robots.txt asks for Crawl-delay: 20; we honour it.

    py scripts/gamereplays_links.py            # crawl (resumable)
    py scripts/gamereplays_links.py --report   # rebuild the CSV from the cache
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "gamereplays-listing"
OUT = ROOT / "replays" / "gamereplays-links.csv"
# Wanted rows, appended as they are found (discovery order). The crawler only
# ever appends, so the user can add columns/marks to it; while the file is
# locked (open in Excel) new rows wait in memory and go in on the next attempt.
# Each flush also re-appends any row this run found that is no longer in the
# file (an editor saved over newer appends), so a save never loses links.
DOWNLOAD_LIST = ROOT / "replays" / "gamereplays-download-list.csv"
LINKS_LIST = ROOT / "replays" / "gamereplays-download-links.txt"  # same rows, bare URLs only
DL_COLS = ["replay_id", "download_url", "file_name", "player_count", "reason", "players", "armies",
           "map", "added", "found_in"]
BASE = "https://www.gamereplays.org/cnczerohour/replays.php"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
DELAY = 20.0
CHINA = {"Vanilla China", "China Tank", "China Nuke"}
RANDOM = "Random Army"

EVENTS_INDEX = BASE + "?s=0&game=2&show=events&st={st}"
EVENT = BASE + "?s=0&game=2&show=event&id={id}&sort_by=rounds&search=&st={st}"
EXPERT = BASE + "?s=0&game=2&show=expert_replays&sort_by=latest_upload&st={st}"
MEMBER = BASE + "?s=0&game=2&tab=upcoming&show=index&active_tab=upcoming&sort_by=latest_upload&st={st}"

_last = 0.0
OFFLINE = False


def fetch(url: str) -> str:
    global _last
    path = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    if path.exists():
        return path.read_text(encoding="latin-1")
    if OFFLINE:
        return ""
    for attempt in range(5):
        wait = _last + DELAY - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("latin-1")
            break
        except Exception as e:  # noqa: BLE001 - retry any transport error
            print(f"  retry {attempt + 1} {url}: {e}", flush=True)
            time.sleep(DELAY * (attempt + 1))
    else:
        raise RuntimeError(f"failed: {url}")
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(body, encoding="latin-1")
    tmp.replace(path)
    return body


def max_st(page: str, marker: str) -> int:
    pat = re.escape(marker) + r"[^\"']*?st=(\d+)"
    return max([int(m) for m in re.findall(pat, html.unescape(page))] + [0])


def rows(page: str):
    starts = [m.start() for m in re.finditer(r'class="replay_index_button index_download"', page)]
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else len(page)
        block = page[a:b]
        end = block.find('class="replay_entry_link"')
        if end >= 0:
            block = block[:end]
        dl = re.search(r'class="replay_index_button index_download" href="([^"]*show=download[^"]*id=(\d+))"[^>]*title="Download ([^"]*?), [\d.]+\s*[a-zA-Z]*"', block)
        if not dl:
            continue
        # Tooltips read "player (Army)" in both the 1v1 and the team layouts.
        armies = []
        for tip in re.findall(r"player_tooltip&quot;&gt;&lt;div&gt;(.*?)&lt;/div&gt;", block):
            m = re.match(r"(.*) \(([^()]*)\)$", html.unescape(html.unescape(tip)))
            armies.append((m.group(2), m.group(1)) if m else ("?", html.unescape(tip)))
        title = re.search(r'<div class="replay_title">(.*?)</div>', block, re.S)
        mp = re.search(r'class="map_overlay" title="([^"]*)"', block)
        added = re.search(r"Added ([^-<]+?) -", block)
        ver = re.search(r'<span class="heavytext">([^<]*)</span>', block)
        yield {
            "replay_id": dl.group(2),
            "download_url": html.unescape(dl.group(1)).replace("&&", "&"),
            "file_name": html.unescape(dl.group(3)),
            "armies": [a for a, _ in armies],
            "players": [p.strip() for _, p in armies],
            "title": html.unescape(re.sub(r"<[^>]+>", "", title.group(1))).strip() if title else "",
            "map": html.unescape(mp.group(1)) if mp else "",
            "added": added.group(1).strip() if added else "",
            "version": ver.group(1).strip() if ver else "",
        }


def crawl_list(template: str, marker: str, label: str, **kw):
    first = fetch(template.format(st=0, **kw))
    last = max_st(first, marker)
    step = 30
    print(f"{label}: {last // step + 1} pages", flush=True)
    for st in range(0, last + 1, step):
        page = first if st == 0 else fetch(template.format(st=st, **kw))
        got = list(rows(page))
        anchors = page.count('class="replay_index_button index_download"')
        if len(got) != anchors or (st < last and len(got) != step) or not got:
            print(f"  WARN {label} st={st}: {len(got)} rows, {anchors} download anchors", flush=True)
        yield from got


def event_ids() -> list[tuple[str, str]]:
    first = fetch(EVENTS_INDEX.format(st=0))
    last = max_st(first, "show=events&")
    found: dict[str, str] = {}
    for st in range(0, last + 1, 25):
        page = first if st == 0 else fetch(EVENTS_INDEX.format(st=st))
        for eid, name in re.findall(r'show=event&amp;id=(\d+)"[^>]*>([^<]*)<', page):
            if name.strip() or eid not in found:
                found[eid] = html.unescape(name.strip()) or found.get(eid, "")
    return sorted(found.items(), key=lambda kv: int(kv[0]))


def classify(armies: list[str]) -> str:
    china = any(a in CHINA for a in armies)
    rnd = RANDOM in armies
    return "+".join(x for x, ok in (("china", china), ("random", rnd)) if ok)


class DownloadList:
    def __init__(self, path: Path):
        self.path = path
        self.pending: list[dict] = []
        self.mine: dict[str, dict] = {}
        self.known = self.ids_in_file()

    def ids_in_file(self) -> set[str]:
        ids: set[str] = set()
        if self.path.exists():
            with self.path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
                for rec in csv.reader(f):
                    if rec and rec[0].strip().isdigit():
                        ids.add(rec[0].strip())
        return ids

    def offer(self, row: dict, found_in: str) -> None:
        reason = classify(row["armies"])
        if not reason or row["replay_id"] in self.known:
            return
        self.known.add(row["replay_id"])
        rec = {
            "replay_id": row["replay_id"], "download_url": row["download_url"], "file_name": row["file_name"],
            "player_count": len(row["armies"]), "reason": reason, "players": " | ".join(row["players"]),
            "armies": " | ".join(row["armies"]), "map": row["map"], "added": row["added"], "found_in": found_in,
        }
        self.mine[rec["replay_id"]] = rec
        self.pending.append(rec)

    def flush(self) -> None:
        try:
            present = self.ids_in_file()
        except PermissionError:
            present = None
        if present is not None:
            queued = {r["replay_id"] for r in self.pending}
            lost = [r for i, r in self.mine.items() if i not in present and i not in queued]
            if lost:
                print(f"  re-adding {len(lost)} rows missing from the download list", flush=True)
                self.pending[:0] = lost
        if not self.pending:
            return
        try:
            self.append(self.pending)
        except PermissionError:
            print(f"  download list locked; {len(self.pending)} rows waiting", flush=True)
            return
        self.pending.clear()

    def append(self, recs: list[dict]) -> None:
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
            w = csv.DictWriter(f, fieldnames=DL_COLS)
            if new:
                w.writeheader()
            w.writerows(recs)


class LinkList(DownloadList):
    """Bare download URLs, one per line, so an editor's ctrl+click opens just the link."""

    def ids_in_file(self) -> set[str]:
        if not self.path.exists():
            return set()
        text = self.path.read_text(encoding="utf-8", errors="replace")
        return set(re.findall(r"[?&]id=(\d+)", text))  # marks typed after a link are fine

    def append(self, recs: list[dict]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            f.writelines(r["download_url"] + "\n" for r in recs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="only rebuild the CSV from cached pages (no fetching)")
    args = ap.parse_args()
    global OFFLINE
    OFFLINE = args.report

    found: dict[str, dict] = {}
    sinks = [DownloadList(DOWNLOAD_LIST), LinkList(LINKS_LIST)]

    class _All:
        pending = property(lambda self: [r for k in sinks for r in k.pending])
        def offer(self, row, found_in):
            for k in sinks:
                k.offer(row, found_in)
        def flush(self):
            for k in sinks:
                k.flush()
    dl = _All()

    def add(row, section, event=""):
        r = found.setdefault(row["replay_id"], {**row, "sections": set(), "events": set()})
        r["sections"].add(section)
        if event:
            r["events"].add(event)
        if not OFFLINE:
            dl.offer(row, f"event {event}" if event else section)

    events = event_ids()
    print(f"events: {len(events)}", flush=True)
    for eid, name in events:
        for row in crawl_list(EVENT, f"show=event&id={eid}&", f"event {eid} {name}", id=eid):
            add(row, "events", f"{eid}:{name}")
        dl.flush()
    for i, row in enumerate(crawl_list(EXPERT, "show=expert_replays&", "expert")):
        add(row, "expert")
        if i % 30 == 29:
            dl.flush()
    for i, row in enumerate(crawl_list(MEMBER, "show=index&", "member")):
        add(row, "member")
        if i % 30 == 29:
            dl.flush()
    while dl.pending:
        dl.flush()
        if dl.pending:
            time.sleep(60)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["replay_id", "wanted", "reason", "player_count", "sections", "events", "players", "armies",
            "map", "title", "added", "version", "file_name", "download_url"]
    wanted = 0
    out = OUT
    try:
        OUT.open("a").close()
    except PermissionError:
        out = OUT.with_name(OUT.stem + time.strftime("-%Y%m%d-%H%M%S") + ".csv")
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(found.values(), key=lambda r: int(r["replay_id"])):
            reason = classify(r["armies"])
            wanted += bool(reason)
            w.writerow({
                **{k: r[k] for k in ("replay_id", "map", "title", "added", "version", "file_name", "download_url")},
                "wanted": "yes" if reason else "no",
                "reason": reason,
                "player_count": len(r["armies"]),
                "sections": ";".join(sorted(r["sections"])),
                "events": ";".join(sorted(r["events"])),
                "players": " | ".join(r["players"]),
                "armies": " | ".join(r["armies"]),
            })
    print(f"wrote {out}: {len(found)} replays, {wanted} wanted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
