#!/usr/bin/env python3
"""
space_objects.py — unified space object database builder (v1.1)

Fixes: DISCOS relative pagination links (links.next) via urljoin.

Sources (cached):
- GCAT satcat.tsv
- Space-Track SATCAT (daily-ish)
- Space-Track GP (hourly-ish; includes analyst objects)
- ESA DISCOSweb Objects API (optional; incremental cache: only fetch missing satnos)

Outputs:
- out/space_objects.json
- out/indexes/norad_compact.json.gz
- out/indexes/cospar_to_norad.json.gz
- out/indexes/name_to_norad.json.gz

Env (.env auto-loaded):
  SPACETRACK_USER=...
  SPACETRACK_PASS=...
  DISCOS_TOKEN=...   # optional
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from spacetrack import SpaceTrackClient
import spacetrack.operators as op

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
import logging


# =============================
# Rich logging setup
# =============================

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
log = logging.getLogger("space_objects")


# =============================
# dotenv (no deps)
# =============================

_DOTENV_EXCLUDE = {".env.example", ".env.sample", ".env.template"}


def _parse_dotenv_line(line: str) -> Optional[Tuple[str, str]]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("export "):
        s = s[len("export ") :].lstrip()
    if "=" not in s:
        return None
    k, v = s.split("=", 1)
    k = k.strip()
    if not k:
        return None
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        v = v[1:-1]
    v = v.replace(r"\n", "\n").replace(r"\r", "\r").replace(r"\t", "\t")
    return k, v


def load_dotenv_files(directory: str = ".") -> None:
    d = Path(directory)
    if not d.exists():
        return

    base = d / ".env"
    others = sorted(
        p
        for p in d.glob(".env*")
        if p.is_file() and p.name not in _DOTENV_EXCLUDE and p.name != ".env"
    )

    files: List[Path] = []
    if base.exists():
        files.append(base)
    files.extend(others)

    if not files:
        return

    for p in files:
        override = p.name != ".env"
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            parsed = _parse_dotenv_line(line)
            if not parsed:
                continue
            k, v = parsed
            if override or (k not in os.environ):
                os.environ[k] = v

    log.info(f"Loaded env from: {', '.join(p.name for p in files)}")


# =============================
# Paths / config
# =============================

DATA = Path("data")
CACHE = DATA / "cache"
STATE_PATH = DATA / "state.json"

OUT = Path("out") / "space_objects.json"
INDEX_DIR = OUT.parent / "indexes"

USER_AGENT = "space-objects/1.1"

GCAT_URL = "https://planet4589.org/space/gcat/tsv/cat/satcat.tsv"
GCAT_CACHE = CACHE / "gcat_satcat.tsv"

ST_SATCAT_CACHE = CACHE / "spacetrack_satcat.json.gz"
ST_GP_CACHE = CACHE / "spacetrack_gp.json.gz"

DISCOS_BASE = os.getenv("DISCOS_BASE_URL", "https://discosweb.esoc.esa.int/api").rstrip(
    "/"
)
DISCOS_VERSION = os.getenv("DISCOS_API_VERSION", "2")
DISCOS_CACHE = CACHE / "discos_objects_by_satno.json.gz"

GP_MAX_AGE = timedelta(hours=1, minutes=5)
SATCAT_MAX_AGE = timedelta(hours=25)
GCAT_MAX_AGE = timedelta(hours=48)
DISCOS_MAX_AGE = timedelta(days=30)

GP_EPOCH_WINDOW_DAYS = 10

WELL_TRACKED_ANALYST = (80000, 89999)
SPACE_FENCE_ANALYST = (270000, 339999)

DISCOS_BATCH_SIZE = 200
DISCOS_INCREMENTAL_ONLY = True


# =============================
# Utilities
# =============================


def ensure(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(str(x).strip())
    except Exception:
        return None


def stale(state: Dict[str, Any], key: str, age: timedelta) -> bool:
    t = state.get(key)
    if not t:
        return True
    try:
        return (now() - parse_iso(t)) > age
    except Exception:
        return True


def gzip_write_json(path: Path, obj: Any) -> None:
    ensure(path.parent)
    with gzip.open(path, "wb") as f:
        f.write(json.dumps(obj).encode("utf-8"))


def gzip_read_json(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def normalize_rows(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        s = payload.strip()
        if not s or "NO RESULTS" in s.upper():
            return []
        payload = json.loads(s)
    if isinstance(payload, dict):
        return [payload]
    return [r for r in list(payload) if isinstance(r, dict)]


def analyst_kind(n: int) -> Optional[str]:
    if WELL_TRACKED_ANALYST[0] <= n <= WELL_TRACKED_ANALYST[1]:
        return "well_tracked_analyst"
    if SPACE_FENCE_ANALYST[0] <= n <= SPACE_FENCE_ANALYST[1]:
        return "space_fence_analyst"
    return None


def canon_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


# =============================
# State
# =============================


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: Dict[str, Any]) -> None:
    ensure(STATE_PATH.parent)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


# =============================
# GCAT
# =============================


def fetch_gcat(state: Dict[str, Any]) -> None:
    gstate = state.setdefault("gcat", {})
    need = stale(gstate, "last_fetch", GCAT_MAX_AGE) or not GCAT_CACHE.exists()

    if not need:
        log.info("GCAT: cache hit")
        return

    log.info("GCAT: downloading satcat.tsv")
    r = requests.get(GCAT_URL, headers={"User-Agent": USER_AGENT}, timeout=180)
    r.raise_for_status()
    ensure(CACHE)
    GCAT_CACHE.write_bytes(r.content)

    gstate["last_fetch"] = iso(now())
    state["gcat"] = gstate
    log.info(f"GCAT: updated ({len(r.content) / 1024 / 1024:.2f} MiB)")


def load_gcat_by_norad() -> Dict[int, Dict[str, Any]]:
    if not GCAT_CACHE.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    with GCAT_CACHE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            n = safe_int(r.get("Satcat"))
            if n is not None:
                out[n] = r
    log.info(f"GCAT: loaded ({len(out):,} rows)")
    return out


# =============================
# Space-Track (cached)
# =============================


def update_spacetrack(
    state: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    user = os.getenv("SPACETRACK_USER")
    pw = os.getenv("SPACETRACK_PASS")
    if not user or not pw:
        log.warning(
            "Space-Track: missing SPACETRACK_USER/PASS; skipping (using cache if present)."
        )
        satcat: List[Dict[str, Any]] = []
        gp: List[Dict[str, Any]] = []
        try:
            if ST_SATCAT_CACHE.exists():
                satcat = gzip_read_json(ST_SATCAT_CACHE)
        except Exception as e:
            log.warning(f"Space-Track: failed to read SATCAT cache: {e}")
        try:
            if ST_GP_CACHE.exists():
                gp = gzip_read_json(ST_GP_CACHE)
        except Exception as e:
            log.warning(f"Space-Track: failed to read GP cache: {e}")
        return satcat, gp

    st_state = state.setdefault("spacetrack", {})
    need_satcat = (
        stale(st_state, "last_satcat_fetch", SATCAT_MAX_AGE)
        or not ST_SATCAT_CACHE.exists()
    )
    need_gp = stale(st_state, "last_gp_fetch", GP_MAX_AGE) or not ST_GP_CACHE.exists()

    if not need_satcat and not need_gp:
        log.info("Space-Track: SATCAT cache hit")
        log.info("Space-Track: GP cache hit")
        return gzip_read_json(ST_SATCAT_CACHE), gzip_read_json(ST_GP_CACHE)

    st = SpaceTrackClient(identity=user, password=pw)

    satcat_rows: List[Dict[str, Any]] = []
    gp_rows: List[Dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        if need_satcat:
            task = progress.add_task("Space-Track: downloading SATCAT", total=None)
            satcat_rows = normalize_rows(st.satcat(format="json"))
            gzip_write_json(ST_SATCAT_CACHE, satcat_rows)
            st_state["last_satcat_fetch"] = iso(now())
            progress.remove_task(task)
            log.info(f"Space-Track: SATCAT updated ({len(satcat_rows):,} rows)")
        else:
            satcat_rows = gzip_read_json(ST_SATCAT_CACHE)
            log.info("Space-Track: SATCAT cache hit")

        if need_gp:
            time.sleep(random.randint(1, 30))
            task = progress.add_task(
                "Space-Track: downloading GP (latest elsets)", total=None
            )
            cutoff = now() - timedelta(days=GP_EPOCH_WINDOW_DAYS)
            gp_rows = normalize_rows(
                st.gp(decay_date=None, epoch=op.greater_than(cutoff), format="json")
            )
            gzip_write_json(ST_GP_CACHE, gp_rows)
            st_state["last_gp_fetch"] = iso(now())
            progress.remove_task(task)
            log.info(f"Space-Track: GP updated ({len(gp_rows):,} objects)")
        else:
            gp_rows = gzip_read_json(ST_GP_CACHE)
            log.info("Space-Track: GP cache hit")

    state["spacetrack"] = st_state
    return satcat_rows, gp_rows


# =============================
# DISCOS (incremental-only caching)
# =============================


class DiscosClient:
    def __init__(self, token: str):
        self.base = DISCOS_BASE  # e.g. https://discosweb.esoc.esa.int/api
        self.s = requests.Session()
        self.s.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.api+json",
                "DiscosWeb-Api-Version": DISCOS_VERSION,
            }
        )

    def fetch_objects(self, satnos: List[int]) -> Dict[int, Dict[str, Any]]:
        if not satnos:
            return {}

        # First page uses full base URL
        url = f"{self.base}/objects"
        satno_list = ",".join(str(x) for x in satnos)
        params = {
            "filter": f"in(satno,({satno_list}))",
            "page[size]": 100,
            "page[number]": 1,
        }

        out: Dict[int, Dict[str, Any]] = {}
        page = 0

        while True:
            page += 1
            r = self.s.get(url, params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()

            data = payload.get("data") or []
            for item in data:
                attrs = item.get("attributes") or {}
                n = safe_int(attrs.get("satno"))
                if n is not None:
                    out[n] = attrs

            links = payload.get("links") or {}
            next_link = links.get("next")

            if not next_link:
                break

            # DISCOS often returns relative links like "/api/objects?...page[number]=2"
            # Convert to absolute URL against the base.
            url = urljoin(self.base + "/", next_link)
            params = {}  # next link already includes query string

        return out


def update_discos_incremental(
    state: Dict[str, Any], norads_needed: List[int]
) -> Dict[int, Dict[str, Any]]:
    token = os.getenv("DISCOS_TOKEN")
    if not token:
        log.warning("DISCOS: DISCOS_TOKEN not set — skipping.")
        return {}

    ensure(CACHE)
    dstate = state.setdefault("discos", {})

    discos_map: Dict[int, Dict[str, Any]] = {}
    if DISCOS_CACHE.exists():
        raw = gzip_read_json(DISCOS_CACHE) or {}
        discos_map = {int(k): v for k, v in raw.items()}
        log.info(f"DISCOS: cache loaded ({len(discos_map):,} objects)")
    else:
        log.info("DISCOS: no cache yet")

    missing = [n for n in norads_needed if n not in discos_map]

    if not missing:
        log.info("DISCOS: nothing missing — no API calls needed")
        return discos_map

    if (not DISCOS_INCREMENTAL_ONLY) and stale(dstate, "last_fetch", DISCOS_MAX_AGE):
        log.info("DISCOS: cache stale and refresh enabled (not incremental-only)")

    client = DiscosClient(token=token)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]DISCOS fetching missing {task.completed}/{task.total}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("DISCOS", total=len(missing))

        i = 0
        while i < len(missing):
            batch = missing[i : i + DISCOS_BATCH_SIZE]
            got = client.fetch_objects(batch)
            discos_map.update(got)
            i += DISCOS_BATCH_SIZE
            progress.update(task, advance=len(batch))

    gzip_write_json(DISCOS_CACHE, {str(k): v for k, v in discos_map.items()})
    dstate["last_fetch"] = iso(now())
    dstate["cached_count"] = len(discos_map)
    state["discos"] = dstate

    log.info(f"DISCOS: updated cache ({len(discos_map):,} objects)")
    return discos_map


# =============================
# Merge + indexes (fast)
# =============================


def index_by_norad(
    rows: List[Dict[str, Any]], key: str = "NORAD_CAT_ID"
) -> Dict[int, Dict[str, Any]]:
    d: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        n = safe_int(r.get(key))
        if n is not None:
            d[n] = r
    return d


def build_unified(
    satcat_rows: List[Dict[str, Any]],
    gp_rows: List[Dict[str, Any]],
    gcat_by_norad: Dict[int, Dict[str, Any]],
    discos_by_norad: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    log.info("Merge: indexing SATCAT/GP for O(1) lookups")
    satcat_by = index_by_norad(satcat_rows, "NORAD_CAT_ID")
    gp_by = index_by_norad(gp_rows, "NORAD_CAT_ID")

    norads = (
        set(gcat_by_norad.keys())
        | set(satcat_by.keys())
        | set(gp_by.keys())
        | set(discos_by_norad.keys())
    )
    norads_sorted = sorted(norads)

    objects: Dict[str, Any] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]Merging {task.completed}/{task.total}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("merge", total=len(norads_sorted))
        for n in norads_sorted:
            sc = satcat_by.get(n, {})
            gr = gp_by.get(n, {})
            gc = gcat_by_norad.get(n, {})
            dc = discos_by_norad.get(n, {})

            name = (
                sc.get("SATNAME")
                or gr.get("OBJECT_NAME")
                or dc.get("name")
                or gc.get("Name")
            )
            cospar = (
                sc.get("INTLDES")
                or gr.get("INTLDES")
                or dc.get("cosparId")
                or gc.get("Launch_Tag")
            )
            obj_type = (
                sc.get("OBJECT_TYPE")
                or gr.get("OBJECT_TYPE")
                or dc.get("objectClass")
                or gc.get("Type")
            )

            tle = None
            if gr.get("TLE_LINE1") and gr.get("TLE_LINE2"):
                tle = {
                    "epoch": gr.get("EPOCH"),
                    "line1": gr.get("TLE_LINE1"),
                    "line2": gr.get("TLE_LINE2"),
                }

            objects[str(n)] = {
                "norad_cat_id": n,
                "cospar_id": cospar,
                "name": name,
                "object_type": obj_type,
                "country": sc.get("COUNTRY") or gr.get("COUNTRY_CODE"),
                "launch_date": sc.get("LAUNCH") or gr.get("LAUNCH_DATE"),
                "decay_date": sc.get("DECAY") or gr.get("DECAY_DATE"),
                "analyst_kind": analyst_kind(n),
                "in_satcat": bool(sc),
                "tle": tle,
                "sources": {
                    "spacetrack_satcat": bool(sc),
                    "spacetrack_gp": bool(gr),
                    "gcat": bool(gc),
                    "discos": bool(dc),
                },
            }

            progress.update(task, advance=1)

    return {"generated_at": iso(now()), "objects": objects}


def write_indexes(unified: Dict[str, Any]) -> None:
    ensure(INDEX_DIR)
    objs: Dict[str, Any] = unified.get("objects", {}) or {}

    norad_compact: Dict[str, Any] = {}
    cospar_to_norad: Dict[str, List[int]] = {}
    name_to_norad: Dict[str, List[int]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Writing indexes {task.completed}/{task.total}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("indexes", total=len(objs))

        for norad_str, rec in objs.items():
            n = safe_int(norad_str)
            if n is None:
                progress.update(task, advance=1)
                continue

            name = rec.get("name")
            cospar = rec.get("cospar_id")
            tle = rec.get("tle")
            has_tle = bool(tle and tle.get("line1") and tle.get("line2"))
            epoch = tle.get("epoch") if tle else None

            norad_compact[norad_str] = {
                "name": name,
                "cospar_id": cospar,
                "object_type": rec.get("object_type"),
                "country": rec.get("country"),
                "analyst_kind": rec.get("analyst_kind"),
                "has_tle": has_tle,
                "epoch": epoch,
                "sources": rec.get("sources"),
            }

            if isinstance(cospar, str) and cospar.strip():
                cospar_to_norad.setdefault(cospar.strip(), []).append(n)

            cn = canon_name(name)
            if cn:
                name_to_norad.setdefault(cn, []).append(n)

            progress.update(task, advance=1)

    gzip_write_json(INDEX_DIR / "norad_compact.json.gz", norad_compact)
    gzip_write_json(INDEX_DIR / "cospar_to_norad.json.gz", cospar_to_norad)
    gzip_write_json(INDEX_DIR / "name_to_norad.json.gz", name_to_norad)


# =============================
# Main
# =============================


def main() -> None:
    load_dotenv_files(".")
    ensure(DATA)
    ensure(CACHE)
    ensure(OUT.parent)

    state = load_state()

    # 1) GCAT (cached)
    fetch_gcat(state)
    gcat_by = load_gcat_by_norad()

    # 2) Space-Track (cached)
    satcat_rows, gp_rows = update_spacetrack(state)

    # 3) Union NORAD list for DISCOS
    norads_union: set[int] = set(gcat_by.keys())
    for r in satcat_rows:
        n = safe_int(r.get("NORAD_CAT_ID"))
        if n is not None:
            norads_union.add(n)
    for r in gp_rows:
        n = safe_int(r.get("NORAD_CAT_ID"))
        if n is not None:
            norads_union.add(n)

    norads_list = sorted(norads_union)

    # 4) DISCOS (incremental-only: fetch missing)
    discos_by = update_discos_incremental(state, norads_list)

    # 5) Merge (fast + progress)
    unified = build_unified(satcat_rows, gp_rows, gcat_by, discos_by)

    # 6) Write output
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold white]Writing out/space_objects.json"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("write", total=None)
        OUT.write_text(json.dumps(unified, indent=2), encoding="utf-8")
        progress.remove_task(task)

    # 7) Indexes
    write_indexes(unified)

    # 8) Save state
    save_state(state)

    console.print(
        Panel.fit(
            f"[bold green]Done[/bold green]\n"
            f"Objects: {len(unified['objects']):,}\n"
            f"Output: {OUT}\n"
            f"Indexes: {INDEX_DIR}",
            title="Space Objects",
        )
    )


if __name__ == "__main__":
    main()
