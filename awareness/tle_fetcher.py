import os
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Local disk cache so we don't hammer Space-Track on every run.
# The optimizer may call fetch_tles() repeatedly; this keeps it fast.
CACHE_PATH  = Path("data/tle_cache.json")

# Space-Track rate-limits accounts — one full catalog pull per day is safe.
TTL_SECONDS = 86400  # 24 hours

# Space-Track requires a session cookie obtained from the login endpoint
# before any data query will succeed.
LOGIN_URL   = "https://www.space-track.org/ajaxauth/login"

# Query: all GP (general perturbations) elements whose epoch is within the
# last 30 days, sorted by NORAD ID.
# EPOCH/>now-30 keeps us under the 100 k object ceiling and discards stale debris.
#
# format/3le, not format/tle. `tle` returns bare two-line pairs with no names,
# which left every ConjunctionResult.name showing a raw TLE line. `3le` prefixes
# each pair with a "0 SATELLITE NAME" line. _parse handles both regardless.
CATALOG_URL = (
    "https://www.space-track.org/basicspacedata/query"
    "/class/gp/EPOCH/%3Enow-30/orderby/NORAD_CAT_ID/format/3le"
)


def fetch_tles(cache_ttl: float = TTL_SECONDS) -> list[tuple[str, str, str]]:
    """Return [(name, line1, line2), ...] from cache or live Space-Track.

    The cache avoids redundant network calls.  Pass cache_ttl=0 to force
    a fresh download (e.g. during integration tests).
    """
    load_dotenv()  # pull SPACE_TRACK_USER / SPACE_TRACK_PASS from .env

    # Serve from cache when the file exists and hasn't expired.
    if CACHE_PATH.exists():
        cached = json.loads(CACHE_PATH.read_text())
        if time.time() - cached["timestamp"] < cache_ttl:
            # json round-trips tuples as lists; convert back so callers get tuples.
            return [tuple(t) for t in cached["tles"]]

    tles = _download()

    # Write cache atomically enough for our purposes (single process).
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({"timestamp": time.time(), "tles": tles}))
    return tles


def _download() -> list[tuple[str, str, str]]:
    """Authenticate with Space-Track and pull the active LEO catalog.

    Raises RuntimeError rather than returning an empty catalog. An empty catalog
    is indistinguishable from "space is empty" to every caller downstream, and
    the conjunction checker would happily declare any orbit safe.
    """
    user = os.environ["SPACE_TRACK_USER"]
    pwd  = os.environ["SPACE_TRACK_PASS"]

    # A requests.Session keeps the auth cookie across the subsequent GET.
    s = requests.Session()
    login = s.post(LOGIN_URL, data={"identity": user, "password": pwd}, timeout=30)

    # The login response must be checked. Space-Track rejects a bad login with a
    # 4xx and a JSON body, but leaves the session usable — the catalog query then
    # answers 204 No Content, which raise_for_status() treats as SUCCESS because
    # it is a 2xx. The old code skipped this check and cached the resulting empty
    # catalog for 24 hours, so an auth failure looked like an empty sky.
    if login.status_code != 200 or '"Login"' in login.text:
        raise RuntimeError(
            f"Space-Track login failed (HTTP {login.status_code}): "
            f"{login.text[:200]}. Check SPACE_TRACK_USER / SPACE_TRACK_PASS in .env."
        )

    resp = s.get(CATALOG_URL, timeout=60)
    resp.raise_for_status()  # surface HTTP errors (401 bad creds, 429 rate-limit, etc.)

    tles = _parse(resp.text)
    if not tles:
        # Covers 204 No Content and any 2xx whose body holds no parseable TLEs.
        # Never let this reach the caller as a valid empty catalog.
        raise RuntimeError(
            f"Space-Track returned no usable TLEs (HTTP {resp.status_code}, "
            f"{len(resp.text)} bytes). This usually means the session was not "
            "authenticated, or the query matched nothing."
        )
    return tles


def _parse(text: str) -> list[tuple[str, str, str]]:
    """Convert raw Space-Track text into a list of (name, line1, line2) tuples.

    Handles both formats Space-Track can return:

        3LE (format/3le)    "0 ISS (ZARYA)" / "1 25544U ..." / "2 25544 ..."
        TLE (format/tle)    "1 25544U ..." / "2 25544 ..."      — no name line

    Objects with no name line are labelled "NORAD <id>" so the identifier in a
    conjunction report is always meaningful.

    This walks the lines rather than stepping by a fixed stride. The previous
    version assumed 3-line groups and advanced `range(0, len, 3)`, which against
    the 2-line data the query actually requested captured **one object in three**
    and used the preceding object's line 2 as the name — silently, because every
    tuple it did emit was internally well-formed.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines) - 1:
        line = lines[i]

        # Bare pair: line 1 immediately followed by line 2.
        if line.startswith("1 ") and lines[i + 1].startswith("2 "):
            out.append((f"NORAD {line[2:7].strip()}", line, lines[i + 1]))
            i += 2
            continue

        # Name line followed by the pair. 3LE prefixes the name with "0 ".
        if (
            i + 2 < len(lines)
            and lines[i + 1].startswith("1 ")
            and lines[i + 2].startswith("2 ")
        ):
            name = line[2:].strip() if line.startswith("0 ") else line
            out.append((name or f"NORAD {lines[i + 1][2:7].strip()}",
                        lines[i + 1], lines[i + 2]))
            i += 3
            continue

        i += 1   # unrecognised line — skip it rather than shifting the grouping

    return out
