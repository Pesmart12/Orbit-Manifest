"""
Phase 2 — Situational Awareness tests.

Test 1: TLE cache write and TTL (network mocked, no credentials needed)
Test 2: Min separation geometry (pure numpy, no SGP4)
Test 3: Full pipeline smoke test with a hardcoded ISS TLE
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from awareness.tle_fetcher import _parse, fetch_tles
from awareness.conjunction import (
    CatalogCache,
    nearest_approach,
    screened_count,
    ConjunctionResult,
    _FILTER_BAND_M,
    _compute_catalog_positions,
    _filter_by_altitude,
    check_conjunctions,
)

MU_EARTH = 3.986004418e14
R_EARTH  = 6.3781e6

# Hardcoded ISS TLE (2024-era, fixed for reproducibility)
ISS_TLE = (
    "ISS (ZARYA)",
    "1 25544U 98067A   24001.50000000  .00016717  00000+0  30309-3 0  9994",
    "2 25544  51.6400 337.6182 0001500  80.0000 280.0000 15.49560000441234",
)

_FAKE_TLE_TEXT = "\n".join([
    "OBJECT A",
    "1 99001U 23001A   24001.00000000  .00000000  00000+0  00000+0 0  9999",
    "2 99001  97.0000 000.0000 0001000  90.0000 270.0000 15.00000000 00001",
    "OBJECT B",
    "1 99002U 23001B   24001.00000000  .00000000  00000+0  00000+0 0  9999",
    "2 99002  51.6000 000.0000 0001000  90.0000 270.0000 15.50000000 00002",
])


def _make_tle(alt_km: float, norad_id: int, ecc: float = 0.0001) -> tuple:
    """Construct a minimal TLE for a given circular-ish altitude."""
    a         = R_EARTH + alt_km * 1e3
    n_rev_day = np.sqrt(MU_EARTH / a ** 3) * 86400.0 / (2.0 * np.pi)
    ecc_str   = f"{int(ecc * 1e7):07d}"
    name  = f"TEST-{alt_km:.0f}KM"
    line1 = (f"1 {norad_id:05d}U 24001A   24001.00000000  .00000000"
             f"  00000+0  00000+0 0  9990")
    line2 = (f"2 {norad_id:05d}  97.0000   0.0000 {ecc_str}  90.0000"
             f" 270.0000 {n_rev_day:11.8f}    10")
    return (name, line1, line2)


def _make_mock_session(text: str) -> MagicMock:
    mock_resp = MagicMock(status_code=200, text=text)
    mock_resp.raise_for_status = MagicMock()

    # A successful Space-Track login is HTTP 200 with no "Login" key in the body.
    # The bare MagicMock this used to return had a MagicMock status_code, which
    # modelled no real response and passed only because nothing checked it.
    mock_login = MagicMock(status_code=200, text="")

    mock_session = MagicMock()
    mock_session.post.return_value = mock_login
    mock_session.get.return_value = mock_resp
    return mock_session


# ---------------------------------------------------------------------------
# Test 1 — Cache write and TTL
# ---------------------------------------------------------------------------
def test_tle_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACE_TRACK_USER", "test@example.com")
    monkeypatch.setenv("SPACE_TRACK_PASS", "testpass")

    cache_file = tmp_path / "tle_cache.json"
    monkeypatch.setattr("awareness.tle_fetcher.CACHE_PATH", cache_file)

    mock_session = _make_mock_session(_FAKE_TLE_TEXT)

    with patch("awareness.tle_fetcher.requests.Session", return_value=mock_session):
        tles1 = fetch_tles()
        assert len(tles1) == 2
        assert mock_session.get.call_count == 1

        # Second call within TTL — should NOT hit network
        tles2 = fetch_tles()
        assert mock_session.get.call_count == 1
        assert tles1 == tles2

    # Call with cache_ttl=0 — forces re-download
    with patch("awareness.tle_fetcher.requests.Session", return_value=mock_session):
        tles3 = fetch_tles(cache_ttl=0)
        assert mock_session.get.call_count == 2
        assert len(tles3) == 2


def test_tle_parse_structure():
    tles = _parse(_FAKE_TLE_TEXT)
    assert len(tles) == 2
    for name, l1, l2 in tles:
        assert l1.startswith("1 ")
        assert l2.startswith("2 ")


# ---------------------------------------------------------------------------
# Test 2 — Min separation geometry (pure numpy, no SGP4)
# ---------------------------------------------------------------------------
def test_separation_geometry():
    """Verify that the distance metric in check_conjunctions is correct
    by testing the numpy norm formula directly."""
    mission_pos = np.array([7e6, 0.0, 0.0])  # meters

    # Coincident — separation = 0
    cat_pos = np.array([7e6, 0.0, 0.0])
    sep = float(np.linalg.norm(cat_pos - mission_pos))
    assert sep == pytest.approx(0.0)

    # 3 km offset — below 5 km threshold (conjunction)
    cat_pos = np.array([7e6 + 3000.0, 0.0, 0.0])
    sep = float(np.linalg.norm(cat_pos - mission_pos))
    assert sep < 5000.0

    # 10 km offset — above threshold (no conjunction)
    cat_pos = np.array([7e6 + 10000.0, 0.0, 0.0])
    sep = float(np.linalg.norm(cat_pos - mission_pos))
    assert sep > 5000.0


# ---------------------------------------------------------------------------
# Test 3 — Full pipeline smoke test (hardcoded ISS TLE, no network)
# ---------------------------------------------------------------------------
def test_check_conjunctions_pipeline():
    # 500 km polar orbit state vector (ECI, t=0 at ascending node)
    a       = R_EARTH + 500e3
    v_circ  = np.sqrt(MU_EARTH / a)
    # Polar orbit: velocity in z-direction at equatorial crossing
    mission_state = np.array([a, 0.0, 0.0, 0.0, 0.0, v_circ])

    epoch    = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    catalog  = [ISS_TLE]
    duration = 3600.0  # 1 hour

    results = check_conjunctions(
        mission_state=mission_state,
        epoch=epoch,
        duration_s=duration,
        catalog=catalog,
        dt=30.0,
        threshold_m=5000.0,
    )

    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, ConjunctionResult)
        assert r.min_separation_m >= 0.0
        assert r.time_of_closest_approach_s >= 0.0
        assert r.time_of_closest_approach_s <= duration


def test_check_conjunctions_empty_catalog():
    a = R_EARTH + 500e3
    v_circ = np.sqrt(MU_EARTH / a)
    mission_state = np.array([a, 0.0, 0.0, 0.0, v_circ, 0.0])
    epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    results = check_conjunctions(mission_state, epoch, 3600.0, catalog=[])
    assert results == []


# ---------------------------------------------------------------------------
# Test 5 — CatalogCache: SGP4 computed once for same epoch/duration/dt/altitude
# ---------------------------------------------------------------------------
def test_catalog_cache_hit():
    a      = R_EARTH + 400e3      # ISS TLE is ~420 km; must be within ±25 km band
    v_circ = np.sqrt(MU_EARTH / a)
    epoch  = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Two different mission states at the same altitude — same cache key.
    state1 = np.array([a, 0.0, 0.0, 0.0, 0.0, v_circ])
    state2 = np.array([a, 0.0, 0.0, 0.0, v_circ * 0.7071, v_circ * 0.7071])

    cache = CatalogCache()

    with patch(
        "awareness.conjunction._compute_catalog_positions",
        wraps=_compute_catalog_positions,
    ) as mock_fn:
        check_conjunctions(state1, epoch, 3600.0, [ISS_TLE], dt=30.0, catalog_cache=cache)
        check_conjunctions(state2, epoch, 3600.0, [ISS_TLE], dt=30.0, catalog_cache=cache)
        # Same epoch / duration / dt / altitude → catalog positions computed once.
        assert mock_fn.call_count == 1


# ---------------------------------------------------------------------------
# Test 6 — Altitude filter: TLE string parsing selects correct orbit shells
# ---------------------------------------------------------------------------
def test_altitude_filter():
    # Circular orbit at the target altitude — should pass.
    tle_500 = _make_tle(500.0, 11001)

    # Eccentric orbit: periapsis ~400 km, apoapsis ~650 km — crosses the band.
    # sma = R_EARTH + 525 km, ecc = 125 km / sma  →  peri ≈ 400 km, apo ≈ 650 km
    a_ecc   = R_EARTH + 525e3
    ecc_val = 125e3 / a_ecc
    tle_ecc = _make_tle(525.0, 11002, ecc=ecc_val)

    # Circular orbit at 1000 km — too high, should be excluded.
    tle_1000 = _make_tle(1000.0, 11003)

    catalog     = [tle_500, tle_ecc, tle_1000]
    target_sma  = R_EARTH + 500e3
    filtered    = _filter_by_altitude(catalog, target_sma, band_m=100_000.0)
    names       = [t[0] for t in filtered]

    assert "TEST-500KM"  in names, "Circular orbit at target altitude should be included"
    assert "TEST-525KM"  in names, "Eccentric orbit crossing the band should be included"
    assert "TEST-1000KM" not in names, "Orbit at 1000 km should be excluded"


# ---------------------------------------------------------------------------
# Test 7 — CatalogCache bucketing: bounded entry count, no loss of coverage
# ---------------------------------------------------------------------------
def test_catalog_cache_buckets_nearby_radii():
    """Candidates spread across an sma band must share cache entries.

    Regression: the key rounded the radius to the nearest 1 km, so an SSO run
    over a +/-20 km band produced ~41 entries, each a full (T, N, 3) array —
    hundreds of MB apiece on a 7-day mission.
    """
    epoch   = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    catalog = [_make_tle(550.0, 12001)]
    cache   = CatalogCache(bucket_m=50_000.0)

    base = R_EARTH + 550e3
    for offset_km in range(-20, 21):          # 41 distinct radii, 1 km apart
        cache.get_or_compute(catalog, epoch, 3600.0, 30.0, base + offset_km * 1e3)

    assert len(cache._store) <= 2, (
        f"41 radii across a 40 km band produced {len(cache._store)} cache entries; "
        "a 50 km bucket grid should collapse them to at most 2"
    )


def test_catalog_cache_bucket_covers_band_edges():
    """A shared bucket must cover every radius that maps into it.

    Bucketing is only safe if the shared object set is a superset of what each
    candidate would have screened alone. A candidate at the edge of a bucket
    screens up to half a band beyond the bucket edge, so an object out there
    must still be present — otherwise sharing an entry hides a conjunction.
    """
    epoch    = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    bucket_m = 50_000.0
    centre   = round((R_EARTH + 550e3) / bucket_m) * bucket_m

    # Worst case: a candidate sitting exactly on the bucket's upper edge.
    edge_target = centre + bucket_m / 2.0
    # An object just inside that candidate's own screening band.
    obj_sma     = edge_target + _FILTER_BAND_M / 2.0 - 1_000.0
    far_object  = _make_tle((obj_sma - R_EARTH) / 1e3, 12002)

    # Sanity: the object really is one the edge candidate would screen alone.
    assert _filter_by_altitude([far_object], edge_target), \
        "test setup: object should fall inside the edge candidate's own band"

    cache = CatalogCache(bucket_m=bucket_m)
    _, names, _ = cache.get_or_compute([far_object], epoch, 3600.0, 30.0, edge_target)

    assert names, (
        "object inside the edge candidate's own screening band was dropped from "
        "the shared bucket entry — bucketing must widen the band, not narrow it"
    )


def test_catalog_cache_budget_is_cumulative():
    """The memory guard applies to the cache as a whole, not one entry at a time.

    Regression: the check was per-array, so a run accumulating many moderate
    entries — the actual failure mode — never tripped it.
    """
    epoch   = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    catalog = [_make_tle(500.0 + 100.0 * i, 12100 + i) for i in range(6)]

    # Budget deliberately larger than any single entry, smaller than the total.
    cache = CatalogCache(bucket_m=50_000.0, warn_bytes=1_000)

    with pytest.warns(ResourceWarning, match="CatalogCache is holding"):
        for i in range(6):
            cache.get_or_compute(
                catalog, epoch, 300.0, 30.0, R_EARTH + 500e3 + 100e3 * i
            )

    assert cache.nbytes > 1_000
    assert len(cache._store) > 1, "budget should be exceeded by accumulation, not one entry"


# ---------------------------------------------------------------------------
# Test 8 — nearest_approach / screened_count: detail that check_conjunctions drops
# ---------------------------------------------------------------------------
def test_nearest_approach_reports_object_above_threshold():
    """The closest object is reported even when nothing violates the threshold.

    check_conjunctions answers "is anything too close?" and returns [] when the
    answer is no, which leaves a SAFE verdict with no margin behind it. This is
    the query that gives the report a number.
    """
    a      = R_EARTH + 400e3
    v_circ = np.sqrt(MU_EARTH / a)
    epoch  = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    state  = np.array([a, 0.0, 0.0, 0.0, 0.0, v_circ])

    cache = CatalogCache()
    hits = check_conjunctions(state, epoch, 3600.0, [ISS_TLE], dt=30.0,
                              threshold_m=5000.0, catalog_cache=cache)
    near = nearest_approach(state, epoch, 3600.0, [ISS_TLE], dt=30.0,
                            catalog_cache=cache)

    assert near is not None, "nearest object should be reported regardless of threshold"
    assert near.name == "ISS (ZARYA)"
    assert near.min_separation_m > 0.0
    assert 0.0 <= near.time_of_closest_approach_s <= 3600.0

    # The whole point: silent threshold check, but the detail still exists.
    if not hits:
        assert near.min_separation_m >= 5000.0


def test_nearest_approach_empty_and_out_of_band():
    epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    a     = R_EARTH + 400e3
    state = np.array([a, 0.0, 0.0, 0.0, 0.0, np.sqrt(MU_EARTH / a)])

    assert nearest_approach(state, epoch, 3600.0, [], dt=30.0) is None
    # An object 500 km above the band never survives the altitude filter.
    far = _make_tle(900.0, 13001)
    assert nearest_approach(state, epoch, 3600.0, [far], dt=30.0) is None


def test_screened_count_is_the_filtered_subset():
    """Screened count must reflect the altitude band, not the whole catalog."""
    epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    a     = R_EARTH + 500e3
    state = np.array([a, 0.0, 0.0, 0.0, 0.0, np.sqrt(MU_EARTH / a)])

    catalog = [
        _make_tle(500.0, 13101),   # in band
        _make_tle(505.0, 13102),   # in band
        _make_tle(1200.0, 13103),  # far above — filtered out
        _make_tle(300.0, 13104),   # far below — filtered out
    ]

    n = screened_count(state, epoch, 3600.0, catalog, dt=30.0)
    assert 0 < n < len(catalog), f"expected a strict subset of {len(catalog)}, got {n}"
    assert screened_count(state, epoch, 3600.0, [], dt=30.0) == 0


# ---------------------------------------------------------------------------
# Test 9 — an auth failure must never look like an empty sky
# ---------------------------------------------------------------------------
def test_login_failure_raises_instead_of_caching_empty(tmp_path, monkeypatch):
    """A rejected login must raise, not cache a 24-hour empty catalog.

    Regression, found on the first live run: Space-Track answers a bad login with
    401 but leaves the session usable, and the catalog query then returns 204 No
    Content. raise_for_status() treats 204 as success because it is a 2xx, so
    _parse("") returned [] and fetch_tles cached an empty catalog for a day. The
    conjunction checker would then declare every orbit safe.
    """
    monkeypatch.setenv("SPACE_TRACK_USER", "u")
    monkeypatch.setenv("SPACE_TRACK_PASS", "short")
    cache_file = tmp_path / "tle_cache.json"
    monkeypatch.setattr("awareness.tle_fetcher.CACHE_PATH", cache_file)

    login = MagicMock(status_code=401,
                      text='{"Login":"Password does not meet minimum length requirements."}')
    query = MagicMock(status_code=204, text="")
    query.raise_for_status = MagicMock()          # 204 is a 2xx — does not raise
    session = MagicMock()
    session.post.return_value = login
    session.get.return_value = query

    with patch("awareness.tle_fetcher.requests.Session", return_value=session):
        with pytest.raises(RuntimeError, match="login failed"):
            fetch_tles(cache_ttl=0)

    assert not cache_file.exists(), "a failed login must not write a cache entry"


def test_empty_catalog_body_raises_instead_of_caching(tmp_path, monkeypatch):
    """Even with a clean login, a body holding no TLEs must not be cached."""
    monkeypatch.setenv("SPACE_TRACK_USER", "u")
    monkeypatch.setenv("SPACE_TRACK_PASS", "p")
    cache_file = tmp_path / "tle_cache.json"
    monkeypatch.setattr("awareness.tle_fetcher.CACHE_PATH", cache_file)

    login = MagicMock(status_code=200, text="")
    query = MagicMock(status_code=204, text="")
    query.raise_for_status = MagicMock()
    session = MagicMock()
    session.post.return_value = login
    session.get.return_value = query

    with patch("awareness.tle_fetcher.requests.Session", return_value=session):
        with pytest.raises(RuntimeError, match="no usable TLEs"):
            fetch_tles(cache_ttl=0)

    assert not cache_file.exists()


# ---------------------------------------------------------------------------
# Test 10 — _parse must not silently drop objects on either Space-Track format
# ---------------------------------------------------------------------------
def _pair(norad: int) -> tuple[str, str]:
    return (f"1 {norad:05d}U 24001A   24001.00000000  .00000000  00000+0  00000+0 0  9990",
            f"2 {norad:05d}  97.0000   0.0000 0001000  90.0000 270.0000 15.00000000 00001")


def test_parse_two_line_format_keeps_every_object():
    """format/tle returns bare pairs with no name line.

    Regression: _parse stepped by 3 assuming 3-line groups, so against 2-line
    data it kept one object in three and used the previous object's line 2 as
    the name. A live pull returned 10,788 objects out of roughly 32,000, and
    every name was a raw TLE line — silently, because each emitted tuple was
    internally well-formed.
    """
    n = 9
    text = "\n".join(l for i in range(n) for l in _pair(20000 + i))
    parsed = _parse(text)

    assert len(parsed) == n, f"expected all {n} objects, kept {len(parsed)}"
    for name, l1, l2 in parsed:
        assert not name.startswith(("1 ", "2 ")), f"name is a TLE line: {name!r}"
        assert l1.startswith("1 ") and l2.startswith("2 ")
        assert l1[2:7] == l2[2:7], "line1/line2 NORAD ids must match"


def test_parse_three_line_format_uses_real_names():
    """format/3le prefixes each pair with '0 NAME' — the name must survive."""
    entries = [("ISS (ZARYA)", 25544), ("STARLINK-1007", 44713), ("COSMOS 2251 DEB", 34561)]
    text = "\n".join(
        line for name, norad in entries for line in (f"0 {name}", *_pair(norad))
    )
    parsed = _parse(text)

    assert len(parsed) == len(entries)
    assert [p[0] for p in parsed] == [e[0] for e in entries], "names should round-trip"


def test_parse_survives_a_malformed_entry_without_losing_the_rest():
    """A stray line must not shift the grouping and swallow later objects."""
    good = "\n".join(l for i in range(4) for l in _pair(30000 + i))
    text = good + "\nGARBAGE LINE THAT IS NOT A TLE\n" + "\n".join(_pair(39999))
    parsed = _parse(text)
    assert len(parsed) == 5, f"expected 4 good + 1 after the garbage, got {len(parsed)}"


# ---------------------------------------------------------------------------
# Test 11 — the GPU and numpy backends must produce identical answers
# ---------------------------------------------------------------------------
from awareness.conjunction import GPU_AVAILABLE  # noqa: E402


def _band_catalog(n: int = 40, centre_km: float = 500.0) -> list[tuple]:
    """Objects spread across the screening band so several survive the filter."""
    return [_make_tle(centre_km - 20.0 + i, 14000 + i) for i in range(n)]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not usable on this machine")
def test_gpu_and_numpy_agree():
    """Same query, both backends, identical results.

    The GPU path exists only for speed. If it ever disagrees with numpy about a
    separation, it is a safety bug — this is a conjunction checker, and the two
    implementations must be interchangeable.
    """
    epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    a = R_EARTH + 500e3
    state = np.array([a, 0.0, 0.0, 0.0, 0.0, np.sqrt(MU_EARTH / a)])
    catalog = _band_catalog()
    duration, dt = 7200.0, 30.0

    gpu_cache = CatalogCache(use_gpu=True)
    cpu_cache = CatalogCache(use_gpu=False)
    assert gpu_cache.use_gpu and not cpu_cache.use_gpu

    g_hits = check_conjunctions(state, epoch, duration, catalog, dt=dt,
                                threshold_m=1e9, catalog_cache=gpu_cache)
    c_hits = check_conjunctions(state, epoch, duration, catalog, dt=dt,
                                threshold_m=1e9, catalog_cache=cpu_cache)

    assert len(g_hits) == len(c_hits) > 0, "test needs objects inside the band"
    for g, c in zip(g_hits, c_hits):
        assert g.norad_id == c.norad_id
        assert g.min_separation_m == pytest.approx(c.min_separation_m, rel=1e-12)
        assert g.time_of_closest_approach_s == c.time_of_closest_approach_s

    g_near = nearest_approach(state, epoch, duration, catalog, dt=dt,
                              catalog_cache=gpu_cache)
    c_near = nearest_approach(state, epoch, duration, catalog, dt=dt,
                              catalog_cache=cpu_cache)
    assert g_near.norad_id == c_near.norad_id
    assert g_near.min_separation_m == pytest.approx(c_near.min_separation_m, rel=1e-12)


def test_numpy_backend_used_when_gpu_disabled():
    """use_gpu=False must keep the catalog on the host regardless of hardware."""
    epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    a = R_EARTH + 500e3
    state = np.array([a, 0.0, 0.0, 0.0, 0.0, np.sqrt(MU_EARTH / a)])
    cache = CatalogCache(use_gpu=False)
    pos, names, _ = cache.get_or_compute(_band_catalog(), epoch, 3600.0, 30.0, a)
    assert names
    assert isinstance(pos, np.ndarray), "forced-numpy cache must hold a host array"
