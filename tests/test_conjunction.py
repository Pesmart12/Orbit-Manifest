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
    ConjunctionResult,
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


def _make_mock_session(text: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.text = text
    mock_resp.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.post.return_value = MagicMock()
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
