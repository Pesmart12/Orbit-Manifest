"""
Validation tests for the constraint solver.

Each test checks that the returned OrbitalBounds are physically correct
(not just that the code runs without error).
"""
import numpy as np
import pytest

from solver.constraint_solver import (
    OrbitalBounds,
    MU_EARTH, R_EARTH, J2, OMEGA_SUN,
    sun_synchronous,
    low_earth_orbit,
    polar,
    iss_rendezvous,
    custom,
    _sun_sync_inclination,
    _sma_from_altitude_km,
)


# ---------------------------------------------------------------------------
# OrbitalBounds sanity / validation
# ---------------------------------------------------------------------------

def test_bounds_rejects_subterranean_sma():
    with pytest.raises(ValueError, match="below Earth's surface"):
        OrbitalBounds(sma_min=500_000, sma_max=600_000,
                      inc_min=0.0, inc_max=np.pi / 2)


def test_bounds_rejects_inverted_sma():
    with pytest.raises(ValueError, match="sma_min must be"):
        OrbitalBounds(sma_min=R_EARTH + 600_000, sma_max=R_EARTH + 400_000,
                      inc_min=0.0, inc_max=1.0)


def test_as_scipy_bounds_length():
    bounds = sun_synchronous(500.0)
    b = bounds.as_scipy_bounds()
    assert len(b) == 5, "optimizer expects exactly 5 element bounds"
    for lo, hi in b:
        assert lo <= hi


# ---------------------------------------------------------------------------
# Sun-synchronous inclination formula
# ---------------------------------------------------------------------------

def test_sun_sync_inclination_known_altitude():
    # At ~500 km the SSO inclination is well-known to be ~97.4°.
    sma = _sma_from_altitude_km(500.0)
    inc_deg = np.rad2deg(_sun_sync_inclination(sma))
    assert 97.0 < inc_deg < 98.0, f"Expected ~97.4°, got {inc_deg:.2f}°"


def test_sun_sync_inclination_retrograde():
    # All LEO SSO inclinations must be retrograde (> 90°) because the
    # J2 drift is prograde for direct orbits — we need the negative sign.
    for alt_km in [300, 500, 700, 900]:
        sma = _sma_from_altitude_km(alt_km)
        inc = _sun_sync_inclination(sma)
        assert inc > np.pi / 2, f"SSO at {alt_km} km should be retrograde"


def test_sun_sync_inclination_drift_rate():
    # Verify the inclination we compute actually produces OMEGA_SUN nodal drift.
    sma = _sma_from_altitude_km(600.0)
    inc = _sun_sync_inclination(sma)
    n = np.sqrt(MU_EARTH / sma**3)
    drift = -1.5 * n * J2 * (R_EARTH / sma)**2 * np.cos(inc)
    assert abs(drift - OMEGA_SUN) < 1e-12, (
        f"Drift {drift:.4e} does not match OMEGA_SUN {OMEGA_SUN:.4e}"
    )


# ---------------------------------------------------------------------------
# sun_synchronous()
# ---------------------------------------------------------------------------

def test_sun_synchronous_bounds_contain_exact_inclination():
    bounds = sun_synchronous(500.0)
    sma_center = (bounds.sma_min + bounds.sma_max) / 2
    inc_exact = _sun_sync_inclination(sma_center)
    assert bounds.inc_min <= inc_exact <= bounds.inc_max


def test_sun_synchronous_sma_band():
    alt, tol = 600.0, 20.0
    bounds = sun_synchronous(alt, tolerance_km=tol)
    expected_center = R_EARTH + alt * 1000
    assert abs((bounds.sma_min + bounds.sma_max) / 2 - expected_center) < 1.0
    assert abs(bounds.sma_max - bounds.sma_min - tol * 2 * 1000) < 1.0


def test_sun_synchronous_near_circular():
    bounds = sun_synchronous(500.0)
    assert bounds.ecc_max <= 0.01


# ---------------------------------------------------------------------------
# low_earth_orbit()
# ---------------------------------------------------------------------------

def test_low_earth_orbit_bounds():
    bounds = low_earth_orbit(400, 600, 45, 55)
    assert bounds.sma_min == pytest.approx(R_EARTH + 400_000, rel=1e-6)
    assert bounds.sma_max == pytest.approx(R_EARTH + 600_000, rel=1e-6)
    assert bounds.inc_min == pytest.approx(np.deg2rad(45), rel=1e-6)
    assert bounds.inc_max == pytest.approx(np.deg2rad(55), rel=1e-6)


def test_low_earth_orbit_rejects_above_leo():
    with pytest.raises(ValueError, match="LEO ceiling"):
        low_earth_orbit(400, 2100, 0, 90)


# ---------------------------------------------------------------------------
# polar()
# ---------------------------------------------------------------------------

def test_polar_centred_on_90_deg():
    bounds = polar(700.0)
    inc_center = np.rad2deg((bounds.inc_min + bounds.inc_max) / 2)
    assert abs(inc_center - 90.0) < 0.01


def test_polar_allows_retrograde():
    bounds = polar(700.0, inc_window_deg=5.0)
    assert bounds.inc_max > np.pi / 2  # must reach past 90°


# ---------------------------------------------------------------------------
# iss_rendezvous()
# ---------------------------------------------------------------------------

def test_iss_rendezvous_inclination():
    bounds = iss_rendezvous()
    inc_center_deg = np.rad2deg((bounds.inc_min + bounds.inc_max) / 2)
    assert abs(inc_center_deg - 51.6) < 0.01


def test_iss_rendezvous_altitude_band():
    bounds = iss_rendezvous(altitude_tolerance_km=15.0)
    alt_center_km = ((bounds.sma_min + bounds.sma_max) / 2 - R_EARTH) / 1000
    assert abs(alt_center_km - 415.0) < 0.1


def test_iss_rendezvous_tight_inclination_window():
    # Plane-change manoeuvres are expensive; the inclination window must be small.
    bounds = iss_rendezvous()
    window_deg = np.rad2deg(bounds.inc_max - bounds.inc_min)
    assert window_deg < 1.0, f"ISS inclination window too wide: {window_deg:.3f}°"


# ---------------------------------------------------------------------------
# custom()
# ---------------------------------------------------------------------------

def test_custom_roundtrip():
    bounds = custom(
        sma_min_m=R_EARTH + 500_000,
        sma_max_m=R_EARTH + 700_000,
        inc_min_deg=30.0,
        inc_max_deg=60.0,
        ecc_max=0.05,
    )
    assert bounds.inc_min == pytest.approx(np.deg2rad(30.0), rel=1e-9)
    assert bounds.inc_max == pytest.approx(np.deg2rad(60.0), rel=1e-9)
    assert bounds.ecc_max == pytest.approx(0.05)
