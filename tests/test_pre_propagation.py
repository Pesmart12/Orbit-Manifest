"""
Unit tests for physics/pre_propagation.py (OrbitScreen).

All tests use Keplerian parameters only — no C++ integrator required.
This keeps the suite fast and isolated from the build system.
"""
import numpy as np
import pytest

from physics.pre_propagation import OrbitScreen, MU_EARTH, R_EARTH


@pytest.fixture
def screen():
    """Default OrbitScreen with standard thresholds."""
    return OrbitScreen()


def test_perigee_clearance(screen):
    """Perigee below the 150 km floor must be rejected.

    sma = 500 km altitude, ecc = 0.06 → perigee ≈ 87 km.
    OrbitalBounds allows ecc=0.06 (it's < 1); OrbitScreen must catch it.
    """
    sma = R_EARTH + 500e3
    ecc = 0.06   # perigee = sma*(1-ecc) ≈ R_EARTH + 87 km — below 150 km floor
    n_steps, dt = 6000, 10.0

    ok, reasons = screen.check_pre_propagation(sma, np.deg2rad(51.6), ecc, 0.0, 0.0, n_steps, dt)

    assert not ok
    assert any("perigee" in r for r in reasons)


def test_apogee_ceiling(screen):
    """Apogee above the 2500 km LEO ceiling must be rejected.

    sma = 1000 km, ecc = 0.22 → apogee ≈ 2623 km.
    """
    sma = R_EARTH + 1000e3
    ecc = 0.22   # apogee = sma*(1+ecc) ≈ R_EARTH + 2623 km — above 2500 km ceiling
    n_steps, dt = 20000, 10.0

    ok, reasons = screen.check_pre_propagation(sma, np.deg2rad(51.6), ecc, 0.0, 0.0, n_steps, dt)

    assert not ok
    assert any("apogee" in r for r in reasons)


def test_integration_window_too_short(screen):
    """Integration window shorter than one orbital period must be rejected.

    1500 km orbit has period ≈ 116 min. Window is set to 85% of that.
    """
    sma     = R_EARTH + 1500e3
    T_orbit = 2.0 * np.pi * np.sqrt(sma**3 / MU_EARTH)
    dt      = 60.0
    n_steps = int(0.85 * T_orbit / dt)   # 85% of the period — too short

    ok, reasons = screen.check_pre_propagation(sma, np.deg2rad(90.0), 0.0, 0.0, 0.0, n_steps, dt)

    assert not ok
    assert any("integration window" in r for r in reasons)


def test_valid_orbit_passes(screen):
    """Standard 500 km circular orbit with adequate integration window passes all checks."""
    sma     = R_EARTH + 500e3
    T_orbit = 2.0 * np.pi * np.sqrt(sma**3 / MU_EARTH)
    dt      = 10.0
    n_steps = int(10 * T_orbit / dt)   # 10 full orbits — well above one-period floor

    ok, reasons = screen.check_pre_propagation(
        sma, np.deg2rad(51.6), 0.001, np.deg2rad(120.0), np.deg2rad(45.0), n_steps, dt
    )

    assert ok, f"Expected valid orbit to pass; got: {reasons}"
    assert reasons == []


def test_custom_perigee_floor(screen):
    """OrbitScreen respects a custom min_perigee_alt_m threshold.

    A 300 km circular orbit passes the default 150 km floor but fails a
    custom 400 km floor.
    """
    strict_screen = OrbitScreen(min_perigee_alt_m=400_000.0)
    sma     = R_EARTH + 300e3
    T_orbit = 2.0 * np.pi * np.sqrt(sma**3 / MU_EARTH)
    dt      = 10.0
    n_steps = int(10 * T_orbit / dt)

    ok_default, _ = screen.check_pre_propagation(sma, 0.0, 0.0, 0.0, 0.0, n_steps, dt)
    ok_strict, reasons = strict_screen.check_pre_propagation(sma, 0.0, 0.0, 0.0, 0.0, n_steps, dt)

    assert ok_default, "300 km orbit should pass the default 150 km floor"
    assert not ok_strict
    assert any("perigee" in r for r in reasons)


def test_multiple_failures_reported(screen):
    """Both perigee and window failures are reported together, not just the first."""
    sma = R_EARTH + 500e3
    ecc = 0.06         # perigee ≈ 87 km — below floor
    dt, n_steps = 10.0, 10   # window = 100 s — far shorter than ~95 min period

    ok, reasons = screen.check_pre_propagation(sma, 0.0, ecc, 0.0, 0.0, n_steps, dt)

    assert not ok
    assert len(reasons) >= 2
    assert any("perigee" in r for r in reasons)
    assert any("integration window" in r for r in reasons)
