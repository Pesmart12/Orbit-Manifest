"""
Unit tests for PhysicsAgreementLayer.

All tests use analytically constructed states — no C++ integrator required.
This keeps the test fast and isolated from the build system.

Constants and formulas mirror those in test_integrator.py so that both
files serve as cross-checks of each other.
"""
import numpy as np
import pytest

from solver.physics_agreement import PhysicsAgreementLayer, MU_EARTH, R_EARTH

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pal():
    """Default PhysicsAgreementLayer with standard thresholds."""
    return PhysicsAgreementLayer()


def _circular_state(altitude_km: float) -> np.ndarray:
    """Equatorial circular orbit at a given altitude — at the +x position."""
    a     = R_EARTH + altitude_km * 1e3
    v_c   = np.sqrt(MU_EARTH / a)
    return np.array([a, 0.0, 0.0, 0.0, v_c, 0.0])


# ---------------------------------------------------------------------------
# Pre-propagation tests
# ---------------------------------------------------------------------------

def test_pre_perigee_clearance(pal):
    """Perigee below the 150 km floor must be rejected.

    sma = 500 km altitude, ecc = 0.06 → perigee ≈ 87 km.
    OrbitalBounds allows ecc=0.06 (it's < 1); the physics layer must catch it.
    """
    sma = R_EARTH + 500e3
    ecc = 0.06   # perigee = sma*(1-ecc) ≈ R_EARTH + 87 km — below 150 km floor
    n_steps, dt = 6000, 10.0

    ok, reasons = pal.check_pre_propagation(sma, np.deg2rad(51.6), ecc, 0.0, 0.0, n_steps, dt)

    assert not ok
    assert any("perigee" in r for r in reasons)


def test_pre_apogee_ceiling(pal):
    """Apogee above the 2500 km LEO ceiling must be rejected.

    sma = 1000 km, ecc = 0.22 → apogee ≈ 2623 km.
    """
    sma = R_EARTH + 1000e3
    ecc = 0.22   # apogee = sma*(1+ecc) ≈ R_EARTH + 2623 km — above 2500 km ceiling
    n_steps, dt = 20000, 10.0

    ok, reasons = pal.check_pre_propagation(sma, np.deg2rad(51.6), ecc, 0.0, 0.0, n_steps, dt)

    assert not ok
    assert any("apogee" in r for r in reasons)


def test_pre_period_vs_window(pal):
    """Integration window shorter than one orbital period must be rejected.

    1500 km orbit has period ≈ 116 min.  100 steps × 60 s = 100 min < 116 min.
    """
    sma     = R_EARTH + 1500e3
    T_orbit = 2.0 * np.pi * np.sqrt(sma**3 / MU_EARTH)
    # Deliberately set a window shorter than one period
    dt      = 60.0
    n_steps = int(0.85 * T_orbit / dt)   # 85% of the period — definitely too short

    ok, reasons = pal.check_pre_propagation(sma, np.deg2rad(90.0), 0.0, 0.0, 0.0, n_steps, dt)

    assert not ok
    assert any("integration window" in r for r in reasons)


def test_pre_valid_orbit(pal):
    """Standard 500 km circular orbit with adequate integration window passes all checks."""
    sma     = R_EARTH + 500e3
    T_orbit = 2.0 * np.pi * np.sqrt(sma**3 / MU_EARTH)
    dt      = 10.0
    n_steps = int(10 * T_orbit / dt)   # 10 full orbits — well above one-period floor

    ok, reasons = pal.check_pre_propagation(
        sma, np.deg2rad(51.6), 0.001, np.deg2rad(120.0), np.deg2rad(45.0), n_steps, dt
    )

    assert ok, f"Expected valid orbit to pass; got: {reasons}"
    assert reasons == []


# ---------------------------------------------------------------------------
# Post-propagation tests
# ---------------------------------------------------------------------------

def test_post_energy_drift(pal):
    """A final state with energy shifted by 50 J/kg must be rejected.

    The velocity is nudged by dv = 50/v_circ ≈ 6.6 mm/s — imperceptible in
    distance, but enough to push energy drift above the 10 J/kg threshold.
    """
    s0     = _circular_state(500.0)
    v_circ = s0[4]

    sf        = s0.copy()
    delta_v   = 50.0 / v_circ   # Δv such that v*Δv ≈ 50 J/kg energy shift
    sf[4]    += delta_v          # nudge vy

    ok, reasons = pal.check_post_propagation(s0, sf)

    assert not ok
    assert any("energy drift" in r for r in reasons)


def test_post_momentum_drift(pal):
    """A final state with angular momentum shifted by 0.1% must be rejected.

    The velocity direction is rotated in the orbital plane (keeping |v| = v_circ
    so energy is unchanged) by θ such that 1 − cos(θ) ≈ 1e-3 > 1e-4 threshold.
    h = a × v_circ × cos(θ), so dh/h₀ = 1 − cos(θ).
    """
    s0     = _circular_state(500.0)
    v_circ = s0[4]

    theta = 0.0448   # 1 - cos(theta) ≈ 1e-3 — just above the 1e-4 threshold
    sf        = s0.copy()
    sf[3] = v_circ * np.sin(theta)   # vx — previously zero
    sf[4] = v_circ * np.cos(theta)   # vy — slightly reduced; |v| unchanged

    ok, reasons = pal.check_post_propagation(s0, sf)

    assert not ok
    assert any("angular momentum" in r for r in reasons)


def test_post_earth_clearance(pal):
    """A final state at 50 km altitude (below 80 km floor) must be rejected."""
    s0 = _circular_state(500.0)   # valid initial state

    # Final state: position at 50 km altitude, velocity near circular
    a_low = R_EARTH + 50e3
    v_low = np.sqrt(MU_EARTH / a_low)
    sf    = np.array([a_low, 0.0, 0.0, 0.0, v_low, 0.0])

    ok, reasons = pal.check_post_propagation(s0, sf)

    assert not ok
    assert any("80 km floor" in r for r in reasons)


def test_post_valid_propagation(pal):
    """Two states on the same circular orbit (same E and |h|) must pass all checks.

    s0 is at the +x position; sf is at the +y position (90° along-track).
    Both share identical specific energy and angular momentum magnitude.
    """
    a     = R_EARTH + 500e3
    v_c   = np.sqrt(MU_EARTH / a)

    s0 = np.array([a,   0.0,  0.0,   0.0,  v_c,  0.0])
    sf = np.array([0.0,  a,   0.0,  -v_c,  0.0,  0.0])   # 90° ahead on the same orbit

    ok, reasons = pal.check_post_propagation(s0, sf)

    assert ok, f"Expected valid state pair to pass; got: {reasons}"
    assert reasons == []


# ---------------------------------------------------------------------------
# Velocity plausibility test (edge case for the post check)
# ---------------------------------------------------------------------------

def test_post_velocity_plausibility(pal):
    """A state with velocity far outside [0.5, 1.5] × v_circ must be rejected.

    This catches unit errors — e.g., velocity passed in km/s instead of m/s.
    """
    # Simulate passing s0 velocity in km/s instead of m/s (factor ~1000 too small).
    # The plausibility check runs on s0 (the converted initial state), not sf.
    s0        = _circular_state(500.0)
    s0[3:6]  /= 1000.0   # velocity is now ~7.6 m/s instead of 7615 m/s
    sf        = _circular_state(500.0)   # valid final state

    ok, reasons = pal.check_post_propagation(s0, sf)

    assert not ok
    assert any("v_circ" in r for r in reasons)
