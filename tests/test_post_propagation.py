"""
Unit tests for physics/post_propagation.py.

All tests use analytically constructed Cartesian states — no C++ integrator
required. Constants and energy formulas mirror test_integrator.py so both
files act as cross-checks of each other.
"""
import numpy as np
import pytest

from physics.post_propagation import check_post_propagation, specific_energy, MU_EARTH, R_EARTH, J2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circular_state(altitude_km: float) -> np.ndarray:
    """Equatorial circular orbit at the given altitude, positioned at +x."""
    a   = R_EARTH + altitude_km * 1e3
    v_c = np.sqrt(MU_EARTH / a)
    return np.array([a, 0.0, 0.0, 0.0, v_c, 0.0])


# ---------------------------------------------------------------------------
# specific_energy tests
# ---------------------------------------------------------------------------

def test_specific_energy_circular():
    """Specific energy of a circular orbit equals -μ/(2a) (2-body term dominates at LEO)."""
    a = R_EARTH + 500e3
    s = _circular_state(500.0)
    E = specific_energy(s)

    E_2body = -MU_EARTH / (2.0 * a)
    # J2 correction is small (~1e-3 of E_2body); allow 1% relative tolerance
    assert abs(E - E_2body) / abs(E_2body) < 0.01


def test_specific_energy_j2_correction():
    """J2 term is non-zero and has the correct sign for an equatorial orbit.

    At the equatorial plane z=0, the J2 potential term is:
    μ·J₂·Re²/(2r³) · (3·0/r² − 1) = −μ·J₂·Re²/(2r³)  (negative)
    """
    s = _circular_state(500.0)
    x, y, z = s[0], s[1], s[2]
    r = np.sqrt(x**2 + y**2 + z**2)

    e_j2_expected = MU_EARTH * J2 * R_EARTH**2 / (2.0 * r**3) * (3.0 * z**2 / r**2 - 1.0)

    v2 = s[3]**2 + s[4]**2 + s[5]**2
    e_2body = 0.5 * v2 - MU_EARTH / r
    E = specific_energy(s)

    assert abs(E - (e_2body + e_j2_expected)) < 1e-6
    assert e_j2_expected < 0   # negative for equatorial orbit


def test_specific_energy_conserved_same_orbit():
    """Two states on the same circular orbit have identical specific energy."""
    a   = R_EARTH + 500e3
    v_c = np.sqrt(MU_EARTH / a)

    s0 = np.array([a,   0.0, 0.0,  0.0,  v_c, 0.0])
    sf = np.array([0.0,  a,  0.0, -v_c,  0.0, 0.0])   # 90° ahead

    assert abs(specific_energy(s0) - specific_energy(sf)) < 1e-6


# ---------------------------------------------------------------------------
# check_post_propagation tests
# ---------------------------------------------------------------------------

def test_energy_drift_rejected():
    """A final state with energy shifted by 50 J/kg must be rejected.

    The velocity is nudged by dv = 50/v_circ ≈ 6.6 mm/s — imperceptible in
    distance, but enough to push energy drift above the 10 J/kg threshold.
    """
    s0    = _circular_state(500.0)
    sf    = s0.copy()
    sf[4] += 50.0 / s0[4]   # Δv such that v·Δv ≈ 50 J/kg energy shift

    ok, reasons = check_post_propagation(s0, sf)

    assert not ok
    assert any("energy drift" in r for r in reasons)


def test_momentum_drift_rejected():
    """A final state with angular momentum shifted by 0.1% must be rejected.

    The velocity direction is rotated in the orbital plane (|v| unchanged, so
    energy is preserved) by θ where 1 − cos(θ) ≈ 1e-3 > 1e-4 threshold.
    """
    s0     = _circular_state(500.0)
    v_circ = s0[4]
    theta  = 0.0448   # 1 - cos(theta) ≈ 1e-3

    sf    = s0.copy()
    sf[3] = v_circ * np.sin(theta)
    sf[4] = v_circ * np.cos(theta)

    ok, reasons = check_post_propagation(s0, sf)

    assert not ok
    assert any("angular momentum" in r for r in reasons)


def test_earth_clearance_rejected():
    """A final state at 50 km altitude (below the 80 km floor) must be rejected."""
    s0    = _circular_state(500.0)
    a_low = R_EARTH + 50e3
    sf    = np.array([a_low, 0.0, 0.0, 0.0, np.sqrt(MU_EARTH / a_low), 0.0])

    ok, reasons = check_post_propagation(s0, sf)

    assert not ok
    assert any("80 km floor" in r for r in reasons)


def test_velocity_plausibility_rejected():
    """A state with velocity far outside [0.5, 1.5] × v_circ must be rejected.

    Simulates passing velocity in km/s instead of m/s — a factor ~1000 too small.
    The plausibility check runs on s0, not sf.
    """
    s0       = _circular_state(500.0)
    s0[3:6] /= 1000.0   # ~7.6 m/s instead of ~7615 m/s
    sf       = _circular_state(500.0)

    ok, reasons = check_post_propagation(s0, sf)

    assert not ok
    assert any("v_circ" in r for r in reasons)


def test_valid_state_pair_passes():
    """Two states on the same circular orbit must pass all checks.

    s0 at +x, sf at +y — both share identical energy and |h|.
    """
    a   = R_EARTH + 500e3
    v_c = np.sqrt(MU_EARTH / a)

    s0 = np.array([a,   0.0, 0.0,  0.0,  v_c, 0.0])
    sf = np.array([0.0,  a,  0.0, -v_c,  0.0, 0.0])

    ok, reasons = check_post_propagation(s0, sf)

    assert ok, f"Expected valid state pair to pass; got: {reasons}"
    assert reasons == []


def test_custom_tolerances():
    """Tighter tolerances reject a state pair that passes at default thresholds.

    A 1 J/kg energy shift is within the default 10 J/kg tolerance but outside
    a custom 0.5 J/kg tolerance.
    """
    s0    = _circular_state(500.0)
    sf    = s0.copy()
    sf[4] += 1.0 / s0[4]   # ~1 J/kg energy shift

    ok_default, _ = check_post_propagation(s0, sf, energy_tol=10.0)
    ok_strict, reasons = check_post_propagation(s0, sf, energy_tol=0.5)

    assert ok_default, "1 J/kg shift should pass the default 10 J/kg tolerance"
    assert not ok_strict
    assert any("energy drift" in r for r in reasons)
