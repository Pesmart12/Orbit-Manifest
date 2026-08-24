"""
Tests for optimizer/optimizer.py.

These tests do NOT require Space-Track credentials — the catalog is either
empty or synthetically populated.  The C++ integrator must be built first.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

try:
    import orbit_integrator as oi
except ImportError as e:
    raise ImportError(
        "orbit_integrator C++ module not found. Build it first with: pip install -e ."
    ) from e

from optimizer.optimizer import (
    keplerian_to_cartesian,
    _mission_objective,
    run_optimizer,
)
from solver.constraint_solver import OrbitalBounds, R_EARTH, MU_EARTH

EPOCH = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Test 1: keplerian_to_cartesian — circular equatorial orbit
# ---------------------------------------------------------------------------
def test_keplerian_to_cartesian_circular():
    a = R_EARTH + 500e3
    elements = np.array([[a, 0.0, 0.0, 0.0, 0.0]])  # circular equatorial

    state = keplerian_to_cartesian(elements)[0]

    # Position should be at (a, 0, 0) for true anomaly = 0, raan = 0, argp = 0
    r = np.linalg.norm(state[:3])
    assert abs(r - a) < 1.0, f"Radius {r:.1f} m should equal sma {a:.1f} m"

    # Speed should equal circular velocity
    v      = np.linalg.norm(state[3:])
    v_circ = np.sqrt(MU_EARTH / a)
    assert abs(v - v_circ) < 1.0, f"Speed {v:.2f} m/s should equal v_circ {v_circ:.2f}"

    # For equatorial, z-component of position and vy should be 0
    assert abs(state[2]) < 1e-9, "z should be 0 for equatorial orbit"


# ---------------------------------------------------------------------------
# Test 2: keplerian_to_cartesian — batch consistency
# ---------------------------------------------------------------------------
def test_keplerian_to_cartesian_batch():
    a   = R_EARTH + 600e3
    inc = np.radians(51.6)
    ecc = 0.001
    elements_single = np.array([a, inc, ecc, 0.5, 1.0])

    N = 10
    batch = np.tile(elements_single, (N, 1))
    states = keplerian_to_cartesian(batch)

    assert states.shape == (N, 6)
    # All rows must be identical since input is tiled
    for i in range(1, N):
        np.testing.assert_allclose(states[i], states[0], atol=1e-10,
                                   err_msg=f"Row {i} differs from row 0")


# ---------------------------------------------------------------------------
# Test 3: keplerian_to_cartesian — 1D input (single element set without batch dim)
# ---------------------------------------------------------------------------
def test_keplerian_to_cartesian_1d_input():
    a = R_EARTH + 400e3
    elements = np.array([a, np.radians(28.5), 0.0, 0.0, 0.0])
    states = keplerian_to_cartesian(elements)
    assert states.shape == (1, 6)
    r = np.linalg.norm(states[0, :3])
    assert abs(r - a) < 1.0


# ---------------------------------------------------------------------------
# Test 4: mission_objective — near-circular orbit scores near 0
# ---------------------------------------------------------------------------
def test_mission_objective_circular():
    a = R_EARTH + 500e3
    elements = np.array([[a, np.radians(45.0), 0.0, 0.0, 0.0]])
    state = keplerian_to_cartesian(elements)[0]

    # Propagate one orbit
    T = 2.0 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
    dt = 30.0
    n_steps = int(T / dt)
    final = np.asarray(oi.propagate_single_final(state, dt, n_steps))

    score = _mission_objective(final)
    # Near-circular orbit should have eccentricity << 0.01
    assert score < 0.01, f"Circular orbit final eccentricity {score:.6f} should be < 0.01"


# ---------------------------------------------------------------------------
# Test 5: mission_objective — eccentric orbit scores higher than circular
# ---------------------------------------------------------------------------
def test_mission_objective_eccentric_greater_than_circular():
    a_circ = R_EARTH + 500e3
    elements_circ = np.array([[a_circ, 0.0, 0.001, 0.0, 0.0]])
    elements_ecc  = np.array([[a_circ, 0.0, 0.1,   0.0, 0.0]])

    state_circ = keplerian_to_cartesian(elements_circ)[0]
    state_ecc  = keplerian_to_cartesian(elements_ecc)[0]

    score_circ = _mission_objective(state_circ)
    score_ecc  = _mission_objective(state_ecc)

    assert score_ecc > score_circ, (
        f"Eccentric orbit ({score_ecc:.4f}) should score worse than "
        f"circular ({score_circ:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 6: run_optimizer — empty catalog, tiny search space, finds safe orbit
# ---------------------------------------------------------------------------
def test_run_optimizer_empty_catalog():
    bounds = OrbitalBounds(
        sma_min=R_EARTH + 490e3,
        sma_max=R_EARTH + 510e3,
        inc_min=np.radians(97.0),
        inc_max=np.radians(98.0),
        ecc_min=0.0,
        ecc_max=0.005,
    )

    result = run_optimizer(
        bounds=bounds,
        epoch=EPOCH,
        # Must exceed one orbital period (~5690 s at 510 km): the pre-propagation
        # screen rejects windows shorter than a revolution, since conjunction
        # screening would then cover only part of the orbit.
        duration_s=7200.0,
        catalog=[],           # empty catalog → no conjunctions possible
        dt=60.0,
        popsize=5,            # tiny population for speed
        maxiter=10,
        seed=42,
    )

    # With no catalog objects, every candidate is safe — optimizer should converge
    assert result.safe, "Best orbit should be safe with an empty catalog"
    # Objective should be low (near-circular)
    assert result.objective < 0.01, f"Objective {result.objective:.4f} should be near-circular"
    # Elements should fall within bounds
    sma, inc, ecc, raan, argp = result.elements
    assert bounds.sma_min <= sma <= bounds.sma_max, "sma out of bounds"
    assert bounds.inc_min <= inc <= bounds.inc_max, "inc out of bounds"
    assert bounds.ecc_min <= ecc <= bounds.ecc_max, "ecc out of bounds"


# ---------------------------------------------------------------------------
# Test 7: run_optimizer — catalog with one object far away; orbit accepted
# ---------------------------------------------------------------------------
def test_run_optimizer_distant_catalog_object():
    # Place a catalog object 200 km above the search band — no conjunctions expected
    a_target = R_EARTH + 500e3
    a_catalog = R_EARTH + 700e3   # 200 km above — outside the 50 km filter band
    inc_catalog = np.radians(45.0)

    # Build a minimal but valid-looking TLE for an object at 700 km
    # Mean motion for 700 km altitude: n = sqrt(mu/a^3) in rad/s → rev/day
    n_rad_s = np.sqrt(MU_EARTH / a_catalog ** 3)
    n_rev_day = n_rad_s * 86400 / (2 * np.pi)

    # Synthetic TLE strings (checksum not validated by sgp4 alt-filter path)
    line1 = "1 99999U 25001A   25001.00000000  .00000000  00000-0  00000-0 0  9990"
    n_str = f"{n_rev_day:11.8f}"
    # Eccentricity field: 7 digits, implied "0." prefix
    ecc_str = "0000001"
    line2 = f"2 99999 {np.degrees(inc_catalog):8.4f}   0.0000   0.0000 {ecc_str}   0.0000 {n_str}    10"

    catalog = [("DISTANT_SAT", line1, line2)]

    bounds = OrbitalBounds(
        sma_min=R_EARTH + 490e3,
        sma_max=R_EARTH + 510e3,
        inc_min=np.radians(97.0),
        inc_max=np.radians(98.0),
        ecc_min=0.0,
        ecc_max=0.005,
    )

    result = run_optimizer(
        bounds=bounds,
        epoch=EPOCH,
        duration_s=7200.0,   # > one orbital period — see the screen note above
        catalog=catalog,
        dt=60.0,
        popsize=5,
        maxiter=10,
        seed=7,
    )

    assert result.safe, "Orbit should be safe when catalog object is 200 km away"


# ---------------------------------------------------------------------------
# Test 8: progress_callback is called each generation
# ---------------------------------------------------------------------------
def test_run_optimizer_progress_callback():
    bounds = OrbitalBounds(
        sma_min=R_EARTH + 490e3,
        sma_max=R_EARTH + 510e3,
        inc_min=np.radians(50.0),
        inc_max=np.radians(52.0),
    )
    calls = []

    def cb(gen: int, best: float) -> None:
        calls.append((gen, best))

    run_optimizer(
        bounds=bounds,
        epoch=EPOCH,
        duration_s=7200.0,   # > one orbital period — see the screen note above
        catalog=[],
        dt=60.0,
        popsize=4,
        maxiter=5,
        seed=0,
        progress_callback=cb,
    )

    assert len(calls) > 0, "progress_callback should have been called at least once"
    gens = [c[0] for c in calls]
    assert gens == sorted(gens), "Generation numbers should be non-decreasing"


# ---------------------------------------------------------------------------
# Test 9: the pre-propagation screen is actually wired into the fitness loop
# ---------------------------------------------------------------------------
def test_pre_propagation_screen_rejects_short_window():
    """A mission shorter than one orbit must be screened out, not propagated.

    Regression: physics/ was implemented and tested but imported by nothing, so
    degenerate candidates reached the integrator and were scored on a final
    state that had not completed a revolution — conjunction screening covering
    only part of the orbit.
    """
    bounds = OrbitalBounds(
        sma_min=R_EARTH + 490e3,
        sma_max=R_EARTH + 510e3,
        inc_min=np.radians(97.0),
        inc_max=np.radians(98.0),
    )

    result = run_optimizer(
        bounds=bounds,
        epoch=EPOCH,
        duration_s=1800.0,   # ~32 min, well under the ~95 min period
        catalog=[],
        dt=60.0,
        popsize=4,
        maxiter=3,
        seed=0,
    )

    assert result.screened_out > 0, "screen did not reject any sub-period candidate"
    assert result.conjunctions_checked == 0, (
        "screened candidates must not reach the conjunction check"
    )
    assert result.objective >= 1e10, "a fully screened population has no valid score"


def test_pre_propagation_screen_passes_valid_orbits():
    """A well-formed mission must survive the screen untouched."""
    bounds = OrbitalBounds(
        sma_min=R_EARTH + 490e3,
        sma_max=R_EARTH + 510e3,
        inc_min=np.radians(97.0),
        inc_max=np.radians(98.0),
        ecc_max=0.005,
    )

    result = run_optimizer(
        bounds=bounds,
        epoch=EPOCH,
        duration_s=7200.0,   # > one orbital period
        catalog=[],
        dt=60.0,
        popsize=4,
        maxiter=3,
        seed=0,
    )

    assert result.screened_out == 0, "well-formed candidates should not be screened out"
    assert result.conjunctions_checked > 0
