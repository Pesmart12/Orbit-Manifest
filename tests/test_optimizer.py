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
from awareness.conjunction import ConjunctionResult
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
# Test 4/5: mission_objective — conjunction margin, lower score = more clearance
# ---------------------------------------------------------------------------
def _result(sep_km: float) -> ConjunctionResult:
    return ConjunctionResult(
        norad_id="00001", name="TEST",
        min_separation_m=sep_km * 1000.0,
        time_of_closest_approach_s=0.0,
    )


def test_mission_objective_is_negated_margin():
    """The score is the margin, negated, in metres — DE minimises."""
    assert _mission_objective(_result(47.5)) == pytest.approx(-47_500.0)


def test_mission_objective_prefers_more_clearance():
    """More margin must score strictly better.

    The old objective gated safety instead of grading it: every candidate above
    the 5 km threshold scored identically on separation, so 5.1 km and 500 km of
    clearance were indistinguishable.
    """
    tight = _mission_objective(_result(5.1))
    roomy = _mission_objective(_result(500.0))
    assert roomy < tight, "500 km of clearance should beat 5.1 km"


def test_mission_objective_empty_shell_beats_any_real_margin():
    """No objects in the altitude band is the safest possible outcome.

    It must outrank every finite margin — scoring it neutrally would rank an
    empty shell *below* an orbit with traffic 50 km away.
    """
    empty = _mission_objective(None)
    assert empty < _mission_objective(_result(2000.0))
    assert empty < 0.0


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
    # Nothing in the shell is the best possible margin, so the objective takes the
    # empty-shell sentinel rather than any finite negated distance.
    assert result.objective == pytest.approx(-1.0e9), (
        f"empty catalog should score the empty-shell sentinel, got {result.objective}"
    )
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


# ---------------------------------------------------------------------------
# Test 10: the margin objective actually buys clearance
# ---------------------------------------------------------------------------
def test_optimizer_finds_more_clearance_than_an_arbitrary_orbit():
    """The optimizer must beat a mid-band orbit on actual separation.

    This is the property the old objective could not have: terminal eccentricity
    was blind to raan entirely and gated separation at a threshold, so every safe
    candidate tied. Margin makes clearance the thing being maximised.
    """
    from awareness.conjunction import CatalogCache, nearest_approach

    # Column layout copied from tests/test_conjunction.py::_make_tle, which the
    # altitude-filter tests already exercise — mean motion must land in cols 52-62.
    a_obj = R_EARTH + 500e3          # a catalog object sitting mid-band
    n_rev_day = np.sqrt(MU_EARTH / a_obj ** 3) * 86400.0 / (2.0 * np.pi)
    line1 = ("1 99998U 25001A   25001.00000000  .00000000"
             "  00000+0  00000+0 0  9990")
    line2 = (f"2 99998  97.5000   0.0000 0000001  90.0000"
             f" 270.0000 {n_rev_day:11.8f}    10")
    catalog = [("MIDBAND_SAT", line1, line2)]

    bounds = OrbitalBounds(
        sma_min=R_EARTH + 480e3,
        sma_max=R_EARTH + 520e3,
        inc_min=np.radians(97.0),
        inc_max=np.radians(98.0),
        ecc_min=0.0,
        ecc_max=0.002,
    )

    duration_s, dt = 7200.0, 60.0
    result = run_optimizer(
        bounds=bounds, epoch=EPOCH, duration_s=duration_s, catalog=catalog,
        dt=dt, popsize=8, maxiter=25, seed=3,
    )

    # Margin of a plain mid-band orbit, for comparison.
    baseline_state = keplerian_to_cartesian(
        np.array([[bounds.sma_center, bounds.inc_center, 0.0, 0.0, 0.0]])
    )[0]
    baseline = nearest_approach(
        baseline_state, EPOCH, duration_s, catalog, dt=dt,
        catalog_cache=CatalogCache(),
    )

    assert result.nearest is not None, "the object should have been screened"
    assert baseline is not None

    assert result.nearest.min_separation_m > baseline.min_separation_m, (
        f"optimizer margin {result.nearest.min_separation_m / 1e3:.1f} km did not "
        f"beat a mid-band orbit's {baseline.min_separation_m / 1e3:.1f} km"
    )
    # And the score must be exactly the negated margin it reports.
    assert result.objective == pytest.approx(-result.nearest.min_separation_m)
