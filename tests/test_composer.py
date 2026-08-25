"""
Tests for output/composer.py.

format_report tests need no external dependencies.
plot_ground_track tests mock the C++ integrator call.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.agent import MissionPlan
from optimizer.optimizer import OptimizationResult
from output.composer import (
    _eci_to_latlon,
    _gmst_rad,
    _sample_trajectory,
    _split_at_wraps,
    format_report,
    plot_ground_track,
)
from solver.constraint_solver import OrbitalBounds, R_EARTH

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EPOCH = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

_SMA = R_EARTH + 500e3   # 500 km altitude


def _make_plan(orbit_type="sun_synchronous", safe=True, ecc=0.0001) -> MissionPlan:
    """Build a minimal MissionPlan fixture without running the optimizer."""
    inc = math.radians(97.6)
    elements = np.array([_SMA, inc, ecc, math.radians(45.0), math.radians(22.0)])

    # Circular equatorial ECI state at perigee for simplicity
    v_circ = math.sqrt(3.986004418e14 / _SMA)
    state = np.array([_SMA, 0.0, 0.0, 0.0, v_circ, 0.0])

    result = OptimizationResult(
        success=True,
        elements=elements,
        state=state,
        objective=ecc,
        conjunctions_checked=1000,
        n_generations=42,
        elapsed_s=17.3,
        message="Optimization terminated successfully.",
        safe=safe,
    )

    bounds = OrbitalBounds(
        sma_min=_SMA - 20e3,
        sma_max=_SMA + 20e3,
        inc_min=math.radians(97.0),
        inc_max=math.radians(98.0),
    )

    intent = {
        "orbit_type": orbit_type,
        "altitude_km": 500.0,
        "duration_days": 3.0,
        "rationale": "SSO at 500 km for 3-day Earth observation",
    }

    return MissionPlan(
        description="3-day sun-synchronous Earth observation at 500 km",
        intent=intent,
        bounds=bounds,
        result=result,
        catalog_size=24500,
    )


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


def test_format_report_contains_mission_description():
    plan = _make_plan()
    report = format_report(plan, epoch=_EPOCH)
    assert "3-day sun-synchronous Earth observation at 500 km" in report


def test_format_report_contains_orbital_elements():
    plan = _make_plan()
    report = format_report(plan, epoch=_EPOCH)
    # Altitude should appear as ~500 km
    assert "500" in report
    # Inclination should appear as ~97.6°
    assert "97.6" in report


def test_format_report_safe_status():
    plan = _make_plan(safe=True)
    report = format_report(plan, epoch=_EPOCH)
    assert "SAFE" in report


def test_format_report_unsafe_status():
    plan = _make_plan(safe=False)
    report = format_report(plan, epoch=_EPOCH)
    assert "CONJUNCTION RISK" in report


def test_format_report_catalog_size():
    plan = _make_plan()
    report = format_report(plan, epoch=_EPOCH)
    assert "24,500" in report or "24500" in report


def test_format_report_optimizer_stats():
    plan = _make_plan()
    report = format_report(plan, epoch=_EPOCH)
    assert "42" in report      # n_generations
    assert "17.3" in report    # elapsed_s
    assert "1,000" in report or "1000" in report  # conjunction_calls


def test_format_report_returns_string():
    plan = _make_plan()
    result = format_report(plan, epoch=_EPOCH)
    assert isinstance(result, str)
    assert len(result) > 100


# ---------------------------------------------------------------------------
# _gmst_rad
# ---------------------------------------------------------------------------


def test_gmst_rad_j2000_epoch():
    """GMST at J2000.0 should be ~280.46° (≈ 4.895 rad)."""
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    gmst = _gmst_rad(j2000)
    assert abs(math.degrees(gmst) - 280.46) < 0.1


# ---------------------------------------------------------------------------
# _eci_to_latlon
# ---------------------------------------------------------------------------


def test_eci_to_latlon_equatorial():
    """Satellite on the equatorial plane should return lat ≈ 0."""
    # Position in equatorial plane: x-axis, y = z = 0
    positions = np.array([[_SMA, 0.0, 0.0]])
    lat, lon = _eci_to_latlon(positions, _EPOCH, dt=60.0)
    assert abs(lat[0]) < 1e-6


def test_eci_to_latlon_north_pole():
    """Satellite directly above north pole should return lat = 90°."""
    positions = np.array([[0.0, 0.0, _SMA]])
    lat, lon = _eci_to_latlon(positions, _EPOCH, dt=60.0)
    assert abs(lat[0] - 90.0) < 1e-4


def test_eci_to_latlon_lon_in_range():
    """Returned longitudes must be in [-180, 180]."""
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((50, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions *= _SMA
    _, lon = _eci_to_latlon(positions, _EPOCH, dt=60.0)
    assert np.all(lon >= -180.0) and np.all(lon <= 180.0)


# ---------------------------------------------------------------------------
# _split_at_wraps
# ---------------------------------------------------------------------------


def test_split_at_wraps_no_wrap():
    lon = np.array([10.0, 20.0, 30.0, 40.0])
    lat = np.zeros(4)
    segs = _split_at_wraps(lon, lat)
    assert len(segs) == 1
    np.testing.assert_array_equal(segs[0][0], lon)


def test_split_at_wraps_one_wrap():
    lon = np.array([170.0, 175.0, -175.0, -170.0])
    lat = np.zeros(4)
    segs = _split_at_wraps(lon, lat)
    assert len(segs) == 2
    assert len(segs[0][0]) == 2
    assert len(segs[1][0]) == 2


# ---------------------------------------------------------------------------
# _sample_trajectory — reported spacing must match the propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "days,dt,n_samples",
    [(1.0, 60.0, 500), (3.0, 60.0, 500), (7.0, 60.0, 500), (1.0, 70.0, 500), (0.25, 60.0, 40)],
)
def test_sample_trajectory_dt_matches_propagation(days, dt, n_samples):
    """sample_dt must equal the wall-clock time each sample actually advances.

    Regression: sample_dt was derived as duration_s / n_samples while the
    integrator advanced `total_steps // n_samples` whole steps per sample. The
    two disagree whenever n_samples does not divide total_steps — a 7% longitude
    error over 3 days and 30% over 1 day, plus a track that stopped short of the
    end of the mission.
    """
    duration_s = days * 86400.0
    total_steps = max(1, int(duration_s / dt))
    recorded: list[int] = []

    def fake_final(state, dt_, n_steps):
        recorded.append(n_steps)
        return np.asarray(state, dtype=float)

    with patch("output.composer.orbit_integrator") as mock_oi:
        mock_oi.propagate_single_final.side_effect = fake_final
        state0 = np.array([_SMA, 0.0, 0.0, 0.0, 7600.0, 0.0])
        states, sample_dt = _sample_trajectory(state0, dt, total_steps, n_samples)

    # Uniform spacing — _eci_to_latlon assumes a single dt between all samples
    assert len(set(recorded)) == 1, "chunks must all be the same size"
    assert sample_dt == pytest.approx(recorded[0] * dt)

    # The track spans the whole propagation horizon, overshooting by under one sample
    propagated = len(recorded) * recorded[0]
    assert propagated >= total_steps, "track stops short of the mission end"
    assert propagated - total_steps < recorded[0]

    # One row per sample, plus the initial state
    assert states.shape == (len(recorded) + 1, 6)


# ---------------------------------------------------------------------------
# plot_ground_track — mock the C++ integrator
# ---------------------------------------------------------------------------


def _fake_propagate(state: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """Advance state by nudging the x-component; returns a near-circular position."""
    # Rotate slightly in the x-y plane so the ground track moves
    angle = n_steps * dt * 7.2921150e-5 * 0.9  # slightly less than Earth's rotation
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = state[0], state[1], state[2]
    vx, vy, vz = state[3], state[4], state[5]
    return np.array([c * x - s * y, s * x + c * y, z, c * vx - s * vy, s * vx + c * vy, vz])


def test_plot_ground_track_returns_figure():
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for tests
    import matplotlib.pyplot as plt
    import matplotlib.figure

    plan = _make_plan()
    with patch("output.composer.orbit_integrator") as mock_oi:
        # propagate_single_final returns a state slightly rotated each call
        call_count = [0]
        current_state = [plan.result.state.copy()]

        def fake_final(state, dt, n_steps):
            current_state[0] = _fake_propagate(current_state[0], dt, n_steps)
            return current_state[0]

        mock_oi.propagate_single_final.side_effect = fake_final

        fig = plot_ground_track(plan, epoch=_EPOCH, dt=60.0, n_samples=20)

    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_ground_track_saves_file(tmp_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = _make_plan()
    out_file = tmp_path / "ground_track.png"

    with patch("output.composer.orbit_integrator") as mock_oi:
        current_state = [plan.result.state.copy()]

        def fake_final(state, dt, n_steps):
            current_state[0] = _fake_propagate(current_state[0], dt, n_steps)
            return current_state[0]

        mock_oi.propagate_single_final.side_effect = fake_final
        fig = plot_ground_track(plan, epoch=_EPOCH, dt=60.0, n_samples=20, save_path=out_file)

    assert out_file.exists()
    assert out_file.stat().st_size > 0
    plt.close(fig)


# ---------------------------------------------------------------------------
# Conjunction detail in the report — previously computed then discarded
# ---------------------------------------------------------------------------


def _plan_with_conjunctions(hits=(), nearest=None, screened=1234) -> MissionPlan:
    from awareness.conjunction import ConjunctionResult  # noqa: F401  (kept local)
    plan = _make_plan(safe=not hits)
    plan.result.conjunctions = list(hits)
    plan.result.nearest = nearest
    plan.result.catalog_screened = screened
    return plan


def _hit(name, norad, km, t_s):
    from awareness.conjunction import ConjunctionResult
    return ConjunctionResult(
        norad_id=norad, name=name,
        min_separation_m=km * 1000.0,
        time_of_closest_approach_s=t_s,
    )


def test_report_distinguishes_screened_from_catalog_size():
    """Quoting the full catalog beside the verdict overstates what was checked."""
    plan = _plan_with_conjunctions(nearest=_hit("SOMESAT", "12345", 47.5, 3600.0),
                                   screened=1234)
    report = format_report(plan, epoch=_EPOCH)

    assert "Screened:" in report
    assert "1,234" in report, "screened subset should be reported"
    assert f"{plan.catalog_size:,}" in report, "full catalog size should still appear"
    assert plan.catalog_size != 1234, "fixture must make the two numbers distinguishable"


def test_report_quantifies_a_safe_verdict():
    """SAFE should carry a margin — 6 km clear and 600 km clear must not read alike."""
    plan = _plan_with_conjunctions(nearest=_hit("SOMESAT", "12345", 47.5, 90061.0))
    report = format_report(plan, epoch=_EPOCH)

    assert "Closest approach:" in report
    assert "47.50 km" in report
    assert "SOMESAT" in report and "12345" in report
    assert "T+1 d 01:01" in report, "TCA should be a readable offset"


def test_report_lists_threshold_violations():
    hits = [_hit("BAD-1", "111", 1.2, 60.0), _hit("BAD-2", "222", 3.4, 120.0)]
    plan = _plan_with_conjunctions(hits=hits, nearest=hits[0])
    report = format_report(plan, epoch=_EPOCH)

    assert "CONJUNCTION RISK" in report
    assert "Objects within threshold (2)" in report
    assert "BAD-1" in report and "BAD-2" in report
    assert "1.20 km" in report


def test_report_handles_nothing_in_band():
    plan = _plan_with_conjunctions(nearest=None, screened=0)
    report = format_report(plan, epoch=_EPOCH)
    assert "No catalog objects crossed" in report


def test_report_never_prints_the_empty_shell_sentinel_as_a_distance():
    """An empty shell scores a sentinel, not a real margin.

    Rendering it verbatim produced 'Objective (margin):1000000.00 km' — a million
    kilometres, which is not a distance anyone measured.
    """
    plan = _plan_with_conjunctions(nearest=None, screened=0)
    plan.result.objective = -1.0e9
    report = format_report(plan, epoch=_EPOCH)

    assert "1000000" not in report and "1,000,000" not in report
    assert "no objects in band" in report


def test_report_shows_margin_in_km_when_there_is_traffic():
    plan = _plan_with_conjunctions(nearest=_hit("SOMESAT", "12345", 13761.4, 60.0))
    plan.result.objective = -13_761_400.0
    report = format_report(plan, epoch=_EPOCH)
    assert "13,761.40 km" in report
