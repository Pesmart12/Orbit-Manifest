"""
Output composer — Phase 6.

Formats a MissionPlan into a human-readable text report and a ground-track plot.

Public API:
    format_report(plan, epoch=None) -> str
    plot_ground_track(plan, epoch, dt=60.0, n_samples=500, save_path=None) -> Figure
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agent.agent import MissionPlan
from physics.constants import MU_EARTH, OMEGA_EARTH, R_EARTH

try:
    import orbit_integrator
except ImportError:
    orbit_integrator = None  # type: ignore  — build with: pip install -e .

_WIDTH = 62
_BAR = "─" * _WIDTH


# ---------------------------------------------------------------------------
# Internal helpers — physics
# ---------------------------------------------------------------------------


def _period_min(sma_m: float) -> float:
    return 2 * math.pi * math.sqrt(sma_m ** 3 / MU_EARTH) / 60.0


def _gmst_rad(epoch: datetime) -> float:
    """Approximate Greenwich Mean Sidereal Time in radians at a UTC epoch."""
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    d = (epoch - j2000).total_seconds() / 86400.0
    return math.radians((280.46061837 + 360.98564736629 * d) % 360.0)


def _eci_to_latlon(
    positions: np.ndarray, epoch: datetime, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an (N, 3) ECI position array to (lat, lon) in degrees.

    Accounts for Earth's rotation between each sample using GMST at epoch.
    Returns lat in [-90, 90] and lon in [-180, 180].
    """
    N = len(positions)
    t = np.arange(N) * dt
    gmst0 = _gmst_rad(epoch)
    theta = gmst0 + OMEGA_EARTH * t   # Earth rotation angle at each step

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    lat = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x) - theta)
    lon = ((lon + 180.0) % 360.0) - 180.0   # wrap to [-180, 180]
    return lat, lon


def _sample_trajectory(
    state0: np.ndarray, dt: float, total_steps: int, n_samples: int
) -> tuple[np.ndarray, float]:
    """Sample the trajectory at uniform intervals spanning the whole mission.

    Returns (states, sample_dt): an (n+1, 6) state array and the exact wall-clock
    spacing between consecutive samples, in seconds.

    Each sample advances a whole number of integrator steps, so the true spacing
    is `chunk * dt` — not `duration_s / n_samples`, which is only equal when
    n_samples divides total_steps exactly. Returning the real value is what keeps
    the caller's Earth-rotation correction in step with the propagation.

    `n_samples` is a target, not a guarantee: the count is adjusted so the samples
    span the full mission instead of stopping short of it.

    Calls the C++ integrator in chunks so the full propagation is shared
    with the optimizer's integrator rather than reimplemented here.
    """
    if orbit_integrator is None:
        raise ImportError(
            "orbit_integrator C++ module not found. Build it first with: pip install -e ."
        )
    chunk = max(1, round(total_steps / n_samples))
    n_actual = max(1, math.ceil(total_steps / chunk))
    states = [state0.copy()]
    current = state0.copy()
    for _ in range(n_actual):
        current = np.asarray(orbit_integrator.propagate_single_final(current, dt, chunk))
        states.append(current.copy())
    return np.array(states), chunk * dt


def _split_at_wraps(
    lon: np.ndarray, lat: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split track into continuous segments at 180° longitude wrap-arounds."""
    breaks = np.where(np.abs(np.diff(lon)) > 180.0)[0] + 1
    idx = np.concatenate([[0], breaks, [len(lon)]])
    return [(lon[idx[i] : idx[i + 1]], lat[idx[i] : idx[i + 1]]) for i in range(len(idx) - 1)]


def _format_duration(seconds: float) -> str:
    """Seconds after epoch as a human-readable offset, e.g. '2 d 03:14'."""
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    return f"{days} d {hours:02d}:{minutes:02d}" if days else f"{hours:02d}:{minutes:02d}"


def _conjunction_detail(plan: MissionPlan) -> list[str]:
    """Report the closest approach, and list any threshold violations.

    A bare 'SAFE' says nothing about margin — 6 km clear and 600 km clear read
    identically. The optimizer already computes this detail for its final check,
    so surfacing it costs nothing.
    """
    result = plan.result
    lines: list[str] = []

    nearest = result.nearest
    if nearest is not None:
        lines += [
            "",
            f"  Closest approach:  {nearest.min_separation_m / 1000.0:>10.2f} km",
            f"    object:          {nearest.name} (NORAD {nearest.norad_id})",
            f"    at:              T+{_format_duration(nearest.time_of_closest_approach_s)}",
        ]
    elif result.catalog_screened == 0:
        lines += ["", "  No catalog objects crossed the mission altitude band."]

    if result.conjunctions:
        lines += ["", f"  Objects within threshold ({len(result.conjunctions)}):"]
        for c in result.conjunctions[:10]:
            lines.append(
                f"    {c.min_separation_m / 1000.0:>8.2f} km  "
                f"T+{_format_duration(c.time_of_closest_approach_s):<10}  "
                f"{c.name} (NORAD {c.norad_id})"
            )
        if len(result.conjunctions) > 10:
            lines.append(f"    … and {len(result.conjunctions) - 10} more")

    return lines


# ---------------------------------------------------------------------------
# Public: text report
# ---------------------------------------------------------------------------


def format_report(plan: MissionPlan, epoch: datetime | None = None) -> str:
    """Format a MissionPlan as a multi-line human-readable mission report.

    Args:
        plan:  Completed MissionPlan from plan_mission().
        epoch: Mission start time (for header timestamp). Defaults to now.

    Returns:
        A formatted string suitable for printing or writing to a .txt file.
    """
    if epoch is None:
        epoch = datetime.now(timezone.utc)

    sma, inc, ecc, raan, argp = plan.result.elements
    alt_peri = (sma * (1 - ecc) - R_EARTH) / 1000.0   # km
    alt_apo  = (sma * (1 + ecc) - R_EARTH) / 1000.0   # km
    period   = _period_min(sma)
    orbit_label = plan.intent["orbit_type"].replace("_", " ").title()
    safety_str = "SAFE  ✓" if plan.result.safe else "CONJUNCTION RISK  ✗"

    lines: list[str] = [
        "=" * _WIDTH,
        "  ORBIT MANIFEST — Mission Report".center(_WIDTH),
        "=" * _WIDTH,
        "",
        f"  Mission:    {plan.description}",
        f"  Orbit type: {orbit_label}",
        f"  Generated:  {epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"  Rationale:  {plan.intent.get('rationale', '')}",
        "",
        _BAR,
        "  OPTIMAL ORBITAL ELEMENTS",
        _BAR,
        f"  Semi-major axis:   {sma / 1000.0:>10.2f} km",
        f"  Altitude (peri):   {alt_peri:>10.2f} km",
        f"  Altitude (apo):    {alt_apo:>10.2f} km",
        f"  Inclination:       {math.degrees(inc):>10.4f} °",
        f"  Eccentricity:      {ecc:>10.6f}",
        f"  Period:            {period:>10.2f} min",
        f"  RAAN:              {math.degrees(raan):>10.2f} °",
        f"  Arg. of perigee:   {math.degrees(argp):>10.2f} °",
        "",
        _BAR,
        "  CONJUNCTION SAFETY",
        _BAR,
        f"  Catalog size:      {plan.catalog_size:>10,} objects",
        # Only objects whose orbit crosses the mission shell are ever propagated;
        # quoting the full catalog alone overstates what was actually checked.
        f"  Screened:          {plan.result.catalog_screened:>10,} objects in the altitude band",
        f"  Safety status:     {safety_str}",
        *_conjunction_detail(plan),
        "",
        _BAR,
        "  OPTIMIZER STATISTICS",
        _BAR,
        f"  Converged:         {'Yes' if plan.result.success else 'No':>10}",
        f"  Objective (ecc):   {plan.result.objective:>10.6f}",
        f"  Generations:       {plan.result.n_generations:>10,}",
        f"  Conjunction calls: {plan.result.conjunctions_checked:>10,}",
        f"  Wall-clock time:   {plan.result.elapsed_s:>10.1f} s",
        f"  Message:           {plan.result.message}",
        "",
        "=" * _WIDTH,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public: ground-track plot
# ---------------------------------------------------------------------------


def plot_ground_track(
    plan: MissionPlan,
    epoch: datetime,
    dt: float = 60.0,
    n_samples: int = 500,
    save_path: str | Path | None = None,
):
    """Plot the ground track of the optimal orbit over the mission duration.

    Propagates the best orbit from the optimizer using the C++ integrator,
    converts each position from ECI to geographic lat/lon, and plots the result.
    Longitude wrap-arounds (±180°) are handled by splitting into segments.

    Args:
        plan:       Completed MissionPlan from plan_mission().
        epoch:      Mission start time (UTC) — required for GMST calculation.
        dt:         Integrator time step in seconds (should match what was used in plan_mission).
        n_samples:  Number of trajectory samples to plot (more = smoother track).
        save_path:  If given, save the figure to this path (PNG/PDF/SVG).

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    duration_s = plan.intent["duration_days"] * 86400.0
    total_steps = max(1, int(duration_s / dt))

    # Propagate the best orbit. Take sample_dt from the propagation itself —
    # deriving it as duration_s / n_samples disagrees with the whole number of
    # integrator steps actually taken per sample, which skews every longitude.
    states, sample_dt = _sample_trajectory(plan.result.state, dt, total_steps, n_samples)
    positions = states[:, :3]                         # (N, 3) ECI metres

    lat, lon = _eci_to_latlon(positions, epoch, sample_dt)
    segments = _split_at_wraps(lon, lat)

    orbit_label = plan.intent["orbit_type"].replace("_", " ").title()
    sma, inc, ecc, _, _ = plan.result.elements
    alt_km = (sma - R_EARTH) / 1000.0
    subtitle = (
        f"{orbit_label}  |  alt ≈ {alt_km:.0f} km  |  "
        f"inc = {math.degrees(inc):.2f}°  |  "
        f"ecc = {ecc:.5f}"
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    # Graticule — stand-in for coastlines (no mapping library required)
    for lat_line in range(-60, 90, 30):
        ax.axhline(lat_line, color="lightgrey", lw=0.5, zorder=0)
    for lon_line in range(-180, 181, 30):
        ax.axvline(lon_line, color="lightgrey", lw=0.5, zorder=0)
    ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=0, label="Equator")

    # Ground track — plot each segment to avoid lines jumping across the map
    for i, (seg_lon, seg_lat) in enumerate(segments):
        ax.plot(
            seg_lon, seg_lat,
            color="royalblue", lw=0.9, alpha=0.75,
            label="Ground track" if i == 0 else None,
        )

    # Start / end markers
    ax.plot(lon[0],  lat[0],  "g^", ms=9, zorder=5, label="Start")
    ax.plot(lon[-1], lat[-1], "rs", ms=9, zorder=5, label="End")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)", fontsize=11)
    ax.set_ylabel("Latitude (°)", fontsize=11)
    ax.set_title(f"Ground Track — {plan.description}\n{subtitle}", fontsize=11)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
