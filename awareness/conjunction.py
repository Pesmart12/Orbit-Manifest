from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, SatrecArray, jday

# C++ RK4 integrator — propagates the *mission* orbit at high fidelity.
# The catalog objects are propagated with SGP4 (see _download) because running
# 20 k objects through RK4 every optimizer call would be prohibitively slow.
import orbit_integrator as oi


@dataclass
class ConjunctionResult:
    norad_id: str          # NORAD catalog number as a string
    name: str              # human-readable satellite name from the TLE header
    min_separation_m: float              # closest approach distance in meters
    time_of_closest_approach_s: float   # seconds after epoch when TCA occurs


def check_conjunctions(
    mission_state: np.ndarray,
    epoch: datetime,
    duration_s: float,
    catalog: list[tuple[str, str, str]],
    dt: float = 30.0,
    threshold_m: float = 5000.0,
) -> list[ConjunctionResult]:
    """
    Check mission orbit against TLE catalog for conjunctions.

    Args:
        mission_state: shape (6,) ECI state [x,y,z,vx,vy,vz] in m / m/s
        epoch:         UTC datetime corresponding to mission_state t=0
        duration_s:    propagation window in seconds
        catalog:       [(name, line1, line2), ...] from tle_fetcher
        dt:            time step in seconds (30 s sufficient for LEO screening)
        threshold_m:   conjunction distance threshold in meters

    Returns:
        List of ConjunctionResult below threshold, sorted by min_separation_m.
    """
    if not catalog:
        return []

    # Ensure epoch is timezone-aware so timedelta arithmetic stays unambiguous.
    epoch = epoch.replace(tzinfo=timezone.utc) if epoch.tzinfo is None else epoch

    names    = [t[0] for t in catalog]
    # Parse every TLE into a Satrec object (SGP4 internal state).
    satrecs  = [Satrec.twoline2rv(t[1], t[2]) for t in catalog]
    norad_ids = [str(s.satnum) for s in satrecs]
    # SatrecArray lets sgp4 propagate the entire catalog in one vectorised call
    # instead of looping over individual Satrec objects — critical for 20 k objects.
    sats     = SatrecArray(satrecs)
    N        = len(satrecs)

    n_steps = int(duration_s / dt)

    # Propagate the mission orbit once up front with the C++ RK4 integrator.
    # Shape: (n_steps+1, 6) — we only need the position columns [:3].
    mission_traj = oi.propagate_single(mission_state, dt, n_steps)
    mission_pos  = np.asarray(mission_traj)[:, :3]  # (n_steps+1, 3) meters

    # Track the minimum separation and its time for every catalog object.
    min_sep = np.full(N, np.inf)
    tca     = np.zeros(N)

    for step in range(n_steps + 1):
        t_epoch = epoch + timedelta(seconds=step * dt)
        jd, fr  = _to_jday(t_epoch)

        # Broadcast scalar Julian date/fraction across all N satellites so
        # sgp4() can evaluate the entire catalog in a single vectorised call.
        jd_arr = np.full(N, jd)
        fr_arr = np.full(N, fr)
        # e: (N,) error codes — nonzero means SGP4 failed for that object
        # r: (N,3) ECI positions in km
        e, r, _ = sats.sgp4(jd_arr, fr_arr)

        valid = (e == 0)  # mask out objects with SGP4 errors (decayed, bad TLE, etc.)
        r_m   = r * 1000.0  # km → meters to match mission_pos units

        diff = r_m - mission_pos[step]       # (N, 3) displacement vectors
        sep  = np.linalg.norm(diff, axis=1)  # (N,) scalar distances

        # Invalidate failed SGP4 results so they never satisfy the threshold check.
        sep[~valid] = np.inf

        # Update min_sep and tca only where this step beat the previous minimum.
        improved          = sep < min_sep
        min_sep[improved] = sep[improved]
        tca[improved]     = step * dt

    # Collect only the objects that came within threshold and sort closest-first.
    results = [
        ConjunctionResult(
            norad_id=norad_ids[i],
            name=names[i],
            min_separation_m=float(min_sep[i]),
            time_of_closest_approach_s=float(tca[i]),
        )
        for i in range(N)
        if min_sep[i] < threshold_m
    ]
    results.sort(key=lambda r: r.min_separation_m)
    return results


def _to_jday(dt_utc: datetime) -> tuple[float, float]:
    """Convert a UTC datetime to the (julian_date, fraction) pair sgp4 expects.

    sgp4 splits the Julian date into an integer day (jd) and a fractional day
    (fr) to preserve floating-point precision near midnight.
    """
    return jday(
        dt_utc.year, dt_utc.month, dt_utc.day,
        dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond * 1e-6,
    )
