"""
Pre-propagation orbit screen for the orbital optimizer.

Checks Keplerian parameters before keplerian_to_cartesian or RK4 propagation,
catching physically degenerate candidates in O(1) arithmetic so they never
enter the OpenMP kernel.

Post-propagation checks (energy/momentum conservation) live in
physics/post_propagation.py and are used by the accuracy experiment, not the
production optimizer.
"""
from __future__ import annotations

import numpy as np

from physics.constants import MU_EARTH, R_EARTH


class OrbitScreen:
    """Pre-propagation rejection gate for the orbital optimizer.

    Instantiate once before the optimization loop and call
    check_pre_propagation on every candidate before the batch propagation.
    All arithmetic is O(1) — no integrator calls.

    Args:
        min_perigee_alt_m: Minimum allowed perigee altitude above surface (m).
                           Default 150 km — below this, atmospheric drag is
                           significant and the no-drag RK4 model is invalid.
        max_apogee_m:      Maximum allowed apogee radius from Earth centre (m).
                           Default R_EARTH + 2500 km — above this the orbit
                           exits the LEO conjunction shell.
    """

    def __init__(
        self,
        min_perigee_alt_m: float = 150_000.0,
        max_apogee_m: float = R_EARTH + 2_500_000.0,
    ) -> None:
        self.min_perigee_alt_m = min_perigee_alt_m
        self.max_apogee_m      = max_apogee_m

    def check_pre_propagation(
        self,
        sma: float,
        inc: float,
        ecc: float,
        raan: float,
        argp: float,
        n_steps: int,
        dt: float,
    ) -> tuple[bool, list[str]]:
        """Screen a Keplerian candidate before keplerian_to_cartesian or RK4.

        All arithmetic is O(1) with no propagation. Runs on every candidate
        in the population before the batch propagation call so that pathological
        orbits never enter the OpenMP kernel.

        Args:
            sma:     Semi-major axis (m)
            inc:     Inclination (radians) — accepted but not currently checked
            ecc:     Eccentricity (dimensionless)
            raan:    RAAN (radians) — accepted but not currently checked
            argp:    Argument of perigee (radians) — accepted but not currently checked
            n_steps: Number of RK4 steps in the mission propagation
            dt:      RK4 time step (s)

        Returns:
            (ok, reasons) — ok is True iff all checks pass; reasons is a list
            of human-readable failure descriptions (empty when ok=True).
        """
        r_perigee = sma * (1.0 - ecc)
        r_apogee  = sma * (1.0 + ecc)
        T_orbit   = 2.0 * np.pi * np.sqrt(sma**3 / MU_EARTH)
        t_total   = n_steps * dt

        reasons: list[str] = []

        if r_perigee < R_EARTH + self.min_perigee_alt_m:
            alt_km   = (r_perigee - R_EARTH) / 1e3
            floor_km = self.min_perigee_alt_m / 1e3
            reasons.append(
                f"perigee {alt_km:.1f} km below {floor_km:.0f} km floor "
                "(drag significant; no-drag RK4 model invalid)"
            )

        if r_apogee > self.max_apogee_m:
            alt_km  = (r_apogee - R_EARTH) / 1e3
            ceil_km = (self.max_apogee_m - R_EARTH) / 1e3
            reasons.append(
                f"apogee {alt_km:.1f} km above {ceil_km:.0f} km ceiling "
                "(exits LEO conjunction shell)"
            )

        if t_total < T_orbit:
            reasons.append(
                f"integration window {t_total / 60:.1f} min < orbital period "
                f"{T_orbit / 60:.1f} min "
                "(mission objective at final state is meaningless)"
            )

        return (not reasons), reasons
