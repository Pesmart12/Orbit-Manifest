"""
Physics agreement layer: validate orbital states before and after RK4 propagation.

This module provides a lightweight rejection gate that the optimizer uses to
catch two distinct failure modes before they silently poison the fitness landscape:

1. Pre-propagation pathology — Keplerian parameters that satisfy the numeric
   search bounds but encode a physically degenerate orbit (perigee inside the
   atmosphere, near-escape apogee, period longer than the integration window).
   These are caught in O(1) arithmetic before any propagation.

2. Post-propagation integrity — the RK4 propagation should conserve specific
   mechanical energy and angular momentum magnitude (both are invariants of the
   conservative 2-body + J2 potential). A significant violation signals
   numerical instability in the trajectory and the result should be discarded.

Neither check is a physics model — they are guard rails. The thresholds are
intentionally loose relative to the test suite's validation tolerances so that
the layer only rejects genuinely pathological candidates, not marginally-off ones.
"""
from __future__ import annotations

import numpy as np

# Mirror constants from integrator/integrator.h and solver/constraint_solver.py.
# If you change these, change them in both other locations too.
MU_EARTH = 3.986004418e14   # m^3/s^2  — Earth's gravitational parameter
R_EARTH  = 6.3781e6         # m        — mean equatorial radius
J2       = 1.08263e-3       # dimensionless — Earth's oblateness coefficient


class PhysicsAgreementLayer:
    """Lightweight physics rejection gate for the orbital optimizer.

    Instantiate once before the optimization loop and pass it to the batch
    fitness function. Both public methods are pure NumPy arithmetic — no
    propagation is performed.

    Args:
        min_perigee_alt_m:  Minimum allowed perigee altitude above surface (m).
                            Default 150 km — below this, atmospheric drag is
                            significant and the no-drag RK4 model is invalid.
        max_apogee_m:       Maximum allowed apogee radius from Earth centre (m).
                            Default R_EARTH + 2500 km — above this the orbit
                            exits the LEO conjunction shell.
        energy_tol:         Maximum allowed specific mechanical energy drift
                            (J/kg) between initial and final states.
                            Default 10 J/kg — 10× the test-suite validation
                            limit, calibrated for a 7-day mission horizon.
        momentum_tol_rel:   Maximum allowed relative drift in specific angular
                            momentum magnitude |Δh|/h₀.
                            Default 1e-4 — J2 is axially symmetric so |h| is
                            a first-order invariant; drift > 0.01% indicates
                            numerical instability.
    """

    def __init__(
        self,
        min_perigee_alt_m: float = 150_000.0,
        max_apogee_m: float = R_EARTH + 2_500_000.0,
        energy_tol: float = 10.0,
        momentum_tol_rel: float = 1e-4,
    ) -> None:
        self.min_perigee_alt_m = min_perigee_alt_m
        self.max_apogee_m      = max_apogee_m
        self.energy_tol        = energy_tol
        self.momentum_tol_rel  = momentum_tol_rel

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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

        All arithmetic is O(1) with no propagation. This check runs on every
        candidate in the population before the batch propagation call so that
        pathological orbits never enter the OpenMP kernel.

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

    def check_post_propagation(
        self,
        s0: np.ndarray,
        sf: np.ndarray,
    ) -> tuple[bool, list[str]]:
        """Verify conservation-law integrity after RK4 propagation.

        Compares the initial state s0 against the final state sf returned by
        propagate_batch_final. Both must be Cartesian ECI in SI units:
        [x, y, z, vx, vy, vz] in metres and m/s.

        Four checks are performed:

        1. Circular velocity plausibility on s0 — |v₀| must lie within
           [0.5, 1.5] × v_circ at the initial position.  Catches unit errors
           or degenerate keplerian_to_cartesian conversions before the
           fitness score is consumed by the optimizer.

        2. Earth clearance of the final state — r_f > R_EARTH + 80 km.
           Guards against numerical blow-up in the absence of drag.

        3. Specific mechanical energy (2-body + J2 correction) is conserved
           to within energy_tol J/kg.  J2 is a conservative potential so
           total energy is an invariant of the model.

        4. Specific angular momentum magnitude is conserved to within
           momentum_tol_rel (relative).  J2 is axially symmetric, so
           |h| = √(μa(1−e²)) is preserved at first order.

        Args:
            s0: Initial Cartesian ECI state, shape (6,)
            sf: Final Cartesian ECI state after propagation, shape (6,)

        Returns:
            (ok, reasons) — ok is True iff all checks pass.
        """
        reasons: list[str] = []

        r0 = np.linalg.norm(s0[:3])
        rf = np.linalg.norm(sf[:3])
        v0 = np.linalg.norm(s0[3:])

        # 1. Circular velocity plausibility (checks s0, the converted state)
        v_circ = np.sqrt(MU_EARTH / r0)
        ratio  = v0 / v_circ
        if not (0.5 <= ratio <= 1.5):
            reasons.append(
                f"|v₀|/v_circ = {ratio:.3f} outside [0.5, 1.5] "
                "(likely unit error or degenerate keplerian_to_cartesian output)"
            )

        # 2. Earth clearance of final state
        if rf < R_EARTH + 80e3:
            reasons.append(
                f"final radius {(rf - R_EARTH) / 1e3:.1f} km below 80 km floor "
                "(numerical blow-up with no drag model)"
            )

        # 3. Specific mechanical energy conservation
        E0 = self._specific_energy(s0)
        Ef = self._specific_energy(sf)
        dE = abs(Ef - E0)
        if dE > self.energy_tol:
            reasons.append(
                f"energy drift {dE:.2f} J/kg > {self.energy_tol} J/kg threshold"
            )

        # 4. Specific angular momentum magnitude conservation
        h0 = np.linalg.norm(np.cross(s0[:3], s0[3:]))
        hf = np.linalg.norm(np.cross(sf[:3], sf[3:]))
        if h0 > 0.0:
            dh_rel = abs(hf - h0) / h0
            if dh_rel > self.momentum_tol_rel:
                reasons.append(
                    f"angular momentum drift {dh_rel:.2e} > "
                    f"{self.momentum_tol_rel:.1e} threshold"
                )

        return (not reasons), reasons

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _specific_energy(s: np.ndarray) -> float:
        """Specific mechanical energy including the J2 potential correction.

        E = v²/2 - μ/r + μ·J₂·Re²/(2r³)·(3z²/r² − 1)

        Identical to the formula in tests/test_integrator.py::test_energy_conservation.
        """
        x, y, z = s[0], s[1], s[2]
        v2 = s[3]**2 + s[4]**2 + s[5]**2
        r  = np.sqrt(x**2 + y**2 + z**2)
        e_2body = 0.5 * v2 - MU_EARTH / r
        e_j2    = MU_EARTH * J2 * R_EARTH**2 / (2.0 * r**3) * (3.0 * z**2 / r**2 - 1.0)
        return e_2body + e_j2
