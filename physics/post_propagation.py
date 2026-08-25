"""
Post-propagation integrity checks: energy and angular momentum conservation.

These functions are kept separate from the pre-propagation screen
(solver/physics_agreement.py) because they serve a different purpose:
the pre-propagation check is a cheap O(1) gate used in the production
optimizer; the post-propagation check is a physics accuracy evaluator
used in experiments comparing propagator variants.

Constants mirror those in solver/physics_agreement.py and
integrator/integrator.h. If you change them, change them everywhere.
"""
from __future__ import annotations

import numpy as np

from physics.constants import J2, MU_EARTH, R_EARTH


def specific_energy(s: np.ndarray) -> float:
    """Specific mechanical energy including the J2 potential correction.

    E = v²/2 - μ/r + μ·J₂·Re²/(2r³)·(3z²/r² − 1)

    Identical to the formula in tests/test_integrator.py::test_energy_conservation.
    J2 is a conservative potential, so this quantity is an invariant of the
    2-body + J2 model and should be conserved across propagation.

    Args:
        s: Cartesian ECI state, shape (6,) — [x, y, z, vx, vy, vz] in m and m/s.

    Returns:
        Specific mechanical energy in J/kg.
    """
    x, y, z = s[0], s[1], s[2]
    v2 = s[3]**2 + s[4]**2 + s[5]**2
    r  = np.sqrt(x**2 + y**2 + z**2)
    e_2body = 0.5 * v2 - MU_EARTH / r
    e_j2    = MU_EARTH * J2 * R_EARTH**2 / (2.0 * r**3) * (3.0 * z**2 / r**2 - 1.0)
    return e_2body + e_j2


def check_post_propagation(
    s0: np.ndarray,
    sf: np.ndarray,
    energy_tol: float = 10.0,
    momentum_tol_rel: float = 1e-4,
) -> tuple[bool, list[str]]:
    """Verify conservation-law integrity after propagation.

    Compares the initial state s0 against the final state sf. Both must be
    Cartesian ECI in SI units: [x, y, z, vx, vy, vz] in metres and m/s.

    Four checks are performed:

    1. Circular velocity plausibility on s0 — |v₀| must lie within
       [0.5, 1.5] × v_circ at the initial position. Catches unit errors
       or degenerate keplerian_to_cartesian conversions. This check fires
       on a bad initial state, not on propagation quality — a retry with
       smaller dt cannot fix it.

    2. Earth clearance of the final state — r_f > R_EARTH + 80 km.
       Guards against numerical blow-up in the absence of drag.

    3. Specific mechanical energy (2-body + J2 correction) is conserved
       to within energy_tol J/kg. J2 is a conservative potential so
       total energy is an invariant of the model.

    4. Specific angular momentum magnitude is conserved to within
       momentum_tol_rel (relative). J2 is axially symmetric, so
       |h| = √(μa(1−e²)) is preserved at first order.

    Args:
        s0:               Initial Cartesian ECI state, shape (6,).
        sf:               Final Cartesian ECI state after propagation, shape (6,).
        energy_tol:       Max allowed energy drift in J/kg (default 10.0).
        momentum_tol_rel: Max allowed relative angular momentum drift (default 1e-4).

    Returns:
        (ok, reasons) — ok is True iff all checks pass; reasons is a list of
        human-readable failure descriptions (empty when ok=True).
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
    E0 = specific_energy(s0)
    Ef = specific_energy(sf)
    dE = abs(Ef - E0)
    if dE > energy_tol:
        reasons.append(
            f"energy drift {dE:.2f} J/kg > {energy_tol} J/kg threshold"
        )

    # 4. Specific angular momentum magnitude conservation
    h0 = np.linalg.norm(np.cross(s0[:3], s0[3:]))
    hf = np.linalg.norm(np.cross(sf[:3], sf[3:]))
    if h0 > 0.0:
        dh_rel = abs(hf - h0) / h0
        if dh_rel > momentum_tol_rel:
            reasons.append(
                f"angular momentum drift {dh_rel:.2e} > "
                f"{momentum_tol_rel:.1e} threshold"
            )

    return (not reasons), reasons
