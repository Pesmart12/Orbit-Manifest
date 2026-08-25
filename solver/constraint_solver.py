"""
Constraint solver: mission goal → orbital element bounds.

Each public function accepts human-readable mission parameters and returns
an OrbitalBounds that defines the scipy search space for the optimizer.
No NL parsing happens here — that's the agent layer's job.  These functions
encode the physics and domain knowledge that maps a goal type to valid
orbital element ranges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Canonical constants live in physics/constants.py. Re-exported here because
# callers and tests have long imported them from this module.
from physics.constants import J2, MU_EARTH, R_EARTH

# Earth's mean angular velocity around the Sun.
# A sun-synchronous orbit's nodal drift must equal this value so the
# orbital plane rotates in step with the Sun and keeps illumination constant.
# Derived from: 2π radians / 365.25 days / 86400 s/day
OMEGA_SUN = 2 * np.pi / (365.25 * 86400)  # rad/s ≈ 1.9910e-7

# Practical LEO altitude ceiling.  Above ~2000 km you enter the inner Van Allen
# radiation belt — sustained operations there damage spacecraft electronics.
# The optimizer must never be given a search space that extends above this.
LEO_MAX_ALT_KM = 2000.0


@dataclass
class OrbitalBounds:
    """Search-space bounds for one orbital element set, all in SI units.

    The optimizer draws candidate orbits uniformly from within these bounds
    and passes them to the C++ integrator + conjunction checker.

    Units:
        sma   — meters (semi-major axis)
        inc   — radians
        ecc   — dimensionless [0, 1)
        raan  — radians [0, 2π]
        argp  — radians [0, 2π]
    """
    sma_min: float
    sma_max: float
    inc_min: float
    inc_max: float
    ecc_min: float = 0.0
    ecc_max: float = 0.01   # near-circular by default
    raan_min: float = 0.0
    raan_max: float = field(default_factory=lambda: 2 * np.pi)
    argp_min: float = 0.0
    argp_max: float = field(default_factory=lambda: 2 * np.pi)

    def __post_init__(self) -> None:
        # Guard against the most common unit mistake: passing altitude in km
        # instead of sma in m.  An sma below R_EARTH means the orbit is inside
        # the planet, which is always a caller error.
        if self.sma_min < R_EARTH:
            raise ValueError(
                f"sma_min {self.sma_min:.0f} m is below Earth's surface. "
                "sma must be in metres, not km or altitude."
            )
        if self.sma_min > self.sma_max:
            raise ValueError("sma_min must be <= sma_max")
        # Inclination is defined on [0, π] in the standard Keplerian element set.
        # Values outside this range indicate degrees were passed instead of radians.
        if self.inc_min < 0 or self.inc_max > np.pi:
            raise ValueError("Inclination must be in [0, π] radians")
        # Eccentricity = 1 is a parabolic escape trajectory; ≥ 1 is hyperbolic.
        # Neither is a valid closed orbit for a LEO mission.
        if self.ecc_min < 0 or self.ecc_max >= 1.0:
            raise ValueError("Eccentricity must be in [0, 1)")

    # ------------------------------------------------------------------
    # Convenience accessors used by the optimizer
    # ------------------------------------------------------------------

    @property
    def sma_center(self) -> float:
        """Midpoint of the sma band — used as the optimizer's initial guess."""
        return (self.sma_min + self.sma_max) / 2

    @property
    def inc_center(self) -> float:
        """Midpoint of the inclination band — used as the optimizer's initial guess."""
        return (self.inc_min + self.inc_max) / 2

    def as_scipy_bounds(self) -> list[tuple[float, float]]:
        """Return bounds as a list of (lo, hi) pairs in the element order
        [sma, inc, ecc, raan, argp] expected by scipy.optimize.

        The optimizer unpacks this list directly into scipy.optimize.Bounds,
        so the order here must match the parameter vector order in optimizer.py.
        """
        return [
            (self.sma_min,  self.sma_max),
            (self.inc_min,  self.inc_max),
            (self.ecc_min,  self.ecc_max),
            (self.raan_min, self.raan_max),
            (self.argp_min, self.argp_max),
        ]


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _sma_from_altitude_km(alt_km: float) -> float:
    """Convert altitude above the surface (km) to semi-major axis (m).

    All internal calculations and the C++ integrator work in SI (metres),
    but human-readable goal descriptions use km — this is the conversion point.
    """
    return R_EARTH + alt_km * 1000.0


def _sun_sync_inclination(sma_m: float, ecc: float = 0.0) -> float:
    """Compute the inclination (radians) that produces a sun-synchronous orbit.

    A sun-synchronous orbit drifts eastward at exactly OMEGA_SUN rad/s so
    that the orbital plane always faces the Sun at the same local solar time.
    This is achieved by exploiting the J2 nodal precession:

        dΩ/dt = -3/2 * n * J2 * (R_e/a)^2 * cos(i) / (1-e²)²

    Setting dΩ/dt = +OMEGA_SUN and solving for cos(i):

        cos(i) = -OMEGA_SUN * (1-e²)² / (3/2 * n * J2 * (R_e/a)²)

    The result is always slightly above 90° (retrograde) because OMEGA_SUN
    is positive but the unperturbed drift for a prograde orbit is negative —
    we need the orbit to be retrograde enough to flip the sign.

    Raises ValueError if no solution exists (altitude too high — the J2
    effect weakens with altitude and eventually can't match OMEGA_SUN).
    """
    n = np.sqrt(MU_EARTH / sma_m**3)   # mean motion, rad/s
    # Denominator bundles the altitude- and eccentricity-dependent terms.
    factor = (3 / 2) * n * J2 * (R_EARTH / sma_m) ** 2 / (1 - ecc**2) ** 2
    cos_i = -OMEGA_SUN / factor
    if abs(cos_i) > 1.0:
        # arccos is only defined on [-1, 1]; beyond that there is no real
        # inclination that satisfies the SSO condition at this altitude.
        raise ValueError(
            f"No sun-synchronous solution at sma={sma_m:.0f} m — "
            "altitude may be too high."
        )
    return np.arccos(cos_i)


# ---------------------------------------------------------------------------
# Public goal constructors
# ---------------------------------------------------------------------------

def sun_synchronous(
    altitude_km: float,
    tolerance_km: float = 20.0,
    ecc_max: float = 0.001,
) -> OrbitalBounds:
    """Sun-synchronous orbit at a given altitude.

    Inclination is tightly constrained to the value that makes the nodal
    drift rate equal to OMEGA_SUN (≈ 0.9856°/day).  A ±0.1° window around
    the exact solution is allowed so the optimizer has a small search band
    rather than a single fixed point — necessary because scipy needs a
    non-degenerate interval, and real satellites tolerate a tiny deviation.

    Eccentricity is kept very small: SSO missions are nearly circular so
    the local solar time at the ascending node stays stable over the mission.

    Args:
        altitude_km:  target altitude above the surface in km
        tolerance_km: ± altitude band around the target, in km
        ecc_max:      maximum eccentricity (keep very small — SSO assumes circular)
    """
    sma_center = _sma_from_altitude_km(altitude_km)
    sma_min    = _sma_from_altitude_km(altitude_km - tolerance_km)
    sma_max    = _sma_from_altitude_km(altitude_km + tolerance_km)

    # Compute SSO inclination at the band centre; the ±0.1° window is the
    # tightest interval that still gives scipy a non-zero search space.
    inc_exact  = _sun_sync_inclination(sma_center)
    inc_window = np.deg2rad(0.1)

    return OrbitalBounds(
        sma_min=sma_min,
        sma_max=sma_max,
        inc_min=inc_exact - inc_window,
        inc_max=inc_exact + inc_window,
        ecc_min=0.0,
        ecc_max=ecc_max,
    )


def low_earth_orbit(
    altitude_min_km: float,
    altitude_max_km: float,
    inc_min_deg: float,
    inc_max_deg: float,
    ecc_max: float = 0.01,
) -> OrbitalBounds:
    """Generic LEO with caller-specified altitude and inclination bands.

    Use this when the mission doesn't require a specific orbit type — e.g.
    "between 400 and 600 km, mid-inclination for regional coverage."
    RAAN and argument of perigee are left fully unconstrained (0–2π) so
    the optimizer can find the safest plane with respect to conjunctions.
    """
    if altitude_max_km > LEO_MAX_ALT_KM:
        # Enforced here rather than in OrbitalBounds because the LEO ceiling
        # is a mission policy limit, not a physics impossibility.
        raise ValueError(
            f"altitude_max_km {altitude_max_km} exceeds LEO ceiling {LEO_MAX_ALT_KM} km"
        )
    return OrbitalBounds(
        sma_min=_sma_from_altitude_km(altitude_min_km),
        sma_max=_sma_from_altitude_km(altitude_max_km),
        inc_min=np.deg2rad(inc_min_deg),
        inc_max=np.deg2rad(inc_max_deg),
        ecc_min=0.0,
        ecc_max=ecc_max,
    )


def polar(
    altitude_km: float,
    tolerance_km: float = 25.0,
    inc_window_deg: float = 2.0,
) -> OrbitalBounds:
    """Near-polar orbit — useful for global coverage or Earth observation.

    Centred on 90° inclination with a configurable window.  Slightly
    retrograde inclinations (>90°) are allowed since many Earth-observation
    missions intentionally use a few degrees past 90° to get a near-SSO
    ground track without fully locking the inclination to the SSO formula.

    Use sun_synchronous() instead if you need the exact SSO drift rate.
    """
    return OrbitalBounds(
        sma_min=_sma_from_altitude_km(altitude_km - tolerance_km),
        sma_max=_sma_from_altitude_km(altitude_km + tolerance_km),
        inc_min=np.deg2rad(90.0 - inc_window_deg),
        inc_max=np.deg2rad(90.0 + inc_window_deg),
        ecc_min=0.0,
        ecc_max=0.01,
    )


def iss_rendezvous(altitude_tolerance_km: float = 15.0) -> OrbitalBounds:
    """Orbit compatible with ISS rendezvous.

    The ISS is at ~415 km, 51.6° inclination.  Rendezvous requires the
    chaser to be in (or very close to) the target's orbital plane — a
    plane-change manoeuvre at LEO speeds costs hundreds of m/s of delta-v,
    so the inclination window is kept at ±0.05° to make that effectively zero.

    Altitude tolerance is wider because phasing (catching up along-track)
    is cheap: you adjust altitude slightly to walk the phase angle and then
    return to the target altitude for the final approach.
    """
    ISS_ALTITUDE_KM  = 415.0
    ISS_INC_DEG      = 51.6
    INC_WINDOW_DEG   = 0.05   # plane-change delta-v is ~60 m/s per degree — keep tight

    return OrbitalBounds(
        sma_min=_sma_from_altitude_km(ISS_ALTITUDE_KM - altitude_tolerance_km),
        sma_max=_sma_from_altitude_km(ISS_ALTITUDE_KM + altitude_tolerance_km),
        inc_min=np.deg2rad(ISS_INC_DEG - INC_WINDOW_DEG),
        inc_max=np.deg2rad(ISS_INC_DEG + INC_WINDOW_DEG),
        ecc_min=0.0,
        ecc_max=0.005,  # final approach requires near-circular — no large eccentricity
    )


def custom(
    sma_min_m: float,
    sma_max_m: float,
    inc_min_deg: float,
    inc_max_deg: float,
    ecc_min: float = 0.0,
    ecc_max: float = 0.01,
    raan_min_deg: float = 0.0,
    raan_max_deg: float = 360.0,
    argp_min_deg: float = 0.0,
    argp_max_deg: float = 360.0,
) -> OrbitalBounds:
    """Fully manual bounds — for cases not covered by the named goal types.

    The agent layer uses this when the NL parser produces constraints that
    don't map cleanly to one of the named goal types above.  It's also useful
    for unit tests that need precise control over every element band.

    Convention:
        sma   in metres (not km, not altitude)
        angles in degrees (converted to radians internally)
    Leaving raan/argp at their defaults (0–360°) means the optimizer is free
    to pick any orbital plane orientation, which is usually what you want
    unless the mission has a specific ground-track requirement.
    """
    return OrbitalBounds(
        sma_min=sma_min_m,
        sma_max=sma_max_m,
        inc_min=np.deg2rad(inc_min_deg),
        inc_max=np.deg2rad(inc_max_deg),
        ecc_min=ecc_min,
        ecc_max=ecc_max,
        raan_min=np.deg2rad(raan_min_deg),
        raan_max=np.deg2rad(raan_max_deg),
        argp_min=np.deg2rad(argp_min_deg),
        argp_max=np.deg2rad(argp_max_deg),
    )
