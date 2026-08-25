"""Canonical physical constants for the Python side of Orbit Manifest.

Every Python module takes its constants from here. Before this existed the same
six values were re-declared in six files, each carrying a comment instructing the
reader to keep the copies in sync by hand — which is a defect waiting to happen,
not a safeguard.

`integrator/integrator.h` necessarily keeps its own copy: the C++ translation
unit cannot import Python. That one duplication is enforced rather than trusted —
`tests/test_constants.py` parses the header and asserts the values agree, so the
two sources cannot drift silently.

This module deliberately has no imports. Anything in the project can depend on it
without risking an import cycle.
"""
from __future__ import annotations

MU_EARTH    = 3.986004418e14   # m^3/s^2 — Earth's gravitational parameter (GM)
R_EARTH     = 6.3781e6         # m       — mean equatorial radius
J2          = 1.08263e-3       # —       — second zonal harmonic (oblateness), EGM96
OMEGA_EARTH = 7.2921150e-5     # rad/s   — Earth's sidereal rotation rate

__all__ = ["MU_EARTH", "R_EARTH", "J2", "OMEGA_EARTH"]
