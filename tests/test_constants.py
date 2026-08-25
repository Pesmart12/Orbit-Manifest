"""The Python constants and the C++ header must not drift apart.

`integrator/integrator.h` has to keep its own copy of these values — a C++
translation unit cannot import Python. Every other copy was consolidated into
`physics/constants.py`; this test makes the one remaining duplication an enforced
invariant rather than a comment asking people to remember.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from physics import constants

_HEADER = pathlib.Path(__file__).resolve().parent.parent / "integrator" / "integrator.h"

# constexpr double MU_EARTH    = 3.986004418e14;
_DECL = re.compile(
    r"constexpr\s+double\s+(?P<name>\w+)\s*=\s*(?P<value>[-+0-9.eE]+)\s*;"
)


def _cpp_constants() -> dict[str, float]:
    text = _HEADER.read_text(encoding="utf-8")
    return {m["name"]: float(m["value"]) for m in _DECL.finditer(text)}


def test_header_is_parseable():
    """Guard the guard: if the regex stops matching, the comparison below is vacuous."""
    cpp = _cpp_constants()
    assert cpp, f"no constexpr doubles found in {_HEADER} — has the declaration style changed?"
    assert "MU_EARTH" in cpp


@pytest.mark.parametrize("name", ["MU_EARTH", "R_EARTH", "J2", "OMEGA_EARTH"])
def test_python_matches_cpp(name: str):
    cpp = _cpp_constants()
    assert name in cpp, f"{name} missing from integrator.h"
    py = getattr(constants, name)
    assert py == cpp[name], (
        f"{name} has drifted: physics/constants.py has {py!r}, "
        f"integrator/integrator.h has {cpp[name]!r}"
    )


def test_no_module_redefines_a_constant():
    """Only physics/constants.py may assign these names a literal value.

    Catches a reintroduced local copy, which is how the six duplicates accumulated
    in the first place.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    names = "|".join(["MU_EARTH", "R_EARTH", "J2", "OMEGA_EARTH"])
    literal_assign = re.compile(rf"^_?({names})\s*=\s*[-+0-9.]", re.MULTILINE)

    offenders = []
    for path in root.rglob("*.py"):
        parts = path.parts
        if "tests" in parts or ".git" in parts or "build" in parts:
            continue
        if path.name == "constants.py":
            continue
        if literal_assign.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(root)))

    assert not offenders, (
        "these modules define a physical constant locally instead of importing "
        f"it from physics.constants: {offenders}"
    )
