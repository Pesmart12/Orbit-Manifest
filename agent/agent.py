"""
Agent layer — Phase 5.

Translates a natural language mission description into an optimized orbit
by orchestrating the constraint solver and optimizer.

Flow:
    NL description → Claude API (structured JSON) → intent dict
                   → OrbitalBounds → run_optimizer → MissionPlan
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import anthropic
from dotenv import load_dotenv

from optimizer.optimizer import OptimizationResult, run_optimizer
from solver.constraint_solver import (
    OrbitalBounds,
    R_EARTH,
    custom,
    iss_rendezvous,
    low_earth_orbit,
    polar,
    sun_synchronous,
)

# ---------------------------------------------------------------------------
# JSON schema for structured mission intent extraction
# ---------------------------------------------------------------------------

_ORBIT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "orbit_type": {
            "type": "string",
            "enum": ["sun_synchronous", "low_earth_orbit", "polar", "iss_rendezvous", "custom"],
        },
        # Single target altitude (used by sun_synchronous, polar, or custom with one altitude)
        "altitude_km": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        # Altitude range (used by low_earth_orbit and custom with a band)
        "altitude_min_km": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "altitude_max_km": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        # Inclination range (used by low_earth_orbit and custom)
        "inclination_min_deg": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "inclination_max_deg": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "duration_days": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["orbit_type", "duration_days", "rationale"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """\
You are an orbital mission analyst. Given a natural language mission description, \
extract the orbital parameters the mission requires.

Orbit types:
- sun_synchronous: Earth observation, imaging, or surveillance needing constant solar \
  lighting across passes. Inclination is fixed by physics (J2 nodal precession) — \
  set altitude_km to the target altitude in km, leave inclination fields null.
- low_earth_orbit: General LEO with explicit altitude and inclination ranges. \
  Set altitude_min_km, altitude_max_km, inclination_min_deg, inclination_max_deg.
- polar: Near-90° inclination for global coverage. Set altitude_km.
- iss_rendezvous: Must match ISS orbit (415 km, 51.6° inclination). \
  Leave all altitude and inclination fields null.
- custom: Any orbit with explicit but non-standard constraints. \
  Set altitude range (altitude_min_km / altitude_max_km) or a single altitude_km, \
  and inclination_min_deg / inclination_max_deg.

Defaults if not stated: altitude 500 km, duration 1.0 day.
Inclination hint: ≤30° equatorial/tropical, 45–60° mid-latitude, ≥80° high-latitude or polar.

Always provide a brief rationale explaining your orbit type choice and parameter selection."""

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class MissionPlan:
    """Outcome of a complete plan_mission() call."""

    description: str
    intent: dict                # Claude's structured extraction
    bounds: OrbitalBounds       # scipy search space used by the optimizer
    result: OptimizationResult  # optimizer output
    catalog_size: int           # number of TLE objects used in conjunction checks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_mission(description: str) -> dict:
    """Call Claude to extract structured mission intent from a NL description.

    Returns a dict with keys: orbit_type, duration_days, rationale, and
    whichever altitude/inclination fields are relevant for the orbit type.
    """
    load_dotenv()
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": description}],
        output_config={"format": {"type": "json_schema", "schema": _ORBIT_SCHEMA}},
    )

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _intent_to_bounds(intent: dict) -> OrbitalBounds:
    """Map a structured mission intent dict to OrbitalBounds for the optimizer."""
    orbit_type = intent["orbit_type"]

    if orbit_type == "sun_synchronous":
        return sun_synchronous(intent.get("altitude_km") or 500.0)

    if orbit_type == "polar":
        return polar(intent.get("altitude_km") or 500.0)

    if orbit_type == "iss_rendezvous":
        return iss_rendezvous()

    if orbit_type == "low_earth_orbit":
        return low_earth_orbit(
            altitude_min_km=intent.get("altitude_min_km") or 400.0,
            altitude_max_km=intent.get("altitude_max_km") or 600.0,
            inc_min_deg=intent.get("inclination_min_deg") or 0.0,
            inc_max_deg=intent.get("inclination_max_deg") or 98.0,
        )

    # custom — use a ±50 km band around altitude_km if range not given
    alt_center = intent.get("altitude_km") or 500.0
    alt_min = intent.get("altitude_min_km") or (alt_center - 50.0)
    alt_max = intent.get("altitude_max_km") or (alt_center + 50.0)
    return custom(
        sma_min_m=R_EARTH + alt_min * 1000.0,
        sma_max_m=R_EARTH + alt_max * 1000.0,
        inc_min_deg=intent.get("inclination_min_deg") or 0.0,
        inc_max_deg=intent.get("inclination_max_deg") or 98.0,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plan_mission(
    description: str,
    catalog: list[tuple[str, str, str]],
    epoch: datetime | None = None,
    *,
    dt: float = 60.0,
    popsize: int = 15,
    maxiter: int = 500,
    seed: int | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
    on_intent: Callable[[dict], None] | None = None,
) -> MissionPlan:
    """Parse a NL mission description and find the optimal safe orbit.

    Args:
        description:       Plain English goal, e.g. "Sun-synchronous Earth observation at 550 km"
        catalog:           TLE catalog from fetch_tles()
        epoch:             Mission start time (UTC). Defaults to now.
        dt:                Integrator time step in seconds.
        popsize:           Differential evolution population multiplier (total = popsize × 5).
        maxiter:           Maximum optimizer generations.
        seed:              RNG seed for reproducibility.
        progress_callback: Optional fn(generation, best_score) called each generation.
        on_intent:         Optional fn(intent) called once, after Claude has parsed the
                           description and before optimization begins. This is the only
                           point where the intent is known but the long run has not yet
                           started, so a CLI can report what it understood before going
                           quiet for several minutes.

    Returns:
        MissionPlan containing the extracted intent, orbital bounds, and optimizer result.
    """
    if epoch is None:
        epoch = datetime.now(timezone.utc)

    intent = _parse_mission(description)
    if on_intent is not None:
        on_intent(intent)
    bounds = _intent_to_bounds(intent)
    duration_s = intent["duration_days"] * 86400.0

    result = run_optimizer(
        bounds=bounds,
        epoch=epoch,
        duration_s=duration_s,
        catalog=catalog,
        dt=dt,
        popsize=popsize,
        maxiter=maxiter,
        seed=seed,
        progress_callback=progress_callback,
    )

    return MissionPlan(
        description=description,
        intent=intent,
        bounds=bounds,
        result=result,
        catalog_size=len(catalog),
    )
