"""
Tests for agent/agent.py.

The Claude API call is mocked so no ANTHROPIC_API_KEY is required.
The C++ integrator must be built first (the optimizer uses it).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agent.agent import MissionPlan, _intent_to_bounds, _parse_mission, plan_mission
from solver.constraint_solver import R_EARTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claude_response(orbit_type: str, **kwargs) -> MagicMock:
    """Build a mock Anthropic messages.create() response with a JSON text block."""
    payload = {"orbit_type": orbit_type, "duration_days": 1.0, "rationale": "test", **kwargs}
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(payload)
    resp = MagicMock()
    resp.content = [block]
    return resp


# ---------------------------------------------------------------------------
# _parse_mission — mock the Claude API
# ---------------------------------------------------------------------------


def test_parse_mission_sun_synchronous():
    mock_resp = _claude_response("sun_synchronous", altitude_km=550.0, duration_days=7.0)
    with patch("agent.agent.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_resp
        intent = _parse_mission("7-day sun-synchronous Earth observation at 550 km")

    assert intent["orbit_type"] == "sun_synchronous"
    assert intent["altitude_km"] == 550.0
    assert intent["duration_days"] == 7.0


def test_parse_mission_iss_rendezvous():
    mock_resp = _claude_response("iss_rendezvous", duration_days=3.0)
    with patch("agent.agent.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_resp
        intent = _parse_mission("Rendezvous with the ISS for a 3-day crew rotation")

    assert intent["orbit_type"] == "iss_rendezvous"
    assert intent["duration_days"] == 3.0


def test_parse_mission_passes_description_as_user_message():
    """Verify the NL description is sent verbatim as the user message."""
    mock_resp = _claude_response("polar", altitude_km=600.0)
    description = "Global ice coverage survey at 600 km polar orbit"
    with patch("agent.agent.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_resp
        _parse_mission(description)

    call_kwargs = MockClient.return_value.messages.create.call_args[1]
    user_msg = call_kwargs["messages"][0]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == description


# ---------------------------------------------------------------------------
# _intent_to_bounds — pure unit tests, no mock needed
# ---------------------------------------------------------------------------


def test_intent_to_bounds_sun_synchronous():
    intent = {"orbit_type": "sun_synchronous", "altitude_km": 500.0, "duration_days": 1.0, "rationale": ""}
    bounds = _intent_to_bounds(intent)
    # SSO inclination at 500 km is slightly above 90° (retrograde)
    assert bounds.inc_min > np.radians(90.0)
    assert bounds.inc_max > np.radians(90.0)
    assert abs(bounds.sma_center - (R_EARTH + 500e3)) < 25e3


def test_intent_to_bounds_iss_rendezvous():
    intent = {"orbit_type": "iss_rendezvous", "duration_days": 1.0, "rationale": ""}
    bounds = _intent_to_bounds(intent)
    # ISS at ~415 km, 51.6°
    assert abs(bounds.sma_center - (R_EARTH + 415e3)) < 20e3
    assert abs(np.degrees(bounds.inc_center) - 51.6) < 0.1


def test_intent_to_bounds_polar():
    intent = {"orbit_type": "polar", "altitude_km": 600.0, "duration_days": 1.0, "rationale": ""}
    bounds = _intent_to_bounds(intent)
    # Polar centred on 90°
    assert abs(np.degrees(bounds.inc_center) - 90.0) < 2.5


def test_intent_to_bounds_low_earth_orbit():
    intent = {
        "orbit_type": "low_earth_orbit",
        "altitude_min_km": 400.0,
        "altitude_max_km": 700.0,
        "inclination_min_deg": 45.0,
        "inclination_max_deg": 55.0,
        "duration_days": 1.0,
        "rationale": "",
    }
    bounds = _intent_to_bounds(intent)
    assert abs(bounds.sma_min - (R_EARTH + 400e3)) < 1.0
    assert abs(bounds.sma_max - (R_EARTH + 700e3)) < 1.0
    assert abs(np.degrees(bounds.inc_min) - 45.0) < 0.01
    assert abs(np.degrees(bounds.inc_max) - 55.0) < 0.01


def test_intent_to_bounds_custom_single_altitude():
    # custom with only altitude_km → ±50 km band
    intent = {
        "orbit_type": "custom",
        "altitude_km": 800.0,
        "inclination_min_deg": 60.0,
        "inclination_max_deg": 70.0,
        "duration_days": 1.0,
        "rationale": "",
    }
    bounds = _intent_to_bounds(intent)
    assert abs(bounds.sma_min - (R_EARTH + 750e3)) < 1.0
    assert abs(bounds.sma_max - (R_EARTH + 850e3)) < 1.0
    assert abs(np.degrees(bounds.inc_min) - 60.0) < 0.01
    assert abs(np.degrees(bounds.inc_max) - 70.0) < 0.01


# ---------------------------------------------------------------------------
# plan_mission — mocked Claude API + empty TLE catalog
# ---------------------------------------------------------------------------


def test_plan_mission_end_to_end():
    """Full pipeline: mocked NL parse → bounds → optimizer (no catalog, for speed)."""
    mock_resp = _claude_response("sun_synchronous", altitude_km=500.0, duration_days=0.5)

    with patch("agent.agent.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_resp
        plan = plan_mission(
            description="Half-day SSO at 500 km",
            catalog=[],
            epoch=datetime(2025, 1, 1, tzinfo=timezone.utc),
            dt=60.0,
            popsize=4,
            maxiter=5,
            seed=42,
        )

    assert isinstance(plan, MissionPlan)
    assert plan.catalog_size == 0
    assert plan.intent["orbit_type"] == "sun_synchronous"
    assert plan.result.safe           # no catalog → no conjunctions
    assert plan.result.objective < 0.05


def test_plan_mission_progress_callback():
    mock_resp = _claude_response("polar", altitude_km=600.0, duration_days=0.25)
    calls: list[tuple[int, float]] = []

    with patch("agent.agent.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_resp
        plan_mission(
            description="Polar survey",
            catalog=[],
            epoch=datetime(2025, 1, 1, tzinfo=timezone.utc),
            dt=60.0,
            popsize=4,
            maxiter=3,
            seed=0,
            progress_callback=lambda g, s: calls.append((g, s)),
        )

    assert len(calls) > 0, "progress_callback should have been called at least once"
    gens = [c[0] for c in calls]
    assert gens == sorted(gens), "generation numbers should be non-decreasing"
