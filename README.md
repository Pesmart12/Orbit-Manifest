# Orbit Manifest

**Natural language mission design for orbital mechanics.**

Describe what you want your satellite to do. Orbit Manifest designs an orbit that achieves it — optimized for your objectives and validated against the live LEO satellite catalog.

```
"Design a sun-synchronous orbit passing over the equator 3 times in 5 days,
 minimizing delta-v, with no conjunctions with active satellites."
```

→ Orbital elements. Ground track. Launch window. Delta-v budget. Conjunction report.

---

## What It Does

Traditional mission design tools (GMAT, STK) require you to already know your orbital elements. Orbit Manifest works the other way: you describe the mission goal, and the system figures out the orbit.

**Example queries:**

- *"Design an orbit that passes over Lagos every 12 hours for 30 days"*
- *"Find the most fuel-efficient LEO orbit with equatorial ground coverage and no debris conjunctions"*
- *"Sun-synchronous orbit for Earth observation, 500km altitude, 3 revisits per day over Europe"*

The system handles the astrodynamics. You handle the mission.

---

## Architecture

```
Natural Language Input
        │
        ▼
  Claude API Agent          ← parses goals, handles ambiguity, composes output
        │
        ▼
  Constraint Solver         ← maps mission goals to orbital mechanics constraints
        │
        ▼
  Orbital Optimizer         ← scipy outer loop over candidate orbits
        ├──► C++ RK4 Integrator (pybind11)     ← fast physics propagation
        └──► Situational Awareness Layer        ← conjunction checks vs live TLE catalog
        │
        ▼
  Mission Plan Output       ← elements, ground track, launch window, delta-v, conjunction report
```

### Components

| Component | Technology | Role |
|---|---|---|
| Agent | Claude API (claude-sonnet-4) | NL parsing, orchestration, output |
| Constraint Solver | Python | Mission goal → orbital constraints |
| Optimizer | scipy | Multi-objective orbit optimization |
| Integrator | C++ / pybind11 | RK4 2-body + J2 propagation |
| Situational Awareness | python-sgp4 | Conjunction analysis vs live catalog |
| Data source | Space-Track.org | Live TLE catalog (~20k objects) |

---

## Physics

The integrator implements:

- **2-body gravitational model** — Newtonian point-mass gravity
- **J2 perturbation** — Earth oblateness; required for accurate sun-synchronous orbit nodal precession
- **RK4 fixed-step integration** — 4th order Runge-Kutta

Written in C++, exposed to Python via pybind11. The optimizer calls it thousands of times per run — performance is non-negotiable.

Planned extensions: atmospheric drag, n-body (lunar/solar), adaptive step size (RK45).

---

## Getting Started

### Prerequisites

- Python 3.10+
- C++17 compiler (GCC 11+ or Clang 14+)
- CMake 3.20+
- Anthropic API key
- Space-Track.org account (free)

### Installation

```bash
git clone https://github.com/yourusername/orbit-manifest
cd orbit-manifest

# Python dependencies
pip install -r requirements.txt

# Build C++ integrator and install as Python module
pip install -e .

# Configure credentials
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and Space-Track credentials
```

### Build C++ integrator manually

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run

```bash
python -m agent.agent "Design a sun-synchronous orbit passing over the equator 3 times in 5 days, minimizing delta-v, no conjunctions"
```

---

## Output

Orbit Manifest returns a full mission plan:

```
═══════════════════════════════════════
  ORBIT MANIFEST — MISSION PLAN
═══════════════════════════════════════

Orbital Elements
  Semi-major axis (a):    6878.1 km
  Eccentricity (e):       0.0001
  Inclination (i):        97.4°
  RAAN (Ω):               312.7°
  Arg. of perigee (ω):    0.0°

Mission Performance
  Orbital period:         90.5 min
  Equatorial passes:      3 in 5 days ✓
  Orbit type:             Sun-synchronous ✓

Delta-V Budget
  Launch to operational:  ~1.8 km/s (from Vandenberg)
  Station-keeping (1yr):  ~10 m/s

Conjunction Report
  Objects screened:       22,847
  Close approaches (< 5km): 0 ✓
  Minimum separation:     12.3 km (NOAA-20, T+2.3 days)

Launch Window
  Next opportunity:       2025-03-15 06:42 UTC
  RAAN alignment from:    Vandenberg SFB (34.7°N, 120.6°W)
```

Ground track and trajectory plots saved to `output/`.

---

## Project Status

| Component | Status |
|---|---|
| C++ RK4 Integrator | Implemented |
| pybind11 bindings | Implemented |
| Situational Awareness | In progress |
| Constraint Solver | Planned |
| Orbital Optimizer | Planned |
| Claude API Agent | Planned |
| Output Composer | Planned |

---

## Why Not GMAT or STK?

| | GMAT | STK | Orbit Manifest |
|---|---|---|---|
| Cost | Free | $$$$ | Free / open source |
| Input | Orbital elements | Orbital elements | Natural language |
| Learning curve | High | High | Low |
| Mission design | Yes | Yes | Yes |
| Conjunction analysis | Yes | Yes | Yes |
| Agentic / NL interface | No | No | Yes |
| Target users | Experts | Enterprise | Small teams, universities |

Orbit Manifest is not trying to replace GMAT or STK for professional missions with large teams. It's trying to make serious orbital mission design accessible to the teams that currently can't use those tools.

---

## References

- Vallado, *Fundamentals of Astrodynamics and Applications*
- Bate, Mueller, White, *Fundamentals of Astrodynamics*
- Brandon Rhodes, [python-sgp4](https://github.com/brandon-rhodes/python-sgp4)
- [Space-Track.org](https://www.space-track.org)
- [pybind11](https://pybind11.readthedocs.io)

---

## License

MIT
