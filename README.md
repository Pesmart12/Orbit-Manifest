# Orbit Manifest

**Natural language mission design for orbital mechanics.**

Describe what you want your satellite to do. Orbit Manifest designs an orbit that achieves it — optimized against your objectives and validated against the live LEO satellite catalog.

```
python run.py "7-day sun-synchronous Earth observation at 550 km"
```

→ Optimal orbital elements. Conjunction safety report. Ground-track plot.

---

## What It Does

Traditional mission design tools (GMAT, STK) require you to already know your orbital elements. Orbit Manifest works the other way: you describe the mission goal in plain English, and the system figures out the orbit.

**Example queries:**

- *"Sun-synchronous Earth observation at 550 km, 7 days"*
- *"ISS rendezvous mission, 3-day crew rotation"*
- *"Polar ice survey at 600 km, 5 days"*
- *"LEO communications orbit, 500–700 km, mid-inclination"*

The system handles the astrodynamics. You handle the mission.

---

## Architecture

```
Natural Language Input
        │
        ▼
  Claude API (claude-opus-4-8)   ← extracts orbit type and parameters
        │
        ▼
  Constraint Solver              ← maps mission goal to OrbitalBounds
        │
        ▼
  Differential Evolution         ← scipy optimizer over 5 Keplerian elements
        ├──► C++ RK4 Integrator (pybind11)    ← fast 2-body + J2 propagation
        └──► Conjunction Checker              ← vectorized SGP4 vs live TLE catalog
        │
        ▼
  Output Composer                ← text report + ground-track PNG
```

### Components

| Component | Technology | Role |
|---|---|---|
| Agent | Claude API (`claude-opus-4-8`) | NL parsing → structured orbit intent |
| Constraint Solver | Python | Mission goal → `OrbitalBounds` |
| Optimizer | scipy `differential_evolution` | Global search over Keplerian elements |
| Integrator | C++17 / pybind11 | RK4 2-body + J2 propagation |
| Conjunction Checker | python-sgp4, numpy | Vectorized catalog vs mission orbit |
| TLE Catalog | Space-Track.org | Live LEO catalog (~20 k objects, 24-hr cache) |
| Output | matplotlib | Ground-track plot + formatted text report |

---

## Physics

The C++ integrator implements:

- **2-body gravitational model** — Newtonian point-mass gravity
- **J2 perturbation** — Earth oblateness; required for accurate sun-synchronous nodal precession
- **RK4 fixed-step integration** — 4th-order Runge-Kutta

The TLE catalog is propagated with **SGP4** (python-sgp4) — the correct model for TLE mean elements. The mission orbit uses RK4 on osculating elements. The two are never mixed.

Conjunction checking is fully vectorized: a single `SatrecArray.sgp4()` call produces a `(T, N, 3)` position array; separation is computed with one numpy broadcast, no Python loop over time steps. Performance: ~2 s/generation vs ~37 s before refactoring.

---

## Getting Started

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- C++17 compiler (GCC 11+, Clang 14+, or MSVC 2022)
- CMake 3.20+
- [Anthropic API key](https://console.anthropic.com) (`sk-ant-...`)
- [Space-Track.org](https://www.space-track.org) account (free)

### Installation

```bash
git clone https://github.com/yourusername/orbit-manifest
cd orbit-manifest

# Create and activate conda environment (installs every dependency)
conda env create -f environment.yml
conda activate orbit-manifest

# Build the C++ integrator (must come last)
pip install -e .
```

### Configure credentials

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
SPACE_TRACK_USER=your@email.com
SPACE_TRACK_PASS=yourpassword
```

No quotes, no spaces around `=`.

### Run

```bash
# Quick pipeline check (~30 s, low-fidelity)
python run.py "7-day sun-synchronous Earth observation at 550 km" --quick

# Full production run (~17 min on 8-core machine)
python run.py "7-day sun-synchronous Earth observation at 550 km"

# Other options
python run.py "ISS rendezvous, 3 days" --output results/ --seed 42
```

Output lands in `results/mission_report.txt` and `results/ground_track.png`.

### Run tests

```bash
pytest tests/
```

---

## Sample Output

```
══════════════════════════════════════════════════════════════
                 ORBIT MANIFEST — Mission Report
══════════════════════════════════════════════════════════════

  Mission:    7-day sun-synchronous Earth observation at 550 km
  Orbit type: Sun Synchronous
  Generated:  2025-06-19 14:32:07 UTC
  Rationale:  SSO at 550 km provides consistent solar lighting for imaging

──────────────────────────────────────────────────────────────
  OPTIMAL ORBITAL ELEMENTS
──────────────────────────────────────────────────────────────
  Semi-major axis:      6928.10 km
  Altitude (peri):       549.8 km
  Altitude (apo):        550.2 km
  Inclination:            97.6412 °
  Eccentricity:         0.000061
  Period:                97.4 min
  RAAN:                  135.22 °
  Arg. of perigee:        22.07 °

──────────────────────────────────────────────────────────────
  CONJUNCTION SAFETY
──────────────────────────────────────────────────────────────
  Catalog size:          24,857 objects
  Safety status:         SAFE  ✓

──────────────────────────────────────────────────────────────
  OPTIMIZER STATISTICS
──────────────────────────────────────────────────────────────
  Converged:                    Yes
  Objective (ecc):         0.000061
  Generations:                  312
  Conjunction calls:         23,400
  Wall-clock time:            982.4 s
  Message:           Optimization terminated successfully.
```

---

## Why Not GMAT or STK?

| | GMAT | STK | Orbit Manifest |
|---|---|---|---|
| Cost | Free | $$$$ | Free / open source |
| Input | Orbital elements | Orbital elements | Natural language |
| Learning curve | High | High | Low |
| Conjunction analysis | Yes | Yes | Yes |
| NL interface | No | No | Yes |
| Target users | Experts | Enterprise | Small teams, universities |

Orbit Manifest is not trying to replace GMAT or STK for professional missions. It's trying to make serious orbital mission design accessible to teams that currently can't use those tools.

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
