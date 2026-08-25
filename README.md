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
        └──► Conjunction Checker              ← two-level SGP4 sieve, GPU or numpy
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
| Conjunction Checker | python-sgp4, CuPy / numpy | Two-level sieve, catalog vs mission orbit |
| TLE Catalog | Space-Track.org | Live catalog (32 k objects, 24-hr cache) |
| Output | matplotlib | Ground-track plot + formatted text report |

---

## Physics

The C++ integrator implements:

- **2-body gravitational model** — Newtonian point-mass gravity
- **J2 perturbation** — Earth oblateness; required for accurate sun-synchronous nodal precession
- **RK4 fixed-step integration** — 4th-order Runge-Kutta

The TLE catalog is propagated with **SGP4** (python-sgp4) — the correct model for TLE mean elements. The mission orbit uses RK4 on osculating elements. The two are never mixed.

### Conjunction screening

The catalog is propagated in one vectorised `SatrecArray.sgp4()` call into a
`(T, N, 3)` position array, cached per altitude bucket for the length of an
optimizer run.

Screening is a **two-level sieve**, because sampling separations on a fixed grid
does not merely mis-measure a close approach — it can miss which approach matters.
Measured against a 1-second reference on the live catalog, a 60-second grid
reported one object's true 4.0 km pass as 43.1 km, having locked onto a different
event 48 minutes away. Level 1 measures every object on the coarse grid; level 2
re-searches the **whole window again** at 10 s for objects that could still reach
the threshold, promoted on a rigorous relative-velocity bound so nothing is
dropped. Sub-sample parabolic interpolation then recovers the true minimum
between samples.

Separation runs on the GPU through **CuPy** when it is available, falling back to
numpy otherwise; both produce bit-identical results and the test suite asserts it.
On a 4,873-object band the GPU is 33.8x faster per candidate, which is what makes
the sieve affordable at all — a full run is ~1.4 h with it against ~52 h without.

---

## Getting Started

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- C++17 compiler (GCC 11+, Clang 14+, or MSVC 2022)
- CMake 3.20+
- [Anthropic API key](https://console.anthropic.com) (`sk-ant-...`)
- [Space-Track.org](https://www.space-track.org) account (free)
- *Optional:* an NVIDIA GPU. With CuPy installed (`conda install -c conda-forge cupy`)
  conjunction screening runs on the GPU; without it everything still works on numpy,
  just slower. Set `ORBIT_MANIFEST_NO_GPU=1` to force the numpy path.

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
# Quick pipeline check — low-fidelity, verifies the pipeline end to end
python run.py "7-day sun-synchronous Earth observation at 550 km" --quick

# Full production run (~1.4 h on an RTX 4070; longer on CPU — see note below)
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

Real output from an abbreviated run (14 generations; a full run searches 500)
against the live catalog. The 550 km sun-synchronous shell is crowded, and a
short search does not find a clear orbit — which is the tool doing its job:

```
==============================================================
                ORBIT MANIFEST — Mission Report               
==============================================================

  Mission:    7-day sun-synchronous Earth observation at 550 km
  Orbit type: Sun Synchronous
  Generated:  2026-08-25 00:00:00 UTC
  Rationale:  SSO at 550 km provides consistent solar lighting for imaging

──────────────────────────────────────────────────────────────
  OPTIMAL ORBITAL ELEMENTS
──────────────────────────────────────────────────────────────
  Semi-major axis:      6913.46 km
  Altitude (peri):       532.81 km
  Altitude (apo):        537.91 km
  Inclination:          97.6467 °
  Eccentricity:        0.000368
  Period:                 95.35 min
  RAAN:                  234.36 °
  Arg. of perigee:       269.98 °

──────────────────────────────────────────────────────────────
  CONJUNCTION SAFETY
──────────────────────────────────────────────────────────────
  Catalog size:          32,364 objects
  Screened:               8,019 objects in the altitude band
  Safety status:     CONJUNCTION RISK  ✗

  Closest approach:        2.66 km
    object:          BREEZE-KM R/B (NORAD 43438)
    at:              T+4 d 00:21

  Objects within threshold (11):
        2.66 km  T+4 d 00:21   BREEZE-KM R/B (NORAD 43438)
        2.82 km  T+21:33       FENGYUN 1C DEB (NORAD 36203)
        3.13 km  T+4 d 07:52   STARLINK-5221 (NORAD 54091)
        3.42 km  T+3 d 20:23   TDS 1 (NORAD 40076)
        3.60 km  T+3 d 04:34   FENGYUN 1C DEB (NORAD 31441)
        3.87 km  T+10:50       DELTA 1 DEB (NORAD 10234)
        3.89 km  T+2 d 18:56   OBJECT B (NORAD 69098)
        4.27 km  T+3 d 13:13   STARLINK-5811 (NORAD 55787)
        4.45 km  T+1 d 05:34   COSMOS 2151 (NORAD 21422)
        4.50 km  T+2 d 21:03   DELTA 1 DEB (NORAD 39102)
    … and 1 more

──────────────────────────────────────────────────────────────
  OPTIMIZER STATISTICS
──────────────────────────────────────────────────────────────
  Converged:                Yes
  Objective (margin): no safe orbit found
  Generations:                2
  Conjunction calls:         40
  Wall-clock time:         97.9 s
  Message:           Optimization terminated successfully.

==============================================================
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
