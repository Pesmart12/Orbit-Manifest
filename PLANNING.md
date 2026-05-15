# Orbit Manifest — PLANNING.md

## Project Summary

Orbit Manifest is a natural language mission design agent that converts human-readable orbital mission goals into viable, optimized orbits with full safety assessments against the live LEO satellite catalog. Users describe what they want a satellite to do; Orbit Manifest designs an orbit that achieves it.

**Target users:** CubeSat teams, university rocketry/satellite programs, early-stage NewSpace companies without STK licenses or GMAT expertise.

---

## Architecture Overview

```
User Input (natural language mission goal)
        │
        ▼
┌─────────────────────────────────┐
│     Claude API Agent            │
│  - Parse mission goal           │
│  - Decompose into constraints   │
│  - Handle ambiguity             │
│  - Compose final output         │
└────────────┬────────────────────┘
             │ Structured constraints
             ▼
┌─────────────────────────────────┐
│   Mission Constraint Solver     │
│  - Map NL goals → orbital params│
│  - Sun-sync inclination calc    │
│  - Coverage requirement parser  │
│  - Generate candidate orbits    │
└────────────┬────────────────────┘
             │ Candidate orbital element sets
             ▼
┌─────────────────────────────────┐
│      Orbital Optimizer          │◄── Optimization loop
│  - scipy outer loop             │    (many iterations)
│  - Objectives: min delta-v,     │
│    max coverage, min time       │
│  - Calls integrator per iter    │
└──────┬──────────────┬───────────┘
       │              │
       │ Propagation  │ Conjunction check
       ▼              ▼
┌────────────┐  ┌──────────────────────────────┐
│ C++ RK4    │  │   Situational Awareness       │
│ Integrator │  │  - Fetch TLE catalog          │
│ (pybind11) │  │    (Space-Track.org)          │
│            │  │  - SGP4 propagate catalog     │
│ - 2-body   │  │  - Compute min separations   │
│ - J2 perturb│  │  - Reject unsafe orbits      │
│ - Batch mode│  └──────────────┬───────────────┘
└────────────┘                  │
                    Safe, optimized orbit
                                │
                                ▼
               ┌────────────────────────────────┐
               │       Output Composer          │
               │  - Orbital elements            │
               │  - Ground track plot           │
               │  - Launch window               │
               │  - Delta-v budget              │
               │  - Conjunction report          │
               │  - Claude narrative summary    │
               └────────────────────────────────┘
```

---

## Components

### 1. Claude API Agent (`agent/`)
**Purpose:** Entry point. Parses natural language mission goals, decomposes them into structured constraints, orchestrates all downstream tools, composes the final output.

**Tools the agent has:**
- `solve_constraints(mission_goal: str) -> CandidateOrbits` — calls Constraint Solver
- `optimize_orbit(candidates, objectives, constraints) -> OptimizedOrbit` — calls Optimizer
- `check_conjunctions(orbit, duration_days) -> ConjunctionReport` — calls Situational Awareness
- `propagate_orbit(state_vector, duration) -> Trajectory` — calls C++ integrator
- `compose_output(orbit, report, trajectory) -> MissionPlan` — formats final result

**Model:** `claude-sonnet-4-20250514`

**Ambiguity handling:** Agent asks clarifying questions when goals are underspecified (e.g. "efficient" → asks whether user means delta-v, fuel mass, or time-to-orbit).

---

### 2. Mission Constraint Solver (`solver/`)
**Purpose:** Translates structured mission goals into orbital mechanics constraints and generates candidate orbital element sets.

**Key mappings:**
- `"sun-synchronous"` → inclination as a function of altitude (i ≈ 90° + small correction from J2)
- `"pass over equator N times in D days"` → constrains orbital period and RAAN
- `"low Earth orbit"` → altitude band 200–2000 km
- `"avoid collisions"` → minimum separation constraint fed to situational awareness layer
- `"most efficient"` → add delta-v minimization to optimizer objectives

**Output:** List of `CandidateOrbit` objects (a, e, i, Ω, ω, ν) with associated objective weights.

---

### 3. Orbital Optimizer (`optimizer/`)
**Purpose:** Optimizes over candidate orbits to find the solution that best satisfies mission objectives within hard constraints.

**Approach:**
- Outer loop: `scipy.optimize` (differential evolution or SLSQP depending on problem structure)
- Inner loop: C++ integrator called via pybind11 for propagation (performance-critical)
- Hard constraints: altitude bounds, inclination limits, conjunction clearance from situational awareness layer
- Soft objectives: minimize delta-v, maximize ground track coverage, minimize time to operational orbit

**Key note:** The optimizer calls the C++ integrator thousands of times. This is why the integrator must be fast and exposed via pybind11 rather than a subprocess call.

---

### 4. C++ RK4 Integrator (`integrator/`)
**Purpose:** Fast, accurate numerical propagation of orbital state vectors. The performance-critical core of the system.

**Files:**
- `integrator.h` — class definition, state vector types
- `integrator.cpp` — RK4 implementation, equations of motion, J2 perturbation
- `bindings.cpp` — pybind11 bindings, numpy array I/O

**Physics:**
- 2-body gravitational model (baseline)
- J2 perturbation (Earth oblateness) — required for accurate sun-synchronous orbit nodal precession
- RK4 fixed-step integration (adaptive step optional later)
- Batch mode: propagate N candidate orbits in parallel

**Build:** CMake (`CMakeLists.txt`). Python wheels via `pip install .` using scikit-build-core or setup.py with pybind11.

**Validation:** Test against known analytical solutions (circular orbit period, J2 nodal drift rate) before wiring into optimizer.

**Future extensions:**
- Atmospheric drag (exponential density model)
- Lunar/solar perturbations (n-body)
- Adaptive step size (RK45)

---

### 5. Situational Awareness Layer (`awareness/`)
**Purpose:** Checks candidate orbits against the live LEO satellite catalog for conjunctions. Acts as a hard constraint in the optimizer loop.

**Data source:** Space-Track.org TLE catalog (free with registration). Updated daily.

**Approach:**
- Fetch current TLE catalog on startup / cache with TTL
- Use `python-sgp4` (Brandon Rhodes) to propagate catalog objects — SGP4 is sufficient for screening thousands of objects quickly
- For each optimizer candidate orbit, compute minimum separation distance over the propagation window
- Reject any orbit with minimum separation below configurable threshold (default: 5 km)
- Flag close approaches for inclusion in output conjunction report

**Key note:** The C++ integrator is used for the mission orbit. SGP4 (Python) is used for the catalog objects. No need to run 20,000 objects through RK4.

---

### 6. Output Composer (`output/`)
**Purpose:** Takes the validated, optimized orbit and produces the full mission plan.

**Outputs:**
- Orbital elements (a, e, i, Ω, ω, ν) with units
- Ground track plot (matplotlib or plotly)
- Launch window (RAAN targeting from launch site)
- Delta-v budget (launch to operational orbit)
- Conjunction report (closest approach distances, object IDs)
- Claude narrative summary in plain English

---

## Data Sources

| Data | Source | Notes |
|---|---|---|
| Live TLE catalog | Space-Track.org | Free with registration, ~20k objects |
| Earth gravity (J2) | EGM96 standard | J2 = 1.08263e-3, hardcoded constant |
| Launch site coords | Hardcoded small DB | ~10 major launch sites to start |
| Solar/atmospheric indices | NOAA Space Weather | F10.7, Kp — for drag modeling later |

---

## Repo Structure

```
orbit-manifest/
├── agent/
│   ├── agent.py              # Claude API orchestration
│   └── tools.py              # Tool definitions (propagate, optimize, check, compose)
├── solver/
│   ├── constraint_solver.py  # NL goal → orbital constraints
│   └── orbital_mechanics.py  # Sun-sync inclination, period calc, etc.
├── optimizer/
│   └── optimizer.py          # scipy outer loop, calls integrator inner loop
├── integrator/
│   ├── integrator.h          # State vector types, class definition
│   ├── integrator.cpp        # RK4 + J2 perturbation implementation
│   └── bindings.cpp          # pybind11 numpy bindings
├── awareness/
│   ├── tle_fetcher.py        # Space-Track.org downloader + cache
│   └── conjunction.py        # SGP4 catalog propagation, min separation
├── output/
│   ├── composer.py           # Mission plan assembly
│   └── plotting.py           # Ground track + trajectory plots
├── tests/
│   ├── test_integrator.py    # Validate RK4 against analytical solutions
│   ├── test_solver.py        # Constraint mapping unit tests
│   └── test_conjunction.py   # Conjunction detection tests
├── scripts/
│   └── fetch_tles.py         # One-shot TLE catalog downloader
├── data/
│   └── launch_sites.json     # Launch site coordinates
├── CMakeLists.txt            # C++ build config
├── setup.py                  # pybind11 Python package build
├── requirements.txt
├── .env.example              # ANTHROPIC_API_KEY, SPACE_TRACK credentials
└── README.md
```

---

## Build Order

### Phase 1 — C++ Integrator (build and validate first)
1. Implement `integrator.h` / `integrator.cpp` — RK4 2-body with J2
2. Write `bindings.cpp` — pybind11, numpy array I/O
3. Configure `CMakeLists.txt` and `setup.py`
4. Validate against analytical solutions in `tests/test_integrator.py`
   - Circular orbit: period T = 2π√(a³/μ)
   - J2 nodal drift: dΩ/dt = -3/2 * n * J2 * (R_e/a)² * cos(i) / (1-e²)²

### Phase 2 — Situational Awareness
5. Implement `tle_fetcher.py` — Space-Track API, local cache with TTL
6. Implement `conjunction.py` — SGP4 propagation, min separation computation
7. Test on a known conjunction scenario

### Phase 3 — Constraint Solver
8. Implement `orbital_mechanics.py` — core calculations (sun-sync inclination, period, RAAN)
9. Implement `constraint_solver.py` — map structured goals to candidate orbital elements
10. Unit test each constraint mapping

### Phase 4 — Optimizer
11. Wire scipy optimizer to integrator (inner loop) and conjunction checker (hard constraint)
12. Test on simple single-objective case (minimize delta-v for circular LEO)
13. Add multi-objective support

### Phase 5 — Agent Layer
14. Define tool schemas for Claude API function calling
15. Implement `agent.py` — full orchestration
16. Implement ambiguity handling and clarification flow
17. Implement `output/composer.py` and `output/plotting.py`

### Phase 6 — End-to-End Demo
18. Query: *"Design a sun-synchronous orbit that passes over the equator 3 times in 5 days, minimizing delta-v, with no conjunctions"*
19. Full pipeline run, validate output against manual calculation
20. Polish output formatting

---

## Key Constants

```python
MU_EARTH    = 3.986004418e14   # m^3/s^2 — Earth gravitational parameter
R_EARTH     = 6.3781e6         # m — Earth equatorial radius
J2          = 1.08263e-3       # Earth oblateness coefficient
OMEGA_EARTH = 7.2921150e-5     # rad/s — Earth rotation rate
```

---

## Environment Setup

```bash
# Python dependencies
pip install anthropic sgp4 scipy numpy matplotlib requests python-dotenv

# C++ build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Or via pip (installs pybind11 module)
pip install -e .

# Credentials (.env)
ANTHROPIC_API_KEY=sk-...
SPACE_TRACK_USER=your@email.com
SPACE_TRACK_PASS=yourpassword
```

---

## Key References

- Brandon Rhodes python-sgp4: https://github.com/brandon-rhodes/python-sgp4
- Vallado, *Fundamentals of Astrodynamics and Applications* — RK4, J2, SGP4 reference
- Bate, Mueller, White, *Fundamentals of Astrodynamics* — orbital mechanics foundation
- Space-Track TLE API: https://www.space-track.org
- pybind11 docs: https://pybind11.readthedocs.io
- EGM96 gravity model: https://cddis.nasa.gov/926/egm96/

---

## Claude Code Notes

- **Start with the integrator.** It is the performance core and can be built and validated in complete isolation before any other component exists.
- **Each component is independently testable.** Build and validate each phase before wiring to the next.
- **The optimizer's inner loop calls the C++ integrator.** Keep the pybind11 interface simple: numpy arrays in, numpy arrays out.
- **The situational awareness layer uses SGP4, not RK4.** Do not over-engineer the catalog propagation — SGP4 is fast enough for screening 20k objects.
- **Agent layer is last.** All physics and optimization must work correctly before adding the NL interface.
- **Validate physics first.** Every integrator PR should pass `test_integrator.py` before merging.
