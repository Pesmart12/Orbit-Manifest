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

**Algorithm:** `scipy.optimize.differential_evolution` — chosen over SLSQP/L-BFGS-B because conjunction penalties create hard discontinuities (fitness jumps to 1e10) that break gradient-based solvers. Differential evolution is gradient-free, population-based, and finds global optima in fragmented feasible spaces.

**Search space:** 5 Keplerian elements `[sma, inc, ecc, raan, argp]`. Bounds come from `OrbitalBounds.as_scipy_bounds()` in the constraint solver.

**Critical scipy settings:**
```python
differential_evolution(
    batch_fitness_fn,
    bounds=orbital_bounds.as_scipy_bounds(),
    workers=1,           # NOT -1 — scipy multiprocessing conflicts with OpenMP
    updating='deferred', # required to batch-evaluate full population each generation
    popsize=15,          # 15 × 5 params = 75 candidates/generation
)
```

**Inner loop per generation:**
1. `keplerian_to_cartesian(population)` → `(N, 6)` ECI state vectors
2. `orbit_integrator.propagate_batch_final(states, dt, n_steps)` → `(N, 6)` terminal states via OpenMP
3. `check_conjunctions(state, ..., catalog_cache=cache)` per candidate → hard constraint filter
4. `mission_objective(final_state)` for surviving candidates → fitness score

**`CatalogCache` usage:** Instantiate once before the optimization loop. The same epoch/duration/dt applies to all candidates in a run, so catalog SGP4 positions are computed once and reused. Never create a new `CatalogCache` inside the fitness function.

**Typical convergence:** 400–700 generations (~15–25 min at ~2s/generation after catalog caching). May reach `maxiter=1000` in congested orbit shells.

**Key note:** The optimizer calls the C++ integrator thousands of times. The GIL is released inside `propagate_batch_final` before the OpenMP parallel region — OpenMP threads are native C++ threads and run truly in parallel. Never use `workers=-1` with batch propagation.

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

**Data source:** Space-Track.org TLE catalog (free with registration). Updated daily. Disk-cached for 24 hours in `data/tle_cache.json`.

**Implemented architecture (all 7 tests passing):**

- **`_filter_by_altitude(catalog, target_sma_m, band_m=50_000)`** — TLE string parsing only (columns 26-32 eccentricity, 52-62 mean motion). Computes periapsis/apoapsis via Kepler's third law. Reduces ~20k objects to ~2–5k in the relevant altitude shell before any SGP4 call.

- **`_compute_catalog_positions(sats, epoch, n_steps, dt)`** — single `SatrecArray.sgp4(jd_arr, fr_arr)` call with T-length time arrays. Returns `(T, N, 3)` meters. Replaces the previous design of T Python-loop SGP4 calls per candidate per generation.

- **`CatalogCache`** — in-memory dict keyed by `(epoch, duration_s, dt, round(sma, -3))`. One instance lives for an optimizer run. Cache key rounds altitude to nearest 1 km so nearby mission orbits share an entry. Memory guard warns at >3 GB.

- **Vectorized separation** — `(T, N, 3) - (T, 1, 3)` numpy broadcast → `linalg.norm(axis=2)` → `(T, N)`. No Python loop over time steps in the hot path.

**Performance:** ~37s/generation (original) → ~2s/generation (after caching).

**Key notes:**
- TLE mean elements are defined relative to SGP4 — do not propagate them with RK4. Running TLEs through an osculating integrator gives physically wrong positions.
- A PINN surrogate for the RK4 integrator is not worthwhile — propagation is ~70ms/generation (not the bottleneck), and position approximation errors could produce false-negative conjunction results.
- Reject any orbit with minimum separation below configurable threshold (default: 5 km).

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

### Phase 1 — C++ Integrator ✓ COMPLETE (4/4 tests passing)
1. [x] Implement `integrator.h` / `integrator.cpp` — RK4 2-body with J2
2. [x] Write `bindings.cpp` — pybind11, numpy array I/O, OpenMP batch parallelism
3. [x] Configure `CMakeLists.txt` and `setup.py`
4. [x] Validate against analytical solutions in `tests/test_integrator.py`
   - Circular orbit period: `T = 2π√(a³/μ)` — error < 1 m after one period
   - Energy conservation over 10 orbits — drift < 1 J/kg
   - J2 nodal drift: `dΩ/dt = -3/2 * n * J2 * (R_e/a)² * cos(i) / (1-e²)²` — error < 0.01 deg/day
   - Batch consistency: 20 identical orbits match single propagation to 10 decimal places

### Phase 2 — Situational Awareness ✓ COMPLETE (7/7 tests passing)
5. [x] Implement `tle_fetcher.py` — Space-Track API, local JSON cache with 24h TTL
6. [x] Implement `conjunction.py` — initial SGP4 propagation + separation computation
7. [x] Refactor `conjunction.py` for optimizer performance:
   - `_filter_by_altitude()` — TLE string parsing pre-filter, no SGP4
   - `_compute_catalog_positions()` — single vectorized SGP4 call across all T steps
   - `CatalogCache` — in-memory cache keyed by epoch/duration/dt/altitude
   - Fully vectorized numpy separation — no Python loop over time steps
   - Result: ~37s/generation → ~2s/generation
8. [x] 7 tests: TLE cache TTL, parse structure, separation geometry, pipeline smoke test, empty catalog, cache hit, altitude filter

### Phase 3 — Constraint Solver ✓ IMPLEMENTED (not fully tested)
9. [x] `solver/constraint_solver.py` — `OrbitalBounds` dataclass with goal constructors:
   - `sun_synchronous(altitude_km)` — solves J2 nodal drift for exact SSO inclination
   - `low_earth_orbit(alt_min, alt_max, inc_min, inc_max)`
   - `polar(altitude_km)`
   - `iss_rendezvous()`
   - `custom(...)`
   - `OrbitalBounds.as_scipy_bounds()` → `[(lo, hi), ...]` ready for scipy
10. [ ] Write `tests/test_solver.py` — unit test each goal constructor
11. [x] `physics/pre_propagation.py` — `OrbitScreen` pre-propagation gate:
    - Perigee clearance, apogee ceiling, window vs. period — O(1) Keplerian arithmetic, no integrator call
    - Used in Phase 4 optimizer to reject degenerate candidates before batch propagation
    - `physics/post_propagation.py` — `check_post_propagation` + `specific_energy` (deferred — see Future Experiments)
    - 8 tests in `tests/test_physics_agreement.py` (all passing)

### Phase 4 — Optimizer ← NEXT
12. Implement `optimizer/optimizer.py`:
    - `keplerian_to_cartesian(params)` — convert 5-element Keplerian → 6-element ECI state
    - `batch_fitness(population, cache, catalog, epoch, duration)` — batch evaluate N candidates; screen each with `check_pre_propagation` before propagation, assign penalty 1e10 on failure
    - `run_optimizer(orbital_bounds, catalog, epoch, duration, objective_fn)` — outer loop
13. Wire `differential_evolution` with `workers=1`, `updating='deferred'`, `popsize=15`
14. Wire `CatalogCache` — instantiate once, pass into fitness function
15. Wire `propagate_batch_final` as the inner propagation loop
16. Test on simple case: minimize SMA (lowest safe altitude) for a sun-synchronous orbit
17. Add delta-v objective once simple case passes

### Future Experiments (post-Phase 4)
- **Post-propagation integrity experiment:** Wire `check_post_propagation` into the optimizer and study when/how often RK4 at dt=10s fails energy or momentum conservation thresholds across a 7-day mission. Use findings to calibrate threshold values and explore adaptive timestep retry (re-propagate failing candidates at dt/2 with 2× n_steps before hard-rejecting).
- **Propagator comparison:** Benchmark RK4 accuracy vs. an alternative propagator using the post-propagation checks as the evaluation metric.

### Phase 5 — Agent Layer
17. Define tool schemas for Claude API function calling
18. Implement `agent.py` — full orchestration
19. Implement ambiguity handling and clarification flow
20. Implement `output/composer.py` and `output/plotting.py`

### Phase 6 — End-to-End Demo
21. Query: *"Design a sun-synchronous orbit that passes over the equator 3 times in 5 days, minimizing delta-v, with no conjunctions"*
22. Full pipeline run, validate output against manual calculation
23. Polish output formatting

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
