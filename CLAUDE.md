# CLAUDE.md — Orbit Manifest

## What This Project Is
A natural language orbital mission design agent. Users describe a mission goal in plain English; the system produces an optimized orbit with conjunction analysis against the live LEO satellite catalog. See PLANNING.md for full architecture.

---

## Build Commands

```bash
# Install build dependencies (first time only)
pip install pybind11 numpy setuptools

# Install Python dependencies
pip install -r requirements.txt

# Build C++ integrator and install as Python module
pip install -e .

# Build C++ manually (alternative)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)   # Linux/Mac
# cmake --build . --config Release   # Windows

# Run tests
pytest tests/
```

---

## Repo Layout

```
orbit-manifest/
├── integrator/           # C++ RK4 integrator — BUILD THIS FIRST
│   ├── integrator.h      # StateVector type, constants, declarations
│   ├── integrator.cpp    # RK4 + 2-body + J2 equations of motion
│   └── bindings.cpp      # pybind11 numpy bindings
├── tests/
│   ├── test_integrator.py  # Validation: period, energy, J2 drift, batch (4 tests)
│   └── test_conjunction.py # Conjunction checker unit tests (7 tests)
├── agent/                # Claude API orchestration (not yet implemented)
├── solver/               # NL goal → orbital constraints
│   └── constraint_solver.py  # OrbitalBounds + goal constructors — IMPLEMENTED
├── optimizer/            # scipy optimization loop (not yet implemented)
├── awareness/            # TLE fetching + conjunction checks
│   ├── tle_fetcher.py    # Space-Track login, catalog pull, 24-hr disk cache
│   └── conjunction.py    # SGP4 catalog vs RK4 mission orbit, CatalogCache
├── output/               # Mission plan composition + plots (not yet implemented)
├── CMakeLists.txt        # C++ build config
├── setup.py              # pybind11 Python extension build
└── .env                  # API keys (not committed)
```

---

## Build Order — Do Not Skip Steps

1. **C++ integrator** — build and validate against analytical solutions first
2. **Situational awareness** — TLE fetcher + conjunction checker
3. **Constraint solver** — mission goal → orbital elements
4. **Optimizer** — scipy outer loop calling integrator inner loop
5. **Agent layer** — Claude API orchestration, last
6. **Output composer** — mission plan formatting

Never wire components together before each is individually validated.

---

## C++ Integrator — Critical Details

- Language: C++17
- Physics: RK4 2-body + J2 perturbation. J2 is required — do not omit it.
- Interface: pybind11, numpy arrays in and out
- The optimizer calls this thousands of times per run. It must be fast.
- Validate before use:
  - Circular orbit period: `T = 2π√(a³/μ)`
  - J2 nodal drift: `dΩ/dt = -3/2 * n * J2 * (R_e/a)² * cos(i) / (1-e²)²`

---

## Key Constants

```cpp
const double MU_EARTH    = 3.986004418e14;  // m^3/s^2
const double R_EARTH     = 6.3781e6;        // m
const double J2          = 1.08263e-3;      // Earth oblateness
const double OMEGA_EARTH = 7.2921150e-5;    // rad/s
```

---

## Architecture Rules — Never Violate

- **C++ integrator propagates the mission orbit.** SGP4 (python-sgp4) propagates the TLE catalog. Do not run 20k catalog objects through RK4. TLE mean elements are defined relative to the SGP4 model — running them through an osculating RK4 integrator gives physically wrong positions.
- **Situational awareness is a hard constraint** in the optimizer, not a post-hoc check. Unsafe orbits are rejected during optimization.
- **Agent layer is last.** All physics and optimization must work before adding the NL interface.
- **State vectors in SI units throughout** — meters, meters/second, seconds. Convert at boundaries only.
- **pybind11 interface stays simple** — numpy arrays in, numpy arrays out. No complex C++ objects crossing the boundary.
- **Optimizer uses `workers=1` with `updating='deferred'`.** Do not use `workers=-1` (scipy multiprocessing) — it conflicts with OpenMP. OpenMP inside `propagate_batch_final` handles parallelism. `updating='deferred'` is required to collect the full population before evaluating, enabling true batch propagation.
- **CatalogCache lives for the lifetime of an optimizer run.** Instantiate once before the optimization loop and pass it to every `check_conjunctions` call. Never create a new CatalogCache inside the fitness function.
- **No PINN surrogate for the integrator.** The RK4 batch propagation (~70 ms/generation) is not the bottleneck — conjunction checking was (~37 s/generation). Approximating the integrator introduces position errors that could produce false-negative conjunction results, which is a safety failure.

---

## Environment Variables (.env)

```
ANTHROPIC_API_KEY=sk-...
SPACE_TRACK_USER=your@email.com
SPACE_TRACK_PASS=yourpassword
```

---

## Conjunction Checker — Performance Architecture

The original conjunction checker looped over T time steps in Python, calling `SatrecArray.sgp4()` once per step. After profiling, this was identified as the optimizer bottleneck: ~37s/generation vs ~70ms for RK4 batch propagation.

**Refactored design (all 7 tests passing):**

1. **`_filter_by_altitude(catalog, target_sma_m, band_m=50_000)`** — TLE string parsing only (no SGP4). Extracts mean motion and eccentricity from line2 columns, computes periapsis/apoapsis, keeps objects whose orbit overlaps the mission altitude band. Reduces N from ~20k to ~2–5k before any propagation.

2. **`_compute_catalog_positions(sats, epoch, n_steps, dt)`** — single `SatrecArray.sgp4(jd_arr, fr_arr)` call with a T-length time array. Returns `(T, N, 3)` position array in meters. Replaces T sequential Python → C extension round trips with one call.

3. **`CatalogCache`** — in-memory dict keyed by `(epoch, duration_s, dt, round(sma, -3))`. One instance per optimizer run. The `round(..., -3)` (nearest 1 km) lets candidate orbits at the same shell share a cache entry. Raises a `ResourceWarning` if the array would exceed 3 GB.

4. **Fully vectorized separation** in `check_conjunctions`: `(T, N, 3) - (T, 1, 3)` broadcast → `np.linalg.norm(axis=2)` → `(T, N)`. No Python loop over time steps.

**Result:** ~37s/generation → ~2s/generation.

---

## Optimizer — Design Decisions

The optimizer is Phase 4 (not yet implemented). Key decisions already settled:

**Algorithm:** `scipy.optimize.differential_evolution`
- Gradient-free — conjunction penalties create hard discontinuities that break gradient-based solvers (SLSQP, L-BFGS-B)
- Population-based — maps directly onto `propagate_batch_final`; the population of ~75 candidates per generation is the batch
- Global search — the safe-orbit landscape has multiple feasible pockets separated by conjunction-blocked corridors

**Search space:** 5 Keplerian elements `[sma, inc, ecc, raan, argp]`
- `OrbitalBounds.as_scipy_bounds()` from the constraint solver feeds directly into scipy's `bounds` argument

**Critical scipy settings:**
```python
differential_evolution(
    batch_fitness_fn,
    bounds=orbital_bounds.as_scipy_bounds(),
    workers=1,          # NOT -1 — OpenMP handles parallelism inside C++
    updating='deferred', # collect full population before evaluating → enables batching
    popsize=15,          # 15 × 5 params = 75 candidates per generation
    maxiter=1000,
)
```

**Batch fitness function pattern:**
```python
def batch_fitness(population):           # shape (N, 5) Keplerian params
    states = keplerian_to_cartesian(population)          # (N, 6) ECI
    finals = oi.propagate_batch_final(states, dt, n_steps)  # (N, 6) — OpenMP inside
    scores = np.full(N, 1e10)
    for i, state in enumerate(states):
        if not check_conjunctions(state, epoch, duration, catalog, catalog_cache=cache):
            scores[i] = mission_objective(finals[i])
    return scores
```

**Performance estimate (8-core machine, 7-day mission, 500 km orbit):**

| Component | Per generation |
|-----------|---------------|
| `propagate_batch_final` (75 candidates, OpenMP) | ~70 ms |
| Conjunction numpy separation (cached catalog) | ~2 s |
| **Total** | **~2 s** |
| 500 generations | **~17 min** |

---

## Current Status

### Phase 1 — C++ Integrator ✓ (4/4 tests passing)
- [x] `integrator/integrator.h` — StateVector, constants, declarations
- [x] `integrator/integrator.cpp` — RK4, 2-body + J2 EOM, propagate functions
- [x] `integrator/bindings.cpp` — pybind11 module (`orbit_integrator`)
- [x] `CMakeLists.txt` + `setup.py` — build system
- [x] `tests/test_integrator.py` — period, energy conservation, J2 drift, batch consistency

### Phase 2 — Situational Awareness ✓ (7/7 tests passing)
- [x] `awareness/tle_fetcher.py` — Space-Track auth, full LEO catalog pull, 24-hr disk cache
- [x] `awareness/conjunction.py` — refactored with `CatalogCache`, altitude pre-filter, vectorized SGP4, fully vectorized numpy separation (no Python loop)
- [x] `tests/test_conjunction.py` — 7 tests: TLE cache TTL, parse structure, separation geometry, pipeline smoke test, empty catalog, cache hit, altitude filter

### Phase 3 — Constraint Solver ✓ (implemented, not fully tested)
- [x] `solver/constraint_solver.py` — `OrbitalBounds` dataclass + goal constructors:
  - `sun_synchronous(altitude_km)` — J2 nodal drift formula to find exact SSO inclination
  - `low_earth_orbit(alt_min, alt_max, inc_min, inc_max)`
  - `polar(altitude_km)`
  - `iss_rendezvous()`
  - `custom(...)`
  - `OrbitalBounds.as_scipy_bounds()` → `[(lo, hi), ...]` for scipy optimizer

### Phase 4 — Optimizer ← next
Design settled (see above). Implement `optimizer/optimizer.py`.

### Phase 5+ — Not yet started
Agent layer, output composer.
