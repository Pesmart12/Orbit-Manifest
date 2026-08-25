# CLAUDE.md — Orbit Manifest

## What This Project Is
A natural language orbital mission design agent. Users describe a mission goal in plain English; the system produces an optimized orbit with conjunction analysis against the live LEO satellite catalog.

**This file is the single source of truth for architecture, status, and roadmap.** README.md is the outward-facing description (what it does, how to install and run it) and deliberately tracks no status.

---

## Build Commands

All Python runs through conda. `environment.yml` is the single source of
dependencies — it covers runtime, build (`pybind11`, `setuptools`, `cmake`) and
`anthropic` via its own `pip:` section. **Do not `pip install` anything it already
provides**: pip-over-conda is what silently replaced `scipy` with a wheel carrying
its own BLAS, and reinstalling it damaged the conda package underneath.

```bash
# Create environment (first time) — installs everything, build deps included
conda env create -f environment.yml
conda activate orbit-manifest

# Build C++ integrator and install as Python module — do this before anything else
pip install -e .

# Build C++ manually (alternative)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)   # Linux/Mac
# cmake --build . --config Release   # Windows

# Run tests
pytest tests/

# End-to-end run
python run.py "7-day sun-synchronous Earth observation at 550 km" --quick   # fast pipeline check
python run.py "7-day sun-synchronous Earth observation at 550 km"             # full run (~17 min)
```

---

## Repo Layout

```
orbit-manifest/
├── integrator/           # C++ RK4 integrator — BUILD THIS FIRST
│   ├── integrator.h      # StateVector type, constants, declarations
│   ├── integrator.cpp    # RK4 + 2-body + J2 equations of motion
│   └── bindings.cpp      # pybind11 numpy bindings
├── physics/              # Pre/post propagation physics agreement gate
│   ├── pre_propagation.py
│   └── post_propagation.py
├── solver/               # NL goal → orbital constraints
│   └── constraint_solver.py  # OrbitalBounds dataclass + goal constructors
├── awareness/            # TLE fetching + conjunction checks
│   ├── tle_fetcher.py    # Space-Track login, catalog pull, 24-hr disk cache
│   └── conjunction.py    # SGP4 catalog vs RK4 mission orbit, CatalogCache
├── optimizer/            # scipy differential evolution loop
│   └── optimizer.py      # keplerian_to_cartesian, run_optimizer, OptimizationResult
├── agent/                # Claude API orchestration
│   └── agent.py          # _parse_mission, _intent_to_bounds, MissionPlan, plan_mission
├── output/               # Report formatting + ground-track plot
│   └── composer.py       # format_report, plot_ground_track
├── tests/                # One test file per module
├── run.py                # End-to-end CLI entry point
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

3. **`CatalogCache`** — in-memory dict keyed by `(epoch, duration_s, dt, bucket)`, one instance per optimizer run. `bucket` snaps the candidate's radius to a 50 km grid (`_BUCKET_M`), and each bucket's screening band is widened by a full bucket width so a candidate anywhere inside it still has its whole ±25 km window covered. **The shared set is always a superset of what a candidate would screen alone, never a subset** — sharing an entry cannot hide a conjunction. Warns once via `ResourceWarning` when the cache's *total* footprint crosses 3 GB.

   The key was previously `round(sma, -3)` — the nearest kilometre — which produced one multi-hundred-MB array per distinct kilometre of the sma band (~41 for a ±20 km SSO run, ~9 GB on a 7-day mission). The per-entry memory guard never fired, because no single entry was the problem. Bucketing cuts that ~20× at the cost of roughly 2× the objects per entry.

4. **Fully vectorized separation** in `check_conjunctions`: `(T, N, 3) - (T, 1, 3)` broadcast → `np.linalg.norm(axis=2)` → `(T, N)`. No Python loop over time steps.

**Result:** ~37s/generation → ~2s/generation.

---

## Optimizer — Design Decisions

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

**86/86 tests passing.** Counts are measured, not estimated — update them when they change.

### Known environment fault — numpy's MKL BLAS (resolved; keep the pin)

**Symptom:** the interpreter dies — exit 127, no traceback, output truncated
mid-run — on a delay-load failure (`0xC06D007F` = `DELAYLOAD_MODULE_NOT_FOUND`).
It looks like a matplotlib or pytest problem and is neither.

**Cause:** the MKL build conda-forge resolved to on this platform crashes on
matrix multiply. A bare `numpy.arange(9).reshape(3,3) @ numpy.eye(3)` is enough
to kill the process. Everything downstream of a matmul inherited it:

- `test_j2_nodal_drift` → `np.polyfit` → `np.linalg.lstsq` → LAPACK
- `plot_ground_track` → `ax.axhline` → matplotlib composes a *blended* transform
  → matrix multiply. `ax.plot()` survives because it reuses `transData` directly
  and never multiplies.

**Fix:** `environment.yml` pins `libblas=*=*openblas`. Without that pin a fresh
`conda env create` resolves to MKL again and the bug returns. To repair an env
that already has MKL:

```bash
conda install -n orbit-manifest -c conda-forge "libblas=*=*openblas"
```

**Do not use `pytest -p no:faulthandler` to make this go away.** It hides the
report, not the crash. If a suite ever ends with truncated dots and no summary
line, run `python -c "import numpy; print(numpy.eye(3) @ numpy.eye(3))"` first.

Ruled out along the way, none of them the cause: channel mixing, the C++
extension, working-directory DLL shadowing, missing C extensions, the Visual C++
runtime, matplotlib backend choice (svg/pdf/agg all failed), a force-reinstall of
matplotlib/freetype/numpy/scipy, and a full env rebuild. One unrelated defect was
fixed en route: `scipy` had been pip-installed over the conda-forge package.

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

### Phase 3 — Constraint Solver ✓ (17/17 tests passing)
- [x] `solver/constraint_solver.py` — `OrbitalBounds` dataclass + goal constructors:
  - `sun_synchronous(altitude_km)` — J2 nodal drift formula to find exact SSO inclination
  - `low_earth_orbit(alt_min, alt_max, inc_min, inc_max)`
  - `polar(altitude_km)`
  - `iss_rendezvous()`
  - `custom(...)`
  - `OrbitalBounds.as_scipy_bounds()` → `[(lo, hi), ...]` for scipy optimizer
- [x] `tests/test_constraint_solver.py` — 17 tests: bounds validation, SSO inclination vs. known altitudes and drift rate, per-goal-constructor bands, custom roundtrip

### Phase 3b — Physics Agreement Layer ✓ (15/15 tests passing)
- [x] `physics/pre_propagation.py` — `OrbitScreen.check_pre_propagation`: perigee floor, apogee ceiling, integration window vs. orbital period. O(1), no integrator call. **Wired into `batch_fitness`**, ahead of `propagate_batch_final`, so degenerate candidates reach neither the OpenMP kernel nor the conjunction check. Rejections are counted in `OptimizationResult.screened_out`.
- [x] `physics/post_propagation.py` — `check_post_propagation` + `specific_energy` (2-body + J2). **Not wired in** — it is an accuracy evaluator for the experiment in the Roadmap, not a production gate.
- [x] `tests/test_pre_propagation.py` (6) + `tests/test_post_propagation.py` (9)
- With bounds from the goal constructors the screen normally rejects nothing; it earns its place on `custom()` bounds and on missions shorter than one orbital period, where conjunction screening would cover only part of a revolution.

### Phase 4 — Optimizer ✓ (8/8 tests passing)
- [x] `optimizer/optimizer.py` — `keplerian_to_cartesian`, `_mission_objective`, `run_optimizer`
- [x] `tests/test_optimizer.py` — Keplerian conversion, objective scoring, DE with empty/distant catalog, progress callback

### Phase 5 — Agent Layer ✓ (10/10 tests passing)
- [x] `agent/__init__.py`
- [x] `agent/agent.py` — `_parse_mission` (Claude API structured JSON), `_intent_to_bounds`, `MissionPlan`, `plan_mission`
- [x] `tests/test_agent.py` — mocked Claude API: parse extraction, bounds mapping for all 5 orbit types, end-to-end pipeline, progress callback

### Phase 6 — Output Composer ✓ (15/15 tests passing)
- [x] `output/__init__.py`
- [x] `output/composer.py` — `format_report`, `plot_ground_track`, `_eci_to_latlon`, `_gmst_rad`, `_split_at_wraps`
- [x] `tests/test_composer.py` — report content/format (7 tests), GMST, lat/lon geometry, wrap-around splitting, figure output and file save (mocked integrator)

---

## Roadmap — Specified but Not Built

None of the following exists in the pipeline. Do not describe any of it as implemented.

### Optimizer objectives
`_mission_objective` scores **terminal eccentricity only**, and its docstring contradicts
itself (claims energy-deviation, computes eccentricity). For a sun-synchronous run with
`ecc_max=0.001` the search is close to degenerate. Intended objectives:
- Delta-v minimization (launch → operational orbit)
- Coverage maximization
- Time-to-orbit minimization

### Output composer
`format_report` emits elements, one safety boolean, and optimizer stats. Intended:
- **Launch window** — RAAN targeting from a launch site (needs a launch-site coordinate table, ~10 sites)
- **Delta-v budget**
- **Conjunction detail** — `ConjunctionResult` already carries NORAD ID, name, min separation
  and TCA, but `OptimizationResult` keeps only `safe: bool`, so all of it is discarded.
  Reporting `catalog_size` (full catalog) next to the verdict also overstates what was
  screened — only the altitude-filtered subset is ever checked.
- **Claude narrative summary** in plain English

### Constraint solver
- Repeat-groundtrack goals ("pass over the equator N times in D days") → period + RAAN constraint
- Coverage requirement parsing

### Integrator physics
- Atmospheric drag (exponential density model; would need NOAA F10.7 / Kp indices)
- Lunar/solar perturbations (n-body)
- Adaptive step size (RK45)

### Post-propagation integrity work
The pre-propagation screen is wired in (see Phase 3b). `check_post_propagation` is not,
and is the remaining half.
- **Post-propagation integrity experiment:** run `check_post_propagation` across a 7-day
  mission to find where RK4 at dt=10 s violates energy/momentum thresholds; use the results
  to calibrate the tolerances. Expect failures to cluster at high eccentricity and long
  horizons — circular orbits drift <1 J/kg over 10 orbits, well inside the 10 J/kg default.
- **Adaptive-timestep retry:** on an energy or momentum failure, re-propagate at `dt/2` with
  2× `n_steps` (same total time). Cap at two halvings (dt → dt/2 → dt/4), then hard-fail.
  A velocity-plausibility failure is a bad *initial* state, not a numerics problem — hard
  reject it with no retry. Logging which check fired, and at what altitude/eccentricity,
  gives the map needed to calibrate the thresholds.

---

## Decisions Recorded

Deliberate choices that earlier drafts described differently. Don't "fix" the code back
toward the older design without revisiting these.

- **The agent is a single structured-output call, not a tool-using orchestrator.**
  `_parse_mission` makes one `messages.create` call with `output_config.format` and returns
  an intent dict. An earlier design gave Claude five tools (`solve_constraints`,
  `optimize_orbit`, `check_conjunctions`, `propagate_orbit`, `compose_output`) plus an
  interactive clarification flow for underspecified goals. Neither was built. The pipeline
  downstream of parsing is fully deterministic and gains nothing from model-driven
  orchestration.
- **`vectorized=True` is what makes batching work**, not `workers=1`. scipy 1.17 overrides
  `vectorized` whenever `workers != 1`, so `workers=1` is a requirement of the vectorized
  contract — not merely a way to avoid fighting OpenMP. `batch_fitness` therefore receives
  `(n_params, S)` and transposes; the population is `popsize × 5`.
- **No PINN surrogate for the integrator.** Investigated and dropped; the full reasoning is
  in Architecture Rules above. A design doc for the experiment (`EXPERIMENT_PINN.md`) was
  removed in the same pass as `PLANNING.md` — recoverable from git history if the question
  ever reopens. The adaptive-RK4 work it proposed was not PINN-specific and survives in the
  Roadmap above.
