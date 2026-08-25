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
python run.py "7-day sun-synchronous Earth observation at 550 km"             # full run (~12 h at 550 km — see Performance)
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
- **Optimizer uses `workers=1` with `updating='deferred'` and `vectorized=True`.** scipy overrides `vectorized` whenever `workers != 1`, so `workers=1` is a requirement of the vectorized contract, not merely a way to avoid fighting OpenMP. `updating='deferred'` is required to collect the full population before evaluating.
- **The optimizer no longer batch-propagates.** Since the objective became conjunction margin, each candidate needs one mission propagation, and `nearest_approach` does it internally. The old loop propagated every candidate twice — once through `propagate_batch_final` for the eccentricity objective, once again inside `check_conjunctions`. `propagate_batch_final` remains part of the C++ API and is still validated by `tests/test_integrator.py`; feeding batched trajectories into the conjunction layer to restore OpenMP batching is an open optimization (see Roadmap).
- **CatalogCache lives for the lifetime of an optimizer run.** Instantiate once before the optimization loop and pass it to every `check_conjunctions` call. Never create a new CatalogCache inside the fitness function.
- **No PINN surrogate for the integrator.** Propagation is not the bottleneck — measured at 7.3 ms/generation batched against ~37 s for conjunction separation (see Performance). Approximating the integrator introduces position errors that could produce false-negative conjunction results, which is a safety failure.

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

**Result:** the catalog SGP4 propagation is now done once per altitude bucket instead of per candidate per generation. Note this removed *catalog propagation* from the hot path, not the per-candidate separation broadcast, which is unchanged and still dominates — see Performance.

---

## Optimizer — Design Decisions

**Algorithm:** `scipy.optimize.differential_evolution`
- Gradient-free — conjunction penalties create hard discontinuities that break gradient-based solvers (SLSQP, L-BFGS-B)
- Population-based — a generation of ~75 candidates is evaluated together via `vectorized=True`
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

**Objective: conjunction margin.** `_mission_objective` returns the negated
closest-approach distance in metres, so minimising the score maximises clearance
between the mission orbit and the nearest catalog object.

It was terminal eccentricity until measurements showed that was meaningless. A
circular LEO orbit under J2 develops short-period eccentricity oscillation of
amplitude ~1e-3 — the same order as the entire `[0, ecc_max]` search range — so
the score tracked the phase of that oscillation at an arbitrary stopping time.
Measured on a 550 km SSO, the initial eccentricity that scored "best" moved with
mission duration (0.0008 at 12 h, 0.0010 at 2 d, 0.0 at 5 d, 0.0002 at 10 d),
`argp`/`sma` phase outweighed the decision variable 2.1x, and `raan` — the
parameter that decides which traffic you fly through — had exactly zero
influence. Eccentricity was already bounded by `ecc_max` in the goal
constructors, so constraining it twice bought nothing.

Margin makes `sma`, `raan` and `argp` real decisions and grades safety rather
than gating it: 5.1 km and 500 km of clearance used to tie. On a synthetic
mid-band object, a plain band-centre orbit clears 7.1 km — nominally "safe" —
while the optimizer finds 13,761 km by moving to a plane where the object sits
on the far side of Earth.

**Batch fitness function pattern:**
```python
def batch_fitness(population):        # scipy hands over (n_params, S) — transpose it
    population = population.T
    scores = np.full(len(population), 1e10)
    keep = [screen.check_pre_propagation(*row, n_steps=n_steps, dt=dt)[0]
            for row in population]                        # O(1), no integrator
    states = keplerian_to_cartesian(population[keep])
    for j, i in enumerate(np.flatnonzero(keep)):
        near = nearest_approach(states[j], epoch, duration, catalog,
                                dt=dt, catalog_cache=cache)   # margin AND gate
        if near is not None and near.min_separation_m < threshold_m:
            continue                                       # unsafe → keep 1e10
        scores[i] = _mission_objective(near)                # -margin, metres
    return scores
```

---

## Performance — measured against the live catalog

Measured 2026-08-25 on the real Space-Track catalog (32,364 objects, EPOCH >
now-30), SSO 550 km, 7-day mission at dt=60 s (T=10,081), on this machine.

**N — the object count surviving the altitude filter — is 4,873.** Every earlier
estimate in this file was wrong because N was guessed. The previously documented
"~2 s/generation, ~17 min" implies N≈110; the real shell is 44x denser than that.

| Component | numpy | CuPy (GPU) | Notes |
|---|---|---|---|
| Catalog SGP4 + upload, per bucket | 10.8 s | 11.1 s | once per bucket, then cached |
| **Separation, per candidate** | **804 ms** | **23.8 ms** | the entire hot path — **33.8x** |
| Propagation, whole population (OpenMP) | 7.3 ms | — | no production caller |
| Cache entry | 1.1 GB RAM | 1.1 GB VRAM | (T x N x 3 x 8 bytes) |

| Run | numpy | CuPy |
|---|---|---|
| `--quick` (popsize=5, maxiter=20) | 20 min | **36 s** |
| full (popsize=15, maxiter=500) | 8.4 h | **~15 min** |

Separation is ~100% of the cost either way. The GPU path keeps float64 — the
4070 throttles FP64 *compute* to 1/64 rate, but this kernel is bandwidth-bound,
so accuracy costs nothing. Both backends produce bit-identical separations;
`tests/test_conjunction.py::test_gpu_and_numpy_agree` asserts it.

**Bucketing costs 39%.** The `CatalogCache` band widening (±25 km → ±50 km, so a
shared bucket entry stays a superset for every candidate in it) takes N from
3,498 to 4,873. That is real but far less than the 2x a uniform-density shell
would predict — LEO object density varies sharply with altitude. `_BUCKET_M` is
tunable: narrowing it cuts N and the per-entry footprint, at the price of more
entries.

**Headroom, in the order worth taking it** (see Roadmap → Conjunction separation
performance): the separation code allocates a 1.1 GB `(T, N, 3)` temporary per
candidate; chunking over time measured 2.1x with identical results, float32 would
roughly halve bandwidth again, and an algorithmic pre-filter that excludes
objects whose orbital planes cannot approach the threshold would cut N directly.
Pedro's preference for the GPU step is CuPy.

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
- [x] `physics/pre_propagation.py` — `OrbitScreen.check_pre_propagation`: perigee floor, apogee ceiling, integration window vs. orbital period. O(1), no integrator call. **Wired into `batch_fitness`**, ahead of any propagation, so degenerate candidates never reach the integrator or the conjunction check. Rejections are counted in `OptimizationResult.screened_out`.
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

## Next Steps — resume here (as of 2026-08-25)

State: 115 tests passing on both backends, working tree clean, everything pushed.
The pipeline has never completed a full 500-generation run; the longest was 14
generations. Ordered by what I would do first.

### 1. Anthropic credits — the only hard blocker
`ANTHROPIC_API_KEY` in `.env` is **valid** (verified against `models.list()`), but
the account has no credits, so `run.py` dies at `_parse_mission` with
`invalid_request_error: credit balance is too low`. Everything downstream works —
the measurement runs bypassed it by supplying the intent dict directly:

```python
intent = {"orbit_type": "sun_synchronous", "altitude_km": 550.0,
          "duration_days": 7.0, "rationale": "..."}
bounds = _intent_to_bounds(intent)          # then run_optimizer(...) as normal
```

One parse call costs ~$0.006, so a minimum top-up covers hundreds of runs.
Space-Track credentials work; note it throttles after a few rapid failures and
returns the same `{"Login":"Failed"}` for throttling as for a wrong password —
if it rejects known-good credentials, wait rather than retrying.

### 2. Tighten the level-2 promotion gate — biggest remaining win
`_refine_gate` bounds relative velocity at two circular velocities (a head-on
retrograde pass). Correct as a worst case, far too generous for an SSO mission
meeting mostly near-polar traffic, and it promotes **63.5%** of the band, making
level 2 the dominant cost (87 ms of 135 ms per candidate).

**Trap:** the obvious triangle-inequality form `|Δcatalog| + |Δmission|` collapses
to exactly the same head-on number and buys nothing. A useful bound needs the
actual *relative* displacement per object, which is already being formed inside
`_closest_approaches` — computing a per-object max there is the cheap place to get
it. Halving promotion roughly halves a run.

### 3. Chase the 23 km worst-case sieve error
Everything else is exact (median and p95 both 0 m against a 1 s reference), so one
outlier at 23 km stands out. Likely a third close approach that level 2's grid
also mis-ranks. Worth identifying before adding machinery: it may be one bad TLE
rather than a structural gap. Reproduce with the dt-ladder comparison in the
session's `sampling_error.py` approach — run `check_conjunctions` at dt=60, 10 and
1 with `threshold_m=1e9` and diff the per-object dictionaries.

### 4. Verify the level-2 budget does not silently degrade a real run
`_FINE_BUDGET_BYTES` is 3 GB. Objects beyond it keep their **level-1** separation,
which can overestimate a close approach — the cache warns once via
`ResourceWarning`. Nobody has watched a long run to see whether it fills. If it
does, either raise the budget or make the eviction preference explicit (currently
first-come). Do not let this fail quietly; it is the one place the sieve can
regress to the behaviour it was built to fix.

### 5. Decide the fate of `propagate_batch` / `propagate_batch_final`
Neither has a production caller. `CMakeLists.txt` carries
`find_package(OpenMP REQUIRED)` and `setup.py` passes `/openmp` for parallel
regions nothing reaches. Either wire batching into the conjunction layer (a ~0.02%
gain — propagation is 7.3 ms against 85+ s of separation) or delete both and drop
the OpenMP build dependency. Leaving it as-is is the one option with no upside.

### 6. Retire `_refine_candidates`
Superseded by `_sieve`, kept opt-in (`refine=True`) for a final verification pass.
It refines around the coarse argmin, which cannot fix a mis-identified event —
the exact limitation the sieve exists to solve. If nothing uses it by the next
pass, delete it.

### Figures are a band, not a constant
Screened count depends on where the optimizer wanders: 4,807 objects at the band
centre, 8,019 in one real run that drifted lower into denser traffic. Quote ranges
when reporting performance.

---

## Roadmap — Specified but Not Built

None of the following exists in the pipeline. Do not describe any of it as implemented.

### Optimizer objectives
The objective is conjunction margin (see Optimizer — Design Decisions). Still unbuilt:
- **Per-goal-type objectives.** The original design took `run_optimizer(..., objective_fn)`;
  the implementation hardcodes one. SSO, polar and rendezvous plausibly want different
  scores — e.g. SSO drift fidelity, `|dΩ/dt − ω_sun|` measured from the propagation, which
  would validate the constraint solver's analytic inclination against the real integrator.
- **Coverage maximization** — needs ground-track → target-region → revisit machinery, and
  the agent extracting a target region from the description.
- **Delta-v minimization.** Noted for completeness, but launch delta-v for a circular LEO
  is essentially analytic in altitude and inclination: it needs neither DE nor propagation,
  and would just pin `sma` to its lower bound.
- **Restore OpenMP batching.** Each candidate now propagates once inside `nearest_approach`.
  Feeding batched trajectories into the conjunction layer would let `propagate_batch` do that
  work in one OpenMP call instead of S serial ones. Deliberately deferred: it is a 0.02%
  gain and would reshape the conjunction API immediately before the separation work
  reshapes it again. Decide it as part of that work, or delete both batch functions and the
  OpenMP build dependency if separation never gets touched.

### Conjunction time resolution — closed by the two-level sieve
Separation is sampled every `dt` and the minimum over samples reported, which
overestimates the true miss distance. The tail was dangerous: measured against a
dt=1 s reference on the live catalog, a 60 s grid reported NORAD 63352's true
4,005 m approach at T+2.897 h as **43,062 m at T+2.102 h** — a *different event*
2,861 s away. The grid was not mis-measuring an approach, it was locking onto the
wrong one, so the optimizer chose orbits against miss distances an order of
magnitude too generous.

`_sieve` fixes it in two levels. Level 1 measures every object on the caller's
grid. Level 2 re-measures the survivors **over the whole window again** at
`dt / _SIEVE_FACTOR` — not around level 1's answer, which is the entire point.
Promotion uses `_refine_gate`'s rigorous `v_rel·dt/2` bound, so nothing that could
reach the gate is dropped. **Measured promotion rate over a 7-day window is 63.5%**
(3,054 of 4,807), not the ~7% a 6-hour window suggested — over a week most objects
drift within the gate at some point, so level 2 is now the dominant cost.

| vs dt=1 s reference | coarse | + parabola | **sieve** |
|---|---|---|---|
| median overestimate | 1,576 m | 2 m | **0 m** |
| p95 | 47,695 m | 1,068 m | **0 m** |
| worst | 331,316 m | 270,204 m | **23,086 m** |
| nearest object | 20,962 m | 14,122 m | **4,006 m** (true 4,005 m) |

Sampling everything at 10 s reaches the same accuracy and needs a 7 GB cache
entry — too large for two buckets in 12 GB of VRAM. Gating first keeps level 2 to
~480 MB. Level-2 positions are cached per object, so the expensive fine SGP4 is
paid once per object across a whole run rather than once per candidate orbit.

**Cost of correctness:** 24 ms -> 135 ms per candidate, so a full run goes from
~15 min to ~85 min, and the cache reaches ~5 GB.

**The GPU is what makes the sieve affordable at all** — measured on the same
7-day case: CuPy 135 ms/candidate (1.4 h per run) against numpy 4,964 ms
(51.7 h). Not an optimisation; a precondition.

The obvious remaining lever is the promotion gate. `_refine_gate` bounds relative
velocity by two circular velocities — a head-on retrograde pass — which is right
for the worst case and far too generous for an SSO mission meeting other
near-polar traffic. A per-object bound would cut the 63.5% promotion rate
directly and is the cheapest large win left. Note the triangle-inequality form
(|Δcatalog| + |Δmission|) reduces to the same head-on figure, so a useful bound
has to use the actual relative motion, not per-body speeds. That is the right trade for a
conjunction checker: the alternative is a fast answer that names the wrong
satellite. Memory is now the binding constraint — a failed GPU upload warns and
degrades to numpy rather than killing the run.

Parabolic sub-sample refinement stays in both levels and costs nothing; squared
distance is exactly quadratic in time for a linear relative pass.
`_refine_candidates` (opt-in, `refine=True`) remains for a final verification
pass, but the sieve supersedes it.

### Conjunction separation performance
Separation is ~100% of a generation (see Performance) and the only worthwhile optimization
target. Agreed order of operations:

1. **Measure first.** A live Space-Track run settles N, which every estimate depends on, and
   shows whether `_BUCKET_M` — which doubled the screening band — is hurting.
2. **Take the free numpy wins.** Chunking over time is 2.1x with identical results; float32
   roughly halves bandwidth again at ~1 m resolution against a 5 km threshold.
3. **Consider an algorithmic pre-filter.** Excluding objects whose orbital planes cannot
   bring them within the threshold, or a coarse-time pass refined near candidates, would cut
   N directly and likely beat any hardware change.
4. **Then GPU.** Pedro's stated preference (2026-08-24) is CUDA via **CuPy**. It is a good
   fit: embarrassingly parallel over objects, bandwidth-bound, and `CatalogCache` already
   holds the catalog array so it uploads once per bucket while each candidate transfers only
   a 242 KB trajectory up and 16 KB of minima back. Build it as an optional import with the
   numpy path as fallback — the project targets teams who may not have an NVIDIA card, and
   keeping both lets them be diffed for correctness. Hardware here: RTX 4070, 12 GB,
   compute 8.9; neither the CUDA toolkit nor CuPy is installed yet.

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
