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
│   └── test_integrator.py  # Validation: period, energy, J2 drift, batch
├── agent/                # Claude API orchestration (not yet implemented)
├── solver/               # NL goal → orbital constraints (not yet implemented)
├── optimizer/            # scipy optimization loop (not yet implemented)
├── awareness/            # TLE fetching + conjunction checks (not yet implemented)
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

- **C++ integrator propagates the mission orbit.** SGP4 (python-sgp4) propagates the TLE catalog. Do not run 20k catalog objects through RK4.
- **Situational awareness is a hard constraint** in the optimizer, not a post-hoc check. Unsafe orbits are rejected during optimization.
- **Agent layer is last.** All physics and optimization must work before adding the NL interface.
- **State vectors in SI units throughout** — meters, meters/second, seconds. Convert at boundaries only.
- **pybind11 interface stays simple** — numpy arrays in, numpy arrays out. No complex C++ objects crossing the boundary.

---

## Environment Variables (.env)

```
ANTHROPIC_API_KEY=sk-...
SPACE_TRACK_USER=your@email.com
SPACE_TRACK_PASS=yourpassword
```

---

## Current Status

### Phase 1 — C++ Integrator
- [x] `integrator/integrator.h` — StateVector, constants, declarations
- [x] `integrator/integrator.cpp` — RK4, 2-body + J2 EOM, propagate functions
- [x] `integrator/bindings.cpp` — pybind11 module (`orbit_integrator`)
- [x] `CMakeLists.txt` + `setup.py` — build system
- [x] `tests/test_integrator.py` — 4 validation tests written
- [ ] **Build and run tests** — `pip install -e .` then `pytest tests/` ← next step

### Phase 2+ — Not yet started
Situational awareness, constraint solver, optimizer, agent layer, output composer.
