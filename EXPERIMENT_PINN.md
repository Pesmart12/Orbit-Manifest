# Experiment: PINN Surrogate vs RK4 Propagator — Physical Accuracy and Performance

## Experiment Question

Does a neural network trained on RK4 propagation data preserve physical accuracy (energy and angular momentum conservation) as well as the RK4 itself? And does adding physics-informed loss terms during training close any accuracy gap — at what performance trade-off?

---

## Background

The Orbit Manifest optimizer uses a C++ RK4 integrator (2-body + J2) to propagate mission candidates. The `check_post_propagation()` method provides model-agnostic accuracy metrics based on conservation laws — energy drift (J/kg) and angular momentum drift (relative). These metrics apply equally to any propagator that outputs a Cartesian ECI final state, making them a natural evaluation tool for comparing propagators.

The motivation for this experiment is to quantify the accuracy cost of replacing RK4 with a neural surrogate, and to test whether physics-informed training recovers any of that accuracy.

This experiment is **isolated from the production optimizer**. The production pipeline uses naive RK4 throughout.

---

## Propagator Variants

### Variant 1 — Baseline RK4
The current production integrator: `orbit_integrator.propagate_single_final(state0, dt=10.0, n_steps)`.
- Fixed timestep, no post-propagation quality check
- ~70 ms/generation for a batch of 75 candidates (OpenMP)
- Serves as the ground truth for training data and accuracy reference

### Variant 2 — Adaptive RK4
Same integrator, but with `check_post_propagation` as a quality gate after each propagation:
- If energy or momentum drift exceeds threshold: re-propagate at `dt/2`, `n_steps*2` (same total time)
- Maximum 2 halvings (dt → dt/2 → dt/4) before hard failure
- Does **not** improve the underlying physics model — only ensures the result satisfies the conservation thresholds before it is used
- Velocity-plausibility failure (bad initial state, not a numerics issue) is a hard reject with no retry

### Variant 3 — Data-Driven PINN (Variant A)
MLP trained on baseline RK4 final states. Pure data loss.
- Input: normalized (6,) initial ECI state + scalar total propagation time T
- Output: (6,) predicted final ECI state
- Loss: MSE(predicted, RK4_final)
- No explicit conservation constraints during training

### Variant 4 — Physics-Informed PINN (Variant B)
Same architecture as Variant A, with additional physics soft constraints in the loss:
- Loss: MSE(predicted, RK4_final) + λ₁·|E_pred − E₀|² + λ₂·||h_pred| − |h₀||²
- E and h computed using `specific_energy()` and cross-product formula — the same formulas used in the evaluation
- λ₁, λ₂ are tunable; start at 1e-3 and sweep

---

## Evaluation Metrics

All variants evaluated using `check_post_propagation(s0, sf)`:

| Metric | Description | Unit |
|--------|-------------|------|
| Energy drift | `|E_final − E_initial|` | J/kg |
| Momentum drift | `|h_final − h_initial| / h_initial` | dimensionless |
| Pass rate | % of test orbits that pass all four post-propagation checks | % |
| Position error | `‖r_pred − r_rk4‖` at terminal state | m |
| Velocity error | `‖v_pred − v_rk4‖` at terminal state | m/s |
| Inference time | Wall-clock time per orbit (single propagation call) | ms |

Position and velocity error use the baseline RK4 result as ground truth.

---

## Experimental Design

### Training Data Generation

Generate a dataset of (initial_state, T, final_state) tuples from baseline RK4:

- **Initial state sampling**: Draw uniformly from the LEO orbital bounds defined by `low_earth_orbit()` in `solver/constraint_solver.py`:
  - sma ∈ [R_EARTH + 200 km, R_EARTH + 2000 km]
  - ecc ∈ [0.0, 0.3] — extended beyond the typical optimizer range to stress-test the PINN
  - inc ∈ [0°, 180°]
  - raan, argp ∈ [0°, 360°]
- Convert to Cartesian ECI with `keplerian_to_cartesian`
- **Time horizons**: T ∈ {1 orbital period, 6 hours, 1 day, 3 days, 7 days}
- **Dataset size**: 10,000 initial states × 5 time horizons = 50,000 samples
- 80/10/10 train/val/test split, stratified by T

### PINN Architecture

```
Input: [x, y, z, vx, vy, vz, T]  (7 features, normalized)
  → Linear(7, 256) + GELU
  → Linear(256, 256) + GELU
  → Linear(256, 256) + GELU
  → Linear(256, 6)               (final state prediction)
```

- Normalize inputs: position by R_EARTH, velocity by v_circ at R_EARTH + 500 km, T by 7 days
- Train with Adam, lr=1e-3, cosine decay, batch size 512, up to 200 epochs
- Train Variant A and Variant B separately; same architecture and hyperparameters, different loss

### Evaluation Protocol

1. For each of the 5,000 test orbits (held out from training):
   - Propagate with all four variants
   - Evaluate `check_post_propagation(s0, sf)` for each
   - Record energy drift, momentum drift, position error, velocity error, inference time

2. Aggregate per variant and per time horizon T:
   - Mean and 95th-percentile of each accuracy metric
   - Pass rate (% passing all four checks)
   - Mean inference time

3. Plot accuracy vs. time horizon to characterize how error scales with T for each variant

### Adaptive RK4 Retry Logging

For Variant 2, log per orbit:
- `n_halvings` used (0, 1, or 2)
- Which check triggered the retry (energy, momentum, or earth clearance)
- Energy/momentum drift before and after retry

This produces a map of which orbit types (altitude, eccentricity, time horizon) cause the RK4 to degrade at dt=10s — useful for calibrating the thresholds and understanding when finer resolution actually matters.

---

## Hypotheses

1. **PINN Variant A (data-only) will show significantly higher energy drift than RK4**, especially at T = 7 days. Conservation is not enforced by training, so the network can produce final states that violate it.

2. **PINN Variant B (physics-informed) will partially close the accuracy gap**, but will not fully match RK4 because physics soft constraints with finite λ are not exact conservation.

3. **Adaptive RK4 retry will almost never trigger for circular orbits at dt=10s** — the existing test suite shows < 1 J/kg energy drift over 10 orbits, well under the 10 J/kg threshold. Retries will cluster at high eccentricity and long T.

4. **PINN inference will be faster per orbit than RK4 single-threaded**, but the batch RK4 (OpenMP) throughput will likely still beat a batched PINN for the optimizer's use case (75 candidates at once).

---

## Implementation Plan

### Phase E1 — Infrastructure
- [ ] `experiments/pinn/data_gen.py` — draw orbital samples, propagate with baseline RK4, save to `experiments/data/rk4_dataset.npz`
- [ ] `experiments/pinn/utils.py` — `keplerian_to_cartesian`, normalization helpers, `specific_energy` import from `physics/post_propagation.py`

### Phase E2 — Adaptive RK4 (Variant 2)
- [ ] `solver/propagation.py` — `PropagationResult` namedtuple, `propagate_validated()` function (see PLANNING.md Future Experiments)
- [ ] `experiments/pinn/eval_rk4_adaptive.py` — run Variant 2 on test set, log retry events, save results

### Phase E3 — PINN Training
- [ ] `experiments/pinn/model.py` — MLP architecture, physics loss terms
- [ ] `experiments/pinn/train.py` — training loop for Variant A and Variant B
- [ ] `experiments/pinn/eval_pinn.py` — run trained models on test set, save results

### Phase E4 — Analysis
- [ ] `experiments/pinn/analyze.py` — load all result files, compute aggregate metrics, generate plots:
  - Accuracy vs. time horizon (energy drift, momentum drift, position error) per variant
  - Pass rate bar chart per variant and T
  - Inference time comparison
  - Retry event map (altitude vs. eccentricity heatmap, colored by n_halvings)

---

## Dependencies

Add to `requirements.txt` (experiment only, not required for production):
```
torch          # PINN training and inference
matplotlib     # result plots (may already be in environment)
```

---

## Key Design Constraints

- `check_post_propagation` is used **only as an evaluator** in this experiment — not to gate training or alter PINN outputs at inference time.
- The PINN is never wired into conjunction checking. Position errors from a neural surrogate could produce false-negative conjunction results, which is a safety failure.
- The production optimizer is untouched by this experiment. All experiment code lives under `experiments/`.
- The `specific_energy()` formula used in PINN physics loss must be identical to the one in `physics/post_propagation.py` — import it directly rather than re-implementing.

---

## Post-Propagation Checker — Full Implementation

Lives at `physics/post_propagation.py`. Import directly in experiment code:

```python
from solver.post_propagation import check_post_propagation, specific_energy
```

```python
"""
Post-propagation integrity checks: energy and angular momentum conservation.

These functions are kept separate from the pre-propagation screen
(physics/pre_propagation.py) because they serve a different purpose:
the pre-propagation check is a cheap O(1) gate used in the production
optimizer; the post-propagation check is a physics accuracy evaluator
used in experiments comparing propagator variants.

Constants mirror those in physics/pre_propagation.py and
integrator/integrator.h. If you change them, change them everywhere.
"""
from __future__ import annotations

import numpy as np

MU_EARTH = 3.986004418e14   # m^3/s^2
R_EARTH  = 6.3781e6         # m
J2       = 1.08263e-3       # dimensionless


def specific_energy(s: np.ndarray) -> float:
    """Specific mechanical energy including the J2 potential correction.

    E = v²/2 - μ/r + μ·J₂·Re²/(2r³)·(3z²/r² − 1)

    Identical to the formula in tests/test_integrator.py::test_energy_conservation.
    J2 is a conservative potential, so this quantity is an invariant of the
    2-body + J2 model and should be conserved across propagation.

    Args:
        s: Cartesian ECI state, shape (6,) — [x, y, z, vx, vy, vz] in m and m/s.

    Returns:
        Specific mechanical energy in J/kg.
    """
    x, y, z = s[0], s[1], s[2]
    v2 = s[3]**2 + s[4]**2 + s[5]**2
    r  = np.sqrt(x**2 + y**2 + z**2)
    e_2body = 0.5 * v2 - MU_EARTH / r
    e_j2    = MU_EARTH * J2 * R_EARTH**2 / (2.0 * r**3) * (3.0 * z**2 / r**2 - 1.0)
    return e_2body + e_j2


def check_post_propagation(
    s0: np.ndarray,
    sf: np.ndarray,
    energy_tol: float = 10.0,
    momentum_tol_rel: float = 1e-4,
) -> tuple[bool, list[str]]:
    """Verify conservation-law integrity after propagation.

    Compares the initial state s0 against the final state sf. Both must be
    Cartesian ECI in SI units: [x, y, z, vx, vy, vz] in metres and m/s.

    Four checks are performed:

    1. Circular velocity plausibility on s0 — |v₀| must lie within
       [0.5, 1.5] × v_circ at the initial position. Catches unit errors
       or degenerate keplerian_to_cartesian conversions. This check fires
       on a bad initial state, not on propagation quality — a retry with
       smaller dt cannot fix it.

    2. Earth clearance of the final state — r_f > R_EARTH + 80 km.
       Guards against numerical blow-up in the absence of drag.

    3. Specific mechanical energy (2-body + J2 correction) is conserved
       to within energy_tol J/kg. J2 is a conservative potential so
       total energy is an invariant of the model.

    4. Specific angular momentum magnitude is conserved to within
       momentum_tol_rel (relative). J2 is axially symmetric, so
       |h| = √(μa(1−e²)) is preserved at first order.

    Args:
        s0:               Initial Cartesian ECI state, shape (6,).
        sf:               Final Cartesian ECI state after propagation, shape (6,).
        energy_tol:       Max allowed energy drift in J/kg (default 10.0).
        momentum_tol_rel: Max allowed relative angular momentum drift (default 1e-4).

    Returns:
        (ok, reasons) — ok is True iff all checks pass; reasons is a list of
        human-readable failure descriptions (empty when ok=True).
    """
    reasons: list[str] = []

    r0 = np.linalg.norm(s0[:3])
    rf = np.linalg.norm(sf[:3])
    v0 = np.linalg.norm(s0[3:])

    # 1. Circular velocity plausibility (checks s0, the converted state)
    v_circ = np.sqrt(MU_EARTH / r0)
    ratio  = v0 / v_circ
    if not (0.5 <= ratio <= 1.5):
        reasons.append(
            f"|v₀|/v_circ = {ratio:.3f} outside [0.5, 1.5] "
            "(likely unit error or degenerate keplerian_to_cartesian output)"
        )

    # 2. Earth clearance of final state
    if rf < R_EARTH + 80e3:
        reasons.append(
            f"final radius {(rf - R_EARTH) / 1e3:.1f} km below 80 km floor "
            "(numerical blow-up with no drag model)"
        )

    # 3. Specific mechanical energy conservation
    E0 = specific_energy(s0)
    Ef = specific_energy(sf)
    dE = abs(Ef - E0)
    if dE > energy_tol:
        reasons.append(
            f"energy drift {dE:.2f} J/kg > {energy_tol} J/kg threshold"
        )

    # 4. Specific angular momentum magnitude conservation
    h0 = np.linalg.norm(np.cross(s0[:3], s0[3:]))
    hf = np.linalg.norm(np.cross(sf[:3], sf[3:]))
    if h0 > 0.0:
        dh_rel = abs(hf - h0) / h0
        if dh_rel > momentum_tol_rel:
            reasons.append(
                f"angular momentum drift {dh_rel:.2e} > "
                f"{momentum_tol_rel:.1e} threshold"
            )

    return (not reasons), reasons
```
