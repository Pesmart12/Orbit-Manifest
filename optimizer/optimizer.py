"""
Orbital optimizer — Phase 4.

Wraps scipy.optimize.differential_evolution around the C++ RK4 batch
propagator and the conjunction checker to find the safest orbit within
the bounds supplied by the constraint solver.

Parameter vector order (matches OrbitalBounds.as_scipy_bounds):
    [sma (m), inc (rad), ecc, raan (rad), argp (rad)]

True anomaly is fixed at 0 for all candidates (orbit starts at perigee).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

import orbit_integrator as oi
from awareness.conjunction import CatalogCache, check_conjunctions
from solver.constraint_solver import OrbitalBounds

MU_EARTH = 3.986004418e14  # m³/s²


# ---------------------------------------------------------------------------
# Orbital mechanics helpers
# ---------------------------------------------------------------------------

def keplerian_to_cartesian(elements: np.ndarray) -> np.ndarray:
    """Convert an (N, 5) array of Keplerian elements to (N, 6) ECI Cartesian.

    Element order: [sma (m), inc (rad), ecc, raan (rad), argp (rad)].
    True anomaly is fixed at 0 (orbit starts at perigee).

    Returns shape (N, 6): [x, y, z, vx, vy, vz] in m / m·s⁻¹.
    """
    elements = np.atleast_2d(elements)
    N = len(elements)
    states = np.empty((N, 6))

    sma  = elements[:, 0]
    inc  = elements[:, 1]
    ecc  = elements[:, 2]
    raan = elements[:, 3]
    argp = elements[:, 4]
    nu   = np.zeros(N)  # true anomaly = 0 → start at perigee

    p     = sma * (1.0 - ecc ** 2)
    r_mag = p / (1.0 + ecc * np.cos(nu))

    # Perifocal frame position and velocity
    r_peri = np.column_stack([r_mag * np.cos(nu),  r_mag * np.sin(nu),  np.zeros(N)])
    v_mag  = np.sqrt(MU_EARTH / p)
    v_peri = np.column_stack([-v_mag * np.sin(nu), v_mag * (ecc + np.cos(nu)), np.zeros(N)])

    # Rotation matrices from perifocal to ECI (vectorised per element)
    co, so = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc),  np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)

    # Row-by-row matrix-vector product (equivalent to R @ r_peri for each row)
    # R columns: [c0*cw - so*sw*ci, -co*sw - so*cw*ci, so*si]
    #            [so*cw + co*sw*ci, -so*sw + co*cw*ci, -co*si]
    #            [sw*si,             cw*si,              ci   ]
    R00 =  co*cw - so*sw*ci;  R01 = -co*sw - so*cw*ci;  R02 = so*si
    R10 =  so*cw + co*sw*ci;  R11 = -so*sw + co*cw*ci;  R12 = -co*si
    R20 =  sw*si;              R21 =  cw*si;              R22 = ci

    states[:, 0] = R00*r_peri[:,0] + R01*r_peri[:,1] + R02*r_peri[:,2]
    states[:, 1] = R10*r_peri[:,0] + R11*r_peri[:,1] + R12*r_peri[:,2]
    states[:, 2] = R20*r_peri[:,0] + R21*r_peri[:,1] + R22*r_peri[:,2]
    states[:, 3] = R00*v_peri[:,0] + R01*v_peri[:,1] + R02*v_peri[:,2]
    states[:, 4] = R10*v_peri[:,0] + R11*v_peri[:,1] + R12*v_peri[:,2]
    states[:, 5] = R20*v_peri[:,0] + R21*v_peri[:,1] + R22*v_peri[:,2]

    return states


def _mission_objective(final_state: np.ndarray) -> float:
    """Score a propagated terminal state. Lower is better.

    Objective: minimise orbital energy deviation from the initial state.
    A stable, near-circular orbit has low specific orbital energy variation.
    We use the vis-viva energy of the final state as the cost — this drives
    the optimizer toward orbits whose energy is preserved over the mission
    (i.e., well-behaved, non-decaying trajectories).

    Specifically: penalise eccentricity of the final state.
    e = sqrt(1 + 2*E*h² / mu²)  where E = specific energy, h = specific angular momentum.
    Lower eccentricity → better orbit.
    """
    r_vec = final_state[:3]
    v_vec = final_state[3:]
    r = np.linalg.norm(r_vec)
    v2 = np.dot(v_vec, v_vec)
    E  = 0.5 * v2 - MU_EARTH / r          # specific orbital energy (negative for bound orbit)
    h  = np.cross(r_vec, v_vec)
    h2 = np.dot(h, h)
    # Eccentricity magnitude from orbital energy and angular momentum
    ecc_sq = 1.0 + 2.0 * E * h2 / MU_EARTH ** 2
    ecc_sq = max(ecc_sq, 0.0)  # numerical noise can give tiny negatives for near-circular
    return float(np.sqrt(ecc_sq))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Outcome of a single optimizer run."""
    success: bool
    elements: np.ndarray          # best [sma, inc, ecc, raan, argp]
    state: np.ndarray             # ECI Cartesian of best candidate (6,)
    objective: float              # best objective value (lower = better)
    conjunctions_checked: int     # total conjunction calls made
    n_generations: int            # number of DE generations completed
    elapsed_s: float              # wall-clock time
    message: str = ""
    safe: bool = True             # False if best candidate still has conjunctions


# ---------------------------------------------------------------------------
# Optimizer entry point
# ---------------------------------------------------------------------------

def run_optimizer(
    bounds: OrbitalBounds,
    epoch: datetime,
    duration_s: float,
    catalog: list[tuple[str, str, str]],
    dt: float = 30.0,
    threshold_m: float = 5000.0,
    popsize: int = 15,
    maxiter: int = 1000,
    tol: float = 1e-6,
    seed: int | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
) -> OptimizationResult:
    """Find the safest orbit within the given bounds.

    Runs scipy differential_evolution with a batch fitness function that:
    1. Converts the DE population (Keplerian elements) to ECI states.
    2. Batch-propagates all N candidates with the C++ RK4 integrator.
    3. Rejects candidates with conjunctions (scores them 1e10).
    4. Scores survivors by orbital eccentricity of the terminal state.

    Args:
        bounds:           OrbitalBounds from the constraint solver
        epoch:            UTC datetime for mission start
        duration_s:       mission duration in seconds
        catalog:          TLE catalog [(name, line1, line2), ...]
        dt:               integration time step (seconds)
        threshold_m:      conjunction distance threshold (meters)
        popsize:          DE population multiplier (total = popsize × 5)
        maxiter:          maximum DE generations
        tol:              convergence tolerance
        seed:             RNG seed for reproducibility
        progress_callback: optional fn(generation, best_score) called each generation

    Returns:
        OptimizationResult
    """
    epoch = epoch.replace(tzinfo=timezone.utc) if epoch.tzinfo is None else epoch
    n_steps = int(duration_s / dt)

    # One cache for the entire run — catalog SGP4 propagation is done once
    # and reused for every conjunction call.
    catalog_cache = CatalogCache()

    stats = {"generations": 0, "conjunction_calls": 0}
    t0 = time.perf_counter()

    def batch_fitness(population: np.ndarray) -> np.ndarray:
        """Evaluate population from scipy vectorized DE.

        scipy passes shape (n_params, S) when vectorized=True; we need (S, n_params).
        """
        population = population.T   # (n_params, S) → (S, n_params)
        N = len(population)
        scores = np.full(N, 1e10)

        # Convert Keplerian → ECI for all N candidates at once
        states = keplerian_to_cartesian(population)   # (N, 6)

        # Batch propagate — OpenMP inside the C++ module
        finals = np.asarray(oi.propagate_batch_final(states, dt, n_steps))  # (N, 6)

        for i in range(N):
            stats["conjunction_calls"] += 1
            hits = check_conjunctions(
                states[i], epoch, duration_s, catalog,
                dt=dt, threshold_m=threshold_m,
                catalog_cache=catalog_cache,
            )
            if hits:
                # Hard rejection — unsafe orbit
                continue
            scores[i] = _mission_objective(finals[i])

        stats["generations"] += 1
        if progress_callback is not None:
            best = scores[scores < 1e10]
            best_val = float(best.min()) if len(best) > 0 else 1e10
            progress_callback(stats["generations"], best_val)

        return scores

    scipy_bounds = bounds.as_scipy_bounds()

    result = differential_evolution(
        batch_fitness,
        bounds=scipy_bounds,
        workers=1,           # OpenMP handles parallelism inside C++; don't fight it
        updating="deferred", # collect full population before evaluating → enables batching
        popsize=popsize,
        maxiter=maxiter,
        tol=tol,
        seed=seed,
        vectorized=True,     # pass full (N, 5) population matrix to batch_fitness
        polish=False,        # no gradient polish — conjunction landscape is discontinuous
    )

    elapsed = time.perf_counter() - t0
    best_elements = result.x
    best_state    = keplerian_to_cartesian(best_elements)[0]

    # Final conjunction check on the winner
    final_hits = check_conjunctions(
        best_state, epoch, duration_s, catalog,
        dt=dt, threshold_m=threshold_m,
        catalog_cache=catalog_cache,
    )

    return OptimizationResult(
        success=result.success,
        elements=best_elements,
        state=best_state,
        objective=float(result.fun),
        conjunctions_checked=stats["conjunction_calls"],
        n_generations=stats["generations"],
        elapsed_s=elapsed,
        message=result.message,
        safe=len(final_hits) == 0,
    )
