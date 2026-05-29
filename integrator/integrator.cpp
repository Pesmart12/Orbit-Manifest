#include "integrator.h"
#include <cmath>
#include <omp.h>

// Returns d/dt [x, y, z, vx, vy, vz] under 2-body + J2 gravity.
// All units SI: positions in metres, velocities in m/s, output in m/s and m/s².
StateVector compute_derivatives(const StateVector& s) {
    const double x  = s[0], y  = s[1], z  = s[2];
    const double vx = s[3], vy = s[4], vz = s[5];

    const double r_sq  = x*x + y*y + z*z;
    const double r     = std::sqrt(r_sq);
    const double r3_inv = 1.0 / (r * r_sq);   // 1/r³ — shared by 2-body and J2

    // 2-body: a = -μ/r³ * r_vec
    const double ax_2b = -MU_EARTH * x * r3_inv;
    const double ay_2b = -MU_EARTH * y * r3_inv;
    const double az_2b = -MU_EARTH * z * r3_inv;

    // J2 perturbation — gradient of the degree-2 zonal geopotential term.
    // Derivation: U_J2 = (μ*J2*Re²/r³) * (3/2*(z/r)² - 1/2)
    // Taking -∇U_J2 gives acceleration components below.
    // C = (3/2) * J2 * μ * Re² / r⁵  — common scalar factor
    const double r5_inv = r3_inv / r_sq;
    const double C  = 1.5 * J2 * MU_EARTH * R_EARTH * R_EARTH * r5_inv;
    const double fz = 5.0 * z * z / r_sq;     // 5(z/r)², appears in all three components

    // x,y acceleration: C * coord * (5(z/r)² - 1)
    // z acceleration:   C * z     * (5(z/r)² - 3)   ← -3, not -1, because J2 breaks
    //                                                   symmetry along the polar axis
    const double ax_J2 = C * x * (fz - 1.0);
    const double ay_J2 = C * y * (fz - 1.0);
    const double az_J2 = C * z * (fz - 3.0);

    // Derivative vector: [v, a_total]
    return { vx, vy, vz,
             ax_2b + ax_J2,
             ay_2b + ay_J2,
             az_2b + az_J2 };
}

// Advances state s by one time step dt using the classical 4th-order Runge-Kutta scheme.
// RK4 weights: (k1 + 2k2 + 2k3 + k4) / 6  — equivalent to Simpson's rule quadrature.
// k2 and k3 sample the slope at the midpoint (h/2), k4 at the full step.
// Local truncation error is O(dt⁵); global error O(dt⁴).
StateVector rk4_step(const StateVector& s, double dt) {
    const double h = dt;
    const double h2 = h * 0.5;
    const double h6 = h / 6.0;

    // Lambda adds two state vectors with a scalar multiplier: a + scale*b.
    // Kept local to avoid polluting the namespace with a one-use helper.
    auto add = [](const StateVector& a, const StateVector& b, double scale) -> StateVector {
        return { a[0] + scale*b[0], a[1] + scale*b[1], a[2] + scale*b[2],
                 a[3] + scale*b[3], a[4] + scale*b[4], a[5] + scale*b[5] };
    };

    const StateVector k1 = compute_derivatives(s);               // slope at t
    const StateVector k2 = compute_derivatives(add(s, k1, h2));  // slope at t + h/2 (using k1)
    const StateVector k3 = compute_derivatives(add(s, k2, h2));  // slope at t + h/2 (using k2)
    const StateVector k4 = compute_derivatives(add(s, k3, h));   // slope at t + h   (using k3)

    return {
        s[0] + h6 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]),
        s[1] + h6 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]),
        s[2] + h6 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]),
        s[3] + h6 * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]),
        s[4] + h6 * (k1[4] + 2.0*k2[4] + 2.0*k3[4] + k4[4]),
        s[5] + h6 * (k1[5] + 2.0*k2[5] + 2.0*k3[5] + k4[5]),
    };
}

// Returns only the terminal state after n_steps steps.
// Avoids allocating an (n_steps+1)-length trajectory vector, which matters
// because the optimizer calls this thousands of times per run.
StateVector propagate_final(const StateVector& s0, double dt, int n_steps) {
    StateVector s = s0;
    for (int i = 0; i < n_steps; ++i) {
        s = rk4_step(s, dt);
    }
    return s;
}

// Returns the full trajectory including the initial state at index 0.
// Use this for conjunction checking and visualisation; use propagate_final
// when only the endpoint is needed (saves memory and copy overhead).
std::vector<StateVector> propagate(const StateVector& s0, double dt, int n_steps) {
    std::vector<StateVector> traj;
    traj.reserve(n_steps + 1);
    traj.push_back(s0);
    StateVector s = s0;
    for (int i = 0; i < n_steps; ++i) {
        s = rk4_step(s, dt);
        traj.push_back(s);
    }
    return traj;
}

// Propagates N independent orbits in parallel using OpenMP.
// schedule(static) is appropriate here because every orbit runs for exactly
// n_steps steps — equal work per thread, no need for dynamic load balancing.
std::vector<StateVector> propagate_batch_final(
    const std::vector<StateVector>& states, double dt, int n_steps)
{
    std::vector<StateVector> results(states.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)states.size(); ++i) {
        results[i] = propagate_final(states[i], dt, n_steps);
    }
    return results;
}
