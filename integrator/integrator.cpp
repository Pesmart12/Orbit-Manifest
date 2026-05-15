#include "integrator.h"
#include <cmath>

StateVector compute_derivatives(const StateVector& s) {
    const double x  = s[0], y  = s[1], z  = s[2];
    const double vx = s[3], vy = s[4], vz = s[5];

    const double r_sq  = x*x + y*y + z*z;
    const double r     = std::sqrt(r_sq);
    const double r3_inv = 1.0 / (r * r_sq);

    // 2-body acceleration
    const double ax_2b = -MU_EARTH * x * r3_inv;
    const double ay_2b = -MU_EARTH * y * r3_inv;
    const double az_2b = -MU_EARTH * z * r3_inv;

    // J2 perturbation
    // C = (3/2) * J2 * mu * Re^2 / r^5
    const double r5_inv = r3_inv / r_sq;
    const double C  = 1.5 * J2 * MU_EARTH * R_EARTH * R_EARTH * r5_inv;
    const double fz = 5.0 * z * z / r_sq;

    // x,y components: C * coord * (fz - 1)
    // z component:    C * z     * (fz - 3)  ← asymmetric, intentional
    const double ax_J2 = C * x * (fz - 1.0);
    const double ay_J2 = C * y * (fz - 1.0);
    const double az_J2 = C * z * (fz - 3.0);

    return { vx, vy, vz,
             ax_2b + ax_J2,
             ay_2b + ay_J2,
             az_2b + az_J2 };
}

StateVector rk4_step(const StateVector& s, double dt) {
    const double h = dt;
    const double h2 = h * 0.5;
    const double h6 = h / 6.0;

    auto add = [](const StateVector& a, const StateVector& b, double scale) -> StateVector {
        return { a[0] + scale*b[0], a[1] + scale*b[1], a[2] + scale*b[2],
                 a[3] + scale*b[3], a[4] + scale*b[4], a[5] + scale*b[5] };
    };

    const StateVector k1 = compute_derivatives(s);
    const StateVector k2 = compute_derivatives(add(s, k1, h2));
    const StateVector k3 = compute_derivatives(add(s, k2, h2));
    const StateVector k4 = compute_derivatives(add(s, k3, h));

    return {
        s[0] + h6 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]),
        s[1] + h6 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]),
        s[2] + h6 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]),
        s[3] + h6 * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]),
        s[4] + h6 * (k1[4] + 2.0*k2[4] + 2.0*k3[4] + k4[4]),
        s[5] + h6 * (k1[5] + 2.0*k2[5] + 2.0*k3[5] + k4[5]),
    };
}

StateVector propagate_final(const StateVector& s0, double dt, int n_steps) {
    StateVector s = s0;
    for (int i = 0; i < n_steps; ++i)
        s = rk4_step(s, dt);
    return s;
}

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

std::vector<StateVector> propagate_batch_final(
    const std::vector<StateVector>& states, double dt, int n_steps)
{
    std::vector<StateVector> results(states.size());
    for (std::size_t i = 0; i < states.size(); ++i)
        results[i] = propagate_final(states[i], dt, n_steps);
    return results;
}
