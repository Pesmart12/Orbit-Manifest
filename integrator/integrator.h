#pragma once

#include <array>
#include <vector>

using StateVector = std::array<double, 6>;

constexpr double MU_EARTH    = 3.986004418e14;
constexpr double R_EARTH     = 6.3781e6;
constexpr double J2          = 1.08263e-3;
constexpr double OMEGA_EARTH = 7.2921150e-5;

// Compute [vx, vy, vz, ax, ay, az] from state [x, y, z, vx, vy, vz]
StateVector compute_derivatives(const StateVector& s);

// Single RK4 step
StateVector rk4_step(const StateVector& s, double dt);

// Propagate from s0 — returns full trajectory including initial state (n_steps+1 rows)
std::vector<StateVector> propagate(const StateVector& s0, double dt, int n_steps);

// Propagate from s0 — returns only the terminal state (optimizer fast path)
StateVector propagate_final(const StateVector& s0, double dt, int n_steps);

// Propagate N orbits — returns only terminal states (main optimizer use case)
std::vector<StateVector> propagate_batch_final(
    const std::vector<StateVector>& states, double dt, int n_steps);
