import numpy as np
import pytest

try:
    import orbit_integrator as oi
except ImportError as e:
    raise ImportError(
        "orbit_integrator C++ module not found. Build it first with: pip install -e ."
    ) from e

MU_EARTH = 3.986004418e14
R_EARTH  = 6.3781e6
J2       = 1.08263e-3


def keplerian_to_cartesian(a, e, inc, raan, argp, nu, mu=MU_EARTH):
    """Convert Keplerian orbital elements to ECI Cartesian state [x,y,z,vx,vy,vz]."""
    p = a * (1.0 - e**2)
    r_mag = p / (1.0 + e * np.cos(nu))

    r_peri = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_peri = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    co, so = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc),  np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)

    R = np.array([
        [ co*cw - so*sw*ci, -co*sw - so*cw*ci,  so*si],
        [ so*cw + co*sw*ci, -so*sw + co*cw*ci, -co*si],
        [ sw*si,             cw*si,              ci   ],
    ])

    return np.concatenate([R @ r_peri, R @ v_peri])


def cartesian_to_raan(state):
    """Extract RAAN (rad) from ECI Cartesian state vector."""
    r = state[:3]
    v = state[3:]
    h = np.cross(r, v)
    n = np.cross([0.0, 0.0, 1.0], h)
    return np.arctan2(n[1], n[0])


# ---------------------------------------------------------------------------
# Test 1: Circular orbit period
# ---------------------------------------------------------------------------
def test_circular_orbit_period():
    a = R_EARTH + 500e3
    # J2-adjusted circular velocity for equatorial orbit: v = sqrt(mu/a * (1 + 1.5*J2*(Re/a)^2))
    v_circ = np.sqrt(MU_EARTH / a * (1.0 + 1.5 * J2 * (R_EARTH / a)**2))
    # J2-adjusted period: T = 2*pi*a / v_circ
    T = 2.0 * np.pi * a / v_circ

    state0 = np.array([a, 0.0, 0.0, 0.0, v_circ, 0.0])
    dt = 10.0
    n_steps = int(T / dt)

    final = oi.propagate_single_final(state0, dt, n_steps)

    r_initial = np.linalg.norm(state0[:3])
    r_final   = np.linalg.norm(final[:3])
    assert abs(r_final - r_initial) < 1.0, \
        f"Radius drift {abs(r_final - r_initial):.3f} m exceeds 1 m"

    # Check angular rate: satellite sweeps v_circ/a rad/s in the equatorial plane.
    # Comparing Cartesian position after int(T/dt) steps fails because the step
    # count truncates the period by up to dt seconds (= ~76 km at orbital speed).
    # Angular position is immune to that truncation artifact.
    theta_expected = (v_circ / a) * (n_steps * dt) % (2.0 * np.pi)
    theta_actual   = np.arctan2(final[1], final[0]) % (2.0 * np.pi)
    angle_err = abs(theta_expected - theta_actual)
    angle_err = min(angle_err, 2.0 * np.pi - angle_err)
    arc_err = a * angle_err
    assert arc_err < 100.0, \
        f"Arc position error {arc_err:.1f} m exceeds 100 m after {n_steps} steps"


# ---------------------------------------------------------------------------
# Test 2: Energy conservation (2-body + J2 total mechanical energy)
# ---------------------------------------------------------------------------
def test_energy_conservation():
    # J2 is conservative — total mechanical energy (2-body + J2 potential) must be conserved.
    a = R_EARTH + 500e3
    v_circ = np.sqrt(MU_EARTH / a * (1.0 + 1.5 * J2 * (R_EARTH / a)**2))
    T = 2.0 * np.pi * a / v_circ

    state0 = np.array([a, 0.0, 0.0, 0.0, v_circ, 0.0])
    dt = 10.0
    n_steps = int(10 * T / dt)

    traj = oi.propagate_single(state0, dt, n_steps)

    def energy(s):
        x, y, z = s[0], s[1], s[2]
        v2 = s[3]**2 + s[4]**2 + s[5]**2
        r  = np.sqrt(x**2 + y**2 + z**2)
        e_2body = 0.5 * v2 - MU_EARTH / r
        # J2 potential: Phi_J2 = mu*J2*Re^2 / (2*r^3) * (3*z^2/r^2 - 1)
        e_j2 = MU_EARTH * J2 * R_EARTH**2 / (2.0 * r**3) * (3.0 * z**2 / r**2 - 1.0)
        return e_2body + e_j2

    E0 = energy(traj[0])
    E_max_drift = max(abs(energy(traj[i]) - E0) for i in range(1, len(traj)))

    assert E_max_drift < 1.0, \
        f"Energy drift {E_max_drift:.4f} J/kg exceeds 1 J/kg over 10 orbits"


# ---------------------------------------------------------------------------
# Test 3: J2 nodal drift (RAAN precession)
# ---------------------------------------------------------------------------
def test_j2_nodal_drift():
    a   = R_EARTH + 500e3
    e   = 0.0
    inc = np.radians(97.4)   # sun-synchronous inclination

    n = np.sqrt(MU_EARTH / a**3)
    drift_analytical = -(1.5 * n * J2 * (R_EARTH / a)**2 * np.cos(inc)
                         / (1.0 - e**2)**2)  # rad/s

    state0 = keplerian_to_cartesian(a=a, e=e, inc=inc,
                                    raan=0.0, argp=0.0, nu=0.0)

    duration_days = 5
    dt = 30.0
    n_steps = int(duration_days * 86400 / dt)

    traj = oi.propagate_single(state0, dt, n_steps)

    # Sample RAAN every 100 steps
    stride = 100
    indices = range(0, n_steps + 1, stride)
    raan_series = np.array([cartesian_to_raan(traj[i]) for i in indices])
    time_series  = np.array([i * dt for i in indices])

    # Unwrap to handle 2π wraps
    raan_series = np.unwrap(raan_series)

    drift_measured = np.polyfit(time_series, raan_series, 1)[0]  # rad/s

    drift_err_deg_per_day = abs(drift_measured - drift_analytical) * 86400 * (180.0 / np.pi)

    assert drift_err_deg_per_day < 0.01, (
        f"J2 nodal drift error {drift_err_deg_per_day:.4f} deg/day "
        f"(measured {np.degrees(drift_measured)*86400:.4f}, "
        f"analytical {np.degrees(drift_analytical)*86400:.4f} deg/day)"
    )


# ---------------------------------------------------------------------------
# Test 4: Batch consistency
# ---------------------------------------------------------------------------
def test_batch_consistency():
    a = R_EARTH + 500e3
    v_circ = np.sqrt(MU_EARTH / a)
    state0 = np.array([a, 0.0, 0.0, 0.0, v_circ, 0.0])

    N = 20
    states0 = np.tile(state0, (N, 1))

    dt = 10.0
    n_steps = 1000

    finals_batch  = oi.propagate_batch_final(states0, dt, n_steps)
    final_single  = oi.propagate_single_final(state0, dt, n_steps)

    for i in range(N):
        np.testing.assert_array_almost_equal(
            finals_batch[i], final_single, decimal=10,
            err_msg=f"Batch row {i} differs from single propagation"
        )
