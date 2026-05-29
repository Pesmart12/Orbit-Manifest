#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "integrator.h"

namespace py = pybind11;

// c_style: require C (row-major) memory order so we can safely stride with plain pointer arithmetic.
// forcecast: silently convert integer or float32 arrays rather than raising a type error at the boundary.
using arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Copies a 6-element numpy array into a StateVector.
// pybind11 has no automatic conversion for std::array, so we copy explicitly.
static StateVector arr_to_sv(const arr& a) {
    auto buf = a.request();
    if (buf.size != 6)
        throw std::runtime_error("state vector must have exactly 6 elements");
    const double* p = static_cast<const double*>(buf.ptr);
    return { p[0], p[1], p[2], p[3], p[4], p[5] };
}

// Copies a StateVector into a freshly allocated (6,) numpy array.
static arr sv_to_arr(const StateVector& s) {
    arr out({6});
    double* p = out.mutable_data();
    for (int i = 0; i < 6; ++i) {
        p[i] = s[i];
    }
    return out;
}

// propagate_single: (6,) -> (n_steps+1, 6)
// Full trajectory; index 0 is the initial state.
arr py_propagate_single(const arr& state0, double dt, int n_steps) {
    auto traj = propagate(arr_to_sv(state0), dt, n_steps);
    arr out({ (py::ssize_t)(n_steps + 1), (py::ssize_t)6 });
    double* p = out.mutable_data();
    for (std::size_t i = 0; i < traj.size(); ++i) {
        for (int j = 0; j < 6; ++j) {
            p[i * 6 + j] = traj[i][j];   // row-major: step i, component j
        }
    }
    return out;
}

// propagate_single_final: (6,) -> (6,)
// Returns only the terminal state; cheaper than propagate_single when the
// intermediate trajectory is not needed (e.g., optimizer cost evaluation).
arr py_propagate_single_final(const arr& state0, double dt, int n_steps) {
    return sv_to_arr(propagate_final(arr_to_sv(state0), dt, n_steps));
}

// propagate_batch: (N,6) -> (N, n_steps+1, 6)
// Full trajectories for N orbits, parallelised with OpenMP inside propagate().
arr py_propagate_batch(const arr& states0, double dt, int n_steps) {
    auto buf = states0.request();
    if (buf.ndim != 2 || buf.shape[1] != 6)
        throw std::runtime_error("states0 must have shape (N, 6)");
    const int N = static_cast<int>(buf.shape[0]);
    const double* src = static_cast<const double*>(buf.ptr);

    std::vector<StateVector> initial(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 6; ++j) {
            initial[i][j] = src[i * 6 + j];
        }
    }

    arr out({ (py::ssize_t)N, (py::ssize_t)(n_steps + 1), (py::ssize_t)6 });
    double* dst = out.mutable_data();

{
        // Release the GIL before entering the OpenMP parallel region.
        // Without this, Python's GIL would serialize all threads to one at a time,
        // defeating the parallelism. Safe here because the loop touches only C++
        // objects (no Python objects, no pybind11 calls inside the parallel block).
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            auto traj = propagate(initial[i], dt, n_steps);
            for (int t = 0; t <= n_steps; ++t) {
                for (int j = 0; j < 6; ++j) {
                    // Flat index for 3-D row-major array [N, n_steps+1, 6]
                    dst[i * (n_steps + 1) * 6 + t * 6 + j] = traj[t][j];
                }
            }
        }
    }
    return out;
}

// propagate_batch_final: (N,6) -> (N,6)
// Terminal states only for N orbits; used by the optimizer's inner loop where
// storing full trajectories would be prohibitively expensive in memory.
arr py_propagate_batch_final(const arr& states0, double dt, int n_steps) {
    auto buf = states0.request();
    if (buf.ndim != 2 || buf.shape[1] != 6)
        throw std::runtime_error("states0 must have shape (N, 6)");
    const int N = static_cast<int>(buf.shape[0]);
    const double* src = static_cast<const double*>(buf.ptr);

    std::vector<StateVector> initial(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 6; ++j) {
            initial[i][j] = src[i * 6 + j];
        }
    }

    std::vector<StateVector> results;
    {
        py::gil_scoped_release release;   // same GIL reasoning as propagate_batch
        results = propagate_batch_final(initial, dt, n_steps);
    }

    arr out({ (py::ssize_t)N, (py::ssize_t)6 });
    double* dst = out.mutable_data();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 6; ++j) {
            dst[i * 6 + j] = results[i][j];
        }
    }
    return out;
}

PYBIND11_MODULE(orbit_integrator, m) {
    m.doc() = "RK4 orbital integrator with J2 perturbation";

    m.def("propagate_single", &py_propagate_single,
          py::arg("state0"), py::arg("dt"), py::arg("n_steps"),
          "Propagate a single orbit. Returns trajectory (n_steps+1, 6).");

    m.def("propagate_single_final", &py_propagate_single_final,
          py::arg("state0"), py::arg("dt"), py::arg("n_steps"),
          "Propagate a single orbit. Returns terminal state (6,).");

    m.def("propagate_batch", &py_propagate_batch,
          py::arg("states0"), py::arg("dt"), py::arg("n_steps"),
          "Propagate N orbits. Returns trajectories (N, n_steps+1, 6).");

    m.def("propagate_batch_final", &py_propagate_batch_final,
          py::arg("states0"), py::arg("dt"), py::arg("n_steps"),
          "Propagate N orbits. Returns terminal states (N, 6).");
}
