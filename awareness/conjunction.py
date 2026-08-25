from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, SatrecArray, jday

# C++ RK4 integrator — propagates the *mission* orbit at high fidelity.
# The catalog objects are propagated with SGP4 because running 20k objects
# through RK4 every optimizer call would be prohibitively slow, and TLE mean
# elements are defined relative to the SGP4 model — not osculating elements.
import orbit_integrator as oi

from physics.constants import MU_EARTH, R_EARTH

# Full width of the altitude screening window around a candidate's radius.
# _filter_by_altitude keeps objects overlapping ±_FILTER_BAND_M / 2.
_FILTER_BAND_M = 50_000.0  # 50 km total → ±25 km

# Default grid that candidate radii are snapped to for cache lookup. See
# CatalogCache for why this exists and why it is this size.
_BUCKET_M = 50_000.0  # 50 km

# Warn (not error) when the cache's *total* footprint exceeds this size.
_WARN_BYTES = 3 * 1024 ** 3  # 3 GB

# Rows of the (T, N, 3) array processed per chunk. Chunking is what makes the
# separation fast on either backend: the unchunked expression allocates a full
# (T, N, 3) temporary — 1.1 GB at the measured catalog size — and reads it back
# twice. Small chunks keep the working set in cache on CPU; larger ones amortise
# kernel-launch overhead on GPU.
_CHUNK_CPU = 64
_CHUNK_GPU = 1024


def _init_gpu():
    """Return the cupy module if it is importable AND can actually run a kernel.

    Set ORBIT_MANIFEST_NO_GPU=1 to force the numpy path — used by the test suite
    to check both backends agree.

    CuPy JIT-compiles its kernels and needs the CUDA headers. In a conda env those
    ship under <prefix>/Library (Windows) or <prefix> (Linux), but nothing points
    CuPy at them, so it fails at the *first kernel launch* with a confusing
    "Failed to find CUDA headers" rather than at import. Set CUDA_PATH ourselves,
    then prove it works by compiling something.
    """
    if os.environ.get("ORBIT_MANIFEST_NO_GPU"):
        return None

    # CUDA_PATH must be set BEFORE the first `import cupy`: CuPy resolves the
    # header location during import and caches it, so setting it afterwards is
    # ignored and every kernel launch fails with "Failed to find CUDA headers".
    if "CUDA_PATH" not in os.environ:
        for cand in (Path(sys.prefix) / "Library", Path(sys.prefix)):
            if (cand / "include" / "cuda_runtime.h").exists():
                os.environ["CUDA_PATH"] = str(cand)
                break
    try:
        import cupy
        int(cupy.arange(4).sum())        # forces a compile + launch
        # A single large cp.asarray() pins that much host memory and fails on a
        # ~1 GB catalog; uploads are chunked instead (see _to_device).
        cupy.cuda.set_pinned_memory_allocator(None)
        return cupy
    except Exception:
        return None


_cp = _init_gpu()
GPU_AVAILABLE = _cp is not None


def _array_module(arr):
    """numpy or cupy, whichever owns this array."""
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp
    return np


def _to_device(host: np.ndarray, rows: int = 512):
    """Copy a host array to the GPU in slices.

    A single cp.asarray() of the whole catalog tries to allocate a pinned host
    staging buffer the same size — 1.1 GB at the measured shape — and fails with
    cudaErrorMemoryAllocation even with 10 GB of VRAM free.
    """
    dev = _cp.empty(host.shape, dtype=host.dtype)
    for s in range(0, host.shape[0], rows):
        dev[s:s + rows] = _cp.asarray(host[s:s + rows])
    return dev


@dataclass
class ConjunctionResult:
    norad_id: str                          # NORAD catalog number as a string
    name: str                              # human-readable satellite name from the TLE header
    min_separation_m: float               # closest approach distance in meters
    time_of_closest_approach_s: float     # seconds after epoch when TCA occurs


def _filter_by_altitude(
    catalog: list[tuple[str, str, str]],
    target_sma_m: float,
    band_m: float = _FILTER_BAND_M,
) -> list[tuple[str, str, str]]:
    """Keep only TLEs whose orbit intersects the altitude band around target_sma_m.

    Uses TLE string parsing only — no SGP4 call needed. Reduces catalog size
    before any expensive propagation so that _compute_catalog_positions works on
    a smaller N.

    TLE field positions used (0-indexed):
        line2[26:33] — eccentricity (7 digits, implied leading "0.")
        line2[52:63] — mean motion in rev/day (11 chars)
    """
    # lo/hi are semi-major axis values (meters from Earth's centre), not altitudes.
    # Using SMA rather than altitude avoids having to add R_EARTH to every comparison.
    lo = target_sma_m - band_m / 2.0
    hi = target_sma_m + band_m / 2.0
    filtered = []
    for entry in catalog:
        _, _, line2 = entry
        try:
            # Read mean motion (rev/day) and eccentricity directly from the TLE
            # string — no SGP4 call needed.  These are the only two fields required
            # to compute periapsis/apoapsis, and parsing is O(1) per object.
            n_rev_day = float(line2[52:63])
            # Eccentricity is stored as 7 digits with an implied leading "0." —
            # e.g., "0001500" → 0.0001500.
            ecc       = float("0." + line2[26:33])
            # Kepler's third law: n² a³ = μ  →  a = (μ / n²)^(1/3)
            n_rad_s   = n_rev_day * 2.0 * np.pi / 86400.0
            a         = (MU_EARTH / n_rad_s ** 2) ** (1.0 / 3.0)
            periapsis = a * (1.0 - ecc)
            apoapsis  = a * (1.0 + ecc)
            # Interval intersection: [periapsis, apoapsis] overlaps [lo, hi] iff
            # periapsis < hi AND apoapsis > lo.  This correctly captures eccentric
            # objects that pass through the mission shell even if their mean
            # altitude is far from the target.
            if periapsis < hi and apoapsis > lo:
                filtered.append(entry)
        except (ValueError, IndexError):
            continue  # skip malformed TLEs silently
    return filtered


def _compute_catalog_positions(
    sats: SatrecArray,
    epoch: datetime,
    n_steps: int,
    dt: float,
) -> np.ndarray:
    """Propagate all catalog objects across every time step in a single SGP4 call.

    Returns shape (T, N, 3) in meters, with np.inf at positions where SGP4
    failed (decayed satellites, bad TLEs, etc.).

    The key optimization over the previous loop-per-step design: SatrecArray.sgp4
    accepts an array of T time points and returns (N, T, 3) in one call, replacing
    T sequential Python → C extension round trips with one.
    """
    T = n_steps + 1

    # Build arrays of Julian date integer and fractional parts for all T time steps.
    # sgp4 splits the Julian date into (jd, fr) to avoid precision loss near midnight
    # when a single float64 would have insufficient resolution (~8.6 ms at J2000).
    # We still use a Python loop here because _to_jday calls sgp4.api.jday() which
    # has no vectorised Python equivalent — the loop is O(T) trivial arithmetic and
    # is not a meaningful bottleneck compared to the SGP4 propagation itself.
    jd_arr = np.empty(T)
    fr_arr = np.empty(T)
    for i in range(T):
        t = epoch + timedelta(seconds=i * dt)
        jd_arr[i], fr_arr[i] = _to_jday(t)

    # Single vectorized call across all N satellites and all T time steps.
    # SatrecArray.sgp4(jd, fr) where jd/fr are length-T arrays returns:
    #   e: (N, T)    — SGP4 error codes (0 = success)
    #   r: (N, T, 3) — ECI positions in km
    # This replaces the previous design of T individual sgp4() calls in a Python
    # loop, eliminating T Python → C extension round trips per generation.
    e, r, _ = sats.sgp4(jd_arr, fr_arr)

    # SGP4 returns (N, T, 3) — satellite-major order.  Transpose to (T, N, 3) so
    # the time axis is first, matching mission_pos shape (T, 3) for broadcasting.
    catalog_pos = np.asarray(r, dtype=np.float64).transpose(1, 0, 2) * 1000.0  # km → m

    # Mask failed SGP4 positions with inf so they can never satisfy threshold_m.
    # valid is (N, T); valid.T is (T, N).  Boolean-indexing catalog_pos (T, N, 3)
    # with a (T, N) mask selects whole (x,y,z) rows, so setting to inf kills all
    # three coordinates of each bad position at once.
    valid = np.asarray(e) == 0  # (N, T); non-zero codes mean decayed / bad TLE
    catalog_pos[~valid.T] = np.inf

    return catalog_pos


class CatalogCache:
    """In-memory cache of pre-computed catalog position arrays.

    One instance should live for the lifetime of an optimization run. Because
    epoch, duration, and dt are fixed across all candidate orbits in a run,
    the catalog SGP4 propagation is done once per altitude bucket and reused for
    every candidate that falls in it.

    **Why bucketing.** A candidate's radius varies continuously across the
    optimizer's sma band, so keying on it directly gives a distinct entry per
    candidate — each a (T, N, 3) float64 array that can run to hundreds of MB.
    An SSO run over a ±20 km band produced ~41 of them. Radii are therefore
    snapped to a `bucket_m` grid before lookup.

    **Why this is safe.** Each bucket's screening band is widened by a full
    bucket width, so a candidate anywhere in the bucket still has its entire
    ±_FILTER_BAND_M / 2 window inside the cached set. The set shared by a bucket
    is always a superset of what any individual candidate would have got on its
    own, never a subset — sharing an entry cannot hide a conjunction. The extra
    objects are simply further away and do not cross the threshold.

    Usage:
        cache = CatalogCache()
        # Pass to check_conjunctions in the optimizer loop:
        results = check_conjunctions(..., catalog_cache=cache)
    """

    def __init__(
        self,
        bucket_m: float = _BUCKET_M,
        warn_bytes: float = _WARN_BYTES,
        use_gpu: bool | None = None,
    ) -> None:
        self._store: dict = {}
        self.bucket_m   = bucket_m
        self._warn_bytes = warn_bytes
        self._nbytes    = 0      # cumulative across every entry held
        self._warned    = False  # warn once per cache, not once per entry
        # None = use the GPU when CuPy is usable. Pass False to force numpy —
        # the test suite runs the same cases both ways and diffs the results.
        self.use_gpu = GPU_AVAILABLE if use_gpu is None else (use_gpu and GPU_AVAILABLE)

    @property
    def nbytes(self) -> int:
        """Total bytes held across all cached position arrays."""
        return self._nbytes

    def get_or_compute(
        self,
        catalog: list[tuple[str, str, str]],
        epoch: datetime,
        duration_s: float,
        dt: float,
        target_sma_m: float,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Return (catalog_pos (T,N,3) m, names, norad_ids) for the given parameters.

        The radius is snapped to the bucket grid so nearby candidates share an
        entry; the bucket's band is widened to compensate. See the class docstring.
        """
        bucket = round(target_sma_m / self.bucket_m) * self.bucket_m
        key = (epoch.isoformat(), duration_s, dt, bucket)
        if key not in self._store:
            self._store[key] = self._compute(catalog, epoch, duration_s, dt, bucket)
        return self._store[key]

    def _compute(
        self,
        catalog: list[tuple[str, str, str]],
        epoch: datetime,
        duration_s: float,
        dt: float,
        bucket_centre_m: float,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        # Widen by a full bucket width: a candidate at the edge of this bucket sits
        # bucket_m / 2 from the centre and still needs its own ±_FILTER_BAND_M / 2
        # window covered, so the half-widths add.
        band_m = _FILTER_BAND_M + self.bucket_m
        filtered = _filter_by_altitude(catalog, bucket_centre_m, band_m=band_m)
        if not filtered:
            return np.empty((0, 0, 3)), [], []

        names       = [t[0] for t in filtered]
        satrec_list = [Satrec.twoline2rv(t[1], t[2]) for t in filtered]
        norad_ids   = [str(s.satnum) for s in satrec_list]
        sats        = SatrecArray(satrec_list)

        n_steps = int(duration_s / dt)
        T, N    = n_steps + 1, len(filtered)

        # T steps × N objects × 3 coords × 8 bytes (float64)
        nbytes = T * N * 3 * 8
        self._nbytes += nbytes
        # Budget the cache as a whole, not each entry: the failure mode is many
        # moderate arrays accumulating across a run, which a per-entry check
        # never catches.  Warn once — a run that trips this trips it repeatedly.
        if self._nbytes > self._warn_bytes and not self._warned:
            self._warned = True
            # stacklevel=4 surfaces the warning at the check_conjunctions call
            # site in user code rather than inside this private method chain.
            warnings.warn(
                f"CatalogCache is holding {self._nbytes / 1024 ** 3:.1f} GB across "
                f"{len(self._store) + 1} altitude buckets "
                f"(latest: {N} objects × {T} steps). Consider reducing duration_s, "
                f"increasing dt, or narrowing the optimizer's sma bounds.",
                ResourceWarning,
                stacklevel=4,
            )

        catalog_pos = _compute_catalog_positions(sats, epoch, n_steps, dt)

        if self.use_gpu:
            # Move the bucket to the GPU once. Per candidate the transfer is then
            # only the mission trajectory up (~236 KB) and the minima back (~38 KB);
            # the catalog itself never crosses the bus again. Dropping the host
            # copy also frees the same bytes of RAM.
            try:
                catalog_pos = _to_device(catalog_pos)
            except Exception as exc:      # OOM, driver fault — degrade, do not fail
                warnings.warn(
                    f"GPU upload failed ({type(exc).__name__}: {exc}); "
                    "falling back to numpy for this bucket.",
                    RuntimeWarning,
                    stacklevel=4,
                )
        return catalog_pos, names, norad_ids


def check_conjunctions(
    mission_state: np.ndarray,
    epoch: datetime,
    duration_s: float,
    catalog: list[tuple[str, str, str]],
    dt: float = 30.0,
    threshold_m: float = 5000.0,
    catalog_cache: CatalogCache | None = None,
) -> list[ConjunctionResult]:
    """
    Check mission orbit against TLE catalog for conjunctions.

    Args:
        mission_state:  shape (6,) ECI state [x,y,z,vx,vy,vz] in m / m·s⁻¹
        epoch:          UTC datetime corresponding to mission_state t=0
        duration_s:     propagation window in seconds
        catalog:        [(name, line1, line2), ...] from tle_fetcher
        dt:             time step in seconds (30 s sufficient for LEO screening)
        threshold_m:    conjunction distance threshold in meters
        catalog_cache:  optional CatalogCache shared across calls; create one
                        per optimizer run so catalog positions are computed once

    Returns:
        List of ConjunctionResult below threshold, sorted by min_separation_m.
    """
    if not catalog:
        return []

    epoch = epoch.replace(tzinfo=timezone.utc) if epoch.tzinfo is None else epoch

    # An ephemeral cache is created when none is provided so the function is
    # correct in isolation, but callers in the optimizer loop should pass a
    # shared CatalogCache instance to actually benefit from caching across calls.
    cache        = catalog_cache if catalog_cache is not None else CatalogCache()
    # For a circular orbit, norm(r_vec) equals the semi-major axis exactly.
    # For a slightly elliptical candidate, it's the radius at the current true
    # anomaly — close enough to SMA for the altitude pre-filter to be correct.
    target_sma_m = float(np.linalg.norm(mission_state[:3]))
    n_steps      = int(duration_s / dt)

    catalog_pos, names, norad_ids = cache.get_or_compute(
        catalog, epoch, duration_s, dt, target_sma_m
    )

    N = len(names)
    if N == 0:
        return []

    min_sep, tca_s = _closest_approaches(
        mission_state, catalog_pos, dt, n_steps
    )

    hits = np.where(min_sep < threshold_m)[0]
    results = [
        ConjunctionResult(
            norad_id=norad_ids[i],
            name=names[i],
            min_separation_m=float(min_sep[i]),
            time_of_closest_approach_s=float(tca_s[i]),
        )
        for i in hits
    ]
    results.sort(key=lambda r: r.min_separation_m)
    return results


def _closest_approaches(
    mission_state: np.ndarray,
    catalog_pos: np.ndarray,
    dt: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-object closest approach over the whole window.

    Returns (min_sep (N,) metres, tca_s (N,) seconds after epoch) as numpy arrays,
    whichever backend did the work. Shared by the threshold check and the
    nearest-approach query so the separation maths exists in exactly one place.

    Runs on the GPU when `catalog_pos` is a cupy array (CatalogCache puts it there
    when CuPy is usable) and on numpy otherwise. Both branches execute the same
    expression; measured on a 4,873-object band they agree exactly in float64.

    Chunked and fused rather than one broadcast expression: the readable version
    materialises a (T, N, 3) temporary — 1.1 GB at the measured shape — then reads
    it back twice. Working per axis over slices measured 2.1x faster on CPU with
    identical results, and is the form the GPU wants too.
    """
    # Propagate mission orbit with C++ RK4; positions only (columns 0-2).
    # Always numpy — the integrator is a CPU extension.
    mission_pos = np.asarray(oi.propagate_single(mission_state, dt, n_steps))[:, :3]

    xp = _array_module(catalog_pos)
    if xp is not np:
        mission_pos = xp.asarray(mission_pos.astype(catalog_pos.dtype, copy=False))
        chunk = _CHUNK_GPU
    else:
        chunk = _CHUNK_CPU

    T, N = catalog_pos.shape[0], catalog_pos.shape[1]
    best     = xp.full(N, xp.inf, dtype=catalog_pos.dtype)
    best_idx = xp.zeros(N, dtype=xp.int64)

    for s in range(0, T, chunk):
        c = catalog_pos[s:s + chunk]
        m = mission_pos[s:s + chunk]
        # Squared distance per axis, accumulated in place — no (chunk, N, 3) temp.
        dx = c[:, :, 0] - m[:, None, 0]; acc  = dx * dx
        dy = c[:, :, 1] - m[:, None, 1]; acc += dy * dy
        dz = c[:, :, 2] - m[:, None, 2]; acc += dz * dz

        idx = acc.argmin(axis=0)               # (N,) index within this chunk
        val = acc.min(axis=0)                  # (N,) best squared distance here
        better = val < best
        best_idx = xp.where(better, idx + s, best_idx)
        best     = xp.where(better, val, best)

    min_sep = xp.sqrt(best)                    # compare squared, sqrt once at the end
    if xp is not np:
        min_sep, best_idx = _cp.asnumpy(min_sep), _cp.asnumpy(best_idx)
    return min_sep, best_idx * dt              # step index → seconds after epoch


def nearest_approach(
    mission_state: np.ndarray,
    epoch: datetime,
    duration_s: float,
    catalog: list[tuple[str, str, str]],
    dt: float = 30.0,
    catalog_cache: CatalogCache | None = None,
) -> ConjunctionResult | None:
    """Closest catalog object over the window, **regardless of any threshold**.

    `check_conjunctions` answers "is anything too close?" and returns nothing when
    the answer is no. That makes a clean safety verdict unquantified — "SAFE" with
    no number behind it. This answers "what was the closest thing?", so a report
    can say how much margin the orbit actually had.

    Returns None only when the catalog is empty or nothing survives the altitude
    filter. Pass the optimizer's `catalog_cache` and this is nearly free: the
    catalog positions are already computed.
    """
    if not catalog:
        return None

    epoch = epoch.replace(tzinfo=timezone.utc) if epoch.tzinfo is None else epoch
    cache = catalog_cache if catalog_cache is not None else CatalogCache()
    target_sma_m = float(np.linalg.norm(mission_state[:3]))
    n_steps = int(duration_s / dt)

    catalog_pos, names, norad_ids = cache.get_or_compute(
        catalog, epoch, duration_s, dt, target_sma_m
    )
    if len(names) == 0:
        return None

    min_sep, tca_s = _closest_approaches(mission_state, catalog_pos, dt, n_steps)
    i = int(np.argmin(min_sep))
    return ConjunctionResult(
        norad_id=norad_ids[i],
        name=names[i],
        min_separation_m=float(min_sep[i]),
        time_of_closest_approach_s=float(tca_s[i]),
    )


def screened_count(
    mission_state: np.ndarray,
    epoch: datetime,
    duration_s: float,
    catalog: list[tuple[str, str, str]],
    dt: float = 30.0,
    catalog_cache: CatalogCache | None = None,
) -> int:
    """How many catalog objects actually got screened for this orbit.

    The altitude pre-filter keeps only objects whose orbit crosses the mission
    shell, so this is far smaller than len(catalog). Reporting the full catalog
    size next to a safety verdict overstates what was checked.
    """
    if not catalog:
        return 0
    epoch = epoch.replace(tzinfo=timezone.utc) if epoch.tzinfo is None else epoch
    cache = catalog_cache if catalog_cache is not None else CatalogCache()
    _, names, _ = cache.get_or_compute(
        catalog, epoch, duration_s, dt, float(np.linalg.norm(mission_state[:3]))
    )
    return len(names)


def _to_jday(dt_utc: datetime) -> tuple[float, float]:
    """Convert a UTC datetime to the (julian_date, fraction) pair sgp4 expects.

    sgp4 splits the Julian date into an integer day (jd) and a fractional day
    (fr) to preserve floating-point precision near midnight.
    """
    return jday(
        dt_utc.year, dt_utc.month, dt_utc.day,
        dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond * 1e-6,
    )
