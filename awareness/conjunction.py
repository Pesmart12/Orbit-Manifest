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

# The second sieve level runs at dt / this. 6 turns the default 60 s grid into
# 10 s, which measured p95 0 m error against a 1 s reference and picked the
# correct nearest object; 20 s and coarser did not.
_SIEVE_FACTOR = 6

# Ceiling on the level-2 position store. Level 2 tends toward holding most of the
# band at fine resolution across an optimizer run — 7 GB for a 7-day mission at
# 10 s — which will not fit beside level 1 on a 12 GB card. Objects beyond the
# budget keep their level-1 separation and the cache warns once.
_FINE_BUDGET_BYTES = 3 * 1024 ** 3   # 3 GB


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
        fine_budget_bytes: float = _FINE_BUDGET_BYTES,
    ) -> None:
        # Ceiling on the level-2 store. It wants to hold most of the band at fine
        # resolution — 7 GB for a 7-day mission at 10 s — which does not fit
        # alongside level 1 and its temporaries on a 12 GB card.
        self.fine_budget_bytes = fine_budget_bytes
        self._store: dict = {}
        self._satrecs: dict = {}   # same keys as _store; used by the fine stage
        self._pending_key = None   # set by get_or_compute around a _compute call
        # Level-2 positions, per bucket: {'rows': {obj_index: column}, 'arr': (T2, M, 3)}.
        # Grown lazily as objects pass the level-1 gate, so the expensive fine SGP4
        # is paid once per object across the whole optimizer run rather than once
        # per candidate orbit.
        self._fine: dict = {}
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
        key = self._key(epoch, duration_s, dt, target_sma_m)
        if key not in self._store:
            self._pending_key = key
            bucket = round(target_sma_m / self.bucket_m) * self.bucket_m
            self._store[key] = self._compute(catalog, epoch, duration_s, dt, bucket)
        return self._store[key]

    def _key(self, epoch: datetime, duration_s: float, dt: float, target_sma_m: float):
        bucket = round(target_sma_m / self.bucket_m) * self.bucket_m
        return (epoch.isoformat(), duration_s, dt, bucket)

    def fine_positions(
        self,
        want: np.ndarray,
        epoch: datetime,
        duration_s: float,
        dt: float,
        fine_dt: float,
        target_sma_m: float,
    ):
        """Level-2 catalog positions, computed once per object and reused.

        Returns (positions (T2, M, 3), object_indices (M,)) covering every object
        ever requested for this bucket — a superset of `want`. Returning the
        superset avoids rebuilding a 480 MB array on each call; the caller maps the
        columns it cares about back through `object_indices`.

        The fine SGP4 is the expensive part (~20M evaluations for a few hundred
        objects over 7 days at 10 s), so it must not be repeated per candidate
        orbit. Level 1 gates hard enough that only ~7% of the band ever lands here.
        """
        key = self._key(epoch, duration_s, dt, target_sma_m)
        satrecs = self._satrecs.get(key, [])
        if not satrecs:
            return None, np.empty(0, dtype=np.int64)

        n_fine = int(duration_s / fine_dt)
        T2 = n_fine + 1
        col_bytes = T2 * 3 * 8

        entry = self._fine.get(key)
        if entry is None:
            # Preallocate to a byte budget and fill columns in place. Growing by
            # concatenate instead would rebuild the array on every promotion and
            # hold old + new at once, doubling peak memory — that is what
            # exhausted a 12 GB card mid-run (19.6 GB allocated, asking 10.2 more).
            cap = max(1, min(len(satrecs), int(self.fine_budget_bytes // col_bytes)))
            xp = _cp if self.use_gpu else np
            try:
                arr = xp.empty((T2, cap, 3), dtype=np.float64)
            except Exception:                       # not enough VRAM — use the host
                arr, xp = np.empty((T2, cap, 3), dtype=np.float64), np
            entry = {"rows": {}, "arr": arr, "used": 0, "cap": cap, "full": False}
            self._fine[key] = entry
            self._nbytes += cap * col_bytes

        missing = [int(j) for j in want if int(j) not in entry["rows"]]
        room = entry["cap"] - entry["used"]
        if len(missing) > room:
            if not entry["full"]:
                entry["full"] = True
                warnings.warn(
                    f"Level-2 sieve cache is full at {entry['cap']:,} objects "
                    f"({self.fine_budget_bytes / 1024**3:.1f} GB budget). Further "
                    "objects keep their coarser level-1 separation, which can "
                    "overestimate a close approach. Raise fine_budget_bytes, "
                    "increase dt, or shorten duration_s.",
                    ResourceWarning,
                    stacklevel=4,
                )
            missing = missing[:room]

        if missing:
            pos = _compute_catalog_positions(
                SatrecArray([satrecs[j] for j in missing]), epoch, n_fine, fine_dt
            )                                        # (T2, len(missing), 3), host
            dst = entry["arr"]
            block = _to_device(pos) if _array_module(dst) is _cp else pos
            lo = entry["used"]
            dst[:, lo:lo + len(missing), :] = block
            for offset, j in enumerate(missing):
                entry["rows"][j] = lo + offset
            entry["used"] += len(missing)

        if entry["used"] == 0:
            return None, np.empty(0, dtype=np.int64)

        idx = np.fromiter(entry["rows"].keys(), dtype=np.int64, count=len(entry["rows"]))
        # Hand back only the filled columns; the tail is uninitialised.
        return entry["arr"][:, :entry["used"], :], idx

    def satrecs_for(self, epoch: datetime, duration_s: float, dt: float,
                    target_sma_m: float) -> list:
        """Satrec objects for a bucket, in the same order as its position array.

        The fine refinement stage re-propagates individual objects at times that
        are not on the coarse grid, which SatrecArray's shared-time-vector call
        cannot express.
        """
        return self._satrecs.get(self._key(epoch, duration_s, dt, target_sma_m), [])

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
        # Kept so the fine stage can re-propagate individual objects at their own
        # close-approach times; see _refine_candidates.
        self._satrecs[self._pending_key] = satrec_list

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
    refine: bool = False,
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

    min_sep, tca_s = _sieve(
        mission_state, catalog_pos, cache, epoch, duration_s, dt,
        gate_m=threshold_m, target_sma_m=target_sma_m,
    )
    if refine:
        min_sep, tca_s = _refine_candidates(
            mission_state, cache.satrecs_for(epoch, duration_s, dt, target_sma_m),
            epoch, duration_s, dt, min_sep, tca_s,
            gate_m=_refine_gate(dt, threshold_m, target_sma_m),
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

    # ---- sub-sample refinement -------------------------------------------
    # The true closest approach almost never lands on a sample, so `best` is an
    # OVERestimate. Measured against a dt=2 s reference on the live catalog, the
    # dt=60 s grid overstated real miss distances by 1.6 km at the median but by
    # up to 331 km in the tail — one object 8.8 km away was reported at 120 km,
    # which for a 5 km threshold is a false negative waiting to happen.
    #
    # For a near-linear relative pass, squared distance is *exactly* quadratic in
    # time: d²(t) = d_min² + v_rel²·(t − t*)². Over one LEO sample step the motion
    # is close enough to linear that fitting a parabola through the three samples
    # around the coarse minimum recovers both the true miss distance and its time,
    # using data already in hand — no re-propagation.
    cols  = xp.arange(N)
    i_prev = xp.clip(best_idx - 1, 0, T - 1)
    i_next = xp.clip(best_idx + 1, 0, T - 1)

    def _d2_at(idx):
        d = catalog_pos[idx, cols] - mission_pos[idx]     # (N, 3) gather
        return (d * d).sum(axis=1)

    y0, y1, y2 = _d2_at(i_prev), best, _d2_at(i_next)

    # Decayed satellites and bad TLEs are marked with inf, so a neighbour sample
    # can be inf and `inf - inf` yields NaN with a RuntimeWarning. The comparisons
    # below already reject NaN (every one is False), so the result was correct —
    # but relying on that is fragile and the warnings drown real ones. Replace
    # non-finite neighbours with the centre sample, which disables the fit for
    # that object explicitly.
    y0 = xp.where(xp.isfinite(y0), y0, y1)
    y2 = xp.where(xp.isfinite(y2), y2, y1)

    # Vertex of the parabola through (-1, y0), (0, y1), (1, y2), in units of dt.
    denom = y0 - 2.0 * y1 + y2
    delta = xp.where(denom > 0.0, 0.5 * (y0 - y2) / xp.where(denom > 0.0, denom, 1.0), 0.0)
    # Reject a fit that is flat, non-convex, or points outside the bracketing
    # samples — and never let refinement land on the array edges, where one of the
    # three points is a clamped duplicate.
    ok = (
        (denom > 0.0)
        & (xp.abs(delta) <= 1.0)
        & (best_idx > 0)
        & (best_idx < T - 1)
        & xp.isfinite(y0) & xp.isfinite(y2)
    )
    d2_ref = y1 - 0.25 * (y0 - y2) * delta
    # A parabola through noisy samples can dip below zero; clamp before the sqrt.
    d2_ref = xp.where(ok, xp.maximum(d2_ref, 0.0), y1)
    t_ref  = (best_idx + xp.where(ok, delta, 0.0)) * dt

    min_sep = xp.sqrt(d2_ref)
    if xp is not np:
        min_sep, t_ref = _cp.asnumpy(min_sep), _cp.asnumpy(t_ref)
    return min_sep, t_ref


def _sieve(
    mission_state: np.ndarray,
    catalog_pos,
    cache: "CatalogCache",
    epoch: datetime,
    duration_s: float,
    dt: float,
    gate_m: float,
    target_sma_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-level search for each object's true closest approach.

    Level 1 measures every object on the caller's grid. Level 2 re-measures the
    survivors **over the whole window again** at dt/_SIEVE_FACTOR — not around
    level 1's answer. That distinction is the entire point: a coarse grid does not
    merely mis-measure an approach, it can lock onto the wrong one. On the live
    catalog NORAD 63352's true 4,005 m approach at T+2.897 h was reported as
    43,062 m at T+2.102 h — a different event 2,861 s away, which no refinement
    around the first answer could ever find.

    Sampling everything at 10 s would also fix it, at 7 GB per cache entry — too
    large to hold two buckets in 12 GB of VRAM. Gating first keeps level 2 to ~7%
    of the band, about 480 MB.

    `gate_m` is what the caller cares about beating: the conjunction threshold, or
    the current best separation. Objects are promoted when they could still get
    below it, judged by `_refine_gate`'s rigorous v_rel·dt/2 bound, so promotion
    never drops something that could reach `gate_m`.
    """
    n_steps = int(duration_s / dt)
    min_sep, tca_s = _closest_approaches(mission_state, catalog_pos, dt, n_steps)

    promote = np.flatnonzero(min_sep < _refine_gate(dt, gate_m, target_sma_m))
    if promote.size == 0:
        return min_sep, tca_s

    fine_dt = dt / _SIEVE_FACTOR
    fine_pos, obj_idx = cache.fine_positions(
        promote, epoch, duration_s, dt, fine_dt, target_sma_m
    )
    if fine_pos is None or obj_idx.size == 0:
        return min_sep, tca_s

    n_fine = int(duration_s / fine_dt)
    fine_min, fine_tca = _closest_approaches(
        mission_state, fine_pos, fine_dt, n_fine
    )

    # Level 2 searched the full window, so its answer supersedes level 1 outright
    # rather than being min()'d with it — level 1 may have found a different event.
    min_sep = np.array(min_sep, copy=True)
    tca_s   = np.array(tca_s, copy=True)
    min_sep[obj_idx] = fine_min
    tca_s[obj_idx]   = fine_tca
    return min_sep, tca_s


def _refine_candidates(
    mission_state: np.ndarray,
    satrecs: list,
    epoch: datetime,
    duration_s: float,
    dt: float,
    min_sep: np.ndarray,
    tca_s: np.ndarray,
    gate_m: float,
    fine_dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-measure objects near the gate on a fine time grid around their approach.

    The coarse grid overstates miss distances, badly in the tail: measured against
    a 1 s reference, dt=60 s reported an object 4.0 km away as 43.5 km, and picked
    the wrong satellite as the closest in the whole run. Parabolic interpolation
    fixes the median but not this — it assumes locally linear relative motion, and
    at 15 km/s a 60 s step spans 900 km, some 15 degrees of orbital arc.

    Sampling everything at dt=10 s would fix it and cost a 7 GB cache entry, too
    large to hold two buckets in 12 GB of VRAM. Instead only the objects that could
    possibly matter are re-measured: `gate_m` is a rigorous bound, so anything
    excluded provably cannot reach the threshold. On the live catalog that is ~7%
    of the band.

    The mission trajectory is propagated once at `fine_dt` and shared by every
    candidate; each object is then SGP4'd at the times in its own window.

    **Off by default, because it is incomplete.** It refines around the coarse
    argmin, which fixes mis-MEASUREMENT of an event but not mis-IDENTIFICATION of
    which event matters. Measured on the live catalog, NORAD 63352's true closest
    approach (4,005 m at T+2.897 h) is 2,861 s away from the approach the coarse
    grid picked (43,062 m at T+2.102 h) — far outside any sane window. It lifts
    p95 error from 1,068 m to 80 m for 6.6x the runtime and still names the wrong
    nearest object. See Roadmap for the multi-level sieve that would fix it.
    """
    cand = np.flatnonzero(np.asarray(min_sep) < gate_m)
    if cand.size == 0 or not satrecs:
        return min_sep, tca_s

    min_sep, tca_s = np.array(min_sep, copy=True), np.array(tca_s, copy=True)

    n_fine = int(duration_s / fine_dt)
    mission_fine = np.asarray(
        oi.propagate_single(mission_state, fine_dt, n_fine)
    )[:, :3]

    # Julian dates for the whole fine grid, built once.
    jd0, fr0 = _to_jday(epoch)

    for j in cand:
        lo = max(0.0, float(tca_s[j]) - dt)
        hi = min(duration_s, float(tca_s[j]) + dt)
        i0, i1 = int(lo / fine_dt), min(int(hi / fine_dt) + 1, len(mission_fine))
        if i1 <= i0:
            continue
        t = np.arange(i0, i1) * fine_dt

        fr = fr0 + t / 86400.0
        jd = np.full(t.shape, jd0)
        e, r, _ = satrecs[j].sgp4_array(jd, fr)
        pos = np.asarray(r, dtype=np.float64) * 1000.0
        pos[np.asarray(e) != 0] = np.inf

        d = pos - mission_fine[i0:i1]
        d2 = (d * d).sum(axis=1)
        k = int(np.argmin(d2))
        if d2[k] < min_sep[j] ** 2:
            min_sep[j] = float(np.sqrt(d2[k]))
            tca_s[j] = float(t[k])

    return min_sep, tca_s


def _refine_gate(dt: float, threshold_m: float, target_sma_m: float) -> float:
    """Coarse separations below this could still hide a real threshold violation.

    If the true closest approach happens at t*, the nearest sample is within dt/2
    of it, so the coarse minimum can exceed the true one by at most v_rel·dt/2.
    Bounding v_rel by two circular velocities at this radius (a head-on pass, the
    worst case) makes the gate rigorous: anything above it provably cannot reach
    threshold_m, so skipping it cannot hide a conjunction.
    """
    v_circ = float(np.sqrt(MU_EARTH / max(target_sma_m, R_EARTH)))
    return threshold_m + 2.0 * v_circ * dt / 2.0


def nearest_approach(
    mission_state: np.ndarray,
    epoch: datetime,
    duration_s: float,
    catalog: list[tuple[str, str, str]],
    dt: float = 30.0,
    catalog_cache: CatalogCache | None = None,
    refine: bool = False,
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

    min_sep, tca_s = _sieve(
        mission_state, catalog_pos, cache, epoch, duration_s, dt,
        gate_m=float(np.min(_closest_approaches(
            mission_state, catalog_pos, dt, n_steps)[0])),
        target_sma_m=target_sma_m,
    )
    if refine:
        # Gate on the current best: only objects that could beat it need re-measuring.
        min_sep, tca_s = _refine_candidates(
            mission_state, cache.satrecs_for(epoch, duration_s, dt, target_sma_m),
            epoch, duration_s, dt, min_sep, tca_s,
            gate_m=_refine_gate(dt, float(np.min(min_sep)), target_sma_m),
        )
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
