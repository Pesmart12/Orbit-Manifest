"""
run.py — Orbit Manifest end-to-end entry point.

Usage:
    python run.py "7-day sun-synchronous Earth observation at 550 km"
    python run.py "ISS rendezvous, 3-day crew rotation" --quick
    python run.py "Polar ice survey at 600 km, 5 days" --output results/ --seed 42
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _progress_bar(gen: int, best: float, t0: float, maxiter: int) -> None:
    elapsed = time.perf_counter() - t0
    pct = min(100, gen * 100 // maxiter)
    filled = pct // 5
    bar = "█" * filled + "░" * (20 - filled)
    print(
        f"\r  Gen {gen:>4}/{maxiter}  [{bar}] {pct:3}%"
        f"  best={best:.6f}  {elapsed:5.1f}s",
        end="",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Orbit Manifest — natural language orbital mission design",
    )
    parser.add_argument("mission", help="Mission description in plain English")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast test run: popsize=5, maxiter=20 (~30 s). Verify pipeline before a full run.",
    )
    parser.add_argument(
        "--output",
        default="results",
        metavar="DIR",
        help="Output directory for the report and ground-track PNG (default: results/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible optimizer runs",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Integrator time step in seconds (default: 60)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    popsize = 5  if args.quick else 15
    maxiter = 20 if args.quick else 500

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    wall_t0 = time.perf_counter()

    # ── Step 1: TLE catalog ────────────────────────────────────────────────
    print("━" * 62)
    print("  Fetching TLE catalog from Space-Track (cached 24 h) …")
    from awareness.tle_fetcher import fetch_tles
    try:
        catalog = fetch_tles()
    except KeyError as exc:
        print(f"\n  ERROR: missing environment variable {exc}.")
        print("  Add SPACE_TRACK_USER and SPACE_TRACK_PASS to your .env file.")
        return 1
    except Exception as exc:
        print(f"\n  ERROR fetching catalog: {exc}")
        return 1
    print(f"  {len(catalog):,} TLE objects loaded.")

    # ── Steps 2 & 3: parse with Claude, then optimize ─────────────────────
    print("━" * 62)
    print(f'  Mission: "{args.mission}"')
    print("  Calling Claude to extract mission parameters …")

    from agent.agent import plan_mission

    epoch  = datetime.now(timezone.utc)
    t_opt  = time.perf_counter()
    parsed = False

    def show_intent(intent: dict) -> None:
        """Report what Claude understood, then open the optimizer section.

        plan_mission calls this between parsing and optimizing — the only moment
        where the intent is known but the long run has not started, so the user
        sees what was parsed before the CLI goes quiet for several minutes.
        """
        nonlocal parsed, t_opt
        parsed = True
        print(f"  Orbit type:  {intent['orbit_type'].replace('_', ' ').title()}")
        print(f"  Duration:    {intent['duration_days']} days")
        print(f"  Rationale:   {intent.get('rationale', '')}")
        print("━" * 62)
        print(f"  Optimizing orbit  [popsize={popsize}, maxiter={maxiter}]")
        if args.quick:
            print("  (--quick mode — low-fidelity run for pipeline verification)")
        print()
        t_opt = time.perf_counter()   # start the clock at the optimizer, not the API call

    def cb(gen: int, best: float) -> None:
        _progress_bar(gen, best, t_opt, maxiter)

    try:
        plan = plan_mission(
            args.mission,
            catalog=catalog,
            epoch=epoch,
            dt=args.dt,
            popsize=popsize,
            maxiter=maxiter,
            seed=args.seed,
            progress_callback=cb,
            on_intent=show_intent,
        )
    except Exception as exc:
        # show_intent fires between the two halves, so whether it ran tells us
        # which one failed.
        if not parsed:
            print(f"\n  ERROR calling Claude API: {exc}")
            return 1
        print(f"\n  ERROR during optimization: {exc}")
        raise

    print()  # newline after progress bar

    # ── Step 4: text report ────────────────────────────────────────────────
    print("━" * 62)
    from output.composer import format_report, plot_ground_track

    report = format_report(plan, epoch=epoch)
    print(report)

    report_path = out_dir / "mission_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved  →  {report_path}")

    # ── Step 5: ground-track plot ─────────────────────────────────────────
    print("  Generating ground-track plot …")
    plot_path = out_dir / "ground_track.png"
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive — write straight to file
        fig = plot_ground_track(
            plan, epoch=epoch, dt=args.dt, n_samples=500, save_path=plot_path
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  Plot saved    →  {plot_path}")
    except ImportError:
        print("  (skipping plot — matplotlib not installed)")

    print("━" * 62)
    print(f"  Total time: {time.perf_counter() - wall_t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
