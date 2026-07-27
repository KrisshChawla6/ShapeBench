#!/usr/bin/env python3
"""
Per-run wall-clock reconstruction: rescale each run's own eval axis to an
absolute time axis using ITS OWN resolved wall-clock duration (walltime_lookup),
then aggregate median/p25/p75/min/max/n_active across runs on a common
log-spaced time grid — mirroring exactly the stats logic in
plot_method_summary.make_summary() / plot_convergence_lam500.compute_band(),
just with time instead of eval count as the x-axis.

Still assumes uniform time-per-eval WITHIN a single run (no per-iteration
timestamps exist anywhere in this repo — see module docstrings in
plot_walltime_common.py) — but BETWEEN runs, each one now uses its own real
measured duration, so run-to-run heterogeneity (e.g. L-BFGS-B nr3 vs nr10,
BO seeds that ran 44h vs 135h) is captured exactly rather than blended.
"""

import os

import numpy as np

from walltime_lookup import resolve_wall_seconds, load_index


def aggregate_time_trajectory(dirs, curve_loader, results_csv="results.csv",
                              n_points=600, verbose_label=None):
    """
    dirs: list of raw run directories for one method.
    curve_loader(dir) -> 1D np.array of the running-best value already in the
        convention to be plotted directly (both minimizing/maximizing handled
        by caller — this function is direction-agnostic).

    Returns dict(time_grid, n_active, mean, median, p25, p75, min, max) or
    None if no run could be resolved. Also prints an inclusion/exclusion
    report so run counts can be sanity-checked against the docs.
    """
    index = load_index()
    curves, walltimes, included, excluded = [], [], [], []

    for d in dirs:
        arr = curve_loader(d)
        if arr is None or len(arr) == 0:
            excluded.append((os.path.basename(d), "empty/unreadable curve"))
            continue
        wall_s, detail = resolve_wall_seconds(d, results_csv=results_csv, index=index)
        if wall_s is None:
            excluded.append((os.path.basename(d), detail))
            continue
        curves.append(np.asarray(arr, dtype=float))
        walltimes.append(wall_s)
        included.append((os.path.basename(d), wall_s, len(arr)))

    tag = f"[{verbose_label}] " if verbose_label else ""
    print(f"{tag}{len(included)}/{len(dirs)} runs resolved to a wall time")
    for name, reason in excluded:
        print(f"{tag}  EXCLUDED {name}: {reason}")

    if not curves:
        return None

    walltimes = np.array(walltimes)
    max_time = float(walltimes.max())
    time_grid = np.unique(np.concatenate([
        np.geomspace(max(1.0, max_time / 1e5), max_time, n_points), [max_time]
    ]))
    n_grid = len(time_grid)

    interp_vals = np.full((len(curves), n_grid), np.nan)
    for i, (c, w) in enumerate(zip(curves, walltimes)):
        n = len(c)
        own_t = (np.arange(n) / max(n - 1, 1)) * w
        interp_vals[i] = np.interp(time_grid, own_t, c)  # clamps flat beyond own_t range

    active_mask = time_grid[None, :] <= walltimes[:, None]

    mean_v   = np.full(n_grid, np.nan)
    median_v = np.full(n_grid, np.nan)
    p25_v    = np.full(n_grid, np.nan)
    p75_v    = np.full(n_grid, np.nan)
    min_v    = np.full(n_grid, np.nan)
    max_v    = np.full(n_grid, np.nan)
    n_active = np.zeros(n_grid, dtype=int)

    for j in range(n_grid):
        active = interp_vals[active_mask[:, j], j]
        if len(active):
            mean_v[j]   = np.mean(active)
            median_v[j] = np.median(active)
            p25_v[j]    = np.percentile(active, 25)
            p75_v[j]    = np.percentile(active, 75)
            min_v[j]    = np.min(active)
            max_v[j]    = np.max(active)
            n_active[j] = len(active)

    return dict(
        time_s=time_grid, n_active=n_active,
        mean=mean_v, median=median_v, p25=p25_v, p75=p75_v, min=min_v, max=max_v,
        n_runs=len(curves), included=included, excluded=excluded,
    )
