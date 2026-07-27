#!/usr/bin/env python3
"""
Shared helpers for reward/objective-vs-wall-clock-time plots.

APPROXIMATION NOTE: per-iteration timestamps are not logged anywhere in this
repo (slurm .out logs only record job Start/End time, not per-eval time). Each
run's OWN measured wall-clock duration is resolved from those logs
(walltime_lookup.py / build_walltime_index.py) and used to rescale that run's
eval axis to time — so between runs, timing is exact. WITHIN a run, time is
still assumed uniform per evaluation (no way to do better without per-eval
timestamps), which is a good approximation for L-BFGS-B/PSO/ShapeEvolve but
understates how much slower Bayesian Optimization's *later* iterations are
relative to its earlier ones (GP refit cost grows ~O(n^3) with the number of
observations, per the docs).
"""

import os
import sys

import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

sys.path.insert(0, os.path.dirname(__file__))
from plot_combined_methods import load_trajectory, load_adjoint_reference  # noqa: E402


def load_trajectory_walltime(csv_path, sec_per_eval, max_evals=None):
    """Load a trajectory CSV and rescale its eval axis to wall-clock seconds."""
    traj = load_trajectory(csv_path, max_evals)
    traj["time_s"] = traj["eval"] * sec_per_eval + 1.0  # +1s: keep log axis valid at eval=0
    return traj


def human_time(seconds):
    """Format a seconds value as a short human string (1s, 1min, 1h, 1d, ...)."""
    if seconds < 60:
        return f"{seconds:g}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:g}min"
    hours = seconds / 3600
    if hours < 24:
        return f"{hours:g}h"
    days = seconds / 86400
    return f"{days:g}d"


def apply_walltime_axis(ax, x_max_seconds):
    """Configure a log-scale x-axis with human-readable time tick labels."""
    ax.set_xscale("log")
    ax.set_xlim(1, x_max_seconds * 1.15)

    def fmt(value, _pos):
        if value <= 0:
            return ""
        return human_time(value)

    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    # Nice round secondary ticks at nameable points across the plotted range
    candidates = [1, 5, 10, 30, 60, 300, 600, 1800, 3600, 3 * 3600,
                  6 * 3600, 12 * 3600, 86400, 2 * 86400, 4 * 86400, 8 * 86400]
    ticks = [t for t in candidates if 1 <= t <= x_max_seconds * 1.15]
    ax.set_xticks(ticks, minor=False)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
