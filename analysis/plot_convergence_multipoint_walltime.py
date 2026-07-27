#!/usr/bin/env python3
"""
Reward vs. wall-clock time for the multi-point HPA drag minimisation benchmark
(reward_exact_notebook, NeuralFoil) — RIGOROUS per-run reconstruction.

Each run's own eval axis is rescaled to an absolute time axis using that run's
OWN measured wall-clock duration (resolved from slurm logs in logs/jack/, with
a results.csv-mtime fallback for jobs cancelled/timed-out without a clean
"End time:" footer — see walltime_lookup.py). Runs are then aggregated
(median/p25/p75/min/max/n_active) on a common log-time grid exactly as
plot_method_summary.py does on the eval axis (walltime_aggregate.py).

This still assumes uniform time-per-eval WITHIN a single run (no per-iteration
timestamps exist anywhere in the logs), but BETWEEN runs each one now carries
its own real duration — so e.g. L-BFGS-B's nr3-vs-nr10 seeds and BO's
44h-vs-135h seeds are no longer blended into one method-wide average.

Raw run-directory selection (verified against the docs and against the
existing eval-axis trajectory CSVs' n_active dropoff patterns):
    L-BFGS-B (40):  per seed 0-39, nr3 result if it succeeded, else the
                     nr10_RETRY result (3 seeds — 8, 14, 31 — never recovered
                     and keep their nr10_RETRY_FAILED result).
    BO (4):          seeds 0, 42 (n5000 runs) + seeds 2, 3 (nr25_rs512_cap2k
                     runs) — confirmed by the existing trajectory CSV's
                     n_active staying flat at 4 all the way to eval~4999.
    PSO/GA (25):     attempts 1-25, 120 particles x 500 iterations each.
    ShapeEvolve (35): 14 Flash attempts (COMPLETE_FOR_PLOT) + 21 Pro attempts.

Output: environments/NeuralFoil/results/combined_method_comparison_reward_exact_notebook/
        NeuralFoil_multipoint_objective_vs_walltime.png/.pdf
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(__file__))
from plot_combined_methods import STYLE, _draw_methods, load_adjoint_reference  # noqa: E402
from plot_method_summary import load_csv, best_fitness_total_trajectory  # noqa: E402
from plot_walltime_common import apply_walltime_axis, human_time  # noqa: E402
from walltime_aggregate import aggregate_time_trajectory  # noqa: E402

BASE = "environments/NeuralFoil/results"
OUT_DIR = f"{BASE}/combined_method_comparison_reward_exact_notebook"


def case1_objective_curve(run_dir):
    """Objective space (positive, lower=better), matching the existing
    *_trajectory_fitness_total.csv convention: obj = -running_max(fitness_total)."""
    rows = load_csv(run_dir)
    if not rows:
        return None
    _, best_ft = best_fitness_total_trajectory(rows)
    if best_ft is None:
        return None
    return -best_ft


# ── Raw dirs ──────────────────────────────────────────────────────────────────
NR3_DIR = f"{BASE}/SAVED_DIRS_run_lbfgsb_reward_exact_notebook_nr3"
NR10_DIR = f"{BASE}/SAVED_DIRS_run_lbfgsb_reward_exact_notebook_nr10_RETRY_of_nr3_FAILED"
lbfgsb_dirs = []
for s in range(40):
    nr3_ok = f"{NR3_DIR}/run_lbfgsb_seed{s}_nr3"
    if os.path.isdir(nr3_ok):
        lbfgsb_dirs.append(nr3_ok)
        continue
    nr10_ok = f"{NR10_DIR}/run_lbfgsb_seed{s}_nr10_RETRY"
    nr10_failed = f"{NR10_DIR}/run_lbfgsb_seed{s}_nr10_RETRY_FAILED"
    if os.path.isdir(nr10_ok):
        lbfgsb_dirs.append(nr10_ok)
    elif os.path.isdir(nr10_failed):
        lbfgsb_dirs.append(nr10_failed)
    else:
        print(f"[warn] L-BFGS-B seed{s}: no nr3/nr10 directory found at all")

BO_DIR = f"{BASE}/SAVED_DIRS_BO_torch_reward_exact_notebook"
bo_dirs = [
    f"{BO_DIR}/run_BO_torch_seed0_n5000",
    f"{BO_DIR}/run_BO_torch_seed42_n5000",
    f"{BO_DIR}/run_BO_torch_seed2_nr25_rs512_cap2k",
    f"{BO_DIR}/run_BO_torch_seed3_nr25_rs512_cap2k",
]

GA_DIR = f"{BASE}/SAVED_DIRS_run_GA_reward_exact_notebook_120particles_500iterations"
ga_dirs = [f"{GA_DIR}/run_GA_reward_exact_notebook_120particles_500iterations_attempt_{a}_AWS"
           for a in range(1, 26)]

V3_DIR = f"{BASE}/SAVED_DIRS_run_v3_dynamic_optimizer_reward_exact_notebook_flash_and_pro"
FLASH_ATT = [9, 10, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25]
PRO_ATT = [3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
v3_dirs = (
    [f"{V3_DIR}/run_v3_dynamic_optimizer_reward_exact_notebook_attempt_{a}_flash2_5_COMPLETE_FOR_PLOT"
     for a in FLASH_ATT]
    + [f"{V3_DIR}/run_v3_dynamic_optimizer_reward_exact_notebook_attempt_{a}_pro2_5"
       for a in PRO_ATT]
)

METHODS = [
    dict(label="L-BFGS-B", dirs=lbfgsb_dirs, color="#e377c2"),
    dict(label="Bayesian Opt. (exact GP)", dirs=bo_dirs, color="#ff7f0e"),
    dict(label=r"PSO (120p $\times$ 500i)", dirs=ga_dirs, color="#1f77b4"),
    dict(label="ShapeEvolve", dirs=v3_dirs, color="#2ca02c"),
]

ADJOINT_DIR = f"{BASE}/adjoint_run_fwbounds_naca0012"
ADJOINT_WALL_S = 31.0


def main():
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")

    trajs = []
    method_legend = []
    all_vals = []
    x_max = 0.0

    for m in METHODS:
        agg = aggregate_time_trajectory(m["dirs"], case1_objective_curve,
                                        verbose_label=m["label"])
        if agg is None:
            print(f"[warn] no usable runs for {m['label']}, skipping")
            continue

        t = agg["time_s"]
        med = np.maximum(agg["median"], 1e-9)
        p25 = np.maximum(agg["p25"], 1e-9)
        p75 = np.maximum(agg["p75"], 1e-9)
        mn  = np.maximum(agg["min"], 1e-9)
        mx  = np.maximum(agg["max"], 1e-9)
        na  = agg["n_active"]

        valid = ~np.isnan(med)
        t, med, p25, p75, mn, mx, na = t[valid], med[valid], p25[valid], p75[valid], mn[valid], mx[valid], na[valid]
        if len(t) == 0:
            continue

        all_vals.extend(med.tolist()); all_vals.extend(mn.tolist()); all_vals.extend(mx.tolist())
        x_max = max(x_max, t.max())

        split_t = None
        max_na = na[0]
        drop = np.where(na < max_na)[0]
        if len(drop):
            split_t = t[drop[0]]
        solid_mask = t <= split_t if split_t is not None else np.ones(len(t), dtype=bool)
        dash_mask  = t >= split_t if split_t is not None else np.zeros(len(t), dtype=bool)

        trajs.append((m["color"], t, med, p25, p75, mn, mx, solid_mask, dash_mask))
        method_legend.append(Line2D([0], [0], color=m["color"], lw=2.0,
                                    label=f"{m['label']}  ({agg['n_runs']} runs)"))

    adjoint_ref = None
    weighted_cd, adj_feasible = load_adjoint_reference(ADJOINT_DIR)
    if weighted_cd is not None:
        adjoint_ref = weighted_cd
        method_legend.append(
            Line2D([0], [0], color="#333333", lw=1.2, linestyle=":",
                   label=f"Adjoint (IPOPT)  (reward=-{adjoint_ref:.4f}, {human_time(ADJOINT_WALL_S)})")
        )

    _draw_methods(ax, trajs, adjoint_ref=adjoint_ref)

    y_lo, y_hi = 0.065, 200
    apply_walltime_axis(ax, x_max)
    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Wall-clock time (measured per-run duration; assumes uniform time/eval within each run)")
    ax.set_ylabel(
        r"Penalized objective $\overline{C_D} + \lambda\!\sum_k v_k$",
        fontsize=11,
    )
    ax.set_title(
        "Multi-point Drag Minimization — Method Comparison vs. Wall-clock Time\n"
        r"($M_\infty$=0.03, Re=$4.42$–$6.25\times10^5$, 6 $C_L$ targets)",
        fontweight="medium", pad=8, fontsize=10,
    )
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    style_legend = [
        Patch(facecolor="grey", alpha=0.18, label="Min–max range"),
        Patch(facecolor="grey", alpha=0.40, label="25th–75th percentile"),
        Line2D([0], [0], color="grey", lw=1.8, label="Median best"),
        Line2D([0], [0], color="grey", lw=1.8, linestyle="--",
               label="Fewer active runs"),
    ]
    leg1 = ax.legend(handles=method_legend, loc="upper right",
                     framealpha=0.95, title="Method")
    ax.add_artist(leg1)
    ax.legend(handles=style_legend, loc="lower left",
              framealpha=0.95, title="Style key",
              bbox_to_anchor=(0.0, 0.1))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_base = f"{OUT_DIR}/NeuralFoil_multipoint_objective_vs_walltime"
    for ext in (".png", ".pdf"):
        fig.savefig(out_base + ext, dpi=200, bbox_inches="tight")
        print(f"[walltime] Plot -> {out_base}{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
