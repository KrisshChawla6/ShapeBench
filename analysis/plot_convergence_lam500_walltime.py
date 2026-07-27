#!/usr/bin/env python3
"""
Reward vs. wall-clock time for the constrained laminar airfoil C_L/C_D
maximisation benchmark (ld_ratio_constrained_m02_re1e7_normalized, LAM500,
NeuralFoil) — RIGOROUS per-run reconstruction.

Identical run-directory selection to plot_convergence_lam500.py (eval-axis
version) — copied verbatim, zero ambiguity there. Each run's own eval axis is
rescaled to an absolute time axis using that run's OWN measured wall-clock
duration (walltime_lookup.py: slurm log Date:/End time:, with a results.csv
mtime fallback for jobs killed by the scheduler's wall-time limit before
writing a clean "End time:" footer — this affects most of the BO_torch and
several ShapeEvolve/v3 runs here, which were long-running/cancelled). Runs are
aggregated (median/p25/p75/min/max/n_active) on a common log-time grid exactly
as compute_band() does on the eval axis (walltime_aggregate.py).

Still assumes uniform time-per-eval WITHIN a single run — no per-iteration
timestamps exist in the logs — but between runs each one now carries its own
real measured duration.

Output: environments/NeuralFoil/results/convergence_plots_LAM500/
        NeuralFoil_LD_convergence_vs_walltime.pdf/.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(__file__))
from plot_walltime_common import apply_walltime_axis, human_time  # noqa: E402
from walltime_aggregate import aggregate_time_trajectory  # noqa: E402

BASE_LAM = (
    "environments/NeuralFoil/results/"
    "SAVED_DIRS_reward_ld_ratio_constrained_m02_re1e7_normalized_LAM500"
)
BASE_RES = "environments/NeuralFoil/results"
OUT_DIR = f"{BASE_RES}/convergence_plots_LAM500"
os.makedirs(OUT_DIR, exist_ok=True)

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 150,
}

COLORS = {
    "L-BFGS-B":                "#e377c2",
    "Bayesian Opt. (exact GP)": "#ff7f0e",
    "PSO":                     "#1f77b4",
    "ShapeEvolve":             "#2ca02c",
}


def curve_from_csv(csv_name):
    def _loader(run_dir):
        f = os.path.join(run_dir, csv_name)
        if not os.path.exists(f):
            return None
        df = pd.read_csv(f)
        col = "gbest_reward" if "gbest_reward" in df.columns and "best_reward" not in df.columns else "best_reward"
        return df[col].values.astype(float)
    return _loader


DEFAULT_LOADER = curve_from_csv("results.csv")


# ── Raw dirs (identical selection to plot_convergence_lam500.py) ────────────
ga_dirs = []
for a in range(1, 26):
    d = (f"{BASE_LAM}/run_GA_ld_ratio_constrained_m02_re1e7_normalized_"
         f"120particles_500iterations_attempt_{a}_AWS")
    if os.path.exists(f"{d}/results.csv"):
        ga_dirs.append(d)
print(f"GA: {len(ga_dirs)} attempt dirs found")

lbfgsb_dirs = []
for s in range(40):
    d = f"{BASE_LAM}/run_lbfgsb_ld_ratio_constrained_m02_re1e7_normalized_seed{s}_nr3"
    if os.path.exists(f"{d}/results.csv"):
        lbfgsb_dirs.append(d)
print(f"L-BFGS-B: {len(lbfgsb_dirs)} seed dirs found")

BO_SAVED = f"{BASE_RES}/SAVED_DIRS_run_BO_torch_ld_ratio_constrained_m02_re1e7_normalized"
bo_dirs = []
bo_csv_names = {}
for s in [0, 1, 2, 3]:
    d = f"{BO_SAVED}/run_BO_torch_ld_ratio_constrained_m02_re1e7_normalized_seed{s}_n5000"
    if os.path.exists(f"{d}/results.csv"):
        bo_dirs.append(d)
        bo_csv_names[d] = "results.csv"
for s in [5, 6, 7, 8, 9]:
    d = f"{BO_SAVED}/run_BO_torch_ld_ratio_constrained_m02_re1e7_normalized_seed{s}_n6000"
    recovered = f"{d}/results_recovered.csv"
    name = "results_recovered.csv" if os.path.exists(recovered) else "results.csv"
    if os.path.exists(f"{d}/{name}"):
        bo_dirs.append(d)
        bo_csv_names[d] = name
print(f"BO_torch: {len(bo_dirs)} seed dirs found")

V3_SAVED = f"{BASE_RES}/SAVED_DIRS_run_v3_dynamic_optimizer_ld_ratio_constrained_m02_re1e7_normalized"
v3_dirs = []
for a in range(1, 25):
    d = (f"{V3_SAVED}/run_v3_dynamic_optimizer_ld_ratio_constrained_m02_re1e7_normalized_"
         f"attempt_{a}_flash_2_5")
    if os.path.exists(f"{d}/results.csv"):
        v3_dirs.append(d)
print(f"v3 LLM: {len(v3_dirs)} attempt dirs found")


def bo_curve_loader(run_dir):
    return curve_from_csv(bo_csv_names[run_dir])(run_dir)


def bo_resolve_csv_name(run_dir):
    return bo_csv_names[run_dir]


CURVE_SETS = {
    "L-BFGS-B": (lbfgsb_dirs, DEFAULT_LOADER, "results.csv"),
    "Bayesian Opt. (exact GP)": (bo_dirs, bo_curve_loader, None),  # per-dir csv name
    "PSO": (ga_dirs, DEFAULT_LOADER, "results.csv"),
    "ShapeEvolve": (v3_dirs, DEFAULT_LOADER, "results.csv"),
}

XFOIL_BEST = {"LBFGSB": 329.5, "BO": 324.9, "GA": 325.6, "v3": 329.0}
NAME_KEY = {"L-BFGS-B": "LBFGSB", "Bayesian Opt. (exact GP)": "BO", "PSO": "GA", "ShapeEvolve": "v3"}

# ── Aggregate + plot ──────────────────────────────────────────────────────────
plt.rcParams.update(STYLE)
fig, ax = plt.subplots(figsize=(8.5, 5), facecolor="white")

x_max_time = 0.0
method_legend = []

for label in ["L-BFGS-B", "Bayesian Opt. (exact GP)", "PSO", "ShapeEvolve"]:
    dirs, loader, fixed_csv_name = CURVE_SETS[label]
    if not dirs:
        continue

    if fixed_csv_name is not None:
        agg = aggregate_time_trajectory(dirs, loader, results_csv=fixed_csv_name,
                                        verbose_label=label)
    else:
        # BO: per-run csv name (results.csv vs results_recovered.csv) —
        # aggregate manually per-dir since walltime_lookup needs the exact
        # file whose mtime marks the run's true end.
        from walltime_lookup import resolve_wall_seconds, load_index
        import numpy as _np
        index = load_index()
        curves, walltimes, included, excluded = [], [], [], []
        for d in dirs:
            arr = loader(d)
            if arr is None or len(arr) == 0:
                excluded.append((os.path.basename(d), "empty/unreadable curve"))
                continue
            wall_s, detail = resolve_wall_seconds(d, results_csv=bo_resolve_csv_name(d), index=index)
            if wall_s is None:
                excluded.append((os.path.basename(d), detail))
                continue
            curves.append(_np.asarray(arr, dtype=float))
            walltimes.append(wall_s)
            included.append((os.path.basename(d), wall_s, len(arr)))
        print(f"[{label}] {len(included)}/{len(dirs)} runs resolved to a wall time")
        for name, reason in excluded:
            print(f"[{label}]   EXCLUDED {name}: {reason}")
        if not curves:
            agg = None
        else:
            walltimes = _np.array(walltimes)
            max_time = float(walltimes.max())
            n_points = 600
            time_grid = _np.unique(_np.concatenate([
                _np.geomspace(max(1.0, max_time / 1e5), max_time, n_points), [max_time]
            ]))
            n_grid = len(time_grid)
            interp_vals = _np.full((len(curves), n_grid), _np.nan)
            for i, (c, w) in enumerate(zip(curves, walltimes)):
                n = len(c)
                own_t = (_np.arange(n) / max(n - 1, 1)) * w
                interp_vals[i] = _np.interp(time_grid, own_t, c)
            active_mask = time_grid[None, :] <= walltimes[:, None]
            median_v = _np.full(n_grid, _np.nan)
            p25_v = _np.full(n_grid, _np.nan)
            p75_v = _np.full(n_grid, _np.nan)
            min_v = _np.full(n_grid, _np.nan)
            max_v = _np.full(n_grid, _np.nan)
            n_active = _np.zeros(n_grid, dtype=int)
            for j in range(n_grid):
                active = interp_vals[active_mask[:, j], j]
                if len(active):
                    median_v[j] = _np.median(active)
                    p25_v[j] = _np.percentile(active, 25)
                    p75_v[j] = _np.percentile(active, 75)
                    min_v[j] = _np.min(active)
                    max_v[j] = _np.max(active)
                    n_active[j] = len(active)
            agg = dict(time_s=time_grid, n_active=n_active, median=median_v,
                       p25=p25_v, p75=p75_v, min=min_v, max=max_v, n_runs=len(curves))

    if agg is None:
        print(f"[warn] no usable runs for {label}, skipping")
        continue

    t = agg["time_s"]
    valid = ~np.isnan(agg["median"])
    t, med, p25, p75, lo, hi = (t[valid], agg["median"][valid], agg["p25"][valid],
                                agg["p75"][valid], agg["min"][valid], agg["max"][valid])
    if len(t) == 0:
        continue

    x_max_time = max(x_max_time, t.max())
    color = COLORS[label]
    disp_label = r"PSO (120p $\times$ 500i)" if label == "PSO" else label
    ax.fill_between(t, lo, hi, color=color, alpha=0.12)
    ax.fill_between(t, p25, p75, color=color, alpha=0.28)
    ax.plot(t, med, color=color, lw=1.8, label=f"{disp_label}  ({agg['n_runs']} runs)")

C_xf = {
    "LBFGSB": COLORS["L-BFGS-B"],
    "BO":     COLORS["Bayesian Opt. (exact GP)"],
    "GA":     COLORS["PSO"],
    "v3":     COLORS["ShapeEvolve"],
}
for key, xf in XFOIL_BEST.items():
    ax.axhline(xf, color=C_xf[key], lw=1.0, ls="--", alpha=0.6)

XF_LABEL_X = {"BO": 2, "GA": 5, "v3": 20, "LBFGSB": 60}
for key, xf in XFOIL_BEST.items():
    ax.text(XF_LABEL_X[key], xf + 0.3, f"XF {xf:.1f}", color=C_xf[key],
            fontsize=6.5, va="bottom", ha="left")

apply_walltime_axis(ax, x_max_time)
ax.set_xlabel("Wall-clock time (measured per-run duration; assumes uniform time/eval within each run)")
ax.set_ylabel(r"$C_L/C_D - 500\!\sum_k v_k$")
ax.set_title(
    r"Constrained Laminar Airfoil $C_L/C_D$ Maximization (Ma=0.2, Re=10$^7$)"
    " — vs. Wall-clock Time\n"
    r"Stage 1 convergence: best penalized reward per run   ($\lambda=500$, "
    "dashed = XFOIL-validated best after IPOPT post-processing)",
    fontsize=9.5,
)
ax.set_ylim(0, 368)
ax.grid(True, which="both", alpha=0.25)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

style_legend = [
    Patch(facecolor="grey", alpha=0.18, label="Min–max range"),
    Patch(facecolor="grey", alpha=0.40, label="25th–75th percentile"),
    Line2D([0], [0], color="grey", lw=1.8, label="Median best"),
]
leg1 = ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95, title="Method")
ax.add_artist(leg1)
ax.legend(handles=style_legend, loc="lower left", fontsize=8.5,
          framealpha=0.95, title="Style key")

plt.tight_layout()

out_pdf = f"{OUT_DIR}/NeuralFoil_LD_convergence_vs_walltime.pdf"
out_png = f"{OUT_DIR}/NeuralFoil_LD_convergence_vs_walltime.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved:\n  {out_pdf}\n  {out_png}")
