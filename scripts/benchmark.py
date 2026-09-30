"""Experiment 3: multi-seed benchmark across every trajectory and condition.

Evaluates the baseline and one or more trained policies (e.g. one per
training seed) on all trajectory types x uncertainty configs, and prints a
Markdown table of RMSE and smoothness averaged over seeds. This is the
table the README reports.

Usage:
    python scripts/benchmark.py --models results/v5_s0/sac_residual_final.zip \
        results/v5_s1/sac_residual_final.zip results/v5_s2/sac_residual_final.zip
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import evaluate          # noqa: E402
from scripts.run_baseline import CONFIGS       # noqa: E402

TRAJECTORIES = ["figure8", "circle", "random"]


def _run(job):
    """One (trajectory, config, model) cell. Runs in a worker process."""
    traj, cfg_name, model_path, episodes = job
    policy = None
    mode = "baseline"
    if model_path is not None:
        from stable_baselines3 import SAC
        policy = SAC.load(model_path, device="cpu")
        mode = "residual"
    agg, _ = evaluate(mode, policy, traj, episodes, CONFIGS[cfg_name])
    return job, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/benchmark.md")
    args = ap.parse_args()

    jobs = [(t, c, m, args.episodes)
            for t in TRAJECTORIES for c in CONFIGS
            for m in [None] + args.models]
    res = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for (t, c, m, _), agg in ex.map(_run, jobs):
            res[(t, c, m)] = agg

    lines = [f"Baseline vs residual SAC — mean over {len(args.models)} "
             f"training seed(s) x {args.episodes} episodes. RMSE in mm "
             f"(± std across seeds); NDJ = normalised jerk, lower is smoother.",
             "",
             "| trajectory | condition | baseline RMSE | policy RMSE | "
             "change | baseline NDJ | policy NDJ |",
             "|---|---|---|---|---|---|---|"]
    for t in TRAJECTORIES:
        for c in CONFIGS:
            b = res[(t, c, None)]
            pr = np.array([res[(t, c, m)]["rmse"] for m in args.models]) * 1000
            pn = np.mean([res[(t, c, m)]["ndj"] for m in args.models])
            change = (pr.mean() / (1000 * b["rmse"]) - 1) * 100
            lines.append(
                f"| {t} | {c} | {1000 * b['rmse']:.1f} | "
                f"{pr.mean():.1f} ± {pr.std():.1f} | {change:+.0f}% | "
                f"{b['ndj']:.0f} | {pn:.0f} |")

    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
