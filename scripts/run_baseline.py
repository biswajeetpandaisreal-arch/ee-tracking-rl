"""Experiment 1: how the classical baseline degrades under uncertainty.

Runs the pure differential-IK controller (no RL) across uncertainty
configurations and reports RMSE / max error / smoothness for each. This
motivates the residual RL policy: the table this script prints is the
'problem statement' the policy is trained to fix.

Usage:
    python scripts/run_baseline.py [--trajectory figure8] [--video]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.tracking_env import PandaTrackingEnv          # noqa: E402
from scripts.eval_utils import rollout, metrics, plot_tracking, save_video  # noqa: E402

CONFIGS = {
    "clean":            dict(obs_noise_std=0.0,   delay_steps=0, unreachable=False),
    "noise":            dict(obs_noise_std=0.005, delay_steps=0, unreachable=False),
    "delay_60ms":       dict(obs_noise_std=0.0,   delay_steps=3, unreachable=False),
    "delay_100ms":      dict(obs_noise_std=0.0,   delay_steps=5, unreachable=False),
    "noise+delay":      dict(obs_noise_std=0.005, delay_steps=3, unreachable=False),
    "unreachable":      dict(obs_noise_std=0.0,   delay_steps=0, unreachable=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="figure8",
                    choices=["circle", "figure8", "random"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--outdir", default="results/baseline")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for name, cfg in CONFIGS.items():
        ms = []
        log = None
        for ep in range(args.episodes):
            env = PandaTrackingEnv(trajectory=args.trajectory,
                                   control_mode="baseline",
                                   randomize=True, seed=1000 + ep, **cfg)
            log = rollout(env, policy=None,
                          record_video=(args.video and ep == 0))
            ms.append(metrics(log))
            if args.video and ep == 0 and log["frames"]:
                save_video(log["frames"],
                           os.path.join(args.outdir, f"baseline_{name}.mp4"))
            env.close()
        agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        rows.append((name, agg))
        plot_tracking(log, f"Baseline (diff-IK) — {name}",
                      os.path.join(args.outdir, f"baseline_{name}.png"))
        print(f"{name:14s} rmse {1000*agg['rmse']:7.1f} mm | "
              f"max {1000*agg['max_err']:7.1f} mm | NDJ {agg['ndj']:9.1f}")

    with open(os.path.join(args.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"trajectory: {args.trajectory}  episodes: {args.episodes}\n")
        f.write(f"{'config':14s} {'rmse_mm':>9s} {'max_mm':>9s} {'ndj':>11s}\n")
        for name, agg in rows:
            f.write(f"{name:14s} {1000*agg['rmse']:9.1f} "
                    f"{1000*agg['max_err']:9.1f} {agg['ndj']:11.1f}\n")
    print(f"\nplots + summary written to {args.outdir}/")


if __name__ == "__main__":
    main()
