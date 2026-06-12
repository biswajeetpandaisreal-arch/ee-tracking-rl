"""Experiment 2: trained policy vs classical baseline, head to head.

Evaluates both controllers over the same uncertainty configurations and
seeds used in scripts/run_baseline.py and writes a comparison table, plots
per configuration, and optional videos.

Usage:
    python scripts/evaluate.py --model results/training/sac_residual_final.zip
    python scripts/evaluate.py --model ... --video --trajectory figure8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.tracking_env import PandaTrackingEnv          # noqa: E402
from scripts.eval_utils import rollout, metrics, plot_tracking, save_video  # noqa: E402
from scripts.run_baseline import CONFIGS                # noqa: E402


def evaluate(control_mode, policy, traj, episodes, cfg, video_path=None):
    ms, log = [], None
    for ep in range(episodes):
        env = PandaTrackingEnv(trajectory=traj, control_mode=control_mode,
                               randomize=True, seed=1000 + ep, **cfg)
        log = rollout(env, policy=policy,
                      record_video=(video_path is not None and ep == 0))
        ms.append(metrics(log))
        if video_path and ep == 0:
            save_video(log["frames"], video_path)
        env.close()
    agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    return agg, log


def main():
    from stable_baselines3 import SAC

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="residual",
                    choices=["residual", "rl_only"])
    ap.add_argument("--trajectory", default="figure8",
                    choices=["circle", "figure8", "random"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--outdir", default="results/comparison")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    policy = SAC.load(args.model)

    header = (f"{'config':14s} | {'baseline rmse':>13s} {'policy rmse':>12s} | "
              f"{'baseline ndj':>12s} {'policy ndj':>11s}")
    lines = [f"trajectory: {args.trajectory}  episodes/config: {args.episodes}",
             header, "-" * len(header)]
    print("\n".join(lines[-2:]))

    for name, cfg in CONFIGS.items():
        base, _ = evaluate("baseline", None, args.trajectory,
                           args.episodes, cfg)
        vid = os.path.join(args.outdir, f"policy_{name}.mp4") \
            if args.video else None
        pol, log = evaluate(args.mode, policy, args.trajectory,
                            args.episodes, cfg, video_path=vid)
        plot_tracking(log, f"SAC {args.mode} — {name}",
                      os.path.join(args.outdir, f"policy_{name}.png"))
        row = (f"{name:14s} | {1000*base['rmse']:10.1f} mm "
               f"{1000*pol['rmse']:9.1f} mm | "
               f"{base['ndj']:12.0f} {pol['ndj']:11.0f}")
        lines.append(row)
        print(row)

    with open(os.path.join(args.outdir, "comparison.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nplots + table written to {args.outdir}/")


if __name__ == "__main__":
    main()
