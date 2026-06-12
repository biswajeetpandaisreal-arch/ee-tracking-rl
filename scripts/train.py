"""Train the SAC residual policy.

Curriculum (default):
  phase 1   clean environment                       (learn to help, not fight, the baseline)
  phase 2   + observation noise + 60 ms delay       (learn filtering / anticipation)
  phase 3   + 100 ms delay, unreachable episodes    (learn graceful degradation)

Each phase continues training the same agent on a harder environment
distribution. Trajectory type is sampled per episode (circle / figure-eight /
smooth-random) with randomised centre, size and period, so the policy learns
*tracking*, not one shape.

Usage:
    python scripts/train.py                       # full curriculum, ~1.5M steps
    python scripts/train.py --steps 200000        # quick run
    python scripts/train.py --mode rl_only        # ablation: RL from scratch
"""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.tracking_env import PandaTrackingEnv  # noqa: E402


class MultiTrajectoryEnv(gym.Wrapper):
    """Samples a trajectory type per episode."""

    NAMES = ["circle", "figure8", "random"]

    def __init__(self, env: PandaTrackingEnv, unreachable_prob: float = 0.0):
        super().__init__(env)
        self.unreachable_prob = unreachable_prob

    def reset(self, **kw):
        env = self.env.unwrapped
        env.trajectory_name = env.rng.choice(self.NAMES)
        env.unreachable = bool(env.rng.random() < self.unreachable_prob)
        return self.env.reset(**kw)


def make_env(noise, delay, unreach_p, mode, seed, sample=False):
    def _f():
        env = PandaTrackingEnv(control_mode=mode, obs_noise_std=noise,
                               delay_steps=delay, randomize=True, seed=seed,
                               sample_uncertainty=sample)
        env = MultiTrajectoryEnv(env, unreachable_prob=unreach_p)
        return gym.wrappers.RecordEpisodeStatistics(env)
    return _f


# Per-episode uncertainty randomisation (delay ~ U{0..5}, noise ~ U[0,5mm],
# 30% unreachable) after a short clean warm-up. One stationary training
# distribution -> consistent replay buffer, no catastrophic forgetting; easy
# episodes never leave the distribution.
PHASES = [
    # (name, fraction, max_noise, max_delay, unreachable_p, sample_per_episode)
    ("warmup",     0.2, 0.000, 0, 0.0, False),
    ("randomized", 0.8, 0.005, 5, 0.3, True),
]


def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--mode", default="residual",
                    choices=["residual", "rl_only"])
    ap.add_argument("--outdir", default="results/training")
    ap.add_argument("--no-curriculum", action="store_true",
                    help="train on the hardest distribution from step 0")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    phases = PHASES if not args.no_curriculum \
        else [("randomized", 1.0, 0.005, 5, 0.3, True)]

    model = None
    for i, (name, frac, noise, delay, up, sample) in enumerate(phases):
        steps = int(args.steps * frac)
        venv = VecMonitor(SubprocVecEnv(
            [make_env(noise, delay, up, args.mode, seed=100 * i + j,
                      sample=sample) for j in range(args.n_envs)]))
        if model is None:
            try:
                import tensorboard  # noqa: F401
                tb = os.path.join(args.outdir, "tb")
            except ImportError:
                tb = None
            model = SAC("MlpPolicy", venv, verbose=1,
                        learning_rate=3e-4, buffer_size=1_000_000,
                        batch_size=512, gamma=0.99, tau=0.005,
                        train_freq=1, gradient_steps=1,
                        policy_kwargs=dict(net_arch=[256, 256]),
                        tensorboard_log=tb)
        else:
            model.set_env(venv)
        print(f"\n=== phase {i+1}/{len(phases)}: {name} "
              f"(max_noise={noise}, max_delay={delay}, unreachable_p={up}, "
              f"sampled={sample}) for {steps} steps ===")
        model.learn(total_timesteps=steps, reset_num_timesteps=False,
                    tb_log_name=f"sac_{args.mode}")
        model.save(os.path.join(args.outdir, f"sac_{args.mode}_{name}"))
        venv.close()

    model.save(os.path.join(args.outdir, f"sac_{args.mode}_final"))
    print(f"\nsaved final model to {args.outdir}/sac_{args.mode}_final.zip")


if __name__ == "__main__":
    main()
