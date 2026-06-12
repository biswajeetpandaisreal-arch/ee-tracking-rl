"""Evaluation utilities: rollouts, metrics, plots, video.

Metrics
-------
rmse        root-mean-square Cartesian tracking error [m]
max_err     worst-case error [m]
mean_err    mean error [m]
ndj         normalised dimensionless jerk of the EE path (lower = smoother)

Normalised dimensionless jerk:
    NDJ = sqrt( 0.5 * (T^5 / L^2) * integral( ||d3x/dt3||^2 dt ) )
A standard smoothness metric from motor-control literature; insensitive to
duration and path length, so runs of different configs are comparable.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.tracking_env import PandaTrackingEnv  # noqa: E402


def rollout(env: PandaTrackingEnv, policy=None, record_video: bool = False):
    """Run one episode. policy=None -> zero action (pure baseline if env is
    in 'baseline'/'residual' mode). Returns a dict of logged arrays."""
    obs, _ = env.reset()
    ee, tgt, err, acts, frames, rew = [], [], [], [], [], []
    done = False
    while not done:
        if policy is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        ee.append(info["ee_pos"])
        tgt.append(info["target_pos"])
        err.append(info["ee_error"])
        acts.append(np.asarray(action))
        rew.append(r)
        if record_video:
            frames.append(env.render())
    return {
        "ee": np.array(ee), "tgt": np.array(tgt), "err": np.array(err),
        "act": np.array(acts), "rew": np.array(rew), "frames": frames,
        "dt": env.dt,
    }


def metrics(log: dict) -> dict:
    err = log["err"]
    ee, dt = log["ee"], log["dt"]
    # third derivative by finite differences
    jerk = np.diff(ee, n=3, axis=0) / dt ** 3
    T = len(ee) * dt
    L = np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1))
    ndj = np.sqrt(0.5 * (T ** 5 / max(L, 1e-6) ** 2)
                  * np.sum(np.linalg.norm(jerk, axis=1) ** 2) * dt)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_err": float(np.max(err)),
        "mean_err": float(np.mean(err)),
        "ndj": float(ndj),
        "mean_reward": float(np.mean(log["rew"])),
    }


def plot_tracking(log: dict, title: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(log["err"])) * log["dt"]
    fig = plt.figure(figsize=(14, 9))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(*log["tgt"].T, "k--", lw=1.5, label="reference")
    ax.plot(*log["ee"].T, "C0", lw=1.2, label="end-effector")
    ax.set_title("3D path")
    ax.legend()
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")

    ax = fig.add_subplot(2, 2, 2)
    for i, lbl in enumerate("xyz"):
        ax.plot(t, log["tgt"][:, i], "k--", lw=0.8)
        ax.plot(t, log["ee"][:, i], f"C{i}", lw=1.0, label=lbl)
    ax.set_title("per-axis position"); ax.set_xlabel("t [s]")
    ax.set_ylabel("position [m]"); ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t, 1000 * log["err"], "C3")
    ax.set_title("tracking error"); ax.set_xlabel("t [s]")
    ax.set_ylabel("error [mm]")
    ax.axhline(1000 * np.sqrt(np.mean(log["err"] ** 2)), color="k",
               ls=":", label=f"RMSE = {1000*np.sqrt(np.mean(log['err']**2)):.1f} mm")
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t, log["act"])
    ax.set_title("policy action (residual)"); ax.set_xlabel("t [s]")
    ax.set_ylabel("normalised action")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_video(frames, path: str, fps: int = 50):
    if not frames:
        return
    try:
        import imageio
        imageio.mimsave(path, frames, fps=fps)
    except ImportError:
        print("imageio not installed — skipping video. pip install imageio[ffmpeg]")
