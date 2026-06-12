"""Live MuJoCo viewer — SAC residual policy with trajectory trail.

Shows:
  - Green sphere  : instantaneous reference target
  - Yellow trail  : reference trajectory ghost (next N seconds ahead)
  - Blue trail    : actual EE path (last N seconds)
  - Terminal HUD  : live RMSE, episode reward, step count

Usage (from project root):
    python scripts/visualise_live.py --model results/v3/sac_residual_final.zip
    python scripts/visualise_live.py --model results/v3/sac_residual_final.zip --trajectory circle
    python scripts/visualise_live.py --model results/v3/sac_residual_final.zip --trajectory random --noise 0.005 --delay 3
    python scripts/visualise_live.py   # baseline only, no model
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.tracking_env import PandaTrackingEnv  # noqa: E402

# ── visual config ────────────────────────────────────────────────────────────
TRAIL_SECONDS   = 2.0     # seconds of EE history to show (blue dots)
LOOKAHEAD_SECS  = 3.0     # seconds of reference path to show ahead (yellow dots)
TRAIL_DOT_R     = 0.006   # radius of trail spheres [m]
TARGET_R        = 0.018   # radius of instantaneous target sphere [m]
TRAIL_SPACING   = 3       # show every Nth step in the EE trail (reduce clutter)

# RGBA colours
COL_TARGET   = np.array([0.15, 0.90, 0.25, 0.90])  # green  — live target
COL_REF      = np.array([0.95, 0.80, 0.10, 0.55])  # yellow — reference path ahead
COL_EE       = np.array([0.20, 0.55, 0.95, 0.65])  # blue   — EE history


def build_sphere(geom, pos, radius, rgba):
    """Fill a pre-allocated mjvGeom as a sphere."""
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )


def main():
    ap = argparse.ArgumentParser(
        description="Live MuJoCo viewer for SAC residual EE-tracking policy")
    ap.add_argument("--model", default=None,
                    help="Path to SAC .zip checkpoint (omit = baseline only)")
    ap.add_argument("--trajectory", default="figure8",
                    choices=["circle", "figure8", "random"])
    ap.add_argument("--delay",  type=int,   default=0,   help="Actuator delay steps")
    ap.add_argument("--noise",  type=float, default=0.0, help="Obs noise std [m]")
    ap.add_argument("--unreachable", action="store_true",
                    help="Use partially-unreachable workspace trajectory")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier (0.5 = half speed, 2.0 = 2× faster)")
    args = ap.parse_args()

    # ── load policy ──────────────────────────────────────────────────────────
    policy = None
    if args.model:
        from stable_baselines3 import SAC  # noqa: PLC0415
        policy = SAC.load(args.model)
        print(f"[visualise] loaded policy: {args.model}")
    else:
        print("[visualise] no model supplied — running baseline (diff-IK) only")

    mode = "residual" if policy else "baseline"

    # ── build env ────────────────────────────────────────────────────────────
    env = PandaTrackingEnv(
        trajectory=args.trajectory,
        control_mode=mode,
        obs_noise_std=args.noise,
        delay_steps=args.delay,
        unreachable=args.unreachable,
        randomize=False,
    )
    obs, _ = env.reset()

    # ── pre-compute reference path dots (static per episode) ─────────────────
    def ref_dots(t_start: float, duration: float, dt: float = 0.04):
        """Sample reference positions from t_start to t_start+duration."""
        times = np.arange(0, duration, dt)
        return np.array([env.ref_pos(t_start + s) for s in times])

    # ── geometry budget ──────────────────────────────────────────────────────
    # MuJoCo passive viewer user_scn has a fixed geom budget.
    # We use: 1 (target) + N_ref + N_ee_trail
    n_ref_dots  = int(LOOKAHEAD_SECS / 0.04)   # ~75 dots
    n_ee_dots   = int(TRAIL_SECONDS  / env.dt / TRAIL_SPACING)  # ~50 dots
    N_GEOMS     = 1 + n_ref_dots + n_ee_dots

    # rolling buffer of recent EE positions
    ee_history: list[np.ndarray] = []

    # ── stats ─────────────────────────────────────────────────────────────────
    ep_errors: list[float] = []
    ep_reward = 0.0
    ep_count  = 0
    step      = 0

    print(f"\n[visualise] trajectory={args.trajectory}  mode={mode}  "
          f"noise={args.noise}  delay={args.delay}  speed={args.speed}×")
    print("[visualise] MuJoCo viewer opening — close window to exit\n")

    sleep_dt = env.dt / max(args.speed, 0.05)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

        # Initialise geom pool once
        viewer.user_scn.ngeom = N_GEOMS
        for i in range(N_GEOMS):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.001, 0.0, 0.0]),
                np.array([0.0, 0.0, -10.0]),   # park off-screen
                np.eye(3).flatten(),
                np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )

        while viewer.is_running():
            t0 = time.perf_counter()

            # ── policy step ──────────────────────────────────────────────────
            if policy is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action, _ = policy.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            ep_errors.append(info["ee_error"])
            ep_reward += reward

            ee_history.append(info["ee_pos"].copy())

            # ── episode reset ────────────────────────────────────────────────
            if terminated or truncated:
                ep_count += 1
                rmse = float(np.sqrt(np.mean(np.array(ep_errors) ** 2)))
                print(f"  ep {ep_count:3d} | steps {step:5d} | "
                      f"RMSE {rmse*1000:6.1f} mm | "
                      f"reward {ep_reward:7.1f}")
                ep_errors, ep_reward, step = [], 0.0, 0
                ee_history.clear()
                obs, _ = env.reset()

            # ── build geoms ──────────────────────────────────────────────────
            g = 0  # geom index

            # 1. Instantaneous target — large green sphere
            build_sphere(viewer.user_scn.geoms[g],
                         info["target_pos"], TARGET_R, COL_TARGET)
            g += 1

            # 2. Reference path lookahead — yellow dots
            rdots = ref_dots(env.t, LOOKAHEAD_SECS)
            for pos in rdots[:n_ref_dots]:
                build_sphere(viewer.user_scn.geoms[g],
                             pos, TRAIL_DOT_R, COL_REF)
                g += 1
            # park any unused ref dot slots
            while g < 1 + n_ref_dots:
                viewer.user_scn.geoms[g].pos[:] = [0, 0, -10]
                viewer.user_scn.geoms[g].rgba[:] = [0, 0, 0, 0]
                g += 1

            # 3. EE history trail — blue dots (thinned)
            thinned = ee_history[::TRAIL_SPACING][-n_ee_dots:]
            for pos in thinned:
                build_sphere(viewer.user_scn.geoms[g],
                             pos, TRAIL_DOT_R, COL_EE)
                g += 1
            while g < N_GEOMS:
                viewer.user_scn.geoms[g].pos[:] = [0, 0, -10]
                viewer.user_scn.geoms[g].rgba[:] = [0, 0, 0, 0]
                g += 1

            viewer.user_scn.ngeom = N_GEOMS
            viewer.sync()

            # ── real-time pacing ─────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            remaining = sleep_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    env.close()
    print("\n[visualise] viewer closed.")


if __name__ == "__main__":
    main()
