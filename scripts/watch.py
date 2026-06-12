"""Live MuJoCo viewer: watch the baseline or a trained policy track."""
import argparse, os, sys, time
import numpy as np
import mujoco, mujoco.viewer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.tracking_env import PandaTrackingEnv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None, help="path to SAC .zip (omit = baseline only)")
ap.add_argument("--trajectory", default="figure8", choices=["circle", "figure8", "random"])
ap.add_argument("--delay", type=int, default=0)
ap.add_argument("--noise", type=float, default=0.0)
ap.add_argument("--unreachable", action="store_true")
args = ap.parse_args()

policy = None
if args.model:
    from stable_baselines3 import SAC
    policy = SAC.load(args.model)

mode = "residual" if policy else "baseline"
env = PandaTrackingEnv(trajectory=args.trajectory, control_mode=mode,
                       obs_noise_std=args.noise, delay_steps=args.delay,
                       unreachable=args.unreachable, randomize=False)
obs, _ = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as v:
    while v.is_running():
        a = np.zeros(7, dtype=np.float32) if policy is None \
            else policy.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
        v.user_scn.ngeom = 1
        mujoco.mjv_initGeom(v.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.015, 0, 0]), info["target_pos"],
                            np.eye(3).flatten(), np.array([0.2, 0.9, 0.2, 0.8]))
        v.sync()
        time.sleep(env.dt)
