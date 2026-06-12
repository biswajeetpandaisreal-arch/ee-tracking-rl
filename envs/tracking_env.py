"""Panda end-effector trajectory-tracking environment (Gymnasium API).

Architecture
------------
                 noisy obs                    noisy obs
                    |                            |
              DiffIK baseline             SAC residual policy
                    \\                          /
                     qdot = qdot_ik + alpha * a
                                |
                        [ delay buffer ]          <- uncertainty
                                |
                  integrate -> position setpoint
                                |
                       MuJoCo position actuators

The policy learns a *residual* joint-velocity correction on top of a
classical differential-IK controller. Three uncertainty sources can be
enabled independently:

  obs_noise_std   Gaussian noise on joint positions/velocities and the
                  measured EE position (fed to BOTH baseline and policy)
  delay_steps     FIFO delay on the commanded joint velocity (actuator /
                  network latency)
  unreachable     trajectory partially leaves the reachable workspace

Control runs at 50 Hz over a 500 Hz physics step (decimation 10).
"""

from __future__ import annotations

import os
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from controllers.diff_ik import DiffIKController          # noqa: E402
from envs.trajectories import make_trajectory              # noqa: E402

_XML = os.path.join(os.path.dirname(__file__), "..", "assets",
                    "franka_emika_panda", "panda_nohand.xml")

LOOKAHEAD = [0.1, 0.2, 0.4]  # seconds of trajectory preview in the obs


class PandaTrackingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self,
                 trajectory: str = "figure8",
                 control_mode: str = "residual",   # residual | rl_only | baseline
                 obs_noise_std: float = 0.0,
                 delay_steps: int = 0,
                 unreachable: bool = False,
                 sample_uncertainty: bool = False,  # treat noise/delay as maxima, sample per episode
                 randomize: bool = True,
                 episode_seconds: float = 15.0,
                 residual_scale: float = 0.4,      # rad/s authority of the policy
                 render_mode: str | None = None,
                 seed: int | None = None):
        super().__init__()
        assert control_mode in ("residual", "rl_only", "baseline")

        self.model = mujoco.MjModel.from_xml_path(_XML)
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                         "attachment_site")
        self.nq = self.model.nq  # 7

        self.physics_dt = self.model.opt.timestep            # 0.002
        self.decimation = 10
        self.dt = self.physics_dt * self.decimation           # 0.02 (50 Hz)

        self.trajectory_name = trajectory
        self.control_mode = control_mode
        self.max_noise_std = obs_noise_std
        self.max_delay_steps = delay_steps
        self.obs_noise_std = obs_noise_std
        self.delay_steps = delay_steps
        self.sample_uncertainty = sample_uncertainty
        self.cmd_hist_len = 5
        self.unreachable = unreachable
        self.randomize = randomize
        self.max_steps = int(episode_seconds / self.dt)
        self.residual_scale = residual_scale
        self.render_mode = render_mode

        self.baseline = DiffIKController(self.model)
        self.rng = np.random.default_rng(seed)

        # --- spaces -----------------------------------------------------
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nq,),
                                       dtype=np.float32)
        # obs: q(7) qd(7) ee_err(3) ee_vel(3) tgt_vel(3)
        #      lookahead errors (3*len(LOOKAHEAD)) baseline qdot(7) prev a(7)
        # obs: q(7) qd(7) ee_err(3) ee_vel(3) tgt_vel(3)
        #      lookahead (9) baseline qdot(7) prev a(7) cmd history (5*7)
        obs_dim = 7 + 7 + 3 + 3 + 3 + 3 * len(LOOKAHEAD) + 7 + 7 + 5 * 7
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,),
                                            dtype=np.float32)

        self._renderer = None
        self._joint_low = self.model.jnt_range[:, 0]
        self._joint_high = self.model.jnt_range[:, 1]

    # ------------------------------------------------------------------
    def _ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def _ee_vel(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)
        return jacp @ self.data.qvel

    def _noisy(self, x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        if self.obs_noise_std > 0:
            return x + self.rng.normal(0, self.obs_noise_std * scale,
                                       size=x.shape)
        return x

    # --- ramped reference --------------------------------------------
    # Blend the reference in from the initial EE pose over ramp_time with a
    # smoothstep, as a real deployment would: the task is *tracking*, not a
    # step response, so metrics should reflect steady-state behaviour.
    def ref_pos(self, t: float) -> np.ndarray:
        if t >= self.ramp_time:
            return self.traj.pos(t)
        s = t / self.ramp_time
        s = s * s * (3 - 2 * s)  # smoothstep
        return self._ee_start + s * (self.traj.pos(t) - self._ee_start)

    def ref_vel(self, t: float) -> np.ndarray:
        if t >= self.ramp_time:
            return self.traj.vel(t)
        s = t / self.ramp_time
        ds = 6 * s * (1 - s) / self.ramp_time
        s = s * s * (3 - 2 * s)
        return s * self.traj.vel(t) + ds * (self.traj.pos(t) - self._ee_start)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.sample_uncertainty:
            # Per-episode domain randomisation over the uncertainty itself:
            # keeps one stationary training distribution (consistent replay
            # buffer) and never lets easy episodes leave the distribution.
            self.delay_steps = int(self.rng.integers(0, self.max_delay_steps + 1))
            self.obs_noise_std = float(self.rng.uniform(0.0, self.max_noise_std))

        self.traj = make_trajectory(self.trajectory_name, rng=self.rng,
                                    randomize=self.randomize,
                                    unreachable=self.unreachable)
        self.t = 0.0
        self.step_count = 0
        self.ramp_time = 2.0  # s: blend reference in from the start pose

        # Home keyframe + small perturbation.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[:] += self.rng.uniform(-0.05, 0.05, size=self.nq)
        self.data.qvel[:] = 0.0
        # Position-actuator setpoint starts at the current pose.
        self.data.ctrl[:] = self.data.qpos[:self.nq]
        mujoco.mj_forward(self.model, self.data)

        self._setpoint = self.data.qpos[:self.nq].copy()
        self._ee_start = self._ee_pos()
        self._delay_buf = deque(
            [np.zeros(self.nq) for _ in range(self.delay_steps)])
        self._prev_action = np.zeros(self.nq)
        self._prev_qdot_cmd = np.zeros(self.nq)
        # History of the last issued commands. A real robot always knows its
        # own past commands, so this is deployable (unlike privileged noise/
        # delay parameters) — and it lets the policy see what is still 'in
        # flight' inside the delay buffer, restoring (approximate) Markovness.
        self._cmd_hist = deque([np.zeros(self.nq)
                                for _ in range(self.cmd_hist_len)],
                               maxlen=self.cmd_hist_len)

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def _measurements(self):
        """Noisy measurements shared by baseline and policy."""
        q = self._noisy(self.data.qpos[:self.nq].copy(), scale=1.0)
        qd = self._noisy(self.data.qvel[:self.nq].copy(), scale=2.0)
        ee = self._noisy(self._ee_pos(), scale=1.0)
        return q, qd, ee

    def _get_obs(self):
        q, qd, ee = self._measurements()
        tgt = self.ref_pos(self.t)
        tgt_v = self.ref_vel(self.t)
        ee_vel = self._ee_vel()  # could also be noised; kept estimated

        look = np.concatenate([self.ref_pos(self.t + h) - ee
                               for h in LOOKAHEAD])
        qdot_ik = self.baseline(self.data, ee, tgt, tgt_v) \
            if self.control_mode != "rl_only" else np.zeros(self.nq)
        self._last_qdot_ik = qdot_ik

        obs = np.concatenate([
            q, qd,
            tgt - ee,
            ee_vel,
            tgt_v,
            look,
            qdot_ik,
            self._prev_action,
            np.concatenate(list(self._cmd_hist)) / 1.5,  # normalised
        ]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)

        if self.control_mode == "baseline":
            qdot_cmd = self._last_qdot_ik
        elif self.control_mode == "rl_only":
            qdot_cmd = 1.5 * action            # full-authority policy
        else:  # residual
            qdot_cmd = self._last_qdot_ik + self.residual_scale * action

        # Actuation delay (uncertainty source): FIFO of length delay_steps.
        if self.delay_steps > 0:
            self._delay_buf.append(qdot_cmd)
            qdot_applied = self._delay_buf.popleft()
        else:
            qdot_applied = qdot_cmd

        # Integrate velocity command into the position-actuator setpoint.
        self._setpoint = np.clip(self._setpoint + qdot_applied * self.dt,
                                 self._joint_low, self._joint_high)
        self.data.ctrl[:] = self._setpoint
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        self.t += self.dt
        self.step_count += 1

        # ---- reward ----------------------------------------------------
        ee = self._ee_pos()                       # true state for reward
        tgt = self.ref_pos(self.t)
        err = np.linalg.norm(tgt - ee)
        vel_err = np.linalg.norm(self.ref_vel(self.t) - self._ee_vel())

        r_track = np.exp(-err / 0.05)             # 1 at zero error
        r_vel = 0.3 * np.exp(-vel_err / 0.3)
        p_rate = 0.08 * np.sum((action - self._prev_action) ** 2)
        p_act = 0.0
        reward = r_track + r_vel - p_rate - p_act

        self._prev_action = action.copy()
        self._prev_qdot_cmd = qdot_cmd.copy()
        self._cmd_hist.append(qdot_cmd.copy())

        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {"ee_error": err, "ee_pos": ee, "target_pos": tgt,
                "qdot_cmd": qdot_cmd}
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
