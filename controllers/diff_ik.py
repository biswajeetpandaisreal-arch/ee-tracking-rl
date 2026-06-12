"""Classical baseline controller: damped-least-squares differential IK.

q_dot = J^T (J J^T + lambda^2 I)^-1 (K_p * e + v_ff)

This is the textbook resolved-rate controller with feedback on EE position
error and velocity feedforward from the reference. On the clean task it
tracks well. It degrades in exactly the regimes this challenge asks about:

  * control delay  -> pure feedback reacts to stale error => lag/oscillation
  * sensor noise   -> noise injected straight into the command
  * unreachable    -> near singularities / joint limits the DLS solution
                      loses directions and error grows

The RL residual is trained on top of this controller to recover
performance in those regimes.
"""

from __future__ import annotations

import mujoco
import numpy as np


class DiffIKController:
    def __init__(self, model: mujoco.MjModel, site_name: str = "attachment_site",
                 kp: float = 4.0, damping: float = 0.05,
                 qdot_max: float = 1.5):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                         site_name)
        self.kp = kp
        self.damping = damping
        self.qdot_max = qdot_max
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def __call__(self, data: mujoco.MjData,
                 ee_pos: np.ndarray,
                 target_pos: np.ndarray,
                 target_vel: np.ndarray) -> np.ndarray:
        """Compute a joint-velocity command (length nv = 7).

        ee_pos is passed in explicitly (rather than read from `data`) so the
        environment can feed the controller *noisy* measurements — the
        baseline must live with the same sensing the policy gets.
        """
        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr,
                          self.site_id)
        J = self._jacp  # 3 x 7, position Jacobian

        err = target_pos - ee_pos
        v_des = self.kp * err + target_vel  # feedback + feedforward

        # Damped least squares: robust near singularities.
        JJt = J @ J.T + (self.damping ** 2) * np.eye(3)
        qdot = J.T @ np.linalg.solve(JJt, v_des)

        # Saturate for safety/smoothness.
        return np.clip(qdot, -self.qdot_max, self.qdot_max)
