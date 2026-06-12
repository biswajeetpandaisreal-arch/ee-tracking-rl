"""Time-varying Cartesian reference trajectories for end-effector tracking.

Each trajectory exposes pos(t) and vel(t) (analytic derivative) so that
controllers and the RL observation can use feedforward / lookahead.
All trajectories are defined relative to a workspace centre and scaled so
they fit (or deliberately exceed, for the 'unreachable' setting) the
Panda's comfortable workspace.
"""

from __future__ import annotations

import numpy as np


class Trajectory:
    """Base class: position and velocity reference in Cartesian space."""

    def __init__(self, center: np.ndarray, period: float):
        self.center = np.asarray(center, dtype=np.float64)
        self.period = float(period)
        self.omega = 2.0 * np.pi / self.period

    def pos(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def vel(self, t: float) -> np.ndarray:
        raise NotImplementedError


class Circle(Trajectory):
    """Circle in a tilted plane (XY by default, optional Z modulation)."""

    def __init__(self, center, radius=0.15, period=8.0, tilt=0.0):
        super().__init__(center, period)
        self.r = radius
        self.tilt = tilt  # fraction of radius used as Z amplitude

    def pos(self, t):
        w = self.omega
        return self.center + np.array([
            self.r * np.cos(w * t),
            self.r * np.sin(w * t),
            self.tilt * self.r * np.sin(w * t),
        ])

    def vel(self, t):
        w = self.omega
        return np.array([
            -self.r * w * np.sin(w * t),
            self.r * w * np.cos(w * t),
            self.tilt * self.r * w * np.cos(w * t),
        ])


class FigureEight(Trajectory):
    """Lissajous 1:2 figure-eight in the YZ plane (vertical eight in front
    of the robot) — a standard 'hard' tracking reference because the
    velocity direction reverses twice per cycle."""

    def __init__(self, center, width=0.18, height=0.12, period=10.0):
        super().__init__(center, period)
        self.a = width
        self.b = height

    def pos(self, t):
        w = self.omega
        return self.center + np.array([
            0.0,
            self.a * np.sin(w * t),
            self.b * np.sin(2.0 * w * t),
        ])

    def vel(self, t):
        w = self.omega
        return np.array([
            0.0,
            self.a * w * np.cos(w * t),
            2.0 * self.b * w * np.cos(2.0 * w * t),
        ])


class SmoothRandom(Trajectory):
    """Sum-of-sinusoids 'moving target' with randomised amplitudes,
    frequencies and phases per axis. Smooth (C-infinity) but aperiodic
    over an episode, so the policy cannot memorise a shape."""

    def __init__(self, center, amplitude=0.12, period=9.0, n_harmonics=3,
                 rng: np.random.Generator | None = None):
        super().__init__(center, period)
        rng = rng or np.random.default_rng()
        self.n = n_harmonics
        # Per-axis, per-harmonic parameters. Amplitudes decay with harmonic
        # index so the path stays smooth; total amplitude normalised.
        self.amps = rng.uniform(0.3, 1.0, size=(3, self.n))
        self.amps *= amplitude / np.sum(self.amps, axis=1, keepdims=True)
        self.freqs = self.omega * rng.uniform(0.5, 1.8, size=(3, self.n))
        self.phases = rng.uniform(0, 2 * np.pi, size=(3, self.n))

    def pos(self, t):
        return self.center + np.sum(
            self.amps * np.sin(self.freqs * t + self.phases), axis=1)

    def vel(self, t):
        return np.sum(
            self.amps * self.freqs * np.cos(self.freqs * t + self.phases),
            axis=1)


# Workspace centre in front of the Panda, chosen from a reachability sweep
# (see README): comfortably inside the dexterous workspace.
DEFAULT_CENTER = np.array([0.45, 0.0, 0.45])


def make_trajectory(name: str,
                    rng: np.random.Generator | None = None,
                    randomize: bool = False,
                    unreachable: bool = False) -> Trajectory:
    """Factory with optional domain randomisation.

    unreachable=True shifts the centre outward and inflates the size so a
    portion of the reference leaves the reachable workspace — the policy
    must degrade gracefully (track the closest feasible point) instead of
    chasing an impossible target.
    """
    rng = rng or np.random.default_rng()
    center = DEFAULT_CENTER.copy()
    scale = 1.0
    if randomize:
        center = center + rng.uniform([-0.05, -0.08, -0.08], [0.05, 0.08, 0.08])
        scale = rng.uniform(0.8, 1.2)
    if unreachable:
        # Push the centre forward/up. The Panda's reach is ~0.855 m from the
        # shoulder (z = 0.333 m), so the far parts of this path are genuinely
        # infeasible while the near parts remain reachable: the controller
        # must degrade gracefully, not chase an impossible target.
        center = center + np.array([0.38, 0.0, 0.18])
        scale *= 1.3

    period = rng.uniform(7.0, 11.0) if randomize else 9.0

    if name == "circle":
        return Circle(center, radius=0.15 * scale, period=period, tilt=0.4)
    if name == "figure8":
        return FigureEight(center, width=0.18 * scale, height=0.12 * scale,
                           period=period)
    if name == "random":
        return SmoothRandom(center, amplitude=0.13 * scale, period=period,
                            rng=rng)
    raise ValueError(f"unknown trajectory '{name}'")
