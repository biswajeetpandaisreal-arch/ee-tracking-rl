# Residual RL for 3D End-Effector Trajectory Tracking

A Franka Panda arm (MuJoCo) tracks time-varying Cartesian trajectories under
sensor noise, control delay, and partially unreachable references. The
controller combines a **classical differential-IK baseline with a SAC residual
policy**: RL is used where it actually earns its place — recovering performance
in the regimes where classical control measurably breaks.

---

## Demo

```bash
# Watch the trained policy track a figure-8 live (MuJoCo viewer window)
python scripts/visualise_live.py --model results/v3/sac_residual_final.zip

# Circle trajectory
python scripts/visualise_live.py --model results/v3/sac_residual_final.zip --trajectory circle

# Aperiodic random trajectory — hardest generalisation test
python scripts/visualise_live.py --model results/v3/sac_residual_final.zip --trajectory random

# Stress test: noise + 60 ms actuator delay
python scripts/visualise_live.py --model results/v3/sac_residual_final.zip --noise 0.005 --delay 3

# Half speed for close inspection
python scripts/visualise_live.py --model results/v3/sac_residual_final.zip --speed 0.5
```

The viewer shows three overlaid trails in real time:

| colour | meaning |
|--------|---------|
| green sphere | instantaneous reference target |
| yellow dots | reference path 3 s ahead |
| blue dots | actual EE path over the last 2 s |

---

## Why residual, not RL from scratch?

Trajectory tracking on a clean simulated arm is a solved problem:
damped-least-squares differential IK with velocity feedforward achieves
millimetre-level RMSE with no learning at all. Applying RL to that version
demonstrates nothing. This project instead **quantifies where the classical
controller fails**, then trains a residual policy to fix exactly those failures:

| config (baseline only) | RMSE | max err | smoothness (NDJ) |
|---|---|---|---|
| clean | 5.8 mm | 8.6 mm | 547 |
| obs noise (σ = 5 mm) | 5.9 mm | 10.2 mm | **15,761 (29×)** |
| control delay 60 ms | 9.7 mm | 15.4 mm | 591 |
| control delay 100 ms | **12.7 mm (2.2×)** | 20.9 mm | 613 |
| noise + delay | 9.8 mm | 15.6 mm | 15,247 |
| partially unreachable path | **56.1 mm (10×)** | 111 mm | 3,451 |

(figure-eight trajectory, 3 episodes/config — reproduced by `python scripts/run_baseline.py`)

Each uncertainty source breaks the baseline differently:

- **Noise** destroys smoothness — feedback injects it straight into the joint command
- **Delay** adds lag — pure feedback reacts to stale error and cannot anticipate
- **Unreachable segments** cause large error and jerk near the workspace boundary

The residual policy sees trajectory *preview* (0.1 / 0.2 / 0.4 s lookahead) and the
baseline's own command, so it can act predictively (compensating delay), filter
(rejecting noise), and yield gracefully at the boundary — things a memoryless
feedback law cannot do.

The residual architecture also keeps the safety story clean: the policy's authority
is bounded (±0.6 rad/s on top of the baseline), so even an untrained or misbehaving
policy degrades toward the classical controller rather than toward chaos — the
property you want before putting a learned policy near hardware.

---

## System design

```
            noisy measurements (shared)
              ┌──────────┴──────────┐
        diff-IK baseline      SAC residual policy
              └──────────┬──────────┘
               q̇ = q̇_ik + 0.6 · a
                         │
                  [delay buffer]        ← uncertainty
                         │
          integrate → position setpoint
                         │
              MuJoCo position actuators (500 Hz physics, 50 Hz control)
```

**State (43-D).** Joint positions and velocities (7+7), EE position error (3),
EE velocity (3), reference velocity (3), lookahead position errors at +0.1/0.2/0.4 s (9),
the baseline's commanded q̇ (7), previous action (7). The lookahead terms are the
key design choice: they turn delay compensation from an inference problem into a
representation problem.

**Action (7-D).** Joint-velocity residual in [−1, 1], scaled by 0.6 rad/s and added
to the baseline command. Velocity-space control (not torque) matches a real arm's
command interface and is inherently smoother.

**Reward.**
```
r = exp(−‖e‖/5cm) + 0.3·exp(−‖ė‖/0.3) − 0.08·‖Δa‖²
```
Dense tracking term, velocity-matching term, and an action-rate penalty as an
explicit anti-jitter regulariser. The residual-magnitude penalty keeps the policy
from fighting the baseline where the baseline is already good.

**Trajectories.** Analytic `pos(t)` / `vel(t)` objects — circle, 1:2 Lissajous
figure-eight, sum-of-sinusoids moving target — domain-randomised per episode in
centre, size, and period. The reference is blended in from the robot's start pose
over 2 s with a smoothstep, so metrics reflect steady-state tracking, not a
step-response transient. The policy is conditioned on lookahead samples, not a
trajectory ID, so it generalises across shapes.

**Uncertainty (all three, independently switchable).**
1. Gaussian observation noise on joint states and measured EE position, fed to
   *both* baseline and policy (no cheating with clean state)
2. Constant control delay via a FIFO buffer on the commanded velocity
   (3 steps = 60 ms, 5 steps = 100 ms at 50 Hz)
3. Trajectories whose far segments exceed the Panda's ~0.855 m reach

**Training.** SAC (Stable-Baselines3), 2-phase curriculum: clean warmup (20%)
→ per-episode domain randomisation over noise, delay, and unreachable probability
(80%). Trajectory type sampled per episode. `--no-curriculum` flag for ablation.

**Evaluation.** RMSE / max / mean Cartesian error, plus **normalised dimensionless
jerk (NDJ)** of the EE path — a duration- and path-length-invariant smoothness
metric from motor-control literature, so "smooth" is a number, not a claim about
how a video looks. `scripts/evaluate.py` runs baseline and policy head-to-head
over identical seeds and configs.

---

## Running it

```bash
pip install -r requirements.txt

# 1. Reproduce the baseline degradation table (fast, no training needed)
python scripts/run_baseline.py --trajectory figure8

# 2. Train the residual policy (~1.5M steps; overnight on laptop CPU,
#    a few hours with CUDA. Use --steps 300000 for a quick smoke test.)
python scripts/train.py

# 3. Head-to-head comparison: baseline vs policy, all uncertainty configs
python scripts/evaluate.py --model results/v3/sac_residual_final.zip --video

# 4. Live MuJoCo visualisation (opens a viewer window)
python scripts/visualise_live.py --model results/v3/sac_residual_final.zip

# Ablations
python scripts/train.py --mode rl_only         # RL from scratch, no baseline
python scripts/train.py --no-curriculum        # hardest distribution from step 0
```

---

## Results

The baseline degradation table is above (real, reproducible via `run_baseline.py`).
The trained policy comparison table, per-axis tracking plots, and episode videos are
generated into `results/comparison/` by `evaluate.py`.

Training ran for 600k steps on a laptop CPU (~47 min). The reward curve shows a
clean plateau phase (426k–558k steps) followed by a sustained ascent to a peak
episode reward of 641, settling at ~637 — consistent with a policy that found a
better attractor late in training and had not yet fully converged.

---

## Repo layout

```
assets/franka_emika_panda/   MuJoCo Menagerie Panda model (BSD-licensed)
envs/trajectories.py         analytic reference trajectories + domain randomisation
envs/tracking_env.py         Gymnasium env: residual control, uncertainty injection
controllers/diff_ik.py       damped-least-squares differential IK baseline
scripts/run_baseline.py      experiment 1: where classical control breaks
scripts/train.py             SAC training with 2-phase curriculum
scripts/evaluate.py          experiment 2: baseline vs policy head-to-head
scripts/eval_utils.py        rollouts, metrics (RMSE, NDJ), plots, video
scripts/visualise_live.py    live MuJoCo viewer with EE trail and reference path
results/v3/                  trained checkpoint (sac_residual_final.zip)
```

---

## Honest limitations

- The residual scale (0.6 rad/s) and reward weights were set by reasoning plus a
  small sweep, not exhaustive tuning. A proper hyperparameter search would likely
  push RMSE below 10 mm.
- Training ran to 600k steps on a laptop CPU; the reward curve had not fully
  converged. Further training is expected to improve performance.
- Orientation tracking is not implemented. The design extends naturally: add
  orientation error as an axis-angle 3-vector and a rotational Jacobian term in
  the baseline.
- Delay is constant per episode; real latency is stochastic. Randomising
  `delay_steps` per episode during training is a one-line change and would likely
  improve robustness further.
- Sim-to-real transfer has not been tested. The main gap is unmodelled joint
  friction and motor dynamics; system identification or domain randomisation over
  those parameters would be the next step toward hardware deployment.