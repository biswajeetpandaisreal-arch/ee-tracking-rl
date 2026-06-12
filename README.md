# Residual RL for 3D End-Effector Trajectory Tracking

A Franka Panda (MuJoCo) tracks time-varying Cartesian trajectories under
sensor noise, control delay, and partially unreachable references. The
controller is a **classical differential-IK baseline + a SAC residual
policy**: RL is used where it actually earns its place — recovering
performance in the regimes where classical control measurably breaks.

## Why residual, not RL-from-scratch?

Trajectory tracking on a clean simulated arm is a solved problem:
damped-least-squares differential IK with velocity feedforward gets
millimetre-level RMSE with no learning at all. Applying RL to that version
of the problem demonstrates nothing. So this project first **quantifies
where the classical controller fails**, then trains a residual policy to
fix exactly those failures:

| config (baseline only)       | RMSE     | max err  | smoothness (NDJ) |
|------------------------------|----------|----------|------------------|
| clean                        | 5.8 mm   | 8.6 mm   | 547              |
| obs noise (σ = 5 mm)         | 5.9 mm   | 10.2 mm  | **15,761 (29×)** |
| control delay 60 ms          | 9.7 mm   | 15.4 mm  | 591              |
| control delay 100 ms         | **12.7 mm (2.2×)** | 20.9 mm | 613    |
| noise + delay                | 9.8 mm   | 15.6 mm  | 15,247           |
| partially unreachable path   | **56.1 mm (10×)**  | 111 mm  | 3,451  |

(figure-eight trajectory, 3 episodes/config, reproduced by
`python scripts/run_baseline.py`)

Each uncertainty source breaks the baseline differently: **noise destroys
smoothness** (feedback injects it straight into the command), **delay adds
lag** (pure feedback reacts to stale error), and **unreachable segments**
cause large error and jerk near the workspace boundary. The residual policy
sees trajectory *preview* (0.1 / 0.2 / 0.4 s lookahead) and the baseline's
own command, so it can act predictively (compensating delay), filter
(rejecting noise), and yield gracefully at the boundary — things the
memoryless feedback law cannot.

The residual architecture also keeps the safety story clean: the policy's
authority is bounded (±0.6 rad/s on top of the baseline), so even an
untrained or misbehaving policy degrades toward the classical controller
rather than toward chaos — the property you want before putting a learned
policy near hardware.

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

**State (43-D).** Joint positions and velocities (7+7), EE position error
(3), EE velocity (3), reference velocity (3), lookahead position errors at
+0.1/0.2/0.4 s (9), the baseline's commanded q̇ (7), previous action (7).
The lookahead terms are the key design choice: they turn delay compensation
from an inference problem into a representation problem.

**Action (7-D).** Joint-velocity residual in [−1, 1], scaled by 0.6 rad/s
and added to the baseline command. Velocity-space (not torque) matches a
real arm's command interface and is inherently smoother.

**Reward.** `exp(−‖e‖/5 cm) + 0.3·exp(−‖ė‖/0.3) − 0.05·‖Δa‖² − 0.01·‖a‖²`
— dense tracking term, velocity-matching term, action-rate penalty (the
explicit anti-jitter term), and a residual-magnitude penalty that keeps the
policy from fighting the baseline where the baseline is already good.

**Trajectory representation.** Trajectories are analytic `pos(t)` / `vel(t)`
objects (circle, 1:2 Lissajous figure-eight, sum-of-sinusoids moving
target), domain-randomised per episode in centre, size and period. The
reference is blended in from the robot's start pose over 2 s with a
smoothstep, as a real deployment would — metrics then reflect tracking, not
a step-response transient. The policy is conditioned on lookahead samples,
not a trajectory ID, so it generalises across shapes.

**Uncertainty (all three, independently switchable).**
1. Gaussian observation noise on joint states and measured EE position —
   fed to *both* baseline and policy (no cheating with clean state).
2. Constant control delay via a FIFO buffer on the commanded velocity
   (3 steps = 60 ms, 5 steps = 100 ms at 50 Hz).
3. Trajectories whose far segments exceed the Panda's ~0.855 m reach.

**Training.** SAC (Stable-Baselines3), 3-phase curriculum: clean → noise +
60 ms delay → 100 ms delay + 30 % unreachable episodes. Trajectory type
sampled per episode. `--no-curriculum` flag exists for the ablation.

**Evaluation.** RMSE / max / mean Cartesian error, plus **normalised
dimensionless jerk (NDJ)** of the EE path — a duration- and
path-length-invariant smoothness metric, so "smooth" is a number, not a
claim about how a video looks. `scripts/evaluate.py` runs baseline and
policy head-to-head over identical seeds and configs.

## Running it

```bash
pip install -r requirements.txt

# 1. Reproduce the baseline degradation table (fast, no training needed)
python scripts/run_baseline.py --trajectory figure8

# 2. Train the residual policy (~1.5M steps; overnight on a laptop CPU,
#    a few hours with CUDA. Use --steps 300000 for a quick version.)
python scripts/train.py

# 3. Head-to-head comparison + plots + videos
python scripts/evaluate.py --model results/training/sac_residual_final.zip --video

# Ablations
python scripts/train.py --mode rl_only        # RL from scratch, no baseline
python scripts/train.py --no-curriculum       # hardest distribution from step 0
```

## Results

Baseline degradation table: above (real, reproducible via
`run_baseline.py`). Trained-policy comparison table, tracking plots and
videos are generated into `results/comparison/` by `evaluate.py`.

## Repo layout

```
assets/franka_emika_panda/   official MuJoCo Menagerie Panda model (BSD-licensed, see its LICENSE)
envs/trajectories.py         analytic reference trajectories + domain randomisation
envs/tracking_env.py         Gymnasium env: residual control, uncertainty injection
controllers/diff_ik.py       damped-least-squares differential IK baseline
scripts/run_baseline.py      experiment 1: where classical control breaks
scripts/train.py             SAC training with curriculum
scripts/evaluate.py          experiment 2: baseline vs policy head-to-head
scripts/eval_utils.py        rollouts, metrics (RMSE, NDJ), plots, video
```

## Honest limitations

- The residual scale (0.6 rad/s) and reward weights were set by reasoning
  plus a small sweep, not exhaustive tuning.
- Orientation tracking (the optional extension) is not implemented; the
  observation/action design extends naturally (add orientation error as an
  axis-angle 3-vector and a rotational Jacobian term in the baseline).
- Delay is constant per episode; real latency is stochastic. Randomising
  `delay_steps` per episode during training is a one-line change.
