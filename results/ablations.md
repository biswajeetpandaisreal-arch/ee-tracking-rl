# Ablations: which training uncertainty helps, which hurts

Policy RMSE in mm on each evaluation condition (lower is better). Every
configuration uses the same warm-up phase and differs only in what is
randomised during the main phase. Seeds: *all three*, A and B = 3 seeds,
C = 1 seed; 10 evaluation episodes per cell. **Bold** = beats the baseline.

| trajectory | condition | baseline | all three (noise+delay+30% unreachable) | A: noise+delay | B: delay only (final) | C: unreachable only |
|---|---|---|---|---|---|---|
| figure8 | clean | 4.5 | 6.1 | **3.9** | **2.9** | 5.2 |
| figure8 | noise | 4.9 | 5.5 | **4.0** | **3.0** | **3.9** |
| figure8 | delay_60ms | 7.5 | 7.8 | **4.5** | **2.7** | **5.4** |
| figure8 | delay_100ms | 9.7 | 9.9 | **7.1** | **4.7** | 15.4 |
| figure8 | noise+delay | 7.8 | **6.9** | **5.0** | **3.7** | **5.7** |
| figure8 | unreachable | 36.2 | 40.9 | 41.3 | 39.0 | 37.8 |
| circle | clean | 2.5 | 6.5 | 4.0 | 3.9 | 6.0 |
| circle | noise | 3.1 | 5.6 | 3.9 | 3.1 | 3.4 |
| circle | delay_60ms | 4.1 | 7.6 | 4.1 | **2.9** | 5.2 |
| circle | delay_100ms | 5.2 | 8.9 | 5.9 | **3.5** | 16.0 |
| circle | noise+delay | 4.6 | 6.3 | **4.1** | **3.4** | 5.1 |
| circle | unreachable | 148.2 | **101.2** | **104.0** | **101.6** | **127.3** |
| random | clean | 2.8 | 5.7 | 3.4 | 3.0 | 2.9 |
| random | noise | 3.3 | 5.0 | 3.4 | **3.0** | 3.5 |
| random | delay_60ms | 4.8 | 7.7 | **3.8** | **2.1** | **4.5** |
| random | delay_100ms | 6.3 | 9.5 | **5.7** | **3.4** | 15.4 |
| random | noise+delay | 5.2 | 6.7 | **4.1** | **3.4** | 5.4 |
| random | unreachable | 57.5 | **55.6** | **54.2** | **56.9** | **55.3** |

Conditions where the policy beats the baseline (of 18): all three **3**, A **11**, B **14**, C **6**.

## Reading

- **Unreachable references poison training.** C (unreachable only) is up to
  +143% worse than the baseline on the *clean* circle. Diagnostics showed the
  policy learns a constant joint-velocity bias ("lean out") that the
  baseline's feedback then fights, leaving a steady-state offset. Out-of-reach
  references should be clamped by the reference generator, not learned.
- **Noise randomisation makes the policy cautious.** A never fails badly but
  gives up most of the gain B gets.
- **Delay is where the residual earns its place.** B cuts RMSE by 26-65% in
  every delayed condition, using the trajectory preview and measured latency.
