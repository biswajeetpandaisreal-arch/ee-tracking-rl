Baseline vs residual SAC — mean over 3 training seed(s) x 10 episodes. RMSE in mm (± std across seeds); NDJ = normalised jerk, lower is smoother.

| trajectory | condition | baseline RMSE | policy RMSE | change | baseline NDJ | policy NDJ |
|---|---|---|---|---|---|---|
| figure8 | clean | 4.5 | 2.9 ± 0.4 | -36% | 560 | 1937 |
| figure8 | noise | 4.9 | 3.0 ± 0.2 | -38% | 18431 | 41948 |
| figure8 | delay_60ms | 7.5 | 2.7 ± 0.1 | -65% | 623 | 4211 |
| figure8 | delay_100ms | 9.7 | 4.7 ± 1.0 | -52% | 645 | 4664 |
| figure8 | noise+delay | 7.8 | 3.7 ± 0.1 | -52% | 17911 | 36564 |
| figure8 | unreachable | 36.2 | 39.0 ± 1.6 | +8% | 3123 | 7924 |
| circle | clean | 2.5 | 3.9 ± 0.7 | +58% | 665 | 2585 |
| circle | noise | 3.1 | 3.1 ± 0.4 | -1% | 24221 | 52888 |
| circle | delay_60ms | 4.1 | 2.9 ± 0.3 | -29% | 731 | 4609 |
| circle | delay_100ms | 5.2 | 3.5 ± 0.2 | -34% | 734 | 5083 |
| circle | noise+delay | 4.6 | 3.4 ± 0.1 | -26% | 23768 | 45781 |
| circle | unreachable | 148.2 | 101.6 ± 13.8 | -31% | 16883 | 16905 |
| random | clean | 2.8 | 3.0 ± 0.7 | +5% | 965 | 2858 |
| random | noise | 3.3 | 3.0 ± 0.4 | -10% | 32379 | 70258 |
| random | delay_60ms | 4.8 | 2.1 ± 0.3 | -56% | 1060 | 5652 |
| random | delay_100ms | 6.3 | 3.4 ± 0.6 | -45% | 1124 | 6945 |
| random | noise+delay | 5.2 | 3.4 ± 0.0 | -35% | 31465 | 59905 |
| random | unreachable | 57.5 | 56.9 ± 2.5 | -1% | 10262 | 15983 |
