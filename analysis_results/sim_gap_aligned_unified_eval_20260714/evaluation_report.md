# Aligned Go2 Policy Gap Cross-Evaluation

All policies use the same 8-command schedule, 2400 steps, 256 environments, seed 400, and clean physics apart from the explicitly injected gap.

Ranking is lexicographic: total non-timeout terminations, number of zero-termination gap cells, worst-cell terminations, then mean velocity-tracking error. Mean return is reported per cell but is not used for ranking because auto-reset makes completed-episode return incomparable when only a subset of environments terminates.

## Robustness Ranking

| Rank | Policy | Total term. | Zero-term gaps | Worst term. | Nominal term. | Matched term. | Mean xy err. | Mean yaw err. |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | p5 | 24 | 4/5 | 24 | 0 | 0 | 0.0600 | 0.0637 |
| 2 | p75 | 789 | 4/5 | 789 | 0 | 0 | 0.0588 | 0.0588 |
| 3 | rr05 | 2425 | 4/5 | 2425 | 0 | 0 | 0.0614 | 0.0610 |
| 4 | rr03 | 4622 | 1/5 | 2782 | 233 | 0 | 0.0677 | 0.0867 |
| 5 | g0 | 6728 | 1/5 | 3849 | 0 | 0 | 0.0837 | 0.0877 |

## Termination Matrix

Rows are policies and columns are evaluation gaps.

| Policy | g0 | RR0.3 | RR0.5 | 5 kg | 7.5 kg |
|---|---:|---:|---:|---:|---:|
| g0 | 0 | 862 | 10 | 2007 | 3849 |
| rr03 | 233 | 0 | 66 | 1541 | 2782 |
| rr05 | 0 | 0 | 0 | 0 | 2425 |
| p5 | 0 | 24 | 0 | 0 | 0 |
| p75 | 0 | 789 | 0 | 0 | 0 |

## Velocity Error Matrix

Each cell is `xy / yaw` mean absolute tracking error.

| Policy | g0 | RR0.3 | RR0.5 | 5 kg | 7.5 kg |
|---|---:|---:|---:|---:|---:|
| g0 | 0.045 / 0.036 | 0.116 / 0.115 | 0.064 / 0.051 | 0.087 / 0.088 | 0.107 / 0.149 |
| rr03 | 0.056 / 0.061 | 0.047 / 0.050 | 0.048 / 0.054 | 0.079 / 0.113 | 0.109 / 0.156 |
| rr05 | 0.060 / 0.033 | 0.049 / 0.037 | 0.040 / 0.030 | 0.057 / 0.078 | 0.101 / 0.126 |
| p5 | 0.053 / 0.047 | 0.076 / 0.132 | 0.056 / 0.056 | 0.050 / 0.036 | 0.066 / 0.047 |
| p75 | 0.058 / 0.066 | 0.090 / 0.080 | 0.064 / 0.063 | 0.039 / 0.043 | 0.044 / 0.042 |
