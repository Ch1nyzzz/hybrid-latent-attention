# Ouro extension candidate comparison: development

Development selection diagnostics for a candidate protocol, not held-out confirmation or authorization to score. T16 must exceed this same model's T4, T6 and T8; recovery from a weak T8 is insufficient.

Initializer, control240, control384 and extension240 are fixed weight endpoints; T denotes evaluation loops. Gains are after minus before.

## Primary d9–12, n=512

| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p | Holm p (15) |
|---|---:|---:|---:|---|---:|---:|---:|
| extension / T4 → extension / T16 | 6.05% | 12.30% | +6.25 pp | [+1.02, +11.35] pp | 55 / 23 | 0.0003778 | 0.0034 |
| extension / T6 → extension / T16 | 9.57% | 12.30% | +2.73 pp | [-2.44, +7.85] pp | 44 / 30 | 0.1302 | 0.6509 |
| extension / T8 → extension / T16 | 14.65% | 12.30% | -2.34 pp | [-7.78, +3.13] pp | 36 / 48 | 0.2299 | 0.9195 |
| initializer / T4 → extension / T16 | 13.48% | 12.30% | -1.17 pp | [-6.57, +4.25] pp | 38 / 44 | 0.5811 | 1 |
| initializer / T6 → extension / T16 | 37.50% | 12.30% | -25.20 pp | [-31.23, -18.67] pp | 17 / 146 | 9.267e-27 | 1.297e-25 |
| initializer / T8 → extension / T16 | 32.81% | 12.30% | -20.51 pp | [-27.51, -13.11] pp | 42 / 147 | 7.438e-15 | 8.925e-14 |
| initializer / T16 → extension / T16 | 9.18% | 12.30% | +3.12 pp | [-1.71, +7.90] pp | 40 / 24 | 0.05994 | 0.3596 |
| control240 / T4 → extension / T16 | 12.50% | 12.30% | -0.20 pp | [-5.64, +5.25] pp | 41 / 42 | 1 | 1 |
| control240 / T6 → extension / T16 | 39.26% | 12.30% | -26.95 pp | [-33.26, -20.12] pp | 21 / 159 | 2.002e-27 | 3.003e-26 |
| control240 / T8 → extension / T16 | 33.01% | 12.30% | -20.70 pp | [-27.78, -13.22] pp | 44 / 150 | 9.791e-15 | 1.077e-13 |
| control240 / T16 → extension / T16 | 8.20% | 12.30% | +4.10 pp | [-0.82, +8.94] pp | 44 / 23 | 0.01393 | 0.09754 |
| control384 / T4 → extension / T16 | 14.26% | 12.30% | -1.95 pp | [-7.39, +3.53] pp | 37 / 47 | 0.3261 | 0.9784 |
| control384 / T6 → extension / T16 | 39.06% | 12.30% | -26.76 pp | [-33.16, -19.84] pp | 23 / 160 | 1.889e-26 | 2.455e-25 |
| control384 / T8 → extension / T16 | 32.62% | 12.30% | -20.31 pp | [-27.42, -12.81] pp | 45 / 149 | 3.303e-14 | 3.303e-13 |
| control384 / T16 → extension / T16 | 7.42% | 12.30% | +4.88 pp | [+0.05, +9.62] pp | 45 / 20 | 0.002626 | 0.02101 |

## Decisions

- development_eligible: False.
- confirmation_supported: not applicable.
- own_exit_task_floors_passed: True.
- d1_d2_retention_passed: True.
- measured_exit_range_supported: not applicable.
- confirmed_deep_gain_with_shallow_cost: not applicable.

Strongest of twelve baseline exits: 39.26%; extension/T16: 12.30%; gain -26.95 pp. The 5pp margin applies only to DEV.

## Per-hop learning and shallow retention

- control240/T4 d1: 100.00% (floor 95%); passed=True.
- control240/T4 d2: 100.00% (floor 95%); passed=True.
- control240/T4 d6: 99.22% (floor 70%); passed=True.
- control240/T4 d8: 93.75% (floor 70%); passed=True.
- control240 d1/T4: initializer 100.00% → 100.00%; drop +0.00 pp; retained=True.
- control240 d2/T4: initializer 100.00% → 100.00%; drop +0.00 pp; retained=True.
- control384/T4 d1: 100.00% (floor 95%); passed=True.
- control384/T4 d2: 100.00% (floor 95%); passed=True.
- control384/T4 d6: 99.22% (floor 70%); passed=True.
- control384/T4 d8: 95.31% (floor 70%); passed=True.
- control384 d1/T4: initializer 100.00% → 100.00%; drop +0.00 pp; retained=True.
- control384 d2/T4: initializer 100.00% → 100.00%; drop +0.00 pp; retained=True.
- extension/T8 d1: 100.00% (floor 95%); passed=True.
- extension/T8 d2: 100.00% (floor 95%); passed=True.
- extension/T8 d6: 88.28% (floor 70%); passed=True.
- extension/T8 d8: 88.28% (floor 70%); passed=True.
- extension d1/T4: initializer 100.00% → 100.00%; drop +0.00 pp; retained=True.
- extension d2/T4: initializer 100.00% → 100.00%; drop +0.00 pp; retained=True.

## Primary-population shallow costs

| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p | Holm p (15) |
|---|---:|---:|---:|---|---:|---:|---:|
| initializer / T4 → extension / T4 | 13.48% | 6.05% | -7.42 pp | [-11.86, -2.84] pp | 12 / 50 | 1.214e-06 | — |
| initializer / T8 → extension / T8 | 32.81% | 14.65% | -18.16 pp | [-25.44, -10.54] pp | 52 / 145 | 2.366e-11 | — |
| control240 / T4 → extension / T4 | 12.50% | 6.05% | -6.45 pp | [-10.85, -1.92] pp | 13 / 46 | 1.917e-05 | — |
| control240 / T8 → extension / T8 | 33.01% | 14.65% | -18.36 pp | [-25.60, -10.76] pp | 51 / 145 | 1.237e-11 | — |
| control384 / T4 → extension / T4 | 14.26% | 6.05% | -8.20 pp | [-12.74, -3.51] pp | 12 / 54 | 1.694e-07 | — |
| control384 / T8 → extension / T8 | 32.62% | 14.65% | -17.97 pp | [-25.23, -10.35] pp | 52 / 144 | 3.496e-11 | — |

Any corresponding T4/T8 drop exceeds 2pp: True. The cost flag alone establishes no gain or range claim.

## Per-hop unrestricted / choice accuracy

| Hop | Endpoint / exit | Unrestricted | Choice |
|---|---|---:|---:|
| 1 | initializer / T4 | 100.00% | 100.00% |
| 1 | initializer / T6 | 100.00% | 100.00% |
| 1 | initializer / T8 | 100.00% | 100.00% |
| 1 | initializer / T16 | 88.28% | 88.28% |
| 1 | control240 / T4 | 100.00% | 100.00% |
| 1 | control240 / T6 | 100.00% | 100.00% |
| 1 | control240 / T8 | 100.00% | 100.00% |
| 1 | control240 / T16 | 86.72% | 86.72% |
| 1 | control384 / T4 | 100.00% | 100.00% |
| 1 | control384 / T6 | 100.00% | 100.00% |
| 1 | control384 / T8 | 100.00% | 100.00% |
| 1 | control384 / T16 | 86.72% | 86.72% |
| 1 | extension / T4 | 100.00% | 100.00% |
| 1 | extension / T6 | 100.00% | 100.00% |
| 1 | extension / T8 | 100.00% | 100.00% |
| 1 | extension / T16 | 94.53% | 94.53% |
| 2 | initializer / T4 | 100.00% | 100.00% |
| 2 | initializer / T6 | 100.00% | 100.00% |
| 2 | initializer / T8 | 100.00% | 100.00% |
| 2 | initializer / T16 | 68.75% | 68.75% |
| 2 | control240 / T4 | 100.00% | 100.00% |
| 2 | control240 / T6 | 100.00% | 100.00% |
| 2 | control240 / T8 | 99.22% | 99.22% |
| 2 | control240 / T16 | 63.28% | 63.28% |
| 2 | control384 / T4 | 100.00% | 100.00% |
| 2 | control384 / T6 | 100.00% | 100.00% |
| 2 | control384 / T8 | 100.00% | 100.00% |
| 2 | control384 / T16 | 64.84% | 64.84% |
| 2 | extension / T4 | 100.00% | 100.00% |
| 2 | extension / T6 | 100.00% | 100.00% |
| 2 | extension / T8 | 100.00% | 100.00% |
| 2 | extension / T16 | 96.09% | 96.09% |
| 3 | initializer / T4 | 100.00% | 100.00% |
| 3 | initializer / T6 | 84.38% | 84.38% |
| 3 | initializer / T8 | 67.19% | 67.19% |
| 3 | initializer / T16 | 65.62% | 65.62% |
| 3 | control240 / T4 | 100.00% | 100.00% |
| 3 | control240 / T6 | 94.53% | 94.53% |
| 3 | control240 / T8 | 76.56% | 76.56% |
| 3 | control240 / T16 | 71.88% | 71.88% |
| 3 | control384 / T4 | 100.00% | 100.00% |
| 3 | control384 / T6 | 93.75% | 93.75% |
| 3 | control384 / T8 | 79.69% | 79.69% |
| 3 | control384 / T16 | 73.44% | 73.44% |
| 3 | extension / T4 | 99.22% | 99.22% |
| 3 | extension / T6 | 99.22% | 99.22% |
| 3 | extension / T8 | 99.22% | 99.22% |
| 3 | extension / T16 | 94.53% | 94.53% |
| 4 | initializer / T4 | 100.00% | 100.00% |
| 4 | initializer / T6 | 36.72% | 36.72% |
| 4 | initializer / T8 | 31.25% | 31.25% |
| 4 | initializer / T16 | 29.69% | 29.69% |
| 4 | control240 / T4 | 100.00% | 100.00% |
| 4 | control240 / T6 | 41.41% | 41.41% |
| 4 | control240 / T8 | 39.06% | 39.06% |
| 4 | control240 / T16 | 36.72% | 36.72% |
| 4 | control384 / T4 | 100.00% | 100.00% |
| 4 | control384 / T6 | 42.97% | 42.97% |
| 4 | control384 / T8 | 35.94% | 35.94% |
| 4 | control384 / T16 | 33.59% | 33.59% |
| 4 | extension / T4 | 91.41% | 91.41% |
| 4 | extension / T6 | 99.22% | 99.22% |
| 4 | extension / T8 | 95.31% | 95.31% |
| 4 | extension / T16 | 81.25% | 81.25% |
| 6 | initializer / T4 | 98.44% | 98.44% |
| 6 | initializer / T6 | 43.75% | 43.75% |
| 6 | initializer / T8 | 33.59% | 33.59% |
| 6 | initializer / T16 | 35.16% | 35.16% |
| 6 | control240 / T4 | 99.22% | 99.22% |
| 6 | control240 / T6 | 39.06% | 39.06% |
| 6 | control240 / T8 | 29.69% | 29.69% |
| 6 | control240 / T16 | 38.28% | 38.28% |
| 6 | control384 / T4 | 99.22% | 99.22% |
| 6 | control384 / T6 | 35.94% | 35.94% |
| 6 | control384 / T8 | 22.66% | 22.66% |
| 6 | control384 / T16 | 37.50% | 37.50% |
| 6 | extension / T4 | 78.12% | 78.12% |
| 6 | extension / T6 | 95.31% | 95.31% |
| 6 | extension / T8 | 88.28% | 88.28% |
| 6 | extension / T16 | 38.28% | 38.28% |
| 8 | initializer / T4 | 95.31% | 95.31% |
| 8 | initializer / T6 | 15.62% | 15.62% |
| 8 | initializer / T8 | 7.03% | 7.03% |
| 8 | initializer / T16 | 23.44% | 23.44% |
| 8 | control240 / T4 | 93.75% | 93.75% |
| 8 | control240 / T6 | 12.50% | 12.50% |
| 8 | control240 / T8 | 7.03% | 7.03% |
| 8 | control240 / T16 | 23.44% | 23.44% |
| 8 | control384 / T4 | 95.31% | 95.31% |
| 8 | control384 / T6 | 15.62% | 15.62% |
| 8 | control384 / T8 | 3.12% | 3.12% |
| 8 | control384 / T16 | 11.72% | 11.72% |
| 8 | extension / T4 | 68.75% | 68.75% |
| 8 | extension / T6 | 88.28% | 88.28% |
| 8 | extension / T8 | 88.28% | 88.28% |
| 8 | extension / T16 | 25.78% | 25.78% |
| 9 | initializer / T4 | 33.59% | 33.59% |
| 9 | initializer / T6 | 76.56% | 76.56% |
| 9 | initializer / T8 | 21.88% | 21.88% |
| 9 | initializer / T16 | 12.50% | 12.50% |
| 9 | control240 / T4 | 34.38% | 34.38% |
| 9 | control240 / T6 | 77.34% | 77.34% |
| 9 | control240 / T8 | 24.22% | 24.22% |
| 9 | control240 / T16 | 11.72% | 11.72% |
| 9 | control384 / T4 | 34.38% | 34.38% |
| 9 | control384 / T6 | 73.44% | 73.44% |
| 9 | control384 / T8 | 20.31% | 20.31% |
| 9 | control384 / T16 | 10.94% | 10.94% |
| 9 | extension / T4 | 8.59% | 8.59% |
| 9 | extension / T6 | 22.66% | 22.66% |
| 9 | extension / T8 | 39.06% | 39.06% |
| 9 | extension / T16 | 31.25% | 31.25% |
| 10 | initializer / T4 | 6.25% | 6.25% |
| 10 | initializer / T6 | 63.28% | 63.28% |
| 10 | initializer / T8 | 66.41% | 66.41% |
| 10 | initializer / T16 | 8.59% | 8.59% |
| 10 | control240 / T4 | 6.25% | 6.25% |
| 10 | control240 / T6 | 70.31% | 70.31% |
| 10 | control240 / T8 | 62.50% | 62.50% |
| 10 | control240 / T16 | 8.59% | 8.59% |
| 10 | control384 / T4 | 10.16% | 10.16% |
| 10 | control384 / T6 | 74.22% | 74.22% |
| 10 | control384 / T8 | 61.72% | 61.72% |
| 10 | control384 / T16 | 7.81% | 7.81% |
| 10 | extension / T4 | 5.47% | 5.47% |
| 10 | extension / T6 | 4.69% | 4.69% |
| 10 | extension / T8 | 10.94% | 10.94% |
| 10 | extension / T16 | 7.81% | 7.81% |
| 11 | initializer / T4 | 10.16% | 10.16% |
| 11 | initializer / T6 | 5.47% | 5.47% |
| 11 | initializer / T8 | 5.47% | 5.47% |
| 11 | initializer / T16 | 8.59% | 8.59% |
| 11 | control240 / T4 | 5.47% | 5.47% |
| 11 | control240 / T6 | 4.69% | 4.69% |
| 11 | control240 / T8 | 6.25% | 6.25% |
| 11 | control240 / T16 | 7.81% | 7.81% |
| 11 | control384 / T4 | 7.03% | 7.03% |
| 11 | control384 / T6 | 3.91% | 3.91% |
| 11 | control384 / T8 | 4.69% | 4.69% |
| 11 | control384 / T16 | 6.25% | 6.25% |
| 11 | extension / T4 | 7.03% | 7.03% |
| 11 | extension / T6 | 6.25% | 6.25% |
| 11 | extension / T8 | 4.69% | 4.69% |
| 11 | extension / T16 | 6.25% | 6.25% |
| 12 | initializer / T4 | 3.91% | 3.91% |
| 12 | initializer / T6 | 4.69% | 4.69% |
| 12 | initializer / T8 | 37.50% | 37.50% |
| 12 | initializer / T16 | 7.03% | 7.03% |
| 12 | control240 / T4 | 3.91% | 3.91% |
| 12 | control240 / T6 | 4.69% | 4.69% |
| 12 | control240 / T8 | 39.06% | 39.06% |
| 12 | control240 / T16 | 4.69% | 4.69% |
| 12 | control384 / T4 | 5.47% | 5.47% |
| 12 | control384 / T6 | 4.69% | 4.69% |
| 12 | control384 / T8 | 43.75% | 43.75% |
| 12 | control384 / T16 | 4.69% | 4.69% |
| 12 | extension / T4 | 3.12% | 3.12% |
| 12 | extension / T6 | 4.69% | 4.69% |
| 12 | extension / T8 | 3.91% | 3.91% |
| 12 | extension / T16 | 3.91% | 3.91% |

NLL, answer mass, tie diagnostics, d6/d8 T4 changes and all secondary/per-hop comparisons are retained in JSON.

Unrestricted correctness is primary. Existing conservative paired intervals and exact McNemar tests are reused unchanged. Holm adjusts the fifteen prespecified contrasts separately for each group and correctness field; only primary d9–12 unrestricted correctness controls confirmation. Secondary statistics and per-hop intervals make no simultaneous-coverage claim. Floors/retention are observed thresholds, not formal noninferiority tests. The 5pp margin applies only to DEV.

The comparator reads only explicit prediction prefixes and checks matched IDs/metadata, score schema, endpoints, counts and balanced answers. It does not verify original data, weight identity, inherited initializer provenance, training completion, frozen sources, reported summary aggregates, or candidate adoption. The controller must bind those separately. This result never triggers or authorizes any scoring, checkpoint selection or training.
