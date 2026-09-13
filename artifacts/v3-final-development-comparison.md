# Ouro v3 paired comparison: development

Development diagnostics only. Confirmation eligibility is a point-gain selection gate, not held-out evidence or goal completion; the full-method flag remains development-scoped.

`fixed` means the fixed4-trained model; T4/T8 denote evaluation loops. All gains are after minus before.

Rows: 1280; each hop: 128.

## Primary: d9–12, n=512

| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p |
|---|---:|---:|---:|---|---:|---:|
| Conditional: T4 → T8 | 6.05% | 6.84% | +0.78 pp | [-3.79, +5.34] pp | 30 / 26 | 0.6889 |
| Fixed4-trained T4 → conditional T8 | 10.35% | 6.84% | -3.52 pp | [-8.86, +1.89] pp | 32 / 50 | 0.05981 |
| Independent T8 → conditional T8 | 12.70% | 6.84% | -5.86 pp | [-11.29, -0.31] pp | 29 / 59 | 0.001824 |
| Fixed4-trained T8 → conditional T8 | 31.45% | 6.84% | -24.61 pp | [-30.79, -17.95] pp | 20 / 146 | 7.784e-25 |
| Fixed4-trained: T4 → T8 | 10.35% | 31.45% | +21.09 pp | [+13.74, +28.04] pp | 148 / 40 | 8.97e-16 |
| Independent: T4 → T8 | 11.91% | 12.70% | +0.78 pp | [-2.34, +3.89] pp | 14 / 10 | 0.5413 |
| Independent T4 → conditional T4 | 11.91% | 6.05% | -5.86 pp | [-10.92, -0.69] pp | 23 / 53 | 0.0007646 |
| Fixed4-trained T4 → conditional T4 | 10.35% | 6.05% | -4.30 pp | [-9.55, +1.04] pp | 29 / 51 | 0.01832 |
| Initializer T4 → conditional T4 | 9.57% | 6.05% | -3.52 pp | [-8.74, +1.77] pp | 30 / 48 | 0.05354 |
| Fixed4-trained T4 → independent T8 | 10.35% | 12.70% | +2.34 pp | [-3.67, +8.31] pp | 58 / 46 | 0.2807 |

## Decisions

- Scope: development.
- DEV confirmation eligibility: False.
- Full-method criterion (three primary CI lower bounds >0 plus d1 retention): False.
- Same-depth fixed control positive CI: False.

D1 T4 retention permits an observed drop of at most 2 percentage points:

- conditional: 100.00% → 100.00%; drop +0.00 pp; retained=True.
- independent: 100.00% → 12.50%; drop +87.50 pp; retained=False.
- fixed: 100.00% → 100.00%; drop +0.00 pp; retained=True.

## Hard-task shallow costs

These comparisons describe the cost at T4 and do not add hard-task noninferiority gates.

### Unseen d9–12, n=512

| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p |
|---|---:|---:|---:|---|---:|---:|
| Independent T4 → conditional T4 | 11.91% | 6.05% | -5.86 pp | [-10.92, -0.69] pp | 23 / 53 | 0.0007646 |
| Fixed4-trained T4 → conditional T4 | 10.35% | 6.05% | -4.30 pp | [-9.55, +1.04] pp | 29 / 51 | 0.01832 |
| Initializer T4 → conditional T4 | 9.57% | 6.05% | -3.52 pp | [-8.74, +1.77] pp | 30 / 48 | 0.05354 |

### Seen d6/8, n=256

| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p |
|---|---:|---:|---:|---|---:|---:|
| Independent T4 → conditional T4 | 12.50% | 6.25% | -6.25 pp | [-14.24, +1.98] pp | 16 / 32 | 0.0293 |
| Fixed4-trained T4 → conditional T4 | 97.27% | 6.25% | -91.02 pp | [-94.52, -84.01] pp | 1 / 234 | 8.549e-69 |
| Initializer T4 → conditional T4 | 8.59% | 6.25% | -2.34 pp | [-9.73, +5.13] pp | 16 / 22 | 0.4177 |

Restricted-choice secondary statistics, all accuracies and every per-hop comparison are in the JSON.

Statistics reuse the existing conservative Bonferroni-Wilson approximate 95% paired interval and exact McNemar test unchanged. McNemar p-values are unadjusted across comparisons; no new simultaneous-coverage claim is made. Unrestricted correctness is primary; restricted-choice correctness is secondary. Evaluator-v2 token-ID tie-breaking is required.

The caller must bind the four evaluations to the common initializer and prespecified final checkpoints. The comparator does not select checkpoints, authorize scoring, or verify dataset provenance from IDs alone.
