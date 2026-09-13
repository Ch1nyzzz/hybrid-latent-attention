# Ouro V4 paired comparison: development

Development selection only; no held-out confirmation claim. All statistics and measured exits remain development-scoped.

Fixed4 and Fixed8 name training depth; T names evaluation loops. Gains are after minus before.

Rows: 1280; each hop: 128.

## Primary d9–12: unrestricted correctness, n=512

| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p | Holm p (5) |
|---|---:|---:|---:|---|---:|---:|---:|
| fixed8 / T8 → fixed8 / T16 | 12.50% | 12.50% | +0.00 pp | [-0.97, +0.97] pp | 0 / 0 | 1 | 1 |
| fixed4 / T16 → fixed8 / T16 | 12.50% | 12.50% | +0.00 pp | [-6.56, +6.56] pp | 64 / 64 | 1 | 1 |
| fixed4 / T4 → fixed8 / T16 | 12.50% | 12.50% | +0.00 pp | [-6.56, +6.56] pp | 64 / 64 | 1 | 1 |
| fixed4 / T6 → fixed8 / T16 | 12.50% | 12.50% | +0.00 pp | [-6.56, +6.56] pp | 64 / 64 | 1 | 1 |
| fixed4 / T8 → fixed8 / T16 | 12.50% | 12.50% | +0.00 pp | [-6.56, +6.56] pp | 64 / 64 | 1 | 1 |

## Decisions

- development_eligible: False.
- confirmation_supported: not applicable.
- own_exit_task_floors_passed: False.
- d1_retention_passed: False.
- measured_exit_interval_supported: not applicable.
- confirmed_deep_gain_with_t8_cost: not applicable.

Best Fixed4 T4/T6/T8: 12.50%; candidate Fixed8/T16: 12.50%; gain +0.00 pp. The at-least-5pp margin is a DEV selection rule only.

## Training-exit floors and d1/T4 retention

- fixed4/T4 d1: 12.50% (floor 95%); passed=False.
- fixed4/T4 d2: 12.50% (floor 95%); passed=False.
- fixed4/T4 d6: 12.50% (floor 70%); passed=False.
- fixed4/T4 d8: 12.50% (floor 70%); passed=False.
- fixed8/T8 d1: 12.50% (floor 95%); passed=False.
- fixed8/T8 d2: 12.50% (floor 95%); passed=False.
- fixed8/T8 d6: 12.50% (floor 70%); passed=False.
- fixed8/T8 d8: 12.50% (floor 70%); passed=False.
- fixed4 d1/T4: 100.00% → 12.50%; drop +87.50 pp; retained=False.
- fixed8 d1/T4: 100.00% → 10.16%; drop +89.84 pp; retained=False.

## Costs at shallower measured exits

| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p | Holm p (5) |
|---|---:|---:|---:|---|---:|---:|---:|
| fixed4 / T4 → fixed8 / T4 | 12.50% | 12.50% | +0.00 pp | [-6.56, +6.56] pp | 64 / 64 | 1 | — |
| fixed4 / T8 → fixed8 / T8 | 12.50% | 12.50% | +0.00 pp | [-6.56, +6.56] pp | 64 / 64 | 1 | — |

Primary-group T8 drop exceeds 2pp: False. This cost flag alone establishes no gain or interval claim.

## Per-hop unrestricted / choice accuracy

| Hop | Model / exit | Unrestricted | Choice |
|---|---|---:|---:|
| 1 | initializer / T4 | 100.00% | 100.00% |
| 1 | initializer / T6 | 100.00% | 100.00% |
| 1 | initializer / T8 | 100.00% | 100.00% |
| 1 | initializer / T16 | 89.84% | 89.84% |
| 1 | fixed4 / T4 | 12.50% | 12.50% |
| 1 | fixed4 / T6 | 12.50% | 12.50% |
| 1 | fixed4 / T8 | 12.50% | 12.50% |
| 1 | fixed4 / T16 | 12.50% | 12.50% |
| 1 | fixed8 / T4 | 10.16% | 10.16% |
| 1 | fixed8 / T8 | 12.50% | 12.50% |
| 1 | fixed8 / T16 | 12.50% | 12.50% |
| 2 | initializer / T4 | 10.16% | 10.16% |
| 2 | initializer / T6 | 23.44% | 23.44% |
| 2 | initializer / T8 | 31.25% | 31.25% |
| 2 | initializer / T16 | 19.53% | 19.53% |
| 2 | fixed4 / T4 | 12.50% | 12.50% |
| 2 | fixed4 / T6 | 12.50% | 12.50% |
| 2 | fixed4 / T8 | 12.50% | 12.50% |
| 2 | fixed4 / T16 | 12.50% | 12.50% |
| 2 | fixed8 / T4 | 11.72% | 11.72% |
| 2 | fixed8 / T8 | 12.50% | 12.50% |
| 2 | fixed8 / T16 | 12.50% | 12.50% |
| 3 | initializer / T4 | 7.81% | 7.81% |
| 3 | initializer / T6 | 5.47% | 5.47% |
| 3 | initializer / T8 | 9.38% | 9.38% |
| 3 | initializer / T16 | 7.03% | 7.03% |
| 3 | fixed4 / T4 | 12.50% | 12.50% |
| 3 | fixed4 / T6 | 12.50% | 12.50% |
| 3 | fixed4 / T8 | 12.50% | 12.50% |
| 3 | fixed4 / T16 | 12.50% | 12.50% |
| 3 | fixed8 / T4 | 12.50% | 12.50% |
| 3 | fixed8 / T8 | 12.50% | 12.50% |
| 3 | fixed8 / T16 | 12.50% | 12.50% |
| 4 | initializer / T4 | 10.16% | 10.16% |
| 4 | initializer / T6 | 10.16% | 10.16% |
| 4 | initializer / T8 | 8.59% | 8.59% |
| 4 | initializer / T16 | 7.81% | 7.81% |
| 4 | fixed4 / T4 | 12.50% | 12.50% |
| 4 | fixed4 / T6 | 12.50% | 12.50% |
| 4 | fixed4 / T8 | 12.50% | 12.50% |
| 4 | fixed4 / T16 | 12.50% | 12.50% |
| 4 | fixed8 / T4 | 11.72% | 11.72% |
| 4 | fixed8 / T8 | 12.50% | 12.50% |
| 4 | fixed8 / T16 | 12.50% | 12.50% |
| 6 | initializer / T4 | 8.59% | 8.59% |
| 6 | initializer / T6 | 7.81% | 7.81% |
| 6 | initializer / T8 | 9.38% | 9.38% |
| 6 | initializer / T16 | 8.59% | 8.59% |
| 6 | fixed4 / T4 | 12.50% | 12.50% |
| 6 | fixed4 / T6 | 12.50% | 12.50% |
| 6 | fixed4 / T8 | 12.50% | 12.50% |
| 6 | fixed4 / T16 | 12.50% | 12.50% |
| 6 | fixed8 / T4 | 14.06% | 14.06% |
| 6 | fixed8 / T8 | 12.50% | 12.50% |
| 6 | fixed8 / T16 | 12.50% | 12.50% |
| 8 | initializer / T4 | 10.94% | 10.94% |
| 8 | initializer / T6 | 9.38% | 9.38% |
| 8 | initializer / T8 | 6.25% | 6.25% |
| 8 | initializer / T16 | 10.94% | 10.94% |
| 8 | fixed4 / T4 | 12.50% | 12.50% |
| 8 | fixed4 / T6 | 12.50% | 12.50% |
| 8 | fixed4 / T8 | 12.50% | 12.50% |
| 8 | fixed4 / T16 | 12.50% | 12.50% |
| 8 | fixed8 / T4 | 10.94% | 10.94% |
| 8 | fixed8 / T8 | 12.50% | 12.50% |
| 8 | fixed8 / T16 | 12.50% | 12.50% |
| 9 | initializer / T4 | 7.81% | 7.81% |
| 9 | initializer / T6 | 6.25% | 6.25% |
| 9 | initializer / T8 | 7.03% | 7.03% |
| 9 | initializer / T16 | 7.03% | 7.03% |
| 9 | fixed4 / T4 | 12.50% | 12.50% |
| 9 | fixed4 / T6 | 12.50% | 12.50% |
| 9 | fixed4 / T8 | 12.50% | 12.50% |
| 9 | fixed4 / T16 | 12.50% | 12.50% |
| 9 | fixed8 / T4 | 14.06% | 14.06% |
| 9 | fixed8 / T8 | 12.50% | 12.50% |
| 9 | fixed8 / T16 | 12.50% | 12.50% |
| 10 | initializer / T4 | 7.81% | 7.81% |
| 10 | initializer / T6 | 5.47% | 5.47% |
| 10 | initializer / T8 | 7.81% | 7.81% |
| 10 | initializer / T16 | 7.81% | 7.81% |
| 10 | fixed4 / T4 | 12.50% | 12.50% |
| 10 | fixed4 / T6 | 12.50% | 12.50% |
| 10 | fixed4 / T8 | 12.50% | 12.50% |
| 10 | fixed4 / T16 | 12.50% | 12.50% |
| 10 | fixed8 / T4 | 13.28% | 13.28% |
| 10 | fixed8 / T8 | 12.50% | 12.50% |
| 10 | fixed8 / T16 | 12.50% | 12.50% |
| 11 | initializer / T4 | 8.59% | 8.59% |
| 11 | initializer / T6 | 8.59% | 8.59% |
| 11 | initializer / T8 | 9.38% | 9.38% |
| 11 | initializer / T16 | 7.81% | 7.81% |
| 11 | fixed4 / T4 | 12.50% | 12.50% |
| 11 | fixed4 / T6 | 12.50% | 12.50% |
| 11 | fixed4 / T8 | 12.50% | 12.50% |
| 11 | fixed4 / T16 | 12.50% | 12.50% |
| 11 | fixed8 / T4 | 11.72% | 11.72% |
| 11 | fixed8 / T8 | 12.50% | 12.50% |
| 11 | fixed8 / T16 | 12.50% | 12.50% |
| 12 | initializer / T4 | 12.50% | 12.50% |
| 12 | initializer / T6 | 11.72% | 11.72% |
| 12 | initializer / T8 | 10.94% | 10.94% |
| 12 | initializer / T16 | 11.72% | 11.72% |
| 12 | fixed4 / T4 | 12.50% | 12.50% |
| 12 | fixed4 / T6 | 12.50% | 12.50% |
| 12 | fixed4 / T8 | 12.50% | 12.50% |
| 12 | fixed4 / T16 | 12.50% | 12.50% |
| 12 | fixed8 / T4 | 10.94% | 10.94% |
| 12 | fixed8 / T8 | 12.50% | 12.50% |
| 12 | fixed8 / T16 | 12.50% | 12.50% |

NLL, answer mass, tie diagnostics, secondary paired statistics and all group comparisons are retained in JSON.

Unrestricted correctness is primary; choice accuracy and other scores are secondary. Paired intervals and exact McNemar reuse the existing implementation. Holm adjustment is across the five prespecified contrasts, separately for each group and correctness field; only primary d9–12 unrestricted correctness controls confirmation. No simultaneous coverage claim is made for the intervals or exploratory groups. Floors and retention are observed thresholds, not formal noninferiority tests. The 5pp margin applies only to DEV selection.

This comparator validates complete matched prediction membership, metadata, endpoints, answer balance and evaluator-v2 score structure. It does not validate training weights, final-budget completion, source identity, original dataset identity or reported summary aggregates. The caller must bind those independently before using these decisions or scoring confirmation; this tool neither authorizes nor performs scoring.

