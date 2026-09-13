# Ouro paired comparison: development

Development diagnostics only; these flags are not held-out test confirmation or goal completion.

`fixed` always denotes the fixed4-trained model; numeric suffixes denote evaluation loops.

Rows: 768. Table: hard IID d=6/8, n=256.

| Comparison | Before | After | Gain | Approx. 95% CI | Wrong→right / right→wrong | Exact p |
|---|---:|---:|---:|---|---:|---:|
| curriculum_4_to_8 | 88.7% | 92.6% | +3.9% | [-2.8%, +10.5%] | 20 / 10 | 0.09874 |
| fixed4_to_curriculum8 | 96.9% | 92.6% | -4.3% | [-10.3%, +1.9%] | 7 / 18 | 0.04329 |
| fixed8_to_curriculum8 | 11.3% | 92.6% | +81.2% | [+73.0%, +86.4%] | 209 / 1 | 2.565e-61 |
| fixed4_to_8 | 96.9% | 11.3% | -85.5% | [-90.1%, -77.7%] | 1 / 220 | 1.318e-64 |
| fixed4_to_curriculum4 | 96.9% | 88.7% | -8.2% | [-14.7%, -1.4%] | 6 / 27 | 0.0003241 |
| initializer4_to_curriculum4 | 10.5% | 88.7% | +78.1% | [+69.1%, +84.2%] | 204 / 4 | 3.756e-55 |
| initializer4_to_fixed4 | 10.5% | 96.9% | +86.3% | [+78.9%, +90.4%] | 221 / 0 | 5.935e-67 |

D1 shallow retention allows an observed drop of 2 percentage points:
- curriculum: drop +0.0%; retained=True
- fixed: drop +0.0%; retained=True

Decision flags (development): {"cross_training_gain_positive": false, "d1_curriculum_retained": true, "d1_fixed_retained": true, "primary_available": true, "primary_gain_positive": false, "primary_unavailable_reason": null, "same_depth_training_gain_positive": true}

Restricted-choice secondary comparisons and every subgroup are in the JSON.

Intervals use the same conservative Bonferroni-Wilson approximation as the trainer; exact McNemar p-values are unadjusted across comparisons. Correctness uses evaluator-v2 deterministic token-ID tie-breaking.
