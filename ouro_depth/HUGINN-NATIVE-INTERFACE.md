# Huginn native-input follow-up

Declared after the completed no-BOS calibration and before any native-BOS model
score. The original `raw-pretrained-path-fix` result remains a failed calibration:
all 512 next-token predictions were newline token10; d1/d2 raw accuracy was zero.
Restricted A–H accuracy was 12.5–14.8%. These are results for that exact interface,
not evidence that all formats or the architecture fail at one-hop reasoning.

The pinned official model card demonstrates plain completion with
`tokenizer.encode(..., add_special_tokens=True)`. Its tokenizer postprocessor
prepends `<|begin_text|>` (65504). Our first calibration reused Ouro's
`add_special_tokens=False`. This follow-up corrects that implementation choice.

Run exactly one separate native-BOS attempt on the same complete ordered 256
d1/d2 DEV rows at R32/R64. Preserve the original manifest, frozen attempt, scores,
and thresholds. The only input change is the official BOS postprocessor. Assert
that every native prompt is exactly `[65504] + original_ids`, contains one BOS,
fits L256, and retains the same canonical space-prefixed answer-token boundary.
No prompt wording, newline suffix, chat template, answer choices, or target
selection changes. No training, test data, generated reasoning, or model saving.

Use the same pinned checkpoint, FP32 parameters, BF16 autocast, B2, L256, no
cache/noise, and evaluation seed18931 with the existing per-question latent
helper. Within this attempt, R32 and R64 receive the exact same latent tensor.
BOS changes valid sequence length, so the new draw is not asserted to be
bitwise aligned to every content position in the no-BOS run. This is a corrected
interface readiness check, not a fully controlled causal estimate of BOS alone.

Readiness still requires unrestricted next-token accuracy d1>=95% and d2>=80%
at both depths. Choice accuracy, NLL and answer mass remain diagnostics; no
choice-only score replaces raw accuracy. Passing supports skipping a future
shared warmup only under this explicitly recorded native interface. Neither
result is a hard-task or deeper-training success. Original failure is retained.

If native BOS still fails, inspect the saved output tokens and record that
first-token behavior. Do not begin a prompt search or declare a format-only
explanation. A generated continuation diagnostic, if needed, requires a separate
bounded declaration and cannot retroactively pass the single-token calibration.
No shared warmup or scientific training is authorized by this protocol alone;
their design follows the complete Ouro final-depth evidence and the user's
existing experimental authorization.

This is a small interface change: verify actual pinned-tokenizer encodings on
the existing train/DEV rows without loading model weights, review the changed
branch, and reuse the already passed paired-latent/native-forward tests. Run on
allocated GPU4 only after the previously registered Ouro probe releases it.
Save source, declaration, outputs and completion status under a new attempt;
refuse overwrite and preserve any failed attempt without automatic restart.
