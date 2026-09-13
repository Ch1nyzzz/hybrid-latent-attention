# Additional length-extrapolation DEV

`data/extrapolation-dev/dev.jsonl` contains 512 pointer questions: 128 each at
9, 10, 11 and 12 hops, with exactly 16 answers per letter A–H in every stratum.
Seed: **17501**. The original 25-node single-cycle graph, random two-letter
labels, shuffled unindented edges and answer-only prompt are preserved. At
12 hops, the reverse path still requires 13 edges. Nine-hop prompts have 387
characters; the two-digit depths have 388. No tokenizer-length equality is claimed.

This is **additional development data**, created to probe lengths beyond v2's
supervised range. It neither replaces sealed test/OOD nor changes the active
training data, final-candidate selection or primary confirmation criterion.
Its planned diagnostic is the two final 2B checkpoints at inference loops
4/6/8/12/16, only after both jobs complete. Do not score intermediate peaks.
No model scoring is part of preparation.

The preparer reuses the verified public generator's OOD random stream and
filters pointer questions; it then changes only the split label to `dev`.
This implementation detail does not make the probe a sealed OOD evaluation.
All canonical graph identities from v1, diagnostic-onehop and every v2 split
are excluded. Sealed reference files contribute only stored identity keys;
their prompts and answers are not solved or scored. Memorization and format
probes are already covered through their diagnostic-onehop parent instances.

Every persisted prompt is independently parsed and solved. A separate cycle
walk checks all 25 unique positions, the gold distance, counts, exact answer
balance, graph/query uniqueness and reference exclusion. Generated and persisted
ID order must match exactly. The manifest records these checks, a content digest
for the generated DEV and the development-only usage policy. Existing output
directories are refused; generation stages data until its persisted audit passes.

```bash
python3 -m ouro_depth.prepare_extrapolation_dev --root .
python3 -m ouro_depth.prepare_extrapolation_dev --root . --verify-only
```

The preparation audit establishes dataset consistency, not task learning or
beneficial deeper inference. Results from this additional DEV remain exploratory.
