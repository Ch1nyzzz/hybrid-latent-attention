# Pointer v3 data

Generation seed: **17601**. Every question retains the original 25-node single
cycle, random two-letter labels, shuffled unindented edge lines and randomized
A–H answer mapping. Prompts end in `Answer:` with no reasoning target.

|Split|Requested hops|Rows per hop|Total|A–H per hop|
|---|---|---:|---:|---:|
|train|1, 2, 3, 4, 6, 8|4,000|24,000|500 each|
|dev|1, 2, 3, 4, 6, 8, 9, 10, 11, 12|128|1,280|16 each|
|sealed test|same ten depths|512|5,120|64 each|

The test primary group is unseen hops **9–12** (2,048 questions). Seen guards
are **d1** (512) and **d6/d8** (1,024); other depths remain separately available.
No test example is used for model scoring or selection during preparation.
Training, checkpoint selection and evaluation gates belong to the separate
experiment protocol, which this preparation does not modify.

The verified public generator supplies the train and seen evaluation streams.
Its independently shuffled OOD stream supplies unseen lengths. Within each
unseen hop/answer cell, the first 16 questions become dev and the remaining
64 become sealed test, changing only the split label. Final split mixing seeds
are 17702/17703/17704 for train/dev/test. Graph and semantic query IDs are preserved.

All **57,312** prior graph identities are excluded: every v1 and v2 split,
diagnostic-onehop train/dev, and extrapolation-dev. Sealed reference files
contribute only stored identity keys; their prompts and answers are not solved
or scored. Memorization and formatting variants are covered through their
onehop parent instances. The 30,400 new rows are also mutually disjoint by graph
identity within and across splits.

Every persisted row is independently parsed and solved, with an additional
25-node cycle walk checking the gold distance. Counts, exact answer balance,
template, graph/query identities and saved ID order are checked for every split.
The manifest records the audit and each split's content digest for transfer
integrity. Existing output is refused and staged data is published only after
its audit passes.

```bash
python3 -m ouro_depth.prepare_v3_data --root .
python3 -m ouro_depth.prepare_v3_data --root . --verify-only
```

Character lengths remain 387 for single-digit hops and 388 for two-digit hops;
equal token lengths are not claimed. At maximum depth12, the reverse cycle
path requires13 edges, so it is not a shorter solution. Dataset consistency
does not establish task learning, useful deeper inference or transfer to
natural-language reasoning.
