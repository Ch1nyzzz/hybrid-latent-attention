# V2 pointer data

This dataset supports a new task-learning curriculum while preserving the original
pointer task. It does not change any v1 or diagnostic file, and its preparation is
not evidence that deeper loops improve reasoning.

Seed: **17401**. Output: `data/v2-pointer`. Template: the original **25-node**
directed cycle with random two-letter node labels, shuffled edges, **unindented**
source lines, and a randomized A–H option mapping. No format intervention is
included. The graph is longer than twice the largest query depth, so a 12-hop
query cannot be solved with a shorter inverse-cycle path.

| Split | Dependency depths | Examples per depth | Total | Each answer letter per depth | Role |
|---|---|---:|---:|---:|---|
| Train | 1, 2, 3, 4, 6, 8 | 4,000 | 24,000 | 500 | Optimization |
| Dev | 1, 2, 3, 4, 6, 8 | 128 | 768 | 16 | Development decisions |
| IID test | 1, 2, 3, 4, 6, 8 | 512 | 3,072 | 64 | Sealed |
| OOD | 10, 12 | 1,024 | 2,048 | 128 | Sealed |

All **29,888** rows have unique underlying fact sets and semantic query IDs.
Canonical instance keys exclude prompt ordering and answer mapping, preventing
the same graph from crossing splits under a different presentation. Node renamings
remain fresh instances of the same abstract task class; this is not a split over
graph isomorphism classes. Prompt length is 387 characters for IID depths and
388 for the two-digit OOD depths. Real token lengths still vary with labels.

The exclusion index includes all four v1 splits and diagnostic-onehop train/dev,
covering **26,912** reference instances. The memorization and paired-format probes
are already covered because their instances are subsets of diagnostic-onehop.
Reference test/OOD files are read only for stored canonical identity keys. New
test/OOD examples are generated and independently verified, with no model scoring
or performance-based selection. They remain sealed while the curriculum and
candidate checkpoints are selected on training/development data.

The existing public generator and independent rendered-text solver are reused.
The generator temporarily produces both families; preparation retains only pointer
examples without changing them. Every persisted v2 row is reparsed and solved;
the audit checks exact answer balance by depth, the complete cycle, query depth,
metadata, identities, within/across-split disjointness, and reference exclusion.

```sh
python3 -m ouro_depth.prepare_v2_data --root .
python3 -m ouro_depth.prepare_v2_data --root . --verify-only
```

Generation refuses to overwrite an existing destination. Smaller per-depth counts
can be requested for a preparation test, but every count must be a positive
multiple of eight; the manifest records the actual counts. Training curricula,
model initialization, budget and loop schedules are specified separately. Positive
hard-task and same-checkpoint deeper-loop results are still unestablished.
