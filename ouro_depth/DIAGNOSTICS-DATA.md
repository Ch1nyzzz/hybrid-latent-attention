# One-hop diagnostics: dataset roles and limits

The next diagnostic isolates learning the existing pointer-chasing task at one
hop. It does **not** replace the goal of learning useful deeper recurrence.

At update 100, each v1 arm has processed 1,600 examples, including only 131
pointer d1 and 149 arithmetic d1 examples (reconstructed from the recorded seed,
batch size and training order). Pointer d1 still requires a lookup among 25
shuffled edges, binding an arbitrary node label, and mapping that node to a random
A–H option. Arithmetic d1 additionally retrieves a base and evaluates an affine
expression modulo 17. Thus “one dependency step” does not mean an already learned
atomic skill. The v2 evaluations show near-chance answer accuracy and strong
answer-position bias; this motivates a task-learning diagnostic, not a conclusion
that extra loops are ineffective.

## Prepared data

Run from the experiment root:

```sh
python3 -m ouro_depth.prepare_diagnostics --root .
python3 -m ouro_depth.prepare_diagnostics --root . --verify-only
```

Generation refuses to overwrite existing diagnostic directories. Every output
split is verified from its rendered prompts using the existing independent solver.
The v1 datasets and generator are unchanged.

| Dataset | Train | Development | Relationship | Permitted interpretation |
|---|---:|---:|---|---|
| `data/diagnostic-onehop` | 12,000 | 512 | Distinct facts across splits | Held-out one-hop task learning |
| `data/diagnostic-memorize32` | 32 | 32 | Same 32 examples, intentionally repeated | Alignment/optimization and memorization only |

Generator seed is **17301**. The learning data are pointer-chasing d1 rows from
the existing verified generator. The format retains **25 nodes**, shuffled edges,
random two-letter labels, eight independently positioned answer options, and one
final answer token. Every prompt is 387 characters; the model's existing
tokenization check remains necessary. A–H counts are exactly 1,500 each in the
12,000 training rows and 64 each in the 512 held-out development rows.

Selection seed **17302** selects four examples per answer letter from the new
one-hop training set. The memorization train/dev files contain the exact same
prompts, answers, semantic identities and metadata, with only the split label
changed for dev. Each manifest explicitly marks this overlap and forbids calling
the memorization dev score held-out accuracy. The memorization examples never
overlap the genuine 512-example development split.

All new underlying instances are excluded against the canonical identity keys of
all four v1 splits. For v1 test/OOD, the preparation script reads only the stored
`metadata.instance_key` field for identity exclusion: no evaluation, optimization,
or answer-based selection is performed. No new test/OOD set is produced here.

## What the diagnostic can distinguish

A high repeated-example score only shows that the current pipeline and optimizer
can fit these inputs. It does not show an algorithm was learned. Failure to fit
also does not by itself prove an implementation bug: the learning rate, numerical
precision and optimization difficulty remain possible explanations.

Fresh-example one-hop accuracy well above 12.5%, together with diverse
input-dependent choices, is the meaningful next prerequisite. As a proposed
progression criterion, require at least **80% on the 512 held-out examples** and
check that shuffled option mappings and edge orders preserve performance. These
are development criteria, not confirmatory tests. Such nuisance-variant probes
are not included in the current prepared files and must be independently derived
and labeled if used.

After one-hop generalization is established, introduce higher dependency depths
while retaining easier examples. Keep d6/d8 development probes instance-disjoint
and unchanged; never train on their actual instances. Later loop-depth comparisons
must still use the same checkpoint and questions, matched training-budget
controls, and paired wrong→right/right→wrong counts. A one-hop-only warmup run
without a concurrent mixed-task control cannot isolate whether its gain came
from concentrated easy exposure, additional optimization, or reduced task-family
interference. Record that limitation rather than treating the diagnostic as a
causal curriculum result.

Training budgets, checkpoint initialization, loop counts, and stopping rules are
specified separately by the experiment protocol. Preparing these files launches
no training.
