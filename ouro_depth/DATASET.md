# Controlled dependency-depth reasoning data

This dataset tests whether additional recurrent computation improves exact answers
as the number of required dependency steps increases. It is a mechanism experiment,
not evidence by itself for improved mathematical reasoning on natural problems.

## Interface

```python
from data import generate_dataset, verify_dataset

manifest = generate_dataset(
    "data", train_count=12000, dev_count=600, test_count=1200,
    ood_count=600, seed=1729,
    train_difficulties=(1, 2, 3, 4, 6, 8),
    ood_difficulties=(10, 12), context_size=16,
)
verification = verify_dataset("data")
```

```sh
python data.py --output-dir data
python data.py --verify-dir data
python -m unittest discover -s tests -p test_data.py -v
```

The output contains `train.jsonl`, `dev.jsonl`, `test.jsonl`, `ood.jsonl`, and
`manifest.json`. Counts, seed, difficulty sets, and context size are configurable.
Each split has a separate deterministic random stream. Changing held-out counts
does not change the training data. Generation replaces these five output files.

Every row contains `id`, `family`, `difficulty`, `split`, `prompt`, `answer`,
`answer_value`, and `metadata`. `answer` is exactly one uppercase letter A–H.
Prompts finish with `Answer:`; they request one selection and contain no reasoning
target. Only `prompt` and `answer` belong in model inputs and training targets.
Metadata includes the program/graph, query, choices, construction trace, and
canonical instance identity for auditing, and must never be included in prompts.

Check the exact tokenizer before training: appending the intended answer format
must yield a single target token and preserve the prompt token prefix. Do not
silently truncate inputs. This generator reports character lengths rather than
claiming model-specific token lengths. The arithmetic family has the longer
prompts and should be checked against the selected model's context limit.

## Two task families

**Pointer chasing.** A random permutation of 25 unique two-letter labels forms
one directed cycle. All 25 edges are shuffled in the prompt. From a random start,
follow exactly *d* edges and select the reached node from eight distinct options.
The options contain the answer and seven randomly selected different graph nodes.
Difficulty is the exact number of forward edge applications. The cycle length is
`max(context_size, 2 * maximum_requested_depth + 1)`, held fixed across all splits.
This prevents solving *d* forward hops using a shorter *N−d* inverse path. There
are also no self-loops, sinks, or early repeated nodes. The default maximum query
depth of 12 therefore needs 25 nodes even though `context_size` defaults to 16.

**Modular arithmetic.** A random base value and 16 affine equations form one
dependency chain over integers modulo 17. Each equation has the form
`vv = (aa * uu + bb) mod 17`. Multipliers are 2–16 and offsets are 0–16. Every
multiplier is invertible modulo 17, so no equation erases its input through a
zero multiplier. Variable names and statement order are randomized. The query
asks for the variable at dependency depth *d*; later equations are distractors.
The answer and seven other residues are presented as two-digit options. Difficulty
is the number of arithmetic equations between the base and the queried variable.

Each family keeps its full context size constant at every difficulty, including
OOD difficulty. The manifest records the actual counts in `context_size_by_family`:
25 edges for pointer chasing and 16 arithmetic operations by default. Arithmetic
prompts have exactly equal character length. Pointer
prompts differ only by the number of digits in the requested hop count. Actual
token counts may vary with random labels, and must be measured and reported by
family and difficulty using the actual tokenizer. The two families have different
prompt lengths and should also be reported separately.

## Split and shortcut controls

Training, development, and the IID test use dependency depths 1, 2, 3, 4, 6, and 8.
The OOD test uses 10 and 12. Each split allocates its count almost equally across
family/depth strata; counts differ by at most one. Correct answer letters are
balanced separately inside each stratum, with counts differing by at most one.
For the default training count, every family/depth stratum has 1,000 examples,
exactly 125 per answer letter. Development, IID test, and OOD have 50, 100, and
150 examples per stratum respectively.

An underlying instance key canonicalizes the full graph or full equation set.
It excludes statement order, answer choices, requested variable/start, and
difficulty. The same facts therefore cannot reappear in another split merely by
changing a query or reordering text. Row IDs additionally identify the semantic
query. Duplicate underlying instances are rejected within and across all splits.
Node renamings are fresh instances of the same abstract task class; the split is
not intended to hold out graph isomorphism classes or language templates.

The generator computes labels during construction. The independent verifier
reparses the rendered prompt, traverses the supplied graph or recursively resolves
the supplied equations, determines the answer option and dependency depth, and
checks them against every stored semantic field. It also validates distractors,
the full cycle/chain, the fixed context size, answer balance, manifests, and split
disjointness. The verifier never trusts the stored answer or construction trace to
solve a prompt. Hand-calculated fixtures test both solvers, including a cyclic
dependency failure, and mutation tests check that corrupted labels and facts fail.

## Interpretation and remaining limitations

Dependency depth is a property of the generated task, not proof of the model's
internal reasoning steps. A model may learn composition algorithms that use fewer
sequential operations; this is acceptable performance but does not identify its
mechanism. Chance accuracy is 12.5%. Balanced letters remove a fixed-position
baseline, but cannot rule out every learned shortcut. In modular arithmetic,
intermediate residues may coincide by chance and composed maps can occasionally
be simple. No claim is made that every example has an information-theoretically
irreducible sequential cost of *d*.

The OOD set tests additional dependency depth within the same templates, context
size, modulus, and task family. It does not establish cross-domain transfer. Use
the development split for pilot selection and stopping; reserve IID/OOD test
results for frozen configurations. Compare all recurrent depths on the same
held-out examples, report paired wrong-to-right and right-to-wrong transitions,
and report family/depth cells. Keep output to one answer token so extra textual
reasoning cannot explain a recurrent-depth effect. A matched training-compute
baseline and an unchanged-checkpoint depth sweep are still needed to attribute
improvement to training useful deeper recurrence.
