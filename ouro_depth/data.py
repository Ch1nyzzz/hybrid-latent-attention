"""Deterministic, verifiable next-token reasoning data for recurrent depth studies.

The generator solves while constructing each instance. The verifier reparses the
actual prompt and solves it independently, without consulting the stored answer
or construction trace. Only prompt and answer should be passed to a model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
from typing import Any, Iterable


SCHEMA_VERSION = 1
FAMILIES = ("pointer_chasing", "modular_arithmetic")
LETTERS = "ABCDEFGH"
DEFAULT_TRAIN_DIFFICULTIES = (1, 2, 3, 4, 6, 8)
DEFAULT_OOD_DIFFICULTIES = (10, 12)
MODULUS = 17
LABELS = tuple(a + b for a in "abcdefghijklmnopqrstuvwxyz" for b in "abcdefghijklmnopqrstuvwxyz")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _key(value: Any) -> str:
    # Hashing is for order-independent semantic identity and duplicate detection.
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _instance_key(family: str, facts: dict[str, Any]) -> str:
    return _key({"family": family, "facts": facts})


def _choices(rng: random.Random, correct: Any, universe: Iterable[Any], answer: str) -> dict[str, Any]:
    distractors = rng.sample([value for value in universe if value != correct], 7)
    rng.shuffle(distractors)
    choices: dict[str, Any] = {}
    for letter in LETTERS:
        choices[letter] = correct if letter == answer else distractors.pop()
    return choices


def _pointer_instance(rng: random.Random, difficulty: int, answer: str, context_size: int) -> dict[str, Any]:
    nodes = rng.sample(LABELS, context_size)
    # A single cycle prevents sinks, shortcuts from self-loops, and early repeats.
    edges = [[nodes[i], nodes[(i + 1) % context_size]] for i in range(context_size)]
    start_index = rng.randrange(context_size)
    start = nodes[start_index]
    path = [nodes[(start_index + i) % context_size] for i in range(difficulty + 1)]
    correct = path[-1]
    choices = _choices(rng, correct, nodes, answer)
    rng.shuffle(edges)
    prompt = (
        f"Follow exactly {difficulty} directed links from {start}. Every node has one outgoing link.\n"
        "Links:\n" + "\n".join(f"{src} -> {dst}" for src, dst in edges)
        + "\nWhich node do you reach?\n"
        + "\n".join(f"{letter}) {choices[letter]}" for letter in LETTERS)
        + "\nAnswer:"
    )
    facts = {"edges": sorted(edges)}
    return {
        "prompt": prompt,
        "answer": answer,
        "answer_value": correct,
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "facts": facts,
            "query": {"start": start, "hops": difficulty},
            "context_size": context_size,
            "choices": choices,
            "construction_trace": path,
        },
    }


def _arithmetic_instance(rng: random.Random, difficulty: int, answer: str, context_size: int) -> dict[str, Any]:
    variables = rng.sample(LABELS, context_size + 1)
    initial = rng.randrange(MODULUS)
    base = [variables[0], initial]
    equations: list[list[Any]] = []
    values = [initial]
    trace: list[dict[str, Any]] = [{"variable": variables[0], "value": initial}]
    for i in range(1, context_size + 1):
        # Invertible, non-identity multipliers retain dependence on earlier values.
        multiplier = rng.randrange(2, MODULUS)
        bias = rng.randrange(MODULUS)
        equations.append([variables[i], multiplier, variables[i - 1], bias])
        values.append((multiplier * values[-1] + bias) % MODULUS)
        trace.append({"variable": variables[i], "value": values[-1]})
    query = variables[difficulty]
    correct = values[difficulty]
    choices = _choices(rng, correct, range(MODULUS), answer)
    statements = [f"{base[0]} = {base[1]:02d}"] + [
        f"{dst} = ({multiplier:02d} * {src} + {bias:02d}) mod {MODULUS}"
        for dst, multiplier, src, bias in equations
    ]
    rng.shuffle(statements)
    prompt = (
        f"All values are integers modulo {MODULUS}. Each equation defines its left-hand variable.\n"
        "Equations:\n" + "\n".join(statements)
        + f"\nWhat is the value of {query}?\n"
        + "\n".join(f"{letter}) {choices[letter]:02d}" for letter in LETTERS)
        + "\nAnswer:"
    )
    facts = {"base": base, "equations": sorted(equations), "modulus": MODULUS}
    return {
        "prompt": prompt,
        "answer": answer,
        "answer_value": correct,
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "facts": facts,
            "query": {"variable": query},
            "context_size": context_size,
            "choices": choices,
            "construction_trace": trace[:difficulty + 1],
        },
    }


def _parse_choices(text: str, numeric: bool) -> dict[str, Any]:
    lines = text.splitlines()
    if len(lines) != 9 or lines[-1] != "Answer:":
        raise ValueError("Prompt must have eight options followed by exactly 'Answer:'")
    choices = {}
    pattern = r"([A-H])\) ([0-9]{2})" if numeric else r"([A-H])\) ([a-z]{2})"
    for expected, line in zip(LETTERS, lines[:-1]):
        match = re.fullmatch(pattern, line)
        if not match or match[1] != expected:
            raise ValueError("Options must occur exactly once in A-H order")
        choices[expected] = int(match[2]) if numeric else match[2]
    if len(set(choices.values())) != 8:
        raise ValueError("Options must contain eight distinct values")
    return choices


def solve_prompt(prompt: str, family: str) -> dict[str, Any]:
    """Independently parse and solve rendered text; raise ValueError if malformed."""
    if family == "pointer_chasing":
        match = re.fullmatch(
            r"Follow exactly ([0-9]+) directed links from ([a-z]{2})\. Every node has one outgoing link\.\n"
            r"Links:\n(.+)\nWhich node do you reach\?\n(.+)", prompt, re.DOTALL,
        )
        if not match:
            raise ValueError("Malformed pointer prompt")
        hops, start = int(match[1]), match[2]
        edges: dict[str, str] = {}
        for line in match[3].splitlines():
            edge = re.fullmatch(r"([a-z]{2}) -> ([a-z]{2})", line)
            if not edge or edge[1] in edges:
                raise ValueError("Malformed or repeated pointer source")
            edges[edge[1]] = edge[2]
        if set(edges) != set(edges.values()) or start not in edges or not 1 <= hops < len(edges):
            raise ValueError("Pointer facts must be a permutation with a valid query")
        # Check the full graph, not only the queried prefix.
        visited = set()
        node = start
        while node not in visited:
            visited.add(node)
            node = edges[node]
        if len(visited) != len(edges) or node != start:
            raise ValueError("Pointer graph must contain one full cycle")
        node = start
        trace = [node]
        for _ in range(hops):
            node = edges[node]
            trace.append(node)
        choices = _parse_choices(match[4], numeric=False)
        if not set(choices.values()) <= set(edges):
            raise ValueError("Pointer options must be nodes in the supplied graph")
        facts = {"edges": sorted([src, dst] for src, dst in edges.items())}
        query = {"start": start, "hops": hops}
        value, difficulty, context_size = node, hops, len(edges)
    elif family == "modular_arithmetic":
        match = re.fullmatch(
            r"All values are integers modulo 17\. Each equation defines its left-hand variable\.\n"
            r"Equations:\n(.+)\nWhat is the value of ([a-z]{2})\?\n(.+)", prompt, re.DOTALL,
        )
        if not match:
            raise ValueError("Malformed arithmetic prompt")
        bases: dict[str, int] = {}
        definitions: dict[str, tuple[int, str, int]] = {}
        for line in match[1].splitlines():
            base_match = re.fullmatch(r"([a-z]{2}) = ([0-9]{2})", line)
            expr_match = re.fullmatch(r"([a-z]{2}) = \(([0-9]{2}) \* ([a-z]{2}) \+ ([0-9]{2})\) mod 17", line)
            if base_match:
                var, number = base_match[1], int(base_match[2])
                if var in bases or var in definitions or not 0 <= number < MODULUS:
                    raise ValueError("Duplicate or invalid base definition")
                bases[var] = number
            elif expr_match:
                var, multiplier, source, bias = expr_match[1], int(expr_match[2]), expr_match[3], int(expr_match[4])
                if var in bases or var in definitions or not 2 <= multiplier < MODULUS or not 0 <= bias < MODULUS:
                    raise ValueError("Duplicate or invalid arithmetic definition")
                definitions[var] = (multiplier, source, bias)
            else:
                raise ValueError("Unrecognized arithmetic statement")
        if len(bases) != 1:
            raise ValueError("Arithmetic instance must have one base")
        cache = {name: (number, 0, [{"variable": name, "value": number}]) for name, number in bases.items()}

        def resolve(variable: str, active: frozenset[str] = frozenset()) -> tuple[int, int, list[dict[str, Any]]]:
            if variable in cache:
                return cache[variable]
            if variable in active or variable not in definitions:
                raise ValueError("Cycle or undefined arithmetic dependency")
            multiplier, source, bias = definitions[variable]
            source_value, source_depth, source_trace = resolve(source, active | {variable})
            result = (multiplier * source_value + bias) % MODULUS
            cache[variable] = (result, source_depth + 1, source_trace + [{"variable": variable, "value": result}])
            return cache[variable]

        for variable in definitions:
            resolve(variable)
        # Every dependency depth occurs once: this is one chain with irrelevant tail equations.
        if sorted(cache[var][1] for var in definitions) != list(range(1, len(definitions) + 1)):
            raise ValueError("Arithmetic definitions must form one chain")
        value, difficulty, trace = resolve(match[2])
        if difficulty == 0:
            raise ValueError("The arithmetic query must require at least one operation")
        choices = _parse_choices(match[3], numeric=True)
        if any(not 0 <= option < MODULUS for option in choices.values()):
            raise ValueError("Arithmetic option is outside the residue range")
        facts = {
            "base": list(next(iter(bases.items()))),
            "equations": sorted([dst, *definition] for dst, definition in definitions.items()),
            "modulus": MODULUS,
        }
        query = {"variable": match[2]}
        context_size = len(definitions)
    else:
        raise ValueError(f"Unknown family: {family}")
    matching = [letter for letter, option in choices.items() if option == value]
    if len(matching) != 1:
        raise ValueError("The correct result must occur in exactly one option")
    return {
        "answer": matching[0], "answer_value": value, "difficulty": difficulty,
        "facts": facts, "query": query, "choices": choices, "context_size": context_size,
        "construction_trace": trace, "instance_key": _instance_key(family, facts),
    }


def verify_row(row: dict[str, Any]) -> dict[str, Any]:
    """Verify prompt, label, difficulty, metadata and canonical identity together."""
    solved = solve_prompt(row["prompt"], row["family"])
    for field in ("answer", "answer_value", "difficulty"):
        if row[field] != solved[field]:
            raise ValueError(f"Stored {field} disagrees with independently solved prompt")
    metadata = row["metadata"]
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported schema version")
    for field in ("facts", "query", "choices", "context_size", "construction_trace", "instance_key"):
        if metadata[field] != solved[field]:
            raise ValueError(f"Metadata {field} disagrees with prompt")
    expected_id = _key({"instance_key": solved["instance_key"], "query": solved["query"]})[:24]
    if row["id"] != expected_id:
        raise ValueError("Row id does not match canonical semantic query")
    if row["split"] not in ("train", "dev", "test", "ood"):
        raise ValueError("Unknown split")
    return solved


def _split_rng(seed: int, split: str) -> random.Random:
    # Separate streams mean changing test_count does not change the training set.
    return random.Random(int(_key({"seed": seed, "split": split}), 16))


def _make_split(
    split: str, count: int, difficulties: tuple[int, ...], seed: int,
    context_sizes: dict[str, int], used_instances: set[str],
) -> list[dict[str, Any]]:
    rng = _split_rng(seed, split)
    strata = [(family, difficulty) for family in FAMILIES for difficulty in difficulties]
    # Randomized remainder allocation avoids privileging easy tasks for small counts.
    rng.shuffle(strata)
    rows = []
    for index, (family, difficulty) in enumerate(strata):
        stratum_count = count // len(strata) + int(index < count % len(strata))
        offset = rng.randrange(8)
        answers = [LETTERS[(i + offset) % 8] for i in range(stratum_count)]
        rng.shuffle(answers)
        generator = _pointer_instance if family == "pointer_chasing" else _arithmetic_instance
        for answer in answers:
            while True:
                row = generator(rng, difficulty, answer, context_sizes[family])
                instance_key = _instance_key(family, row["metadata"]["facts"])
                if instance_key not in used_instances:
                    break
            used_instances.add(instance_key)
            row.update({"family": family, "difficulty": difficulty, "split": split})
            row["metadata"]["instance_key"] = instance_key
            row["id"] = _key({"instance_key": instance_key, "query": row["metadata"]["query"]})[:24]
            verify_row(row)
            rows.append(row)
    rng.shuffle(rows)
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['family']}/d{row['difficulty']}"].append(row)
    lengths = [len(row["prompt"]) for row in rows]
    return {
        "count": len(rows),
        "answer_counts": dict(sorted(Counter(row["answer"] for row in rows).items())),
        "prompt_characters": {"min": min(lengths, default=0), "max": max(lengths, default=0), "mean": statistics.mean(lengths) if lengths else 0},
        "strata": {
            key: {"count": len(group), "answer_counts": dict(sorted(Counter(row["answer"] for row in group).items()))}
            for key, group in sorted(grouped.items())
        },
    }


def generate_dataset(
    output_dir: str | Path, train_count: int = 12000, dev_count: int = 600,
    test_count: int = 1200, ood_count: int = 600, seed: int = 1729,
    train_difficulties: Iterable[int] = DEFAULT_TRAIN_DIFFICULTIES,
    ood_difficulties: Iterable[int] = DEFAULT_OOD_DIFFICULTIES,
    context_size: int = 16,
) -> dict[str, Any]:
    """Write JSONL splits and a manifest; return the manifest.

    Counts are exact. Family/difficulty counts differ by at most one within a
    split; A-H counts differ by at most one within each family/difficulty cell.
    Token lengths must be measured separately with the model's actual tokenizer.
    """
    train_difficulties, ood_difficulties = tuple(train_difficulties), tuple(ood_difficulties)
    counts = {"train": train_count, "dev": dev_count, "test": test_count, "ood": ood_count}
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts.values()):
        raise ValueError("Split counts must be nonnegative integers")
    for values in (train_difficulties, ood_difficulties):
        if not values or len(values) != len(set(values)) or any(not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in values):
            raise ValueError("Difficulty sets must be nonempty, distinct positive integers")
    if set(train_difficulties) & set(ood_difficulties):
        raise ValueError("OOD and training difficulty sets must be disjoint")
    if min(ood_difficulties) <= max(train_difficulties):
        raise ValueError("OOD must require greater dependency depth than training")
    if not isinstance(context_size, int) or context_size < 8 or context_size <= max((*train_difficulties, *ood_difficulties)) or context_size + 1 > len(LABELS):
        raise ValueError("context_size must exceed every difficulty and provide at least eight choices")
    # A short directed cycle permits solving k hops with only N-k inverse steps.
    # Keep N > 2*k for every train and held-out query so the forward path is shorter.
    pointer_context_size = max(context_size, 2 * max((*train_difficulties, *ood_difficulties)) + 1)
    if pointer_context_size > len(LABELS):
        raise ValueError("Pointer context requires more unique labels than are available")
    context_sizes = {"pointer_chasing": pointer_context_size, "modular_arithmetic": context_size}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    used_instances: set[str] = set()
    summaries = {}
    for split, count in counts.items():
        rows = _make_split(split, count, ood_difficulties if split == "ood" else train_difficulties, seed, context_sizes, used_instances)
        destination = output / f"{split}.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(row) + "\n")
        temporary.replace(destination)
        summaries[split] = _summarize(rows)
    manifest = {
        "schema_version": SCHEMA_VERSION, "seed": seed,
        "families": list(FAMILIES), "train_difficulties": list(train_difficulties),
        "ood_difficulties": list(ood_difficulties), "context_size": context_size,
        "context_size_by_family": context_sizes,
        "modulus": MODULUS, "splits": summaries,
        "verification": {"all_rows_independently_solved": True, "unique_underlying_instances": len(used_instances), "instance_overlap": 0},
        "token_lengths": "Not measured here; validate using the actual Ouro/Huginn tokenizer before training.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_dataset(output_dir: str | Path) -> dict[str, Any]:
    """Re-read all files, independently verify examples and check cross-split overlap."""
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema version")
    if min(manifest["ood_difficulties"]) <= max(manifest["train_difficulties"]):
        raise ValueError("Manifest OOD difficulty must exceed training difficulty")
    seen_instances: set[str] = set()
    seen_ids: set[str] = set()
    summaries = {}
    for split in ("train", "dev", "test", "ood"):
        rows = []
        allowed_depths = set(manifest["ood_difficulties"] if split == "ood" else manifest["train_difficulties"])
        with (output / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                solved = verify_row(row)
                if row["split"] != split or row["difficulty"] not in allowed_depths:
                    raise ValueError("Row appears in the wrong split")
                if solved["context_size"] != manifest["context_size_by_family"][row["family"]]:
                    raise ValueError("Context size differs from the dataset specification")
                if row["family"] == "pointer_chasing" and 2 * row["difficulty"] >= solved["context_size"]:
                    raise ValueError("Pointer query has an equally short or shorter inverse-cycle solution")
                if solved["instance_key"] in seen_instances or row["id"] in seen_ids:
                    raise ValueError("Duplicate underlying instance or query, within or across splits")
                seen_instances.add(solved["instance_key"])
                seen_ids.add(row["id"])
                rows.append(row)
        summaries[split] = _summarize(rows)
        if summaries[split] != manifest["splits"][split]:
            raise ValueError(f"Manifest summary disagrees with {split} data")
        for stratum in summaries[split]["strata"].values():
            balance = [stratum["answer_counts"].get(letter, 0) for letter in LETTERS]
            if max(balance) - min(balance) > 1:
                raise ValueError("Answer letters are unbalanced within a difficulty/family cell")
        stratum_sizes = [
            summaries[split]["strata"].get(f"{family}/d{difficulty}", {"count": 0})["count"]
            for family in FAMILIES for difficulty in allowed_depths
        ]
        if max(stratum_sizes) - min(stratum_sizes) > 1:
            raise ValueError("Family/difficulty strata have unequal sample allocation")
    expected_verification = {
        "all_rows_independently_solved": True,
        "unique_underlying_instances": len(seen_instances), "instance_overlap": 0,
    }
    if manifest["verification"] != expected_verification:
        raise ValueError("Manifest verification claims disagree with observed data")
    return {"verified_rows": len(seen_ids), "unique_underlying_instances": len(seen_instances), "instance_overlap": 0, "splits": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--verify-dir", type=Path)
    parser.add_argument("--train-count", type=int, default=12000)
    parser.add_argument("--dev-count", type=int, default=600)
    parser.add_argument("--test-count", type=int, default=1200)
    parser.add_argument("--ood-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--train-difficulties", default="1,2,3,4,6,8")
    parser.add_argument("--ood-difficulties", default="10,12")
    parser.add_argument("--context-size", type=int, default=16)
    args = parser.parse_args()
    if args.verify_dir:
        result = verify_dataset(args.verify_dir)
    else:
        result = generate_dataset(
            args.output_dir, args.train_count, args.dev_count, args.test_count, args.ood_count,
            args.seed, tuple(map(int, args.train_difficulties.split(","))),
            tuple(map(int, args.ood_difficulties.split(","))), args.context_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
