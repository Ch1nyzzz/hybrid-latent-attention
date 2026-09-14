"""V10 corpus: paired long (R1 trace) / short (Numina solution) answers from OpenR1-Math-220k (PROTOCOL-v10.md §2).

  python -m ouro_depth.prepare_v10_data --model-path base_model --output-dir data/v10-cot [--seed 20260921]

Filters: math-word-problem only, short answer, both levels end in a boxed answer equivalent to the gold,
Ouro token budgets (prompt<=512, short 24..1024, long<=6144), problem dedup, 13-gram decontamination against
the five evaluation sets. Requires internet (datasets) and the local Ouro tokenizer.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SOURCE = ('open-r1/OpenR1-Math-220k', 'default')
EVAL_SETS = {'math500': 'HuggingFaceH4/MATH-500', 'aime24': 'HuggingFaceH4/aime_2024', 'aime25': 'math-ai/aime25',
             'hmmt_feb25': 'MathArena/hmmt_feb_2025', 'beyondaime': 'ByteDance-Seed/BeyondAIME'}
LIMITS = {'prompt': 512, 'short_min': 24, 'short_max': 1024, 'long_max': 6144, 'answer_chars': 40}
NGRAM = 13
TAIL_CHARS = 300


def answer_in_tail(tail, answer, g):
    normalized = g.normalize(answer)
    if re.fullmatch(r'-?\d+', normalized):
        return re.search(r'(?<![0-9.\-])' + re.escape(normalized) + r'(?![0-9.])', tail) is not None
    return normalized in g.normalize(tail)


def grader():
    path = Path(__file__).resolve().parent / 'matheval' / 'math_grader.py'
    spec = importlib.util.spec_from_file_location('math_grader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def words(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).split()


def ngrams(text):
    w = words(text)
    return {' '.join(w)} if len(w) < NGRAM else {' '.join(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def first_split(dataset):
    return dataset[sorted(dataset.keys())[0]]


def problem_text(row):
    for key in ('problem', 'question', 'Problem', 'Question'):
        if key in row:
            return row[key]
    raise KeyError(f'No problem field in {list(row)}')


def eval_ngrams():
    from datasets import load_dataset
    banned, counts = set(), {}
    for name, repo in EVAL_SETS.items():
        rows = first_split(load_dataset(repo))
        counts[name] = len(rows)
        for row in rows:
            banned |= ngrams(problem_text(row))
    return banned, counts


def select(example, g):
    """Return (reason, None) when dropped, else (None, candidate row without token counts)."""
    if example['question_type'] != 'math-word-problem':
        return 'question_type', None
    answer = example['answer']
    normalized = g.normalize(answer)
    if not normalized or len(normalized) > LIMITS['answer_chars']:
        return 'answer', None
    short = example['solution'].strip()
    boxed = g.last_boxed(short)
    appended = False
    if boxed is None:
        # Many Numina solutions state the answer in prose without \boxed{}; require the gold answer in the tail, then box it.
        if not answer_in_tail(short[-TAIL_CHARS:], answer, g):
            return 'short_no_answer', None
        short, appended = short + '\n\nThe final answer is \\boxed{' + answer.strip() + '}', True
    elif not g.is_equiv(boxed, answer):
        return 'short_boxed_mismatch', None
    index = next((i for i, ok in enumerate(example['correctness_math_verify'] or []) if ok), None)
    if index is None:
        return 'no_correct_generation', None
    long = example['generations'][index].strip()
    if '</think>' not in long:
        return 'no_think_close', None
    boxed = g.last_boxed(long)
    if boxed is None or not g.is_equiv(boxed, answer):
        return 'long_boxed', None
    return None, {'id': f"v10-{example['uuid']}", 'uuid': example['uuid'], 'source': example['source'],
                  'problem_type': example['problem_type'], 'problem': example['problem'].strip(), 'answer': answer,
                  'short': short, 'long': long, 'generation_index': index, 'short_boxed_appended': appended}


def prompt_for(tokenizer, problem):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': problem + INSTR}], tokenize=False, add_generation_prompt=True)


def count_tokens(tokenizer, rows):
    prompts = tokenizer([prompt_for(tokenizer, r['problem']) for r in rows], add_special_tokens=False)['input_ids']
    shorts = tokenizer([r['short'] for r in rows], add_special_tokens=False)['input_ids']
    longs = tokenizer([r['long'] for r in rows], add_special_tokens=False)['input_ids']
    for row, p, s, l in zip(rows, prompts, shorts, longs):
        row['n_prompt'], row['n_short'], row['n_long'] = len(p), len(s) + 1, len(l) + 1  # +1 for <|im_end|>


def length_ok(row):
    return (row['n_prompt'] <= LIMITS['prompt'] and LIMITS['short_min'] <= row['n_short'] <= LIMITS['short_max']
            and row['n_long'] <= LIMITS['long_max'])


def stats(rows, key):
    values = sorted(r[key] for r in rows)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return {'mean': round(sum(values) / len(values), 1), 'p50': pick(.5), 'p90': pick(.9), 'max': values[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=20260921)
    ap.add_argument('--train', type=int, default=24000)
    ap.add_argument('--dev', type=int, default=512)
    ap.add_argument('--limit', type=int, default=0, help='debug: only scan the first N source rows')
    args = ap.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'{output} exists; inspect it, do not overwrite')
    from datasets import load_dataset
    from transformers import AutoTokenizer
    g = grader()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    banned, eval_counts = eval_ngrams()
    source = load_dataset(*SOURCE, split='train')
    if args.limit:
        source = source.select(range(args.limit))
    reasons, seen, candidates = Counter(), set(), []
    for example in source:
        reason, row = select(example, g)
        if reason:
            reasons[reason] += 1
            continue
        key = ' '.join(words(row['problem']))
        if key in seen:
            reasons['duplicate'] += 1
            continue
        if ngrams(row['problem']) & banned:
            reasons['contaminated'] += 1
            continue
        seen.add(key)
        candidates.append(row)
    for offset in range(0, len(candidates), 1000):
        count_tokens(tokenizer, candidates[offset:offset + 1000])
    kept = [r for r in candidates if length_ok(r)]
    reasons['length'] += len(candidates) - len(kept)
    random.Random(args.seed).shuffle(kept)
    if len(kept) < args.train + args.dev:
        raise ValueError(f'Only {len(kept)} rows survive the filters; need {args.train + args.dev}')
    splits = {'dev': kept[:args.dev], 'train': kept[args.dev:args.dev + args.train]}
    output.mkdir(parents=True)
    hashes = {}
    for name, rows in splits.items():
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        (output / f'{name}.jsonl').write_text(text)
        hashes[name] = hashlib.sha256(text.encode()).hexdigest()
    manifest = {'dataset_type': 'cot_pair_v10', 'protocol': 'PROTOCOL-v10.md', 'source': SOURCE, 'seed': args.seed,
                'scanned': len(source), 'surviving': len(kept), 'dropped': dict(reasons), 'limits': LIMITS, 'ngram': NGRAM,
                'eval_sets': eval_counts, 'counts': {k: len(v) for k, v in splits.items()}, 'split_sha256': hashes,
                'lengths': {split: {key: stats(rows, key) for key in ('n_prompt', 'n_short', 'n_long')} for split, rows in splits.items()},
                'sources': dict(Counter(r['source'] for r in splits['train']))}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'sources'}, indent=1))


if __name__ == '__main__':
    main()
