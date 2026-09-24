"""Summarize diagnostic suite outputs across all ranks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


def summarize_slices(files: List[Path]) -> Dict[str, Any]:
    all_slices = []
    for f in files:
        all_slices.extend(json.loads(f.read_text()))

    # Group metrics by variant across all records, layers, and loops
    variants = ['baseline', 'k_restored', 'v_restored', 'loop1_restored', 'prompt_restored', 'response_restored']
    stats = {v: {'kl': [], 'mse': []} for v in variants}

    for item in all_slices:
        for loop_result in item['slices']:
            for v in variants:
                if v in loop_result:
                    stats[v]['kl'].append(loop_result[v]['kl'])
                    stats[v]['mse'].append(loop_result[v]['mse'])

    summary = {}
    for v in variants:
        kls = stats[v]['kl']
        mses = stats[v]['mse']
        summary[v] = {
            'mean_kl': sum(kls) / max(len(kls), 1),
            'mean_mse': sum(mses) / max(len(mses), 1),
            'count': len(kls)
        }
    return summary


def summarize_oracle(files: List[Path]) -> Dict[str, Any]:
    all_oracles = []
    for f in files:
        all_oracles.extend(json.loads(f.read_text()))

    initial_losses = [x['initial_loss'] for x in all_oracles]
    final_losses = [x['final_loss'] for x in all_oracles]
    reductions = [x['loss_reduction_pct'] for x in all_oracles]

    return {
        'mean_initial_loss': sum(initial_losses) / max(len(initial_losses), 1),
        'mean_final_loss': sum(final_losses) / max(len(final_losses), 1),
        'mean_reduction_pct': sum(reductions) / max(len(reductions), 1),
        'probes': len(all_oracles)
    }


def summarize_credit(files: List[Path]) -> Dict[str, Any]:
    all_credits = []
    for f in files:
        all_credits.extend(json.loads(f.read_text()))

    keys = set()
    for item in all_credits:
        keys.update(item['metrics'].keys())

    summary = {}
    for k in sorted(keys):
        cos_sims = [item['metrics'][k]['cosine_similarity'] for item in all_credits if k in item['metrics']]
        rel_norms = [item['metrics'][k]['relative_norm'] for item in all_credits if k in item['metrics']]
        rel_errs = [item['metrics'][k]['relative_error'] for item in all_credits if k in item['metrics']]
        summary[k] = {
            'mean_cosine_similarity': sum(cos_sims) / max(len(cos_sims), 1),
            'mean_relative_norm': sum(rel_norms) / max(len(rel_norms), 1),
            'mean_relative_error': sum(rel_errs) / max(len(rel_errs), 1),
            'count': len(cos_sims)
        }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--world', type=int, default=8)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    slice_files = sorted(out_dir.glob('slice-rank-*.json'))
    oracle_files = sorted(out_dir.glob('oracle-rank-*.json'))
    credit_files = sorted(out_dir.glob('credit-rank-*.json'))

    full_summary = {}

    if slice_files:
        full_summary['slices'] = summarize_slices(slice_files)
        print("\n=== SLICE ORACLE RESTORATION DIAGNOSTIC ===")
        print(f"{'Variant':<20} | {'Mean Attention KL':<18} | {'Mean Relative Output MSE':<24}")
        print("-" * 68)
        for v, s in full_summary['slices'].items():
            print(f"{v:<20} | {s['mean_kl']:<18.6f} | {s['mean_mse']:<24.6f}")

    if oracle_files:
        full_summary['oracle'] = summarize_oracle(oracle_files)
        print("\n=== FREE LATENT ORACLE DIAGNOSTIC (SAME BUDGET) ===")
        o = full_summary['oracle']
        print(f"Probes: {o['probes']}")
        print(f"Initial Writer Loss: {o['mean_initial_loss']:.6f}")
        print(f"Free Latent Loss:    {o['mean_final_loss']:.6f}")
        print(f"Loss Reduction:      {o['mean_reduction_pct']:.2f}%")

    if credit_files:
        full_summary['prompt_credit'] = summarize_credit(credit_files)
        print("\n=== PROMPT CREDIT PROBE (K-HOP DETACH VS ATTACH) ===")
        print(f"{'Parameter Family':<15} | {'Cosine Sim':<12} | {'Norm Ratio (Det/Att)':<20} | {'Relative Diff':<15}")
        print("-" * 70)
        for k, s in full_summary['prompt_credit'].items():
            print(f"{k:<15} | {s['mean_cosine_similarity']:<12.4f} | {s['mean_relative_norm']:<20.4f} | {s['mean_relative_error']:<15.4f}")

    (out_dir / 'summary.json').write_text(json.dumps(full_summary, indent=2))
    print(f"\nSaved complete diagnostic summary to {out_dir / 'summary.json'}")


if __name__ == '__main__':
    main()
