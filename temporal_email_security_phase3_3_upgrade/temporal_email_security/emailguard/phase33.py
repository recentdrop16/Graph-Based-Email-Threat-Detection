"""Repeat Phase 3.2 with longer training and report variability across random seeds.

Thresholds are selected on validation only by Phase 3.2; test results are untouched.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .phase32 import run

MODELS = ('tfidf_logistic', 'text_only', 'temporal_gnn', 'gnn_no_history')
METRICS = ('precision', 'recall', 'f1', 'false_positive_rate')


def summarize(reports):
    if not reports:
        raise ValueError('At least one experiment is required')
    result = {'runs': len(reports), 'seeds': [r['seed'] for r in reports], 'models': {}}
    for name in MODELS:
        values = {}
        for metric in METRICS:
            observed = [float(r['models'][name]['test'][metric]) for r in reports]
            values[metric] = {'mean': float(np.mean(observed)),
                              'std': float(np.std(observed, ddof=1)) if len(observed) > 1 else 0.0,
                              'per_seed': observed}
        values['validation_selected_thresholds'] = [float(r['models'][name]['validation']['threshold']) for r in reports]
        result['models'][name] = values
    return result


def main():
    p = argparse.ArgumentParser(description='Multi-seed follow-up with longer training and validation-selected thresholds')
    p.add_argument('--csv', default='sample_data/challenge.csv')
    p.add_argument('--generate-challenge', action='store_true')
    p.add_argument('--encoder', choices=('minilm', 'hash'), default='minilm')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--users', type=int, default=12)
    p.add_argument('--messages', type=int, default=36)
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--chunk-size', type=int, default=12)
    p.add_argument('--output', default='artifacts/phase33_summary.json')
    args = p.parse_args()
    if args.epochs < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        p.error('epochs must be positive and seeds must be distinct')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed in args.seeds:
        print(f'\n===== Seed {seed} =====', flush=True)
        per_seed = output.with_name(f'{output.stem}_seed{seed}.json')
        options = vars(args).copy()
        options.update(seed=seed, output=str(per_seed))
        # When generating, each seed gets its own dataset file so the original remains intact.
        if args.generate_challenge:
            source = Path(args.csv)
            options['csv'] = str(source.with_name(f'{source.stem}_seed{seed}{source.suffix}'))
        reports.append(run(SimpleNamespace(**options)))
    summary = summarize(reports)
    summary.update({'epochs': args.epochs, 'encoder': args.encoder,
                    'synthetic': bool(args.generate_challenge),
                    'limitations': [
                        'Synthetic generated data are not real-world validation',
                        'Training lasts a fixed number of epochs; early stopping is not implemented',
                        'Thresholds are chosen from validation data independently per seed and model',
                        'Template-separated partitions may still share users and semantic patterns',
                        'No guarantee that temporal history improves performance']})
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nMEAN TEST RESULTS (standard deviation across seeds)')
    for name, item in summary['models'].items():
        f1, fpr = item['f1'], item['false_positive_rate']
        print(f'{name:18s} F1={f1["mean"]:.3f} +/- {f1["std"]:.3f} FPR={fpr["mean"]:.3f} +/- {fpr["std"]:.3f}')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
