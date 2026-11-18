"""Dataset quality and cross-split leakage audit; no network access."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from .data import load_events
from .phase3 import user_split


def _digest(row):
    # Canonicalize message content, not identifiers, to catch templated duplicates.
    payload = '\n'.join(' '.join(str(row[k]).casefold().split()) for k in ('subject', 'body'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def audit_dataset(csv, seed=42):
    df = load_events(csv)
    parts = dict(zip(('train', 'validation', 'test'), user_split(df, seed=seed)))
    hashes = {name: set(part.apply(_digest, axis=1)) for name, part in parts.items()}
    pairs = {}
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        overlap = hashes[a] & hashes[b]
        pairs[f'{a}_vs_{b}'] = {
            'shared_message_templates': len(overlap),
            'rows_in_' + a: int(parts[a].apply(_digest, axis=1).isin(overlap).sum()),
            'rows_in_' + b: int(parts[b].apply(_digest, axis=1).isin(overlap).sum()),
        }
    result = {
        'source': str(csv), 'seed': seed,
        'splits': {name: {'users': int(part.user_id.nunique()), 'rows': len(part),
                          'positive': int(part.label.sum()),
                          'unique_subject_body': len(hashes[name]),
                          'timestamp_min': str(part.timestamp.min()),
                          'timestamp_max': str(part.timestamp.max())}
                   for name, part in parts.items()},
        'cross_split_content_overlap': pairs,
        'warning': ('Shared normalized subject/body text across splits can inflate results. '
                    'This audit does not prove dataset provenance or checkpoint independence.'),
    }
    return result


def main():
    p = argparse.ArgumentParser(description='Audit dataset quality and cross-split message duplication')
    p.add_argument('--csv', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='artifacts/dataset_audit.json')
    args = p.parse_args()
    result = audit_dataset(args.csv, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'Results saved: {output}')


if __name__ == '__main__':
    main()
