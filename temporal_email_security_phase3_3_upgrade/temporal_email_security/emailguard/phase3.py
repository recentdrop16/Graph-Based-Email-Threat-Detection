"""Phase 3: explicit-label CSV normalization and leakage-aware model evaluation.

No labels are inferred from message wording or folder names. No remote data is fetched.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix,
                             balanced_accuracy_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .data import COLUMNS, TextEncoder, attach_embeddings, load_events
from .model import TemporalEmailNet, TextOnlyNet, GraphState


def normalize_csv(source, output, mapping, positive='1', negative='0'):
    """Map explicitly named input columns to the canonical schema."""
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = {'user_id', 'timestamp', 'sender', 'recipient', 'label'}
    if required - mapping.keys():
        raise ValueError(f'Missing mappings: {sorted(required - mapping.keys())}')
    missing = [src for src in mapping.values() if src not in raw.columns]
    if missing:
        raise ValueError(f'Source columns not found: {missing}')
    if positive == negative:
        raise ValueError('Positive and negative label values must differ')
    result = pd.DataFrame({name: raw[src] for name, src in mapping.items()})
    for optional in ('subject', 'body'):
        if optional not in result:
            result[optional] = ''
    labels = result.label.str.strip()
    unexpected = sorted(set(labels) - {positive, negative})
    if unexpected:
        raise ValueError(f'Unknown labels {unexpected[:10]}; supply explicit --positive and --negative')
    result['label'] = (labels == positive).astype(int)
    result = result[COLUMNS]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    load_events(out)  # validate timestamps and schema
    return len(result)


def user_split(df, seed=42, train_fraction=.6, validation_fraction=.2):
    """Three disjoint user partitions. Require at least 3 users."""
    users = np.array(sorted(df.user_id.astype(str).unique()))
    if len(users) < 3:
        raise ValueError('At least three distinct user IDs required for train/validation/test')
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n = len(users)
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * validation_fraction))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    partitions = [set(users[:n_train]), set(users[n_train:n_train+n_val]), set(users[n_train+n_val:])]
    return tuple(df[df.user_id.astype(str).isin(part)].copy() for part in partitions)


def score_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    result = {'samples': int(len(labels)), 'positive_samples': int(labels.sum()),
              'threshold': float(threshold), 'precision': float(precision),
              'recall': float(recall), 'f1': float(f1),
              'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
              'false_positive_rate': fp / max(1, fp + tn),
              'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
              'average_precision': None, 'roc_auc': None}
    if len(set(labels)) == 2:
        result['average_precision'] = float(average_precision_score(labels, scores))
        result['roc_auc'] = float(roc_auc_score(labels, scores))
    return result


def choose_threshold(labels, scores):
    """Select on validation data only, never test data."""
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, scores])
    ranked = [(score_metrics(labels, scores, float(t))['f1'], -abs(float(t) - .5), float(t)) for t in candidates]
    return max(ranked)[2]


def predict_checkpoint(checkpoint, df, encoder):
    from .pyg_model import PyGTemporalEmailNet, DirectedGraphState
    obj = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = obj['config']
    if config['encoder'] != encoder.mode or config['text_dim'] != encoder.dim:
        raise ValueError(f'Checkpoint encoder mismatch: expected {config["encoder"]} dim {config["text_dim"]}')
    is_text = Path(checkpoint).stem.endswith('text_only')
    if is_text:
        model = TextOnlyNet(encoder.dim, config['hidden'])
    elif config.get('architecture') == 'pyg':
        model = PyGTemporalEmailNet(encoder.dim, config['hidden'])
    else:
        model = TemporalEmailNet(encoder.dim, config['hidden'])
    model.load_state_dict(obj['state_dict'])
    model.eval()
    scores = pd.Series(index=df.index, dtype=float)
    with torch.no_grad():
        for _, group in df.groupby('user_id', sort=True):
            state = DirectedGraphState() if config.get('architecture') == 'pyg' else GraphState()
            for index, row in group.sort_values('timestamp', kind='stable').iterrows():
                if is_text:
                    logit = model(torch.tensor(row.embedding, dtype=torch.float32))
                else:
                    logit = model.step(row, row.embedding, state, config.get('neighbors', 8))
                scores.loc[index] = float(torch.sigmoid(logit))
    return scores.to_numpy()


def run_evaluation(args):
    df = load_events(args.csv)
    train, validation, test = user_split(df, seed=args.seed)
    encoder = TextEncoder(args.encoder)
    # Text embeddings for checkpoint inference; TF-IDF below is fitted ONLY on training users.
    validation_emb = attach_embeddings(validation, encoder)
    test_emb = attach_embeddings(test, encoder)
    report = {'source': str(args.csv), 'seed': args.seed,
              'warning': 'Evaluate on independently sourced, labeled data before making deployment claims.',
              'split': {name: {'users': int(part.user_id.nunique()), 'emails': len(part),
                               'positive': int(part.label.sum())}
                        for name, part in [('train', train), ('validation', validation), ('test', test)]},
              'models': {}}
    if len(set(train.label)) < 2:
        raise ValueError('Training users must contain both classes for TF-IDF baseline')
    text = lambda frame: (frame.subject.astype(str) + ' ' + frame.body.astype(str)).tolist()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=1)
    x_train = tfidf.fit_transform(text(train))
    baseline = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=args.seed)
    baseline.fit(x_train, train.label)
    val_scores = baseline.predict_proba(tfidf.transform(text(validation)))[:, 1]
    test_scores = baseline.predict_proba(tfidf.transform(text(test)))[:, 1]
    threshold = choose_threshold(validation.label, val_scores)
    report['models']['tfidf_logistic'] = {'validation': score_metrics(validation.label, val_scores, threshold),
                                           'test': score_metrics(test.label, test_scores, threshold)}
    for path in args.checkpoint:
        name = Path(path).stem
        if name in report['models']:
            raise ValueError(f'Duplicate model name: {name}')
        val_scores = predict_checkpoint(path, validation_emb, encoder)
        test_scores = predict_checkpoint(path, test_emb, encoder)
        threshold = choose_threshold(validation.label, val_scores)
        report['models'][name] = {'validation': score_metrics(validation.label, val_scores, threshold),
                                  'test': score_metrics(test.label, test_scores, threshold),
                                  'checkpoint': str(path),
                                  'note': 'Checkpoint is pre-trained; confirm its training users/data do not overlap this evaluation dataset.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Results saved: {output}')


def main():
    parser = argparse.ArgumentParser(description='Phase 3 real-data normalization and held-out-user evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Map a labeled source CSV into the required canonical format')
    prepare.add_argument('--input', required=True)
    prepare.add_argument('--output', required=True)
    for field in COLUMNS:
        prepare.add_argument('--' + field.replace('_', '-'), dest=field)
    prepare.add_argument('--positive', default='1')
    prepare.add_argument('--negative', default='0')
    evaluate = sub.add_parser('evaluate', help='Disjoint-user test with validation-only threshold selection')
    evaluate.add_argument('--csv', required=True)
    evaluate.add_argument('--checkpoint', action='append', default=[])
    evaluate.add_argument('--encoder', choices=['minilm', 'hash'], default='minilm')
    evaluate.add_argument('--seed', type=int, default=42)
    evaluate.add_argument('--output', default='artifacts/phase3_evaluation.json')
    args = parser.parse_args()
    if args.command == 'prepare':
        mapping = {field: getattr(args, field) for field in COLUMNS if getattr(args, field)}
        count = normalize_csv(args.input, args.output, mapping, args.positive, args.negative)
        print(f'Normalized {count} labeled emails into {args.output}')
    else:
        run_evaluation(args)


if __name__ == '__main__':
    main()
