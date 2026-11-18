import numpy as np
import pandas as pd
import pytest
from emailguard.phase3 import normalize_csv, user_split, choose_threshold, score_metrics


def test_user_disjoint():
    frame = pd.DataFrame({'user_id': [f'u{i}' for i in range(12)], 'label': [i % 2 for i in range(12)]})
    train, val, test = user_split(frame)
    sets = [set(x.user_id) for x in (train, val, test)]
    assert all(sets)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(map(len, sets)) == 12


def test_metrics_and_threshold():
    labels = [0, 0, 1, 1]
    scores = [.1, .2, .8, .9]
    threshold = choose_threshold(labels, scores)
    result = score_metrics(labels, scores, threshold)
    assert result['f1'] == 1.0
    assert result['confusion_matrix'] == {'tn': 2, 'fp': 0, 'fn': 0, 'tp': 2}


def test_normalize_explicit_labels(tmp_path):
    source, dest = tmp_path / 'source.csv', tmp_path / 'output.csv'
    pd.DataFrame({'who': ['u1'], 'when': ['2026-01-01T00:00:00Z'], 'from': ['a@b.com'],
                  'to': ['c@d.com'], 'category': ['phish'], 'message': ['verify']}).to_csv(source, index=False)
    mapping = {'user_id': 'who', 'timestamp': 'when', 'sender': 'from', 'recipient': 'to',
               'label': 'category', 'body': 'message'}
    assert normalize_csv(source, dest, mapping, positive='phish', negative='safe') == 1
    assert pd.read_csv(dest).label.tolist() == [1]
    with pytest.raises(ValueError, match='Unknown labels'):
        normalize_csv(source, dest, mapping)
