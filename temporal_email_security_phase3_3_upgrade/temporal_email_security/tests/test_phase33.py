import pytest
from emailguard.phase33 import summarize


def sample(seed, f1):
    models = {}
    for name in ('tfidf_logistic', 'text_only', 'temporal_gnn', 'gnn_no_history'):
        models[name] = {'validation': {'threshold': 0.4}, 'test': {
            'precision': 0.6, 'recall': 0.7, 'f1': f1, 'false_positive_rate': 0.2}}
    return {'seed': seed, 'models': models}


def test_summary_aggregates_seeds():
    result = summarize([sample(42, 0.4), sample(43, 0.6)])
    assert result['runs'] == 2
    assert result['models']['temporal_gnn']['f1']['mean'] == pytest.approx(0.5)
    assert result['models']['temporal_gnn']['f1']['std'] == pytest.approx(0.141421356)
    assert result['models']['gnn_no_history']['validation_selected_thresholds'] == [0.4, 0.4]


def test_summary_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])
