import pandas as pd
from emailguard.audit import audit_dataset
from emailguard.data import COLUMNS


def test_cross_split_duplicate_detection(tmp_path):
    rows = []
    for u in range(6):
        rows.append([f'u{u}', '2026-01-01T00:00:00Z', f's{u}@test.org', f'r{u}@test.org', 'Shared', 'Same body', u % 2])
    path = tmp_path / 'data.csv'
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    report = audit_dataset(path)
    assert report['cross_split_content_overlap']['train_vs_test']['shared_message_templates'] == 1
    assert sum(s['rows'] for s in report['splits'].values()) == 6


def test_distinct_content_has_no_overlap(tmp_path):
    rows = []
    for u in range(6):
        rows.append([f'u{u}', '2026-01-01T00:00:00Z', f's{u}@test.org', f'r{u}@test.org', f'Subject {u}', f'Body {u}', u % 2])
    path = tmp_path / 'data.csv'
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    report = audit_dataset(path)
    assert all(v['shared_message_templates'] == 0 for v in report['cross_split_content_overlap'].values())
