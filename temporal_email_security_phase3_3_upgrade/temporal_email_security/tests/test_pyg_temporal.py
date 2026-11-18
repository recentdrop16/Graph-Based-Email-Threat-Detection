from datetime import datetime, timezone, timedelta
import numpy as np
import torch
from emailguard.pyg_model import PyGTemporalEmailNet, DirectedGraphState

def test_pyg_forward_and_directed_history():
    model = PyGTemporalEmailNet(text_dim=8, hidden=16)
    state = DirectedGraphState()
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = dict(sender='alice@example.com', recipient='bob@example.org', timestamp=t,
               subject='urgent', body='Please verify password')
    emb = np.zeros(8, dtype=np.float32)
    first = model.step(row, emb, state, update=False)
    assert first.ndim == 0 and torch.isfinite(first)
    assert not state.incoming and not state.memory
    model.step(row, emb, state)
    assert list(state.incoming['bob@example.org']) == ['alice@example.com']
    assert not state.incoming['alice@example.com']
    row['timestamp'] = t + timedelta(hours=1)
    later = model.step(row, emb, state)
    assert torch.isfinite(later)

def test_multiple_recipients_share_one_email_update():
    model = PyGTemporalEmailNet(text_dim=8, hidden=16)
    state = DirectedGraphState()
    row = dict(sender='a@x.com', recipient='b@y.com;c@z.com',
               timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
    score = model.step(row, np.zeros(8, dtype=np.float32), state)
    assert torch.isfinite(score)
    assert state.pair_count[('a@x.com', 'b@y.com')] == 1
    assert state.pair_count[('a@x.com', 'c@z.com')] == 1
