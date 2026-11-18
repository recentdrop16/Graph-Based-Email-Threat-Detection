# Phase 1: PyTorch Geometric graph construction

This update **adds** `emailguard/graph.py` and tests, without replacing the trained temporal model. The existing `artifacts/temporal.pt` checkpoint, `emailguard/model.py`, `emailguard/api.py`, and training pipeline remain unchanged. The graph builder is not yet connected to the classifier, so it does not yet satisfy the final PyG model requirement.

## Installation (PowerShell, from project folder)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

If PyG fails to install on your Windows/Python/Torch combination, preserve the existing working environment and report the exact error before changing PyTorch.

## Graph example

```python
from emailguard.graph import build_graph
rows = [{'user_id':'u1', 'timestamp':'2026-01-01T12:00:00Z',
         'sender':'person@school.edu', 'recipient':'friend@work.com;team@school.edu',
         'subject':'Project update', 'body':'See you tomorrow'}]
graph = build_graph(rows, user_id='u1')
print(graph)
print(graph['account','emails','account'].edge_index)
```

## Added behavior

- Directed sender-to-recipient account edges, one per recipient, with shared event ID
- Account-to-domain membership edges and reverse relation for heterogeneous message passing
- Edge timestamps, optional MiniLM embedding and five explicit security indicators
- Per-user graph construction and chronological event ordering

## Still to implement and validate

- PyG GAT/TGN-inspired classifier consuming this heterogeneous graph (with checkpoint migration/retraining)
- Direction-aware temporal memory and contact/domain novelty features
- CERT and supplemental datasets, leakage-safe chronological evaluation
- Static features, GCN/GAT, temporal-only, semantic-temporal baselines
- Neighbor-limit/latency/memory ablations, Outlook prototype

Note: URL/keyword flags are observable features, not proof of maliciousness.
