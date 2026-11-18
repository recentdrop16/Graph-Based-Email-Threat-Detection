# Phase 2: PyG message passing

The legacy checkpoint (`artifacts/temporal.pt`) and default training/API behavior are retained.

## What changed
- `emailguard/pyg_model.py` adds a separate `PyGTemporalEmailNet` using **PyG `GATConv`** over recent, directed historical sender→recipient relationships, with per-account GRU memory.
- The email is classified **before** its edges and memories are updated. Multiple recipients are handled in one event; email-level risk uses the maximum recipient logit.
- The new model uses the MiniLM embedding, three historical timing/frequency features, and five observable security features.
- The training CLI has `--architecture pyg`, saving **separate** `artifacts/pyg_temporal.pt`, `artifacts/pyg_text_only.pt`, and `artifacts/pyg_results.json`.
- The API selects the model based on the checkpoint's `config.architecture`.

## PowerShell from any working directory

```powershell
cd "C:\Users\Anthony\Downloads\temporal_email_security_phase2_upgrade\temporal_email_security"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m emailguard.train --architecture pyg --epochs 4
$env:EMAILGUARD_MODEL = "artifacts/pyg_temporal.pt"
python -m uvicorn emailguard.api:app --host 127.0.0.1 --port 8000
```

If the extracted directory name differs, adjust the `cd` path. If an existing PowerShell terminal has a running server, stop it with Ctrl+C before launching a new one. For the original model, remove the override with `Remove-Item Env:EMAILGUARD_MODEL` before launching the server.

## Limitations
This is a **TGN-inspired, causal, directed GAT/GRU prototype**, not the full Temporal Graph Networks algorithm. The separate Phase 1 `build_graph` heterogeneous graph is still a data representation; this Phase 2 model currently runs `GATConv` on a small account-only subgraph of prior edges, not on the full heterogeneous graph including domain nodes. The new security features are heuristic observations, not verified attack indicators. Results on the synthetic demonstration dataset cannot establish real-world detection performance. The 0.5 threshold is not calibrated.

This package was syntax-checked, but its PyG runtime tests must be run on the user's Windows machine because PyG was unavailable in the packaging environment.
