# Phase 3.3 — Longer training and repeatability

Phase 3.3 adds a separate experiment runner without changing the Phase 3.2 model, API, or previous outputs.

## Run

```powershell
python -m pytest -q
python -m emailguard.phase33 --generate-challenge --encoder minilm --epochs 10 --seeds 42 43 44 --output artifacts/phase33_summary.json
```

This trains all neural configurations **from scratch** for ten epochs for each seed. It may take a while on CPU. The challenge dataset is generated separately for each seed, and each seed uses a separate template-based split. Each model's classification threshold is selected using **validation data only**, then applied to its test set. It does not use test results to tune thresholds. No early stopping has been implemented; all ten epochs run.

For a faster smoke test:

```powershell
python -m emailguard.phase33 --generate-challenge --encoder hash --epochs 2 --seeds 42 --output artifacts/phase33_smoke.json
```

Results are written to the requested summary JSON and individual `*_seed42.json`, etc. The summary includes per-seed test metrics, means, standard deviations, and validation-selected thresholds. **Do not treat differences as evidence of generalization to real email**: the dataset remains synthetic, users can overlap between splits, and semantic patterns may recur.

The `gnn_no_history` condition reuses the same trained graph model but resets graph state before each message. It is a diagnostic, not a separately trained baseline.
