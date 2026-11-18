# Phase 3.2 — controlled experiments

This release adds `emailguard/phase32.py`. It keeps the Phase 3.1 code intact.

## Run on Windows PowerShell

From the project directory, activate `.venv`, then:

```powershell
python -m pytest -q
python -m emailguard.phase32 --generate-challenge --csv sample_data/challenge.csv --encoder minilm --epochs 3 --output artifacts/phase32_results.json
```

This trains all three model families **from scratch** on the same partition and reports a fourth no-history condition using the trained GNN with its state cleared before each email. It does **not** retrain a separate no-history model. For a faster offline smoke test, use `--encoder hash --epochs 1`; hash embeddings are placeholders and are not suitable for final research conclusions.

The synthetic challenge has 432 messages, 12 users, more varied wording shared by both labels, and sender familiarity patterns that are correlated with labels but not perfectly. The generated dataset is **synthetic** and is not a substitute for independently sourced real messages. It is intentionally less trivial than `demo.csv`, but scores are not guaranteed to be different or to favor any model.

## Splitting and limitations

Every identical normalized subject/body combination is assigned to **one** partition only. This prevents exact text-template overlap, but does not prevent semantic near-duplicates. Users may appear in more than one partition, so **do not describe this as a held-out-user evaluation**. State is reset separately for each partition; the comparison does not simulate a chronological production deployment.

The TF-IDF model is fitted on training messages only. Both neural models are newly initialized and trained on the same training partition. Classification thresholds are selected on validation only, and the test partition is used only for final scores.

## Evaluate a real labeled dataset

If you have a CSV with the canonical columns `user_id,timestamp,sender,recipient,subject,body,label` and **authentic** communication metadata, run:

```powershell
python -m emailguard.phase32 --csv data/real_labeled.csv --encoder minilm --epochs 3 --output artifacts/real_phase32_results.json
```

Use `python -m emailguard.phase3 prepare ...` to normalize actual field names. **Do not fabricate** sender/recipient, timestamps, or account IDs for text-only corpora; those corpora can support text-only experiments but not a credible temporal-graph comparison.

## Reporting

Only quote numbers actually produced by your own command. Include the dataset type, split method, sample counts, and limitations. This phase does not download or claim to include any external real-world dataset.
