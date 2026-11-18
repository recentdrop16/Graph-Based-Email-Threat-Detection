# Phase 3: Labeled dataset evaluation and baseline comparison

This release **adds** a separate Phase 3 evaluation CLI and tests. Existing Phase 2 training, FastAPI, models, and checkpoints remain unchanged. It does **not** include a real-world email corpus or claim real-world performance.

## Install and verify (PowerShell, from project folder)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

## Quick demonstration (synthetic data only)

```powershell
python -m emailguard.phase3 evaluate --csv sample_data/demo.csv --checkpoint artifacts/pyg_temporal.pt --checkpoint artifacts/pyg_text_only.pt --output artifacts/phase3_demo_evaluation.json
```

This tests an already-trained checkpoint against the included synthetic data. **Because those checkpoints were trained on this same synthetic corpus (different users), this is NOT an independent real-world validation**. The TF-IDF logistic baseline is trained on Phase 3's training-user split, and threshold selection uses validation users only.

## Prepare an external labeled dataset

Input must have genuine email timestamps, sender, recipient, an explicit binary ground-truth label, and a stable user/account ID for graph isolation. Subject and body are optional but recommended. Some public phishing datasets have only message text and labels; they are **not sufficient** to evaluate directed temporal email relationships unless authentic metadata is available. Do not invent sender, recipient, timestamp, or user IDs.

If a CSV already has columns `user_id,timestamp,sender,recipient,subject,body,label` with numeric 0/1 labels, use it directly.

Otherwise, explicitly map its columns (example):

```powershell
python -m emailguard.phase3 prepare --input data\raw_emails.csv --output data\normalized.csv --user-id account --timestamp sent_at --sender from_address --recipient to_address --subject subject --body message --label category --positive phishing --negative legitimate
```

Then evaluate:

```powershell
python -m emailguard.phase3 evaluate --csv data\normalized.csv --output artifacts\external_evaluation.json
```

To compare previously trained checkpoints, append `--checkpoint artifacts/pyg_temporal.pt` and `--checkpoint artifacts/pyg_text_only.pt`, **only if** they are compatible with the data and their training data is disjoint from the external evaluation corpus. For a proper real-data model comparison, retrain each model using the same training users and assess on the same held-out test users; the current CLI does not yet retrain the PyG checkpoint automatically.

## Metrics and methodological safeguards

- Three disjoint user groups: train, validation, test (default deterministic 60/20/20 allocation).
- TF-IDF vectorizer and logistic regression fit on training users only.
- Decision threshold selected on validation users, never test users.
- Precision, recall, F1, false-positive rate, balanced accuracy, confusion matrix, average precision, ROC AUC.
- ROC AUC and average precision are null when a split contains only one class.
- Pretrained PyG checkpoint inference replays emails chronologically per user and resets graph state between users.
- No email dataset downloads or invented ground-truth labels.

**Research limitation:** If the same organization, communication thread, or near-duplicate messages span splits, user isolation alone may not eliminate all leakage. Confirm dataset provenance, class balance, and temporal coverage before publishing results. Checkpoint thresholds and F1 scores should not be interpreted as calibrated security probabilities.
