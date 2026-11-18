# Phase 3.1 — Dataset audit and external-data preparation

This is a complete copy of the Phase 3 project with two additional dataset-audit tests and a fix for the Pandas fillna warning. **No external real-world dataset is bundled or downloaded.** The included `sample_data/demo.csv` is synthetic.

## Verify

From the project folder, activate your existing virtual environment or create a new one, install `requirements.txt`, then:

```powershell
python -m pytest -q
```

Expected: **11 passed**.

## Audit synthetic demonstration data

```powershell
python -m emailguard.audit --csv sample_data/demo.csv --output artifacts/demo_audit.json
```

Look at `cross_split_content_overlap`. The synthetic corpus reuses short attack/normal templates, so high overlap is expected. Perfect metrics on this corpus are not evidence of generalization.

## Integrate external data

Obtain an independently sourced, appropriately licensed labeled email CSV yourself. If its columns are not already `user_id,timestamp,sender,recipient,subject,body,label`, map them explicitly:

```powershell
python -m emailguard.phase3 prepare --input data/raw.csv --output data/normalized.csv --user-id account --timestamp sent_at --sender from_address --recipient to_address --subject subject --body message --label category --positive phishing --negative legitimate
python -m emailguard.audit --csv data/normalized.csv --output artifacts/external_audit.json
python -m emailguard.phase3 evaluate --csv data/normalized.csv --output artifacts/external_baseline.json
```

**Important:** Authentic sender, recipient, timestamp, and user/account identifiers are necessary for temporal graph evaluation. Do not manufacture those fields for text-only phishing datasets. For a text-only dataset, the evaluation workflow needs a separate text-only adapter; do not run graph evaluation on fabricated relationships.

**Checkpoint caution:** `artifacts/pyg_temporal.pt` and `artifacts/pyg_text_only.pt` were trained on the original synthetic data. Do not report their scores on that synthetic data as independent generalization. A real-data study requires training comparable models on the same real-data training split, with validation-only threshold selection and untouched test data. The existing Phase 3 CLI evaluates pretrained checkpoints; it does not yet retrain them on new datasets.

**Audit limitations:** This audit checks exact normalized subject/body matches across user splits, not semantic near-duplicates, threads, organizational overlap, or data provenance. It cannot establish checkpoint training-data independence.
