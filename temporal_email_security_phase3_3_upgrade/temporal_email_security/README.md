# Temporal Email Security — runnable research prototype

A **working local prototype** based on the proposal *Real-Time Temporal Graph Learning for Social-Engineering Detection*. It includes:

- Directed, timestamped sender→recipient email events, partitioned by `user_id`.
- MiniLM sentence embeddings (`all-MiniLM-L6-v2`) for subject and body.
- Shared PyTorch event classifier with per-node GRU temporal memory, learned multi-head attention over sampled recent neighbors, and post-prediction memory updates.
- Held-out-user evaluation, precision/recall/F1/false-positive rate, and a text-only neural baseline.
- FastAPI endpoint that scores messages in timestamp order and retains each user's graph state in memory.
- Synthetic labeled emails for **demonstration only**; synthetic results are **not** evidence of real-world attack detection.

## Setup (Windows PowerShell)

Install Python 3.10–3.12, then from this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m emailguard.train --epochs 4
```

The first MiniLM run downloads a model (~90 MB) from Hugging Face, so internet is required. PyTorch installation may take time. For a quick **offline smoke test** that does **not** use semantic embeddings:

```powershell
python -m emailguard.train --encoder hash --epochs 2
```

## Run API

```powershell
uvicorn emailguard.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs and use `POST /predict` with:

```json
{
  "user_id": "local-user-1",
  "sender": "outside@example.com",
  "recipient": "me@example.org",
  "subject": "Urgent account verification",
  "body": "Please send your password immediately.",
  "timestamp": "2026-11-06T12:00:00Z"
}
```

`POST /reset/{user_id}` clears that user's in-memory history. State is **not persisted** across server restarts. Run one worker only: multi-worker deployments do not share memory. The API is localhost-only and has no authentication; do not expose it to the internet or submit private email without authorization.

## Real dataset CSV format

Columns: `user_id,timestamp,sender,recipient,subject,body,label`.

- `user_id`: graph partition; do not use the same person in both training and held-out testing.
- `timestamp`: ISO 8601 timestamp, ideally with timezone.
- `label`: 0 = normal, 1 = risky. Labels must be independently validated.
- One row per directed sender→recipient edge. For multiple recipients, add rows with the same message data.
- Missing subject/body is allowed but reduces semantic information.

Train using:

```powershell
python -m emailguard.train --csv sample_data/your_emails.csv --epochs 8 --neighbors 8
```

Outputs in `artifacts/`: `temporal.pt`, `text_only.pt`, and `results.json`.

## What is implemented vs. not yet implemented

**Implemented:** actual trainable temporal node memory, learned multi-head neighbor attention, MiniLM embeddings, shared model weights with isolated per-user state, event-level risk predictions, recent-neighbor sampling, held-out-user testing, a text-only baseline, and a local HTTP service.

**Not implemented:** Outlook Office add-in or Microsoft Graph authentication; real-world CERT preprocessing; explicit GCN/static-graph baselines; complete neighbor-limit sweep and runtime/memory benchmark; production model calibration; persistent or encrypted user memory; concurrency safety; attack-resistant security feature extraction; evaluation on verified real phishing labels.

This is a starting implementation for research, **not** a validated phishing detector or a completed experimental study. In particular, this code uses PyTorch's attention layer instead of a PyTorch Geometric GATConv and a simplified GRU memory update instead of the full configurable TGN reference implementation. It should be described as a **TGN-style temporal attention model**, not as a faithful reproduction of the published architecture.

## Tests

```powershell
pytest -q
```
