"""Directed heterogeneous email graph construction for PyTorch Geometric.

This module is additive: the v0.1 temporal checkpoint and API are unchanged.
Each email is represented by directed sender->recipient account edges and
account->domain membership edges. Domain and account nodes have distinct types.
"""
from collections import defaultdict
from datetime import datetime, timezone
import re
import numpy as np
import torch

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
URGENCY = re.compile(r"\b(urgent|immediately|asap|within 24 hours|suspended)\b", re.IGNORECASE)
CREDENTIAL = re.compile(r"\b(password|verification code|credentials|one.time code|otp)\b", re.IGNORECASE)
SENSITIVE = re.compile(r"\b(confidential|employee records|bank account|wire transfer|ssn)\b", re.IGNORECASE)


def recipients(value):
    """Accept one address, semicolon/comma-separated addresses, or a list."""
    if isinstance(value, (list, tuple, set)):
        values = [part for v in value for part in recipients(v)]
    else:
        values = re.split(r"[;,]", str(value))
    result = []
    for v in values:
        address = v.strip().lower()
        if address and address not in result:
            if not EMAIL.fullmatch(address):
                raise ValueError(f"Invalid recipient address: {address}")
            result.append(address)
    if not result:
        raise ValueError("At least one recipient required")
    return result


def security_features(subject, body, sender, recipient):
    """Explicit observable indicators; not independently verified attack labels."""
    content = f"{subject or ''} {body or ''}"
    links = URL.findall(content)
    sender_domain = sender.rsplit('@', 1)[-1].lower()
    recipient_domain = recipient.rsplit('@', 1)[-1].lower()
    return np.asarray([
        float(bool(links)), float(bool(URGENCY.search(content))),
        float(bool(CREDENTIAL.search(content))), float(bool(SENSITIVE.search(content))),
        float(sender_domain != recipient_domain),
    ], dtype=np.float32)


def _timestamp(value):
    if hasattr(value, 'timestamp'):
        return float(value.timestamp())
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).replace('Z', '+00:00')
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def build_graph(events, user_id=None, text_dim=384):
    """Return a PyG HeteroData graph for one user, sorted chronologically.

    Event rows must contain user_id, timestamp, sender, recipient, subject, body.
    Optional embedding is a fixed-length numeric vector. One email with multiple
    recipients becomes multiple edges with the same event_id and timestamp.
    This is a graph-data pipeline, not yet the temporal classifier's input.
    """
    from torch_geometric.data import HeteroData
    rows = events.to_dict('records') if hasattr(events, 'to_dict') else list(events)
    if user_id is None:
        users = {str(r['user_id']) for r in rows}
        if len(users) != 1: raise ValueError('Specify user_id when events contain multiple users')
        user_id = next(iter(users))
    rows = [r for r in rows if str(r['user_id']) == str(user_id)]
    rows.sort(key=lambda r: _timestamp(r['timestamp']))
    accounts = {}; domains = {}; edges = []; memberships = set()
    edge_attr = []; edge_time = []; event_ids = []
    for event_id, row in enumerate(rows):
        sender = str(row['sender']).strip().lower()
        if not EMAIL.fullmatch(sender): raise ValueError(f'Invalid sender address: {sender}')
        targets = recipients(row['recipient'])
        for address in [sender] + targets:
            if address not in accounts: accounts[address] = len(accounts)
            domain = address.rsplit('@', 1)[1]
            if domain not in domains: domains[domain] = len(domains)
            memberships.add((accounts[address], domains[domain]))
        emb = row.get('embedding')
        if emb is None: emb = np.zeros(text_dim, dtype=np.float32)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if len(emb) != text_dim: raise ValueError(f'Expected embedding dimension {text_dim}, got {len(emb)}')
        for target in targets:
            edges.append((accounts[sender], accounts[target]))
            edge_attr.append(np.concatenate([emb, security_features(row.get('subject',''),row.get('body',''),sender,target)]))
            edge_time.append(_timestamp(row['timestamp']))
            event_ids.append(event_id)
    graph = HeteroData()
    graph['account'].num_nodes = len(accounts)
    graph['domain'].num_nodes = len(domains)
    graph['account', 'emails', 'account'].edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.empty((2,0), dtype=torch.long)
    graph['account', 'emails', 'account'].edge_attr = torch.tensor(np.stack(edge_attr), dtype=torch.float32) if edge_attr else torch.empty((0,text_dim+5), dtype=torch.float32)
    graph['account', 'emails', 'account'].timestamp = torch.tensor(edge_time, dtype=torch.float64)
    graph['account', 'emails', 'account'].event_id = torch.tensor(event_ids, dtype=torch.long)
    pairs = sorted(memberships)
    graph['account', 'belongs_to', 'domain'].edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous() if pairs else torch.empty((2,0), dtype=torch.long)
    graph['domain', 'has_account', 'account'].edge_index = graph['account', 'belongs_to', 'domain'].edge_index.flip(0)
    graph.account_addresses = list(accounts)
    graph.domain_names = list(domains)
    graph.user_id = str(user_id)
    return graph
