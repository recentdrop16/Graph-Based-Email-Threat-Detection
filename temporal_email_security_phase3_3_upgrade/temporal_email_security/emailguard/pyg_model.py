"""Phase 2: causal, directed PyG GATConv with per-user GRU memory.

This is a TGN-inspired research prototype, not a full TGN implementation.
Historical edges alone are used for scoring; the current edge is added afterward.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
import torch
from torch import nn
from torch_geometric.nn import GATConv
from .graph import recipients, security_features

@dataclass
class DirectedGraphState:
    memory: dict = field(default_factory=dict)
    incoming: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=256)))
    last_seen: dict = field(default_factory=dict)
    pair_count: dict = field(default_factory=lambda: defaultdict(int))

class PyGTemporalEmailNet(nn.Module):
    def __init__(self, text_dim=384, hidden=48, heads=2):
        super().__init__()
        self.text_dim, self.hidden = text_dim, hidden
        self.event = nn.Sequential(nn.Linear(text_dim+8, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gat = GATConv(hidden, hidden, heads=heads, concat=False, add_self_loops=False)
        self.gru = nn.GRUCell(hidden*2, hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden*5, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def zero(self, device):
        return torch.zeros(self.hidden, device=device)

    def _aggregate(self, node, state, limit, device):
        # Directed historical sender -> recipient edges; recent unique sources.
        sources = []
        for source in reversed(state.incoming.get(node, ())):
            if source not in sources:
                sources.append(source)
            if len(sources) >= limit:
                break
        if not sources:
            return self.zero(device)
        # Local subgraph: source memories send messages to the target node.
        nodes = sources + ([node] if node not in sources else [])
        indices = {name: i for i, name in enumerate(nodes)}
        x = torch.stack([state.memory.get(name, self.zero(device)) for name in nodes])
        edge_index = torch.tensor([[indices[s] for s in sources],
                                   [indices[node]] * len(sources)], dtype=torch.long, device=device)
        return self.gat(x, edge_index)[indices[node]]

    def step(self, row, embedding, state, neighbor_limit=8, update=True):
        device = next(self.parameters()).device
        sender = str(row['sender']).strip().lower()
        targets = recipients(row['recipient'])
        ts_value = row['timestamp']
        ts = float(ts_value.timestamp()) if hasattr(ts_value, 'timestamp') else float(ts_value)
        vec = torch.as_tensor(embedding, dtype=torch.float32, device=device)
        if vec.numel() != self.text_dim:
            raise ValueError(f'Expected {self.text_dim} embedding values')
        sm = state.memory.get(sender, self.zero(device))
        sa = self._aggregate(sender, state, neighbor_limit, device)
        logits, events, old_recipient_memories = [], [], []
        for recipient in targets:
            rm = state.memory.get(recipient, self.zero(device))
            ra = self._aggregate(recipient, state, neighbor_limit, device)
            dt_s = math.log1p(max(0., ts-state.last_seen.get(sender, ts))/3600)
            dt_r = math.log1p(max(0., ts-state.last_seen.get(recipient, ts))/3600)
            freq = math.log1p(state.pair_count.get((sender, recipient), 0))
            security = security_features(row.get('subject',''), row.get('body',''), sender, recipient)
            extras = torch.tensor([dt_s, dt_r, freq, *security], dtype=torch.float32, device=device)
            event = self.event(torch.cat([vec, extras]))
            logits.append(self.classifier(torch.cat([sm, rm, sa, ra, event])).reshape(()))
            events.append(event)
            old_recipient_memories.append(rm)
        if update:
            # Simultaneous update: no recipient sees this email until all predictions finish.
            state.memory[sender] = self.gru(torch.cat([torch.stack(events).mean(0),
                torch.stack(old_recipient_memories).mean(0)]), sm)
            for recipient, event, rm in zip(targets, events, old_recipient_memories):
                if recipient != sender:
                    state.memory[recipient] = self.gru(torch.cat([event, sm]), rm)
                state.incoming[recipient].append(sender)
                state.pair_count[(sender, recipient)] += 1
                state.last_seen[recipient] = ts
            state.last_seen[sender] = ts
        return torch.stack(logits).max()  # email-level risk: highest recipient risk
