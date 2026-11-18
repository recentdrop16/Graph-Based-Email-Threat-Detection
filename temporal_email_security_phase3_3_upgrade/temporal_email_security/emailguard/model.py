"""Research-sized temporal graph model with attention, GRU node memory and event classification.

Each user has an isolated state. States are updated after predictions to avoid
using the current event as historical evidence. The model weights are shared.
"""
from collections import defaultdict,deque
from dataclasses import dataclass,field
import math
import torch
from torch import nn

@dataclass
class GraphState:
    memory:dict=field(default_factory=dict)
    neighbors:dict=field(default_factory=lambda:defaultdict(lambda:deque(maxlen=256)))
    last_seen:dict=field(default_factory=dict)

class TemporalEmailNet(nn.Module):
    def __init__(self,text_dim=384,hidden=48,heads=2):
        super().__init__()
        self.text_dim=text_dim
        self.hidden=hidden
        self.heads=heads
        self.event=nn.Sequential(nn.Linear(text_dim+3,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.attention=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.gru=nn.GRUCell(hidden*2,hidden)
        self.classifier=nn.Sequential(nn.Linear(hidden*5,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def zero(self,device):
        return torch.zeros(self.hidden,device=device)

    def _aggregate(self,node,state,limit,device):
        # Temporal neighbor sampling: most recent unique contacts, bounded by limit.
        seen=set(); contacts=[]
        for other in reversed(state.neighbors[node]):
            if other not in seen:
                seen.add(other); contacts.append(other)
            if len(contacts)>=limit: break
        if not contacts: return self.zero(device)
        keys=torch.stack([state.memory.get(c,self.zero(device)) for c in contacts]).unsqueeze(0)
        query=state.memory.get(node,self.zero(device)).reshape(1,1,-1)
        return self.attention(query,keys,keys,need_weights=False)[0].reshape(-1)

    def step(self,row,embedding,state,neighbor_limit=8,update=True):
        device=next(self.parameters()).device
        sender,recipient=str(row['sender']),str(row['recipient'])
        ts=float(row['timestamp'].timestamp())
        sm=state.memory.get(sender,self.zero(device))
        rm=state.memory.get(recipient,self.zero(device))
        sa=self._aggregate(sender,state,neighbor_limit,device)
        ra=self._aggregate(recipient,state,neighbor_limit,device)
        last_s=state.last_seen.get(sender,ts)
        last_r=state.last_seen.get(recipient,ts)
        dt_s=math.log1p(max(0,ts-last_s)/3600)
        dt_r=math.log1p(max(0,ts-last_r)/3600)
        frequency=sum(1 for x in state.neighbors[sender] if x==recipient)
        extras=torch.tensor([dt_s,dt_r,math.log1p(frequency)],dtype=torch.float32,device=device)
        vec=torch.as_tensor(embedding,dtype=torch.float32,device=device)
        event=self.event(torch.cat([vec,extras]))
        logit=self.classifier(torch.cat([sm,rm,sa,ra,event])).squeeze()
        if update:
            # Graph updates happen after scoring the event.
            state.memory[sender]=self.gru(torch.cat([event,rm]),sm)
            state.memory[recipient]=self.gru(torch.cat([event,sm]),rm)
            state.neighbors[sender].append(recipient)
            state.neighbors[recipient].append(sender)
            state.last_seen[sender]=ts
            state.last_seen[recipient]=ts
        return logit

class TextOnlyNet(nn.Module):
    def __init__(self,text_dim=384,hidden=48):
        super().__init__()
        self.classifier=nn.Sequential(nn.Linear(text_dim,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,embedding):
        return self.classifier(embedding).squeeze(-1)
