import os
from datetime import datetime,timezone
from pathlib import Path
import torch
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from .data import TextEncoder
from .model import TemporalEmailNet,GraphState
from .pyg_model import PyGTemporalEmailNet,DirectedGraphState

app=FastAPI(title='Temporal Email Security Research API',version='0.1.0')
_model=None
_encoder=None
_config=None
_states={}
_last_time={}

class EmailRequest(BaseModel):
    user_id:str=Field(min_length=1)
    sender:str=Field(min_length=1)
    recipient:str=Field(min_length=1)
    subject:str=''
    body:str=''
    timestamp:datetime=Field(default_factory=lambda:datetime.now(timezone.utc))

@app.on_event('startup')
def startup():
    global _model,_encoder,_config
    ckpt=Path(os.environ.get('EMAILGUARD_MODEL','artifacts/temporal.pt'))
    if not ckpt.exists():
        raise RuntimeError(f'Missing {ckpt}; train first: python -m emailguard.train')
    obj=torch.load(ckpt,map_location='cpu',weights_only=True)
    _config=obj['config'];_encoder=TextEncoder(_config['encoder'])
    _model=(PyGTemporalEmailNet(_config['text_dim'],_config['hidden']) if _config.get('architecture')=='pyg' else TemporalEmailNet(_config['text_dim'],_config['hidden']))
    _model.load_state_dict(obj['state_dict']);_model.eval()

@app.get('/health')
def health():return {'status':'ok','encoder':_config['encoder'] if _config else None}

@app.post('/predict')
def predict(event:EmailRequest):
    ts=event.timestamp
    if ts.tzinfo is None:ts=ts.replace(tzinfo=timezone.utc)
    timestamp=ts.timestamp()
    previous=_last_time.get(event.user_id)
    if previous is not None and timestamp<previous:
        raise HTTPException(400,'Events for each user must arrive in chronological order; reset user state to replay history.')
    state=_states.setdefault(event.user_id,DirectedGraphState() if _config.get('architecture')=='pyg' else GraphState())
    emb=_encoder.encode([event.subject+' [SEP] '+event.body])[0]
    with torch.no_grad():
        score=float(torch.sigmoid(_model.step({'sender':event.sender,'recipient':event.recipient,'timestamp':ts,'subject':event.subject,'body':event.body},emb,state,_config['neighbors'])))
    _last_time[event.user_id]=timestamp
    return {'risk_score':round(score,6),'threshold':0.5,'predicted_label':int(score>=0.5),'note':'Research output; not a verified security verdict.'}

@app.post('/reset/{user_id}')
def reset(user_id:str):
    _states.pop(user_id,None);_last_time.pop(user_id,None)
    return {'reset':user_id}
