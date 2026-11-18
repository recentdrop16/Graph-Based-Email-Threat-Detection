import hashlib
import numpy as np
import pandas as pd

COLUMNS = ['user_id', 'timestamp', 'sender', 'recipient', 'subject', 'body', 'label']

def load_events(path):
    df = pd.read_csv(path)
    for col in ('subject', 'body'):
        if col in df.columns:
            df[col] = df[col].astype('string').fillna('')
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='raise')
    df['label'] = df['label'].astype(int)
    if not df['label'].isin([0,1]).all():
        raise ValueError('Labels must be 0 or 1')
    return df.sort_values(['user_id','timestamp'],kind='stable').reset_index(drop=True)

def split_users(df, test_fraction=0.25, seed=42):
    users = sorted(df.user_id.astype(str).unique())
    if len(users)<2:
        raise ValueError('At least two users required for held-out-user evaluation')
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n = max(1, min(len(users)-1, round(len(users)*test_fraction)))
    test = set(users[:n])
    return df[~df.user_id.astype(str).isin(test)].copy(), df[df.user_id.astype(str).isin(test)].copy()

def make_demo(path, users=12, messages=45, seed=42):
    rng=np.random.default_rng(seed)
    rows=[]
    normal_subjects=['Project update','Meeting notes','Weekly report','Invoice question','Schedule change']
    normal_bodies=['Attached are the meeting notes.','Can we meet tomorrow?','Please review the project update.','Thanks for the information.']
    attack_subjects=['Urgent account verification','Password reset required','Confidential data request']
    attack_bodies=['Send your password and verification code immediately.','Urgent: upload employee records to this external site.','Transfer funds to the new account today.']
    for u in range(users):
        owner=f'person{u}@example.org'
        contacts=[f'contact{u}_{j}@example.org' for j in range(7)]
        for i in range(messages):
            attack=int(rng.random()<0.20)
            incoming=bool(rng.random()<0.7)
            other=(f'unknown{u}_{i}@external.test' if attack and rng.random()<0.7 else str(rng.choice(contacts)))
            sender,recipient=(other,owner) if incoming else (owner,other)
            subject=str(rng.choice(attack_subjects if attack else normal_subjects))
            body=str(rng.choice(attack_bodies if attack else normal_bodies))
            ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(hours=i*4+int(rng.integers(0,3)))
            rows.append([f'user{u}',ts.isoformat(),sender,recipient,subject,body,attack])
    df=pd.DataFrame(rows,columns=COLUMNS)
    df.to_csv(path,index=False)
    return df

class TextEncoder:
    def __init__(self, mode='minilm', dim=384):
        self.mode=mode
        self.dim=dim
        if mode=='minilm':
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
            self.dim=self.model.get_sentence_embedding_dimension()
        elif mode!='hash':
            raise ValueError('encoder must be minilm or hash')

    def encode(self,texts):
        if self.mode=='minilm':
            return np.asarray(self.model.encode(texts,normalize_embeddings=True,show_progress_bar=False),dtype=np.float32)
        # Deterministic offline placeholder for smoke tests; not a semantic encoder.
        out=np.zeros((len(texts),self.dim),dtype=np.float32)
        for i,text in enumerate(texts):
            for word in str(text).lower().split():
                digest=hashlib.sha256(word.encode()).digest()
                idx=int.from_bytes(digest[:4],'big')%self.dim
                out[i,idx]+=1 if digest[4]%2 else -1
            norm=np.linalg.norm(out[i]); out[i]/=max(norm,1e-8)
        return out

def attach_embeddings(df,encoder):
    df=df.copy()
    texts=(df.subject.astype(str)+' [SEP] '+df.body.astype(str)).tolist()
    df['embedding']=list(encoder.encode(texts))
    return df
