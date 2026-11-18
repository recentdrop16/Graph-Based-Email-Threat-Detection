import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from .data import load_events,split_users,TextEncoder,attach_embeddings,make_demo
from .model import TemporalEmailNet,TextOnlyNet,GraphState
from .pyg_model import PyGTemporalEmailNet,DirectedGraphState

def metrics(y,p):
    if not y:return {}
    pred=(np.asarray(p)>=0.5).astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'precision':float(precision),'recall':float(recall),'f1':float(f1),'false_positive_rate':float(fp/max(1,fp+tn)),'samples':len(y),'positive_samples':int(sum(y))}

def detach_state(state):
    state.memory={k:v.detach() for k,v in state.memory.items()}

def train_temporal(model,df,epochs=4,lr=0.001,neighbor_limit=8,chunk_size=12):
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn=nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train(); losses=[]
        # Each user graph and memory starts fresh each epoch; order is chronological.
        for user,group in df.groupby('user_id',sort=True):
            state=(DirectedGraphState() if isinstance(model,PyGTemporalEmailNet) else GraphState()); chunk=[]
            for _,row in group.sort_values('timestamp').iterrows():
                logit=model.step(row,row.embedding,state,neighbor_limit)
                label=torch.tensor(float(row.label),device=logit.device)
                chunk.append(loss_fn(logit,label))
                if len(chunk)>=chunk_size:
                    optimizer.zero_grad();loss=torch.stack(chunk).mean();loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step();losses.append(float(loss.item()))
                    detach_state(state);chunk=[]
            if chunk:
                optimizer.zero_grad();loss=torch.stack(chunk).mean();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step();losses.append(float(loss.item()))
        print(f'epoch {epoch+1}/{epochs}: temporal loss={np.mean(losses):.4f}')

def train_text(model,df,epochs=4,lr=0.001):
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    x=torch.tensor(np.stack(df.embedding.values),dtype=torch.float32)
    y=torch.tensor(df.label.values,dtype=torch.float32)
    for epoch in range(epochs):
        model.train();optimizer.zero_grad()
        loss=nn.BCEWithLogitsLoss()(model(x),y)
        loss.backward();optimizer.step()
        print(f'epoch {epoch+1}/{epochs}: text loss={loss.item():.4f}')

@torch.no_grad()
def evaluate(model,df,kind='temporal',neighbor_limit=8):
    model.eval(); y=[];p=[]
    for user,group in df.groupby('user_id',sort=True):
        state=(DirectedGraphState() if isinstance(model,PyGTemporalEmailNet) else GraphState())
        for _,row in group.sort_values('timestamp').iterrows():
            if kind=='temporal':
                logit=model.step(row,row.embedding,state,neighbor_limit)
            else:
                logit=model(torch.tensor(row.embedding,dtype=torch.float32))
            y.append(int(row.label));p.append(float(torch.sigmoid(logit)))
    return metrics(y,p)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',default='sample_data/demo.csv')
    parser.add_argument('--output',default='artifacts')
    parser.add_argument('--encoder',choices=['minilm','hash'],default='minilm')
    parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--neighbors',type=int,default=8)
    parser.add_argument('--hidden',type=int,default=48)
    parser.add_argument('--chunk-size',type=int,default=12)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--architecture',choices=['legacy','pyg'],default='legacy')
    args=parser.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    path=Path(args.csv)
    if not path.exists():
        if path.name!='demo.csv':raise FileNotFoundError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        make_demo(path)
        print(f'Generated synthetic demonstration dataset: {path}')
    df=load_events(path)
    train,test=split_users(df,seed=args.seed)
    print(f'Train users={train.user_id.nunique()}, held-out test users={test.user_id.nunique()}')
    encoder=TextEncoder(args.encoder)
    df=attach_embeddings(df,encoder)
    train=df[df.user_id.isin(train.user_id.unique())]
    test=df[df.user_id.isin(test.user_id.unique())]
    temporal=(PyGTemporalEmailNet(encoder.dim,args.hidden) if args.architecture=='pyg' else TemporalEmailNet(encoder.dim,args.hidden))
    baseline=TextOnlyNet(encoder.dim,args.hidden)
    train_temporal(temporal,train,args.epochs,neighbor_limit=args.neighbors,chunk_size=args.chunk_size)
    train_text(baseline,train,args.epochs)
    results={'temporal':evaluate(temporal,test,'temporal',args.neighbors),'text_only':evaluate(baseline,test,'text')}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    temporal_name='pyg_temporal.pt' if args.architecture=='pyg' else 'temporal.pt'
    results_name='pyg_results.json' if args.architecture=='pyg' else 'results.json'
    config={'encoder':args.encoder,'text_dim':encoder.dim,'hidden':args.hidden,'neighbors':args.neighbors,'seed':args.seed,'architecture':args.architecture}
    torch.save({'state_dict':temporal.state_dict(),'config':config},out/temporal_name)
    torch.save({'state_dict':baseline.state_dict(),'config':config},out/('pyg_text_only.pt' if args.architecture=='pyg' else 'text_only.pt'))
    (out/results_name).write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
    print(f'Saved checkpoints and results in {out}')
if __name__=='__main__':main()
