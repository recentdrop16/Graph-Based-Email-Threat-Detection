"""Controlled, from-scratch comparison on disjoint email-text templates.

Synthetic challenge is explicitly labeled as synthetic. No external dataset is invented.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .data import COLUMNS, load_events, TextEncoder, attach_embeddings
from .audit import _digest
from .phase3 import score_metrics, choose_threshold
from .model import TextOnlyNet


def make_challenge(path, seed=42, users=12, messages=36):
    """Generated examples with overlapping benign/malicious vocabulary and varying contacts.

    Labels are assigned by synthetic scenario, not claimed to be real phishing ground truth.
    """
    rng = np.random.default_rng(seed)
    requests = ['review the document', 'confirm the invoice', 'check the login notice',
                'update the account details', 'verify the meeting time', 'open the shared file',
                'check the payment details', 'review the payroll update', 'confirm the schedule',
                'respond to the support ticket', 'review the access request', 'verify the delivery']
    openings = ['Please', 'When you have time,', 'Today, please', 'Could you', 'For the team,',
                'Before the deadline,', 'Just a reminder to', 'Following up to', 'I need you to',
                'For the weekly review,', 'As discussed,', 'At your convenience,']
    endings = ['and reply to this message.', 'using the usual process.', 'before the end of the day.',
               'and let me know.', 'when you are available.', 'using the shared portal.',
               'and send a confirmation.', 'for our records.', 'after the next meeting.',
               'without sharing credentials.', 'with the finance team.', 'through the help desk.']
    subjects = ['Account notice', 'Project request', 'Team follow-up', 'Invoice update',
                'Schedule reminder', 'Security review', 'Shared document', 'Payment question',
                'Access request', 'Meeting follow-up', 'Help desk message', 'Weekly update']
    rows = []
    for u in range(users):
        owner = f'employee{u}@company.test'
        contacts = [f'colleague{u}_{k}@company.test' for k in range(4)]
        for i in range(messages):
            attack = int(rng.random() < .34)
            # Most malicious requests come from new outsiders, but not all;
            # legitimate messages can also come from outsiders.
            outsider = rng.random() < (.76 if attack else .24)
            other = f'external{u}_{i}@mail.test' if outsider else str(rng.choice(contacts))
            incoming = rng.random() < .83
            sender, recipient = (other, owner) if incoming else (owner, other)
            # 12*12*12 combinations yield substantially more variety than demo.csv.
            subject = str(rng.choice(subjects))
            body = f'{rng.choice(openings)} {rng.choice(requests)} {rng.choice(endings)}'
            ts = pd.Timestamp('2026-02-01', tz='UTC') + pd.Timedelta(minutes=u*3 + i*110)
            rows.append([f'user{u}', ts.isoformat(), sender, recipient, subject, body, attack])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def template_split(df, seed=42):
    """Partition unique normalized subject/body hashes, then assign all matching rows together.

    User IDs may appear in multiple partitions. Temporal state resets for each partition.
    This tests unseen content, NOT generalization to unseen users or future-only events.
    """
    df = df.copy()
    df['_template'] = df.apply(_digest, axis=1)
    unique = np.array(sorted(df['_template'].unique()))
    if len(unique) < 6:
        raise ValueError('At least six distinct message templates are required')
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(len(unique)*.6)
    n_val = int(len(unique)*.2)
    groups = [set(unique[:n_train]), set(unique[n_train:n_train+n_val]), set(unique[n_train+n_val:])]
    parts = tuple(df[df['_template'].isin(group)].drop(columns=['_template']).copy() for group in groups)
    if any(len(set(part.label)) < 2 for part in parts):
        raise ValueError('Each partition must contain both labels; try a different --seed')
    return parts


def predict(model, df, use_history):
    from .pyg_model import DirectedGraphState
    model.eval()
    results = pd.Series(index=df.index, dtype=float)
    with torch.no_grad():
        for _, group in df.groupby('user_id', sort=True):
            state = DirectedGraphState()
            for idx, row in group.sort_values('timestamp', kind='stable').iterrows():
                vec = torch.tensor(row.embedding, dtype=torch.float32)
                if use_history is None:
                    logit = model(vec)
                else:
                    # No-history condition: reset state before every email.
                    if not use_history:
                        state = DirectedGraphState()
                    logit = model.step(row, row.embedding, state)
                results.loc[idx] = float(torch.sigmoid(logit))
    return results.to_numpy()


def run(args):
    from .train import train_temporal, train_text
    from .pyg_model import PyGTemporalEmailNet
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.generate_challenge:
        make_challenge(args.csv, args.seed, args.users, args.messages)
        print(f'Generated synthetic challenge dataset: {args.csv}')
    df = load_events(args.csv)
    train, val, test = template_split(df, args.seed)
    keys = [set(p.apply(_digest, axis=1)) for p in (train,val,test)]
    assert not (keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2])
    text = lambda part: (part.subject.astype(str)+' '+part.body.astype(str)).tolist()
    report = {'source':str(args.csv), 'synthetic':bool(args.generate_challenge),
              'seed':args.seed,'encoder':args.encoder,'epochs':args.epochs,
              'split_method':'disjoint normalized subject/body templates; users may overlap; states reset per partition',
              'cross_split_exact_template_overlap':0,
              'limitations':['Not a real-world dataset when generated with --generate-challenge',
                             'Text templates are disjoint but users may occur in multiple partitions',
                             'No guarantee of independent semantic meaning or label quality',
                             'No future-only temporal split: do not claim prospective performance'],
              'split':{k:{'emails':len(p),'users':int(p.user_id.nunique()),'positive':int(p.label.sum()),
                          'unique_templates':len(s)} for k,p,s in zip(('train','validation','test'),(train,val,test),keys)},
              'models':{}}
    tfidf = TfidfVectorizer(ngram_range=(1,2),max_features=20000)
    xtrain = tfidf.fit_transform(text(train))
    baseline = LogisticRegression(max_iter=1000,class_weight='balanced',random_state=args.seed)
    baseline.fit(xtrain,train.label)
    for name, model, mode in [('tfidf_logistic',baseline,'tfidf')]:
        v = model.predict_proba(tfidf.transform(text(val)))[:,1]
        t = model.predict_proba(tfidf.transform(text(test)))[:,1]
        threshold = choose_threshold(val.label,v)
        report['models'][name] = {'validation':score_metrics(val.label,v,threshold),'test':score_metrics(test.label,t,threshold)}
    encoder = TextEncoder(args.encoder)
    train, val, test = (attach_embeddings(p,encoder) for p in (train,val,test))
    text_model = TextOnlyNet(encoder.dim,args.hidden)
    train_text(text_model,train,args.epochs)
    graph_model = PyGTemporalEmailNet(encoder.dim,args.hidden)
    train_temporal(graph_model,train,args.epochs,chunk_size=args.chunk_size)
    for name,model,history in [('text_only',text_model,None),('temporal_gnn',graph_model,True),
                                ('gnn_no_history',graph_model,False)]:
        v = predict(model,val,history)
        t = predict(model,test,history)
        threshold = choose_threshold(val.label,v)
        report['models'][name] = {'validation':score_metrics(val.label,v,threshold),
                                  'test':score_metrics(test.label,t,threshold)}
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nTEST RESULTS (same held-out templates for all models)')
    for name,item in report['models'].items():
        m=item['test']
        print(f'{name:18s} precision={m["precision"]:.3f} recall={m["recall"]:.3f} F1={m["f1"]:.3f} FPR={m["false_positive_rate"]:.3f}')
    print(f'Saved: {path}')
    return report


def main():
    p=argparse.ArgumentParser(description='Controlled from-scratch, duplicate-free template comparison')
    p.add_argument('--csv',default='sample_data/challenge.csv')
    p.add_argument('--generate-challenge',action='store_true')
    p.add_argument('--users',type=int,default=12)
    p.add_argument('--messages',type=int,default=36)
    p.add_argument('--encoder',choices=['minilm','hash'],default='minilm')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--hidden',type=int,default=48)
    p.add_argument('--chunk-size',type=int,default=12)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',default='artifacts/phase32_results.json')
    args=p.parse_args()
    run(args)

if __name__=='__main__':
    main()
