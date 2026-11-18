from pathlib import Path
from emailguard.data import make_demo,load_events,split_users,TextEncoder,attach_embeddings
from emailguard.model import TemporalEmailNet,GraphState

def test_demo_and_forward(tmp_path):
    path=tmp_path/'demo.csv'
    make_demo(path,users=4,messages=6)
    df=load_events(path)
    train,test=split_users(df)
    assert set(train.user_id).isdisjoint(set(test.user_id))
    encoder=TextEncoder('hash',dim=32)
    df=attach_embeddings(df,encoder)
    model=TemporalEmailNet(32,hidden=16,heads=2)
    state=GraphState()
    row=df.iloc[0]
    score=model.step(row,row.embedding,state,neighbor_limit=3)
    assert score.ndim==0
    assert row.sender in state.memory
