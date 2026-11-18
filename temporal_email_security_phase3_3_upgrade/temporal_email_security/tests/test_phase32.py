from pathlib import Path
import pandas as pd
from emailguard.phase32 import make_challenge, template_split
from emailguard.data import load_events
from emailguard.audit import _digest


def test_challenge_template_split_no_duplicates(tmp_path):
    path = tmp_path/'challenge.csv'
    make_challenge(path)
    train, val, test = template_split(load_events(path))
    hashes = [set(p.apply(_digest,axis=1)) for p in (train,val,test)]
    assert not hashes[0].intersection(hashes[1])
    assert not hashes[0].intersection(hashes[2])
    assert not hashes[1].intersection(hashes[2])
    assert sum(map(len,(train,val,test))) == 432
    assert all(set(p.label)=={0,1} for p in (train,val,test))


def test_challenge_is_repeatable(tmp_path):
    a,b=tmp_path/'a.csv',tmp_path/'b.csv'
    make_challenge(a);make_challenge(b)
    assert a.read_bytes()==b.read_bytes()
