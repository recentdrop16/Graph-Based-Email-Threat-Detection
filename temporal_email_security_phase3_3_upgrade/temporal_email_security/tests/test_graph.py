import numpy as np
import pytest

pytest.importorskip('torch_geometric')
from emailguard.graph import build_graph, recipients, security_features


def test_recipients():
    assert recipients('A@EXAMPLE.com; b@example.com, a@example.com') == ['a@example.com', 'b@example.com']


def test_directed_multirecipient_domains():
    rows = [{'user_id':'u1','timestamp':'2026-01-01T00:00:00Z','sender':'owner@school.edu',
             'recipient':'a@outside.com;b@school.edu','subject':'Urgent password',
             'body':'Visit https://example.com','embedding':np.zeros(4)},
            {'user_id':'u2','timestamp':'2026-01-01T01:00:00Z','sender':'other@else.com',
             'recipient':'owner@school.edu','subject':'Hi','body':'','embedding':np.zeros(4)}]
    g = build_graph(rows, user_id='u1', text_dim=4)
    e = g['account','emails','account']
    assert g['account'].num_nodes == 3
    assert g['domain'].num_nodes == 2
    assert e.edge_index.shape == (2,2)
    assert e.edge_attr.shape == (2,9)
    assert e.event_id.tolist() == [0,0]
    assert e.edge_index[0].tolist() == [0,0]
    assert g['account','belongs_to','domain'].edge_index.shape == (2,3)
    assert 'other@else.com' not in g.account_addresses
    assert e.edge_attr[:,4].tolist() == [1.,1.]


def test_user_isolation():
    rows = [{'user_id':'a','timestamp':'2026-01-01T00:00:00Z','sender':'a@x.com','recipient':'b@y.com'},
            {'user_id':'b','timestamp':'2026-01-01T00:00:00Z','sender':'c@z.com','recipient':'d@y.com'}]
    with pytest.raises(ValueError): build_graph(rows)
    assert build_graph(rows, user_id='a').account_addresses == ['a@x.com','b@y.com']
