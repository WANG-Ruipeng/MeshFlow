from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pytest,torch
from experiments.a_radial.prepare_data import build
from experiments.a_radial.exporter import ASSETS,MeshExporter
from experiments.a_radial.io_utils import file_hash,read_mesh
from experiments.a_radial.mesh_ops import topology,ray_hits
from experiments.a_radial.legacy.exact_accel import audit

@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    torch.set_num_threads(1)
    return build(ASSETS/'parents.json',tmp_path_factory.mktemp('real16')/'data')

def test_all16_exact512(prepared):
    d=json.loads((prepared/'DATASET.json').read_text());assert len(d['rows'])==16
    for r in d['rows']:
        assert r['faces']==512 and r['face_bin']==25 and r['old_vertices_bitwise_preserved']
        assert r['max_raw_refined_radial_diff_bbox']<1e-10
        v,f=read_mesh(prepared/'meshflow512/objaverse_occ_v5_ids'/f'{r["uid"]}.npz')
        assert len(f)==512 and topology(v,f)['closed_oriented_g0']

def test_all16_refined_exact_intersections(prepared):
    d=json.loads((prepared/'DATASET.json').read_text())
    for r in d['rows']:
        v,f=read_mesh(prepared/'meshflow512/objaverse_occ_v5_ids'/f'{r["uid"]}.npz');a=audit(v,f)
        assert a['embedded_endpoint_pass'],(r['uid'],a)

def test_scale_inverse(prepared):
    p=json.loads((prepared/'DATASET.json').read_text())['rows'][0]
    with np.load(prepared/'radial_train.npz',allow_pickle=False) as z:y=z['logr'][0]
    with np.load(prepared/'lineage'/f'{p["uid"]}.npz',allow_pickle=False) as z:c=z['centre'];s=z['model_to_input_scale']
    ex=MeshExporter();v,f=ex.support.mesh(y)
    assert np.isfinite(v*s+c).all()
    assert abs(np.sqrt(np.mean(np.exp(y.astype(float))**2))-1)<1e-6

def test_three_exports_same_nodes(prepared):
    ex=MeshExporter(10)
    with np.load(prepared/'radial_train.npz',allow_pickle=False) as z:y=z['logr'][0].copy()
    before=y.copy();proxy=ex.proxy(y);outs={}
    for arm in ['fixed','a','qem']:
        m,meta=ex.export(y,512,arm,proxy if arm=='a' else None);outs[arm]=m
        assert meta['actual_faces']==len(m['faces'])
        assert np.array_equal(before,y)
        a=audit(m['vertices'],m['faces']);assert a['embedded_endpoint_pass']
    assert np.array_equal(ex.support.query(y,outs['a']['u']),outs['a']['r'])
    base,_=ex.template(512);assert np.max(abs(ex.support.query(y,base)-outs['a']['r']))>1e-6
    # Warm all dependencies before read trap. No target, cache or source-file read is allowed in A export.
    with patch('builtins.open',side_effect=AssertionError('forbidden IO')),patch('numpy.load',side_effect=AssertionError('forbidden np.load')):
        m,_=ex.export(y,512,'a',proxy)
    assert np.array_equal(m['vertices'],outs['a']['vertices'])
