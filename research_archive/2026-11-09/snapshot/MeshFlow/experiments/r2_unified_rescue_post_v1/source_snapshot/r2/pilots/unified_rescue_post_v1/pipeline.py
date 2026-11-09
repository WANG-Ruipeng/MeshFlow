"""Frozen CPU adapter; reusable on existing or independently authorized D0 caches."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from meshflow_control import geo_io
from meshflow_control.source_winding import export_source_winding
from r2.pilots.seam_slot_c200_v1.postprocess import validate_anchor_indices
from r2.tools.hinge_opposite_v1 import engine as H
from r2.tools.hinge_v2_cpu_v1 import alpha,holes
from . import floor,guard_geo

VERSION='D0_EDGE_FLOOR_1p5R_1p25R_ANTI_SELF_COLLAPSE_3R_1R_v1'

def geometry_from_cached(raw,proposal,source):
    """No model/GT reads. RAW, restored proposal and all maps use canonical slots."""
    original=raw['raw'];K=int(raw['known_faces'])
    pre,protected,records,restored=floor.protect(original,proposal,raw['raw_anchor_ids'],raw['C_vertices'],K,guard_geo.RADIUS)
    geo=guard_geo.process(pre,K,protected_faces=protected)
    canonical=geo_io._validate_result(pre,K,geo)
    oriented=export_source_winding(canonical,source);post=oriented['oriented_arrays']
    extra,anchors=validate_anchor_indices(post,raw);final=dict(post,**extra)
    final.update(identity_domain=np.asarray('generated_RAW_slot_not_GT'),occupied=raw['occupied'],
        occupancy_logits=raw['occupancy_logits'],edge_records=raw['edge_records'],C_vertices=raw['C_vertices'],
        C_faces_original=raw['C_faces_original'],C_faces_model=raw['C_faces_model'])
    assert final['vertices'][final['faces']].tobytes()==final['output'].tobytes()
    assert final['C'].tobytes()==source['C_original'].tobytes()
    if 'torch' in sys.modules:raise RuntimeError('CPU adapter imported torch')
    return pre,canonical,final,geo,oriented,anchors,records

def finish_stage(stage,current,pre,raw,pairs):
    """Exactly one frozen pass per stage, no threshold scan or extra replays."""
    if stage=='HINGE_OPP015':return H.decode(current,pre,raw,pairs,.015,require_opposite_winding=True)
    if stage=='ALPHA':return alpha.decode(current,pre,raw,pairs,.015,alphas=True)
    if stage=='FINAL':
        out,npairs,edges,records,summary=holes.decode(current,pairs)
        return out,records,summary,None,None,npairs
    raise ValueError(stage)
