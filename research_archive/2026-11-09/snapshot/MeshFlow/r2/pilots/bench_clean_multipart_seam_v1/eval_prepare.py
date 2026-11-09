"""Freeze the original Bench 48 requests and old C-only layouts, no forwards."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from r2.pilots.bench_parts_convergence_v1 import panel as P

ROOT=P.ROOT
OUT=ROOT/'experiments/r2_bench_clean_multipart_seam_v1'
OLD=ROOT/'experiments/r2_bench_parts_convergence_v1'
LABELS=('NEW_HEAD','OLD_S15000_LAYOUT')
HISTORICAL=('NATIVE_S21000','SEAM_S15000','SEAM_S21000')
MODEL='BENCH_CLEAN_MULTIPART_S1500'

def verify(files):
    for path,digest in files.items():
        if P.sha(path)!=digest:raise ValueError('Protected input changed: '+path)

def run():
    dest=OUT/'eval_manifest.json'
    if dest.exists():
        value=P.read(dest);verify(value['protected']);return value
    original=OLD/'evaluation_inputs/manifest.json'
    score=OLD/'evaluation/score_registrations/SEAM_S15000.json'
    m=P.read(original);prior=P.read(score)
    descriptors={q['request_id']:q['query_descriptor'] for q in prior['requests']}
    protected={str(p):P.sha(p) for p in (original,score)}
    rows=[]
    for original_q in m['requests']:
        q=dict(original_q)
        q['query_descriptor']=descriptors[q['request_id']]
        for key in ('inference_npz','noise_npz','metadata_npz','source_npz'):
            digest=P.sha(q[key]);q[key+'_sha256']=digest;protected[q[key]]=digest
        inp=P.resolve(q['inference_npz'])
        with np.load(inp,allow_pickle=False) as z:
            assert int(z['N'])==q['N'] and len(z['C_faces_model'])==q['K']
        d=q['query_descriptor'];assert P.sha(d['path'])==d['sha256']
        protected[d['path']]=d['sha256']
        lp=OLD/'evaluation/SEAM_S15000/layouts'/q['condition_id']/'layout.npz'
        rp=lp.parent/'receipt.json';r=P.read(rp)
        assert r['status']=='COMPLETE' and r['source_labels_read'] is False
        assert P.sha(lp)==r['layout_sha256']
        q['old_layout_npz']=str(lp);q['old_layout_receipt']=str(rp)
        q['old_layout_sha256']=P.sha(lp)
        protected[str(lp)]=P.sha(lp);protected[str(rp)]=P.sha(rp)
        rows.append(q)
    assert len(rows)==48 and len({q['condition_id'] for q in rows})==24
    assert sum(q['closed_C'] for q in rows)==8
    value=dict(status='FROZEN',schema='bench_clean_same48_two_layouts_v1',requests=rows,
        new_labels=list(LABELS),historical_labels=list(HISTORICAL),protected=protected,
        new_noise_arrays=0,new_query_arrays=0,raw_outputs=96,D0_outputs=96,
        generator_forwards=4800,new_head_layout_calls=20,closed_C_head_skips=4,
        old_head_calls_for_layout=0,all_RAW_before_D0=96,
        old_layout_policy='Exact frozen SEAM_S15000 C-only occupied mask; source labels never at inference',
        no_unique_GT_style_success=True)
    P.immutable(dest,value)
    # Legacy D0 wrapper needs this manifest shape, without regenerating inputs.
    P.immutable(OUT/'evaluation_inputs/manifest.json',dict(value,process_request_ids=[]))
    return value

def requests(root=OUT,subset='full'):
    if subset!='full':raise ValueError('Only the frozen full48 panel is authorized')
    value=P.read(P.resolve(root)/'eval_manifest.json');verify(value['protected'])
    return value['requests']

def cpu_check():
    from r2.pilots.bench_parts_convergence_v1.data import inference_layout
    from meshflow_control.source_winding import validate_source
    import ast
    names=('eval_prepare','sample','post_d0','evaluate','comparison','gallery','layout_diagnostic')
    for name in names:ast.parse(Path(__file__).with_name(name+'.py').read_text(encoding='utf8'))
    checked=[]
    for q in requests():
        with np.load(P.resolve(q['inference_npz']),allow_pickle=False) as z:
            core={k:z[k].copy() for k in ('C_vertices','C_faces_original','C_faces_model')}
            core['total_N']=int(z['N'])
        with np.load(P.resolve(q['old_layout_npz']),allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
        lay=inference_layout(**core,occupied=old['occupied'])
        for key in ('anchor_ids','edge_records','seam_slots','bulk_slots','known_mask'):
            assert np.array_equal(lay[key],old[key]),(q['request_id'],key)
        with np.load(P.resolve(q['source_npz']),allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
        validate_source(source,core['C_vertices'][core['C_faces_model']])
        with np.load(P.resolve(q['noise_npz']),allow_pickle=False) as z:
            assert z['z'].shape==(q['N']-q['K'],3,3) and z['z'].dtype==np.float32
        if q['closed_C']:
            assert len(lay['occupied'])==0 and len(lay['seam_slots'])==0
        checked.append(dict(request_id=q['request_id'],closed_C=q['closed_C'],old_layout_byte_identity=True,source_C_identity=True,noise_shape=True))
    result=dict(status='PASS',requests=48,conditions=24,closed_requests=8,files_parsed=names,
        model_forwards=0,GPU_calls=0,OT_calls=0,new_noise=0,checks=checked)
    P.immutable(OUT/'evaluation_cpu_checks.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:r=cpu_check();print(dict(status=r['status'],requests=r['requests'],model_forwards=0),flush=True)
    else:
        m=run();print(dict(status=m['status'],requests=len(m['requests']),labels=m['new_labels']),flush=True)
