"""Full CPU scorer check on an explicitly labelled SOURCE_FIXTURE, never a model output."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
import difflib
from pathlib import Path
import sys
import time
import numpy as np
from . import panel as P, evaluation as V
from r2.pilots.seam_abc_v1 import evaluate as E, eval_common as C
from r2.pilots.seam_pool47_couple_v1.endpoint_scoring import metric_row


def arrays(path):
    with np.load(P.resolve(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def run(root=P.OUT):
    root=P.resolve(root);out=root/'audits/closed_C_SOURCE_FIXTURE';out.mkdir(parents=True,exist_ok=True)
    final=out/'verification.json'
    if final.exists():
        old=P.read(final)
        for path,digest in old['protected'].items():
            if P.sha(path)!=digest:raise ValueError('Protected SOURCE_FIXTURE input/evidence changed')
        return old
    start=time.perf_counter();C.resources(out/'resources_before.json')
    registration=P.read(root/'evaluation/score_registrations/NATIVE_S12000.json')
    q=dict(next(q for q in registration['requests'] if q['closed_C']))
    source_manifest=P.read(root/'data/manifest.json')
    record=next(r for r in source_manifest['rows'] if r['condition_id']==q['condition_id'])
    case=arrays(record['case_path']);inp=arrays(q['inference_npz']);meta=arrays(q['metadata_npz'])
    n,k=q['N'],q['K'];order=np.concatenate([case['known_ids'],case['free_ids']]).astype(np.int64)
    if len(order)!=n or len(np.unique(order))!=n:raise AssertionError('Source face partition invalid')
    tri=case['full_target'][order].copy();source_ids=case['source_face_vertex_ids'][order].copy()
    tri[:k]=inp['C_vertices'][inp['C_faces_original']]
    source_ids[:k]=meta['C_source_vertex_ids'][inp['C_faces_original']]
    coords={}
    for vid,point in zip(source_ids.ravel(),tri.reshape(-1,3)):
        vid=int(vid)
        if vid in coords and coords[vid].tobytes()!=point.tobytes():raise AssertionError('Source vertex ID coordinate conflict')
        coords[vid]=point.copy()
    ids=sorted(coords);index={vid:i for i,vid in enumerate(ids)}
    vertices=np.asarray([coords[i] for i in ids],np.float32)
    faces=np.asarray([[index[int(v)] for v in face] for face in source_ids],np.int64)
    fixture=dict(output=tri,vertices=vertices,faces=faces,raw_slot_ids=np.arange(n,dtype=np.int64),
        raw_seam_slot_mask=np.zeros(n,bool),known_faces=np.asarray(k,np.int64),input_faces=np.asarray(n,np.int64),
        source_face_kept=np.ones(n,bool),source_face_reason=np.full(n,'SOURCE_FIXTURE_KEEP'),
        duplicate_of_source_face=np.full(n,-1,np.int64),index_domain=np.asarray('SOURCE_FIXTURE_ORIGINAL_IDS_NOT_MODEL'))
    path=out/'source_fixture.npz';P.array_file(path,fixture)
    protected={str(P.resolve(record['case_path'])):P.sha(record['case_path']),q['inference_npz']:P.sha(q['inference_npz']),
        q['metadata_npz']:P.sha(q['metadata_npz']),q['source_npz']:P.sha(q['source_npz']),
        q['query_descriptor']['path']:P.sha(q['query_descriptor']['path'])}
    receipt=out/'SOURCE_FIXTURE_RECEIPT.json'
    P.immutable(receipt,dict(status='SOURCE_FIXTURE_NOT_GENERATED',purpose='CPU empty-boundary scorer engineering test only',
        source_case_path=record['case_path'],source_case_sha256=P.sha(record['case_path']),source_face_order=order.tolist(),
        C_original_winding=True,free_orientation='Original canonical training-target corner order',
        generator_forwards=0,D0_forwards=0,training_updates=0,noise_draws=0,new_query_draws=0,
        source_fixture_sha256=P.sha(path),not_for_formal_results=True))
    label='SOURCE_FIXTURE';q['request_id']='SOURCE_FIXTURE__'+q['condition_id']
    api=V.private_score(q,label,True)
    def fixture_artifact(_root,_label,_request,_stage):
        if (_label,_request,_stage)!=(label,q['request_id'],'RAW'):raise ValueError('Unexpected fixture identity')
        return path,receipt
    api['artifact']=fixture_artifact
    result=api['score_one']((str(out),label,q,'RAW',P.sha(receipt)))
    score=P.read(result['path']);folder=Path(result['path']).parent
    row,roles,deleted=metric_row(score,fixture,P.read(folder/'pairs.json'))
    numeric=P.read(P.ROOT/'experiments/r2_bench_control_transfer_v1/evaluation/summary_completion.json')['numeric_fields']
    if len(numeric)!=124 or any(key not in row for key in numeric):raise AssertionError('Original 124 fields missing')
    if score['seam_to_bulk']['status']!='NA_CLOSED_C_NO_SEAM_INTERFACE':raise AssertionError('Closed seam must be NA')
    if any(v is not None for key,v in score['seam_to_bulk'].items() if key!='status'):raise AssertionError('Closed seam leaked zero/success')
    if len(score['boundary'])!=8 or any(b['status']!='NA_EMPTY_BOUNDARY' or b['VMR'] is not None or b['perfect'] is not None for b in score['boundary']):
        raise AssertionError('Empty boundary must be NA in all eight settings')
    if not score['C_preserved_exact'] or score['free_face_denominator']!=n-k:raise AssertionError('Source fixture C/free mismatch')
    if score['status_by_metric']['S']!='NA_NO_REQUIRED_INTERFACE':raise AssertionError('Unspecified required interface became mandatory')
    for key in ('FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction','ConflictCov_C'):
        if not V.number(row[key]):raise AssertionError('Ordinary quality metric did not run: '+key)
    C.csvwrite(out/'SOURCE_FIXTURE_124_FIELDS_NOT_GENERATION.csv',[row])
    # Assert every inference/source metadata archive contains C identity only.
    whitelist={'inference_npz':{'C_vertices','C_faces_original','C_faces_model','N','K'},
        'metadata_npz':{'C_source_vertex_ids','known_ids'}}
    input_checks=[]
    for item in P.read(root/'evaluation_inputs/manifest.json')['requests']:
        for key in ('inference_npz','metadata_npz','source_npz'):
            with np.load(item[key],allow_pickle=False) as z:keys=set(z.files)
            if key in whitelist and keys!=whitelist[key]:raise AssertionError('Inference field whitelist differs')
            if any(name in keys for name in ('Y','full_target','occupied_labels','edge_owner_face','source_alias_labels')):
                raise AssertionError('Target/source owner leaked into inference metadata')
            input_checks.append(dict(path=item[key],kind=key,keys=sorted(keys),target_owner_absent=True))
    for filename in ('source_fixture.npz','SOURCE_FIXTURE_RECEIPT.json','SOURCE_FIXTURE_124_FIELDS_NOT_GENERATION.csv'):
        protected[str(out/filename)]=P.sha(out/filename)
    protected[result['path']]=P.sha(result['path'])
    for path,digest in protected.items():
        if P.sha(path)!=digest:raise AssertionError('Protected artifact changed')
    if 'torch' in sys.modules:raise AssertionError('CPU scorer imported neural framework')
    report=dict(status='PASS_SOURCE_FIXTURE_ONLY',not_generation_result=True,model_constructions=0,model_forwards=0,
        D0_forwards=0,training_updates=0,noise_draws=0,new_query_draws=0,condition_id=q['condition_id'],N=n,K=k,
        full_scorer_status=score['status'],numeric_field_count=124,AVMR_PMR_all8='NA_EMPTY_BOUNDARY',
        seam_graph_status=score['seam_to_bulk']['status'],CF_FF_I='Finite ordinary quality values computed; not model performance',
        source_metadata_checks=input_checks,protected=protected,seconds=time.perf_counter()-start,code_sha256=P.sha(__file__))
    P.immutable(final,report);C.resources(out/'resources_finished.json');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(P.OUT));a=p.parse_args();r=run(a.root)
    print(r['status'],r['condition_id'],r['AVMR_PMR_all8'],r['seam_graph_status'])
