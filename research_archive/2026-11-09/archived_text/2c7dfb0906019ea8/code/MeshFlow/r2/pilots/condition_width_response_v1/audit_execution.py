"""Independent saved-artifact execution audit, CPU-only; no model or geometry execution."""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np
from . import common as C
from . import verify_saved as V


def source_freeze():
    f=C.read(C.OUT/'execution/gpu_code_freeze.json')
    V.require(f['registration_sha256']==C.sha(C.OUT/'registration.json'), 'registration changed')
    V.require(all(C.sha(p)==h for p,h in f['files'].items()), 'frozen source changed')
    V.require('torch' not in sys.modules, 'auditor unexpectedly imported Torch')
    return dict(gpu_code_freeze_sha256=C.sha(C.OUT/'execution/gpu_code_freeze.json'),
                registration_sha256=C.sha(C.OUT/'registration.json'),
                verifier_sha256=C.sha(Path(V.__file__)),auditor_sha256=C.sha(Path(__file__)))

def original_transform(q):
    original=C.arrays(q['original_input_path']);current=C.arrays(q['input_path'])
    V.require(C.sha(q['original_input_path'])==q['original_input_sha256'], 'original input hash')
    expected=original['C'].copy();expected[...,0]*=np.float32(q['alpha'])
    V.require(C.same(expected,current['C']) and C.same(current['z'],original['z']), 'C-only X transform/noise identity')
    source=C.arrays(q['condition_source']);before=C.arrays(q['original_condition_source'])
    V.require(C.sha(q['original_condition_source'])==q['original_condition_source_sha256'], 'original source hash')
    for key in ('C_source_corner_permutations','C_source_vertex_ids','C_original_source_vertex_ids',
                'source_face_ids','free_source_ids','C_original_source_face_ids'):
        if key in before: V.require(C.same(before[key],source[key]), 'source face/corner/vertex identity changed: '+key)
    return dict(C_only_X_transform_exact=True,original_noise_all_variants_bitwise=True,source_identity_and_corner_order_preserved=True)

def generator():
    hashes=source_freeze();r=C.read(C.OUT/'registration.json');rows=[]
    requests=r['requests'];expected={q['request_id'] for q in requests}
    root=C.OUT/'evaluation/raw'/V.LABEL
    V.require({p.name for p in root.iterdir() if p.is_dir()}==expected, 'unregistered/missing RAW directory')
    V.require({p.parent.name for p in root.glob('*/calls.jsonl')}==expected, 'unexpected RAW call ledger')
    for q in requests:
        rec=C.read(root/q['request_id']/'receipt.json')
        facts=V.verify_raw(q,rec,r['checkpoint']);facts.update(original_transform(q))
        rows.append(dict(kind='RAW',**facts))
    bases=[q for q in requests if q['variant']=='base']
    probe_dirs={C.OUT/'probes'/q['base_request_id']/('step_'+str(s).zfill(2)) for q in bases for s in V.STEPS}
    actual_dirs={p for p in (C.OUT/'probes').glob('*/*') if p.is_dir()}
    V.require(actual_dirs==probe_dirs, 'unregistered/missing probe directory')
    actual_calls=set((C.OUT/'probes').glob('*/*/*_call.json'))
    actual_returns=set((C.OUT/'probes').glob('*/*/*_return.json'))
    V.require(actual_calls=={p/(v+'_call.json') for p in probe_dirs for v in ('m095','p105')}, 'unregistered/missing probe call')
    V.require(actual_returns=={p/(v+'_return.json') for p in probe_dirs for v in ('m095','p105')}, 'unregistered/missing probe return')
    for base in bases:
        for step in V.STEPS:
            dest=C.OUT/'probes'/base['base_request_id']/('step_'+str(step).zfill(2))
            rows.append(dict(kind='PROBE_PAIR',**V.verify_probe(base,step,dest,r['checkpoint'])))
    completed=C.read(C.OUT/'execution/generator_complete.json')
    V.require(completed['status']=='COMPLETE' and completed['new_sampling_F']==2400
        and completed['new_probe_F']==128 and completed['training']==completed['backward']==0, 'generator completion count')
    summary=dict(status='PASS',**hashes,raw_outputs=48,raw_rollout_reserves=48,
        sampling_forward_reserves=2400,sampling_forward_returns=2400,probe_pairs=64,
        probe_forward_reserves=128,probe_forward_returns=128,total_generator_forwards=2528,
        pending_calls=0,extra_saved_call_cells=0,training_updates=0,backwards=0,
        distinguishable_probe_pairs=sum(x.get('known_embedding_changed') is True for x in rows),
        unresolved_indistinguishable_probe_pairs=sum(x.get('known_embedding_changed') is False for x in rows),
        all_C_X_only_noise_and_source_mapping_checks=True,all_50_step_Euler_recurrences_bitwise=True,
        actual_call_scope='Closed saved ledgers and frozen single-call implementation; verifier executes no model.',
        model_calls_by_auditor=0,geometry_calls_by_auditor=0,torch_imported=False,
        generator_completion_sha256=C.sha(C.OUT/'execution/generator_complete.json'))
    C.csvwrite(C.OUT/'audit/generator_execution_percell.csv',rows)
    summary['percell_csv_sha256']=C.sha(C.OUT/'audit/generator_execution_percell.csv')
    C.write(C.OUT/'audit/generator_execution_validation.json',summary)
    return summary

def _ply(path):
    with Path(path).open('rb') as f:
        lines=[]
        for _ in range(32):
            line=f.readline(1024).decode('ascii').strip();lines.append(line)
            if line=='end_header': break
        V.require(lines[-1]=='end_header' and 'format binary_little_endian 1.0' in lines, 'POST PLY header')
        nv=int(next(x for x in lines if x.startswith('element vertex ')).split()[-1])
        nf=int(next(x for x in lines if x.startswith('element face ')).split()[-1])
        vertices=np.frombuffer(f.read(nv*12),dtype='<f4').copy().reshape(nv,3)
        a=np.frombuffer(f.read(nf*13),dtype=np.dtype([('n','u1'),('v','<i4',(3,))]))
        V.require(len(a)==nf and np.all(a['n']==3) and not f.read(1), 'POST PLY body')
    return vertices,a['v'].astype(np.int64)

def post():
    hashes=source_freeze();r=C.read(C.OUT/'registration.json');requests=r['requests']
    complete=C.read(C.OUT/'execution/D0_complete.json');index=C.read(C.OUT/'evaluation/post_index.json')
    V.require(complete['status']==index['status']=='COMPLETE' and len(index['rows'])==48, 'POST not closed')
    events=[json.loads(x) for x in (C.OUT/'execution/events.jsonl').read_text().splitlines()]
    V.require(not any(e['event'] in ('GENERATOR_STOPPED','POST_STOPPED') for e in events), 'saved failure event requires separate audit')
    d0=[e for e in events if e['event'].startswith('D0_FORWARD_')]
    V.require(len(d0)==96, 'D0 event count')
    ids={q['request_id'] for q in requests}
    root=C.OUT/'evaluation/post'/V.LABEL
    V.require({p.name for p in root.iterdir() if p.is_dir()}==ids, 'unregistered/missing POST cell')
    for i,q in enumerate(requests):
        a,b=d0[2*i:2*i+2]
        V.require(a['event']=='D0_FORWARD_RESERVED' and b['event']=='D0_FORWARD_RETURNED'
            and a['request_id']==b['request_id']==q['request_id'], 'D0 reserve/return sequence')
    rows=[]
    for q in requests:
        dest=root/q['request_id'];rec=C.read(dest/'receipt.json')
        V.require(rec['status']=='COMPLETE' and rec['GEO_calls']==1, 'POST completion/GEO count')
        V.require(rec['request_id']==q['request_id'] and rec['condition_source_sha256']==q['condition_source_sha256'], 'POST request/source binding')
        for path,h in rec['files'].items():
            V.require(C.resolve(path).parent.resolve()==dest.resolve() and C.sha(path)==h, 'POST artifact hash/location')
        V.require(set(rec['files'])=={str(p) for p in dest.iterdir() if p.is_file() and p.name!='receipt.json'}, 'POST file manifest coverage')
        rawrec=C.read(C.OUT/'evaluation/raw'/V.LABEL/q['request_id']/'receipt.json')
        V.require(rec['raw_sha256']==rawrec['raw_sha256'], 'POST RAW binding')
        raw=C.arrays(rec['raw_path'])['raw'].reshape(-1,3,3);k=q['K'];n=q['N']
        proposal=C.arrays(rec['proposal_path'])['official_unconstrained_proposal']
        pr=C.read(rec['proposal_receipt']);start=C.read(dest/'D0_started.json')
        V.require(pr['status']=='PASS' and pr['model_forwards']==1 and start['status']=='RESERVED', 'D0 call receipts')
        V.require(pr['raw_sha256']==start['raw_sha256']==rec['raw_sha256']
            and pr['checkpoint_sha256']==start['checkpoint_sha256']==r['d0_checkpoint']['sha256']
            and pr['output_sha256']==C.sha(rec['proposal_path']), 'D0 input/model/output identity')
        V.require(proposal.dtype==np.float32 and proposal.shape==raw.shape and np.isfinite(proposal).all(), 'D0 proposal shape')
        pre=C.arrays(rec['pre_geo_path']);canonical=C.arrays(rec['canonical_path']);p=C.arrays(rec['post_path'])
        src=C.arrays(q['condition_source']);V.require(C.sha(q['condition_source'])==q['condition_source_sha256'], 'source hash')
        expected=proposal.copy();expected[:k]=raw[:k]
        V.require(C.same(pre['output'],expected) and C.same(pre['C'],raw[:k]), 'C restoration before GEO altered free or C')
        V.require(C.same(canonical['C'],raw[:k]) and C.same(canonical['output'][:k],raw[:k]), 'GEO canonical C preservation')
        original=src['C_original'];V.require(C.same(p['C'],original) and C.same(p['C_original'],original)
            and C.same(p['output'][:k],original), 'source-order C not exactly restored')
        V.require(C.same(p['vertices'],canonical['vertices']) and C.same(p['source_face_ids'],canonical['source_face_ids']), 'orientation changed vertices or face identity')
        perm=p['corner_permutation_to_MODEL_POST'];m=len(p['faces'])
        V.require(perm.shape==(m,3) and np.array_equal(np.sort(perm,axis=1),np.tile(np.arange(3),(m,1))), 'export permutation invalid')
        V.require(C.same(p['faces'],canonical['faces'][np.arange(m)[:,None],perm])
            and C.same(p['output'],canonical['output'][np.arange(m)[:,None],perm]), 'export changed unordered geometry')
        V.require(C.same(p['output'],p['vertices'][p['faces']]) and C.same(p['free'],p['output'][k:]), 'POST indexed geometry mismatch')
        V.require(int(p['known_faces'])==k and int(p['input_faces'])==n, 'POST N/K metadata')
        v,f=_ply(dest/'post.ply');V.require(C.same(v,p['vertices']) and C.same(f,p['faces']), 'POST PLY readback')
        original_ids=src['C_original_source_vertex_ids'];V.require(C.same(original_ids,p['C_original_source_vertex_ids']), 'original C vertex IDs changed')
        forward={};reverse={}
        for old,new in zip(original_ids.reshape(-1),p['faces'][:k].reshape(-1)):
            old,new=int(old),int(new)
            V.require(old not in forward or forward[old]==new, 'source C vertex ID split')
            V.require(new not in reverse or reverse[new]==old, 'distinct source C IDs merged')
            forward[old]=new;reverse[new]=old
        V.require(rec['source_restoration']['status']=='PASS' and rec['source_restoration']['orientation_solver_calls']==1
            and rec['source_restoration']['model_calls']==0 and rec['source_restoration']['quality_calls']==0, 'source export execution counts')
        rows.append(dict(request_id=q['request_id'],variant=q['variant'],status='PASS',D0_reserved=1,D0_returned=1,
            GEO_calls_recorded=1,orientation_calls_recorded=1,pre_GEO_C_exact=True,GEO_C_exact=True,
            source_order_C_exact=True,source_ID_mapping_injective=True,orientation_geometry_unchanged=True,
            post_PLY_exact=True,model_calls_by_auditor=0,geometry_calls_by_auditor=0))
    V.require('torch' not in sys.modules, 'auditor unexpectedly imported Torch')
    summary=dict(status='PASS',**hashes,POST_outputs=48,D0_reserves=48,D0_returns=48,
        recorded_GEO_calls=48,recorded_orientation_solver_calls=48,pending_D0_calls=0,
        extra_saved_D0_cells=0,all_stage_C_and_source_identity_checks=True,
        model_calls_by_auditor=0,geometry_calls_by_auditor=0,torch_imported=False,
        topology_audit_scope='Saved source vertex IDs mapped bijectively to restored C indices; no edge graph/detector recomputation.',
        orientation_scope='Coordinate/face identity and permutation invariants only; does not certify global topology or geometry quality.',
        D0_completion_sha256=C.sha(C.OUT/'execution/D0_complete.json'),post_index_sha256=C.sha(C.OUT/'evaluation/post_index.json'),
        closed_D0_event_log_sha256=C.sha(C.OUT/'execution/events.jsonl'))
    C.csvwrite(C.OUT/'audit/post_execution_percell.csv',rows)
    summary['percell_csv_sha256']=C.sha(C.OUT/'audit/post_execution_percell.csv')
    C.write(C.OUT/'audit/post_execution_validation.json',summary)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('generator','post'),required=True)
    args=parser.parse_args();value=generator() if args.stage=='generator' else post()
    print(json.dumps(value,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
