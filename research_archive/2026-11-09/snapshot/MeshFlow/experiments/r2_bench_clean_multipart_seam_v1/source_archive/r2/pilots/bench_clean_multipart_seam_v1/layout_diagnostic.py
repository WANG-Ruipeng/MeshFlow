"""Full316-condition C-only head reference audit; no generator/D0 calls."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,collections,csv,json,time
import numpy as np
from .eval_prepare import P,OUT,OLD,MODEL,requests,verify

def arrays(path):
    with np.load(P.resolve(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def csvwrite(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def prepare():
    path=OUT/'HEAD_DIAGNOSTIC_REGISTRATION.json'
    value=dict(status='FROZEN',version='bench_clean_full316_reference_layout_audit_v1',
        source_manifest_sha256=P.sha(OUT/'data/manifest.json'),eval_manifest_sha256=P.sha(OUT/'eval_manifest.json'),
        models=['OLD_S15000_HEAD','NEW_HEAD'],total_conditions_per_head=316,closed_conditions_per_head=49,
        open_conditions_per_head=267,reused_eval_open_conditions_per_head=20,new_GPU_head_calls_per_head=247,
        new_GPU_head_calls_total=494,source_labels_at_inference=False,threshold_logit=0.,
        calibration='NONE',ECE_bins=dict(count=10,boundaries=[i/10 for i in range(11)],domain='owner-positive probability vs source-owner frequency',
            summaries=['pooled edge ECE','C-equal mean of within-C ECE']),
        dtype_policy='GPU BF16 autocast for all new head calls; reuse original frozen GPU BF16 eval logits; FP32 parameters. No CPU precision mixture.',
        interpretation='Reference completion layout agreement, not correctness of all possible open-ended completions',
        generator_calls=0,D0_calls=0,backward_calls=0)
    P.immutable(path,value);return value

def collect(model):
    prepare()
    if model not in ('OLD_S15000_HEAD','NEW_HEAD'):raise ValueError('Unregistered head')
    for label in ('NEW_HEAD','OLD_S15000_LAYOUT'):
        rr=P.read(OUT/'evaluation'/label/'raw_complete_full.json')
        assert rr['status']=='COMPLETE' and rr['outputs']==48
        verify(rr['files'])
    import torch
    torch.set_num_threads(1)
    from meshflow_control import runtime
    runtime.configure_stable_runtime()
    from . import data as D,head as H
    from meshflow_control.seam.occupancy import OccupancyHead
    from r2.pilots.bench_parts_convergence_v1.data import edge_features
    from meshflow_control.checkpoints import state_hash
    m=P.read(OUT/'data/manifest.json');rows=m['rows']
    dest=OUT/'layout_diagnostic'/model
    if (dest/'completion.json').exists():
        done=P.read(dest/'completion.json');verify(done['files']);return done
    if dest.exists():raise RuntimeError('Unresolved diagnostic attempt; no silent forward retry')
    dest.mkdir(parents=True)
    from .common import resources
    from r2.pilots.seam_slot_c200_v1.generate import dispatch_guard
    from r2.pilots.bench_control_transfer_v1.train import resources as legacy_resources
    P.immutable(dest/'resources_before.json',resources())
    dispatch_guard(legacy_resources(),700)
    source=OUT/'inference'/f'{MODEL}_occupancy.pt' if model=='NEW_HEAD' else OLD/'training/SEAM/generator_step15000.pt'
    P.immutable(dest/'attempt.json',dict(model=model,source=str(source),source_sha256=P.sha(source),conditions=316,
        max_new_head_forwards=247,layout_cache_reuse=20,closedC_skips=49,device='GPU_BF16',source_labels_used_by_head=False))
    if model=='NEW_HEAD':head,payload=H.load_head(source,device='cuda')
    else:
        payload=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
        head=OccupancyHead(24).eval().requires_grad_(False);head.load_state_dict(payload['head_state'],strict=True);head.to('cuda')
    expected=state_hash(head.state_dict());eval_by={q['condition_id']:q for q in requests()}
    source_sha=P.sha(source)
    active={r['condition_id'] for r in m['train']};edges=[];conditions=[];calls=0
    for row in rows:
        cid=row['condition_id'];case=D.load_case(row);b=len(case['occupied_labels'])
        reused=False
        if cid in eval_by:
            lp=(OUT/'evaluation/NEW_HEAD/layouts' if model=='NEW_HEAD' else OLD/'evaluation/SEAM_S15000/layouts')/cid
            receipt=P.read(lp/'receipt.json')
            if 'files' in receipt:verify(receipt['files'])
            a=arrays(lp/'layout.npz');logits=a['occupancy_logits'];reused=True
            if model=='OLD_S15000_HEAD':
                assert receipt['checkpoint_sha256']==source_sha and receipt['layout_sha256']==P.sha(lp/'layout.npz')
            else:assert receipt['head']['sha256']==source_sha
            assert np.array_equal(a['edge_records'],edge_features(case['C_vertices'],case['C_faces_original'],int(case['N']))[0])
        elif not b:logits=np.zeros(0,np.float32)
        else:
            feat=H.features(case['C_vertices'],case['C_faces_original'],int(case['N'])) if model=='NEW_HEAD' else edge_features(case['C_vertices'],case['C_faces_original'],int(case['N']))[1]
            with (dest/'calls.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(event='HEAD_FORWARD_STARTED',condition_id=cid))+'\n')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                x=torch.from_numpy(feat).cuda()
                logits=(head(x) if model=='NEW_HEAD' else head(x,torch.ones(b,dtype=torch.bool,device='cuda'))).float().cpu().numpy()
            calls+=1
            with (dest/'calls.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(event='HEAD_FORWARD_RETURNED',condition_id=cid))+'\n')
        assert logits.shape==(b,)
        common=dict(model=model,condition_id=cid,parent_id=row['parent_id'],split=row['split'] if 'split' in row else row['group'],
                    part_type=row['part_type'],eligible_train=cid in active,cache_reused=reused,
                    precision_domain='CACHED_EVAL_GPU_BF16' if reused and b else 'GPU_BF16' if b else 'NA_CLOSED_C')
        pred=logits>=0;truth=np.asarray(case['occupied_labels'],bool);alias=np.asarray(case['source_alias_labels'],bool)
        supported=np.asarray(case['source_supported_labels'],bool)
        counts=collections.Counter()
        for i,z in enumerate(logits):
            ref='OWNER' if truth[i] else 'ALIAS' if alias[i] else 'NULL'
            hit='TP' if truth[i] and pred[i] else 'FN' if truth[i] else 'FP' if pred[i] else 'TN';counts[hit]+=1
            edges.append(dict(**common,edge_id=i,logit=float(z),probability=float(1/(1+np.exp(-float(np.clip(z,-80,80))))),
                predicted_active=bool(pred[i]),source_role=ref,source_owner=bool(truth[i]),source_supported=bool(supported[i]),reference_class=hit))
        conditions.append(dict(**common,boundary_edges=b,closed_C=b==0,active=int(pred.sum()),owner=int(truth.sum()),
            **{k:counts[k] for k in ('TP','FN','FP','TN')},source_alias=int(alias.sum()),source_null=int((~supported).sum())))
    assert state_hash(head.state_dict())==expected
    assert calls==247,(calls,'Expected267open minus20cached')
    P.immutable(dest/'resources_after.json',resources())
    csvwrite(dest/'per_edge.csv',edges);csvwrite(dest/'per_condition.csv',conditions)
    result=dict(status='COMPLETE',model=model,head_forwards=calls,device='GPU_BF16',parameters_before_after_equal=True,
        closed_conditions=49,eval_open_cache_reuse=20,conditions=len(rows),generator_forwards=0,D0_forwards=0,
        caveat='Held-out reference-layout agreement only; source completion does not define unique correct generation. All predictions use GPU BF16 forward with FP32 parameters.',
        files={str(p):P.sha(p) for p in dest.iterdir() if p.is_file()})
    P.immutable(dest/'completion.json',result);return result

def summarize():
    from . import common as C
    allrows=[];conditions=[]
    for model in ('OLD_S15000_HEAD','NEW_HEAD'):
        d=OUT/'layout_diagnostic'/model;verify(P.read(d/'completion.json')['files'])
        for name,target in (('per_edge',allrows),('per_condition',conditions)):
            with (d/(name+'.csv')).open(encoding='utf8') as f:target.extend(csv.DictReader(f))
    slit_path=OUT/'topology_audit/internal_slit_edges.csv'
    with slit_path.open(encoding='utf-8-sig') as f:
        slit={(r['condition_id'],int(r['edge_id'])):r for r in csv.DictReader(f)}
    for r in allrows:
        s=slit[r['condition_id'],int(r['edge_id'])]
        r['is_slit_candidate']=s['is_slit_candidate']=='True'
    summary=[];bins=[];per_c=[]
    for model in ('OLD_S15000_HEAD','NEW_HEAD'):
        for split in ('TRAIN','DEV'):
            for subset in ('ALL','ELIGIBLE','EXCLUDED'):
                rr=[r for r in allrows if r['model']==model and r['split']==split and (subset=='ALL' or (r['eligible_train']=='True')==(subset=='ELIGIBLE'))]
                if not rr:continue
                cc=collections.Counter(r['reference_class'] for r in rr)
                null=[r for r in rr if r['source_role']=='NULL'];alias=[r for r in rr if r['source_role']=='ALIAS']
                slit_edges=[r for r in rr if r['is_slit_candidate']]
                slit_null=[r for r in slit_edges if r['source_role']=='NULL']
                ece=0.
                for bi in range(10):
                    br=[r for r in rr if min(9,int(float(r['probability'])*10))==bi]
                    if not br:continue
                    confidence=np.mean([float(r['probability']) for r in br]);frequency=np.mean([r['source_owner']=='True' for r in br])
                    ece+=len(br)/len(rr)*abs(confidence-frequency)
                    bins.append(dict(model=model,split=split,subset=subset,bin=bi,edges=len(br),probability_mean=confidence,owner_frequency=frequency))
                ce=[]
                for cid in sorted({r['condition_id'] for r in rr}):
                    cr=[r for r in rr if r['condition_id']==cid];v=0.
                    for bi in range(10):
                        br=[r for r in cr if min(9,int(float(r['probability'])*10))==bi]
                        if br:v+=len(br)/len(cr)*abs(np.mean([float(r['probability']) for r in br])-np.mean([r['source_owner']=='True' for r in br]))
                    ce.append(v)
                    per_c.append(dict(model=model,split=split,subset=subset,condition_id=cid,edges=len(cr),ECE_10_equal_width=v))
                summary.append(dict(model=model,split=split,subset=subset,edges=len(rr),conditions=len(ce),**{k:cc[k] for k in ('TP','FN','FP','TN')},
                    owner_recall=cc['TP']/(cc['TP']+cc['FN']) if cc['TP']+cc['FN'] else None,
                    null_rejection=sum(r['predicted_active']=='False' for r in null)/len(null) if null else None,
                    alias_rejection=sum(r['predicted_active']=='False' for r in alias)/len(alias) if alias else None,
                    slit_candidate_edges=len(slit_edges),slit_candidate_active=sum(r['predicted_active']=='True' for r in slit_edges),
                    slit_source_null_edges=len(slit_null),slit_source_null_rejection=sum(r['predicted_active']=='False' for r in slit_null)/len(slit_null) if slit_null else None,
                    ECE_10_equal_width=ece,ECE_condition_equal=float(np.mean(ce)) if ce else None))
    dest=OUT/'layout_diagnostic'
    csvwrite(dest/'summary.csv',summary);csvwrite(dest/'ece_bins.csv',bins);csvwrite(dest/'ece_per_condition.csv',per_c)
    csvwrite(dest/'per_edge_with_slit.csv',allrows)
    P.immutable(dest/'completion.json',dict(status='COMPLETE',threshold=0.,calibration='NONE',ECE_bins=10,
        head_forward_total=494,generator_forwards=0,D0_forwards=0,files={str(dest/n):P.sha(dest/n) for n in ('summary.csv','ece_bins.csv','ece_per_condition.csv','per_edge_with_slit.csv')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','collect','summarize'))
    p.add_argument('--model',choices=('OLD_S15000_HEAD','NEW_HEAD'));a=p.parse_args()
    if a.action=='prepare':print(prepare(),flush=True)
    elif a.action=='collect':print(collect(a.model),flush=True)
    else:summarize()
