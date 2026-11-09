"""CPU-only TRAIN47 source-edge preparation and exact old8000 stream audit."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,collections,csv,json,time
import numpy as np
from . import training_data as D

OLD=D.ROOT/'experiments/r2_seam_pool47_couple_v1'
QUALITY=D.ROOT/'experiments/r2_seam_quality_wave1_wave2_v1/wave2_inputs/source_labels.json'

def edges_for_parent(record,label):
    source=D.source_arrays(record['parent_npz'],record['parent_sha256'])
    quality=D.read(D.resolve(record['quality_path']))
    if D.digest(record['quality_path'])!=record['quality_sha256']:raise ValueError('Source detector table changed')
    faces=source['source_faces'];canonical=source['source_face_vertex_ids']
    contacts={tuple(sorted(map(int,r['pair']))):r['classification'] for r in quality['pair_contacts']}
    edge_inc=collections.defaultdict(list)
    for fi,face in enumerate(faces):
        for j in range(3):
            u,v=map(int,(face[j],face[(j+1)%3]));edge_inc[tuple(sorted((u,v)))].append((fi,u,v))
    free=set(label['free_source_ids']);owner=set(label['owner_source_ids']);dirty=set(label['corrupted_source_ids'])
    rows=[];audit=[];counts=collections.Counter()
    for edge,inc in sorted(edge_inc.items()):
        if len(inc)!=2:continue
        (a,u,v),(b,s,t)=inc
        if a not in free or b not in free:continue
        category='SEAM_SEAM' if a in owner and b in owner else 'SEAM_BULK' if a in owner or b in owner else 'BULK_BULK'
        if category=='SEAM_SEAM':reason='EXCLUDE_SEAM_SEAM_SCOPE'
        elif a in dirty or b in dirty:reason='EXCLUDE_DIRTY_OR_UNKNOWN_FACE'
        elif (u,v)!=(t,s):reason='EXCLUDE_SAME_DIRECTION'
        elif contacts.get(tuple(sorted((a,b))))!='shared_edge_contact':reason='EXCLUDE_NOT_RECORDED_SHARED_EDGE_CONTACT'
        else:reason='ACCEPT'
        counts[reason]+=1
        audit.append(dict(parent_id=record['parent_id'],face_a=int(a),face_b=int(b),vertex_u=u,vertex_v=v,category=category,
                          source_detector_class=contacts.get(tuple(sorted((a,b))),'NA'),decision=reason))
        if reason=='ACCEPT':
            ca=[int(np.flatnonzero(canonical[a]==x).item()) for x in (u,v)]
            cb=[int(np.flatnonzero(canonical[b]==x).item()) for x in (u,v)]
            rows.append([int(a),int(b),*ca,*cb]);counts['ACCEPT_'+category]+=1
    return rows,audit,dict(counts)

def cpu_checks():
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only preparation')
    z=np.arange(45,dtype=np.float32).reshape(5,3,3)/7
    mask=np.array([False,True,True,False,True]);out=D.transform_bulk_noise(z,mask)
    if not np.array_equal(out[~mask],z[~mask]):raise ValueError('Nonbulk prior altered')
    if not np.allclose(out.mean(1),z.mean(1),atol=1e-6,rtol=0):raise ValueError('Centroid moved')
    inv=out.mean(1,keepdims=True)+(out-out.mean(1,keepdims=True))/np.float32(.25)
    if not np.allclose(inv[mask],z[mask],atol=3e-6,rtol=0):raise ValueError('Prior invertibility check')
    # Toy endpoint gradient only; no model construction/forward, no optimizer.
    p=torch.arange(18,dtype=torch.float32).reshape(1,2,9).requires_grad_()
    sample=dict(t=np.float32(.5),xt=np.zeros((2,3,3),np.float32),anchor_ids=np.array([[0,-1,-1],[-1,-1,-1]]),
                edge_indices=np.array([[0,1,3,4]],np.int64),edge_count=1)
    loss,_=D.edge_coherence(p,sample,2);loss.backward();actual=p.grad.detach().numpy().reshape(-1,3)
    end=.5*p.detach().numpy().reshape(-1,3);end[0]=0
    d0=end[0]-end[3];d1=end[1]-end[4]
    expected=np.zeros((6,3),np.float32)
    expected[0]=(2*d0-d1)*.5/2;expected[1]=(2*d1-d0)*.5/2
    expected[3]=-expected[0];expected[4]=-expected[1];expected[0]=0
    if not np.allclose(actual,expected,atol=1e-6,rtol=0):raise ValueError('Edge formula gradient mismatch')
    return dict(status='PASS',neural_constructions=0,neural_forwards=0,neural_backwards=0,toy_coordinate_backward=1,
                optimizer_updates=0,CUDA_initialized=False,anchor_gradient_zero=True,whole_batch_edge_denominator=True,
                prior_centroid_preserved=True,prior_nonbulk_unchanged=True,prior_invertible=True)

def prepare(root):
    root=D.resolve(root).resolve();root.mkdir(parents=True,exist_ok=True);out=root/'training_inputs';out.mkdir(exist_ok=True)
    regpath=root/'training_registration.json'
    if regpath.exists():
        reg=D.read(regpath)
        for p,h in reg['protected_hashes'].items():
            if D.digest(p)!=h:raise ValueError('Frozen dependency changed '+p)
        return reg
    if (out/'started.json').exists():raise RuntimeError('Partial prepare requires explicit engineering recovery')
    started=time.perf_counter()
    D.write_new(out/'started.json',dict(status='CPU_PREPARATION_STARTED',time=time.time(),new_OT=0,model_calls=0))
    prior=D.read(OLD/'training_registration.json');manifest=D.read(OLD/'shared_data_manifest.json');schedule=D.read(OLD/'shared_schedule.json')
    if manifest['status']!='COMPLETE' or len(manifest['steps'])!=3000:raise ValueError('Old complete stream needed')
    labels=D.read(QUALITY)['labels_by_parent'];records=prior['records']
    if len(labels)!=47 or len(records)!=47 or any(r['split']!='TRAIN' for r in records):raise ValueError('Fixed TRAIN47 only')
    checkpoint=OLD/'checkpoints/PLAIN3000_step1000.pt';receipt=D.read(checkpoint.with_suffix('.json'))
    if D.digest(checkpoint)!=receipt['sha256'] or receipt['global_step']!=14000 or receipt['local_step']!=1000:raise ValueError('Required PLAIN_S1000 live checkpoint differs')
    import sys
    sys.path.insert(0,str(D.ROOT/'r2/src'))
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.training.schedule40 import tree_hash
    from meshflow_control.config import MODEL_CONFIG
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if (payload['schema'],payload['arm'],payload['global_step'],payload['seam_total_step'],payload['update_in_progress'],payload['EMA'])!=('r2_seam_pool47_couple_training_v1','PLAIN3000',14000,2000,False,False):raise ValueError('Live source payload contract differs')
    if payload['model_config']!=MODEL_CONFIG:raise ValueError('Model config changed')
    for key,fn,hkey in (('generator_state',state_hash,'generator_state_sha256'),('optimizer',tree_hash,'optimizer_state_sha256'),('rng',tree_hash,'rng_sha256')):
        if fn(payload[key])!=receipt[hkey]:raise ValueError('Source internal state hash failed '+key)
    states=payload['optimizer']['state'];steps=sorted({float(s['step']) for s in states.values()})
    if len(states)!=194 or steps!=[14000.]:raise ValueError('Matched AdamW source steps differ')
    D.write_new(out/'source_checkpoint_audit.json',dict(status='PASS',source_sha256=receipt['sha256'],
        global_step=14000,generator_state_sha256=receipt['generator_state_sha256'],optimizer_state_sha256=receipt['optimizer_state_sha256'],
        rng_sha256=receipt['rng_sha256'],optimizer_states=len(states),optimizer_steps=steps,
        source_schema=payload['schema'],CUDA_initialized=torch.cuda.is_initialized(),model_constructions=0))
    del payload,states
    protected={str(p):D.digest(p) for p in (OLD/'training_registration.json',OLD/'shared_data_manifest.json',OLD/'shared_schedule.json',QUALITY,checkpoint,checkpoint.with_suffix('.json'))}
    rows=[];counts={}
    for record in records:
        uid=record['parent_id'];edge,detail,cnt=edges_for_parent(record,labels[uid]);rows+=detail;counts[uid]=cnt
        labels[uid]['clean_edges']=edge
        labels[uid]['edge_categories']={k[7:]:v for k,v in cnt.items() if k.startswith('ACCEPT_')}
        for field,hfield in (('parent_npz','parent_sha256'),('condition_path','condition_sha256'),('quality_path','quality_sha256')):
            if D.digest(record[field])!=record[hfield]:raise ValueError('Source asset changed')
            protected[record[field]]=record[hfield]
    with (out/'source_edge_selection.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    D.write_new(out/'source_labels.json',dict(status='FROZEN',labels_by_parent=labels,old_quality_labels=str(QUALITY),
        inherited_dirty_definition='1138 ILLEGAL + 3 UNKNOWN Y/Y pairs; whole incident source faces excluded conservatively',
        eligible_edges='Original source indexed incidence2, opposite winding, primary shared_edge_contact, both Y faces clean',
        edge_counts=counts,DEV_labels_read=False))
    chosen=[]
    for oldstep,oldplan in zip(manifest['steps'][1000:2000],schedule['steps'][1000:2000]):
        if oldstep['step']!=oldplan['step']:raise ValueError('Schedule window mismatch')
        rr=[]
        for m,(record,seedrow) in enumerate(zip(oldstep['samples'],oldplan['rows'])):
            if record['uid']!=seedrow['uid'] or record['alpha']!=seedrow['alpha'] or record['sample_index']!=m:raise ValueError('Exact exposure mismatch')
            rr.append({**record,'frozen_random_stream':seedrow})
        chosen.append(dict(step=oldstep['step']-1000,source_continuation_step=oldstep['step'],source_stream_step=oldstep['source_step'],samples=rr))
    exp=collections.Counter();total_edges=0;stream=__import__('hashlib').sha256()
    max_prior_roundoff=0.
    for item in chosen:
        for micro,r in enumerate(item['samples']):
            s=D.load_sample(r,root,D.ARMS[0],item['step'],micro,labels)
            source=D.source_arrays(r['parent_npz'],r['parent_sha256']);full=source['full_target'].reshape(s['N'],3,3).copy();full[:,:,0]*=np.float32(s['alpha'])
            expected=full[s['source_face_ids'][:,None],s['source_corner_permutations']]
            if not np.array_equal(expected,s['x1']):raise ValueError('Source target transport mismatch')
            z,eps=D.recover_existing_noise(s,r)
            smooth=D.load_sample(r,root,D.ARMS[1],item['step'],micro,labels)
            bulk=~s['known_mask'] & s['free_corner_mask'].all(1)
            alternative=D.transform_bulk_noise(s['x0'],bulk)
            max_prior_roundoff=max(max_prior_roundoff,float(np.max(np.abs(smooth['x0']-alternative))))
            exp[s['uid']]+=1;total_edges+=s['edge_count'];stream.update(json.dumps(r,sort_keys=True).encode())
        if item['step']%100==0:print(json.dumps(dict(stage='SOURCE_OT_PRIOR_AUDIT',step=item['step'],samples=sum(exp.values()),models=0)),flush=True)
    if len(exp)!=47 or sum(exp.values())!=8000:raise ValueError('Exposure mismatch')
    D.write_new(out/'shared_data_manifest.json',dict(status='COMPLETE',schema='hinge_prior_existing_8000_v1',steps=chosen,samples=8000,
        source_manifest=str(OLD/'shared_data_manifest.json'),source_window=[1001,2000],original_OT_assignment_preserved=True,new_OT_calls=0,
        parent_exposures=dict(sorted(exp.items())),record_sequence_sha256=stream.hexdigest()))
    D.write_new(out/'transport_audit.json',dict(status='PASS',samples=8000,parents=47,edge_instances=total_edges,
        neural_forwards=0,source_targets_bitwise=True,shared_endpoints_after_OT_bitwise=True,original_x0_gaussian_replay_bitwise=True,
        smooth_Z_and_epsilon_both_transformed=True,smooth_T_after_hybrid_roundoff_max=max_prior_roundoff,
        source_edge_total=sum(len(l['clean_edges']) for l in labels.values()),cpu_contract=cpu_checks(),seconds=time.perf_counter()-started))
    for name in ('source_labels.json','shared_data_manifest.json','transport_audit.json','source_edge_selection.csv'):protected[str(out/name)]=D.digest(out/name)
    code=[Path(__file__),Path(D.__file__),Path(__file__).with_name('training.py')]
    for p in code:protected[str(p)]=D.digest(p)
    reg=dict(status='FROZEN',schema='r2_seam_hinge_training_registration_v1',arms=list(D.ARMS),source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=receipt['sha256'],source_generator_sha256=receipt['generator_state_sha256'],
        source_optimizer_sha256=receipt['optimizer_state_sha256'],source_rng_sha256=receipt['rng_sha256'],
        source_global_step=14000,final_global_step=15000,source_seam_step=2000,final_seam_step=3000,updates_per_arm=1000,
        batch=8,microbatch=1,records=records,shared_data_manifest=str(out/'shared_data_manifest.json'),source_labels=str(out/'source_labels.json'),
        dirty_bulk_weight=.2,owner_third_weight=1,edge_lambda=.1,prior_rho=.25,
        edge_denominator='WHOLE_BATCH_EDGE_COUNT; formula uses sum of3 coordinates, no additional /3',
        fm_denominator='CLEAN weighted free scalar sum; SMOOTH ordinary free scalar sum',
        diagnostic_time_bin_edges=[0,.2,.4,.6,.8,1],optimizer=dict(lr=1e-5,betas=[.9,.95],eps=1e-8,weight_decay=0.,clip=1.),
        optimizer_restoration='EXACT_MATCHED_PLAIN_S1000_LIVE_ADAMW_RNG_NO_RESET',
        source_window=[1001,2000],new_OT_calls=0,model_probe_budget=0,head_policy='NO_LOAD_NO_CALL_NO_UPDATE',
        smooth_prior_assignment='Keep exact cached original-noise OT; transform the same Z and epsilon only on bulk before HYBRID',
        no_automatic_retry=True,protected_hashes=protected,code_hashes={str(p):D.digest(p) for p in code})
    D.write_new(regpath,reg);return reg

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='experiments/r2_seam_hinge_quality_v1');a=p.parse_args()
    r=prepare(a.root);print(json.dumps(dict(status=r['status'],arms=r['arms'],updates=r['updates_per_arm'])))
