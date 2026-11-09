"""Freeze complete source patches and existing full-N noise; zero model calls."""
from pathlib import Path
import copy
from . import common as C
BASE=C.ROOT/'experiments/r2_c_reference_frame_v1/evaluation'
LABELS=['SMALL_C200','CREF_C200']

def prepare():
    target=C.OUT/'registration.json'
    if target.exists(): return C.read(target)
    if list((C.OUT/'evaluation').glob('raw/*/*/calls.jsonl')): raise RuntimeError('No reprepare after inference')
    audit=C.read(C.OUT/'preparation/condition_audit.json'); old=C.read(BASE/'registration.json')
    reqs=[]; queries=[]; models={LABELS[0]:old['models'][LABELS[0]],LABELS[1]:C.read(BASE/'bindings/CREF_C200.json')}
    for model in [*models.values(),old['d0_checkpoint']]:
        if C.sha(model['path'])!=model['sha256']:raise ValueError('Model hash mismatch')
    for ci,cond in enumerate(audit['conditions']):
        uid=cond['parent_id']; source=C.arrays(cond['condition_path']); n,k=cond['N'],cond['K']; c=source['C'].reshape(k,3,3)
        if C.sha(cond['condition_path'])!=cond['condition_sha256']:raise ValueError('Input source changed')
        if cond['topology']['component_count']!=1 or cond['topology']['boundary_loop_count']!=1 or k<2:raise ValueError('Patch qualification')
        selected=sorted([q for q in old['requests'] if q['parent_id']==uid and q['candidate'] in range(4)],key=lambda q:q['candidate'])
        if len(selected)!=4:raise ValueError('Four frozen noises not available')
        cache=cond['already_FULL_in_previous_panel']; cid=uid+'__SEAT_FULL__base'
        for original in selected:
            q=copy.deepcopy(original); noise=C.arrays(q['full_noise_path'])['full_Gaussian']
            if C.sha(q['full_noise_path'])!=q['full_noise_file_sha256'] or C.array_sha(noise)!=q['full_Gaussian_sha256']:raise ValueError('Full noise changed')
            z=noise.copy();z[:,:k]=0
            q.update(action='CACHE' if cache else 'NEW',original_partial_request_id=original['request_id'],
                     condition_id=cid,base_condition_id=uid+'__SEAT_FULL',condition_type='SEAT_FULL',tier='FULL',K=k,N=n,ratio=k/n,
                     pair_id='REGULAR_PATCH_'+str(ci).zfill(2),selection_order=ci,condition_source=cond['condition_path'],condition_source_sha256=cond['condition_sha256'],
                     source_CC_CF_FF=cond['existing_source_quality'],source_quality='INHERITED_SOURCE_PAIR_AUDIT',
                     source_interface_edges=None,patch_shape=cond['input_review']['shape_description'],
                     condition_training_seen=ci<3,condition_seen_in_200step_focus=uid.startswith('85b'),
                     official_pretrain_seen='UNKNOWN',full_noise_reused=True,
                     noise_provenance='EXISTING_FULL_N_GAUSSIAN_BYTE_EXACT; zero first K_FULL; remaining absolute slots unchanged; CREF uses same array in local frame')
            if cache:
                if not C.same(c,C.old().read_input(original)['C']) or not C.same(z,C.old().read_input(original)['z']):raise ValueError('Cache changed C/z')
                q['cache']={}
                for arm in LABELS:
                    q['cache'][arm]={}
                    for stage in ['raw','post']:
                        oldrow=next(x for x in C.read(BASE/(stage+'_index.json'))['rows'] if x['label']==arm and x['request_id']==original['request_id'])
                        p=C.resolve(oldrow[stage+'_path']).parent/'receipt.json'
                        q['cache'][arm].update({stage+'_path':str(p),stage+'_sha256':C.sha(p)})
            else:
                rid='REGULARPATCH__'+q['panel']+'__'+cid+'__n'+str(q['candidate'])
                p=C.OUT/'inputs/requests'/(rid+'.npz');C.save_npz(p,dict(C=c,z=z,N=n,K=k))
                q.update(request_id=rid,base_request_id=rid,input_path=str(p),input_sha256=C.sha(p),C_sha256=C.array_sha(c),Gaussian_sha256=C.array_sha(z),free_Gaussian_sha256=C.array_sha(z[:,k:]))
                q.pop('cache',None)
            C.old().read_input(q); reqs.append(q)
        if cache:
            query=copy.deepcopy(next(x for x in old['queries'] if x['condition_id']==cid))
            if C.sha(query['path'])!=query['sha256']:raise ValueError('Query cache changed')
        else:
            a,seed=C.old().query_bank(c,cid);p=C.OUT/'queries'/(cid+'.npz');C.save_npz(p,a)
            query=dict(condition_id=cid,base_condition_id=uid+'__SEAT_FULL',parent_id=uid,panel=selected[0]['panel'],split=selected[0]['split'],path=str(p),sha256=C.sha(p),
                       array_sha256={key:C.array_sha(v) for key,v in a.items()},identity=dict(condition_id=cid,C_seed=seed,C_sha256=C.array_sha(c),L0=C.L),query_count=2048,weight_denominator=float(a['C_weights'].sum()))
        query['transform_audit']=dict(transform='IDENTITY_FULL_SOURCE_PATCH',changed_geometry=False,new_FULL_area_query=not cache,query_reused_byte_exact=cache)
        queries.append(query)
    manifest=dict(version=C.VERSION,requests=reqs,conditions=audit['conditions'],queries=queries)
    C.write(C.OUT/'inputs/manifest.json',manifest)
    files=[C.CODE/'common.py',C.CODE/'prepare.py',C.CODE/'inference.py',C.CODE/'data_audit.py',C.ROOT/'r2/pilots/c_reference_frame_v1/model.py',C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/gpu.py',*list((C.ROOT/'r2/src/meshflow_control').glob('*.py'))]
    hashes={str(p):C.sha(p) for p in files}
    r=dict(version=C.VERSION,status='FROZEN',utc=C.utc(),labels=LABELS,requests=reqs,conditions=audit['conditions'],queries=queries,models=models,d0_checkpoint=old['d0_checkpoint'],code_hashes=hashes,
        input_audit_sha256=C.sha(C.OUT/'preparation/condition_audit.json'),manifest_sha256=C.sha(C.OUT/'inputs/manifest.json'),
        resources=dict(allocator_cap_MiB=5120,aggregate_experiment_budget_MiB=6144,generator_incremental_estimate_MiB=2600,D0_incremental_estimate_MiB=3000,headroom_MiB=2048,CPU_threads_per_worker=2),
        budget=dict(training_updates=0,new_rollouts=32,new_generator_forwards=1600,new_D0_forwards=32,new_Gaussian_draws=0,logical_RAW=48,logical_POST=48,cached_RAW=16,cached_POST=16,new_stage_scores=64,cached_stage_scores=32),
        scientific_scope=dict(primary='SMALL_C200 versus CREF_C200 on same six complete connected source patches and fixed n0..3',
        exploratory='Four HALF-to-FULL changes jointly alter K, area, boundary, free set and CREF center; not isolated K ablation. Two unchanged FULL are controls, not new-input benefit.',
        shape_exception='3f7417 full surface has rear notches; report separately from rectangle-like inputs.',
        outputs='All RAW before new D0; one official D0 + GEO_C + source winding/free orientation; no RBF.',
        replay='No silent replay. Completed receipts resume; incomplete cell blocks pending recorded engineering diagnosis.',
        review='Actual nonblind AI_PROXY whole-shape and C adoption separate; uncertain allowed; not unique source-GT style grading.'))
    C.write(target,r);C.event('REGISTRATION_FROZEN',sha256=C.sha(target),budget=r['budget']);return r

if __name__=='__main__':
    r=prepare();print({'status':r['status'],'requests':len(r['requests']),'budget':r['budget']},flush=True)