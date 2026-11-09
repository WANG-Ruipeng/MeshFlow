"""Independent bounded CPU audit; never imports a model or executes a forward.

Use --stage preflight while training. Use --stage final after evaluation, with
--checkpoints for mmap CPU parameter/AdamW inspection and large-file hashing.
Only independent_audit is writable; incomplete stages are explicit PENDING.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '2'
import argparse
import collections
import csv
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'experiments/r2_anchor_aware_denoiser_v1'
OUT = RUN / 'independent_audit'
OLD = ROOT / 'experiments/r2_postprocess_avmr_pmr_v1'
ARMS = ('D_CONT_NO_ROLE', 'D_ANCHOR')
LEGAL = {'shared_edge_contact', 'shared_vertex_contact', 'coplanar_boundary_or_tolerance_contact',
         'noncoplanar_edge_or_tangent_contact', 'point_or_tangent_contact'}
ILLEGAL = {'proper_transverse_penetration', 'coplanar_positive_area_overlap'}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def csvread(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = Path(p).absolute()
    for q in (p, *p.parents):
        if q.is_symlink():
            raise ValueError('Symlink audit output rejected')
    p = p.resolve()
    if p == OUT.resolve() or not p.is_relative_to(OUT.resolve()):
        raise ValueError('Audit write escapes owned directory')
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    if tmp.is_symlink():
        raise ValueError('Symlink temporary target rejected')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def close(a, b, label='', atol=1e-11):
    if a is None or a == '':
        assert b is None or b == '', (label, a, b)
    else:
        assert b is not None and b != '' and math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=atol), (label,a,b)


def logs(p):
    if not Path(p).exists():
        return []
    data = Path(p).read_bytes()
    # A running writer may leave its latest line partial. Never parse a prefix
    # as a complete event or count that pending line as a successful update.
    return [json.loads(line) for line in data.split(b'\n')[:-1] if line.strip()]


def pending(reason):
    return dict(status='PENDING', reason=reason)


def protected_audit(large):
    reg = read(RUN/'registration.json')
    checked, skipped = [], []
    for p, h in reg['protected'].items():
        path = Path(p)
        if path.stat().st_size > 256*(1 << 20) and not large:
            skipped.append(dict(path=p, bytes=path.stat().st_size, expected_sha256=h))
            continue
        assert sha(path) == h, 'Protected file changed: ' + p
        checked.append(p)
    for p, h in reg['training_source_hashes'].items():
        assert sha(p) == h, 'Frozen training source changed: ' + p
    # Import only the filesystem helper, not any model/runtime module.
    sys.path.insert(0, str(ROOT))
    from r2.pilots.anchor_aware_denoiser_v1 import common
    denied=[]
    for p in (OLD/'assets/chair_denoiser_0030000.pt', OLD/'REPORT.md',
              Path(reg['records'][0]['raw_path']), Path(reg['records'][0]['condition_source']), RUN/'../outside.json'):
        try:
            common.checked(p)
        except ValueError:
            denied.append(str(p))
        else:
            raise AssertionError('Old write target not rejected')
    return dict(status='PASS' if not skipped else 'PASS_SMALL_FILES_LARGE_HASH_PENDING', checked_files=len(checked),
        skipped_large=skipped, frozen_training_source_files=len(reg['training_source_hashes']),
        actual_old_write_guard_negatives=denied, old_files_mutated=0)


def preflight_audit():
    receipt = RUN/'preflight/completion.json'
    if not receipt.exists():
        return pending('Preflight incomplete')
    c = read(receipt)
    p = read(RUN/'gpu_phases/preflight.json')
    assert c['status']=='PASS' and p['status']=='COMPLETE'
    assert (c['forwards'],c['backwards'],c['discarded_updates'])==(32,16,2)
    assert (p['forward_attempts'],p['backward_attempts'],p['optimizer_updates'])==(32,16,2)
    call = [r for r in logs(RUN/'gpu_calls.jsonl') if r.get('phase')=='preflight']
    f=[r for r in call if r['event']=='FORWARD_STARTED'];b=[r for r in call if r['event']=='BACKWARD_STARTED']
    assert [r['call'] for r in f]==list(range(1,33)) and [r['call'] for r in b]==list(range(1,17))
    assert sum(r['official_reference'] for r in f)==8
    observed={}
    plan=read(RUN/'stream/plan.json')['rows']
    chosen=[min(plan,key=lambda r:r['K']),max(plan,key=lambda r:r['N'])]
    chosen += [next(r for r in plan if r['region_type']==part) for part in ('seat','back')]
    for arm in ARMS:
        for case,row in enumerate(chosen):
            p=RUN/'preflight'/f'{arm}_case{case}_export.npz'
            proof=next(v for v in c['zero_equivalence'] if v['arm']==arm and v['export_sha256']==sha(p))
            assert proof['zero_role_bitwise'] and proof['C_projection_export_byte_exact']
            z=arrays(p);batch=arrays(RUN/'stream/batches'/f"{row['step']:06d}.npz")
            lo,hi=map(int,batch['offsets'][row['slot']:row['slot']+2])
            original=batch['target_model'][lo:hi];known=batch['known'][lo:hi]
            assert same(z['known'],known) and same(z['C'],original[known])
            assert same(z['output'][known],original[known]) and same(z['output'][~known],z['proposal'][~known])
            assert z['output'].dtype==np.float32 and np.isfinite(z['proposal']).all()
            observed[(arm,case)]=z
        stats=read(RUN/'preflight'/f'{arm}.json')
        assert stats['discarded_update'] and stats['gradient']['parameters_without_grad']==[]
        assert stats['gradient']['parameters_with_grad']==(297 if arm=='D_ANCHOR' else 296)
        assert all(math.isfinite(x) and x>0 for x in stats['known_hidden_gradient_l2'])
        assert stats['gradient']['selected_gradient_l2']['base.post_quant_linear.weight']>0
        if arm=='D_ANCHOR':
            assert stats['gradient']['selected_gradient_l2']['e_known']>0 and stats['role_l2']>0
        else:
            assert stats['role_l2']==0
    a,b=[read(RUN/'preflight'/f'{arm}.json') for arm in ARMS]
    assert a['rng_before']==b['rng_before'] and a['rng_after']==b['rng_after'] and a['loss']==b['loss']
    for case in range(4):
        assert same(observed[(ARMS[0],case)]['proposal'],observed[(ARMS[1],case)]['proposal'])
    return dict(status='PASS', official_forwards=32, official_backwards=16, discarded_updates=2,
        distinct_preflight_conditions=len({r['condition_id'] for r in chosen}), coverage_checks_per_arm=4,
        includes_small_K=any(r['K']==min(q['K'] for q in plan) for r in chosen),
        includes_max_N=any(r['N']==max(q['N'] for q in plan) for r in chosen),
        region_types=sorted({r['region_type'] for r in chosen}), cross_arm_saved_proposals_byte_equal=True,
        C_exports_independently_reconstructed=True, free_loss_reaches_known_hidden_features=True,
        official_reference_scope='Runtime torch.equal assertions and frozen source/ledger verified; reference residual arrays were not separately saved. No independent model rerun claimed.',
        replayed_model_calls=0)


def stream_audit():
    cachepath=OUT/'stream_complete.json'
    reg=read(RUN/'registration.json')
    if cachepath.exists():
        old=read(cachepath)
        if old['manifest_sha256']==sha(RUN/'stream/manifest.json') and old['plan_sha256']==sha(RUN/'stream/plan.json'):
            return dict(old, audit_execution='REUSE_PREVIOUS_FULL_CPU_STREAM_CHECK')
    manifest=read(RUN/'stream/manifest.json');plan=read(RUN/'stream/plan.json');white=read(RUN/'data_audit/training_whitelist.json')
    assert sha(RUN/'stream/manifest.json')==reg['stream_sha256']
    assert sha(RUN/'data_audit/training_whitelist.json')==reg['whitelist_sha256']
    assert len(manifest['batches'])==1000 and len(plan['rows'])==8000
    sources={r['condition_id']:r for r in white['rows']};source_cache={}
    denominators=[];exposure=collections.Counter();allconditions=collections.Counter()
    for step,b in enumerate(manifest['batches'],1):
        assert b['step']==step and sha(b['path'])==b['sha256']
        rows=plan['rows'][(step-1)*8:step*8]
        assert b['rows']==rows and len({r['uid'] for r in rows})==8 and sum(r['clean'] for r in rows)==4
        a=arrays(b['path']);den=9*int(np.count_nonzero(~a['known']))
        assert den==int(a['free_coordinate_denominator'])==b['free_coordinate_denominator']
        assert a['offsets'].tolist()==b['offsets']
        assert hashlib.sha256(a['x_d'].tobytes()).hexdigest()==b['input_array_sha256']
        assert hashlib.sha256(a['target_d'].tobytes()).hexdigest()==b['target_array_sha256']
        assert hashlib.sha256(a['known'].tobytes()).hexdigest()==b['known_array_sha256']
        assert hashlib.sha256(a['epsilon'].tobytes()).hexdigest()==b['epsilon_array_sha256']
        for j,r in enumerate(rows):
            lo,hi=map(int,a['offsets'][j:j+2]);n,k=r['N'],r['K'];cid=r['condition_id']
            assert hi-lo==n and cid in sources and r['uid'] in white['parent_ids']
            if cid not in source_cache:
                src=sources[cid];assert sha(src['npz'])==src['sha256'];z=arrays(src['npz'])
                source_cache[cid]=np.concatenate((z['C'],z['Y']))
            target=source_cache[cid].copy()
            target[...,0]*=np.float32(r['alpha'])
            assert .9-1e-7<=r['alpha']<=1.1+1e-7 and (r['alpha']==1)==r['clean']
            face=a['face_permutations'][lo:hi];corner=a['corner_permutations'][lo:hi]
            assert np.array_equal(np.sort(face),np.arange(n))
            assert np.array_equal(np.sort(corner,axis=1),np.tile(np.arange(3),(n,1)))
            # Independent scatter back to original C-then-Y order.
            def scatter(value):
                out=np.empty_like(value)
                out[face[:,None],corner]=value
                return out
            tm=scatter(a['target_model'][lo:hi]);td=scatter(a['target_d'][lo:hi]);xd=scatter(a['x_d'][lo:hi]);eps=scatter(a['epsilon'][lo:hi])
            assert same(tm,target) and same(td,target*np.float32(.95*.3762))
            expected=td.copy();expected[k:]+=np.float32(.02)*eps[k:]
            assert same(expected,xd) and same(xd[:k],td[:k]) and not np.any(eps[:k])
            assert np.array_equal(a['known'][lo:hi],face<k)
            assert np.isfinite(xd).all() and np.isfinite(eps).all()
            exposure[r['uid']]+=1;allconditions[cid]+=1
        denominators.append(den)
    assert dict(exposure)==plan['actual_parent_exposure'] and dict(allconditions)==plan['actual_condition_exposure']
    result=dict(status='PASS',batches=1000,samples=8000,parents=len(exposure),conditions=len(allconditions),
        independent_permutation_scatter=True,original_CY_and_width_byte_match=True,known_clean=True,
        noise_arrays_reused_no_new_draws=True,scalar_free_MSE_denominator_verified_for_every_batch=True,
        denominator_min=min(denominators),denominator_max=max(denominators),
        manifest_sha256=sha(RUN/'stream/manifest.json'),plan_sha256=sha(RUN/'stream/plan.json'),
        audit_execution='FULL_CPU_BYTE_CHECK',no_model_calls=True)
    write(cachepath,result);return result


def train_audit():
    reg=read(RUN/'registration.json');manifest=read(RUN/'stream/manifest.json');allcalls=logs(RUN/'gpu_calls.jsonl')
    outcomes={};perarm={};initials={}
    for arm in ARMS:
        initp=RUN/'training'/arm/'initialization.json';comp=RUN/'training'/arm/'completion.json'
        if not initp.exists():outcomes[arm]=pending('Arm not initialized');continue
        init=read(initp);initials[arm]=init
        assert init['optimizer_state_entries']==0 and init['identity']['fresh_AdamW']
        assert init['identity']['stream_sha256']==reg['stream_sha256']
        assert init['identity']['registration_sha256']==sha(RUN/'registration.json')
        rows=logs(RUN/'training'/arm/'ledger.jsonl')
        starts=[r for r in rows if r['event']=='UPDATE_STARTED'];done=[r for r in rows if r['event']=='UPDATE_COMPLETE']
        assert [r['update'] for r in starts]==list(range(1,len(starts)+1))
        assert [r['update'] for r in done]==list(range(1,len(done)+1))
        assert len(done)<=1000 and len(starts) in (len(done),len(done)+1)
        for r in done:
            b=manifest['batches'][r['update']-1]
            assert r['batch_sha256']==b['sha256'] and r['free_denominator']==b['free_coordinate_denominator']
            assert math.isfinite(r['loss']) and math.isfinite(r['gradient_norm_before_clip']) and r['loss']>=0
            if arm=='D_CONT_NO_ROLE':assert r['role_l2']==0
        perarm[arm]=done
        fc=[r for r in allcalls if r['event']=='FORWARD_STARTED' and r['phase'].startswith('train_'+arm)]
        bc=[r for r in allcalls if r['event']=='BACKWARD_STARTED' and r['phase'].startswith('train_'+arm)]
        if not comp.exists():
            outcomes[arm]=dict(status='PENDING',completed_updates=len(done),observed_forward_starts=len(fc),observed_backward_starts=len(bc));continue
        finish=read(comp)
        if finish['status']!='COMPLETE':
            outcomes[arm]=dict(status=finish['status'],completed_updates=len(done));continue
        assert len(done)==len(starts)==1000 and len(fc)==len(bc)==8000
        assert finish['forwards']==finish['backwards']==8000 and finish['optimizer_updates']==finish['completed_updates']==1000
        phases=[read(p) for p in (RUN/'gpu_phases').glob('train_'+arm+'*.json')]
        assert sum(p['forward_attempts'] for p in phases)==sum(p['backward_attempts'] for p in phases)==8000
        assert sum(p['optimizer_updates'] for p in phases)==1000
        role=done[-1]['role']
        assert role is not None
        if arm=='D_ANCHOR':assert done[-1]['role_l2']>0 and role['requested_add_l2']>0
        else:assert role['effective_add_l2']==0
        outcomes[arm]=dict(status='PASS',updates=1000,forwards=8000,backwards=8000,fresh_optimizer=True,
            initial_full_state_hash=init['identity']['base_initial_state_hash'],final_role_l2=done[-1]['role_l2'],
            final_effective_role_add_l2=role['effective_add_l2'],final_changed_hidden_scalars=role['changed_hidden_scalars'],
            effective_role_add_nonzero=role['effective_add_l2']>0)
    pairing=pending('Both arms have no common completed updates')
    if all(arm in initials for arm in ARMS):
        a,b=[initials[arm] for arm in ARMS]
        assert a['identity']['base_initial_state_hash']==b['identity']['base_initial_state_hash'] and a['rng']==b['rng']
    if all(arm in perarm for arm in ARMS):
        n=min(map(lambda arm:len(perarm[arm]),ARMS));count=0
        for a,b in zip(perarm[ARMS[0]],perarm[ARMS[1]]):
            for key in ('update','batch_sha256','free_denominator','rng_before','rng_after'):
                assert a[key]==b[key],(key,a['update'])
            count+=1
        pairing=dict(status='PASS' if count==1000 else 'PENDING',matched_completed_updates=count,
            shared_batch_identity=True,paired_torch_CPU_and_CUDA_rng_before_after=True)
    return dict(status='PASS' if all(v['status']=='PASS' for v in outcomes.values()) and pairing['status']=='PASS' else 'PENDING',arms=outcomes,pairing=pairing)


def checkpoint_audit(enabled):
    if not enabled:return pending('CPU mmap checkpoint inspection deferred until final --checkpoints')
    if any(not (RUN/'training'/arm/'completion.json').exists() or read(RUN/'training'/arm/'completion.json')['status']!='COMPLETE' for arm in ARMS):
        return pending('Both formal arms must complete before checkpoint inspection')
    avail=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    if avail<16*(1<<30):return pending('Require16GiB available host RAM for bounded CPU mmap checkpoint inspection')
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    result=[]
    for arm in ARMS:
        full=RUN/'checkpoints'/arm/'step1000.pt';export=full.with_name('inference.pt')
        fr=read(full.with_suffix('.json'));er=read(export.with_suffix('.json'))
        assert fr['step']==er['step']==1000 and fr['identity']==er['identity']
        assert sha(full)==fr['sha256'] and sha(export)==er['sha256']
        a=torch.load(full,map_location='cpu',weights_only=True,mmap=True)
        b=torch.load(export,map_location='cpu',weights_only=True,mmap=True)
        assert a['step']==b['step']==1000 and a['arm']==b['arm']==arm and a['identity']==b['identity']
        assert list(a['model'])==list(b['model']) and len(a['model'])==297
        finite_scalars=0
        basehash=hashlib.sha256()
        for name,value in a['model'].items():
            assert value.dtype==torch.float32 and value.device.type=='cpu'
            one=value.reshape(-1).numpy();two=b['model'][name].reshape(-1).numpy()
            for lo in range(0,len(one),1<<20):
                v=one[lo:lo+(1<<20)];w=two[lo:lo+(1<<20)]
                assert np.isfinite(v).all() and v.tobytes()==w.tobytes()
                finite_scalars+=len(v)
            if name.startswith('base.'):
                basehash.update(name[5:].encode());basehash.update(value.numpy().tobytes())
        assert basehash.hexdigest()!=a['identity']['base_initial_state_hash']
        role=a['model']['e_known'].numpy();role_norm=float(np.linalg.norm(role.astype(float)))
        assert role.shape==(1024,) and (role_norm>0 if arm=='D_ANCHOR' else role_norm==0)
        optimizer=a['optimizer'];expected=297 if arm=='D_ANCHOR' else 296
        assert len(optimizer['state'])==expected and len(optimizer['param_groups'])==1
        group=optimizer['param_groups'][0]
        assert len(group['params'])==expected and group['lr']==1e-5 and tuple(group['betas'])==(.9,.95) and group['weight_decay']==0
        tensor_entries=0
        for p,state in optimizer['state'].items():
            assert float(state['step'])==1000 and set(state)=={'step','exp_avg','exp_avg_sq'}
            for key in ('exp_avg','exp_avg_sq'):
                data=state[key].reshape(-1).numpy()
                for lo in range(0,len(data),1<<20):assert np.isfinite(data[lo:lo+(1<<20)]).all()
                if key=='exp_avg_sq':assert bool((data>=0).all())
                tensor_entries+=1
        result.append(dict(arm=arm,status='PASS',model_state_keys=297,finite_FP32_model_scalars=finite_scalars,
            full_equals_inference_bitwise=True,optimizer_parameter_states=expected,all_optimizer_steps=1000,
            finite_optimizer_moment_tensors=tensor_entries,final_role_l2=role_norm,base_changed_from_initial=True,
            full_checkpoint_sha256=fr['sha256'],inference_checkpoint_sha256=er['sha256']))
        del a,b;gc.collect()
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',arms=result,CPU_mmap=True,host_available_before=avail,model_instantiations=0,model_forwards=0,CUDA_initialized=False)


def inferred_audit():
    reg=read(RUN/'registration.json');real=[];synthetic=[]
    for arm in ARMS:
        for r in reg['records']:
            dest=RUN/'inference'/arm/r['label']/r['request_id'];receipt=dest/'receipt.json'
            if not receipt.exists():continue
            p=read(receipt);path=dest/'denoised.npz'
            assert p['status']=='PASS' and p['raw_sha256']==r['raw_sha256'] and p['output_sha256']==sha(path)
            raw=arrays(r['raw_path'])['raw'].reshape(-1,3,3);z=arrays(path);K=int(r['K'])
            assert z['output'].shape==raw.shape and z['output'].dtype==np.float32 and np.isfinite(z['output']).all()
            assert same(z['C'],raw[:K]) and same(z['output'][:K],raw[:K]) and same(z['output'][K:],z['proposal'][K:])
            assert same(z['displacement_from_RAW'],z['output'].astype(float)-raw.astype(float))
            assert p['no_rng_consumed'] and p['C_original_bits_restored']
            real.append(dict(arm=arm,label=r['label'],request_id=r['request_id']))
    manifest=read(RUN/'stream/synthetic/manifest.json')
    for arm in ('D0_OFFICIAL',)+ARMS:
        for r in manifest['rows']:
            dest=RUN/'synthetic_results'/arm/r['synthetic_id'];receipt=dest/'receipt.json'
            if not receipt.exists():continue
            p=read(receipt);z=arrays(dest/'prediction.npz');src=arrays(r['path']);K=int(r['K']);m=p['metrics']
            assert p['status']=='PASS' and sha(dest/'prediction.npz')==p['output_sha256']
            assert same(z['C'],src['C']) and same(z['output'][:K],src['C'])
            delta=z['proposal_denoiser'].astype(float)-src['target_d'].astype(float)
            d=delta[K:];close(m['free_sse'],np.square(d).sum());close(m['free_mse'],np.square(d).mean())
            assert same(d,z['free_error_denoiser'])
            ix=src['supported_Y_local_corner_indices'];sub=d[ix[:,0],ix[:,1]]
            assert same(sub,z['supported_interface_error_denoiser'])
            if len(sub):close(m['supported_interface_mse'],np.square(sub).mean())
            else:assert m['supported_interface_mse'] is None
            if r['clean']:
                motion=z['proposal_denoiser'][K:].astype(float)-src['x_d'][K:].astype(float)
                close(m['clean_unnecessary_movement_mse'],np.square(motion).mean())
            assert p['no_rng_consumed'] and p['C_original_bits_restored']
            synthetic.append(dict(arm=arm,synthetic_id=r['synthetic_id']))
    events=logs(RUN/'inference_ledger.jsonl')
    starts=[tuple(e['key']) for e in events if e['event']=='EVAL_FORWARD_STARTED']
    dones=[tuple(e['key']) for e in events if e['event']=='EVAL_FORWARD_COMPLETE']
    assert len(starts)==len(set(starts)) and len(dones)==len(set(dones)) and set(dones)<=set(starts)
    assert len(starts)<=150
    return dict(status='PASS' if len(real)==96 and len(synthetic)==54 and len(dones)==150 else 'PENDING',
        real_completed=len(real),synthetic_completed=len(synthetic),ledger_started=len(starts),ledger_complete=len(dones),
        real_original_C_and_free_proposal_bitwise=True,synthetic_free_MSE_and_interface_MSE_independent=True,
        no_unregistered_noise_draws_runtime_evidence=True,independent_model_calls=0)


def independent_graph(v,f,K,pairs):
    tri=v[f].astype(float);areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
    groups=collections.Counter(tuple(sorted(map(int,x))) for x in f)
    dup=np.array([groups[tuple(sorted(map(int,x)))]>1 for x in f])
    edges=collections.defaultdict(list)
    for i,face in enumerate(f):
        for j in range(3):edges[tuple(sorted((int(face[j]),int(face[(j+1)%3]))))].append((i,int(face[j]),int(face[(j+1)%3])))
    cp={tuple(x['pair']):x['classification'] for x in pairs['CF']['contacts']}
    fp={tuple(sorted(x['pair'])):x['classification'] for x in pairs['FF']['contacts']}
    parent=list(range(len(f)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    shared=valid=opposite=nm=0;length=0.
    for (u,w),items in edges.items():
        cf=any(i<K for i,_,_ in items) and any(i>=K for i,_,_ in items)
        shared+=int(cf);nm+=int(cf and len(items)>2)
        if len(items)!=2:continue
        (i,a,b),(j,c,d)=items
        if i<K and j<K:continue
        if i==j or u==w or areas[i]<=0 or areas[j]<=0 or dup[i] or dup[j]:continue
        if cf:
            ci,fi=(i,j) if i<K else (j,i);label=cp.get((ci,fi-K),'UNKNOWN')
        else:label=fp.get(tuple(sorted((i-K,j-K))),'UNKNOWN')
        if label not in LEGAL:continue
        parent[find(i)]=find(j)
        if cf:
            valid+=1;opposite+=int(a==d and b==c);length+=float(np.linalg.norm(v[u].astype(float)-v[w]))
    roots={find(i) for i in range(K)}
    reached=[i for i in range(K,len(f)) if find(i) in roots]
    return dict(C_F_shared_edge_count=shared,C_F_valid_edge_count=valid,C_F_has_valid_edge=int(valid>0),
        C_F_valid_opposite_winding_edge_count=opposite,C_F_nonmanifold_edge_count=nm,
        C_F_valid_edge_total_length=length,reached_free_faces=len(reached),reached_free_area=float(areas[reached].sum()),
        free_area=float(areas[K:].sum()),reached_free_area_fraction=float(areas[reached].sum()/areas[K:].sum()) if areas[K:].sum() else None)


def evaluation_source_audit():
    import ast
    E=RUN/'evaluation';registration=read(E/'registration.json')
    code=ROOT/'r2/pilots/anchor_aware_denoiser_v1/evaluation.py'
    if sha(code)==registration['evaluation_source_sha256']:
        return dict(status='PASS',source_status='ORIGINAL_REGISTERED_SOURCE')
    correction=read(E/'compatibility/evaluation_source_revision.json')
    assert correction['original_source_sha256']==registration['evaluation_source_sha256']
    assert correction['revised_source_sha256']==sha(code)
    assert correction['core_registration_sha256']==sha(E/'registration.json')==sha(E/'compatibility/registration_initial.json')
    original=Path(correction['original_source_snapshot']);revised=Path(correction['revised_source_snapshot'])
    assert sha(original)==correction['original_source_sha256'] and sha(revised)==sha(code)
    before=ast.parse(original.read_text());after=ast.parse(revised.read_text())
    identity_revision=correction.get('metadata_roundtrip_correction')
    if identity_revision:
        previous=read(correction['previous_revision_manifest'])
        assert sha(correction['previous_revision_manifest'])==correction['previous_revision_manifest_sha256']
        assert previous['revised_source_sha256']==identity_revision['previous_source_sha256']
        assert sha(identity_revision['previous_source_snapshot'])==identity_revision['previous_source_sha256']
        assert identity_revision['revised_source_sha256']==sha(code)
        assert identity_revision['integer_fields']==['candidate','N','K']
        assert identity_revision['quality_started_before']==identity_revision['quality_completed_before']==identity_revision['unfinished_quality_jobs']==identity_revision['new_model_forwards']==0
        assert identity_revision['GEO_C_started_before']==identity_revision['GEO_C_complete_before']==96
        failure=read(identity_revision['failure_provenance'])
        assert sha(identity_revision['failure_provenance'])==identity_revision['failure_provenance_sha256']
        for path,digest in failure['cached_files_before'].items():assert sha(path)==digest
        helper=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='canonical_identity')
        expected=ast.parse("def canonical_identity(identity):\n    return {k: int(v) if k in ('candidate', 'N', 'K') else v for k, v in identity.items()}").body[0]
        helper.body=helper.body[1:] if isinstance(helper.body[0],ast.Expr) and isinstance(helper.body[0].value,ast.Constant) and isinstance(helper.body[0].value.value,str) else helper.body
        assert ast.dump(helper,include_attributes=False)==ast.dump(expected,include_attributes=False)
        after.body.remove(helper)
        class UndoIdentity(ast.NodeTransformer):
            count=0
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name) and node.func.id=='canonical_identity':
                    assert len(node.args)==1 and not node.keywords
                    self.count+=1
                    return node.args[0]
                return node
        undo=UndoIdentity();after=undo.visit(after);assert undo.count==2
    for tree in (before,after):
        tree.body=[node for node in tree.body if not(isinstance(node,ast.FunctionDef) and node.name=='prepare')]
    renamed=0
    for node in ast.walk(after):
        if isinstance(node,ast.keyword) and node.arg=='denoiser_checkpoint_sha256':
            node.arg='checkpoint_sha256';renamed+=1
    assert renamed==1 and ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
    assert correction['formal_GEO_C_jobs_started']==0 and correction['new_model_forwards']==0 and correction['new_quality_scoring']==0
    return dict(status='PASS',source_status='PROVENANCE_VERIFIED_NONSCIENTIFIC_CORRECTION',
        old_source_sha256=sha(original),new_source_sha256=sha(code),core_registration_sha256=sha(E/'registration.json'),
        scientific_AST_unchanged_after_single_metadata_keyword_rename=True,
        changed_metadata='generator checkpoint preserved; separate denoiser checkpoint; cached identity candidate/N/K normalized after CSV roundtrip',
        integer_identity_roundtrip_correction_verified=bool(identity_revision),
        D0_cache_registration_identity_unchanged=True)


def evaluation_audit():
    E=RUN/'evaluation';done=E/'completion.json'
    if not done.exists():return pending('Formal evaluation incomplete')
    from scipy.spatial import cKDTree
    reg=read(E/'registration.json');index=csvread(E/'post_index.csv')
    assert len(index)==144
    table=csvread(E/'per_output.csv');support=csvread(E/'source_supported_reference_per_output.csv')
    assert len(table)==len(support)==2304
    metrics={(r['denoiser'],r['label'],r['request_id'],r['stage'],r['setting'],r['domain']):r for r in table+support}
    topo={(r['denoiser'],r['label'],r['request_id']):r for r in csvread(E/'topology_per_output.csv')}
    qtable={(r['denoiser'],r['label'],r['request_id']):r for r in csvread(E/'quality_per_output.csv')}
    conditions={};vm_cells=topo_cells=quality_cells=0;vertex_distances=0
    for r in index:
        key=(r['denoiser'],r['label'],r['request_id']);K=int(r['K']);post=arrays(r['post_path']);before=arrays(r['denoised_path'])['output']
        diag=read(E/'diagnostics'/r['denoiser']/r['label']/r['request_id']/'diagnostic.json')
        source=arrays(r['condition_source']);C=source['C'];ids=source['C_source_vertex_ids']
        edges=collections.Counter();coords={}
        for face,vs in zip(C,ids):
            for x,p in zip(vs,face):coords[int(x)]=p
            for j in range(3):edges[tuple(sorted((int(vs[j]),int(vs[(j+1)%3]))))]+=1
        bid=sorted({x for e,n in edges.items() if n==1 for x in e});pts=np.array([coords[x] for x in bid],float)
        di=read(RUN/'data_audit/interfaces'/(r['condition_id']+'.json'));supported=set(di['supported_C_source_vertex_ids'])
        for stage,tri,vkey in (('POST',post['output'],'vertices'),('PRE_GEO_C_RESTORED',before,'pre_GEO_vertices')):
            assert same(tri[:K],C)
            ds=cKDTree(tri[K:].reshape(-1,3).astype(float)).query(pts,workers=1)[0] if len(tri)>K else np.full(len(pts),np.inf)
            for x,y in zip(ds,diag[vkey]):
                if np.isfinite(x):close(x,y['distance_model'],atol=1e-14)
                else:assert y['distance_model'] is None
                vertex_distances+=1
            for domain,d in (('ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY',ds),('SOURCE_SUPPORTED_INTERFACE_REFERENCE_ONLY',ds[[x in supported for x in bid]])):
                for setting in reg['settings']:
                    row=metrics[key+(stage,setting['name'],domain)]
                    assert int(row['boundary_vertex_count'])==len(d)
                    matched=int(np.sum(d<setting['threshold_model']))
                    assert int(row['matched_count'])==matched
                    close(row['VMR'],matched/len(d) if len(d) else None)
                    close(row['perfect'],int(matched==len(d)) if len(d) else None)
                    vm_cells+=4
        graph=independent_graph(post['vertices'],post['faces'],K,read(diag['quality']['pair_audit_path']))
        for field,value in graph.items():close(topo[key][field],value);topo_cells+=1
        assert same(post['vertices'][post['faces']],post['output'])
        for field,value in diag['quality']['metrics'].items():
            if isinstance(value,(int,float)) or value is None:close(qtable[key][field],value);quality_cells+=1
            else:assert qtable[key][field]==value
    # Independent arithmetic of every stored mean/median/p95; geometry detector
    # itself is unchanged/frozen, not reimplemented or re-run here.
    agg_cells=0
    for per,summary in (('quality_per_output.csv','quality_summary.csv'),('topology_per_output.csv','topology_summary.csv'),('per_output.csv','AVMR_summary.csv'),('paired_per_output.csv','paired_summary.csv'),('AVMR_paired_per_output.csv','AVMR_paired_summary.csv')):
        original=csvread(E/per)
        for row in csvread(E/summary):
            selector={k:row[k] for k in ('denoiser','label','stage','setting','region_type','parent_id','candidate','comparison') if row.get(k,'')!=''}
            group=[r for r in original if all(r.get(k)==v for k,v in selector.items())]
            assert len(group)==int(row['outputs'])
            for key,value in row.items():
                if not key.endswith('_valid_n'):continue
                field=key[:-8];vals=[float(r[field]) for r in group if r.get(field,'')!='']
                assert len(vals)==int(value)
                for suffix,fun in (('_mean',lambda x:math.fsum(x)/len(x)),('_median',lambda x:float(np.median(x))),('_p95',lambda x:float(np.quantile(x,.95)))):
                    close(row[field+suffix],fun(vals) if vals else None);agg_cells+=1
    paired_cells=0
    for row in csvread(E/'paired_per_output.csv'):
        new,base=row['comparison'].split('_minus_')
        k=(row['label'],row['request_id'])
        for field,value in row.items():
            if not field.endswith('_delta'):continue
            original=field[:-6]
            table_to_use=qtable if original in qtable[(new,)+k] else topo
            a=table_to_use[(new,)+k][original];b=table_to_use[(base,)+k][original]
            expected=float(a)-float(b) if a!='' and b!='' else None
            close(value,expected);paired_cells+=1
    for row in csvread(E/'AVMR_paired_per_output.csv'):
        new,base=row['comparison'].split('_minus_')
        suffix=(row['label'],row['request_id'],row['stage'],row['setting'],'ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY')
        a,b=metrics[(new,)+suffix],metrics[(base,)+suffix]
        for field in ('VMR','perfect'):
            close(row[field+'_delta'],float(a[field])-float(b[field]) if a[field]!='' and b[field]!='' else None);paired_cells+=1
    fate_checks=0
    for row in csvread(E/'FF_fates_per_output.csv'):
        base=read(E/'diagnostics/D0_OFFICIAL'/row['label']/row['request_id']/'diagnostic.json')
        newer=read(E/'diagnostics'/row['denoiser']/row['label']/row['request_id']/'diagnostic.json')
        K=int(row['K'])
        def source_pairs(diagnostic):
            ids=diagnostic['source_face_ids'];pairs=read(diagnostic['quality']['pair_audit_path'])['FF']['contacts']
            return {tuple(sorted((ids[K+pair['pair'][0]],ids[K+pair['pair'][1]]))):pair['classification'] for pair in pairs}
        a,b=source_pairs(base),source_pairs(newer)
        old_bad={pair for pair,label in a.items() if label in ILLEGAL};new_bad={pair for pair,label in b.items() if label in ILLEGAL}
        kept=set(newer['source_face_ids'])
        removed={pair for pair in old_bad if not set(pair)<=kept}
        persistent=old_bad&new_bad
        candidates=old_bad-removed-persistent
        resolved={pair for pair in candidates if b.get(pair,'disjoint')=='disjoint' or b.get(pair) in LEGAL}
        expected={'D0_illegal_pairs':len(old_bad),'new_illegal_pairs':len(new_bad),
            'D0_ILLEGAL_FACE_REMOVED':len(removed),'PERSISTENT_ILLEGAL':len(persistent),
            'BOTH_RETAINED_RESOLVED_NONILLEGAL':len(resolved),'BOTH_RETAINED_UNKNOWN':len(candidates-resolved),
            'NEW_ILLEGAL':len(new_bad-old_bad)}
        for field,value in expected.items():assert int(row[field])==value;(fate_checks:=fate_checks+1)
    ledger=logs(E/'ledger.jsonl')
    job_counts={event:sum(row['event']==event for row in ledger) for event in ('GEO_C_STARTED','GEO_C_COMPLETE','QUALITY_STARTED','QUALITY_COMPLETE')}
    assert all(value==96 for value in job_counts.values()),job_counts
    assert csvread(E/'RAW_quality_reference.csv')==csvread(ROOT/'experiments/r2_user_mix_from_official_v1/evaluation/per_output.csv')
    assert csvread(E/'RAW_AVMR_reference.csv')==csvread(ROOT/'experiments/r2_avmr_pmr_offline_v1/per_output.csv')
    return dict(status='PASS',primary_POST=144,RAW_references_unchanged=48,independent_boundary_distance_values=vertex_distances,
        independent_threshold_scalar_cells=vm_cells,independent_graph_scalar_cells=topo_cells,
        quality_scalar_origin_cells=quality_cells,aggregate_scalar_cells=agg_cells,paired_delta_cells=paired_cells,FF_source_pair_fate_cells=fate_checks,new_CPU_job_ledger_counts=job_counts,
        distance_algorithm='scipy.cKDTree',graph_algorithm='independent union-find over qualifying actual indexed edges',
        geometry_detector_scope='Frozen saved CF/FF classifications and quality scalars verified; no independent detector rewrite or new pair scans.',model_calls=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('preflight','final'),default='preflight');p.add_argument('--checkpoints',action='store_true');args=p.parse_args()
    start=time.perf_counter();report=dict(stage=args.stage,status='RUNNING',source_sha256=sha(__file__),model_forwards=0,new_noise=0)
    try:
        report['protected']=protected_audit(args.stage=='final' and args.checkpoints)
        report['evaluation_code_provenance']=evaluation_source_audit()
        report['preflight']=preflight_audit()
        report['frozen_stream']=stream_audit()
        report['training']=train_audit()
        report['checkpoints']=checkpoint_audit(args.checkpoints)
        report['inference']=inferred_audit() if args.stage=='final' else pending('Not yet dispatched')
        report['evaluation']=evaluation_audit() if args.stage=='final' else pending('Not yet dispatched')
        report['status']='PASS' if args.stage=='final' and all(v.get('status')=='PASS' for v in report.values() if isinstance(v,dict)) else 'PASS_COMPLETED_STAGES_PENDING_REMAINDER'
        report['seconds']=time.perf_counter()-start
        write(OUT/(args.stage+'_REPORT.json'),report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except BaseException as exc:
        report['status']='FAIL';report['error']=repr(exc);report['seconds']=time.perf_counter()-start
        write(OUT/(args.stage+'_FAILURE.json'),report)
        raise


if __name__=='__main__':main()
