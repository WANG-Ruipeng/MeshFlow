"""Frozen contour-readout candidate evaluation; bounded inference and original metrics.

Every invocation owns one registered model/stage. Historical modules are loaded
under private names and only those private instances receive the new OUT binding.
No training, checkpoint selection, noise draws or implicit replay lives here.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, concurrent.futures, functools, json, multiprocessing, sys, time, traceback
from pathlib import Path
import numpy as np
from . import common as C

VERSION='r2_full_contour_readout_evaluation_v1'
TRAIN_LABELS=('CONTOUR_S0','CONTOUR_S1000','CONTOUR_S3000','CONTOUR_S6000')
REFERENCES=('FULL_S0000','FULL_S1000','FULL_S3000','FULL_S6000')
LABELS=TRAIN_LABELS+REFERENCES
STEPS=dict(zip(TRAIN_LABELS,(0,1000,3000,6000)));STEPS.update(zip(REFERENCES,(0,1000,3000,6000)))
EV=C.OUT/'evaluation'
BASE=C.ROOT/'experiments/r2_full_seat_learning_curve_v1'
OLD_INFERENCE=C.ROOT/'r2/pilots/regular_patch_control_v1/inference.py'
OLD_SCORE=C.ROOT/'r2/pilots/regular_patch_control_v1/score.py'
META=('request_id','parent_id','condition_id','panel','candidate','region_type','tier','N','K','ratio',
      'base_request_id','base_condition_id','variant','alpha','pair_id','split','condition_type')


def verify(files):
    for p,h in files.items():
        if C.sha(p)!=h:raise ValueError('Frozen file changed: '+str(p))


def registration():
    r=C.load_registration()
    if r.get('status')!='FROZEN':raise ValueError('Evaluation requires frozen registration')
    qs=r['requests']
    if len(qs)!=24 or len({q['request_id'] for q in qs})!=24 or len({q['condition_id'] for q in qs})!=6:
        raise ValueError('Expected fixed six FULL conditions / 24 requests')
    if tuple(r.get('evaluation_labels',TRAIN_LABELS))!=TRAIN_LABELS:raise ValueError('Learning-curve endpoints changed')
    if len(r['queries'])!=6:raise ValueError('Frozen C-query count changed')
    old=C.read(BASE/'registration.json');reference={q['request_id']:q for q in old['requests']}
    oldqueries={q['condition_id']:q for q in old['queries']}
    for query in r['queries']:
        if query['sha256']!=oldqueries[query['condition_id']]['sha256']:raise ValueError('Frozen C query changed')
    for q in qs:
        prior=reference[q['request_id']]
        for key in ('C_sha256','Gaussian_sha256','input_sha256','condition_source_sha256','N','K','noise_seed','condition_id'):
            if q[key]!=prior[key]:raise ValueError('Frozen FULL request changed: '+key)
        if any(key not in q for key in META):raise ValueError('Missing scoring metadata')
    return r


def check_code():
    r=registration();files=r.get('evaluation_code_hashes',r.get('code_hashes',{}))
    if not any(C.resolve(p).resolve()==Path(__file__).resolve() for p in files):raise ValueError('Evaluator source not frozen')
    verify(files)
    return dict(status='PASS',requests=24,evaluation_endpoints=list(TRAIN_LABELS))


def binding(label):
    if label not in LABELS:raise ValueError('Unregistered label')
    r=registration();path=EV/'bindings'/f'{label}.json'
    if label=='CONTOUR_S0':
        # Preserve the actual historical sampler file; this is inferred zero-exit
        # equivalence after eight normal training forwards, not 24 new rollouts.
        proof=zero_proof()
        b=dict(C.read(BASE/'evaluation/bindings/FULL_S0000.json'))
        b.update(label=label,zero_equivalence_receipt=proof['path'],zero_equivalence_receipt_sha256=proof['sha256'],
            sampling_identity='HISTORICAL_EXP67_FILE; NEW_ZERO_BRANCH_NOT_RESAMPLED')
    elif label in REFERENCES:b=dict(C.read(BASE/'evaluation/bindings'/f'{label}.json'))
    elif path.exists():b=C.read(path)
    elif label in r.get('models',{}):b=dict(r['models'][label])
    else:raise FileNotFoundError('Export fixed endpoint first: '+str(path))
    if label in TRAIN_LABELS:
        step=STEPS[label]
        if int(b.get('local_step',step))!=step or int(b.get('global_step',6000+step))!=6000+step:
            raise ValueError('Checkpoint must be exact registered endpoint')
    if 'path' not in b or 'sha256' not in b:raise ValueError('Incomplete checkpoint binding')
    return b


def zero_proof():
    p=C.OUT/'audits/zero_initialization.json';r=C.read(p)
    if r.get('status')!='PASS' or r.get('normal_training_microbatches')!=8:
        raise ValueError('Step0 reuse awaits first eight normal-training zero checks')
    for key in ('zero_residual_exact','unchanged_hidden_exact','first_batch_FM_matches','old_state_sha256_matches'):
        if r.get(key) is not True:raise ValueError('Step0 equivalence failed: '+key)
    return dict(path=str(p),sha256=C.sha(p))

def requests(label):
    r=registration();result=[]
    for original in r['requests']:
        q=dict(original);q.pop('cache',None)
        if label=='CONTOUR_S0' or label in REFERENCES:
            historical='FULL_S0000' if label=='CONTOUR_S0' else label
            item=dict(historical_label=historical)
            for stage in ('raw','post'):
                p=BASE/'evaluation'/stage/historical/q['request_id']/'receipt.json'
                item[stage+'_path']=str(p);item[stage+'_sha256']=C.sha(p)
            q.update(action='CACHE',cache={label:item})
        else:q['action']='NEW'
        result.append(q)
    return result

def arm_registration(label):
    r=dict(registration());r['requests']=requests(label);r['labels']=[label]
    return r


@functools.lru_cache(None)
def topology_inputs():
    from .topology import TopologyInputs
    # Verify immutable C-only dependencies once per fresh owned process; the
    # manifest digest and current request source remain checked per cell.
    return TopologyInputs()


@functools.lru_cache(None)
def inference(label):
    # This module object is new, not sys.modules' historical public experiment.
    I=C.module('r2.pilots.regular_patch_control_v1._contour_readout_'+label,OLD_INFERENCE)
    I.C=C;I.EV=EV;I.LABELS=(label,)
    I.registration=lambda:arm_registration(label)
    I.binding=binding;I.check_code=check_code
    I.all_raw=lambda r,arm=None:[I.read_raw(q,label) for q in requests(label)]
    inherited_contract=I._raw_contract
    def raw_contract(q,rec,arm):
        facts=inherited_contract(q,rec,arm)
        if arm in TRAIN_LABELS[1:]:
            verify({rec['topology_path']:rec['topology_sha256'],
                rec['model_input_contract_path']:rec['model_input_contract_sha256']})
            mask=C.arrays(rec['topology_path'])['boundary_edge_mask']
            expected=topology_inputs().for_request(q['request_id'])
            if not C.same(mask,expected):raise ValueError('Consumed C topology differs from actual FULL input')
            contract=C.read(rec['model_input_contract_path'])
            if contract['topology_manifest_sha256']!=C.sha(C.OUT/'inputs/topology_manifest.json'):
                raise ValueError('Consumed topology manifest differs')
            if contract['effective_model_inputs']!=['C','N','Gaussian','time','C_source_connectivity_boundary_mask']:
                raise ValueError('Candidate input inventory is incomplete')
            facts['actual_FULL_C_topology']=True
        return facts
    I._raw_contract=raw_contract
    return I


def cache(label):
    check_code();I=inference(label);count=0
    for q in requests(label):
        if q['action']!='CACHE':continue
        item=q['cache'][label];verify({item[s+'_path']:item[s+'_sha256'] for s in ('raw','post')})
        raw=C.read(item['raw_path']);post=C.read(item['post_path'])
        for key in ('input_sha256','C_sha256','Gaussian_sha256','condition_source_sha256','N','K','noise_seed'):
            if raw[key]!=q[key]:raise ValueError('Cache input mismatch: '+key)
        # For step0 the binding deliberately points to the same historical EXP67
        # inference file. No permissive same-name or approximate state matching.
        if raw['checkpoint_sha256']!=binding(label)['sha256']:raise ValueError('Cached model file differs')
        if raw['label']!=post['label']:raise ValueError('Cache RAW/POST model identity differs')
        expected=item['historical_label']
        if raw['label']!=expected:raise ValueError('Wrong historical model label')
        I._raw_contract(q,raw,label);I._post_contract(q,post,raw,label)
        for stage,old in (('raw',raw),('post',post)):
            value=dict(old);value.update(q);value.update(label=label,status='COMPLETE',new_rollouts=0,new_sampling_F=0,
                historical_label=old['label'],historical_request_id=old['request_id'],historical_receipt=item[stage+'_path'],
                historical_receipt_sha256=item[stage+'_sha256'])
            if stage=='post':value.update(new_D0_F=0,GEO_calls=0,post_status='VERIFIED_HISTORICAL_POST_REUSE')
            else:value['generation_status']='VERIFIED_HISTORICAL_RAW_REUSE'
            C.write(I.record_path(q,label,stage),value)
        I.read_post(q,label);count+=1
    result=dict(status='COMPLETE',label=label,reused_requests=count,new_model_calls=0)
    C.write(EV/'cache_reuse'/f'{label}.json',result);return result


def generate(label,worker=0,workers=1):
    if label not in TRAIN_LABELS[1:]:raise ValueError('Only three trained candidate endpoints generate; step0 and baseline are cache-only')
    check_code();r=arm_registration(label);I=inference(label);b=binding(label);G=I._gpu(label)
    with I.owner('generator',label,worker,workers):
        pending=[]
        for q in I.partition(r,worker,workers):
            p=I.record_path(q,label,'raw')
            if p.exists():I.read_raw(q,label)
            elif p.parent.exists():raise RuntimeError('Unresolved rollout; explicit recovery required, no replay')
            else:C.old().read_input(q);pending.append(q)
        if not pending:return dict(status='COMPLETE',new_rollouts=0)
        torch=I.configure_GPU(G,'generator_'+label,worker,workers,r.get('resources',{}).get('generator_incremental_estimate_MiB',2600))
        verify({b['path']:b['sha256']})
        from meshflow_control.checkpoints import state_hash
        from .model import build_candidate
        from meshflow_control.sampling import sample
        payload=torch.load(C.resolve(b['path']),map_location='cpu',weights_only=True,mmap=True)
        if any(k in payload for k in ('optimizer','rng','teacher_state','aligner_state')):raise ValueError('Use inference export, not training checkpoint')
        if int(payload['global_step'])!=6000+STEPS[label] or int(payload.get('local_step',0))!=STEPS[label]:raise ValueError('Wrong fixed checkpoint endpoint')
        expected_arm='CONTOUR'
        if payload['arm']!=expected_arm:raise ValueError('Wrong training arm')
        if state_hash(payload['generator_state'])!=payload['generator_state_sha256']:raise ValueError('Generator state hash mismatch')
        if b.get('generator_state_sha256',payload['generator_state_sha256'])!=payload['generator_state_sha256']:raise ValueError('Bound generator state differs')
        model=build_candidate(payload['model_config'],trainable=False).eval()
        model.load_state_dict(payload['generator_state'],strict=True);model.to('cuda').set_trainable(False)
        for p in model.parameters():p.requires_grad_(False)
        C.write(I.worker_path('generator',label,worker,workers,'identity'),dict({k:v for k,v in payload.items() if k!='generator_state'},file_sha256=b['sha256']))
        G.snap('generator_resident_'+label,worker,workers)
        class Budget(C.old().CellBudget):
            def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.velocities=[]
            def call(self,kind,fn,context=None):
                value=super().call(kind,fn,context)
                if kind=='sampling_forward':self.velocities.append(value.detach().float().cpu().numpy()[0])
                return value
        topology=topology_inputs()
        count=0;tick=time.perf_counter();allocated=reserved=0
        try:
            for q in pending:
                G.snap('generator_cell_'+label,worker,workers)
                inp=C.old().read_input(q);G.source(q,inp['C'])
                dest=I.record_path(q,label,'raw').parent;dest.mkdir(parents=True,exist_ok=False)
                context=dict(label=label,request_id=q['request_id'],input_sha256=q['input_sha256'],checkpoint_sha256=b['sha256'],
                    topology_manifest_sha256=C.sha(C.OUT/'inputs/topology_manifest.json'),
                    effective_model_inputs=['C','N','Gaussian','time','C_source_connectivity_boundary_mask'])
                budget=Budget(dest/'calls.jsonl',context)
                mask=topology.for_request(q['request_id'])
                mask_tensor=torch.from_numpy(mask[None].copy()).to(device='cuda')
                C.frozen_npz(dest/'topology_input.npz',boundary_edge_mask=mask)
                C.write(dest/'model_input_contract.json',dict(context,base_sampler_model_inputs=['C','N','Gaussian','time'],
                    input_adapter='Owned model.forward partial injects C-only source mask; original Euler50 unchanged',
                    model_metadata=model.contour_metadata(),boundary_mask_array_sha256=C.array_sha(mask),
                    source_condition_sha256=q['condition_source_sha256']))
                original_forward=model.forward
                model.forward=functools.partial(original_forward,boundary_edge_mask=mask_tensor)
                try:
                    _,_,audit=sample(model,inp['C'],int(q['N']),seed=q['noise_seed'],z=inp['z'],budget=budget,out=dest,context=context)
                finally:
                    model.forward=original_forward
                allocated=max(allocated,audit['cuda_peak_allocated_bytes']);reserved=max(reserved,audit['cuda_peak_reserved_bytes'])
                C.frozen_npz(dest/'velocities.npz',dict(velocity=np.stack(budget.velocities),source_dtype=np.array('torch.bfloat16')))
                rec=dict(q);rec.update(context);rec.update(status='COMPLETE',new_rollouts=1,new_sampling_F=50,
                    generation_status='NEW_WORLD_FRAME_EULER50_C_ONLY_CONTOUR_READOUT',topology_path=str(dest/'topology_input.npz'),topology_sha256=C.sha(dest/'topology_input.npz'),
                    model_input_contract_path=str(dest/'model_input_contract.json'),model_input_contract_sha256=C.sha(dest/'model_input_contract.json'),velocities_path=str(dest/'velocities.npz'),velocities_sha256=C.sha(dest/'velocities.npz'),worker=worker,workers=workers)
                for field,name in (('raw','raw.npz'),('ply','raw.ply'),('result','run.json')):rec[field+'_path']=str(dest/name);rec[field+'_sha256']=C.sha(dest/name)
                rec['contracts']=I._raw_contract(q,rec,label);C.write(dest/'receipt.json',rec);I.read_raw(q,label);count+=1
                C.write(I.worker_path('generator',label,worker,workers,'progress'),dict(completed=count,seconds=time.perf_counter()-tick,peak_allocated_bytes=allocated,peak_reserved_bytes=reserved),replace=True)
                print(json.dumps(dict(stage='RAW',label=label,completed=count,request_id=q['request_id'],seconds=audit['seconds'])),flush=True)
            if state_hash(model.state_dict())!=payload['generator_state_sha256']:raise ValueError('Frozen model changed')
            G.snap('generator_finish_'+label,worker,workers)
            result=dict(status='COMPLETE',label=label,new_rollouts=count,new_sampling_F=50*count,seconds=time.perf_counter()-tick,
                peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,training=0,backward=0,
                peak_scope='Per-rollout allocator peak, not full process VRAM')
            C.write(I.worker_path('generator',label,worker,workers,'complete'),result);return result
        except BaseException as exc:
            C.event('EVALUATION_GENERATOR_STOPPED',label=label,completed=count,error=repr(exc),traceback=traceback.format_exc());raise
        finally:del model;torch.cuda.empty_cache()


def post(label,worker=0,workers=1):
    if label not in TRAIN_LABELS[1:]:raise ValueError('Step0 and baseline cannot run D0')
    # The private helper checks all 24 RAW of this endpoint before any D0.
    return inference(label).post(label,worker,workers)


def collect(label):
    check_code();I=inference(label);qs=requests(label)
    raw=[I.read_raw(q,label) for q in qs];postrows=[I.read_post(q,label) for q in qs]
    d=EV/'indices'/label
    for stage,rows in (('raw',raw),('post',postrows)):C.write(d/(stage+'_index.json'),dict(status='COMPLETE',label=label,rows=rows))
    result=dict(status='COMPLETE',label=label,RAW=len(raw),POST=len(postrows));C.write(d/'complete.json',result);return result


@functools.lru_cache(None)
def scorer():
    # CPU metric implementation is unchanged; this private adapter redirects
    # output and keeps the already-reviewed explanatory metadata.
    S=C.module('r2.pilots.regular_patch_control_v1._contour_readout_metrics',OLD_SCORE)
    S.C=C;S.EV=EV;S.VERSION=VERSION;S.META=META
    W=S.cpu_scorer();previous=W.definition_hashes()
    W.definition_hashes=lambda:{**previous,str(Path(__file__)):C.sha(__file__)}
    return W


def _indices(label):
    """Read collected identities without importing generator or CREF classes."""
    qs={q['request_id']:q for q in requests(label)};indices={}
    for stage in ('raw','post'):
        saved=C.read(EV/'indices'/label/(stage+'_index.json'));rows=saved['rows']
        if saved['status']!='COMPLETE' or len(rows)!=24 or {x['request_id'] for x in rows}!=set(qs):
            raise ValueError('Collect all 24 RAW and POST before CPU metrics')
        index={}
        for rec in rows:
            q=qs[rec['request_id']]
            if rec['label']!=label or rec['input_sha256']!=q['input_sha256']:raise ValueError('Collected identity differs')
            if rec['checkpoint_sha256']!=binding(label)['sha256']:raise ValueError('Collected checkpoint differs')
            verify({rec[stage+'_path']:rec[stage+'_sha256']})
            if stage=='post':verify(rec['files'])
            index[q['request_id']]=rec
        indices[stage]=index
    return indices


def _score_one(job):
    q,raw,postrec,query,stage,rh=job
    if 'torch' in sys.modules:raise RuntimeError('CPU scoring must have its own model-free process')
    query=dict(query,transform_audit=dict(method='FROZEN_ORIGINAL_FULL_C_QUERY',new_queries=0,query_identity_sha256=query['sha256']))
    result=scorer().score_one((q,raw,postrec,query,stage,rh));result.update(label=raw['label'],request_id=q['request_id'],stage=stage,
        cache_action='NEW_REGISTERED_SCORE',new_quality_scans_this_call=0 if result['reused'] else 1)
    return result


def _historical_score(q,label,stage,raw,postrec,query):
    item=q['cache'][label];oldlabel=item['historical_label'];rid=q['request_id']
    cp=BASE/'evaluation/score_completion'/f'{oldlabel}.json';completion=C.read(cp)
    if completion['status']!='COMPLETE':raise ValueError('Baseline score incomplete')
    candidates=[ref for ref in completion['receipts'] if (ref['label'],ref['request_id'],ref['stage'])==(oldlabel,rid,stage)]
    if len(candidates)!=1:raise ValueError('Expected exact baseline score identity')
    ref=candidates[0];verify({ref['path']:ref['sha256']});saved=C.read(ref['path']);identity=saved['identity']
    rec=raw if stage=='RAW' else postrec
    if identity['mesh_sha256']!=rec[stage.lower()+'_sha256'] or identity['query_sha256']!=query['sha256']:
        raise ValueError('Baseline score mesh/query differs; no fallback re-score')
    verify(saved['files']);verify(identity['definition_hashes'])
    for key in ('input_sha256','C_sha256','Gaussian_sha256','condition_source_sha256'):
        if identity[key]!=q[key]:raise ValueError('Cached score input differs: '+key)
    if int(saved['N'])!=int(q['N']) or int(saved['K'])!=int(q['K']) or saved['C_query_count']!=2048:
        raise ValueError('Baseline score denominator differs')
    path=EV/'score_reuse'/label/rid/stage/'receipt.json'
    proof=dict(status='VERIFIED_REUSE',label=label,request_id=rid,stage=stage,historical_label=oldlabel,
        historical_request_id=rid,score_path=ref['path'],score_sha256=ref['sha256'],
        original_completion_path=str(cp),original_completion_sha256=C.sha(cp),
        registration_sha256=C.sha(C.OUT/'registration.json'),geometry_sha256=identity['mesh_sha256'],
        query_sha256=query['sha256'],new_quality_scans=0,model_calls=0)
    C.write(path,proof)
    return dict(path=ref['path'],sha256=ref['sha256'],label=label,request_id=rid,stage=stage,reused=True,
        cache_action='VERIFIED_HISTORICAL_SCORE_REUSE',reuse_receipt_path=str(path),reuse_receipt_sha256=C.sha(path),new_quality_scans_this_call=0,seconds=0)

def score(label,workers=2):
    check_code();r=registration();qs=requests(label);idx=_indices(label);tick=time.perf_counter()
    dest=EV/'score_completion'/f'{label}.json'
    if dest.exists():
        result=C.read(dest)
        if result['definition_hashes']!=scorer().definition_hashes():raise ValueError('Scoring definitions changed')
        for ref in result['receipts']:verify({ref['path']:ref['sha256']});verify(C.read(ref['path'])['files'])
        return result
    query={v['condition_id']:v for v in r['queries']};rh=C.sha(C.OUT/'registration.json');jobs=[];receipts=[]
    for q in qs:
        raw=idx['raw'][q['request_id']];pr=idx['post'][q['request_id']]
        for stage in ('RAW','POST'):
            if q['action']=='CACHE':receipts.append(_historical_score(q,label,stage,raw,pr,query[q['condition_id']]))
            else:jobs.append((q,raw,pr,query[q['condition_id']],stage,rh))
    done={};pending=[]
    for i,j in enumerate(jobs):
        p=EV/'scores'/label/j[0]['request_id']/j[4]/'score.json'
        if p.exists():done[i]=_score_one(j)
        else:pending.append((i,j))
    actual=0
    if pending:
        i,job=pending.pop(0);first=_score_one(job);done[i]=first;snapshot=C.resources()
        mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (snapshot.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
        parallel=(os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and (snapshot.get('loadavg') or [999])[0]<(os.cpu_count() or 1)-2
        actual=min(workers,2) if parallel else 1
        C.write(EV/'execution'/label/f'score_dispatch_{time.time_ns()}.json',dict(workers=actual,resources=snapshot,first_authorized_score=first))
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(_score_one,j):i for i,j in pending}
            for f in concurrent.futures.as_completed(futures):
                done[futures[f]]=f.result();print(json.dumps(dict(stage='SCORE',label=label,completed=len(done)+len(receipts),registered=48)),flush=True)
    for i in sorted(done):
        ref=done[i];ref['sha256']=C.sha(ref['path']);receipts.append(ref)
    if len(receipts)!=48:raise ValueError('Need all 24 RAW/POST metric records')
    result=dict(status='COMPLETE',label=label,stages=48,receipts=receipts,workers=actual,seconds=time.perf_counter()-tick,
        definition_hashes=scorer().definition_hashes(),new_quality_scans_this_invocation=sum(x['new_quality_scans_this_call'] for x in receipts),model_calls=0)
    C.write(dest,result);return result


def aggregate(label):
    r=registration();refs=C.read(EV/'score_completion'/f'{label}.json')['receipts'];qs={x['request_id']:x for x in r['requests']};rows=[];vmr=[]
    for ref in refs:
        verify({ref['path']:ref['sha256']});s=C.read(ref['path']);q=qs[ref['request_id']]
        row={k:q[k] for k in META};row.update(label=label,stage=ref['stage'],local_step=STEPS.get(label),
            checkpoint_role='LEARNING_CURVE' if label in TRAIN_LABELS else 'FROZEN_CONTEXT_REFERENCE',score_origin=ref['cache_action'],score_path=ref['path'],score_sha256=ref['sha256'])
        row.update(s['metrics']);row.update({k:v for k,v in s['topology'].items() if isinstance(v,(int,float,bool)) or v is None});row.update(s['distance'])
        row.update({'DESC__'+k:v for k,v in s['descriptive_boundary_distance']['metrics'].items()})
        for item in s['boundary']:
            row['A_VMR__'+item['setting']]=item['VMR'];row['PMR__'+item['setting']]=item['perfect'];vmr.append(dict(label=label,request_id=q['request_id'],stage=ref['stage'],**item))
        row.update(C_preserved_exact=s['C_preserved_exact'],I_status=s['status_by_metric']['I'],free_face_denominator=s['free_face_denominator'],
            C_query_weight_denominator=s['C_query_weight_denominator'],boundary_vertex_denominator=s['boundary'][0]['boundary_vertex_count'],source_C_connectivity_status=s['source_C_connectivity_status'])
        rows.append(row)
    C.csvwrite(EV/'tables'/label/'per_output.csv',rows);C.csvwrite(EV/'tables'/label/'AVMR_PMR_all8.csv',vmr)
    return dict(status='COMPLETE',label=label,rows=len(rows))


def status():
    r=registration();rows=[]
    for label in LABELS:
        rows.append(dict(label=label,raw=sum((EV/'raw'/label/q['request_id']/'receipt.json').exists() for q in r['requests']),
            post=sum((EV/'post'/label/q['request_id']/'receipt.json').exists() for q in r['requests']),
            scores=(EV/'score_completion'/f'{label}.json').exists()))
    return dict(status='READ_ONLY',rows=rows)


def _boundary_one(job):
    import resource
    from r2.tools.full_boundary_attribution_v1 import geometry as B
    q,label,stage,ref,postrec,rh=job
    if 'torch' in sys.modules:raise RuntimeError('Boundary analysis is CPU-only')
    s=C.read(ref['path']);verify({ref['path']:ref['sha256']});verify(s['files'])
    source=C.arrays(q['condition_source']);K=int(q['K'])
    p=(C.resolve(ref['path']).parent/'RAW_source_direction.npz') if stage=='RAW' else C.resolve(postrec['post_path'])
    a=C.arrays(p);tri=a['output'];v=a['vertices'];faces=a['faces']
    if not C.same(v[faces],tri) or not C.same(tri[:K],source['C_original']):raise ValueError('Boundary geometry identity')
    dest=EV/'boundary'/label/q['request_id']/stage;done=dest/'completion.json'
    identity=dict(label=label,request_id=q['request_id'],stage=stage,geometry_path=str(p),geometry_sha256=C.sha(p),
        score_sha256=ref['sha256'],source_sha256=q['condition_source_sha256'],registration_sha256=rh,
        definition_hashes={str(C.ROOT/'r2/tools/full_boundary_attribution_v1/geometry.py'):C.sha(C.ROOT/'r2/tools/full_boundary_attribution_v1/geometry.py'),
        str(C.ROOT/'r2/tools/cref_layer_audit_v1/geometry.py'):C.sha(C.ROOT/'r2/tools/cref_layer_audit_v1/geometry.py')},config=B.DEFAULTS)
    if done.exists():
        previous=C.read(done)
        if previous['identity']!=identity:raise ValueError('Boundary identity changed')
        verify(previous['files']);return previous
    if label=='CONTOUR_S0' or label in REFERENCES:
        oldlabel='FULL_S0000' if label=='CONTOUR_S0' else label
        cp=BASE/'evaluation/boundary_completion'/f'{oldlabel}.json';prior=C.read(cp)
        matches=[x for x in prior['receipts'] if (x['request_id'],x['stage'])==(q['request_id'],stage)]
        if len(matches)!=1:raise ValueError('Expected exact baseline boundary receipt')
        old=matches[0]
        for key in ('geometry_sha256','score_sha256','source_sha256','definition_hashes','config'):
            if old['identity'][key]!=identity[key]:raise ValueError('Baseline boundary differs: '+key)
        verify(old['files'])
        result=dict(status='COMPLETE',identity=identity,label=label,request_id=q['request_id'],stage=stage,
            historical_completion_path=str(cp),historical_completion_sha256=C.sha(cp),
            files=old['files'],summary_path=old['summary_path'],vectors_path=old['vectors_path'],reused=True,
            new_model_calls=0,new_quality_scans=0,seconds=0,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        C.write(done,result);return result
    if dest.exists():raise RuntimeError('Incomplete CPU boundary cell requires explicit recovery')
    C.write(dest/'started.json',identity);tick=time.perf_counter()
    bundle=dict(C_original=source['C_original'],C_source_vertex_ids=source['C_original_source_vertex_ids'],F=tri[K:],
        vertices=v,faces=faces,known_mask=np.arange(len(faces))<K,pairs=C.read(C.resolve(ref['path']).parent/'pairs.json'),L=C.L,
        graph_identity='POST_SAVED_ACTUAL_VERTEX_INDICES' if stage=='POST' else 'RAW_SAVED_EXACT_COORDINATE_OBSERVATIONAL_INDEX_SOURCE_DIRECTION')
    value=B.analyze(bundle,dict(B.DEFAULTS));files={}
    for key,item in value.items():
        if key in ('vectors','normal_line_hits','coplanar_intervals'):out=dest/(key+'.npz');C.frozen_npz(out,item)
        elif key=='summary':out=dest/'summary.json';C.write(out,item)
        else:out=dest/(key+'.csv');C.csvwrite(out,item)
        files[str(out)]=C.sha(out)
    result=dict(status='COMPLETE',identity=identity,label=label,request_id=q['request_id'],stage=stage,files=files,
        summary_path=str(dest/'summary.json'),vectors_path=str(dest/'vectors.npz'),reused=False,
        new_model_calls=0,new_quality_scans=0,seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    C.write(done,result);return result


def boundary(label,workers=2):
    check_code();r=registration();qs={q['request_id']:q for q in requests(label)}
    refs=C.read(EV/'score_completion'/f'{label}.json')['receipts'];rh=C.sha(C.OUT/'registration.json')
    postrows=_indices(label)['post']
    jobs=[(qs[ref['request_id']],label,ref['stage'],ref,postrows[ref['request_id']],rh) for ref in refs]
    tick=time.perf_counter();first=_boundary_one(jobs[0]);snapshot=C.resources()
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (snapshot.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    parallel=(os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and (snapshot.get('loadavg') or [999])[0]<(os.cpu_count() or 1)-2
    actual=min(workers,2) if parallel else 1
    C.write(EV/'execution'/label/f'boundary_dispatch_{time.time_ns()}.json',dict(workers=actual,first_authorized_cell=first['identity'],resources=snapshot,peak_RSS_KiB=first['peak_RSS_KiB']))
    result={0:first}
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(_boundary_one,j):i for i,j in enumerate(jobs[1:],1)}
        for f in concurrent.futures.as_completed(futures):result[futures[f]]=f.result()
    receipts=[result[i] for i in sorted(result)]
    if len(receipts)!=48:raise ValueError('Expected 48 RAW/POST boundary records')
    final=dict(status='COMPLETE',label=label,cells=48,receipts=receipts,workers=actual,
        historical_reuses=sum(x['reused'] for x in receipts),new_model_calls=0,new_quality_scans=0,seconds=time.perf_counter()-tick)
    p=EV/'boundary_completion'/f'{label}.json'
    if p.exists():
        old=C.read(p)
        if [(x['identity'],x['files']) for x in old['receipts']]!=[(x['identity'],x['files']) for x in receipts]:raise ValueError('Boundary completion changed')
        return old
    C.write(p,final);return final


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('check','cache','generate','post','collect','score','boundary','aggregate','status'))
    p.add_argument('--label',choices=LABELS);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if a.stage not in ('check','status') and a.label is None:p.error('--label required')
    if a.stage=='check':value=check_code()
    elif a.stage=='status':value=status()
    elif a.stage in ('generate','post'):value=globals()[a.stage](a.label,a.worker,a.workers)
    elif a.stage in ('score','boundary'):value=globals()[a.stage](a.label,a.workers)
    else:value=globals()[a.stage](a.label)
    print(json.dumps(C.clean({k:v for k,v in value.items() if k!='receipts'}),allow_nan=False),flush=True)

if __name__=='__main__':main()
