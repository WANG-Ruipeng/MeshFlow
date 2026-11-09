"""Frozen N-sensitivity evaluation; original metrics, no newly drawn noise.

Historical implementation is instantiated privately. Only experiment bindings,
fixed endpoint identities and the 48-RAW barrier are changed.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, functools, json, sys, time, traceback
from pathlib import Path
import numpy as np
from . import common as C

VERSION='r2_dedup_n_sensitivity_evaluation_v1'
TRAIN_LABELS=('ORIGINAL_LOW_N','DEDUP_LOW_N')
REFERENCES=('ORIGINAL_ORIG_N','DEDUP_ORIG_N')
LABELS=TRAIN_LABELS+REFERENCES
STEPS={label:200 for label in LABELS}
EV=C.OUT/'evaluation'
PRIOR=C.ROOT/'experiments/r2_target_cleanup_c200_v1'
PRIOR_CODE=C.ROOT/'r2/pilots/full_seat_learning_curve_v1/evaluate.py'
SOURCE_ARM={label:('ORIGINAL_C200' if label.startswith('ORIGINAL_') else 'DEDUP_C200') for label in LABELS}
META=('request_id','parent_id','condition_id','panel','candidate','region_type','tier','N','K','ratio',
      'base_request_id','base_condition_id','variant','alpha','pair_id','split','condition_type')

def verify(files):
    for path,digest in files.items():
        if C.sha(path)!=digest:raise ValueError('Frozen file changed: '+str(path))

def registration():
    r=C.load_registration()
    if r.get('status')!='FROZEN' or tuple(r['evaluation_labels'])!=LABELS:
        raise ValueError('Expected frozen N-sensitivity registration')
    if r['budget']['new_rollouts']!=48 or r['budget']['new_D0_forward']!=48:
        raise ValueError('Registered scientific budget changed')
    qs=r['requests'];old=C.read(PRIOR/'registration.json')
    if len(qs)!=24 or len({q['request_id'] for q in qs})!=24 or len({q['condition_id'] for q in qs})!=6:
        raise ValueError('Expected six FULL conditions / 24 fixed requests')
    if r['original_requests']!=old['requests'] or r['queries']!=old['queries']:
        raise ValueError('Original panel/query changed')
    originals={q['request_id']:q for q in r['original_requests']}
    for q in qs:
        a=originals[q['request_id']];k=int(a['K']);n=int(a['N'])
        expected=max(128,min(256,k+((n-k)*2717*2+3047)//(2*3047)))
        if int(q['N'])!=expected:raise ValueError('Fixed half-up low-N prescription changed')
        if abs(float(q['ratio'])-k/expected)>1e-12:raise ValueError('Low-N condition ratio metadata differs')
        for key in META:
            if key in ('N','ratio'):continue
            if q[key]!=a[key]:raise ValueError('Condition/panel identity changed: '+key)
        for key in ('C_sha256','condition_source','condition_source_sha256','noise_seed'):
            if q[key]!=a[key]:raise ValueError('Original C/source/noise identity changed: '+key)
    return r

def freeze():
    r=registration();originals={q['request_id']:q for q in r['original_requests']};prefix=[]
    for q in r['requests']:
        a=originals[q['request_id']];low=C.old().read_input(q);high=C.old().read_input(a)
        if not C.same(low['C'],high['C']) or not C.same(low['z'],high['z'][:,:q['N']]):
            raise ValueError('Low-N must keep exact old C and Gaussian prefix')
        prefix.append(dict(request_id=q['request_id'],N_low=q['N'],N_original=a['N'],K=q['K'],
            exact_C=True,exact_free_noise_prefix=True,Gaussian_sha256=q['Gaussian_sha256'],original_Gaussian_sha256=a['Gaussian_sha256']))
    files=[Path(__file__),Path(__file__).with_name('render.py'),Path(C.__file__),PRIOR_CODE,
        C.ROOT/'r2/pilots/full_seat_learning_curve_v1/common.py',
        C.ROOT/'r2/pilots/full_seat_learning_curve_v1/render.py',
        C.ROOT/'r2/pilots/regular_patch_control_v1/inference.py',
        C.ROOT/'r2/pilots/regular_patch_control_v1/score.py',
        C.ROOT/'r2/pilots/regular_patch_control_v1/depth_gallery.py',
        C.ROOT/'r2/pilots/target_cleanup_c200_v1/evaluate.py',
        C.ROOT/'r2/pilots/target_cleanup_c200_v1/render.py']
    hashes={str(p):C.sha(p) for p in files}
    hashes.update(scorer().definition_hashes())
    # GPU-only modules are hashed as files without importing torch on this path.
    for p in [C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/gpu.py',
        C.ROOT/'r2/src/meshflow_control/sampling.py',C.ROOT/'r2/src/meshflow_control/models/native.py']:
        hashes[str(p)]=C.sha(p)
    value=dict(status='FROZEN',version=VERSION,registration_sha256=C.sha(C.OUT/'registration.json'),
        evaluation_code_hashes=hashes,labels=LABELS,requests_per_label=24,
        budgets=dict(new_rollouts=48,new_sampling_forwards=2400,new_D0_forwards=48),
        original_metric_definitions=True,model_calls=0)
    C.write(C.OUT/'evaluation_freeze.json',value)
    C.write(C.OUT/'inputs/prefix_verification.json',dict(status='PASS',rows=prefix,new_random_draws=0,model_calls=0))
    return check_code()

def check_code():
    registration();f=C.read(C.OUT/'evaluation_freeze.json')
    if f['status']!='FROZEN' or f['registration_sha256']!=C.sha(C.OUT/'registration.json'):
        raise ValueError('Evaluation/main registration binding changed')
    if str(Path(__file__)) not in f['evaluation_code_hashes']:raise ValueError('Missing evaluator freeze')
    verify(f['evaluation_code_hashes'])
    return dict(status='PASS',labels=LABELS,requests=24,new_rollouts=48,new_D0_forwards=48)

def binding(label):
    if label not in LABELS:raise ValueError('Unregistered label')
    r=registration();p=EV/'bindings'/f'{label}.json'
    b=C.read(p) if p.exists() else dict(r['models'][label])
    expected=(200,12200)
    if (b['local_step'],b['global_step'])!=expected:raise ValueError('Wrong fixed endpoint')
    if b['source_arm']!=SOURCE_ARM[label]:raise ValueError('Frozen source training arm differs')
    if not {'path','sha256'}<=set(b):raise ValueError('Missing export binding')
    return b

def requests(label):
    r=registration();cache=r.get('cache',{}).get(label,{})
    result=[]
    for original in (r['requests'] if label in TRAIN_LABELS else r['original_requests']):
        q=dict(original);q.pop('cache',None);q['action']='NEW'
        if q['request_id'] in cache:q.update(action='CACHE',cache={label:cache[q['request_id']]})
        if label in REFERENCES and q['action']!='CACHE':raise ValueError('Baseline must be fully cached')
        if label in TRAIN_LABELS and q['action']!='NEW':raise ValueError('New endpoints cannot borrow old outputs')
        result.append(q)
    return result

def arm_registration(label):
    r=dict(registration());r['requests']=requests(label);r['labels']=[label]
    return r

@functools.lru_cache(None)
def inference(label):
    I=C.module('r2.pilots.regular_patch_control_v1._dedup_n_inference_'+label,
        C.ROOT/'r2/pilots/regular_patch_control_v1/inference.py')
    I.C=C;I.EV=EV;I.LABELS=(label,)
    I.registration=lambda:arm_registration(label)
    I.binding=binding;I.check_code=check_code
    def barrier(r,arm=None):
        # Verify all 48, then return only this arm: the D0 helper indexes by
        # request_id, which is intentionally shared across paired model arms.
        rows={a:[inference(a).read_raw(q,a) for q in requests(a)] for a in TRAIN_LABELS}
        return rows[label]
    I.all_raw=barrier
    return I

@functools.lru_cache(None)
def old():
    F=C.module('r2.pilots.full_seat_learning_curve_v1._dedup_n_eval_private',PRIOR_CODE)
    F.C=C;F.VERSION=VERSION;F.TRAIN_LABELS=TRAIN_LABELS;F.REFERENCES=REFERENCES;F.LABELS=LABELS
    F.STEPS=STEPS;F.EV=EV
    F.registration=registration;F.check_code=check_code;F.binding=binding;F.requests=requests
    F.arm_registration=arm_registration;F.inference=inference
    F._target_original_score_one=F._score_one;F._score_one=_score_one
    F._historical_score=_historical_score
    return F

def scorer():return old().scorer()
def _score_one(job):return old()._target_original_score_one(job)

def _historical_score(q,label,stage,raw,postrec,query):
    if label not in REFERENCES:raise ValueError('Only original-N scores may be reused')
    historical=SOURCE_ARM[label]
    cp=PRIOR/'evaluation/score_completion'/(historical+'.json');completion=C.read(cp)
    matching=[ref for ref in completion['receipts'] if (ref.get('label'),ref.get('request_id'),ref.get('stage'))==(historical,q['request_id'],stage)]
    if len(matching)!=1:raise ValueError('Exact cached score identity unavailable')
    ref=matching[0];verify({ref['path']:ref['sha256']});s=C.read(ref['path']);ident=s['identity']
    rec=raw if stage=='RAW' else postrec
    if ident['mesh_sha256']!=rec[stage.lower()+'_sha256'] or ident['query_sha256']!=query['sha256']:
        raise ValueError('Cached score mesh/query differs')
    for key in ('input_sha256','C_sha256','Gaussian_sha256','condition_source_sha256'):
        if ident[key]!=q[key]:raise ValueError('Cached score input differs: '+key)
    verify(s['files']);verify(ident['definition_hashes'])
    current=scorer().definition_hashes()
    for path in set(current)&set(ident['definition_hashes']):
        if current[path]!=ident['definition_hashes'][path]:raise ValueError('Metric implementation changed')
    if int(s['N'])!=q['N'] or int(s['K'])!=q['K'] or s['C_query_count']!=2048:
        raise ValueError('Cached denominator changed')
    path=EV/'score_reuse'/label/q['request_id']/stage/'receipt.json'
    proof=dict(status='VERIFIED_REUSE',label=label,request_id=q['request_id'],stage=stage,
        historical_label=historical,historical_request_id=q['request_id'],score_path=ref['path'],score_sha256=ref['sha256'],
        original_completion_path=str(cp),original_completion_sha256=C.sha(cp),
        registration_sha256=C.sha(C.OUT/'registration.json'),geometry_sha256=ident['mesh_sha256'],query_sha256=query['sha256'],new_quality_scans=0,model_calls=0)
    C.write(path,proof)
    return dict(path=ref['path'],sha256=ref['sha256'],label=label,request_id=q['request_id'],stage=stage,reused=True,
        cache_action='VERIFIED_HISTORICAL_SCORE_REUSE',reuse_receipt_path=str(path),reuse_receipt_sha256=C.sha(path),new_quality_scans_this_call=0,seconds=0)

def cache(label):return old().cache(label)
def collect(label):return old().collect(label)
def score(label,workers=2):return old().score(label,workers)
def status():return old().status()

def generate(label,worker=0,workers=1):
    if label not in TRAIN_LABELS:raise ValueError('Baseline cannot generate')
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
        from meshflow_control.models.native import build_model
        from meshflow_control.sampling import sample
        payload=torch.load(C.resolve(b['path']),map_location='cpu',weights_only=True,mmap=True)
        if any(k in payload for k in ('optimizer','rng','teacher_state','aligner_state')):raise ValueError('Use inference export only')
        if (payload['global_step'],payload['local_step'],payload['arm'])!=(12200,200,b['source_arm']):raise ValueError('Wrong frozen source endpoint')
        if state_hash(payload['generator_state'])!=payload['generator_state_sha256']:raise ValueError('Generator hash differs')
        if b['generator_state_sha256']!=payload['generator_state_sha256']:raise ValueError('Bound generator differs')
        model=build_model(payload['model_config'],readout_mode='none',trainable=False).eval()
        model.load_state_dict(payload['generator_state'],strict=True);model.to('cuda').set_trainable(False)
        for p in model.parameters():p.requires_grad_(False)
        C.write(I.worker_path('generator',label,worker,workers,'identity'),dict({k:v for k,v in payload.items() if k!='generator_state'},file_sha256=b['sha256']))
        G.snap('generator_resident_'+label,worker,workers)
        class Budget(C.old().CellBudget):
            def __init__(self,*a,**kw):super().__init__(*a,**kw);self.velocities=[]
            def call(self,kind,fn,context=None):
                value=super().call(kind,fn,context)
                if kind=='sampling_forward':self.velocities.append(value.detach().float().cpu().numpy()[0])
                return value
        count=0;tick=time.perf_counter();allocated=reserved=0
        try:
            for q in pending:
                G.snap('generator_cell_'+label,worker,workers)
                inp=C.old().read_input(q);G.source(q,inp['C'])
                dest=I.record_path(q,label,'raw').parent;dest.mkdir(parents=True,exist_ok=False)
                context=dict(label=label,request_id=q['request_id'],input_sha256=q['input_sha256'],checkpoint_sha256=b['sha256'])
                budget=Budget(dest/'calls.jsonl',context)
                _,_,audit=sample(model,inp['C'],int(q['N']),seed=q['noise_seed'],z=inp['z'],budget=budget,out=dest,context=context)
                allocated=max(allocated,audit['cuda_peak_allocated_bytes']);reserved=max(reserved,audit['cuda_peak_reserved_bytes'])
                C.frozen_npz(dest/'velocities.npz',dict(velocity=np.stack(budget.velocities),source_dtype=np.array('torch.bfloat16')))
                rec=dict(q);rec.update(context);rec.update(status='COMPLETE',new_rollouts=1,new_sampling_F=50,
                    generation_status='NEW_WORLD_FRAME_EULER50',velocities_path=str(dest/'velocities.npz'),velocities_sha256=C.sha(dest/'velocities.npz'),worker=worker,workers=workers)
                for field,name in (('raw','raw.npz'),('ply','raw.ply'),('result','run.json')):rec[field+'_path']=str(dest/name);rec[field+'_sha256']=C.sha(dest/name)
                rec['contracts']=I._raw_contract(q,rec,label);C.write(dest/'receipt.json',rec);I.read_raw(q,label);count+=1
                C.write(I.worker_path('generator',label,worker,workers,'progress'),dict(completed=count,seconds=time.perf_counter()-tick,peak_allocated_bytes=allocated,peak_reserved_bytes=reserved),replace=True)
                print(json.dumps(dict(stage='RAW',label=label,completed=count,request_id=q['request_id'],seconds=audit['seconds'])),flush=True)
            if state_hash(model.state_dict())!=payload['generator_state_sha256']:raise ValueError('Frozen generator changed')
            G.snap('generator_finish_'+label,worker,workers)
            result=dict(status='COMPLETE',label=label,new_rollouts=count,new_sampling_F=50*count,seconds=time.perf_counter()-tick,
                peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,training=0,backward=0,
                peak_scope='Per-rollout allocator peak, not full process VRAM')
            C.write(I.worker_path('generator',label,worker,workers,'complete'),result);return result
        except BaseException as exc:
            C.event('EVALUATION_GENERATOR_STOPPED',label=label,completed=count,error=repr(exc),traceback=traceback.format_exc());raise
        finally:del model;torch.cuda.empty_cache()

def post(label,worker=0,workers=1):
    if label not in TRAIN_LABELS:raise ValueError('Baseline cannot call D0')
    # I.all_raw verifies all 48 authorized new RAW files before any D0 import/load.
    return inference(label).post(label,worker,workers)

def aggregate(label):
    refs=C.read(EV/'score_completion'/f'{label}.json')['receipts']
    qs={x['request_id']:x for x in requests(label)};rows=[];vmr=[]
    originals={x['request_id']:x for x in registration()['original_requests']}
    for ref in refs:
        verify({ref['path']:ref['sha256']});s=C.read(ref['path']);q=qs[ref['request_id']]
        row={k:q[k] for k in META}
        row.update(label=label,stage=ref['stage'],local_step=200,global_step=12200,source_arm=SOURCE_ARM[label],
            N_policy='LOW_N_FIXED_FREE_RATIO' if label in TRAIN_LABELS else 'ORIGINAL_N_CACHED',
            original_N=originals[q['request_id']]['N'],free_slots=q['N']-q['K'],face_bin=20,N_embedding_bin=q['N']//20,
            checkpoint_role='FROZEN_STEP200_WEIGHT_N_INTERVENTION',score_origin=ref['cache_action'],score_path=ref['path'],score_sha256=ref['sha256'])
        if int(s['N'])!=q['N'] or int(s['K'])!=q['K']:raise ValueError('Score/table N metadata differs')
        row.update(s['metrics']);row.update({k:v for k,v in s['topology'].items() if isinstance(v,(int,float,bool)) or v is None});row.update(s['distance'])
        row.update({'DESC__'+k:v for k,v in s['descriptive_boundary_distance']['metrics'].items()})
        for item in s['boundary']:
            row['A_VMR__'+item['setting']]=item['VMR'];row['PMR__'+item['setting']]=item['perfect']
            vmr.append(dict(label=label,request_id=q['request_id'],stage=ref['stage'],**item))
        row.update(C_preserved_exact=s['C_preserved_exact'],I_status=s['status_by_metric']['I'],free_face_denominator=s['free_face_denominator'],
            C_query_weight_denominator=s['C_query_weight_denominator'],boundary_vertex_denominator=s['boundary'][0]['boundary_vertex_count'],source_C_connectivity_status=s['source_C_connectivity_status'])
        row['GEO_deleted_fraction_of_requested_free']=row['deleted_free_faces']/(q['N']-q['K'])
        row['retained_fraction_of_requested_free']=row['actual_free_faces']/(q['N']-q['K'])
        rows.append(row)
    C.csvwrite(EV/'tables'/label/'per_output.csv',rows);C.csvwrite(EV/'tables'/label/'AVMR_PMR_all8.csv',vmr)
    return dict(status='COMPLETE',label=label,rows=len(rows))

def paired():
    check_code();rows=[]
    for label in LABELS:
        aggregate(label);rows+=C.csvread(EV/'tables'/label/'per_output.csv')
    numeric=[]
    excluded=set(META)|{'label','stage','local_step','global_step','source_arm','N_policy','original_N','face_bin','N_embedding_bin','checkpoint_role','score_origin','score_path','score_sha256'}
    for key in rows[0]:
        if key in excluded:continue
        try:
            values=[float(r[key]) for r in rows if r.get(key) not in ('','None','NA')]
            if values and all(np.isfinite(values)):numeric.append(key)
        except (ValueError,TypeError):pass
    lookup={(r['label'],r['request_id'],r['stage']):r for r in rows};pairs=[]
    comparisons=[('ORIGINAL_LOW_N','ORIGINAL_ORIG_N'),('DEDUP_LOW_N','DEDUP_ORIG_N'),
        ('DEDUP_ORIG_N','ORIGINAL_ORIG_N'),('DEDUP_LOW_N','ORIGINAL_LOW_N')]
    for label,comparator in comparisons:
        for q in registration()['requests']:
            for stage in ('RAW','POST'):
                a=lookup[label,q['request_id'],stage];b=lookup[comparator,q['request_id'],stage]
                row=dict(label=label,reference=comparator,request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],stage=stage,noise=q['candidate'],
                    N=int(a['N']),reference_N=int(b['N']))
                for key in numeric:
                    try:row[key+'__delta']=float(a[key])-float(b[key])
                    except (ValueError,TypeError):row[key+'__delta']=None
                pairs.append(row)
    summaries=[]
    for label,reference,stage,parent in sorted({(r['label'],r['reference'],r['stage'],r['parent_id']) for r in pairs}):
        group=[r for r in pairs if (r['label'],r['reference'],r['stage'],r['parent_id'])==(label,reference,stage,parent)]
        out=dict(label=label,reference=reference,stage=stage,parent_id=parent,split=group[0]['split'],paired_noises=len(group))
        for key in numeric:
            vals=[r[key+'__delta'] for r in group if r[key+'__delta'] is not None]
            out[key+'__mean_delta']=float(np.mean(vals)) if vals else None
        summaries.append(out)
    C.csvwrite(EV/'tables/per_output.csv',rows);C.csvwrite(EV/'tables/paired_per_noise.csv',pairs);C.csvwrite(EV/'tables/paired_per_parent.csv',summaries)
    interaction=[]
    for q in registration()['requests']:
        for stage in ('RAW','POST'):
            row=dict(request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],noise=q['candidate'],stage=stage,
                definition='(DEDUP_LOW_N-DEDUP_ORIG_N)-(ORIGINAL_LOW_N-ORIGINAL_ORIG_N)')
            for key in numeric:
                try:
                    vals={label:float(lookup[label,q['request_id'],stage][key]) for label in LABELS}
                    row[key+'__interaction']=vals['DEDUP_LOW_N']-vals['DEDUP_ORIG_N']-vals['ORIGINAL_LOW_N']+vals['ORIGINAL_ORIG_N']
                except (ValueError,TypeError):row[key+'__interaction']=None
            interaction.append(row)
    C.csvwrite(EV/'tables/weight_by_N_interaction.csv',interaction)
    groups=[]
    for label in LABELS:
        for stage in ('RAW','POST'):
            for split in ('ALL','TRAIN','DEV'):
                group=[r for r in rows if r['label']==label and r['stage']==stage and (split=='ALL' or r['split']==split)]
                out=dict(label=label,stage=stage,historical_split=split,outputs=len(group),
                    legal_CF_outputs=sum(float(r['C_F_valid_edge_count'])>0 for r in group),
                    oriented_reach_ge30_outputs=sum(float(r['opposite_winding_reached_free_area_fraction'])>=.3 for r in group),
                    C_exact_all=all(str(r['C_preserved_exact'])=='True' for r in group))
                for key in numeric:
                    vals=[]
                    for r in group:
                        try:vals.append(float(r[key]))
                        except (ValueError,TypeError):pass
                    out[key+'__mean']=float(np.mean(vals)) if vals else None
                groups.append(out)
    C.csvwrite(EV/'tables/group_summary.csv',groups)
    return dict(status='COMPLETE',outputs=len(rows),noise_pairs=len(pairs),parent_pairs=len(summaries),interactions=len(interaction),model_calls=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('freeze','check','cache','generate','post','collect','score','aggregate','paired','status'))
    p.add_argument('--label',choices=LABELS);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if a.stage not in ('freeze','check','paired','status') and a.label is None:p.error('--label required')
    if a.stage=='check':value=check_code()
    elif a.stage in ('freeze','paired','status'):value=globals()[a.stage]()
    elif a.stage in ('generate','post'):value=globals()[a.stage](a.label,a.worker,a.workers)
    elif a.stage=='score':value=score(a.label,a.workers)
    else:value=globals()[a.stage](a.label)
    print(json.dumps(C.clean({k:v for k,v in value.items() if k!='receipts'}),allow_nan=False),flush=True)

if __name__=='__main__':main()
