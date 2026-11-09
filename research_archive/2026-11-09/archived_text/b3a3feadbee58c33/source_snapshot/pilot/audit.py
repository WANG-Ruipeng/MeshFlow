"""Independent CPU accounting and saved-artifact audit; no model forwards.

Unfinished stages remain PENDING. This audit never repairs, retries, selects a
checkpoint, or changes any scorer. Checkpoint inspection uses CPU mmap only.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='2'
import argparse,collections,csv,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.single_step_denoiser_distillation_v1 import common as C

OUT=C.OUT/'independent_audit'
LEGAL={'shared_edge_contact','shared_vertex_contact','coplanar_boundary_or_tolerance_contact',
       'noncoplanar_edge_or_tangent_contact','point_or_tangent_contact'}


def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def near(a,b):return bool(np.isclose(float(a),float(b),rtol=2e-12,atol=1e-14))
def present(path):return Path(path).exists()


def data():
    d=C.read(C.OUT/'manifest/data.json');groups=('train_inputs','internal_val','dev_real','dev_synthetic')
    assert tuple(len(d[k]) for k in groups)==(1120,25,48,18)
    allrows=sum((d[k] for k in groups),[]);byid={r['sample_id']:r for r in allrows}
    assert len(byid)==1211 and len(d['generation_plan'])==249
    tr=d['train_inputs'];dev=d['dev_real']+d['dev_synthetic']
    assert len({r['uid'] for r in tr})==25 and len({r['condition_id'] for r in tr})==112
    for k in ('uid','group_id'):
        assert not {r[k] for r in tr}&{r[k] for r in dev}
    counts=collections.Counter((r['domain'],r['variant']) for r in tr)
    assert sum(r['domain']=='real' for r in tr)==224 and sum(r['domain']=='synthetic' for r in tr)==896
    for cid in {r['condition_id'] for r in tr}:
        rr=[r for r in tr if r['condition_id']==cid]
        assert sorted(r['variant'] for r in rr if r['domain']=='real')==[0,1]
        assert sorted(r['variant'] for r in rr if r['domain']=='synthetic')==list(range(8))
    stream=C.read(d['train_stream_path']);assert C.sha(d['train_stream_path'])==d['train_stream_sha256']
    assert stream['updates']==4000 and len(stream['batches'])==4000
    exposure=collections.Counter();slots=collections.Counter()
    for index,b in enumerate(stream['batches']):
        assert b['step']==index and len(b['sample_ids'])==8
        rr=[byid[i] for i in b['sample_ids']]
        assert len({r['uid'] for r in rr})==8 and all(r['training_eligible'] for r in rr)
        assert sum(r['domain']=='real' for r in rr)==4
        assert b['free_scalar_denominator']==9*sum(r['N']-r['K'] for r in rr)
        for slot,r in enumerate(rr):exposure[(r['uid'],r['domain'])]+=1;slots[(slot,r['domain'])]+=1
    assert sum(exposure.values())==32000
    existing=0;missing=[]
    for r in allrows:
        p=Path(r['input_path'])
        if not p.exists():missing.append(r['sample_id']);continue
        b=arrays(p);n,k=int(r['N']),int(r['K']);x=b['input_d'];known=b['known_mask']
        assert x.dtype==np.float32 and x.shape==(n,3,3) and np.isfinite(x).all()
        assert known.dtype==np.bool_ and bits(known,np.arange(n)<k)
        assert b['C'].dtype==np.float32 and b['C'].shape==(k,3,3)
        if r.get('input_sha256'):assert C.sha(p)==r['input_sha256']
        if r['domain']=='real':
            assert bits(b['input_model'][:k],b['C'])
            assert bits(np.multiply(b['input_model'],np.float32(C.SCALE),dtype=np.float32),x)
        existing+=1
    return dict(status='PASS' if not missing else 'PENDING',conditions=112,training_parents=25,real_train=224,synthetic_train=896,
        internal_validation=25,DEV_real=48,DEV_synthetic=18,stream_updates=4000,stream_exposures=32000,
        distinct_parents_each_batch=8,real_synthetic_each_batch=[4,4],inputs_present=existing,inputs_pending=len(missing),
        pending_input_ids=missing,UID_group_isolation=True,stream_sha256=C.sha(d['train_stream_path']),
        parent_domain_exposure=[dict(uid=u,domain=dom,count=n) for (u,dom),n in sorted(exposure.items())],
        slot_domain_exposure=[dict(slot=sl,domain=dom,count=n) for (sl,dom),n in sorted(slots.items())])


def engineering_exceptions(phases,calls):
    allowed=[];proofs=[]
    for phase in phases:
        if phase['status']=='COMPLETE':continue
        if phase['name']!='generate_training_raw' or phase['status']!='FAILED':continue
        root=C.OUT/'audit/engineering_fix_001';receipt=C.read(root/'receipt.json')
        original=C.read(root/'failed_phase_original.json');before=(root/'common.before.py').read_text()
        assert receipt['status']=='ENGINEERING_FIX_BEFORE_FIRST_MODEL_CALL'
        assert receipt['actual_forward']==receipt['actual_backward']==receipt['updates']==0 and receipt['no_new_input_or_noise']
        assert C.sha(root/'failed_phase_original.json')==receipt['failed_phase_original_sha256']
        assert original['error']==phase['error']=='TypeError("event() got multiple values for argument \'kind\'")'
        assert original['counts']=={'generator_F':1} and original['pending']['call']==1
        assert phase['counts']=={'generator_F':0} and phase['actual_calls_proved']==0 and phase['reservation_attempts']=={'generator_F':1}
        assert phase['pending'] is None and phase['gpu_occupancy_seconds']==original['gpu_occupancy_seconds']
        assert not any(r['phase']==phase['name'] for r in calls)
        assert "def event(kind,ledger=" in before and "event('OP_STARTED',ledger='gpu_calls.jsonl',phase=self.name,**self.pending)" in before
        assert "self.pending=dict(kind=kind,call=" in before and 'self.begin(kind,identity)\n  value=fn()' in before
        # Python binds event's positional kind and **pending.kind before entry;
        # this TypeError precedes fn(), so reservation is not a model forward.
        after=before.replace("def event(kind,ledger=","def event(event_type,ledger=").replace("event=kind,**kw)","event=event_type,**kw)")
        assert after==(C.CODE/'common.py').read_text()
        allowed.append(phase['name']);proofs.append(dict(phase=phase['name'],actual_model_calls=0,
            reserved_attempts=1,failed_time_counted_seconds=phase['gpu_occupancy_seconds'],receipt_sha256=C.sha(root/'receipt.json'),
            original_failure_sha256=C.sha(root/'failed_phase_original.json'),fix_diff_sha256=C.sha(root/'code.diff')))
    return set(allowed),proofs


def budgets(final=False):
    phases=[C.read(p) for p in sorted((C.OUT/'gpu_phases').glob('*.json'))]
    path=C.OUT/'gpu_calls.jsonl';calls=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    starts={};complete={}
    for row in calls:
        key=(row['phase'],row['kind'],row['call'])
        dest=starts if row['event']=='OP_STARTED' else complete
        assert key not in dest;dest[key]=row
    assert set(complete)<=set(starts)
    for key,r in complete.items():assert r['identity']==starts[key]['identity']
    counts=collections.Counter(r['kind'] for r in starts.values())
    for key,limit in C.LIMITS.items():assert counts[key]<=limit
    for p in phases:
        local=collections.Counter(r['kind'] for r in starts.values() if r['phase']==p['name'])
        assert all(local[k]==v for k,v in p['counts'].items())
        assert p['dispatch']['workers']==1
        if p['status']=='COMPLETE':assert p['pending'] is None and all(k in complete for k in starts if k[0]==p['name'])
    seconds=sum(p['gpu_occupancy_seconds'] for p in phases)
    assert seconds<=14400
    expected=dict(generator_F=12450,teacher_target_F=1193,student_train_F=32000,student_train_B=32000,
        student_updates=4000,student_diagnostic_F=165,student_eval_F=66,teacher_benchmark_F=228,student_benchmark_F=228,
        student_preflight_B=16,student_discarded_updates=2)
    reached=all(counts[k]==v for k,v in expected.items())
    manifest=C.read(C.OUT/'manifest/data.json');stream=C.read(manifest['train_stream_path'])['batches']
    semantic=collections.defaultdict(set)
    trainids={r['sample_id'] for r in manifest['train_inputs']};validids={r['sample_id'] for r in manifest['internal_val']}
    realids={r['sample_id'] for r in manifest['dev_real']};syntheticids={r['sample_id'] for r in manifest['dev_synthetic']}
    genids={r['sample_id'] for r in manifest['generation_plan']};monitorids=validids|set(manifest['monitor_ids'])
    for operation in starts.values():
        kind=operation['kind'];identity=operation['identity'];tag=None
        if kind in ('student_train_F','student_train_B'):
            step,micro=int(identity['step']),int(identity['micro']);assert 1<=step<=4000 and 0<=micro<8
            if kind=='student_train_F':assert identity['sample_id']==stream[step-1]['sample_ids'][micro]
            tag=(step,micro)
        elif kind=='student_updates':tag=int(identity['step']);assert 1<=tag<=4000
        elif kind=='generator_F':
            tag=(identity['sample_id'],int(identity['euler_step']));assert tag[0] in genids and 0<=tag[1]<50
        elif kind=='teacher_target_F':tag=identity;assert tag in trainids|validids|realids
        elif kind=='student_eval_F':tag=identity;assert tag in realids|syntheticids
        elif kind=='student_diagnostic_F':
            tag=(int(identity['step']),identity['sample_id']);assert tag[0] in (0,500,1000,2000,4000) and tag[1] in monitorids
        if tag is not None:
            assert tag not in semantic[kind],('DUPLICATE_SCIENTIFIC_OPERATION',kind,tag)
            semantic[kind].add(tag)
    if final:
        assert semantic['student_train_F']==semantic['student_train_B']=={(step,micro) for step in range(1,4001) for micro in range(8)}
        assert semantic['student_updates']==set(range(1,4001))
        assert semantic['generator_F']=={(sid,i) for sid in genids for i in range(50)}
        assert semantic['teacher_target_F']==trainids|validids|realids and semantic['student_eval_F']==realids|syntheticids
        assert semantic['student_diagnostic_F']=={(step,sid) for step in (0,500,1000,2000,4000) for sid in monitorids}
    allowed,exceptions=engineering_exceptions(phases,calls)
    if final:assert reached and len(starts)==len(complete) and all(p['status']=='COMPLETE' or p['name'] in allowed for p in phases)
    return dict(status='PASS' if reached and len(starts)==len(complete) else 'PENDING',started=dict(counts),
        closed_operations=len(complete),unclosed_operations=len(starts)-len(complete),GPU_seconds=seconds,
        GPU_time_scope='COMPLETED_TOTAL' if all(p['status']!='RUNNING' for p in phases) else 'PARTIAL_STORED_PHASE_SNAPSHOTS_RUNNING_TIME_NOT_YET_CLOSED',
        max_GPU_workers=1,expected_complete=expected,teacher_backward_updates=0,model_audit_forwards=0,engineering_exceptions=exceptions,
        actual_operation_identities_unique=True,actual_microstream_ids_match_frozen_stream=True,semantic_operation_counts={k:len(v) for k,v in semantic.items()})


def targets():
    d=C.read(C.OUT/'manifest/data.json');cache=C.OUT/'teacher_cache/completion.json'
    if not cache.exists():return dict(status='PENDING',reason='Teacher target stage not complete')
    done=C.read(cache);assert done['status']=='COMPLETE' and len(done['rows'])==1211
    assert done['new_teacher_forwards']==1193 and done['exact_synthetic_reuses']==18
    assert done['teacher_parameter_before']==done['teacher_parameter_after'] and done['teacher_gradients']==0
    verified=0
    for r in sum((d[k] for k in ('train_inputs','internal_val','dev_real','dev_synthetic')),[]):
        p=Path(r['teacher_cache_path']);rec=C.read(p.with_suffix('.json'));t=arrays(p);inp=arrays(r['input_path'])
        assert rec['status']=='PASS' and C.sha(p)==rec['sha256'] and C.sha(r['input_path'])==rec['input_sha256']
        assert rec['teacher_sha256']==C.EXPECTED and rec['no_rng_consumed']
        assert bits(t['input_d'],inp['input_d']) and bits(t['r_T_d'],np.subtract(t['proposal_T_d'],t['input_d'],dtype=np.float32))
        k=int(r['K']);assert bits(t['C'],inp['C']) and bits(t['output'][:k],inp['C'])
        assert bits(t['output'][k:],t['proposal_T_model'][k:])
        assert all(np.isfinite(t[name]).all() for name in ('input_d','proposal_T_d','r_T_d','proposal_T_model','output'))
        if r.get('historical_teacher_model_space_cache'):
            assert bits(t['proposal_T_model'],arrays(r['historical_teacher_model_space_cache'])['official_unconstrained_proposal'])
        if r['split']=='dev_synthetic':
            old=arrays(r['teacher_reuse_path']);assert rec['reused'] and bits(t['proposal_T_d'],old[r['teacher_proposal_key']])
        verified+=1
    return dict(status='PASS',teacher_targets_verified=verified,real_DEV_exact_D0_replay=48,synthetic_exact_reuses=18,
        teacher_parameter_unchanged=True,effective_FP32_residual_exact=True,C_byte_exact=True)


def preflight():
    paths=[C.OUT/'preflight'/f'{name}_completion.json' for name in ('teacher','student')]
    if not all(p.exists() for p in paths):return dict(status='PENDING',reason='Preflight incomplete')
    details=[]
    for p in paths:
        r=C.read(p);assert r['status']=='PASS' and C.sha(r['result_path'])==r['result_sha256'];details.append(C.read(r['result_path']))
    teacher,student=details
    assert student['formal_updates']==0 and student['discarded_updates']==2 and student['student_backwards']==16
    assert student['student_forwards']<=32 and student['no_rng_consumed'] and not student['teacher_loaded']
    assert student['export_independent_reload']
    spec=student['initial_structure']
    assert spec['actual_num_layers']==8 and spec['actual_hidden_dim']==512 and spec['actual_num_heads']==8
    assert spec['total_parameters']<=50_000_000 and spec['face_count_dropout_actual']==0 and spec['final_residual_zero']
    assert spec['known_role'] is False and spec['teacher_checkpoint_loaded'] is False
    return dict(status='PASS',student_forwards=student['student_forwards'],student_backwards=16,discarded_updates=2,
        student_parameters=spec['total_parameters'],teacher_parameters=spec['teacher_parameters'],actual_architecture=[8,512,8],
        random_initial_state_sha256=student['initial_state_sha256'],no_teacher_weight_initialization=True,
        teacher_preflight_record=C.sha(paths[0]),student_preflight_record=C.sha(paths[1]))


def independent_graph(a,K,pairs):
    v,f=a['vertices'],a['faces'];t=v[f].astype(np.float64)
    ar=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
    dup=collections.Counter(tuple(sorted(map(int,face))) for face in f)
    ed=collections.defaultdict(list)
    for i,face in enumerate(f):
        for q in range(3):
            u,w=int(face[q]),int(face[(q+1)%3]);ed[tuple(sorted((u,w)))].append((i,u,w))
    cf={tuple(map(int,x['pair'])):x['classification'] for x in pairs['CF']['contacts']}
    ff={tuple(sorted(map(int,x['pair']))):x['classification'] for x in pairs['FF']['contacts']}
    parent=list(range(len(f)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    nvalid=ncf=nop=0;length=0.
    for (u,w),inc in ed.items():
        if len(inc)!=2:continue
        (i,u0,w0),(j,u1,w1)=inc
        if max(i,j)<K or i==j or u==w or not np.linalg.norm(v[u].astype(float)-v[w])>0:continue
        if ar[i]<=0 or ar[j]<=0 or dup[tuple(sorted(map(int,f[i])))]>1 or dup[tuple(sorted(map(int,f[j])))]>1:continue
        cross=min(i,j)<K
        label=cf.get((min(i,j),max(i,j)-K)) if cross else ff.get(tuple(sorted((i-K,j-K))))
        if label not in LEGAL:continue
        nvalid+=1;parent[find(i)]=find(j)
        if cross:
            ncf+=1;nop+=int(u0==w1 and w0==u1);length+=float(np.linalg.norm(v[u].astype(float)-v[w]))
    seeded={find(i) for i in range(K)};reached=[i for i in range(K,len(f)) if find(i) in seeded]
    return dict(graph_valid_edges=nvalid,C_F_valid_edge_count=ncf,C_F_valid_opposite_winding_edge_count=nop,
        C_F_valid_edge_total_length=length,reached_free_faces=len(reached),reached_free_area=float(ar[reached].sum()),
        free_area=float(ar[K:].sum()),reached_free_area_fraction=float(ar[reached].sum()/ar[K:].sum()) if ar[K:].sum()>0 else None)


def evaluation():
    root=C.OUT/'evaluation';completion=root/'completion.json'
    if not completion.exists():return dict(status='PENDING',reason='Final geometry scores not complete')
    er=C.read(root/'registration.json');compat=None
    if C.sha(C.CODE/'evaluation.py')!=er['evaluation_source_sha256']:
        import ast
        compat=C.read(root/'compatibility/engineering_fix_001.json')
        assert compat['original_source_sha256']==er['evaluation_source_sha256'] and compat['revised_source_sha256']==C.sha(C.CODE/'evaluation.py')
        assert compat['registration_sha256']==C.sha(root/'registration.json') and compat['no_scientific_definition_change'] and compat['no_model_or_quality_replay']
        C.verify(compat['files'])
        def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef)}
        before,after=functions(compat['original_source_snapshot']),functions(C.CODE/'evaluation.py')
        assert sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))==['event','fidelity','job_open','prepare']
        proof=C.read(root/'compatibility/engineering_fix_001/legacy_quality_evidence.json')
        assert len(proof['rows'])==48 and proof['historical_START_reconstructed'] is False
        for item in proof['rows']:
            assert C.sha(item['score_path'])==item['score_sha256'] and C.sha(item['diagnostic_path'])==item['diagnostic_sha256']
    C.verify(er['protected'])
    events=[json.loads(x) for x in (root/'ledger.jsonl').read_text().splitlines()]
    events += [C.read(p) for p in (root/'events').glob('*/*/*/*.json')]
    def event_ids(name):
        values=[(r['method'],r['label'],r['request_id']) for r in events if r['event']==name]
        assert len(values)==len(set(values));return set(values)
    gs,gc=event_ids('GEO_STARTED'),event_ids('GEO_COMPLETE');qs,qc=event_ids('QUALITY_STARTED'),event_ids('QUALITY_COMPLETE')
    assert gs==gc and len(gc)==96 and len(qc)==96 and qs<=qc
    exceptions={(r['method'],r['label'],r['request_id']) for r in compat['legacy_quality_missing_START_jobs']} if compat else set()
    assert qc-qs==exceptions and len(exceptions)==1

    idx=rows(root/'post_index.csv');assert len(idx)==144
    assert collections.Counter(r['method'] for r in idx)==dict(IDENTITY_GEO=48,D0_GEO=48,STUDENT_GEO=48)
    quality=rows(root/'quality_per_output.csv');assert len(quality)==144
    qmap={(r['method'],r['label'],r['request_id']):r for r in quality}
    checks=thresholds=0
    for r in idx:
        d=C.read(r['diagnostic_path']);a=arrays(r['post_path']);k=int(r['K'])
        raw=arrays(r['raw_path'])['raw'].reshape(-1,3,3)
        assert bits(a['output'][:k],raw[:k]) and bits(a['vertices'][a['faces']],a['output'])
        g=independent_graph(a,k,C.read(d['quality']['pair_audit_path']))
        for field,val in g.items():
            if val is None:assert d['topology'][field] is None
            else:assert near(val,d['topology'][field]),(field,val,d['topology'][field])
        qr=qmap[(r['method'],r['label'],r['request_id'])]
        for field,val in d['quality']['metrics'].items():
            if val is None:assert qr[field]==''
            elif isinstance(val,(int,float)):assert near(qr[field],val)
            else:assert qr[field]==str(val)
        # Independent boundary IDs/counts, using source connectivity only.
        src=arrays(r['condition_source']);source_ids=src['C_source_vertex_ids'];edges=collections.Counter();points={}
        for face,ids in zip(src['C'],source_ids):
            for q in range(3):
                points[int(ids[q])]=face[q].astype(np.float64)
                edges[tuple(sorted((int(ids[q]),int(ids[(q+1)%3]))))]+=1
        ids=sorted({i for e,n in edges.items() if n==1 for i in e})
        for stage,mk,vk in (('POST','metrics','vertices'),('PRE','pre_GEO_metrics','pre_GEO_vertices')):
            tri=a['output'] if stage=='POST' else arrays(r['denoised_path'])[r['pre_GEO_key']].reshape(-1,3,3)
            fv=tri[k:].astype(np.float64).reshape(-1,3)
            dist=[float(np.linalg.norm(fv-points[i],axis=1).min()) if len(fv) else np.inf for i in ids]
            assert ids==[int(v['source_vertex_id']) for v in d[vk]]
            for x,v in zip(dist,d[vk]):assert (not np.isfinite(x) and v['distance_model'] is None) or near(x,v['distance_model'])
            mm=[x for x in d[mk] if x['domain']=='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY'];assert len(mm)==8
            for m in mm:
                matched=sum(x<float(m['threshold_model']) for x in dist)
                assert matched==m['matched_count'] and len(ids)==m['boundary_vertex_count']
                if ids:assert near(m['VMR'],matched/len(ids)) and m['perfect']==int(matched==len(ids))
                thresholds+=1
        checks+=1
    vm=rows(root/'per_output.csv');assert len(vm)==2304
    sums=rows(root/'quality_summary.csv')
    for summary in sums:
        group=[r for r in quality if r['method']==summary['method'] and r['label']==summary['label']]
        for field in ('parent_id','candidate','region_type'):
            if summary.get(field):group=[r for r in group if r[field]==summary[field]]
        assert len(group)==int(summary['outputs'])
        for field in summary:
            if not field.endswith('_mean'):continue
            src=field[:-5];vals=[float(r[src]) for r in group if r.get(src,'')!='']
            assert int(summary[src+'_valid_n'])==len(vals)
            if vals:assert near(summary[field],np.mean(vals))
    tmap={(r['method'],r['label'],r['request_id']):r for r in rows(root/'topology_per_output.csv')}
    paired=rows(root/'paired_per_output.csv');assert len(paired)==144
    for r in paired:
        newer,older=r['comparison'].split('_minus_')
        nk=(newer,r['label'],r['request_id']);ok=(older,r['label'],r['request_id'])
        new=dict(qmap[nk],**tmap[nk]);old=dict(qmap[ok],**tmap[ok])
        for field,value in r.items():
            if not field.endswith('_delta'):continue
            source=field[:-6]
            if new[source]=='' or old[source]=='':assert value==''
            else:assert near(value,float(new[source])-float(old[source]))
    vmap={(r['method'],r['label'],r['request_id'],r['stage'],r['setting']):r for r in vm}
    vp=rows(root/'AVMR_paired_per_output.csv');assert len(vp)==2304
    for r in vp:
        newer,older=r['comparison'].split('_minus_');tail=(r['label'],r['request_id'],r['stage'],r['setting'])
        for field in ('VMR','perfect'):
            a,b=vmap[(newer,*tail)][field],vmap[(older,*tail)][field]
            if a=='' or b=='':assert r[field+'_delta']==''
            else:assert near(r[field+'_delta'],float(a)-float(b))
    return dict(status='PASS',outputs=checks,independent_union_find_graphs=checks,independent_boundary_threshold_checks=thresholds,
        paired_quality_rows=len(paired),paired_AVMR_rows=len(vp),quality_aggregate_arithmetic=True,original_quality_detector_reexecutions=0,source_reconstruction=0,
        actual_source_protected_files=len(er['protected']),GEO_started_complete=[len(gs),len(gc)],quality_started_complete=[len(qs),len(qc)],
        disclosed_missing_CPU_START_lines=len(exceptions),quality_reexecutions=0,compatibility_provenance_verified=True)


def fidelity():
    path=C.OUT/'evaluation/distillation_per_output.csv'
    if not path.exists():return dict(status='PENDING',reason='Saved fidelity table not complete')
    rr=rows(path);assert len(rr)==66
    for r in rr:
        t,s=arrays(r['teacher_path']),arrays(r['student_path']);k=int(r['K']);n=int(r['N'])
        assert bits(t['input_d'],s['input_d'])
        rt=np.subtract(t['proposal_T_d'],t['input_d'],dtype=np.float32)[k:]
        rs=np.subtract(s['proposal_S_d'],s['input_d'],dtype=np.float32)[k:]
        err=np.subtract(rs,rt,dtype=np.float32).astype(np.float64)
        energy=float(np.sum(rt.astype(np.float64)**2));sse=float(np.sum(err**2));den=9*(n-k)
        assert near(r['E'],sse/den) and near(r['B'],energy/den) and int(r['free_scalar_count'])==den
        low=energy/den<=1e-14
        if low:assert r['R']=='' and r['residual_norm_ratio']=='' and r['residual_cosine']=='' and r['signal_status']=='LOW_SIGNAL'
        else:assert near(r['R'],sse/energy)
        assert bits(s['output'][:k],s['C']) and bits(s['r_S_d'],np.subtract(s['proposal_S_d'],s['input_d'],dtype=np.float32))
    summaries=rows(C.OUT/'evaluation/distillation_summary.csv')
    for s in summaries:
        group=[r for r in rr if all(not s.get(k) or s[k]==r.get(k) for k in ('phase','label','uid','candidate','variant'))]
        assert len(group)==int(s['outputs'])
        den=sum(int(r['free_scalar_count']) for r in group);err=sum(float(r['error_SSE']) for r in group);en=sum(float(r['teacher_energy_sum']) for r in group)
        assert near(s['pooled_E'],err/den) and near(s['pooled_B'],en/den)
        if en/den>1e-14:assert near(s['pooled_R'],err/en)
    return dict(status='PASS',outputs=66,independent_effective_residual_arithmetic=True,weighted_denominators_verified=True,
        LOW_SIGNAL_policy_verified=True,C_original_bits=True,model_calls=0)



def training():
    done=C.OUT/'training/completion.json'
    if not done.exists():return dict(status='PENDING',reason='Formal training incomplete')
    result=C.read(done);assert result['status']=='COMPLETE'
    assert (result['updates'],result['train_forwards'],result['train_backwards'],result['diagnostic_forwards'])==(4000,32000,32000,165)
    assert result['monitor_steps']==[0,500,1000,2000,4000] and not result['teacher_loaded'] and result['no_rng_consumed']
    init=C.read(C.OUT/'training/initialization.json');initial=C.read(C.OUT/'preflight/student_initial_fp32.json')
    assert init['fresh_optimizer'] and init['initial_file_sha256']==initial['sha256'] and init['initial_state_sha256']==initial['state_sha256']
    assert init['source_sha256']==C.sha(C.CODE/'train.py')
    curves=rows(C.OUT/'training/learning_curve.csv');assert len(curves)==4000
    manifest=C.read(C.OUT/'manifest/data.json');stream=C.read(manifest['train_stream_path'])['batches']
    for step,(curve,batch) in enumerate(zip(curves,stream),1):
        assert int(curve['step'])==step and int(curve['free_scalars'])==batch['free_scalar_denominator']
        lr=1e-6+(1e-4-1e-6)*(step-1)/99 if step<=100 else 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*(step-100)/3900))
        assert near(curve['lr'],lr)
        den=int(curve['real_scalars'])+int(curve['synthetic_scalars']);assert den==int(curve['free_scalars'])
        assert near(curve['loss'],(float(curve['real_E'])*int(curve['real_scalars'])+float(curve['synthetic_E'])*int(curve['synthetic_scalars']))/den)
        assert all(math.isfinite(float(curve[k])) for k in ('loss','gradient_norm','real_E','real_B','synthetic_E','synthetic_B'))
    byid={r['sample_id']:r for r in manifest['train_inputs']+manifest['internal_val']}
    expected={r['sample_id'] for r in manifest['internal_val']}|set(manifest['monitor_ids'])
    assert len(expected)==33
    monitor_checks=0
    for step in result['monitor_steps']:
        rr=rows(C.OUT/'monitor'/f'step{step:04d}.csv');assert {r['sample_id'] for r in rr}==expected and len(rr)==33
        for row in rr:
            sid=row['sample_id'];teacher=arrays(byid[sid]['teacher_cache_path']);student=arrays(C.OUT/'monitor'/f'step{step:04d}'/(sid+'.npz'))
            k=int(byid[sid]['K']);x=teacher['input_d'];rt=np.subtract(teacher['proposal_T_d'],x,dtype=np.float32)[k:]
            rs=np.subtract(student['proposal_S_d'],x,dtype=np.float32)[k:];err=np.subtract(rs,rt,dtype=np.float32).astype(np.float64)
            den=9*(len(x)-k);en=float(np.sum(rt.astype(np.float64)**2));sse=float(np.sum(err**2))
            assert near(row['E'],sse/den) and near(row['B'],en/den)
            if en/den<=1e-14:assert row['R']==''
            else:assert near(row['R'],sse/en)
            if step==0:assert not np.count_nonzero(rs)
            monitor_checks+=1
    return dict(status='PASS',updates=4000,train_forwards=32000,train_backwards=32000,diagnostic_forwards=165,
        monitor_saved_arithmetic_checks=monitor_checks,learning_rate_steps_verified=4000,initial_zero_effective_residual33=True,
        initial_full_state_matches_preflight=True,fresh_AdamW=True,no_online_teacher=True,no_RNG_consumed=True)

def checkpoints():
    path=C.OUT/'checkpoints/student_inference.pt'
    if not path.exists():return dict(status='PENDING',reason='Terminal student export not present')
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    if available<8<<30:return dict(status='PENDING',reason='Less than 8GiB available RAM for optional CPU mmap audit')
    import torch
    torch.set_num_threads(2)
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True);rec=C.read(path.with_suffix('.json'))
    assert C.sha(path)==rec['sha256'] and p['step']==4000 and p['student']==C.STUDENT
    state=p['model'];n=sum(t.numel() for t in state.values());assert n<=50_000_000
    assert all(t.dtype==torch.float32 and torch.isfinite(t).all().item() for t in state.values())
    h=hashlib.sha256()
    for key,t in sorted(state.items()):
        a=t.detach().contiguous().numpy();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(a.tobytes())
    assert h.hexdigest()==p['state_sha256']==rec['state_sha256']
    assert not any('teacher' in k or 'e_known' in k for k in state)
    fullpath=C.OUT/'checkpoints/student_final4000.pt';fullrec=C.read(fullpath.with_suffix('.json'))
    assert C.sha(fullpath)==fullrec['sha256'] and fullrec['optimizer_steps']==[4000]
    full=torch.load(fullpath,map_location='cpu',weights_only=True,mmap=True)
    assert full['step']==4000 and full['state_sha256']==p['state_sha256'] and full['stream_sha256']==C.sha(C.OUT/'manifest/train_stream.json')
    assert set(full['model'])==set(state)
    for key,t in state.items():assert torch.equal(t,full['model'][key])
    opt=full['optimizer'];assert len(opt['state'])==104
    for item in opt['state'].values():
        assert int(item['step'])==4000
        for name,value in item.items():
            if torch.is_tensor(value):assert torch.isfinite(value).all().item()
    for group in opt['param_groups']:
        assert tuple(group['betas'])==(.9,.95) and group['weight_decay']==0 and near(group['lr'],1e-5)
    assert full['monitor_steps']==[0,500,1000,2000,4000]
    return dict(status='PASS',state_tensors=len(state),parameters=n,state_sha256=h.hexdigest(),
        full_and_inference_exact=True,AdamW_state_entries=len(opt['state']),AdamW_every_step=4000,optimizer_finite=True,
        all_parameters_FP32_finite=True,export_file_bytes=path.stat().st_size,teacher_embedded=False,
        CPU_mmap=True,model_instances_created=0,model_forwards=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('data','budget','targets','preflight','training','evaluation','fidelity','checkpoints','final'),required=True)
    args=parser.parse_args();start=time.perf_counter();C.write(OUT/(args.stage+'_resources_before.json'),C.resources())
    if args.stage=='final':
        results={name:globals()[name]() for name in ('data','targets','preflight','training','evaluation','fidelity','checkpoints')}
        results['budget']=budgets(final=True)
        registration=C.read(C.OUT/'registration.json');C.verify(registration['protected'])
        results['protected']=dict(status='PASS',files=len(registration['protected']))
        status='PASS' if all(v['status']=='PASS' for v in results.values()) else 'PENDING'
        report=dict(status=status,results=results,model_calls=0,seconds=time.perf_counter()-start,source_sha256=C.sha(__file__))
    else:
        result=budgets() if args.stage=='budget' else globals()[args.stage]()
        report=dict(result);report.update(model_calls=0,seconds=time.perf_counter()-start,source_sha256=C.sha(__file__))
    C.write(OUT/(args.stage+'_REPORT.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','parent_domain_exposure','slot_domain_exposure','pending_input_ids')},indent=2))


if __name__=='__main__':main()
