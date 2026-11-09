"""Natural-N Native data: exact-source upgrades, fixed C-only evaluation assets.

No target simplification, topology repair, model/torch import, or generated-result
selection. Historical data and audit receipts remain read-only.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse, collections, copy, hashlib, json, subprocess, sys, time, types
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
OUT=ROOT/'experiments/r2_bench_natural_n_recipe_v1'
DATA=OUT/'data'
OLD=ROOT/'experiments/r2_bench_seat_coverage_v1/data/manifest.json'
EXPOSURE=ROOT/'experiments/r2_bench_exposure_allocation_v1'
PREP=EXPOSURE/'data_expansion'
TASK=ROOT/'experiments/r2_bench_deployment_task_data_v1'
HIGH_IDS=('28b9f2908f0cbb179c21c139c0613648','abf250a01f526137a035c800c58595d9')
HOLD_IDS=('617fa8d6d65af35d71295682a9280b8b','756108c9b72e95aa57d85583f4ad07cf')
FOCUS_IDS=('a8e987a6a6486b962d6c8e51365a5a87','27eb79ddf26fd7ac7898580cb9d16864')
DF6='df6fd36040a5719662701a18243b45d3'
VERSION='BENCH_NATURAL_N_NATIVE_DATA_V1'
L=float(np.sqrt(3)*2/.3762)

def resolve(p):
    s=str(p).replace('\\','/')
    for prefix in ('/mnt/e/MeshFlow/','E:/MeshFlow/'):
        if s.startswith(prefix):return ROOT/s[len(prefix):]
    p=Path(s);return p if p.is_absolute() else ROOT/p
def read(p):return json.loads(resolve(p).read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with resolve(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def arrays(p):
    with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,v):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if read(p)!=v:raise ValueError('Immutable JSON differs: '+str(p))
    else:p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def save(p,v):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        a=arrays(p)
        if set(a)!=set(v) or any(not np.array_equal(a[k],v[k]) for k in v):raise ValueError('Immutable arrays differ')
    else:
        with p.open('xb') as f:np.savez_compressed(f,**v)
def protect(book,p,expected=None):
    p=resolve(p);h=sha(p)
    if expected is not None and h!=expected:raise ValueError('Source hash differs: '+str(p))
    if str(p) in book and book[str(p)]!=h:raise ValueError('Source changed during prepare')
    book[str(p)]=h;return h
def resources():
    import psutil
    def query(args):
        r=subprocess.run(args,capture_output=True,text=True)
        return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    return dict(unix=time.time(),cpu_cores=os.cpu_count(),available_RAM_bytes=psutil.virtual_memory().available,
        cpu_utilization_percent=psutil.cpu_percent(.2),load_average=list(os.getloadavg()),workers=1,numeric_threads=1,
        GPU_calls=0,gpu=query(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
        gpu_processes=query(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']))
def boundary(f):
    book=collections.defaultdict(list)
    for i,face in enumerate(f):
        for j in range(3):
            u,v=int(face[j]),int(face[(j+1)%3]);book[tuple(sorted((u,v)))].append((i,u,v))
    return np.asarray([v[0] for k,v in sorted(book.items()) if len(v)==1],np.int64).reshape(-1,3)
def verify_case(row):
    protect({},row['case_path'],row['case_sha256']);a=arrays(row['case_path']);n,k=row['N'],row['K']
    if not 128<=n<=799 or not 2<=k<n:raise ValueError('Natural N/K domain')
    assert (int(a['N']),int(a['K']))==(n,k)
    assert a['full_target'].shape==(n,3,3) and a['full_target'].dtype==np.float32
    assert np.isfinite(a['full_target']).all()
    assert np.array_equal(a['pre_ot_full']*np.float32(2),a['full_target'])
    assert np.array_equal(np.sort(np.r_[a['known_ids'],a['free_ids']]),np.arange(n))
    assert not set(a['known_ids'])&set(a['free_ids'])
    assert np.array_equal(a['C'],a['full_target'][a['known_ids']])
    assert np.array_equal(a['Y'],a['full_target'][a['free_ids']])
    assert np.array_equal(a['C_vertices'][a['C_faces_model']],a['C'])
    assert np.array_equal(a['model_vertices'][a['source_face_vertex_ids']],a['full_target'])
    assert np.array_equal(np.sort(a['C_faces_original'],1),np.sort(a['C_faces_model'],1))
    be=boundary(a['C_faces_original'])
    assert len(a['occupied_labels'])==len(be)
    return dict(parent_id=row['parent_id'],condition_id=row['condition_id'],N=n,K=k,
        boundary_edges=len(be),closed_C=len(be)==0,exact_C_Y_partition=True,half_scale_exact=True,
        source_corner_identity=True,model_input='C geometry,N only; no source-owner layout',status='PASS')

def held_case(row,protected):
    """Consolidate an existing reviewed source/partition; no geometry recomputation."""
    from r2.data_tools.bench_parts_v1.construct import partition
    for key in ('source_case','source_finite_receipt','input','paired','quality','selection','review_source'):
        protect(protected,row[key+'_path'],row[key+'_sha256'])
    a=arrays(row['source_case_path']);pair=arrays(row['paired_path']);q=read(row['quality_path'])
    assert q['mapping']=='PASS_BITWISE_C_PLUS_Y_EQUALS_SOURCE_M' and not q['reasons']
    p=partition(a,pair['known_ids'])
    for k in p:assert np.array_equal(p[k],pair[k]),k
    case={**a,**pair};edges=boundary(pair['C_faces_original']);ne=len(edges)
    occ=np.zeros(ne,bool);supported=np.zeros(ne,bool);alias=np.zeros(ne,bool)
    owner=np.full(ne,-1,np.int64);triple=np.full((ne,3),-1,np.int64)
    for it in q['source_interface_rows']:
        e=int(it['edge_id']);occ[e]=it['role']=='OWNER';alias[e]=it['role']=='ALIAS'
        supported[e]=it['role'] in ('OWNER','ALIAS');owner[e]=int(it['free_source_face_id']);triple[e]=it['source_triple']
    case.update(occupied_labels=occ,source_supported_labels=supported,source_alias_labels=alias,
        edge_owner_face=owner,edge_source_triple=triple,edge_records=edges,
        source_third_in_C=np.asarray([bool(ok and x[2] in set(pair['C_source_vertex_ids'])) for ok,x in zip(supported,triple)],bool))
    path=DATA/'conditions'/row['condition_id']/'case.npz';save(path,case)
    return path

def prepare():
    if (DATA/'COMPLETE.json').exists():return verify()
    start=time.perf_counter();resource_path=DATA/'resources_before.json'
    if not resource_path.exists():write(resource_path,resources())
    protected={};base=read(OLD);prior=read(PREP/'candidate_manifest.json');split=read(TASK/'split_audit/new_data_split.json')
    source_tasks=read(TASK/'dataset/source_tasks_pending_adapter/manifest.json')
    for p in (OLD,PREP/'candidate_manifest.json',PREP/'source_visual_review.json',TASK/'split_audit/new_data_split.json',TASK/'dataset/source_tasks_pending_adapter/manifest.json'):
        protect(protected,p)
    old=[]
    for row in base['train']:
        r=copy.deepcopy(row);r['prior_replay_group']=r['replay_group'];r['replay_group']='FOCUS' if r['parent_id'] in FOCUS_IDS else 'OLD'
        for key in ('case','input','paired'):
            if key+'_path' in r:protect(protected,r[key+'_path'],r[key+'_sha256'])
        old.append(r)
    assert len(old)==24 and collections.Counter(r['replay_group'] for r in old)=={'OLD':22,'FOCUS':2}
    additions=[];reviews=[];previews={r['parent_id']:r for r in read(PREP/'source_visual_review.json')['rows']}
    for pid in HIGH_IDS:
        prev=next(r for r in prior['candidates'] if r['parent_id']==pid);family=next(r for r in split['rows'] if r['parent_id']==pid)
        for key in ('source_case','source_finite_receipt','case','input','quality'):
            protect(protected,prev[key+'_path'],prev[key+'_sha256'])
        visual=previews[pid];protect(protected,visual['image_path'],visual['image_sha256'])
        assert visual['actually_viewed'] and visual['semantic_decision']=='ACCEPT_SOURCE_BROAD_UPPER_BOARD_ASSEMBLY'
        a=arrays(prev['case_path']);assert np.array_equal(a['full_target'],arrays(prev['source_case_path'])['full_target'])
        q=read(prev['quality_path']);assert q['geometry_status']=='PASS_PENDING_SEMANTIC_REVIEW' and not q['reasons']
        r=dict(parent_id=pid,condition_id=prev['condition_id'],N=prev['N'],K=prev['K'],replay_group='HIGH',
            part_type='SEAT_SURFACE',split='HIGH_TRAIN',eval_group='HIGH_TRAIN',report_group='HIGH_TRAIN',
            case_path=prev['case_path'],case_sha256=prev['case_sha256'],input_path=prev['input_path'],input_sha256=prev['input_sha256'],
            paired_path=prev['case_path'],paired_sha256=prev['case_sha256'],boundary_edges=prev['C_indexed_open_edges'],
            source_case_path=prev['source_case_path'],source_case_sha256=prev['source_case_sha256'],
            quality_file=prev['quality_path'],quality_sha256=prev['quality_sha256'],
            source_finite_receipt_path=prev['source_finite_receipt_path'],source_finite_receipt_sha256=prev['source_finite_receipt_sha256'],
            full_seat_review='ACCEPT',review_status='PASS',training_ready=True,native_array_compatible=True,
            eligibility='ACCEPTED_NATIVE_NATURAL_N_COMPLETE_SEAT',admission_version=VERSION,N_contract=[128,799],
            previous_block='N_EXCEEDS_256_ONLY; old source evidence unchanged',source_geometry_modified=False,
            model_input_policy='C,N,known/free task role only; no source occupancy/owner role input',
            source_labels_usage='SUPERVISION_OR_ANALYSIS_ONLY_NOT_PRODUCTION_LAYOUT',source_semantic_review=visual,
            history_scope=family,extent_description=prev['C_scope'],closed_C=False)
        additions.append(r);reviews.append(dict(parent_id=pid,condition_id=r['condition_id'],image_path=visual['image_path'],
            image_sha256=visual['image_sha256'],actually_viewed=True,view_tool='functions.view_image',reviewer='AI_PROXY_backend_data',
            nonblind=True,columns=['M','C','Y','C+Y'],views=2,decision='ACCEPT_NATIVE_NATURAL_N',
            note=visual['source_region_observation'],review_scope='Current task re-view of same authenticated image; no new geometry claim.'))
    held=[]
    for pid in HOLD_IDS:
        prev=next(r for r in source_tasks['rows'] if r['parent_id']==pid and r['recipe_id']=='R1')
        family=next(r for r in split['rows'] if r['parent_id']==pid)
        assert family['split_role']=='ADAPTATION_UNSEEN_PARENT_STAGING' and prev['semantic_decision']=='ACCEPT_SOURCE_TASK'
        path=held_case(prev,protected);protect(protected,prev['image_path'],prev['image_sha256'])
        r=dict(parent_id=pid,condition_id=prev['condition_id'],N=prev['N'],K=prev['K'],replay_group='HOLDOUT',
            part_type='SEAT_SURFACE',split='HIGH_HOLDOUT',eval_group='HIGH_HOLDOUT',report_group='HIGH_HOLDOUT',
            case_path=str(path),case_sha256=sha(path),input_path=prev['input_path'],input_sha256=prev['input_sha256'],
            paired_path=prev['paired_path'],paired_sha256=prev['paired_sha256'],boundary_edges=prev['C_open_boundary_count'],
            source_case_path=prev['source_case_path'],source_case_sha256=prev['source_case_sha256'],
            source_finite_receipt_path=prev['source_finite_receipt_path'],source_finite_receipt_sha256=prev['source_finite_receipt_sha256'],
            quality_file=prev['quality_path'],quality_sha256=prev['quality_sha256'],full_seat_review='ACCEPT',review_status='PASS',
            training_ready=False,evaluation_ready=True,eligibility='ACCEPTED_NATIVE_NATURAL_N_HOLDOUT',admission_version=VERSION,
            N_contract=[128,799],source_geometry_modified=False,extent_description=prev['proposed_region'],
            history_scope=family,closed_C=prev['C_open_boundary_count']==0,source_owner_count=prev['source_owner_count'],
            model_input_policy='C,N only; no source-owner layout',source_labels_usage='ANALYSIS_ONLY',
            required_interface='NONE_SPECIFIED',source_semantic_review=dict(path=prev['review_source_path'],sha256=prev['review_source_sha256']))
        held.append(r);reviews.append(dict(parent_id=pid,condition_id=r['condition_id'],image_path=prev['image_path'],
            image_sha256=prev['image_sha256'],actually_viewed=True,view_tool='functions.view_image',reviewer='AI_PROXY_backend_data',
            nonblind=True,columns=['M','C','Y','C+Y'],views=2,decision='ACCEPT_NATIVE_HIGH_N_HOLDOUT',
            note=prev['review_observation'],review_scope='Current task actual image view; closed cushion is kept as complete-part input, not mislabeled an open upper-surface patch.'))
    train=old+additions;assert len({r['parent_id'] for r in train})==26
    assert not ({r['parent_id'] for r in train}&set((*HOLD_IDS,DF6)))
    assert len({r['source_case_sha256'] for r in additions+held})==4
    checks=[verify_case(r) for r in train+held]
    for r in additions+held:
        fq=read(r['source_finite_receipt_path'])
        assert not fq['illegal_pairs'] and not fq['exact_zero_area_faces']
    write(DATA/'source_visual_review.json',dict(schema=VERSION+'_actual_source_review',rows=reviews,model_calls=0))
    write(DATA/'cpu_checks.json',dict(status='PASS',rows=checks,old24_preserved=True,new_train_parents=2,
        new_highN_holdout_parents=2,training_eval_parent_disjoint=True,df6_excluded=True,new_full_geometry_tests=0,
        geometry_policy='Exact source/partition and authenticated prior finite-pair audits; not a new whole-source audit.',model_calls=0))
    manifest=dict(schema=VERSION,status='READY_NATIVE_DATA',train=train,dev=base['dev'],high_holdout=held,
        train_old22=[r for r in train if r['replay_group']=='OLD'],train_focus2=[r for r in train if r['replay_group']=='FOCUS'],
        train_new2=additions,counts=dict(train_parents=26,conditions=26,old=22,focus=2,high=2,high_holdout=2),
        N_contract=[128,799],train_N_range=[min(r['N'] for r in train),max(r['N'] for r in train)],
        evaluation_N_range=[min(r['N'] for r in held),max(r['N'] for r in held)],protected=protected,
        source_manifest=str(OLD),source_manifest_sha256=sha(OLD),model_input_policy='Native C,N only; source owner labels never configure input',
        experiment_scope='Dataset/recipe bundle comparison; highN and parent/structure coverage change together',
        exclusions=['No simplification','No source repair','No random C cutting','No other highN parent','No df6 training','No SEAM/head'],
        family_limit='Inherited visual-family audit, not proof of exhaustive genealogy/pretraining independence.',model_calls=0)
    write(DATA/'manifest.json',manifest)
    receipt=dict(status='COMPLETE',schema=VERSION,seconds=time.perf_counter()-start,workers=1,numeric_threads=1,
        actual_train_parents=26,new_train_parents=2,new_high_holdout_parents=2,new_consolidated_cases=2,
        full_target_mutations=0,new_full_geometry_tests=0,model_calls=0,GPU_calls=0,training_updates=0,
        files={str(p):sha(p) for p in (DATA/'manifest.json',DATA/'cpu_checks.json',DATA/'source_visual_review.json',resource_path)})
    for p,h in protected.items():assert sha(p)==h
    write(DATA/'COMPLETE.json',receipt);return receipt

def prepare_panel():
    if (DATA/'eval_candidates.json').exists():return verify_panel()
    m=read(DATA/'manifest.json');protected={}
    for p in (DATA/'manifest.json',DATA/'source_visual_review.json',EXPOSURE/'data/eval_manifest.json',Path(__file__)):
        protect(protected,p)
    old=read(EXPOSURE/'data/eval_manifest.json');rows=copy.deepcopy(old['requests']);assert len(rows)==30
    # Only bind the existing pure CPU bundle constructor to this output root;
    # no global mutation of legacy modules and no old output writes.
    from r2.pilots.bench_full_seat_curriculum_v1 import panel as P
    fn=types.FunctionType(P.new_bundle.__code__,dict(vars(P),OUT=OUT),P.new_bundle.__name__,P.new_bundle.__defaults__)
    new=m['train_new2']+m['high_holdout'];conditions=[]
    for r in new:
        row=dict(r,panel_origin='NATURAL_N_FIXED_SOURCE_INPUT',baseline_reuse=False,
            planned_current_training_C=r['replay_group']=='HIGH')
        base=fn(row,protected);conditions.append(base)
        for ni in (0,1):
            cid=row['condition_id'];rid=cid+'__n'+str(ni)
            seed=int.from_bytes(hashlib.sha256((VERSION+'|EVAL|'+cid+'|'+str(ni)).encode()).digest()[:4],'little')
            z=np.random.RandomState(seed).standard_normal((row['N']-row['K'],3,3)).astype(np.float32)
            dest=OUT/'evaluation_inputs/requests'/rid/'noise.npz';save(dest,dict(z=z,seed=np.asarray(seed,np.int64)))
            h=protect(protected,dest);rows.append(dict(base,request_id=rid,noise_index=ni,seed=seed,noise_npz=str(dest),noise_sha256=h,noise_npz_sha256=h))
    assert rows[:30]==old['requests'];assert len({r['request_id'] for r in rows})==38
    assert 'torch' not in sys.modules
    candidate=dict(status='CPU_READY_FOR_EVALUATION_FREEZE',schema=VERSION+'_panel_candidates',requests=rows,
        new_conditions=conditions,blocked_rows=[],protected=protected,new_noise_arrays=8,new_query_arrays=4,
        old_requests_exact_json_equal=True,original_manifest=str(EXPOSURE/'data/eval_manifest.json'),
        original_manifest_sha256=sha(EXPOSURE/'data/eval_manifest.json'),model_calls=0,scoring_calls=0,
        seed_rule=VERSION+'|EVAL|condition_id|noise_index; sha256[:4]little; NumPy RandomState standard_normal FP32')
    write(DATA/'eval_candidates.json',candidate);return verify_panel()

def verify_panel():
    from meshflow_control.source_winding import validate_source
    m=read(DATA/'eval_candidates.json');old=read(EXPOSURE/'data/eval_manifest.json')
    assert m['requests'][:30]==old['requests']
    for p,h in m['protected'].items():assert sha(p)==h
    for q in m['requests']:
        for key,hk in (('inference_npz','inference_npz_sha256'),('metadata_npz','metadata_sha256'),('source_npz','source_npz_sha256'),('noise_npz','noise_sha256')):
            assert sha(q[key])==q[hk]
        inp=arrays(q['inference_npz']);z=arrays(q['noise_npz'])['z']
        assert set(inp)=={'C_vertices','C_faces_original','C_faces_model','N','K'}
        assert z.shape==(q['N']-q['K'],3,3) and z.dtype==np.float32 and np.isfinite(z).all()
        validate_source(arrays(q['source_npz']),inp['C_vertices'][inp['C_faces_model']])
        bank=arrays(q['query_descriptor']['path']);assert sha(q['query_descriptor']['path'])==q['query_descriptor']['sha256']
        assert len(bank['C_points'])==2048 and np.isclose(bank['C_weights'].sum(),1)
    receipt=dict(status='PASS',requests=38,inherited30_exact=True,new_conditions=4,new_noise_arrays=8,
        groups=dict(collections.Counter(q['report_group'] for q in m['requests'])),model_calls=0,
        query_definitions_unchanged=True,eval_candidates_sha256=sha(DATA/'eval_candidates.json'))
    write(DATA/'eval_cpu_checks.json',receipt);return receipt

def verify():
    done=read(DATA/'COMPLETE.json')
    for p,h in done['files'].items():assert sha(p)==h
    m=read(DATA/'manifest.json')
    for p,h in m['protected'].items():assert sha(p)==h
    for r in m['train']+m['high_holdout']:verify_case(r)
    return dict(status='PASS',train=26,high_train=2,high_holdout=2,model_calls=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','panel','verify'));a=p.parse_args()
    print(json.dumps({'prepare':prepare,'panel':prepare_panel,'verify':verify}[a.action](),ensure_ascii=False))
