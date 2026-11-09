"""Independent read-only final POST audit; no production scorer/model imports."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

BASE=Path('/mnt/e/MeshFlow')
OUT=BASE/'experiments/r2_multistep_denoise_v1'
PREV=BASE/'experiments/r2_postprocess_avmr_pmr_v1'
DEST=OUT/'independent_review'
OLD=BASE/'experiments/r2_avmr_pmr_offline_v1'
L=float(np.sqrt(3.)*2/.3762)
SETTINGS=[('raw_tau_1e-6',1e-6)]+[(f'L_tau_{t:g}',t*L) for t in (1e-6,1e-4,.001,.002,.005,.01,.02)]
LABELS=('R2_FROZEN','R2_CONT_OLD','R2_CONT_USER','R2_USER_MIX')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def csv_read(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def check(v,reason):
    if not v:raise AssertionError(reason)
def write(name,v):
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/name).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csv_write(name,rows):
    DEST.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (DEST/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def ply(path):
    data=Path(path).read_bytes();marker=b'end_header\n';cut=data.index(marker)+len(marker)
    header=data[:cut].decode('ascii').splitlines()
    check('format binary_little_endian 1.0' in header,'PLY format')
    nv=int(next(x for x in header if x.startswith('element vertex ')).split()[-1])
    nf=int(next(x for x in header if x.startswith('element face ')).split()[-1])
    verts=np.frombuffer(data,dtype='<f4',count=nv*3,offset=cut).reshape(nv,3).copy()
    typ=np.dtype([('n','u1'),('ids','<i4',(3,))])
    faces=np.frombuffer(data,dtype=typ,count=nf,offset=cut+nv*12)
    check(np.all(faces['n']==3),'PLY face arity')
    check(len(data)==cut+nv*12+nf*13,'PLY trailing payload')
    return verts,faces['ids'].astype(np.int64)


def boundary(C,source_ids):
    edges=source_ids[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    unique,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    ids=np.unique(unique[counts==1])
    flat=source_ids.reshape(-1);coords=C.reshape(-1,3)
    pts=[]
    for uid in ids:
        rows=coords[flat==uid]
        check(all(same(rows[0],p) for p in rows),'Source vertex consistency')
        pts.append(rows[0])
    return ids,np.asarray(pts,np.float64).reshape(-1,3)


def exact_raw_index(raw):
    vals=np.ascontiguousarray(raw.reshape(-1,3))
    uv,inverse=np.unique(vals.view(np.uint32).reshape(-1,3),axis=0,return_inverse=True)
    return uv.view(np.float32),inverse.reshape(-1,3)


def topo(vertices,faces,K):
    n=len(faces)
    directed=faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    ed,inv,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    ccounts=np.bincount(inv[:K*3],minlength=len(ed));fcounts=counts-ccounts
    cb=ccounts==1;attached=cb&(fcounts>0)
    orient=np.where(directed[:,0]<directed[:,1],1,-1)
    signed=np.bincount(inv,weights=orient,minlength=len(ed))
    length=np.linalg.norm(vertices[ed[:,0]].astype(float)-vertices[ed[:,1]].astype(float),axis=1)
    parent=np.arange(n)
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    order=np.argsort(inv,kind='stable');bounds=np.r_[0,np.cumsum(counts)]
    for k in range(len(ed)):
        owners=order[bounds[k]:bounds[k+1]]//3
        for j in owners[1:]:parent[root(int(j))]=root(int(owners[0]))
    return dict(vertices=len(vertices),faces=n,C_open_edges=int(cb.sum()),C_open_edges_with_free=int(attached.sum()),
        C_open_edge_attached_fraction=float(attached.sum()/cb.sum()) if cb.any() else None,
        C_open_length_attached_fraction=float(length[attached].sum()/length[cb].sum()) if length[cb].sum()>0 else None,
        C_free_two_face_opposite_winding_edges=int((attached&(counts==2)&(signed==0)).sum()),
        boundary_edges=int((counts==1).sum()),nonmanifold_edges=int((counts>2).sum()),
        all_edges_two_faces=bool(len(ed)>0 and np.all(counts==2)),shared_edge_components=len({root(i) for i in range(n)}))


def num_eq(a,b,tol=1e-12):
    if a is None or b is None:return a is b
    if isinstance(a,bool):return bool(a)==bool(b)
    return abs(float(a)-float(b))<=tol


STAGES=('RAW','D1','D3','D5')
LABEL='R2_USER_MIX'

def main():
    tick=time.perf_counter()
    reg=read(OUT/'registration.json')
    records=reg['records'];indices=csv_read(OUT/'post_index.csv')
    check(len(records)==12 and len({r['request_id'] for r in records})==12,'Unique 12 inputs')
    check(len(indices)==48 and {(r['request_id'],r['stage']) for r in indices}=={(r['request_id'],s) for r in records for s in STAGES},'Complete 48 output index')
    old_records={r['request_id']:r for r in read(PREV/'registration.json')['records'] if r['label']==LABEL}
    allowed_extra={'D1_denoised_path','D1_denoised_sha256','D1_receipt_path','D1_post_path','D1_score_path'}
    for r in records:
        old=old_records[r['request_id']]
        check({k:r[k] for k in old}==old and set(r)-set(old)==allowed_extra,'Original 12 request identities unchanged; exact five ancestry additions')
    check(reg['settings']==read(PREV/'registration.json')['settings'],'Original 8 thresholds retained')
    changed=[];actual={}
    for path,h in reg['protected'].items():
        actual[path]=sha(path)
        if actual[path]!=h:changed.append(path)
    check(not changed,'Protected source/asset mismatch')
    oldprot=read(OLD/'registration.json')['protected']
    check(all(sha(p)==h for p,h in oldprot.items()),'All original 77 files remain unchanged')
    chains=check_chains(records,reg)
    values=[];distances=[];topologies=[];identities=[];comparison=[]
    oldidx={r['request_id']:r for r in csv_read(PREV/'post_index.csv') if r['label']==LABEL}
    for idx in indices:
        rid=idx['request_id'];stage=idx['stage'];r=old_records[rid];K,N=int(r['K']),int(r['N'])
        check(idx['label']==LABEL and idx['raw_sha256']==r['raw_sha256'],'Indexed raw identity')
        for field in ('raw','input','condition_source'):
            p=r[field+'_path'] if field!='condition_source' else r[field]
            check(sha(p)==r[field+'_sha256'],field+' hash')
        raw=arrays(r['raw_path'])['raw'].reshape(N,3,3)
        source=arrays(r['condition_source']);C=arrays(r['input_path'])['C']
        check(same(C,source['C']) and same(C,raw[:K]),'Source/raw/input C bits')
        check(sha(idx['post_path'])==idx['post_sha256'],'Final stage file hash')
        if stage=='RAW':
            tri=raw;vertices,faces=exact_raw_index(raw)
            check(idx['post_path']==r['raw_path'],'RAW path reuse')
        else:
            p=arrays(idx['post_path']);tri=p['output'];vertices,faces=p['vertices'],p['faces']
            check(tri.dtype==np.float32 and same(C,tri[:K]) and same(C,p['C']),'POST C bits/FP32')
            check(same(vertices[faces],tri),'Indexed NPZ equals triangle soup')
            pv,pf=ply(idx['post_ply']);check(same(pv,vertices) and same(pf,faces),'PLY/NPZ exact equality')
            check(np.array_equal(p['source_face_ids'],np.flatnonzero(p['source_face_kept'])),'POST face subset mapping')
            check(np.array_equal(p['source_face_ids'][:K],np.arange(K)),'C face provenance')
            den=arrays(idx['denoised_path'])['output'];pp=read(Path(idx['post_path']).with_name('postprocess.json'))
            check(pp['status']=='PASS' and pp['input']['file_sha256']==sha(idx['denoised_path']),'GEO_C denoiser ancestry')
            check(pp['config']==read(PREV/'registration.json')['postprocess_config'],'Identical GEO_C config')
            for name,detail in pp['files'].items():check(sha(Path(idx['post_path']).parent/name)==detail['sha256'],'GEO_C file identity')
            mapped=den.reshape(-1,3)[p['corner_target_input_ids']]
            check(same(mapped[p['source_face_ids']],tri),'Full corner provenance')
            check(np.all(np.linalg.norm(mapped.astype(float)-den.astype(float),axis=-1)<=pp['config']['radius_model']),'Single GEO_C displacement bound')
            if stage=='D1':check(idx['post_path']==oldidx[rid]['post_path'] and idx['post_sha256']==oldidx[rid]['post_sha256'],'D1 old file reused')
        check(np.isfinite(tri).all() and len(tri)==int(idx['output_N']),'Finite output and N')
        ids,points=boundary(C,source['C_source_vertex_ids'])
        d,_=cKDTree(tri[K:].reshape(-1,3).astype(float)).query(points,k=1) if len(tri)>K else (np.full(len(ids),np.inf),None)
        meta=dict(label=LABEL,stage=stage,request_id=rid,parent_id=r['parent_id'],region_type=r['region_type'])
        for setting,threshold in SETTINGS:
            matched=int(np.sum(d<threshold));values.append(dict(meta,setting=setting,boundary_vertex_count=len(ids),matched_count=matched,VMR=matched/len(ids),perfect=int(matched==len(ids))))
        distances.extend(dict(meta,source_vertex_id=int(i),distance_model=float(x)) for i,x in zip(ids,d))
        t=topo(vertices,faces,K);topologies.append(dict(meta,**t))
        if stage!='RAW':
            saved=read((PREV/'scores'/LABEL/rid/'score.json') if stage=='D1' else (OUT/'scores'/stage/rid/'score.json'))
            check(saved['status']=='PASS' and saved['identity']['raw_sha256']==r['raw_sha256'] and saved['identity']['post_sha256']==idx['post_sha256'],'Root score identity')
            expected={v['setting']:v for v in values[-8:]}
            for m in saved['metrics']:
                for key in ('boundary_vertex_count','matched_count','VMR','perfect'):check(num_eq(m[key],expected[m['setting']][key]),'Metric receipt mismatch '+key)
            md=0.
            for v,uid,di in zip(saved['vertices'],ids,d):
                check(int(v['source_vertex_id'])==int(uid),'Receipt per-vertex ID');md=max(md,abs(v['distance_model']-di))
            check(md<=1e-12,'Independent cKDTree difference')
            for key,val in t.items():check(num_eq(saved['topology_POST_INDEXED'][key],val),'Independent topology difference '+key)
            q=saved['quality'];check(q['free_face_denominator']==len(tri)-K and q['C_query_count']==2048 and q['definition']['L']==L,'Quality denominator/query/scale')
            comparison.append(dict(meta,maximum_distance_error=md,status='PASS'))
        identities.append(dict(meta,N_RAW=N,N_stage=len(tri),K=K,free_face_denominator=len(tri)-K,C_exact=True,raw_sha256=r['raw_sha256'],stage_sha256=idx['post_sha256'],PLY_NPZ_exact=stage!='RAW'))
    main_per=csv_read(OUT/'per_output.csv');main_dist=csv_read(OUT/'per_vertex_distances.csv');main_topo=csv_read(OUT/'topology_per_output.csv')
    check(len(main_per)==384 and len(values)==384,'384 per-output threshold metrics')
    by={(r['request_id'],r['stage'],r['setting']):r for r in values}
    for r in main_per:
        v=by[(r['request_id'],r['stage'],r['setting'])]
        for k in ('boundary_vertex_count','matched_count','VMR','perfect'):check(num_eq(float(r[k]),v[k]),'Final metric table mismatch '+k)
    byv={(r['request_id'],r['stage'],r['source_vertex_id']):r for r in distances}
    check(len(byv)==len(distances)==len(main_dist)==496,'All 496 stage-boundary distances')
    dist_err=0.
    for r in main_dist:
        v=byv[(r['request_id'],r['stage'],int(r['source_vertex_id']))];dist_err=max(dist_err,abs(float(r['distance_model'])-v['distance_model']))
    check(dist_err<=1e-12,'Final table distance mismatch')
    bytop={(r['request_id'],r['stage']):r for r in topologies}
    for r in main_topo:
        v=bytop[(r['request_id'],r['stage'])]
        for k,val in v.items():
            if k in ('label','stage','request_id','parent_id','region_type'):continue
            observed=r[k]=='True' if isinstance(val,bool) else None if r[k]=='' else float(r[k])
            check(num_eq(observed,val),'Final topology table mismatch '+k)
    summaries=[]
    for stage in STAGES:
        for setting,_ in SETTINGS:
            for group in ('ALL','seat','back'):
                sub=[r for r in values if r['stage']==stage and r['setting']==setting and (group=='ALL' or r['region_type']==group)]
                summaries.append(dict(stage=stage,setting=setting,group=group,outputs=len(sub),A_VMR=float(np.mean([r['VMR'] for r in sub])),PMR=float(np.mean([r['perfect'] for r in sub]))))
    bs={(r['stage'],r['setting'],r['group']):r for r in summaries};main_sum=csv_read(OUT/'summary.csv')
    check(len(main_sum)==len(summaries)==96,'96 macro-averaged summary rows')
    for r in main_sum:
        for k in ('outputs','A_VMR','PMR'):check(num_eq(float(r[k]),bs[(r['stage'],r['setting'],r['group'])][k]),'Summary aggregation mismatch')
    pairs=csv_read(OUT/'paired_vs_D1.csv');check(len(pairs)==288,'288 paired rows')
    for r in pairs:
        a=by[(r['request_id'],'D1',r['setting'])];b=by[(r['request_id'],r['stage'],r['setting'])]
        for col,key in [('delta_VMR','VMR'),('delta_matched_count','matched_count')]:check(num_eq(float(r[col]),b[key]-a[key]),'Pair difference mismatch')
    check(all(sha(path)==h for path,h in reg['protected'].items()),'Protected recheck after CPU audit')
    csv_write('per_output_independent.csv',values);csv_write('per_vertex_independent.csv',distances);csv_write('topology_independent.csv',topologies);csv_write('summary_independent.csv',summaries);csv_write('identity_checks.csv',identities)
    write('protected_final_hashes.json',dict(status='PASS',count=len(actual),files=actual,differences=changed,original77_all_unchanged=True))
    report=dict(status='PASS',inputs=12,stages=list(STAGES),stage_outputs=48,new_intermediate_C_exact=48,POST_C_exact=36,new_POST_PLY_NPZ_exact=24,reused_D1_PLY_NPZ_exact=12,
        metric_rows=384,vertex_rows=496,summary_rows=96,paired_rows=288,topology_rows=48,protected_files=len(actual),protected_differences=0,model_calls=0,model_loads=0,
        source_sha256=sha(__file__),wall_seconds=time.perf_counter()-tick,maximum_vertex_distance_error=dist_err,chain_audit=chains,
        implementation='Independent NumPy source-ID boundary extraction, SciPy cKDTree, manual binary PLY parser and edge-incidence union-find; no main scorer imports.',
        limitations='Receipt/source/geometry audit does not repeat neural forwards or independently prove numerical model execution equivalence. Quality metrics are identity/denominator checked, not independently reimplemented.')
    write('REPORT.json',report);print(json.dumps(report))
def check_chains(records,reg,require_geo=True):
    check(reg['new_rounds']==[2,3,4,5] and reg['GEO_terminal_rounds']==[3,5] and reg['stages']==list(STAGES),'Fixed rounds/endpoints')
    check(reg['coordinate_scale']==.95*.3762,'Fixed original coordinate scale')
    b=reg['budget'];check(b['new_denoiser_forwards']==48 and b['new_GEO_outputs']==24 and b['new_generator_forwards']==b['training_updates']==b['new_noise']==0,'Fixed budget')
    for path,h in reg['runner_hashes'].items():check(sha(path)==h,'Registered runner source identity')
    expected_weight='6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36'
    check(reg['checkpoint_sha256']==expected_weight==sha(reg['checkpoint_path']),'Original official weight identity')
    registration_sha=sha(OUT/'registration.json');rows=[]
    for r in records:
        rid=r['request_id'];K,N=int(r['K']),int(r['N'])
        raw=arrays(r['raw_path'])['raw'].reshape(N,3,3);C=raw[:K]
        prevpath=Path(r['D1_denoised_path']);one=arrays(prevpath)['output'];prev=one
        check(same(one[:K],C) and one.shape==raw.shape and sha(prevpath)==r['D1_denoised_sha256'],'Original D1 geometry/hash')
        check(prevpath==PREV/'denoised'/LABEL/rid/'denoised.npz','Original unprocessed D1 input path')
        for n in (2,3,4,5):
            path=OUT/'denoised'/f'D{n}'/rid/'denoised.npz';a=arrays(path);p=read(path.with_name('receipt.json'))
            wanted=dict(status='PASS',request_id=rid,label=LABEL,round=n,raw_sha256=r['raw_sha256'],input_path=str(prevpath),input_sha256=sha(prevpath),registration_sha256=registration_sha,checkpoint_sha256=expected_weight,N=N,K=K)
            check(all(p.get(k)==v for k,v in wanted.items()),'Actual round receipt identity/chain')
            output=a['output'];proposal=a['official_unconstrained_proposal']
            check(sha(path)==p['output_sha256'],'New round file hash')
            check(output.shape==proposal.shape==raw.shape and output.dtype==proposal.dtype==np.float32,'Round original N/FP32')
            check(same(output[:K],C) and same(a['C'],C),'Every new round exact C')
            check(same(output[K:],proposal[K:]),'F equals actual unconstrained proposal')
            check(np.isfinite(output).all() and np.isfinite(proposal).all(),'Finite round geometry')
            check(int(a['round'])==n and int(a['input_faces'])==N and int(a['known_faces'])==K and float(a['coordinate_scale'])==.95*.3762,'Round metadata/scale')
            for field,base in [('displacement_from_previous',prev),('displacement_from_RAW',raw),('displacement_from_D1',one)]:
                check(same(a[field],output.astype(float)-base.astype(float)),'Actual stored displacement '+field)
            d=np.linalg.norm((output.astype(float)-prev.astype(float))[K:],axis=-1)
            check(num_eq(float(d.max()),p['max_step_free_displacement_model']) and num_eq(float(d.mean()),p['mean_step_free_displacement_model']),'Round free displacement receipt')
            check(p['C_bitwise'] and p['no_rng_consumed'] and p['no_intermediate_GEO'] and p['residual_dtype']=='torch.bfloat16','Runtime flags')
            rows.append(dict(request_id=rid,round=n,status='PASS',input_sha256=p['input_sha256'],output_sha256=p['output_sha256'],N=N,K=K,C_exact=True,max_step_free_displacement_model=float(d.max()),mean_step_free_displacement_model=float(d.mean())))
            prevpath=path;prev=output
    log=[json.loads(l) for l in (OUT/'denoise_ledger.jsonl').read_text().splitlines() if l]
    starts=[r for r in log if r['event']=='DENOISER_FORWARD_STARTED'];done=[r for r in log if r['event']=='DENOISER_FORWARD_COMPLETE'];fails=[r for r in log if r['event']=='DENOISER_FAILED']
    keys={(r['request_id'],n) for r in records for n in (2,3,4,5)}
    check(len(starts)==len(done)==48 and not fails,'Exactly48 successful forward starts/completions')
    check({(r['request_id'],r['round']) for r in starts}=={(r['request_id'],r['round']) for r in done}==keys,'Forward identity coverage')
    last={r['request_id']:1 for r in records}
    for row in starts:
        check(row['round']==last[row['request_id']]+1,'Per-request increasing dependency order');last[row['request_id']]=row['round']
    completion=read(OUT/'denoise_completion.json')
    check(completion['status']=='PASS' and completion['total_forward_attempts']==completion['completed_round_outputs']==48,'Final denoise completion')
    check(completion['training_updates']==completion['generator_forwards']==completion['new_noise']==0,'No extra generation/training/noise')
    geolog=[json.loads(l) for l in (OUT/'score_ledger.jsonl').read_text().splitlines() if l] if (OUT/'score_ledger.jsonl').exists() else []
    gs=[r for r in geolog if r['event']=='GEO_C_STARTED'];gd=[r for r in geolog if r['event']=='GEO_C_COMPLETE']
    if require_geo:check(len(gs)==len(gd)==24 and {(r['request_id'],r['stage']) for r in gd}=={(r['request_id'],s) for r in records for s in ('D3','D5')},'Exactly24 terminal-only GEO operations')
    csv_write('round_chain_checks.csv',rows)
    return dict(status='PASS',additional_forward_starts=48,additional_forward_completions=48,failures=0,terminal_GEO_C_calls=len(gd),terminal_GEO_C_status='PASS' if require_geo else 'NOT_REQUIRED_FOR_CHAIN_ONLY',intermediate_GEO_C_calls=0,all_original_N_preserved=True,all_C_bits_preserved=True,no_RNG_consumed_receipts=True,official_checkpoint_hash_verified=True)

if __name__=='__main__':main()