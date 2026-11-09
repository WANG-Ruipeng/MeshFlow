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
OUT=BASE/'experiments/r2_postprocess_avmr_pmr_v1'
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


def main():
    tick=time.perf_counter();records=csv_read(OUT/'post_index.csv')
    check(len(records)==48 and len({(r['label'],r['request_id']) for r in records})==48,'Unique48 outputs')
    registration=read(OUT/'registration.json');denreg=read(OUT/'denoise_registration.json');denregsha=sha(OUT/'denoise_registration.json')
    check(denreg['coordinate_scale']==.95*.3762,'Fixed denoise scale')
    check(denreg['budget']['denoiser_forwards']==48,'Denoise budget')
    check(denreg['source_sha256']==sha(BASE/'r2/tools/postprocess_avmr_pmr_v1/denoise_run.py'),'Denoise source frozen')
    oldprot=read(OLD/'registration.json')['protected'];actual={p:sha(p) for p in oldprot};changed=[p for p,h in oldprot.items() if actual[p]!=h]
    check(not changed,'Protected77 final hash mismatch')
    incident=read(OUT/'engineering_registration_path_incident/RESTORATION.json')
    check(incident['status']=='RESTORED_BYTE_IDENTICAL','Incident restoration status')
    check(sha(incident['target'])==incident['original_expected_sha256'],'Restoration final actual hash')
    identities=[];values=[];topologies=[];distances=[];comparisons=[];pending=[]
    for r in records:
        label,rid=r['label'],r['request_id'];K,N=int(r['K']),int(r['N'])
        for field in ('raw','post','input','condition_source'):
            path=r[field+'_path'] if field!='condition_source' else r[field]
            expected=r[field+'_sha256']
            check(sha(path)==expected,f'{rid} {field} hash')
        raw=arrays(r['raw_path']);inp=arrays(r['input_path']);condition=arrays(r['condition_source']);post=arrays(r['post_path']);den=arrays(r['denoised_path'])
        rt=raw['raw'].reshape(N,3,3);pt=post['output'];C=inp['C'];F=pt[K:]
        check(C.dtype==rt.dtype==pt.dtype==np.float32,'FP32 geometry')
        check(all(same(a,C) for a in (raw['C'].reshape(K,3,3),rt[:K],condition['C'],den['C'].reshape(K,3,3),den['output'][:K],post['C'],pt[:K])),'All C original bits')
        check(np.isfinite(pt).all() and np.isfinite(den['official_unconstrained_proposal']).all(),'Finite output')
        check(same(den['output'][K:],den['official_unconstrained_proposal'][K:]),'Free is unaltered inverse-scale denoise proposal')
        check(float(den['coordinate_scale'])==.95*.3762,'Saved denoise scale')
        check(same(den['displacement_from_RAW'],den['output'].astype(float)-rt.astype(float)),'Saved denoise displacement')
        dr=read(Path(r['denoised_path']).with_name('receipt.json'));pr=read(Path(r['post_path']).with_name('postprocess.json'))
        check(dr['status']=='PASS' and dr['label']==label and dr['request_id']==rid and dr['raw_sha256']==r['raw_sha256'],'Denoise raw identity')
        check(dr['denoise_registration_sha256']==denregsha and dr['checkpoint_sha256']==denreg['checkpoint_sha256'],'Denoise contract/weight identity')
        check(dr['output_sha256']==sha(r['denoised_path']),'Denoise saved hash')
        check(dr['no_rng_consumed'] and dr['C_original_bits_restored'] and dr['residual_dtype']=='torch.bfloat16','Denoise precision/RNG receipt')
        check(pr['status']=='PASS' and pr['input']['file_sha256']==dr['output_sha256'],'GEO input chain')
        check(pr['config']==registration['postprocess_config'],'Frozen GEO config')
        for file,v in pr['files'].items():check(sha(Path(r['post_path']).parent/file)==v['sha256'],'GEO artifact hash')
        check(same(post['vertices'][post['faces']],pt),'Indexed NPZ triangles')
        pv,pf=ply(r['post_ply']);check(same(pv,post['vertices']) and same(pf,post['faces']),'PLY NPZ bitwise identity')
        check(len(pt)==int(r['output_N'])==pr['output_faces'],'Actual POST face count')
        check(np.array_equal(post['source_face_ids'],np.flatnonzero(post['source_face_kept'])),'GEO kept face mapping')
        check(np.array_equal(post['source_face_ids'][:K],np.arange(K)),'C face provenance/order')
        corrected=den['output'].reshape(-1,3)[post['corner_target_input_ids']]
        check(same(corrected[post['source_face_ids']],pt),'Full GEO corner provenance')
        radius=pr['config']['radius_model'];moves=np.linalg.norm(corrected.astype(float)-den['output'].astype(float),axis=-1)
        check(np.all(moves<=radius),'GEO single-move radius')
        ids,points=boundary(C,condition['C_source_vertex_ids'])
        pairmetrics={}
        for stage,tr in [('RAW',rt),('DENOISE_C_GEO_C',pt)]:
            d,_=cKDTree(tr[K:].reshape(-1,3).astype(float)).query(points,k=1) if len(tr)>K else (np.full(len(points),np.inf),None)
            stagevals=[]
            for setting,threshold in SETTINGS:
                matched=int(np.sum(d<threshold));v=dict(label=label,request_id=rid,parent_id=r['parent_id'],region_type=r['region_type'],stage=stage,setting=setting,
                    boundary_vertex_count=len(ids),matched_count=matched,VMR=matched/len(ids),perfect=int(matched==len(ids)))
                values.append(v);stagevals.append(v)
            pairmetrics[stage]=stagevals
            if stage!='RAW':
                distances.extend(dict(label=label,request_id=rid,source_vertex_id=int(i),distance_model=float(x)) for i,x in zip(ids,d))
        rv,rf=exact_raw_index(rt);ts={'RAW_EXACT_COORD_INDEX':topo(rv,rf,K),'POST_INDEXED':topo(post['vertices'],post['faces'],K)}
        for stage,t in ts.items():topologies.append(dict(label=label,request_id=rid,parent_id=r['parent_id'],region_type=r['region_type'],stage=stage,**t))
        sf=OUT/'scores'/label/rid/'score.json'
        if sf.exists():
            saved=read(sf);check(saved['status']=='PASS','Root score PASS')
            check(saved['identity']['raw_sha256']==r['raw_sha256'] and saved['identity']['post_sha256']==r['post_sha256'],'Root score identity')
            for j,v in enumerate(saved['metrics']):
                expected=pairmetrics['DENOISE_C_GEO_C'][j]
                for key in ('matched_count','boundary_vertex_count','VMR','perfect'):check(num_eq(v[key],expected[key]),'Boundary score mismatch '+key)
            mdiff=0.
            for a,b in zip(saved['vertices'],distances[-len(ids):]):
                check(int(a['source_vertex_id'])==b['source_vertex_id'],'Per-vertex source identity');mdiff=max(mdiff,abs(a['distance_model']-b['distance_model']))
            check(mdiff<=1e-12,'cKDTree distance mismatch')
            for stage,t in ts.items():
                rootvals=saved['topology_'+stage]
                for k,v in t.items():check(num_eq(rootvals[k],v),'Topology mismatch '+stage+' '+k)
            check(saved['quality']['free_face_denominator']==len(F),'Quality actualfree denominator')
            check(saved['quality']['C_query_count']==2048 and saved['quality']['definition']['L']==L,'Quality frozen scale/query')
            comparisons.append(dict(label=label,request_id=rid,status='PASS',maximum_distance_error=mdiff))
        else:pending.append(dict(label=label,request_id=rid))
        identities.append(dict(label=label,request_id=rid,status='PASS',raw_N=N,K=K,POST_N=len(pt),free_face_denominator=len(F),boundary_vertices=len(ids),RAW_sha256=r['raw_sha256'],POST_sha256=r['post_sha256'],denoised_sha256=dr['output_sha256']))
    summaries=[]
    for stage in ('RAW','DENOISE_C_GEO_C'):
        for label in LABELS:
            for setting,_ in SETTINGS:
                for group in ('ALL','seat','back'):
                    subset=[r for r in values if r['stage']==stage and r['label']==label and r['setting']==setting and (group=='ALL' or r['region_type']==group)]
                    summaries.append(dict(stage=stage,label=label,setting=setting,group=group,outputs=len(subset),A_VMR=float(np.mean([r['VMR'] for r in subset])),PMR=float(np.mean([r['perfect'] for r in subset]))))
    summary_status='PENDING_ROOT_SUMMARY';pair_status='PENDING_ROOT_PAIRS';root_summary=OUT/'summary.csv'
    if root_summary.exists():
        actual=csv_read(root_summary);check(len(actual)==len(summaries),'Summary count')
        by={(r['stage'],r['label'],r['setting'],r['group']):r for r in summaries}
        for r in actual:
            e=by[r['stage'],r['label'],r['setting'],r['group']]
            for k in ('outputs','A_VMR','PMR'):check(num_eq(float(r[k]),e[k]),'Summary mismatch '+k)
        summary_status='PASS'
    if (OUT/'paired_RAW_POST.csv').exists():
        pair=csv_read(OUT/'paired_RAW_POST.csv');check(len(pair)==384,'Paired rows count')
        by={(r['label'],r['request_id'],r['setting'],r['stage']):r for r in values}
        for r in pair:
            a=by[r['label'],r['request_id'],r['setting'],'RAW'];b=by[r['label'],r['request_id'],r['setting'],'DENOISE_C_GEO_C']
            check(num_eq(float(r['delta_VMR']),b['VMR']-a['VMR']),'delta VMR')
            check(num_eq(float(r['delta_matched_count']),b['matched_count']-a['matched_count']),'delta matched count')
        pair_status='PASS'
    cpu_ledger=[json.loads(l) for l in (OUT/'denoise_ledger.jsonl').read_text().splitlines() if l]
    starts=[r for r in cpu_ledger if r['event']=='DENOISER_FORWARD_STARTED'];done=[r for r in cpu_ledger if r['event']=='DENOISER_FORWARD_COMPLETE']
    check(len(starts)==len(done)==48,'Exactly48 completed denoiser forward ledger')
    csv_write('per_output_independent.csv',values);csv_write('summary_independent.csv',summaries)
    csv_write('per_vertex_independent.csv',distances);csv_write('topology_independent.csv',topologies);csv_write('identity_checks.csv',identities)
    write('original_77_final_hashes.json',dict(status='PASS_FINAL_BYTES',expected=oldprot,actual=actual if isinstance(actual,dict) else {p:sha(p) for p in oldprot},differences=changed,
        incident='One parent NPZ was temporarily overwritten by erroneous registration path before any forward, then restored byte-identically from historical archive; final hashes all match. Do not claim originals were never modified.',restoration_receipt=incident))
    report=dict(status='PASS' if not pending and summary_status==pair_status=='PASS' else 'PENDING_ROOT_AGGREGATION',outputs_checked=48,
        original_protected_files=len(oldprot),original_final_hash_differences=len(changed),C_chain_bitwise_pass=48,PLY_NPZ_indexed_bitwise_pass=48,
        fixed_scale_chain_receipts_pass=48,denoiser_forward_starts=len(starts),denoiser_forward_completions=len(done),independent_metric_rows=len(values),
        independent_vertex_rows=len(distances),independent_summary_rows=len(summaries),independent_topology_rows=len(topologies),root_scores_compared=len(comparisons),
        root_score_pending=pending,root_summary_comparison=summary_status,root_paired_comparison=pair_status,
        max_root_vertex_distance_error=max((r['maximum_distance_error'] for r in comparisons),default=None),
        implementation='Independent NumPy unique source-ID edges + SciPy cKDTree; manual binary PLY parser; vectorized actual-index edge incidence and union-find. No production scorer import.',
        source_sha256=sha(__file__),model_loads=0,model_forwards=0,new_generation=0,wall_seconds=time.perf_counter()-tick,
        limitation='Verifies serialized scale/weight/runtime contracts and final geometry without rerunning network; no independent numerical model-forward equivalence claim.',
        restored_asset_incident='Acknowledged; exact original14375 bytes restored before formal denoising; final77 hashes pass.')
    write('REPORT.json',report);print(json.dumps(report))

if __name__=='__main__':main()