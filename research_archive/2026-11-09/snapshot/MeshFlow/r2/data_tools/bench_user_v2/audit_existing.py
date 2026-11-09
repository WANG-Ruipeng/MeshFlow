"""CPU-only audit of existing Bench sources; no cleanup, models, or scoring.

Reuses exact-equality definitions from the frozen 20261021 source audit.
New 44 conditions are computed; old 277 condition rows are reused verbatim.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,csv,hashlib,importlib.util,itertools,json,subprocess,time
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_bench_user_data_v2/audit'
MAN=ROOT/'experiments/r2_bench_coverage_expansion_v1/data/manifest.json'
OLD=ROOT/'experiments/r2_bench_data_chain_audit_20261021/geometry'
spec=importlib.util.spec_from_file_location('_bench_frozen_exact_audit',OLD/'audit_geometry.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
PROTECTED={};CHECKS=collections.Counter()

def check(ok,name):
    if not ok:raise ValueError(name)
    CHECKS[name]+=1

def protect(path,expected=None):
    path=A.resolve(path);digest=A.sha(path)
    check(expected is None or digest==expected,'input_registered_hash')
    PROTECTED[str(path)]=digest
    return path

def arrays(path,expected=None):
    with np.load(protect(path,expected),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def read(path):return json.loads(protect(path).read_text('utf-8-sig'))
def jsonwrite(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def csvwrite(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def resources():
    def cmd(args):
        r=subprocess.run(args,capture_output=True,text=True)
        return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    mem={k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemTotal','MemAvailable')}
    return dict(cpu_logical=os.cpu_count(),load_average=list(os.getloadavg()),memory=mem,
        gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']),
        CPU_workers=1,numeric_threads=1,GPU_compute=False)

def duplicate_groups(keys,oriented):
    book=collections.defaultdict(list)
    for i,key in enumerate(keys):book[key].append(i)
    groups=[]
    for ids in book.values():
        if len(ids)<2:continue
        pairs=[dict(face_a=a,face_b=b,winding=A.winding(tuple(oriented[a]),tuple(oriented[b]))) for a,b in itertools.combinations(ids,2)]
        groups.append(dict(source_face_ids=ids,excess_faces=len(ids)-1,pairs=pairs))
    return groups

def parent_audit(row,s):
    f=s['source_faces'];orig=s['model_vertices'][f];canon=s['full_target'];ids=s['source_face_vertex_ids'];p=s['source_corner_permutations']
    check(np.array_equal(np.take_along_axis(f,p,axis=1),ids),'source_corner_permutation_maps_original_ids')
    check(np.array_equal(s['model_vertices'][ids],canon),'canonical_matches_source')
    check(np.array_equal(s['pre_ot_full']*np.float32(2),canon),'pre_OT_half_scale')
    oriented=[tuple(tuple(map(float,v)) for v in t) for t in orig]
    ik=A.id_keys(f);gk=A.geom_keys(orig);d=A.dup_metrics(ik,f);gd=A.dup_metrics(gk,oriented)
    zeros={name:np.flatnonzero(A.area2(t)==0).tolist() for name,t in [('source64',s['source_vertices'][f]),('model64',orig),('canonical64',canon)]}
    metrics=dict(parent_id=row['parent_id'],group='TRAIN',official_split=row.get('official_split','NA'),N=len(f),
        source_case_path=row['source_case_path'],source_case_sha256=row['source_case_sha256'],
        exact_zero_area_faces_source64=len(zeros['source64']),exact_zero_area_faces_model64=len(zeros['model64']),exact_zero_area_faces_canonical64=len(zeros['canonical64']),
        full_selfintersection='NOT_RUN',**A.prefix(d,'source_ID_'),**A.prefix(gd,'exact_geometry_'))
    detailed=dict(**metrics,source_array_fields={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in s.items()},
        zero_area_source_face_ids=zeros,source_ID_duplicate_groups=duplicate_groups(ik,f),exact_geometry_duplicate_groups=duplicate_groups(gk,oriented),
        definitions='Same-ID and exact stored model coordinates are separate; source winding is not author intent; source face order is unchanged.')
    return metrics,detailed

def condition_audit(r,s):
    c=arrays(r['case_path'],r['case_sha256']);pair=arrays(r['paired_path'],r['paired_sha256'])
    known=c['known_ids'];free=c['free_ids'];full=c['full_target'];f=s['source_faces'];orig=s['model_vertices'][f]
    check(np.array_equal(full,s['full_target']),'current_target_matches_untouched_source')
    check(np.array_equal(np.sort(np.r_[known,free]),np.arange(len(full))),'C_Y_source_face_partition')
    check(np.array_equal(c['C'],full[known]) and np.array_equal(c['Y'],full[free]),'C_Y_exact_target_partition')
    for key in ('C','Y','known_ids','free_ids','source_face_ids','free_source_face_ids','C_source_faces_original','C_source_corner_permutations'):
        check(np.array_equal(c[key],pair[key]),'independent_pair_'+key)
    check(np.array_equal(known,c['source_face_ids']) and np.array_equal(free,c['free_source_face_ids']),'face_ID_identity_mapping')
    check(np.array_equal(c['C_source_faces_original'],f[known]),'C_original_winding_matches_source')
    check(np.array_equal(c['C_vertices'][c['C_faces_model']],c['C']),'C_indexed_canonical_matches_target')
    check(np.array_equal(c['C_vertices'][c['C_faces_original']],orig[known]),'C_indexed_original_matches_source')
    yc=A.id_keys(f[free]);cc=A.id_keys(f[known]);yg=A.geom_keys(orig[free]);cg=A.geom_keys(orig[known])
    ygo=[tuple(tuple(map(float,p)) for p in t) for t in orig[free]]
    yd=A.dup_metrics(yc,f[free]);ygd=A.dup_metrics(yg,ygo);ygmap=collections.defaultdict(list)
    for idx,key in zip(free,yg):ygmap[key].append(int(idx))
    counts=collections.Counter();tid=set();tg=set();fid=set();fg=set();pairs=[]
    for ci,(si,ki,kg) in enumerate(zip(known,cc,cg)):
        for fi in ygmap.get(kg,[]):
            same_id=ki==tuple(sorted(map(int,f[fi])))
            wa=A.winding(tuple(map(int,f[si])),tuple(map(int,f[fi]))) if same_id else A.winding(tuple(tuple(map(float,p)) for p in orig[si]),tuple(tuple(map(float,p)) for p in orig[fi]))
            counts['geometry_pairs']+=1;tg.add(ci);fg.add(fi)
            if same_id:counts['ID_pairs']+=1;counts['ID_'+wa]+=1;tid.add(ci);fid.add(fi)
            else:counts['geometry_only_pairs']+=1
            pairs.append(dict(condition_id=r['condition_id'],parent_id=r['parent_id'],group=r['group'],part_type=r['part_type'],C_local_face=ci,C_source_face=int(si),Y_source_face=fi,same_source_vertex_ID_set=same_id,original_winding_relation=wa,overlap='EXACT_SAME_UNORDERED_TRIANGLE_COORDINATES',C_triangle_area_model=float(A.area2(orig[si:si+1])[0]/2)))
    ca=A.area2(orig[known])/2;ya=A.area2(orig[free])/2
    result=dict(condition_id=r['condition_id'],parent_id=r['parent_id'],group=r['group'],part_type=r['part_type'],active_current_pool=True,N=int(c['N']),K=int(c['K']),Y_faces=len(free),
        C_source_components=r.get('source_edge_components','NA'),C_exact_geometry_duplicate_excess=A.dup_metrics(cg,[tuple(tuple(map(float,p)) for p in t) for t in orig[known]])['excess_faces'],Y_exact_zero_area_model64_faces=int(np.count_nonzero(ya==0)),
        C_Y_ID_overlap_pairs=counts['ID_pairs'],C_Y_ID_overlap_C_faces=len(tid),C_Y_ID_overlap_Y_faces=len(fid),C_Y_ID_overlap_same_winding_pairs=counts['ID_SAME'],C_Y_ID_overlap_opposite_winding_pairs=counts['ID_OPPOSITE'],
        C_Y_exact_geometry_overlap_pairs=counts['geometry_pairs'],C_Y_exact_geometry_overlap_C_faces=len(tg),C_Y_exact_geometry_overlap_Y_faces=len(fg),C_Y_geometry_only_distinct_ID_overlap_pairs=counts['geometry_only_pairs'],
        C_area_model=float(ca.sum()),C_exactly_duplicated_in_Y_area_model=float(ca[list(tg)].sum()) if tg else 0.,C_exactly_duplicated_in_Y_area_fraction=float(ca[list(tg)].sum()/ca.sum()) if tg and ca.sum()>0 else 0.,
        source_reconstruction_as_success=False,full_Y_selfintersection='NOT_RUN',**A.prefix(yd,'Y_source_ID_'),**A.prefix(ygd,'Y_exact_geometry_'))
    return result,pairs

def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=True)
    run=OUT/('run_'+str(time.time_ns()));run.mkdir()
    jsonwrite(run/'started.json',dict(started_unix=start,pid=os.getpid(),resources=resources(),models=0,GPU_compute=0))
    try:
        m=read(MAN);protect(__file__);protect(OLD/'audit_geometry.py')
        for name in ('AGENTS.md','WORKSPACE.md','r2/docs/bench_data_pool_policy.md'):protect(ROOT/name)
        oldc=list(csv.DictReader(protect(OLD/'per_condition.csv').open(encoding='utf-8-sig')))
        oldp={r['parent_id']:r for r in csv.DictReader(protect(OLD/'per_parent.csv').open(encoding='utf-8-sig'))}
        rows=m['train'];new=m['new_train'];check(len(rows)==321 and len(new)==44,'frozen_current_counts')
        oldids={r['condition_id'] for r in oldc};newids={r['condition_id'] for r in new}
        check(len(newids)==44 and not newids&oldids,'new_condition_ID_disjoint')
        check({r['condition_id'] for r in rows}-newids<oldids,'old_condition_coverage')
        sources={};parents=[];details=[];byparent={}
        for r in rows:
            pid=r['parent_id']
            if pid in byparent:
                check((r['source_case_path'],r['source_case_sha256'])==(byparent[pid]['source_case_path'],byparent[pid]['source_case_sha256']),'same_parent_one_source_version')
                continue
            byparent[pid]=r;s=arrays(r['source_case_path'],r['source_case_sha256']);sources[pid]=s
            p,d=parent_audit(r,s);parents.append(p);details.append(d)
            if pid in oldp:
                for key in ('N','source_ID_excess_faces','source_ID_same_winding_pairs','source_ID_opposite_winding_pairs','exact_zero_area_faces_source64','exact_zero_area_faces_model64','exact_geometry_excess_faces'):
                    check(p[key]==int(oldp[pid][key]),'old64_parent_reuse_match_'+key)
        check(len(parents)==79 and len({r['parent_id'] for r in new})==15,'parent_counts79_15')
        cond=[];pairs=[]
        for r in new:
            c,cp=condition_audit(r,sources[r['parent_id']]);cond.append(c);pairs.extend(cp)
        allids={r['condition_id'] for r in rows};oldtrain=[r for r in oldc if r['condition_id'] in allids]
        check(len(oldtrain)==277,'reused277_conditions')
        check(set(cond[0])==set(oldtrain[0]),'new_and_old_condition_schema_equal')
        csvwrite(run/'new44_per_condition.csv',cond);csvwrite(run/'new15_per_parent.csv',[r for r in parents if r['parent_id'] not in oldp]);csvwrite(run/'all79_per_parent.csv',parents)
        csvwrite(run/'all321_per_condition.csv',oldtrain+cond)
        if pairs:csvwrite(run/'new44_C_Y_exact_overlap_pairs.csv',pairs)
        jsonwrite(run/'source_index79.json',dict(parents=details))
        for path,digest in PROTECTED.items():check(A.sha(path)==digest,'source_hash_after_unchanged')
        jsonwrite(run/'input_hashes.json',PROTECTED)
        summary=dict(status='COMPLETE',new_conditions=44,new_parents=15,all_train_conditions=321,all_train_parents=79,reused_old_condition_rows=277,
            new_conditions_with_C_Y_exact_overlap=sum(r['C_Y_exact_geometry_overlap_pairs']>0 for r in cond),new_C_Y_exact_overlap_pairs=sum(r['C_Y_exact_geometry_overlap_pairs'] for r in cond),
            new_conditions_with_Y_exact_zero_area=sum(r['Y_exact_zero_area_model64_faces']>0 for r in cond),new_conditions_with_Y_ID_duplicates=sum(r['Y_source_ID_excess_faces']>0 for r in cond),
            new_conditions_passing_two_exact_checks=sum(r['C_Y_exact_geometry_overlap_pairs']==r['Y_exact_zero_area_model64_faces']==0 for r in cond),
            full_selfintersection='NOT_RUN',partial_coplanar_overlap='NOT_RUN',repair_actions=0,checks=dict(CHECKS),wall_seconds=time.time()-start,resources=resources())
        jsonwrite(run/'summary.json',summary)
        jsonwrite(run/'ledger.json',dict(status='COMPLETE',wall_seconds=time.time()-start,pid=os.getpid(),CPU_workers=1,numeric_threads=1,model_imports=0,model_forwards=0,GPU_compute=0,training_updates=0,new_generation=0,new_random_noise=0,OT_calls=0,geometry_repairs=0,source_mutations=0,protected_input_count=len(PROTECTED),all_input_hashes_unchanged=True,NumPy_version=np.__version__))
        jsonwrite(OUT/'latest.json',dict(status='COMPLETE',run=str(run),summary_sha256=A.sha(run/'summary.json'),script_sha256=A.sha(__file__)))
        print(json.dumps(dict(run=str(run),**{k:v for k,v in summary.items() if k not in ('resources','checks')})))
    except Exception as exc:
        jsonwrite(run/'failure.json',dict(status='FAILED',exception_type=type(exc).__name__,message=str(exc),wall_seconds=time.time()-start))
        raise

if __name__=='__main__':main()
