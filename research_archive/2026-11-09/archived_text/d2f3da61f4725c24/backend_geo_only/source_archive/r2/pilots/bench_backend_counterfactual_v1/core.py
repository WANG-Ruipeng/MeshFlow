"""Frozen CPU-only GEO(RAW)/GEO(source) counterfactuals, with slot ledgers.

Resume: python -B -m r2.pilots.bench_backend_counterfactual_v1.core run
Complete task receipts are verified and reused. Partial attempts are never
silently replayed. No model, optimizer, OT, random draw or new D0 is used.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
OUT=ROOT/'experiments/r2_bench_backend_counterfactual_v1'
OLD=ROOT/'experiments/r2_bench_seat_coverage_v1'
VERSION='bench_backend_counterfactual_v1'
LABELS=('NATIVE_S6000','COVERAGE_FINAL')
PARENTS=('a8e987','27eb79')
L=float(np.sqrt(3.)*2/.3762)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    return value
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(clean(value),f,indent=2,ensure_ascii=False,allow_nan=False)
def npz(path,value):
    with Path(path).open('xb') as f:np.savez_compressed(f,**value)
def verify(files):
    for p,h in files.items():
        if sha(p)!=h:raise RuntimeError('Protected file changed: '+p)
def log(event,**value):
    p=OUT/'ledger.jsonl'
    with p.open('a',encoding='utf-8') as f:
        f.write(json.dumps(clean(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**value)),ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())
def csv_write(path,records):
    cols=list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(records)
def dependencies():
    from r2.pilots.seam_slot_c200_v1 import postprocess as P
    from meshflow_control import geo_c,geo_io,source_winding
    paths=[Path(__file__),Path(P.__file__),Path(P.G.__file__),Path(geo_c.__file__),Path(geo_io.__file__),Path(source_winding.__file__)]
    assert 'torch' not in sys.modules,'Unexpected neural framework import'
    return P,geo_c,geo_io,{str(p):sha(p) for p in paths}

def prepare():
    target=OUT/'manifest.json'
    if target.exists():
        m=read(target);verify(m['protected']);verify(m['source_hashes']);return m
    P,geo,io,code=dependencies()
    base=read(OLD/'eval_manifest.json')
    qs=sorted([q for q in base['requests'] if q['parent_id'].startswith(PARENTS)],key=lambda q:(q['parent_id'],q['noise_index']))
    assert len(qs)==4 and {q['noise_index'] for q in qs}=={0,1}
    tasks=[];protected={str(OLD/'eval_manifest.json'):sha(OLD/'eval_manifest.json')}
    for label in LABELS:
        for q in qs:
            original=OLD/'evaluation'/label/q['request_id']
            job=label+'__'+q['request_id']+'__GEO_RAW';dest=OUT/'outputs'/job
            paths=dict(raw_path=str(original/'raw/raw.npz'),proposal_path=str(original/'post/proposal.npz'),
                pre_geo_path=str(original/'post/pre_geo.npz'),base_post_path=str(original/'post/post.npz'),
                base_canonical_path=str(original/'post/post_canonical.npz'),geometry_path=str(original/'post/geometry.json'),
                raw_receipt=str(original/'raw/receipt.json'),post_receipt=str(original/'post/receipt.json'),
                proposal_receipt=str(original/'post/proposal_receipt.json'),
                raw_score_dir=str(OLD/'evaluation/scores'/label/q['request_id']/'RAW'),
                base_score_dir=str(OLD/'cpu'/label/q['request_id']/'BASE_POST'),
                raw_score_path=str(OLD/'evaluation/scores'/label/q['request_id']/'RAW/score.json'),
                base_score_path=str(OLD/'cpu'/label/q['request_id']/'BASE_POST/scores/score.json'))
            source=original/'raw/source_condition.npz'
            for p in list(paths.values())+[str(source),q['case_path']]:
                p=Path(p)
                if p.is_file():protected[str(p)]=sha(p)
                elif not p.is_dir():raise FileNotFoundError(str(p))
            tasks.append(dict(job_id=job,kind='RAW_GEO',label=label,request_id=q['request_id'],
                condition_id=q['condition_id'],parent_id=q['parent_id'],N=q['N'],K=q['K'],request=q,
                case_path=q['case_path'],source_condition_path=str(source),original=paths,output_dir=str(dest),
                new_raw_path=paths['raw_path'],new_pre_geo_path=str(dest/'pre_geo.npz'),
                new_post_path=str(dest/'post.npz'),new_geometry_path=str(dest/'geometry.json')))
    for q in (q for q in qs if q['noise_index']==0):
        template=next(x for x in tasks if x['label']==LABELS[0] and x['request_id']==q['request_id'])
        job=q['condition_id']+'__SOURCE_GEO';dest=OUT/'outputs'/job
        tasks.append(dict(job_id=job,kind='SOURCE_GEO',label='SOURCE_REFERENCE',request_id=q['condition_id']+'__source',
            condition_id=q['condition_id'],parent_id=q['parent_id'],N=q['N'],K=q['K'],request=q,
            case_path=q['case_path'],source_condition_path=template['source_condition_path'],original=template['original'],
            output_dir=str(dest),new_raw_path=str(dest/'source_raw.npz'),new_pre_geo_path=str(dest/'pre_geo.npz'),
            new_post_path=str(dest/'post.npz'),new_geometry_path=str(dest/'geometry.json'),
            scope='Two independent source reference targets; no noise/model samples or open-generation success claim'))
    m=dict(schema=VERSION,status='FROZEN',created_utc=datetime.now(timezone.utc).isoformat(),
        budget=dict(RAW_GEO=8,SOURCE_GEO=2,GEO_calls=10,source_orientation_calls=10,model_calls=0,D0=0,training=0,new_noise=0,OT=0),
        radius_model=geo.RADIUS,L=L,geo_config=geo.FIXED_CONFIG,source_hashes=code,protected=protected,tasks=tasks,
        task_order='lexical parent then noise within NATIVE_S6000, COVERAGE_FINAL, then unique two SOURCE references',
        stage_contract='RAW/D0_PROPOSAL/RESTORED_C and all-input MERGED before cleanup preserve each model own generated slot; POST only retains kept slots.',
        source_contract='C+Y canonical triangle soup, same N/K/scale; source identities carried only for reference rows; no indexed shortcut',
        limits=['Same slot tracks one output across backend stages, not semantic correspondence across models.',
          'GEO(source) tests backend compatibility with two source reference meshes, not autonomous completion.',
          'GEO(RAW) recomputes original GEO clusters from RAW; never reuses D0 clusters.',
          'No sweep or threshold tuning; cached D0 models are not invoked.'])
    OUT.mkdir(parents=True,exist_ok=True);write(target,m);log('REGISTERED',budget=m['budget'])
    return m

def source_raw(task):
    """Repack exactly source C+Y into the Native triangle-soup identity contract."""
    source=arrays(task['case_path']);template=arrays(task['original']['raw_path']);k=int(source['K']);n=int(source['N'])
    C=source['C'];Y=source['Y'];full=source['full_target']
    ids=np.concatenate((source['known_ids'],source['free_ids'])).astype(np.int64)
    assert np.array_equal(np.sort(ids),np.arange(n))
    assert np.array_equal(C,full[source['known_ids']]) and np.array_equal(Y,full[source['free_ids']])
    assert np.array_equal(C,template['raw'][:k])
    soup=np.concatenate((C,Y)).astype(np.float32);assert np.array_equal(soup,full[ids])
    raw={key:value.copy() for key,value in template.items()}
    # Preserve only C connectivity at the incoming stage. Free corners remain
    # independent even when coordinates happen to match a source vertex.
    vertices=np.concatenate((raw['C_vertices'],Y.reshape(-1,3)))
    faces=np.concatenate((raw['C_faces_model'],np.arange(len(raw['C_vertices']),len(vertices)).reshape(-1,3)))
    assert np.array_equal(vertices[faces],soup)
    raw.update(raw=soup,output=soup.copy(),C=C.copy(),free=Y.copy(),vertices=vertices,faces=faces,
        known_faces=np.asarray(k),input_faces=np.asarray(n),source_face_ids=np.arange(n,dtype=np.int64),
        raw_slot_ids=np.arange(n,dtype=np.int64),source_corner_ids=np.tile(np.arange(3),(n,1)),
        source_target_face_ids=ids,source_target_vertex_ids=source['source_face_vertex_ids'][ids],
        source_target_original_corner_ids=source['source_corner_permutations'][ids],
        identity_domain=np.asarray('SOURCE_C_THEN_Y_REFERENCE_NOT_GENERATED'))
    assert not raw['raw_seam_slot_mask'].any() and (raw['raw_anchor_ids'][k:]<0).all()
    receipt=dict(status='PASS',packing='C_then_Y_canonical_soup',N=n,K=k,
        C_equals_production_input=True,full_target_partition_exact=True,source_face_permutation_bijective=True,
        free_input_indexing='independent corner indices; no source-index shortcut',
        source_target_face_ids=ids.tolist(),source_target_case_sha256=sha(task['case_path']))
    return raw,receipt

def geometry_quantities(triangles):
    t=np.asarray(triangles,np.float64)
    edges=np.stack((np.linalg.norm(t[:,1]-t[:,0],axis=1),np.linalg.norm(t[:,2]-t[:,1],axis=1),np.linalg.norm(t[:,0]-t[:,2],axis=1)),axis=1)
    cross=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    shortest=edges.min(axis=1);longest=edges.max(axis=1)
    height=np.divide(cross,longest,out=np.zeros_like(cross),where=longest>0)
    return shortest,height,cross*.5

def merged(pre,canonical):
    ids=canonical['corner_target_input_ids'];assert ids.shape==pre.shape[:2] and ids.min()>=0 and ids.max()<pre.size//3
    corrected=pre.reshape(-1,3)[ids]
    assert np.array_equal(corrected[canonical['source_face_ids']],canonical['output'])
    return corrected

def stage_records(task,stage,soup,slot_ids,kept=None,reasons=None,source_ids=None):
    k=task['K'];n=task['N'];short,height,area=geometry_quantities(soup)
    slot_ids=np.asarray(slot_ids);free=slot_ids>=k
    metadata=dict(job_id=task['job_id'],kind=task['kind'],label=task['label'],request_id=task['request_id'],
                  condition_id=task['condition_id'],parent_id=task['parent_id'],stage=stage,N=n,K=k)
    rows=[]
    for i,slot in enumerate(slot_ids):
        rows.append(dict(metadata,input_slot_id=int(slot),is_C=bool(slot<k),
            source_target_face_id=int(source_ids[slot]) if source_ids is not None else '',
            shortest_edge=float(short[i]),minimum_altitude=float(height[i]),area=float(area[i]),
            shortest_edge_L=float(short[i]/L),minimum_altitude_L=float(height[i]/L),area_L2=float(area[i]/L**2),
            eventual_GEO_kept=bool(kept[slot]) if kept is not None else '',
            eventual_GEO_reason=str(reasons[slot]) if reasons is not None else '',
            identity_domain='SOURCE_C_THEN_Y_REFERENCE' if source_ids is not None else 'WITHIN_REQUEST_GENERATED_SLOT'))
    summary=dict(metadata,present_total_faces=len(soup),present_free_faces=int(free.sum()),free_area=float(area[free].sum()),
        free_area_L2=float(area[free].sum()/L**2),free_zero_area_faces=int((area[free]==0).sum()),
        free_shortest_edge_p50=float(np.median(short[free])) if free.any() else None,
        free_minimum_altitude_p50=float(np.median(height[free])) if free.any() else None,
        original_free_denominator=n-k,post_kept_free_faces=int(kept[k:].sum()) if kept is not None else None,
        post_deleted_free_faces=int((~kept[k:]).sum()) if kept is not None else None)
    return rows,summary

def one(task,manifest_sha):
    P,geo,io,_=dependencies();dest=Path(task['output_dir']);receipt_path=dest/'receipt.json'
    if receipt_path.exists():
        r=read(receipt_path);assert r['manifest_sha256']==manifest_sha;verify(r['files']);return r
    if dest.exists():raise RuntimeError('Unresolved CPU attempt, no silent replay: '+str(dest))
    dest.mkdir(parents=True);write(dest/'attempt.json',dict(job_id=task['job_id'],manifest_sha256=manifest_sha,GEO_budget=1,model_calls=0))
    tick=time.perf_counter();calls=0
    try:
        if task['kind']=='SOURCE_GEO':
            raw,receipt=source_raw(task);npz(task['new_raw_path'],raw);write(dest/'source_packing.json',receipt)
        else:raw=arrays(task['original']['raw_path'])
        source=arrays(task['source_condition_path']);original_hash=hashlib.sha256(raw['raw'].tobytes()).hexdigest()
        log('GEO_STARTED',job_id=task['job_id'],budget=1)
        pre,canonical,post,geometry,oriented,anchors=P.geometry_from_proposal(raw,raw['raw'].copy(),source);calls=1
        assert hashlib.sha256(raw['raw'].tobytes()).hexdigest()==original_hash
        assert np.array_equal(pre,raw['raw']) and geometry['stats']['radius_model']==geo.RADIUS
        if task['kind']=='SOURCE_GEO':
            sf,sc=post['source_face_ids'],post['source_corner_ids']
            post.update(identity_domain=np.asarray('SOURCE_C_THEN_Y_REFERENCE_NOT_GENERATED'),
                source_target_face_ids=raw['source_target_face_ids'],
                retained_source_target_face_ids=raw['source_target_face_ids'][sf],
                retained_source_target_vertex_ids=raw['source_target_vertex_ids'][sf[:,None],sc],
                retained_source_original_corner_ids=raw['source_target_original_corner_ids'][sf[:,None],sc])
        log('GEO_RETURNED',job_id=task['job_id'],calls=1,deleted=geometry['stats']['deleted_free_faces'])
        npz(dest/'pre_geo.npz',dict(output=pre,C=raw['raw'][:task['K']],known_faces=raw['known_faces'],
            raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
        npz(dest/'post_canonical.npz',canonical);npz(dest/'post.npz',post)
        io.write_indexed_ply(dest/'post.ply',post['vertices'],post['faces'])
        write(dest/'geometry.json',dict(stats=geometry['stats'],face_records=geometry['face_records']))
        write(dest/'anchor_identity.json',anchors);write(dest/'source_restoration.json',oriented['restoration_summary'])
        orient=oriented['orientation_result'];write(dest/'orientation.json',dict(summary=orient['summary'],components=orient['component_rows'],edges=orient['edge_rows']))
        face_rows=[];stage_rows=[];n=task['N'];ids=np.arange(n)
        source_ids=raw.get('source_target_face_ids') if task['kind']=='SOURCE_GEO' else None
        stages=[]
        if task['kind']=='RAW_GEO':
            old_pre=arrays(task['original']['pre_geo_path'])['output'];old_can=arrays(task['original']['base_canonical_path'])
            proposal=arrays(task['original']['proposal_path'])['official_unconstrained_proposal']
            assert np.array_equal(old_pre[task['K']:],proposal[task['K']:])
            assert np.array_equal(old_pre[:task['K']],raw['raw'][:task['K']])
            stages=[('RAW',raw['raw'],ids,old_can),('D0_PROPOSAL',proposal,ids,old_can),
                ('RESTORED_C',old_pre,ids,old_can),('D0_GEO_MERGED',merged(old_pre,old_can),ids,old_can),
                ('BASE_POST',old_can['output'],old_can['source_face_ids'],old_can),
                ('RAW_GEO_MERGED',merged(pre,canonical),ids,canonical),('RAW_GEO_POST',canonical['output'],canonical['source_face_ids'],canonical)]
        else:
            stages=[('SOURCE',raw['raw'],ids,canonical),('SOURCE_GEO_MERGED',merged(pre,canonical),ids,canonical),
                ('SOURCE_GEO_POST',canonical['output'],canonical['source_face_ids'],canonical)]
        for name,soup,slots,can in stages:
            rr,ss=stage_records(task,name,soup,slots,can['source_face_kept'],can['source_face_reason'],source_ids)
            face_rows.extend(rr);stage_rows.append(ss)
        csv_write(dest/'per_face_stages.csv',face_rows);csv_write(dest/'stage_summary.csv',stage_rows)
        assert 'torch' not in sys.modules
        value=dict(status='COMPLETE',job_id=task['job_id'],kind=task['kind'],label=task['label'],request_id=task['request_id'],
            manifest_sha256=manifest_sha,GEO_calls=1,orientation_calls=1,model_calls=0,D0_forwards=0,
            seconds=time.perf_counter()-tick,post_sha256=sha(dest/'post.npz'),raw_path=task['new_raw_path'],
            files={str(p):sha(p) for p in sorted(dest.iterdir()) if p.is_file()})
        write(receipt_path,value);return value
    except BaseException as error:
        write(dest/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),returned_GEO_calls=calls,model_calls=0,no_silent_retry=True))
        raise

def run():
    m=prepare();verify(m['protected']);verify(m['source_hashes'])
    complete=OUT/'core_complete.json'
    if complete.exists():
        r=read(complete);verify(r['files']);return r
    # Resource scheduling is owned by the root; the written receipt is required
    # before the first geometry task, not guessed from library allocation.
    if not list((OUT/'resources').glob('snapshot_*.json')):raise RuntimeError('Await root resource receipt before CPU execution')
    tick=time.perf_counter();results=[one(task,sha(OUT/'manifest.json')) for task in m['tasks']]
    assert len(results)==10 and sum(r['GEO_calls'] for r in results)==10
    all_faces=[];all_stages=[]
    for task in m['tasks']:
        for name,store in [('per_face_stages.csv',all_faces),('stage_summary.csv',all_stages)]:
            with (Path(task['output_dir'])/name).open(encoding='utf-8-sig',newline='') as f:store.extend(csv.DictReader(f))
    csv_write(OUT/'per_face_stages.csv',all_faces);csv_write(OUT/'stage_summary.csv',all_stages)
    verify(m['protected']);verify(m['source_hashes'])
    value=dict(status='COMPLETE',tasks=10,RAW_GEO=8,SOURCE_GEO=2,GEO_calls=10,orientation_calls=10,
        model_calls=0,D0_forwards=0,training=0,OT=0,new_noise=0,CPU_threads=1,CPU_workers=1,
        seconds=time.perf_counter()-tick,face_stage_rows=len(all_faces),stage_summary_rows=len(all_stages),
        files={str(Path(t['output_dir'])/'receipt.json'):sha(Path(t['output_dir'])/'receipt.json') for t in m['tasks']})
    value['files'].update({str(OUT/name):sha(OUT/name) for name in ('manifest.json','per_face_stages.csv','stage_summary.csv')})
    write(complete,value);log('CORE_COMPLETE',**{k:v for k,v in value.items() if k!='files'});return value

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run'));args=parser.parse_args()
    result=prepare() if args.command=='prepare' else run()
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','tasks','protected','source_hashes')},ensure_ascii=False))
