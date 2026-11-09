"""Resumable 24 x 2 x 3 CPU experiment; original assets are read-only."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, concurrent.futures, csv, json, sys, time, traceback, hashlib, resource
from collections import Counter
from pathlib import Path
import numpy as np
from . import assets as A, engine as E, metrics as Q

OUT=A.OUT; CODE=Path(__file__).resolve().parent
KEYS=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
      'free_minimum_angle_lt5_fraction','free_q_gt10_fraction')

def clean(x):
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list,set)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(clean(obj),ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    if p.exists():
        if p.read_text(encoding='utf-8')!=data:raise ValueError('Refusing overwrite '+str(p))
        return
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(data,encoding='utf-8');tmp.replace(p)

def csvwrite(p,rows):
    rows=clean(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)

def label(mode,d):return mode+'_'+str(int(round(d*1000))).zfill(3)

def freeze():
    p=OUT/'registration.json'
    if p.exists():
        r=A.read(p)
        amendment=OUT/'engineering_amendments.json'
        if amendment.exists():
            for item in A.read(amendment)['items']:
                if r['source_sha256'][item['path']]!=item['old_sha256']:raise ValueError('Amendment chain mismatch')
                r['source_sha256'][item['path']]=item['new_sha256']
        return r
    rows=A.read(OUT/'input_manifest.json')['rows']
    protected={k:v for r in rows for k,v in r['protected_files'].items()}
    defs=Q.definition_hashes();protected.update(defs)
    for f in ('geo_c.py','winding_solver.py','source_winding.py'):
        pp=A.ROOT/'r2/src/meshflow_control'/f;protected[str(pp)]=A.sha(pp)
    src={str(CODE/n):A.sha(CODE/n) for n in ('run.py','engine.py','assets.py','metrics.py')}
    value=dict(status='FROZEN',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256=src,protected_files=protected,input_manifest_sha256=A.sha(OUT/'input_manifest.json'),
        cells=144,requests=24,modes=['A','B'],delta_fractions=list(E.DELTAS),L=E.L,
        ordering='mode-specific maximum pre_geo member displacement, source face, local edge, C edge and target IDs',
        geometry='Original retained POST representatives; A corner-only, B all original GEO_C vertex-cluster members. No radius rerun; exact cleanup only.',
        budget='all bound original corner displacements including dropped-face B members; never resurrect faces',
        capacity='Any existing C/F incidence occupies C edge to prevent a third face; direction-independent free incidence=1 candidates',
        normals='physical displacement in fixed original canonical corner order, separate from orientation-only flips',
        orientation='restore source order; fixed-C solver; affected anchored component must be satisfiable; not outward certified',
        gates='area >1e-12 L^2; new/worse angle<5 or q>10 rejected; physical normal rotation>60 rejected; zero new CF illegal pairs',
        precision='FP32 source/target geometry; FP64 distances and predicates; old exact robust pair scanner; no model framework',
        tolerance='shape worsening comparison 1e-10 degrees/q; normal cosine 1e-12; original exact geometry predicates unchanged',
        CPU_workers=2,threads_per_worker=2,model_calls=0,GPU_dispatches=0,
        closest_prior='R10: original 10.16-57.22%L candidate motions,125 anchor conflicts and35 all-invalid; current <=3.5%L, actual free-open topology, identity/shape/CF gates and local clusters',
        interpretation='attainable geometric feasibility under fixed proxy, not oracle upper bound or pointer learnability proof')
    write(p,value)
    for path,h in {**src,**defs}.items():
        pp=Path(path);dest=OUT/'actual_code'/(h[:12]+'__'+pp.name);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(pp.read_bytes())
    return value

def worker(row):
    start=time.perf_counter();a=A.load_asset(row);K=row['K'];base=a['post'];canonical=a['canonical']
    bp=A.read(Path(row['score_path']).parent/'pairs.json')
    bl=Q.connectivity_ledger(base['vertices'],base['faces'],K,bp,base['source_face_ids'],base['source_corner_ids'])
    assert bl['summary']['C_F_valid_edge_count']==row['baseline_valid_cf_edges']
    write(OUT/'baseline'/row['short_id']/'connectivity.json',bl)
    result=[]
    for mode in ('A','B'):
        for d in E.DELTAS:
            lab=label(mode,d);dest=OUT/'cells'/lab/row['short_id'];done=dest/'complete.json'
            if done.exists():
                record=A.read(done);A.verify(record['files']);result.append(record['row']);continue
            if dest.exists():raise RuntimeError('Unfinished cell; explicit recovery required: '+str(dest))
            tick=time.perf_counter();write(dest/'started.json',dict(request_id=row['request_id'],label=lab,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            try:
                inv=np.argsort(a['source']['C_source_corner_permutations'],axis=1)
                output,candidates,deleted,summary=E.decode(a['pre_geo'],canonical,K,inv,mode,d,Q.classify_cf)
                np.savez_compressed(dest/'post.npz',**output)
                write(dest/'binding.json',summary);csvwrite(dest/'candidates.csv',candidates);write(dest/'candidate_records.json',candidates);csvwrite(dest/'deleted_faces.csv',deleted)
                from .verify import verify_arrays, verify_candidate_records
                assignments=dict(zip(map(int,output['assignment_input_corner_ids']),map(int,output['assignment_target_C_vertex_ids'])))
                audit=verify_arrays(base,output,a['pre_geo'],K,mode,E.L*d,assignments,reference_length=E.L)
                write(dest/'array_verification.json',audit)
                write(dest/'candidate_verification.json',verify_candidate_records(a,candidates,summary,output))
                score=Q.score_arrays(output['vertices'],output['faces'],K,a['C_canonical'],a['C_canonical_source_vertex_ids'],row['query_descriptor'],row['N'],output['source_face_ids'])
                vectors=score.pop('vectors');np.savez_compressed(dest/'vectors.npz',**vectors)
                write(dest/'score.json',score)
                nl=Q.connectivity_ledger(output['vertices'],output['faces'],K,score['pairs'],output['source_face_ids'],output['source_corner_ids'])
                aff={i for c in candidates if c['decision']=='ACCEPTED' for i in c['affected_source_faces']}
                ledger=Q.compare_ledgers(bl,nl,bp,score['pairs'],aff)
                write(dest/'connectivity.json',nl);write(dest/'paired_ledger.json',ledger)
                for field in ('baseline_edge_fates','new_edges','illegal_pair_fates'):csvwrite(dest/(field+'.csv'),ledger[field])
                if ledger['summary']['CF_new_illegal_pairs']!=0:raise AssertionError('Final zero-new-CF gate failed')
                m=score['metrics'];g=score['topology']
                rec=dict(label=lab,mode=mode,delta_fraction=d,request_id=row['request_id'],short_id=row['short_id'],parent_id=row['parent_id'],split=row['split'],noise=row['candidate'],
                    raw_adoption=row['raw_adoption'],baseline_joint=int(row['baseline_joint']),baseline_adopted_unjoined=int(row['baseline_adopted_unjoined']),
                    baseline_valid_cf_edges=row['baseline_valid_cf_edges'],valid_cf_edges=g['C_F_valid_edge_count'],valid_cf_length=g['C_F_valid_edge_total_length'],
                    screening_joint=int(row['raw_adoption']=='YES' and g['C_F_valid_edge_count']>0),
                    accepted=summary['accepted'],candidates=summary['candidate_count'],max_bound_displacement=summary['max_bound_corner_displacement'],
                    post_path=str(dest/'post.npz'),seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    **ledger['summary'],**{k:v for k,v in m.items() if isinstance(v,(int,float,bool)) or v is None})
                for key in KEYS:
                    old=float(row['score_table_row'][key]);rec['baseline_'+key]=old;rec['delta_pp_'+key]=100*(m[key]-old)
                for b in score['boundary']:
                    rec['A_VMR__'+b['setting']]=b['VMR'];rec['PMR__'+b['setting']]=b['perfect']
                rec.update({'reject_'+k:v for k,v in summary['rejections'].items() if k!='ACCEPTED'})
                files={str(p):A.sha(p) for p in dest.iterdir() if p.is_file()}
                write(done,dict(status='COMPLETE',files=files,row=rec,model_calls=0,CPU_only='torch' not in sys.modules))
                result.append(rec)
                print(json.dumps(dict(cell=lab+'/'+row['short_id'],accepted=summary['accepted'],joint=rec['screening_joint'],newCF=rec['CF_new_legal_edges'],lostFF=rec['FF_lost_legal_edges'],seconds=round(rec['seconds'],2))),flush=True)
            except Exception:
                write(dest/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),model_calls=0));raise
    return result

def aggregate():
    rows=[]
    for p in sorted((OUT/'cells').glob('*/*/complete.json')):rows.append(A.read(p)['row'])
    csvwrite(OUT/'per_output.csv',rows);summaries=[]
    for lab in sorted({r['label'] for r in rows}):
        for split in ('ALL','TRAIN','DEV'):
            rr=[r for r in rows if r['label']==lab and (split=='ALL' or r['split']==split)]
            if not rr:continue
            s=dict(label=lab,split=split,outputs=len(rr),screening_joint=sum(r['screening_joint'] for r in rr),
                baseline_joint=sum(r['baseline_joint'] for r in rr),new_joint=sum(r['screening_joint'] and not r['baseline_joint'] for r in rr),
                lost_joint=sum(r['baseline_joint'] and not r['screening_joint'] for r in rr),
                any_interface=sum(r['valid_cf_edges']>0 for r in rr),accepted=sum(r['accepted'] for r in rr),
                legal_cf_edges=sum(r['valid_cf_edges'] for r in rr),legal_cf_length=sum(r['valid_cf_length'] for r in rr))
            for k in ('CF_retained_legal_edges','CF_lost_legal_edges','CF_new_legal_edges','FF_lost_legal_edges','FF_lost_baseline_length','FF_new_illegal_pairs','FF_illegal_pairs_removed','deleted_source_faces','deleted_source_area'):
                s[k]=sum(r[k] for r in rr)
            for k in KEYS:s['delta_pp_'+k]=float(np.mean([r['delta_pp_'+k] for r in rr]))
            summaries.append(s)
    csvwrite(OUT/'summary.csv',summaries)
    funnel=[]
    for lab in sorted({r['label'] for r in rows}):
        cc=Counter()
        for p in (OUT/'cells'/lab).glob('*/binding.json'):
            b=A.read(p);cc.update(b['rejections']);cc.update({'prefilter_'+k:v for k,v in b['prefilter'].items()})
        funnel.append(dict(label=lab,**cc))
    csvwrite(OUT/'rejection_funnel.csv',funnel)
    return dict(completed=len(rows),summary=[s for s in summaries if s['split']=='ALL'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('freeze','run','aggregate','status'),default='status');p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    if args.stage=='freeze':
        r=freeze();print(json.dumps(dict(status=r['status'],cells=r['cells'],protected_files=len(r['protected_files']))));return
    if args.stage=='status':print(json.dumps(dict(complete=len(list((OUT/'cells').glob('*/*/complete.json'))),failed=len(list((OUT/'cells').glob('*/*/failure.json'))))));return
    if args.stage=='aggregate':print(json.dumps(aggregate()));return
    reg=freeze();A.verify(reg['protected_files']);A.verify(reg['source_sha256'])
    if args.workers not in (1,2):raise ValueError('Measured CPU budget permits at most2 workers')
    rows=A.read(OUT/'input_manifest.json')['rows'];tick=time.perf_counter()
    write(OUT/'execution_started.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),workers=args.workers,threads_per_worker=2,resource_snapshot='resources_before.json',GPU_workers=0))
    # Start one authorized request; remaining independent requests then use the
    # measured small CPU footprint. No synthetic or extra scientific probes.
    if not list((OUT/'cells').glob('*/*/complete.json')):
        worker(rows[0]);rows=rows[1:]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(worker,rows):pass
    result=aggregate();A.verify(reg['protected_files']);A.verify(reg['source_sha256'])
    assert result['completed']==144
    write(OUT/'execution_complete.json',dict(status='COMPLETE',cells=144,seconds=time.perf_counter()-tick,workers=args.workers,model_calls=0,GPU_calls=0,protected_files_verified=len(reg['protected_files'])))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    from collections import Counter
    main()
