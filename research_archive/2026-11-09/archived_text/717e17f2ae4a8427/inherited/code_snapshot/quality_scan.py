"""Bounded, recoverable CPU source scans; reuse the frozen USER detector."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
import traceback
import numpy as np

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_user_data_scaling_12h_v1'
SOURCE=ROOT/'r2/data_tools/user_conditions_v1/quality.py'
SCHEMA='r2_user_scaling_source_quality_v1'


def old_api():
    spec=importlib.util.spec_from_file_location('_r2_scaling_frozen_quality_wrapper',SOURCE)
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    return q


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text('utf-8-sig'))


def output_path(path):
    path=Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('All quality output must stay within registered OUT: '+str(path))
    for parent in path.parents:
        if parent==OUT.parent: break
        if parent.is_symlink(): raise ValueError('Output symlink refused')
    return path


def save(path,value):
    path=output_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text('utf-8')!=text: raise FileExistsError(str(path))
        return
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('x',encoding='utf-8') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    if path.exists(): raise FileExistsError(str(path))
    os.replace(temp,path)


def row_adapter(row):
    p=dict(row)
    p['uid']=row.get('uid',row.get('parent_uid'))
    p['variant_uid']=row.get('variant_uid','registered_variant')
    p['npz']=row.get('npz',row.get('prepared_npz'))
    p['sha256']=row.get('sha256',row.get('prepared_sha256'))
    p['split']=row.get('split',row.get('official_split','UNASSIGNED'))
    if not all(p[k] is not None for k in ('uid','npz','sha256')): raise ValueError('Incomplete parent row')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',p['uid']+'__'+p['variant_uid']): raise ValueError('Unsafe identity')
    if not 128<=int(p['N'])<=256: raise ValueError('Out-of-scope N')
    return p


def scan_one(row):
    q=old_api(); scan_fn,policy,source_hashes=q.frozen_detector()
    parent=row_adapter(row);uid=parent['uid'];variant=parent['variant_uid'];n=int(parent['N'])
    dest=OUT/'data/quality'/(uid+'__'+variant+'.json')
    receipt=dest.with_name(dest.stem+'.receipt.json')
    start=dest.with_name(dest.stem+'.started.json')
    identity=dict(uid=uid,variant_uid=variant,N=n,input_npz=parent['npz'],input_npz_sha256=parent['sha256'],
        detector_source_hashes=source_hashes,quality_source_sha256=sha(SOURCE),L=q.L)
    if dest.exists():
        result=read(dest)
        if result['status']!='COMPLETE' or result['identity']!=identity or sha(parent['npz'])!=parent['sha256']:
            raise ValueError('Completed quality receipt identity mismatch')
        summary=result['receipt']
        if receipt.exists():
            r=read(receipt)
            if r['quality_sha256']!=sha(dest) or r['identity']!=identity: raise ValueError('Quality payload changed')
        else:
            save(receipt,dict(identity=identity,quality_sha256=sha(dest),**summary))
        return dict(summary,reused=True,quality_path=str(dest),quality_sha256=sha(dest))
    if start.exists():
        raise RuntimeError('Unfinished quality attempt requires explicit engineering review, not silent replay: '+str(start))
    tick=time.monotonic();save(start,dict(identity=identity,started_unix=time.time(),pid=os.getpid(),scientific_calls='CPU source pair detector only'))
    try:
        tri,original,face_ids,mapping=q.load_mesh(parent)
        basic=q.basic_geometry(tri,original)
        if basic['nonfinite_source_face_ids'] or basic['exact_zero_area_source_face_ids']:
            raise ValueError('Invalid complete M: finite positive-area faces required before pair scan')
        calls={};scan=scan_fn(tri,tri,q.L,self_pairs=True,independent=True,ledger=calls)
        expected=n*(n-1)//2
        if scan['tested_pairs']!=expected or sum(scan['classification_counts'].values())!=expected or not scan['independent_verification']:
            raise ValueError('Incomplete source pair scan')
        contacts=[];seen=set();mask=np.zeros(n,np.uint8)
        for contact in scan['contacts']:
            pair=contact['pair']
            if len(pair)!=2 or not 0<=pair[0]<pair[1]<n or tuple(pair) in seen: raise ValueError('Invalid pair ID')
            seen.add(tuple(pair));cat=q.category(contact['classification'])
            item=dict(pair=pair,original_face_ids=None if original is None else [int(original[i]) for i in pair],
                classification=contact['classification'],category=cat)
            if cat in ('proper','coplanar_positive_overlap','UNKNOWN'):
                item['detector']=contact['detector'];mask[pair]|={'proper':1,'coplanar_positive_overlap':2,'UNKNOWN':4}[cat]
            contacts.append(item)
        for group in basic['exact_duplicate_groups']: mask[group]|=32
        whole=q.summarize_pairs(contacts,expected)
        if sha(parent['npz'])!=parent['sha256']: raise ValueError('Source changed during scan')
        summary=dict(status='COMPLETE',uid=uid,variant_uid=variant,N=n,wall_seconds=time.monotonic()-tick,
            confirmed_illegal_pair_count=whole['confirmed_illegal_pair_count'],unknown_pair_count=whole['unknown_pair_count'],
            new_unique_pair_tests=scan['tested_pairs'],source_pair_ledger=calls,model_forwards=0,training_updates=0,
            OT_calls=0,mesh_modifications=0,process_pid=os.getpid(),numeric_threads=2)
        result=dict(schema=SCHEMA,status='COMPLETE',identity=identity,uid=uid,variant_uid=variant,N=n,
            split=parent['split'],input_npz=parent['npz'],input_npz_sha256=parent['sha256'],
            full_target_sha256=hashlib.sha256(tri.tobytes()).hexdigest(),L=q.L,
            source_vertex_mapping=mapping,original_face_ids=None if original is None else original.tolist(),
            face_id_policy='Unchanged model full_target row IDs; original IDs from saved indexed source mapping.',
            basic_geometry=basic,pair_contacts=contacts,whole_parent=whole,faceproblem_mask=mask.tolist(),
            faceproblem_mask_bits={'proper':1,'coplanar_positive_overlap':2,'UNKNOWN':4,'nonfinite':8,'zero_area':16,'exact_duplicate':32},
            known_problem_face_ids=np.flatnonzero(mask&3).tolist(),unknown_face_ids=np.flatnonzero(mask&4).tolist(),
            detector_policy=policy,independent_verification=True,automatic_parent_rejection=False,
            interpretation='Reject C intersecting confirmed-illegal/unknown pair endpoints; Y-only risk stays separately reported.',
            receipt=summary)
        save(dest,result);save(receipt,dict(identity=identity,quality_sha256=sha(dest),**summary))
        return dict(summary,reused=False,quality_path=str(dest),quality_sha256=sha(dest))
    except BaseException as e:
        save(dest.with_name(dest.stem+'.failure.json'),dict(identity=identity,status='STOPPED',error=repr(e),
            traceback=traceback.format_exc(),wall_seconds=time.monotonic()-tick,model_calls=0))
        raise


def partition(parent_quality,face_ids):
    """No new detector calls; preserve original CC/CF/FF denominators."""
    q=old_api();p=read(parent_quality) if isinstance(parent_quality,(str,Path)) else parent_quality
    known=set(map(int,face_ids));n=p['N'];k=len(known);f=n-k
    if len(known)!=len(face_ids) or not 0<k<n or min(known)<0 or max(known)>=n: raise ValueError('Invalid C IDs')
    rows={g:[] for g in ('CC','CF','FF')}
    for contact in p['pair_contacts']:
        i,j=contact['pair'];g='CC' if i in known and j in known else 'FF' if i not in known and j not in known else 'CF'
        rows[g].append(contact)
    expected={'CC':k*(k-1)//2,'CF':k*f,'FF':f*(f-1)//2}
    partitions={g:q.summarize_pairs(rows[g],expected[g]) for g in rows}
    ok=all(partitions[g]['confirmed_illegal_pair_count']==0 and partitions[g]['unknown_pair_count']==0 for g in ('CC','CF'))
    return dict(N=n,K=k,partitions=partitions,local_CC_CF_qualified=ok,
        whole_Y_risk='ILLEGAL' if partitions['FF']['confirmed_illegal_pair_count'] else 'UNKNOWN' if partitions['FF']['unknown_pair_count'] else 'NO_DETECTED_PAIR_PROBLEM')


def run(manifest):
    q=old_api(); manifest=Path(manifest);data=read(manifest)
    rows=sorted((row_adapter(r) for r in data['rows']),key=lambda r:(r['uid'],r['variant_uid']))
    if len({(r['uid'],r['variant_uid']) for r in rows})!=len(rows): raise ValueError('Duplicate parent variants')
    if not rows: raise ValueError('Empty source inventory')
    h=sha(manifest);res=q.resources(); stamp=str(time.time_ns())
    save(OUT/'data/quality/dispatch'/('before_'+stamp+'.json'),res)
    # First real source scan measures duration/memory. No synthetic probe task.
    first=scan_one(rows[0]);after=q.resources()
    workers=2 if after['available_ram_bytes']>2*1024**3 and after['cpu_affinity_count']>=4 else 1
    save(OUT/'data/quality/dispatch'/('plan_'+stamp+'.json'),dict(first_real_scan=first,resources_after_first=after,
        actual_workers=workers,per_worker_threads=2,max_workers=2,reason='Two bounded CPU workers after first actual scan, available RAM and CPU affinity check.' if workers==2 else 'Resources restrict scan to one CPU worker.'))
    results=[first];print(json.dumps(first),flush=True)
    if workers==1:
        for row in rows[1:]:
            result=scan_one(row);results.append(result);print(json.dumps(result),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=2) as pool:
            # map returns stable input order; completion speed never selects data.
            for result in pool.map(scan_one,rows[1:],chunksize=1):
                results.append(result);print(json.dumps(result),flush=True)
    if sha(manifest)!=h: raise ValueError('Parent candidate manifest changed during scan')
    save(OUT/'data/quality'/('run_'+stamp+'.json'),dict(schema=SCHEMA,manifest=str(manifest),manifest_sha256=h,
        rows=results,workers=workers,numeric_threads=2,resources_after=q.resources(),implementation_sha256=sha(__file__)))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default=str(OUT/'data/parent_candidates.json'))
    args=p.parse_args();run(args.manifest)
