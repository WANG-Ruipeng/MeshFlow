"""Isolated CPU runner for frozen regional RAW/D0 coordinate proposals.

This module does not generate proposals, dispatch pools, train, call D0, or
assign visual labels. Complete jobs are verified and reused; partial jobs are
never replayed automatically. Run only through the root's measured dispatch.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import hashlib
import io
import json
import re
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_regional_mix_dual_v1'
L = float(np.sqrt(3.) * 2 / .3762)
VERSION = 'bench_regional_mix_cpu_runtime_v1'
for _path in (ROOT, ROOT / 'r2/src'):
    sys.path.insert(0, str(_path))


def utc():
    return datetime.now(timezone.utc).isoformat()


def resolve(value):
    text = str(value).replace('\\','/')
    for prefix in ('E:/MeshFlow/','/mnt/e/MeshFlow/'):
        if text.startswith(prefix):
            return ROOT / text[len(prefix):]
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def write_new(path,value):
    p=resolve(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(clean(value),f,ensure_ascii=False,indent=2,allow_nan=False)


def npz(path,value):
    p=resolve(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f: np.savez_compressed(f,**value)


def arrays(path):
    with np.load(resolve(path),allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def csvwrite(path,rows):
    p=resolve(path);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def verify(book):
    for path,expected in book.items():
        if sha(path)!=expected: raise ValueError('Frozen hash changed: '+str(path))


def assert_cpu():
    if 'torch' in sys.modules:
        raise RuntimeError('CPU-only boundary violated: torch imported')


def resources():
    """Measured raw GPU query plus normalized CPU fields; no GPU work."""
    assert_cpu()
    from r2.pilots.bench_adaptation_strength_v1.common import resources as original
    r=original()
    return dict(utc=utc(),cpu_logical=r['CPU_cores'],cpu_percent=r['CPU_percent'],
                ram_total_bytes=r['RAM_total'],ram_available_bytes=r['RAM_available'],
                loadavg=r['load'],gpu=r['gpu'],gpu_processes=r['gpu_processes'],
                science_and_python_processes=r['science_and_python_processes'],
                threads_per_worker=1,scope='Aggregate measured resources; no process-VRAM estimate',
                peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def modules():
    assert_cpu()
    from meshflow_control import geo_c,geo_io,source_winding
    from r2.pilots.seam_slot_c200_v1 import postprocess as P
    from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
    from r2.pilots.bench_four_routes_v1.hybrid.run import ledger_compare
    assert_cpu()
    return geo_c,geo_io,source_winding,P,CPU,ledger_compare


def definition_hashes():
    geo,io_,sw,P,CPU,ledger_compare=modules()
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    from r2.pilots.bench_four_routes_v1.hybrid import run as old_run
    paths=[Path(__file__),Path(geo.__file__),Path(io_.__file__),Path(sw.__file__),
           Path(P.__file__),Path(P.G.__file__),Path(CPU.__file__),Path(CPU.H.__file__),
           Path(CPU.alpha.__file__),Path(CPU.holes.__file__),Path(CPU.M.__file__),
           Path(old_run.__file__),Path(G.__file__),Path(G.D.__file__),Path(G.H.__file__)]
    result={str(p):sha(p) for p in paths};result.update(CPU.H.Q.definition_hashes())
    assert_cpu();return result


def job_dir(job_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',job_id): raise ValueError('Unsafe job_id')
    return OUT/'runs'/job_id


def load_jobs(path):
    p=resolve(path);reg=read(p)
    if reg.get('status')!='FROZEN': raise ValueError('Jobs must be frozen')
    jobs=reg['jobs'];ids=[j['job_id'] for j in jobs]
    if len(ids)!=len(set(ids)) or not 1<=len(ids)<=30: raise ValueError('Invalid finite job budget')
    if len({j['request']['request_id'] for j in jobs})>6: raise ValueError('More than six fixed requests')
    if len(reg['metric_fields'])!=177 or len(set(reg['metric_fields']))!=177: raise ValueError('Expected original177 metric fields')
    for j in jobs: job_dir(j['job_id'])
    verify(reg['protected']);verify(reg['source_hashes'])
    return reg,sha(p),{j['job_id']:j for j in jobs}


def proposal_contract(job):
    """Validate arrays without GEO, detector, score, renderer or model."""
    q=job['request'];p=job['paths'];raw=arrays(p['raw']);source=arrays(p['source'])
    prop=job['proposal'];verify({prop['path']:prop['sha256']});z=arrays(prop['path'])
    x=z[prop.get('output_key','output')];N,K=int(q['N']),int(q['K'])
    if x.dtype!=np.float32 or x.shape!=(N,3,3) or not np.isfinite(x).all(): raise ValueError('Malformed canonical proposal')
    if raw['raw'].shape!=x.shape or raw['raw'].dtype!=x.dtype: raise ValueError('RAW domain differs')
    if int(z['known_faces'])!=K or int(raw['known_faces'])!=K: raise ValueError('K differs')
    if not np.array_equal(z['raw_slot_ids'],np.arange(N)): raise ValueError('RAW slots reordered')
    if not np.array_equal(z['source_corner_ids'],np.tile(np.arange(3),(N,1))): raise ValueError('Proposal must use RAW canonical corner order')
    if x[:K].tobytes()!=raw['raw'][:K].tobytes(): raise ValueError('Proposal C changed')
    if raw['raw_seam_slot_mask'].any() or (raw['raw_anchor_ids'][K:]>=0).any(): raise ValueError('Native-only input required')
    inp=arrays(q['inference_npz'])
    if source['C_original'].tobytes()!=inp['C_vertices'][inp['C_faces_original']].tobytes(): raise ValueError('Source C identity differs')
    identity=dict(request_id=q['request_id'],condition_id=q['condition_id'],parent_id=q['parent_id'],N=N,K=K,
        noise_index=q['noise_index'],raw_sha256=sha(p['raw']),source_sha256=sha(p['source']),
        input_sha256=sha(q['inference_npz']),metadata_sha256=sha(q['metadata_npz']),
        query_sha256=sha(q['query_descriptor']['path']),query_descriptor=q['query_descriptor'],
        proposal_array_sha256=hashlib.sha256(x.tobytes()).hexdigest(),proposal_shape=list(x.shape),proposal_dtype=str(x.dtype))
    return identity,raw,source,x


def verify_result(receipt):
    if receipt.get('status')!='COMPLETE': raise ValueError('Incomplete producer')
    verify(receipt['files'])
    artifacts=receipt['artifacts'];cpu=read(artifacts['cpu_complete'])
    if cpu.get('status')!='COMPLETE': raise ValueError('Incomplete CPU cascade')
    verify(cpu['files']);assert_cpu();return receipt


def _query_identity(q):
    return dict(request_id=q['request_id'],condition_id=q['condition_id'],parent_id=q['parent_id'],
        noise_index=q['noise_index'],N=q['N'],K=q['K'],
        input_sha256=sha(q['inference_npz']),metadata_sha256=sha(q['metadata_npz']),
        query_sha256=sha(q['query_descriptor']['path']),query_descriptor=q['query_descriptor'])


def reuse_external(job,identity,x):
    """Only a predeclared same-request official-chain result can be reused."""
    c=job['reuse_of_external'];verify({c['registration_path']:c['registration_sha256'],c['complete_path']:c['complete_sha256']})
    oldreg=read(c['registration_path']);old=read(c['complete_path']);oldroot=resolve(c['complete_path']).parent
    if old.get('status')!='COMPLETE' or old['registration_sha256']!=c['registration_sha256']: raise ValueError('Old producer incomplete or rebound')
    matches=[t for t in oldreg['tasks'] if t['request']['request_id']==c['request_id']]
    if len(matches)!=1: raise ValueError('Old request is not unique')
    task=matches[0];q=job['request'];p=task['paths']
    if _query_identity(task['request'])!=_query_identity(q): raise ValueError('Reuse query/input identity differs')
    if sha(p['raw'])!=identity['raw_sha256'] or sha(p['source'])!=identity['source_sha256']: raise ValueError('Reuse RAW/source differs')
    # Verify the existing chain and selector's frozen source identities, and
    # every old output/cascade file, without running the old selector.
    verify(oldreg['source_hashes']);verify(oldreg['protected']);verify(old['files'])
    if arrays(oldroot/'pre_geo.npz')['output'].tobytes()!=x.tobytes(): raise ValueError('Reuse canonical proposal differs')
    cpu_dir=resolve(c['cpu_dir']);cpu=read(cpu_dir/'complete.json');verify(cpu['files'])
    if cpu.get('status')!='COMPLETE' or cpu['request_id']!=q['request_id']: raise ValueError('Reuse cascade differs')
    for name in ('base','pre','raw','pairs'):
        expected={'base':oldroot/'post.npz','pre':oldroot/'pre_geo.npz','raw':resolve(p['raw']),'pairs':oldroot/'pairs.json'}[name]
        if cpu['identity']['sources'].get(str(expected))!=sha(expected): raise ValueError('Old cascade source binding differs')
    if cpu['identity']['seam_model'] is not False: raise ValueError('Expected original Native scorer applicability')
    return dict(base=str(oldroot/'post.npz'),pre=str(oldroot/'pre_geo.npz'),pairs=str(oldroot/'pairs.json'),
        final=str(cpu_dir/'CLEAN_FINAL_POST/post.npz'),final_pairs=str(cpu_dir/'CLEAN_FINAL_POST/pairs.json'),
        cpu_dir=str(cpu_dir),cpu_complete=str(cpu_dir/'complete.json'),
        metric_rows=str(cpu_dir/'metric_rows.json'),geometry=str(oldroot/'geometry.json'),
        ledger_BASE_POST=str(oldroot/'ledger_BASE_POST.json'),ledger_CLEAN_FINAL_POST=str(oldroot/'ledger_CLEAN_FINAL_POST.json'))


def reuse_original_d0(job,identity,x):
    """Verify an exact cached M proposal and its original complete CPU chain."""
    p=job['paths'];q=job['request'];pre=arrays(p['pre'])['output']
    if pre.dtype!=x.dtype or pre.shape!=x.shape or pre.tobytes()!=x.tobytes():
        raise ValueError('Original-D0 reuse requires exact canonical proposal bytes')
    cpu_dir=resolve(p['original_final']).parent.parent
    cp=cpu_dir/'complete.json';cpu=read(cp)
    oldroot=ROOT/'experiments/r2_bench_group_weight_match_v1'
    sp=oldroot/'evaluation/score_registration.json';sr=read(sp)
    manifest=read(oldroot/'data/eval_manifest.json')
    if sha(oldroot/'data/eval_manifest.json')!=sr['eval_manifest_sha256']:
        raise ValueError('Original M evaluation manifest differs')
    matches=[v for v in manifest['requests'] if v['request_id']==q['request_id']]
    if len(matches)!=1 or _query_identity(matches[0])!=_query_identity(q):
        raise ValueError('Original M request/query differs')
    if cpu.get('status')!='COMPLETE' or cpu['request_id']!=q['request_id']:
        raise ValueError('Original M CPU producer incomplete')
    ci=cpu['identity']
    if (ci['registration_sha256']!=sha(sp) or ci['seam_model'] is not False
            or ci['label']!='M' or ci['version']!='bench_clean_original_hinge_opposite_alpha_hole_v1'):
        raise ValueError('Original M pipeline identity differs')
    verify(sr['source_hashes']);verify(cpu['files']);verify(ci['sources'])
    for key in ('raw','pre','original_base'):
        full=str(resolve(p[key]))
        if ci['sources'].get(full)!=sha(full): raise ValueError('Original M source binding differs: '+key)
    if sha(p['raw'])!=identity['raw_sha256'] or sha(p['source'])!=identity['source_sha256']:
        raise ValueError('Original M RAW/source identity differs')
    post_dir=resolve(p['original_base']).parent
    return dict(base=str(resolve(p['original_base'])),pre=str(resolve(p['pre'])),
        pairs=str(resolve(p['original_base_pairs'])),final=str(resolve(p['original_final'])),
        final_pairs=str(resolve(p['original_final_pairs'])),cpu_dir=str(cpu_dir),cpu_complete=str(cp),
        metric_rows=str(cpu_dir/'metric_rows.json'),geometry=str(post_dir/'geometry.json'),
        source_score_registration=str(sp),source_eval_manifest=str(oldroot/'data/eval_manifest.json'))


def contract(path):
    reg,rh,jobs=load_jobs(path);rows=[]
    identities={}
    for job in reg['jobs']:
        ident,raw,source,x=proposal_contract(job);identities[job['job_id']]=ident
        reuse=job.get('reuse_of')
        if reuse:
            if reuse not in identities or reuse==job['job_id']: raise ValueError('Reuse owner must precede alias in registration')
            if identities[reuse]!=ident: raise ValueError('Alias scientific identity differs')
        if job.get('reuse_of_external'): reuse_external(job,ident,x)
        if job.get('reuse_original_d0'): reuse_original_d0(job,ident,x)
        if sum(bool(job.get(v)) for v in ('reuse_of','reuse_of_external','reuse_original_d0'))>1:
            raise ValueError('Multiple reuse sources')
        rows.append(dict(job_id=job['job_id'],request_id=ident['request_id'],identity=ident,
                         execution='REUSE_INTERNAL' if reuse else 'REUSE_EXTERNAL' if job.get('reuse_of_external')
                         else 'REUSE_ORIGINAL_D0' if job.get('reuse_original_d0') else 'NEW'))
    assert_cpu()
    return dict(status='PASS',version=VERSION,registration_sha256=rh,jobs=len(rows),rows=rows,
        maximum_new_GEO_calls=sum(r['execution']=='NEW' for r in rows),
        maximum_new_score_rows=2*sum(r['execution']=='NEW' for r in rows),model_calls=0,geometry_calls=0,scoring_calls=0)


def event(path,name,**values):
    with resolve(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(clean(dict(event=name,utc=utc(),monotonic=time.perf_counter(),**values)),allow_nan=False)+'\n');f.flush()


def _areas(tri):
    t=tri.astype(np.float64)
    return .5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)


def write_slot_ledgers(dest,raw,x,artifacts,K):
    N=len(x);rawarea=_areas(raw['raw']);prearea=_areas(x);rows=[]
    for stage,key in [('BASE_POST','base'),('CLEAN_FINAL_POST','final')]:
        m=arrays(artifacts[key]);ids=m.get('parent_source_raw_slot',m.get('source_face_ids'))
        if ids is None: raise ValueError('Missing original face identity')
        originals=ids[ids>=0]
        if len(set(originals.tolist()))!=len(originals): raise ValueError('Original slot duplicated')
        areas=_areas(m['output']);lookup={int(i):float(a) for i,a in zip(ids,areas) if i>=0}
        for slot in range(K,N):
            rows.append(dict(stage=stage,raw_slot=slot,retained=slot in lookup,raw_area=float(rawarea[slot]),
                proposal_area=float(prearea[slot]),actual_area=lookup.get(slot),
                max_corner_displacement_from_RAW=float(np.linalg.norm(x[slot].astype(float)-raw['raw'][slot],axis=1).max())))
    csvwrite(dest/'per_slot_deletion_area_displacement.csv',rows)


def run_one(path,job_id):
    reg,rh,jobs=load_jobs(path);job=jobs[job_id];dest=job_dir(job_id);done=dest/'complete.json'
    if done.exists():
        receipt=read(done)
        if receipt['registration_sha256']!=rh or receipt['job_sha256']!=canonical_hash(job): raise ValueError('Completed job contract changed')
        return verify_result(receipt)
    if dest.exists(): raise RuntimeError('Partial job retained; explicit engineering recovery required: '+str(dest))
    ident,raw,source,x=proposal_contract(job)
    reuse_kind=None;artifacts=None;reuse_receipt=None
    if job.get('reuse_of'):
        owner=jobs[job['reuse_of']];op=job_dir(owner['job_id'])/'complete.json'
        if not op.exists(): raise RuntimeError('Reuse dependency is not complete')
        o=verify_result(read(op))
        if o['registration_sha256']!=rh or o['scientific_identity']!=ident: raise ValueError('Reuse owner differs')
        artifacts=o['artifacts'];reuse_kind='INTERNAL';reuse_receipt=str(op)
    elif job.get('reuse_of_external'):
        artifacts=reuse_external(job,ident,x);reuse_kind='EXTERNAL';reuse_receipt=job['reuse_of_external']['complete_path']
    elif job.get('reuse_original_d0'):
        artifacts=reuse_original_d0(job,ident,x);reuse_kind='ORIGINAL_D0';reuse_receipt=artifacts['cpu_complete']
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.mkdir()  # Atomic ownership claim: another worker cannot enter this job.
    start=time.perf_counter();log=dest/'events.jsonl';phase='CLAIMED'
    write_new(dest/'attempt.json',dict(version=VERSION,job_id=job_id,job_sha256=canonical_hash(job),
        registration_sha256=rh,pid=os.getpid(),utc=utc(),scientific_identity=ident,reuse_kind=reuse_kind,
        threads_per_worker=1,automatic_retry=False))
    try:
        write_new(dest/'resources_before.json',resources())
        if reuse_kind is None:
            geo,io_,sw,P,CPU,compare=modules();K=int(job['request']['K']);phase='GEO_SOURCE_EXPORT'
            event(log,'CALL_STARTED',kind=phase)
            output,canonical,post,g,orientation,anchors=P.geometry_from_proposal(raw,x,source)
            if output.tobytes()!=x.tobytes(): raise AssertionError('Anchor restoration changed canonical candidate')
            npz(dest/'pre_geo.npz',dict(output=output,known_faces=np.asarray(K),raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
            npz(dest/'post_canonical.npz',canonical);npz(dest/'post.npz',post)
            write_new(dest/'geometry.json',dict(stats=g['stats'],face_records=g['face_records']))
            write_new(dest/'source_restoration.json',orientation['restoration_summary']);write_new(dest/'anchor_identity.json',anchors)
            event(log,'CALL_COMPLETE',kind=phase)
            phase='BASE_PAIR_DETECTOR';event(log,'CALL_STARTED',kind=phase)
            pairs=CPU.H.scan(post['output'],K);write_new(dest/'pairs.json',pairs);event(log,'CALL_COMPLETE',kind=phase)
            phase='FULL_CPU_CASCADE';event(log,'CALL_STARTED',kind=phase)
            CPU.process_one(job['request'],job['label'],dest/'post.npz',dest/'pre_geo.npz',job['paths']['raw'],
                dest/'pairs.json',dest/'cpu',False,rh)
            event(log,'CALL_COMPLETE',kind=phase)
            artifacts=dict(base=str(dest/'post.npz'),pre=str(dest/'pre_geo.npz'),pairs=str(dest/'pairs.json'),
                final=str(dest/'cpu/CLEAN_FINAL_POST/post.npz'),final_pairs=str(dest/'cpu/CLEAN_FINAL_POST/pairs.json'),
                cpu_dir=str(dest/'cpu'),cpu_complete=str(dest/'cpu/complete.json'),metric_rows=str(dest/'cpu/metric_rows.json'),
                geometry=str(dest/'geometry.json'))
        else:
            event(log,'REUSED_WITHOUT_SCIENTIFIC_CALL',reuse_kind=reuse_kind,receipt=reuse_receipt)
        # Original M reuse has no M-vs-M cross-backend ledger. Derive this
        # identity comparison from preserved meshes/pair caches, never detectors.
        if 'ledger_BASE_POST' not in artifacts:
            geo,io_,sw,P,CPU,compare=modules();K=int(job['request']['K'])
            phase='READ_ONLY_LEDGER';event(log,'LEDGER_STARTED')
            for stage,oldkey,oldpairkey,newkey,newpairkey in [('BASE_POST','original_base','original_base_pairs','base','pairs'),
                    ('CLEAN_FINAL_POST','original_final','original_final_pairs','final','final_pairs')]:
                ledger=compare(arrays(job['paths'][oldkey]),arrays(artifacts[newkey]),read(job['paths'][oldpairkey]),read(artifacts[newpairkey]),K,CPU)
                lp=dest/('ledger_'+stage+'.json');write_new(lp,ledger);artifacts['ledger_'+stage]=str(lp)
            event(log,'LEDGER_COMPLETE')
        phase='FINAL_RECEIPT'
        write_slot_ledgers(dest,raw,x,artifacts,int(job['request']['K']))
        rows=read(artifacts['metric_rows'])
        if len(rows)!=2 or {r['stage'] for r in rows}!={'BASE_POST','CLEAN_FINAL_POST'}: raise AssertionError('Expected exactly two scored stages')
        if any(any(f not in row for f in reg['metric_fields']) for row in rows): raise AssertionError('Missing original metric field')
        write_new(dest/'logical_metric_alias.json',dict(job_id=job_id,label=job['label'],request_id=job['request']['request_id'],
            metric_source=artifacts['metric_rows'],metric_source_sha256=sha(artifacts['metric_rows']),
            source_labels=[r['label'] for r in rows],relabel_policy='Display metadata only; original177 values unchanged',reuse_kind=reuse_kind))
        write_new(dest/'resources_after.json',resources());assert_cpu();verify(reg['protected']);verify(reg['source_hashes'])
        result=dict(status='COMPLETE',version=VERSION,job_id=job_id,label=job['label'],request_id=job['request']['request_id'],
            registration_sha256=rh,job_sha256=canonical_hash(job),scientific_identity=ident,
            execution='REUSED_'+reuse_kind if reuse_kind else 'NEW',reuse_receipt=reuse_receipt,artifacts=artifacts,
            seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            RSS_scope='Process-lifetime peak; not job-incremental',GEO_calls=0 if reuse_kind else 1,
            base_pair_detector_calls=0 if reuse_kind else 1,new_score_rows=0 if reuse_kind else 2,
            full_cascade_calls=0 if reuse_kind else 1,logical_score_rows=2,
            model_calls=0,D0_calls=0,new_noise=0,new_queries=0,ratings_generated=0,
            files={str(p):sha(p) for p in dest.rglob('*') if p.is_file()})
        # Seal reused physical artifacts too, without copying or changing them.
        result['files'].update({str(resolve(p)):sha(p) for p in artifacts.values() if resolve(p).is_file()})
        if reuse_receipt: result['files'][str(resolve(reuse_receipt))]=sha(reuse_receipt)
        write_new(done,result);return verify_result(result)
    except BaseException as error:
        event(log,'FAILED',phase=phase,error=repr(error))
        write_new(dest/'failure.json',dict(status='STOPPED_PARTIAL_NO_AUTOMATIC_RETRY',job_id=job_id,
            phase=phase,error=repr(error),traceback=traceback.format_exc(),utc=utc(),
            model_calls=0,D0_calls=0,scientific_call_events=str(log)))
        raise


def status(path):
    reg,rh,jobs=load_jobs(path);rows=[]
    for j in reg['jobs']:
        d=job_dir(j['job_id']);p=d/'complete.json'
        state='COMPLETE' if p.exists() else 'FAILED' if (d/'failure.json').exists() else 'PARTIAL' if d.exists() else 'NOT_STARTED'
        rows.append(dict(job_id=j['job_id'],state=state,path=str(d)))
    return dict(status='READ_ONLY_STATUS',registration_sha256=rh,jobs=rows,model_calls=0,scoring_calls=0)


def collect(path):
    reg,rh,jobs=load_jobs(path);dest=OUT/'results';done=dest/'complete.json'
    if done.exists():
        r=read(done)
        if r['registration_sha256']!=rh:raise ValueError('Collection registration changed')
        verify(r['files']);return r
    if dest.exists():raise RuntimeError('Partial collection retained')
    rows=[];book={};receipts=[]
    for job in reg['jobs']:
        p=job_dir(job['job_id'])/'complete.json';r=verify_result(read(p))
        if r['registration_sha256']!=rh:raise ValueError('Job registration mismatch')
        book[str(p)]=sha(p);receipts.append(r)
        for source in read(r['artifacts']['metric_rows']):
            row=dict(source,job_id=job['job_id'],label=job['label'],request_id=job['request']['request_id'],
                condition_id=job['request']['condition_id'],parent_id=job['request']['parent_id'],
                report_group=job['request']['report_group'],noise_index=job['request']['noise_index'],
                physical_execution=r['execution'],physical_metric_source=r['artifacts']['metric_rows'],
                original_metric_label=source['label'])
            assert all(row[f]==source[f] for f in reg['metric_fields'])
            rows.append(row)
    candidate_rows=len(rows);baseline_rows=[]
    for reference in reg['references']:
        q=reference['request'];p=reference['paths']
        cp=resolve(p['original_final']).parent.parent/'complete.json';producer=read(cp)
        if producer.get('status')!='COMPLETE' or producer['request_id']!=q['request_id']:
            raise ValueError('Original baseline producer incomplete')
        verify(producer['files']);book[str(cp)]=sha(cp)
        metric_path=cp.parent/'metric_rows.json'
        for source in read(metric_path):
            if source['stage'] not in ('BASE_POST','CLEAN_FINAL_POST'):raise ValueError('Original baseline stage differs')
            row=dict(source,job_id='REFERENCE_M_D0',label='M_D0',request_id=q['request_id'],
                condition_id=q['condition_id'],parent_id=q['parent_id'],report_group=q['report_group'],
                noise_index=q['noise_index'],physical_execution='REUSED_ORIGINAL_BASELINE',
                physical_metric_source=str(metric_path),original_metric_label=source['label'])
            if any(f not in source for f in reg['metric_fields']):raise ValueError('Baseline metric missing')
            assert all(row[f]==source[f] for f in reg['metric_fields'])
            baseline_rows.append(row)
    if len(baseline_rows)!=2*len(reg['references']):raise ValueError('Expected two baseline rows per request')
    rows.extend(baseline_rows)
    dest.mkdir();write_new(dest/'metric_rows.json',rows);csvwrite(dest/'per_output.csv',rows)
    result=dict(status='COMPLETE',registration_sha256=rh,logical_jobs=len(receipts),logical_score_rows=candidate_rows,
        reused_original_baseline_rows=len(baseline_rows),collected_rows=len(rows),
        new_GEO_calls=sum(r['GEO_calls'] for r in receipts),new_score_rows=sum(r['new_score_rows'] for r in receipts),
        physical_new_jobs=sum(r['execution']=='NEW' for r in receipts),reused_jobs=sum(r['execution']!='NEW' for r in receipts),
        fields=reg['metric_fields'],model_calls=0,D0_calls=0,new_noise=0,new_queries=0,
        input_job_receipts=book,files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
    write_new(done,result);return result


def render_one(path,job_id,stage):
    if stage not in ('BASE_POST','CLEAN_FINAL_POST'):raise ValueError('Unregistered render stage')
    reg,rh,jobs=load_jobs(path);job=jobs[job_id];receipt=verify_result(read(job_dir(job_id)/'complete.json'))
    if receipt['registration_sha256']!=rh:raise ValueError('Render producer registration differs')
    q=job['request'];p=job['paths'];a=receipt['artifacts'];K=int(q['K'])
    if job['label']=='RAW_ALL':raise ValueError('RAW_ALL is a comparison column, not another primary review page')
    raw_jobs=[j for j in reg['jobs'] if j['label']=='RAW_ALL' and j['request']['request_id']==q['request_id']]
    if len(raw_jobs)!=1:raise ValueError('Exactly one same-request RAW_ALL reference required')
    raw_receipt=verify_result(read(job_dir(raw_jobs[0]['job_id'])/'complete.json'))
    if raw_receipt['registration_sha256']!=rh:raise ValueError('RAW_ALL registration differs')
    cells=[dict(model='RAW',stage='RAW',path=p['raw']),
           dict(model='M_D0',stage=stage,path=p['original_base' if stage=='BASE_POST' else 'original_final']),
           dict(model='RAW_ALL',stage=stage,path=raw_receipt['artifacts']['base' if stage=='BASE_POST' else 'final']),
           dict(model=job['label'],stage=stage,path=a['base' if stage=='BASE_POST' else 'final'])]
    definition=dict(job_id=job_id,stage=stage,cells=cells,request=q,registration_sha256=rh,
                    cameras='Inherited azimuth -60/120 elevation23',sections='Inherited front/side position index1')
    dh=canonical_hash(definition);dest=job_dir(job_id)/'gallery';image=dest/(stage+'.png');done=image.with_suffix('.json')
    if done.exists():
        r=read(done)
        if r['definition_sha256']!=dh or sha(image)!=r['image_sha256']:raise ValueError('Render changed')
        verify(r['source_meshes']);return r
    if image.exists():raise RuntimeError('Unsealed image retained')
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    D,H=G.D,G.H;plt=H._mpl();start=time.perf_counter();dest.mkdir(exist_ok=True)
    fig=plt.figure(figsize=(21.2,13.5),dpi=110,facecolor='white');grid=fig.add_gridspec(4,4,left=.025,right=.99,bottom=.045,top=.92,wspace=.15,hspace=.23)
    inp=arrays(q['inference_npz']);original=inp['C_vertices'][inp['C_faces_original']]
    bounds=q['gallery_bounds'];depth=q['gallery_depth_bounds'];sources={};display=[]
    try:
        for col,cell in enumerate(cells):
            tri,role=G.roles(arrays(cell['path']),K);sources[str(resolve(cell['path']))]=sha(cell['path'])
            g=dict(C=tri[:K],F=tri[K:],C_original=original,tri=tri)
            facts=dict(model=cell['model'],stage=cell['stage'],role_counts={k:int((role==k).sum()) for k in ('C','SEAM','BULK','BRIDGE')},views=[])
            for row,az in enumerate((-60,120)):
                raster,scope=G.role_raster(tri,role,K,D.basis(az,23),np.asarray(depth['center']),depth['span'])
                ax=fig.add_subplot(grid[row,col]);ax.imshow(raster);ax.set_axis_off();ax.set_title(cell['model']+' '+cell['stage']+' | '+str(az),fontsize=9)
                if scope['projected_vertices_outside_inherited_frame']:ax.text(.01,.01,'Inherited-frame clipping',transform=ax.transAxes,fontsize=7,color='#b34b27')
                facts['views'].append(dict(azimuth=az,**scope))
            for row,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[row,col]);H.draw_section(ax,g,np.flatnonzero(role[K:]=='SEAM'),bounds,L,view,1)
                from matplotlib.collections import LineCollection
                axis,project=(2,(0,1)) if view=='front' else (0,(2,1));pos=bounds['front_positions' if view=='front' else 'side_positions'][1]
                ff,ids=H.finite_sections(g['F'],axis,pos,1e-12*L);bm=role[K:][ids]=='BRIDGE'
                if bm.any():ax.add_collection(LineCollection(ff[bm][:,:,project],colors='#41a46c',linewidths=1.6,zorder=3))
                ax.set_title(cell['model']+' | '+view+f'={pos:.5f}',fontsize=8)
            display.append(facts)
        fig.suptitle(q['request_id']+' | '+job_id+f' | N={q["N"]} K={K}\nOriginal RAW / official D0 chain / whole RAW chain / regional candidate; blue C, gray free, green bridge',fontsize=10)
        fig.text(.5,.012,'Actual AI_PROXY review is separate. Fixed sections alone do not certify cracks. No source-style ground truth.',ha='center',fontsize=8)
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110)
        with image.open('xb') as f:f.write(buffer.getvalue())
    finally:plt.close(fig)
    assert_cpu();verify(sources)
    result=dict(status='PENDING_ACTUAL_VIEW',job_id=job_id,request_id=q['request_id'],stage=stage,
        definition_sha256=dh,registration_sha256=rh,image_path=str(image),image_sha256=sha(image),
        cells=cells,source_meshes=sources,display=display,seconds=time.perf_counter()-start,
        model_calls=0,scoring_calls=0,ratings_generated=0)
    write_new(done,result);return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('contract','status','resources','run-one','verify','collect','render-one','code-hashes'))
    ap.add_argument('--jobs',type=Path);ap.add_argument('--job-id');ap.add_argument('--stage',choices=('BASE_POST','CLEAN_FINAL_POST'))
    args=ap.parse_args()
    if args.command=='resources':result=resources()
    elif args.command=='code-hashes':result=definition_hashes()
    else:
        if args.jobs is None:ap.error('--jobs required')
        if args.command=='contract':result=contract(args.jobs)
        elif args.command=='status':result=status(args.jobs)
        elif args.command=='collect':result=collect(args.jobs)
        else:
            if not args.job_id:ap.error('--job-id required')
            if args.command=='run-one':result=run_one(args.jobs,args.job_id)
            elif args.command=='verify':
                reg,rh,jobs=load_jobs(args.jobs);result=verify_result(read(job_dir(args.job_id)/'complete.json'))
                if result['registration_sha256']!=rh:raise ValueError('Registration differs')
            else:
                if not args.stage:ap.error('--stage required')
                result=render_one(args.jobs,args.job_id,args.stage)
    # Do not dump large request manifests, detector ledgers or file maps.
    if args.command in ('code-hashes','resources'):print(json.dumps(clean(result),ensure_ascii=False))
    else: print(json.dumps(clean({k:v for k,v in result.items() if k not in ('files','rows','source_meshes','scientific_identity','fields','input_job_receipts','cells','display')}),ensure_ascii=False))


if __name__=='__main__':main()
