"""Saved Bench GEO_C deletion attribution. CPU arrays only, no model/scorer calls.

Widths are triangle minimum altitudes (2A / longest edge), not semantic timber
thickness. Original slots are generated RAW identities, never source GT faces.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import collections
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'r2/src'))
from meshflow_control import geo_c
SOURCE = REPO / 'experiments/r2_bench_parts_convergence_v1'
OUT = REPO / 'experiments/r2_bench_post_rescue_diagnostic_v1/deletion'
LABELS = ('SEAM_S15000', 'SEAM_S21000')
R = geo_c.RADIUS
STAGES = ('RAW', 'D0_PROPOSAL', 'PRE_GEO', 'CLUSTERED_BEFORE_DELETE')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def same(a, b): return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    payload=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
    with p.open('x', encoding='utf-8') as f: f.write(payload)


def csvwrite(p, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def metrics(tri):
    t = tri.astype(np.float64)
    lengths = np.linalg.norm(t[:, [1, 2, 0]] - t, axis=2)
    cross = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
    area = .5*np.linalg.norm(cross, axis=1)
    emax = lengths.max(axis=1); emin = lengths.min(axis=1)
    height = np.divide(2*area, emax, out=np.zeros_like(area), where=emax>0)
    return dict(area=area, edge_min=emin, edge_max=emax, minimum_altitude=height)


def facekey(t): return tuple(sorted(tuple(float(x) for x in p) for p in t))


def ratio(a, b): return float(a/b) if b else None


def resource_snapshot():
    result = dict(cpu_logical=os.cpu_count(), loadavg=list(os.getloadavg()),
        meminfo=Path('/proc/meminfo').read_text(), BLAS_threads=1,
        scientific_GPU_jobs=0, GPU_access='nvidia-smi read-only resource metadata')
    for k, args in (
        ('gpu', ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader']),
        ('gpu_processes', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])):
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        result[k] = dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    return result


def work(job):
    label, q = job; started = time.perf_counter()
    d = SOURCE / 'evaluation' / label / q['request_id']; postdir = d/'post'
    rawp=d/'raw/raw.npz'; pp=postdir/'proposal.npz'; prep=postdir/'pre_geo.npz'
    canonp=postdir/'post_canonical.npz'; postp=postdir/'post.npz'; gp=postdir/'geometry.json'
    hashes={str(p):sha(p) for p in (rawp,pp,prep,canonp,postp,gp,postdir/'receipt.json',postdir/'proposal_receipt.json')}
    receipt=read(postdir/'receipt.json'); prop_receipt=read(postdir/'proposal_receipt.json')
    assert receipt['status']=='COMPLETE' and receipt['D0_forwards']==receipt['GEO_calls']==1
    assert receipt['post_sha256']==hashes[str(postp)] and receipt['raw_sha256']==hashes[str(rawp)]
    assert prop_receipt['proposal_sha256']==hashes[str(pp)]
    for p in (prep,canonp,gp): assert receipt['files'][str(p)]==hashes[str(p)]
    ra=arrays(rawp); raw=ra['raw']; proposal=arrays(pp)['official_unconstrained_proposal']; pre=arrays(prep)['output']
    ca=arrays(canonp); post=arrays(postp); geometry=read(gp); K=int(ra['known_faces']); N=len(raw)
    assert N==q['N'] and K==q['K'] and geometry['stats']['radius_model']==R
    assert all(t.shape==(N,3,3) and t.dtype==np.float32 and np.isfinite(t).all() for t in (raw,proposal,pre))
    fixed=ra['raw_anchor_ids']>=0; restored=proposal.copy()
    restored[fixed]=ra['C_vertices'][ra['raw_anchor_ids'][fixed]]
    assert same(restored,pre)
    required=('corner_target_input_ids','corner_vertex_ids','source_face_kept','source_face_reason','duplicate_of_source_face','source_face_ids','source_corner_ids','corner_status')
    missing=[k for k in required if k not in ca]
    if missing: raise ValueError('Required historical GEO mappings missing: '+str(missing))
    target=ca['corner_target_input_ids']; corrected=pre.reshape(-1,3)[target]
    kept=ca['source_face_kept']; sf=ca['source_face_ids']; sc=post['source_corner_ids']
    assert same(sf,np.flatnonzero(kept)) and kept[:K].all()
    assert same(corrected[sf],ca['output']) and same(ca['vertices'][ca['faces']],ca['output'])
    assert same(post['source_face_ids'],sf) and same(corrected[sf[:,None],sc],post['output'])
    assert same(post['vertices'],ca['vertices'])
    cv=ca['corner_vertex_ids']; present=cv>=0
    assert same(ca['vertices'][cv[present]],corrected[present]) and same(cv[sf],ca['faces'])
    delta=np.linalg.norm(corrected.astype(float)-pre.astype(float),axis=2)
    assert np.all(delta<=R) and np.allclose(delta,ca['corner_displacement'],rtol=0,atol=1e-15)
    records=geometry['face_records']; reason=ca['source_face_reason']; dup=ca['duplicate_of_source_face']
    assert len(records)==N and int((~kept).sum())==geometry['stats']['deleted_free_faces']
    assert dict(collections.Counter(reason[~kept].tolist()))==geometry['stats']['deleted_by_reason']
    stages=dict(zip(STAGES,(raw,proposal,pre,corrected))); mm={s:metrics(t) for s,t in stages.items()}
    role=np.where(np.arange(N)<K,'C',np.where(ra['raw_seam_slot_mask'],'SEAM','BULK'))
    rows=[]
    for i in range(N):
        record=records[i]
        assert record['source_face_id']==i and record['reason']==reason[i] and record['kept']==bool(kept[i])
        assert record['duplicate_of_source_face']==int(dup[i])
        why=str(reason[i]); partner=int(dup[i]); pre_area=mm['PRE_GEO']['area'][i]
        if kept[i]: primary='KEPT'
        elif why in ('exact_repeated_vertex','exact_collinear'):
            primary='GEO_COLLAPSE_NONZERO_PRE' if pre_area>0 else 'PRE_GEO_ALREADY_DEGENERATE'
            assert mm['CLUSTERED_BEFORE_DELETE']['area'][i]==0
        elif why in ('duplicate_C','duplicate_free'):
            assert 0<=partner<i and facekey(corrected[i])==facekey(corrected[partner])
            primary='PRE_GEO_ALREADY_DUPLICATE' if facekey(pre[i])==facekey(pre[partner]) else 'GEO_DUPLICATE_AFTER_MERGE'
        else: primary='OTHER'
        rh=mm['RAW']['minimum_altitude'][i]; dh=mm['D0_PROPOSAL']['minimum_altitude'][i]; ph=mm['PRE_GEO']['minimum_altitude'][i]
        if rh<R: crossing='RAW_ALREADY_BELOW_R'
        elif dh<R: crossing='D0_FIRST_BELOW_R'
        elif ph<R: crossing='ANCHOR_RESTORE_FIRST_BELOW_R'
        else: crossing='PRE_GEO_NOT_BELOW_R'
        r=dict(label=label,request_id=q['request_id'],condition_id=q['condition_id'],parent_id=q['parent_id'],split=q['split'],part_type=q['part_type'],
            source_raw_slot=i,role=str(role[i]),kept=bool(kept[i]),delete_reason=why,primary_attribution=primary,
            duplicate_of_raw_slot=partner,duplicate_partner_role=str(role[partner]) if partner>=0 else '',
            raw_exact_duplicate_of_partner=partner>=0 and facekey(raw[i])==facekey(raw[partner]),
            pre_exact_duplicate_of_partner=partner>=0 and facekey(pre[i])==facekey(pre[partner]),
            minimum_altitude_crossing=crossing,D0_over_RAW_altitude=ratio(dh,rh),PRE_over_RAW_altitude=ratio(ph,rh),
            D0_at_least_halved_altitude=bool(rh>0 and dh<=rh*.5),
            raw_nonzero_long_thin_relative_R=bool(rh>0 and rh<R and mm['RAW']['edge_max'][i]>4*R),
            pre_shortest_edge_over_radius=mm['PRE_GEO']['edge_min'][i]/R,
            pre_shortest_edge_over_2radius=mm['PRE_GEO']['edge_min'][i]/(2*R),
            pre_minimum_altitude_over_radius=ph/R,radius_model=R,
            merged_distinct_numeric_corners=len(set(tuple(float(x) for x in p) for p in corrected[i])),
            merged_distinct_bit_corner_ids=len(set(tuple(map(int,p.view(np.uint32))) for p in corrected[i])),
            retained_mesh_corner_vertex_ids=json.dumps(cv[i].tolist()),target_input_corner_ids=json.dumps(target[i].tolist()),
            corner_status=json.dumps(ca['corner_status'][i].tolist()),maximum_GEO_corner_displacement=float(delta[i].max()),
            semantic_normal_slat_or_erroneous_deletion='UNKNOWN_FROM_GEOMETRY_ONLY')
        for s,m in mm.items():
            for name,val in m.items(): r[s+'_'+name]=float(val[i])
        rows.append(r)
    outrows=[]
    for rr in ('C','SEAM','BULK','ALL_FREE'):
        mask=(np.arange(N)>=K) if rr=='ALL_FREE' else (role==rr)
        for fate,mask2 in (('ALL',mask),('KEPT',mask&kept),('DELETED',mask&~kept)):
            for s,m in mm.items():
                vals=m['minimum_altitude'][mask2]
                outrows.append(dict(label=label,request_id=q['request_id'],split=q['split'],part_type=q['part_type'],role=rr,fate=fate,stage=s,
                    faces=int(mask2.sum()),nonzero_faces=int((m['area'][mask2]>0).sum()),area_sum=float(m['area'][mask2].sum()),
                    minimum_altitude_below_radius=int((vals<R).sum()),median_minimum_altitude_over_R=float(np.median(vals/R)) if len(vals) else None))
    deleted=[r for r in rows if not r['kept']]
    summary=dict(label=label,request_id=q['request_id'],split=q['split'],part_type=q['part_type'],N=N,K=K,
        free_input=N-K,free_kept=int(kept[K:].sum()),free_deleted=len(deleted),deleted_fraction=len(deleted)/(N-K),
        seam_deleted=sum(r['role']=='SEAM' for r in deleted),bulk_deleted=sum(r['role']=='BULK' for r in deleted),
        deleted_RAW_area=sum(r['RAW_area'] for r in deleted),deleted_PRE_GEO_area=sum(r['PRE_GEO_area'] for r in deleted),
        RAW_free_area=float(mm['RAW']['area'][K:].sum()),PRE_GEO_free_area=float(mm['PRE_GEO']['area'][K:].sum()),
        POST_free_area=float(mm['CLUSTERED_BEFORE_DELETE']['area'][kept & (np.arange(N)>=K)].sum()),
        replay_GEO_calls=0,cache_roundtrip='PASS',source_slot_domain='GENERATED_RAW_SLOT_NOT_SOURCE_GT')
    for category in ('GEO_COLLAPSE_NONZERO_PRE','PRE_GEO_ALREADY_DEGENERATE','GEO_DUPLICATE_AFTER_MERGE','PRE_GEO_ALREADY_DUPLICATE','OTHER'):
        summary[category]=sum(r['primary_attribution']==category for r in deleted)
    return dict(rows=rows,stages=outrows,summary=summary,hashes=hashes,seconds=time.perf_counter()-started,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def examples(allrows):
    free=[r for r in allrows if r['role']!='C']; deleted=[r for r in free if not r['kept']]
    # Registered descriptive selectors, not chosen by generated visual appearance.
    groups=[('GEO_collapse_largest_PRE_area',[r for r in deleted if r['primary_attribution']=='GEO_COLLAPSE_NONZERO_PRE'],'PRE_GEO_area'),
        ('D0_width_crossing',[r for r in deleted if r['minimum_altitude_crossing']=='D0_FIRST_BELOW_R'],'RAW_minimum_altitude'),
        ('RAW_already_long_thin',[r for r in deleted if r['raw_nonzero_long_thin_relative_R']],'RAW_edge_max'),
        ('duplicate_largest_PRE_area',[r for r in deleted if 'DUPLICATE' in r['primary_attribution']],'PRE_GEO_area'),
        ('kept_thin_counterexample',[r for r in free if r['kept'] and r['pre_minimum_altitude_over_radius']<1],'PRE_GEO_edge_max'),
        ('collapse_width_above_R',[r for r in deleted if r['primary_attribution']=='GEO_COLLAPSE_NONZERO_PRE' and r['pre_minimum_altitude_over_radius']>=1],'PRE_GEO_area')]
    selected=[]
    for category,rr,field in groups:
        for row in sorted(rr,key=lambda r:(-r[field],r['label'],r['request_id'],r['source_raw_slot']))[:2]:
            label=row['label'];rid=row['request_id'];i=row['source_raw_slot'];d=SOURCE/'evaluation'/label/rid
            raw=arrays(d/'raw/raw.npz');pre=arrays(d/'post/pre_geo.npz')['output'];ca=arrays(d/'post/post_canonical.npz')
            prop=arrays(d/'post/proposal.npz')['official_unconstrained_proposal'];corr=pre.reshape(-1,3)[ca['corner_target_input_ids']]
            dest=OUT/'examples'/(category+'__'+label+'__'+rid+'__slot'+str(i));dest.mkdir(parents=True)
            center=pre[i].mean(axis=0);half=max(float(row['PRE_GEO_edge_max'])*1.5,8*R)
            local=np.flatnonzero(np.linalg.norm(pre.mean(axis=1)-center,axis=1)<=half)
            with (dest/'geometry.npz').open('xb') as f:
                np.savez_compressed(f,RAW=raw['raw'],D0_PROPOSAL=prop,PRE_GEO=pre,CLUSTERED_BEFORE_DELETE=corr,
                    POST=arrays(d/'post/post.npz')['output'],selected_raw_slot=np.asarray(i),nearby_raw_slots=local,
                    raw_seam_slot_mask=raw['raw_seam_slot_mask'],source_face_kept=ca['source_face_kept'],
                    corner_target_input_ids=ca['corner_target_input_ids'],corner_vertex_ids=ca['corner_vertex_ids'],
                    view_center=center,view_half_extent=np.asarray(half),known_faces=raw['known_faces'])
            write(dest/'metadata.json',dict(selector=category,per_face=row,geometry_npz=str(dest/'geometry.npz'),
                interpretation='Geometry data for matched local views; semantic slat quality UNKNOWN; not a visual review'))
            selected.append(dict(selector=category,**row,geometry_npz=str(dest/'geometry.npz'),metadata_json=str(dest/'metadata.json')))
    csvwrite(OUT/'examples.csv',selected)
    return selected


def run():
    if OUT.exists() and (OUT/'completion.json').exists(): raise FileExistsError('Completed diagnostic is immutable')
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    qs=read(SOURCE/'evaluation_inputs/manifest.json')['requests'];assert len(qs)==48
    for label in LABELS:
        assert read(SOURCE/'evaluation'/label/'post_complete_full.json')['outputs']==48
    resources=resource_snapshot();write(OUT/'resources_before.json',resources)
    jobs=[(label,q) for label in LABELS for q in qs]
    # First actual case measures the same array-only workload, then two independent readers.
    first=work(jobs[0]);write(OUT/'dispatch.json',dict(workers=2,threads_per_worker=1,first_case_seconds=first['seconds'],
        observed_first_case_peak_RSS_KiB=first['peak_RSS_KiB'],reason='96 independent array cases; two CPU workers within authorized resource ceiling',GPU_workers=0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=[first,*list(pool.map(work,jobs[1:]))]
    faces=[r for x in results for r in x['rows']];stages=[r for x in results for r in x['stages']]
    outputs=[x['summary'] for x in results];hashes={p:h for x in results for p,h in x['hashes'].items()}
    csvwrite(OUT/'per_face.csv',faces);csvwrite(OUT/'per_output.csv',outputs);csvwrite(OUT/'per_output_stage_area.csv',stages)
    grouped=collections.defaultdict(list)
    for row in faces:
        if row['role']!='C': grouped[(row['label'],row['split'],row['role'],row['primary_attribution'])].append(row)
    aggregate=[]
    for (label,split,role,reason),rr in sorted(grouped.items()):
        aggregate.append(dict(label=label,split=split,role=role,primary_attribution=reason,faces=len(rr),
            RAW_area=sum(r['RAW_area'] for r in rr),D0_area=sum(r['D0_PROPOSAL_area'] for r in rr),PRE_GEO_area=sum(r['PRE_GEO_area'] for r in rr),
            CLUSTERED_area=sum(r['CLUSTERED_BEFORE_DELETE_area'] for r in rr),
            raw_already_below_R=sum(r['minimum_altitude_crossing']=='RAW_ALREADY_BELOW_R' for r in rr),
            D0_first_below_R=sum(r['minimum_altitude_crossing']=='D0_FIRST_BELOW_R' for r in rr),
            anchor_restore_first_below_R=sum(r['minimum_altitude_crossing']=='ANCHOR_RESTORE_FIRST_BELOW_R' for r in rr),
            PRE_not_below_R=sum(r['minimum_altitude_crossing']=='PRE_GEO_NOT_BELOW_R' for r in rr)))
    csvwrite(OUT/'attribution_summary.csv',aggregate)
    selected=examples(faces)
    summaries=[]
    for label in LABELS:
        rr=[r for r in faces if r['label']==label and r['role']!='C'];dd=[r for r in rr if not r['kept']]
        summaries.append(dict(label=label,outputs=48,free_slots=len(rr),deleted=len(dd),
            role_deleted=dict(collections.Counter(r['role'] for r in dd)),reasons=dict(collections.Counter(r['delete_reason'] for r in dd)),
            primary=dict(collections.Counter(r['primary_attribution'] for r in dd)),width_crossings=dict(collections.Counter(r['minimum_altitude_crossing'] for r in dd)),
            deleted_PRE_area=sum(r['PRE_GEO_area'] for r in dd),total_PRE_free_area=sum(r['PRE_GEO_area'] for r in rr),
            kept_below_R=sum(r['kept'] and r['pre_minimum_altitude_over_radius']<1 for r in rr),
            deleted_pre_min_edge_gt_R=sum(r['pre_shortest_edge_over_radius']>1 for r in dd)))
    assert 'torch' not in sys.modules
    result=dict(status='PASS',requests_per_label=48,total_requests=96,per_face_rows=len(faces),role_summary=summaries,
        cache_roundtrip_outputs=96,actual_GEO_replay_calls=0,model_calls=0,GPU_calls=0,scoring_calls=0,training_updates=0,
        geometric_definition=geo_c.FIXED_CONFIG,minimum_altitude_definition='2 * triangle area / longest edge; not semantic timber width',
        width_threshold='Descriptive comparison only to the existing frozen GEO radius; no new threshold or mesh modification',
        area_definition='Sum of triangle areas; not union surface coverage. Deleted PRE area is not proof of missing visible surface.',
        lineage='Generated RAW slot, not source-parent GT face',examples=len(selected),seconds=time.perf_counter()-start,
        input_hashes=hashes,code_hashes={str(Path(__file__)):sha(__file__),str(Path(geo_c.__file__)):sha(geo_c.__file__)},
        files={str(p):sha(p) for p in OUT.glob('*.csv')})
    write(OUT/'completion.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','files','geometric_definition')},ensure_ascii=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),flush=True)


def finalize_existing():
    """Serialization-only recovery from already-written audited CSV; no GEO replay."""
    tick=time.perf_counter()
    def readcsv(name):
        with (OUT/name).open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        for row in rows:
            for k,v in row.items():
                if v in ('True','False'):row[k]=v=='True'
                elif v=='':row[k]=None
                else:
                    try:row[k]=json.loads(v)
                    except (ValueError,TypeError):pass
        return rows
    faces=readcsv('per_face.csv');outputs=readcsv('per_output.csv');selected=readcsv('examples.csv')
    assert len(outputs)==96 and len({(r['label'],r['request_id']) for r in outputs})==96
    summaries=[]
    for label in LABELS:
        rr=[r for r in faces if r['label']==label and r['role']!='C'];dd=[r for r in rr if not r['kept']]
        summaries.append(dict(label=label,outputs=48,free_slots=len(rr),deleted=len(dd),
            role_deleted=dict(collections.Counter(r['role'] for r in dd)),reasons=dict(collections.Counter(r['delete_reason'] for r in dd)),
            primary=dict(collections.Counter(r['primary_attribution'] for r in dd)),width_crossings=dict(collections.Counter(r['minimum_altitude_crossing'] for r in dd)),
            deleted_PRE_area=sum(r['PRE_GEO_area'] for r in dd),total_PRE_free_area=sum(r['PRE_GEO_area'] for r in rr),
            kept_below_R=sum(r['kept'] and r['pre_minimum_altitude_over_radius']<1 for r in rr),
            deleted_pre_min_edge_gt_R=sum(r['pre_shortest_edge_over_radius']>1 for r in dd)))
    qs=read(SOURCE/'evaluation_inputs/manifest.json')['requests'];hashes={}
    for label in LABELS:
        for q in qs:
            d=SOURCE/'evaluation'/label/q['request_id'];receipt=read(d/'post/receipt.json')
            for p in (d/'raw/raw.npz',*[d/'post'/n for n in ('proposal.npz','pre_geo.npz','post_canonical.npz','post.npz','geometry.json','receipt.json','proposal_receipt.json')]):
                hashes[str(p)]=sha(p)
                if str(p) in receipt['files']:assert hashes[str(p)]==receipt['files'][str(p)]
    result=dict(status='PASS',requests_per_label=48,total_requests=96,per_face_rows=len(faces),role_summary=summaries,
        cache_roundtrip_outputs=96,actual_GEO_replay_calls=0,model_calls=0,GPU_calls=0,scoring_calls=0,training_updates=0,
        geometric_definition=geo_c.FIXED_CONFIG,minimum_altitude_definition='2 * triangle area / longest edge; not semantic timber width',
        width_threshold='Descriptive comparison only to existing GEO radius; no threshold sweep or mesh modification',
        area_definition='Triangle area sum, not union coverage; deleted area does not certify missing visible surface',
        lineage='Generated RAW slot, not source-parent GT face',examples=len(selected),input_hashes=hashes,
        engineering_recovery=dict(original_exit=1,reason='Final JSON serialization of numpy.int64; all geometry identity checks and CSV/example writes had completed',
            original_exec_wall_seconds=5.2706135,original_partial='completion.serialization_failure.json.partial',
            recovery='Read saved CSV only and reverify input hashes; no repeated geometry processing or scoring',seconds=time.perf_counter()-tick),
        code_hashes={str(Path(__file__)):sha(__file__),str(Path(geo_c.__file__)):sha(geo_c.__file__)},
        files={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json'})
    write(OUT/'completion.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','files','geometric_definition')},ensure_ascii=False))


if __name__=='__main__':
    if '--finalize-existing' in sys.argv:finalize_existing()
    else:run()
