"""Four fixed high-N RAWs through original GEO and frozen CPU cascade.

No model framework, D0, training, new queries/noise or parameter sweep.
Complete receipts are reused. Any partial science attempt stops for review.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, concurrent.futures, csv, hashlib, json, multiprocessing, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
from r2.pilots.bench_adaptation_strength_v1.common import resolve,read,sha,resources,write_new
from r2.pilots.bench_backend_counterfactual_v1 import core as BC
from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
from r2.pilots.bench_natural_n_recipe_v1 import evaluation as EV
OUT=ROOT/'experiments/r2_bench_high_joint_coverage_v1/backend_geo_only'
SOURCE=ROOT/'experiments/r2_bench_natural_n_recipe_v1'
VERSION='natural_high_fixed_raw_geo_only_full_cascade_v1'

def arrays(p):return BC.arrays(resolve(p))
def write(p,x):write_new(p,BC.clean(x))
def npz(p,x):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True);BC.npz(p,x)
def csvwrite(p,rows):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True);BC.csv_write(p,rows)
def verify(files):
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('Frozen file changed '+p)
def textnew(p,s):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(s)

def prepare():
    target=OUT/'manifest.json'
    if target.exists():
        m=read(target);verify(m['protected']);verify(m['code']);return m
    P,geo,io,code=BC.dependencies()
    from r2.pilots.chair_back_transfer_v1.evaluate import source_hashes
    code.update(source_hashes())
    for mod in (CPU,CPU.H,CPU.alpha,CPU.holes,CPU.M):code[str(Path(mod.__file__))]=sha(mod.__file__)
    code[str(Path(__file__))]=sha(__file__)
    source_manifest=SOURCE/'data/eval_manifest.json';m=read(source_manifest)
    qs=[q for q in m['requests'] if q['report_group']=='HIGH_TRAIN']
    assert len(qs)==4 and len({q['parent_id'] for q in qs})==2
    assert {q['noise_index'] for q in qs}=={0,1}
    protected={str(source_manifest):sha(source_manifest)};tasks=[]
    # Existing GEO_ONLY experiments only contain earlier Coverage RAW identities.
    checked=[]
    for folder in ('r2_bench_backend_counterfactual_v1','r2_bench_geo_only_acceptance_v1','r2_bench_raw_local_repair_v1'):
        p=ROOT/'experiments'/folder/'manifest.json'
        if p.exists():
            txt=p.read_text(encoding='utf-8-sig')
            checked.append(dict(path=str(p),sha256=sha(p),matching_request_ids=[q['request_id'] for q in qs if q['request_id'] in txt]))
    if any(x['matching_request_ids'] for x in checked):raise RuntimeError('Possible matching historical geometry must be identity-checked before execution')
    for q in qs:
        rid=q['request_id'];rawdir=EV.producer('B',q,'raw');postdir=EV.producer('B',q,'post')
        oldcpu=SOURCE/'cpu/B'/rid;oldrec=read(oldcpu/'complete.json');verify(oldrec['files'])
        source=rawdir/'source_condition.npz'
        paths=dict(raw=str(rawdir/'raw.npz'),source_condition=str(source),old_pre=str(postdir/'pre_geo.npz'),
            old_canonical=str(postdir/'post_canonical.npz'),old_base=str(postdir/'post.npz'),
            old_geometry=str(postdir/'geometry.json'),old_final=str(oldcpu/'CLEAN_FINAL_POST/post.npz'))
        for p in list(paths.values())+[str(rawdir/'receipt.json'),str(postdir/'receipt.json'),str(oldcpu/'complete.json'),q['inference_npz'],q['query_descriptor']['path']]:
            protected[str(resolve(p))]=sha(p)
        for stage in ('RAW','BASE_POST','CLEAN_FINAL_POST'):
            row,score,folder=EV.authenticate_score('B',q,stage)
            for name in ('row.json','score.json','complete.json'):protected[str(folder/name)]=sha(folder/name)
        r=arrays(paths['raw']);s=arrays(source)
        assert r['raw'].shape==(int(q['N']),3,3) and int(r['known_faces'])==int(q['K'])
        assert not r['raw_seam_slot_mask'].any()
        assert s['C_original'].shape==(int(q['K']),3,3)
        tasks.append(dict(request=q,paths=paths,reuse=False,output_dir=str(OUT/'outputs'/rid)))
    OUT.mkdir(parents=True,exist_ok=True)
    textnew(OUT/'PROTOCOL.md', '''# Frozen high-N GEO(RAW) backend pilot

Exactly four existing NATURAL_HIGH HIGH_TRAIN outputs (28b9/abf, n0/n1).
RAW -> original GEO_C -> original source-winding export -> HINGE_OPP015
(opposite winding required) -> ALPHA -> HOLE. Compare the same model/noise's
original single-D0 FINAL; keep original RAW and BASE scores read-only.
No model/D0/training/OT/new noise/new query, no threshold scan, no data editing.
Original C geometry/index identity and all guards remain unchanged.
Original 177 metric fields are preserved. I has zero interior support for both
conditions and is NA in primary comparisons; raw numeric fields remain intact.
Face counts/areas/displacements are generated-slot paired, not GT reconstruction.
Summed triangle area counts overlapping surfaces and is not semantic completeness.
Actual nonblind AI_PROXY review checks both fixed views and both original sections,
particularly 28b9 seat-board sides and abf end supports. No default promotion.
CPU at most two workers, one thread each; freeze source/code hashes before work.
Each complete receipt is verified and reused; a partial attempt requires explicit
engineering review and is never silently replayed. Exactly one GEO and one each
HINGE/ALPHA/HOLE per new request, two expanded scores per request.
''')
    write(OUT/'resources_before.json',resources())
    result=dict(schema=VERSION,status='FROZEN',tasks=tasks,protected=protected,code=code,
        budget=dict(requests=4,GEO=4,source_orientation=4,HINGE=4,ALPHA=4,HOLE=4,expanded_scores=8,
                    model=0,D0=0,training=0,OT=0,new_noise=0,new_queries=0),
        cache_check=checked,cache_conclusion='No matching high-N RAW identity in prior GEO_ONLY manifests; source natural-N has only registered D0 chain',
        original_radius=geo.RADIUS,geo_config=geo.FIXED_CONFIG,metric_fields=m['metric_fields'],
        resources=str(OUT/'resources_before.json'),CPU_workers=2,CPU_threads=1,
        prior_route='Bench GEO_ONLY acceptance: shape preservation possible with conflict costs; same backend, new high-N generator outputs, not new mechanism',
        protocol_sha256=sha(OUT/'PROTOCOL.md'))
    write(target,result);return result

def geometry(task,manifest_sha):
    dest=resolve(task['output_dir'])/'geometry';done=dest/'receipt.json'
    if done.exists():
        r=read(done);assert r['manifest_sha256']==manifest_sha;verify(r['files']);return r
    if dest.exists():raise RuntimeError('Partial GEO attempt requires review: '+str(dest))
    P,geo,io,_=BC.dependencies();q=task['request'];paths=task['paths'];start=time.perf_counter();calls=0
    write(dest/'attempt.json',dict(manifest_sha256=manifest_sha,request_id=q['request_id'],GEO_budget=1))
    try:
        raw=arrays(paths['raw']);source=arrays(paths['source_condition']);k=int(q['K'])
        pre,canonical,post,g,orient,anchors=P.geometry_from_proposal(raw,raw['raw'].copy(),source);calls=1
        assert np.array_equal(pre,raw['raw']) and g['stats']['radius_model']==geo.RADIUS
        npz(dest/'pre_geo.npz',dict(output=pre,C=pre[:k],known_faces=raw['known_faces'],raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
        npz(dest/'post_canonical.npz',canonical);npz(dest/'post.npz',post);io.write_indexed_ply(dest/'post.ply',post['vertices'],post['faces'])
        write(dest/'geometry.json',dict(stats=g['stats'],face_records=g['face_records']))
        write(dest/'anchor_identity.json',anchors);write(dest/'source_restoration.json',orient['restoration_summary'])
        od=orient['orientation_result'];write(dest/'orientation.json',dict(summary=od['summary'],components=od['component_rows'],edges=od['edge_rows']))
        pairs=CPU.H.scan(post['output'],k);write(dest/'pairs.json',pairs)
        receipt=dict(status='COMPLETE',request_id=q['request_id'],manifest_sha256=manifest_sha,GEO_calls=1,source_orientation_calls=1,
            model_calls=0,D0_calls=0,new_noise=0,new_queries=0,full_pair_scans=1,seconds=time.perf_counter()-start,
            files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
        write(done,receipt);return receipt
    except BaseException as exc:
        write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),GEO_returns=calls,no_silent_replay=True));raise

def one(job):
    task,msha=job;q=task['request'];dest=resolve(task['output_dir'])
    geometry(task,msha);g=dest/'geometry'
    r=CPU.process_one(q,'B_GEO_ONLY',g/'post.npz',g/'pre_geo.npz',task['paths']['raw'],g/'pairs.json',dest/'cascade',False,msha)
    assert 'torch' not in sys.modules
    return r

def run():
    m=prepare();done=OUT/'compute_complete.json'
    if done.exists():
        r=read(done);verify(r['files']);return r
    tick=time.perf_counter();jobs=[(t,sha(OUT/'manifest.json')) for t in m['tasks']]
    before=resources();write(OUT/'dispatch.json',dict(resources=before,workers=2,threads=1,
        decision='Four independent CPU geometry/scoring tasks; memory/CPU headroom observed, no GPU jobs. Fixed maximum two workers.'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=list(pool.map(one,jobs))
    verify(m['protected']);verify(m['code'])
    files={str(resolve(t['output_dir'])/s):sha(resolve(t['output_dir'])/s) for t in m['tasks'] for s in ('geometry/receipt.json','cascade/complete.json')}
    value=dict(status='COMPLETE',requests=4,GEO=4,source_orientation=4,HINGE=4,ALPHA=4,HOLE=4,expanded_scores=8,
        model_calls=0,D0_calls=0,new_noise=0,new_queries=0,CPU_workers=2,CPU_threads=1,seconds=time.perf_counter()-tick,
        files=files,resources_finished=resources())
    write(done,value);return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run','verify'));a=p.parse_args()
    if a.command=='prepare':r=prepare()
    elif a.command=='run':r=run()
    else:
        m=prepare();r=read(OUT/'compute_complete.json');verify(r['files'])
        for t in m['tasks']:
            for f in ('geometry/receipt.json','cascade/complete.json'):verify(read(resolve(t['output_dir'])/f)['files'])
    print(json.dumps({k:v for k,v in r.items() if k not in ('tasks','protected','code','files','resources_finished')},ensure_ascii=False))
