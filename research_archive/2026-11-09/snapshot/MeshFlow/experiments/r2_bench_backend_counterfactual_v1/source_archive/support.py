"""Frozen RAW-only below-C candidate region; no semantic or GT matching claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_backend_counterfactual_v1'
OLD=ROOT/'experiments/r2_bench_seat_coverage_v1'
L=9.208138264587333
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def table(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prepare():
    p=OUT/'support/registration.json'
    if p.exists():
        r=read(p)
        for x in r['rows']:assert sha(x['raw_path'])==x['raw_sha256']
        return r
    qs=[q for q in read(OLD/'eval_manifest.json')['requests'] if q['parent_id'].startswith(('a8','27'))]
    rows=[];faces=[]
    for q in qs:
        for label in ('NATIVE_S6000','COVERAGE_FINAL'):
            rawpath=OLD/'evaluation'/label/q['request_id']/'raw/raw.npz'
            a=arr(rawpath);k=int(a['known_faces']);tri=a['raw'].reshape(-1,3,3).astype(float)
            ymin=float(tri[:k,:,1].min());chosen=[]
            for i,t in enumerate(tri[k:],k):
                low=float(t[:,1].min());mid=float(t[:,1].mean());yes=low<ymin-.02*L and mid<ymin
                if yes:chosen.append(i)
                faces.append(dict(label=label,request_id=q['request_id'],slot_id=i,raw_min_y=low,
                    raw_centroid_y=mid,C_min_y=ymin,below_C_candidate=yes))
            rows.append(dict(label=label,request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],
                raw_path=str(rawpath),raw_sha256=sha(rawpath),K=k,N=len(tri),C_min_y=ymin,slots=chosen))
    assert len(rows)==8
    r=dict(status='FROZEN',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),rows=rows,
        rule='free face minY < min(C.Y)-0.02L AND centroidY < min(C.Y)',L=L,
        interpretation='RAW below-C candidate region, not exhaustive semantic leg labels; may include underside/walls and miss high-centroid frames.',
        selection_inputs='Only original RAW and its C; no new GEO/POST outcome used; no cross-model slot correspondence.',
        raw_gallery_review='Root actually viewed four original paired RAW pages (a8/27 n0/n1), nonblind AI_PROXY.',
        observations={'a8_NATIVE':'panel end supports/crossbar; n1 box-like body',
                      'a8_COVERAGE':'four thin leg-like supports visible in both noises',
                      '27_NATIVE':'large back/end panels with low crossbar',
                      '27_COVERAGE':'back panel plus slim end arm/leg frames in both noises'})
    table(OUT/'support/raw_selection_per_face.csv',faces);save(p,r)
    print(json.dumps({'stage':'SUPPORT_FROZEN','counts':[(x['label'],x['parent_id'][:2],x['noise_index'],len(x['slots'])) for x in rows]}))
    return r
def stats(tri):
    tri=np.asarray(tri).reshape(-1,3,3)
    if not len(tri):return dict(count=0,area=0.,x_span=0.,y_span=0.,z_span=0.,min_height_median=None,min_edge_median=None)
    lens=np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2)
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    return dict(count=len(tri),area=float(area.sum()),**dict(zip(('x_span','y_span','z_span'),np.ptp(tri.reshape(-1,3),axis=0).tolist())),
        min_height_median=float(np.median(2*area/np.maximum(lens.max(1),1e-30))),min_edge_median=float(np.median(lens.min(1))))
def analyse():
    r=prepare();m=read(OUT/'manifest.json');lookup={(t['label'],t['request_id']):t for t in m['tasks'] if t['kind']=='RAW_GEO'}
    rows=[];checks=[]
    for x in r['rows']:
        t=lookup[x['label'],x['request_id']];orig=t['original'];ids=np.array(x['slots'],dtype=int)
        raw=arr(x['raw_path'])['raw'].reshape(-1,3,3)
        prop=arr(orig['proposal_path'])['official_unconstrained_proposal'].reshape(-1,3,3)
        pre=arr(orig['pre_geo_path'])['output'].reshape(-1,3,3)
        checks.append(dict(label=x['label'],request_id=x['request_id'],free_proposal_equals_restored_pre=np.array_equal(prop[x['K']:],pre[x['K']:]),
            max_C_restore_displacement=float(np.linalg.norm(prop[:x['K']]-pre[:x['K']],axis=-1).max())))
        stages={'RAW':raw[ids],'D0_PROPOSAL':prop[ids],'PRE_GEO':pre[ids]}
        for name,path in [('BASE_POST',orig['base_post_path']),('GEO_RAW',t['new_post_path'])]:
            a=arr(path);mask=np.isin(a['source_face_ids'],ids);stages[name]=a['vertices'][a['faces'][mask]]
        initial=stats(stages['RAW'])
        for stage,tris in stages.items():
            s=stats(tris);rows.append(dict(label=x['label'],request_id=x['request_id'],parent_id=x['parent_id'],noise_index=x['noise_index'],stage=stage,
                **s,raw_selected_count=len(ids),raw_area=initial['area'],area_ratio_to_raw=s['area']/initial['area'] if initial['area'] else None))
    table(OUT/'support/per_output_stage.csv',rows);save(OUT/'support/restoration_check.json',checks)
    print(json.dumps({'stage':'SUPPORT_DONE','rows':len(rows),'all_free_restore_unchanged':all(x['free_proposal_equals_restored_pre'] for x in checks)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analyse',action='store_true');a=p.parse_args()
    analyse() if a.analyse else prepare()
