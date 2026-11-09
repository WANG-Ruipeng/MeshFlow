"""CPU-only, parent-disjoint identity/synthetic data for bounded D0 calibration.

No generated RAW is paired with a source target. Source arrays are immutable.
Existing reviewed complete-target audits are reused only after identity checks;
held-out source versions retain independent known-defect masks.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse
from collections import Counter,defaultdict
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from r2.data_tools.bench_parts_v1 import quality as Q

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_backend_shape_preservation_v1/data'
POOL=ROOT/'experiments/r2_bench_user_data_v2_expansion_v1/manifest.json'
EVAL=ROOT/'experiments/r2_bench_seat_coverage_v1/eval_manifest.json'
SEED=2026102307
SCALE=float(.95*.3762)
L=float(np.sqrt(3)*2/.3762)

def resolve(p):
    p=Q.local_path(p)
    return p if p.is_absolute() else ROOT/p
def read(p):return json.loads(resolve(p).read_text('utf-8-sig'))
def arrays(p):
    with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return Q.sha(resolve(p))
def write(p,x):Q.write_new(resolve(p),x)
def npz(p,a):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        b=arrays(p)
        assert set(a)==set(b) and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in a),str(p)
    else:
        with p.open('xb') as f:np.savez_compressed(f,**a)
def verify(book):
    for p,h in book.items():assert sha(p)==h,p
def csvwrite(p,rows):
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with resolve(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,cols);w.writeheader();w.writerows(rows)

def source_arrays(case):
    n,k=int(case['N']),int(case['K'])
    ids=np.r_[case['known_ids'],case['free_ids']].astype(np.int64)
    assert len(ids)==n and np.array_equal(np.sort(ids),np.arange(n))
    target=np.concatenate((case['C'],case['Y'])).astype(np.float32)
    assert np.array_equal(target,case['full_target'][ids])
    vids=case['source_face_vertex_ids'][ids].astype(np.int64)
    assert np.array_equal(case['model_vertices'][vids],target)
    return target,ids,vids,k

def defect_mask(case,audit=None):
    target,ids,vids,k=source_arrays(case);n=len(target)
    if audit is None:
        au=Q.SourceAuditor(case);illegal=[];unknown=[];counts=Counter()
        for i in range(n):
            for j in range(i+1,n):
                r=au.pair(i,j);counts[r['classification']]+=1
                if r.get('illegal') is True:illegal.append(dict(face_i=i,face_j=j,detector=r))
                if not r.get('verified',False):unknown.append(dict(face_i=i,face_j=j,detector=r))
        audit=dict(illegal_pairs=illegal,unresolved_pairs=unknown,classifications=dict(counts),detector_calls=dict(au.calls),scope='recomputed held-out source complete finite pairs; not generated-output scoring')
    bad=defaultdict(list)
    for name in ('illegal_pairs','unresolved_pairs'):
        for r in audit[name]:
            for fi in (int(r['face_i']),int(r['face_j'])):bad[fi].append(name)
    full=np.asarray(case['full_target'],np.float64)
    area=np.linalg.norm(np.cross(full[:,1]-full[:,0],full[:,2]-full[:,0]),axis=1)/2
    for fi in np.flatnonzero(area==0):bad[int(fi)].append('EXACT_ZERO_AREA')
    groups=defaultdict(list)
    for fi,tri in enumerate(full):groups[tuple(sorted(map(tuple,tri)))].append(fi)
    duplicates=[g for g in groups.values() if len(g)>1]
    for g in duplicates:
        for fi in g:bad[fi].append('EXACT_COORDINATE_DUPLICATE_ALL_COPIES')
    good=np.asarray([int(fi) not in bad for fi in ids],bool);good[:k]=False
    mask=np.repeat(good[:,None],3,axis=1)
    if not mask.any():raise ValueError('No qualified held-out free corner')
    details=dict(source_face_bad_reasons={str(fi):sorted(set(reasons)) for fi,reasons in bad.items()},
        duplicate_groups=duplicates,exact_zero_area_source_faces=np.flatnonzero(area==0).tolist(),
        source_illegal_pairs=len(audit['illegal_pairs']),source_unresolved_pairs=len(audit['unresolved_pairs']),
        free_faces=n-k,masked_free_faces=int(good.sum()),excluded_free_faces=int((~good[k:]).sum()),
        mask_rule='Exclude every free face incident to verified illegal or unresolved source pair, any exact-zero area or any coordinate duplicate; C has no loss. Mask is source-only and fixed across perturbations.',
        claim='Qualified supervised free faces under frozen detector, not a globally perfect/manifold/semantic certificate')
    return target,ids,vids,k,mask,audit,details

def choose_sources():
    pool=read(POOL);evalm=read(EVAL)
    evaluation={q['parent_id']:q for q in evalm['requests']}
    assert len(evaluation)==15 and len(pool['parents'])==25
    groups=defaultdict(list)
    for r in pool['train']:groups[r['parent_id']].append(r)
    selected=[]
    for pr in sorted(pool['parents'],key=lambda r:r['parent_id']):
        uid=pr['parent_id']
        if uid in evaluation:continue
        assert pr.get('eligibility','ACCEPTED_TARGET')=='ACCEPTED_TARGET' and pr['review']['actually_viewed'] and pr['review']['decision']=='ACCEPT'
        options=groups[uid]
        options.sort(key=lambda r:(r['part_type']!='SEAT_SURFACE',r['part_type'],r['condition_id']))
        r=options[0]
        assert r['review']['actually_viewed'] and r['review']['decision']=='ACCEPT'
        verify(pr['artifact_hashes']);assert sha(r['case_path'])==r['case_sha256']
        case=arrays(r['case_path']);target=arrays(pr['target_case_path']);audit=read(pr['geometry_path'])
        for key in ('model_vertices','source_faces','source_face_vertex_ids','source_corner_permutations','full_target'):
            assert np.array_equal(case[key],target[key]),(uid,key,'condition target version mismatch')
        assert not audit['illegal_pairs'] and not audit['unresolved_pairs'] and not audit['exact_zero_area_faces']
        for key in ('nonmanifold_edges','same_winding_double_edges'):
            assert not audit['topology'][key],(uid,key)
        selected.append(dict(parent_id=uid,condition_id=r['condition_id'],case_path=r['case_path'],case_sha256=sha(r['case_path']),
            parent_record=pr,condition_record=r,audit_path=pr['geometry_path'],case=case,audit=audit,split='TRAIN'))
    held=[]
    for uid,q in sorted(evaluation.items()):
        assert sha(q['case_path'])==q['case_sha256']
        case=arrays(q['case_path']);supplement=None
        if 'model_vertices' not in case:
            supplement=q['source_case_path'];assert sha(supplement)==q['source_case_sha256']
            source=arrays(supplement)
            for key in ('full_target','pre_ot_full','source_face_vertex_ids'):
                assert np.array_equal(case[key],source[key]),(uid,key,'historical source mismatch')
            for key in ('model_vertices','source_faces','source_corner_permutations'):
                case[key]=source[key].copy()
        held.append(dict(parent_id=uid,condition_id=q['condition_id'],case_path=q['case_path'],case_sha256=sha(q['case_path']),
            source_supplement=supplement,source_supplement_sha256=sha(supplement) if supplement else None,
            condition_record=q,case=case,audit=None,split='VALIDATION'))
    assert len(selected)==16 and not ({x['parent_id'] for x in selected}&set(evaluation))
    return selected,held,pool,evalm

def instance(src,kind,sigma,seed,path):
    target=src['target'];vids=src['vids'];k=src['K'];rng=np.random.RandomState(seed)
    delta=np.zeros_like(target)
    if kind=='CORNER_NOISE':delta[k:]=rng.standard_normal(delta[k:].shape).astype(np.float32)*np.float32(sigma/SCALE)
    elif kind=='VERTEX_NOISE':
        values=rng.standard_normal((int(vids.max())+1,3)).astype(np.float32)*np.float32(sigma/SCALE)
        delta[k:]=values[vids[k:]]
    elif kind!='IDENTITY':raise ValueError(kind)
    x=(target+delta).astype(np.float32);assert np.array_equal(x[:k],target[:k])
    a=dict(x_model=x,target_model=target,loss_mask=src['mask'],known_faces=np.asarray(k,np.int64),
        source_face_ids=src['ids'],source_vertex_ids=vids,delta_model=delta,
        noise_seed=np.asarray(seed,np.int64),sigma_denoiser=np.asarray(sigma,np.float64),denoiser_scale=np.asarray(SCALE,np.float64))
    npz(path,a)
    return dict(id=path.stem,row_id=path.stem,path=str(path),sha256=sha(path),parent=src['parent_id'],parent_id=src['parent_id'],
        condition_id=src['condition_id'],kind=kind,sigma=sigma,noise_seed=seed,N=len(target),K=k,
        mask_coordinates=int(src['mask'].sum())*3,case_path=src['case_path'],case_sha256=src['case_sha256'],
        source_artifact=str(src['source_artifact']),source_sha256=sha(src['source_artifact']))

def render(sources):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    gallery=OUT/'gallery';gallery.mkdir(exist_ok=True)
    pages=[]
    for split in ('TRAIN','VALIDATION'):
        ss=[s for s in sources if s['split']==split]
        fig=plt.figure(figsize=(16,4*((len(ss)+3)//4)),dpi=140)
        for i,s in enumerate(ss):
            ax=fig.add_subplot((len(ss)+3)//4,4,i+1,projection='3d');t=s['target'];k=s['K'];good=s['mask'][:,0]
            colors=['#269fd0' if j<k else '#9ca3af' if good[j] else '#d66a4a' for j in range(len(t))]
            ax.add_collection3d(Poly3DCollection(t,facecolors=colors,edgecolors='#4b5563',linewidths=.12,alpha=1))
            v=t.reshape(-1,3);center=(v.min(0)+v.max(0))/2;rad=(v.max(0)-v.min(0)).max()/2*1.06
            ax.set_xlim(center[0]-rad,center[0]+rad);ax.set_ylim(center[1]-rad,center[1]+rad);ax.set_zlim(center[2]-rad,center[2]+rad)
            # Model Y is vertical; view standard geometry coordinates via matching limits.
            ax.view_init(elev=25,azim=-65,vertical_axis='y');ax.set_box_aspect((1,1,1));ax.set_axis_off()
            ax.set_title(s['parent_id'][:6]+f" N={len(t)} K={k}\nmask {int(good.sum())}/{len(t)-k} free faces",fontsize=9)
        fig.suptitle(split+' SOURCE TARGETS | blue C / gray supervised free / red excluded known defects\nActual source meshes, no denoiser/generator outputs',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.95));path=gallery/(split.lower()+'.png');fig.savefig(path);plt.close(fig);pages.append(str(path))
    (gallery/'index.html').write_text('<meta charset="utf-8"><h1>Source/mask audit</h1>'+''.join('<img style="width:100%" src="'+Path(p).name+'">' for p in pages),encoding='utf8')
    return pages

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);done=OUT/'manifest.json'
    if done.exists():
        m=read(done);verify(m['protected_inputs']);verify(m['artifact_hashes']);print('DATA_VERIFIED',len(m['train_rows']),len(m['validation_rows']));return m
    tick=time.perf_counter();selected,held,pool,evalm=choose_sources()
    protected={str(POOL):sha(POOL),str(EVAL):sha(EVAL)};sources=[];mask_rows=[]
    for s in selected+held:
        protected[s['case_path']]=s['case_sha256']
        if s.get('source_supplement'):protected[s['source_supplement']]=s['source_supplement_sha256']
        if s.get('parent_record'):
            protected.update(s['parent_record']['artifact_hashes'])
        t,ids,vids,k,mask,audit,detail=defect_mask(s['case'],s['audit'])
        dest=OUT/'sources'/s['parent_id'];dest.mkdir(parents=True,exist_ok=True)
        npz(dest/'source.npz',dict(target_model=t,loss_mask=mask,source_face_ids=ids,source_vertex_ids=vids,known_faces=np.asarray(k,np.int64)))
        write(dest/'audit.json',audit);write(dest/'mask.json',detail)
        provenance={key:value for key,value in s.items() if key not in ('case','audit') and value is not None}
        write(dest/'provenance.json',provenance)
        s.update(target=t,ids=ids,vids=vids,K=k,mask=mask,source_artifact=dest/'source.npz')
        sources.append(s);mask_rows.append(dict(parent_id=s['parent_id'],condition_id=s['condition_id'],split=s['split'],N=len(t),K=k,
            existing_nonmanifold_vertices=len(audit.get('topology',{}).get('nonmanifold_vertices',[])) if 'topology' in audit else 'NA_NOT_RECOMPUTED',
            **{x:detail[x] for x in ('free_faces','masked_free_faces','excluded_free_faces','source_illegal_pairs','source_unresolved_pairs')}))
        print('SOURCE',s['split'],s['parent_id'][:6],detail['masked_free_faces'],'/',len(t)-k,flush=True)
    rng=np.random.RandomState(SEED);parent_sequence=np.concatenate([rng.permutation(16) for _ in range(125)])
    train_rows=[];stream=[];event_rows=[];exposure=Counter();kind_exposure=Counter()
    for step in range(500):
        batch=[]
        for micro in range(4):
            event=step*4+micro;src=selected[int(parent_sequence[event])]
            kind=('IDENTITY','IDENTITY','CORNER_NOISE','VERTEX_NOISE')[micro]
            sigma=0. if kind=='IDENTITY' else (.01,.03,.05)[step%3]
            seed=SEED+100000+event
            path=OUT/'train'/f'event_{event:04d}.npz'
            row=instance(src,kind,sigma,seed,path);row.update(step=step+1,micro=micro)
            train_rows.append(row);batch.append(event);exposure[src['parent_id']]+=1;kind_exposure[(src['parent_id'],kind)]+=1
            event_rows.append(dict(step=step+1,micro=micro,row_index=event,parent_id=src['parent_id'],kind=kind,sigma=sigma,seed=seed,sha256=row['sha256']))
        stream.append(batch)
    assert set(exposure.values())=={125} and len(exposure)==16
    validation=[]
    for i,src in enumerate(held):
        for kind in ('IDENTITY','CORNER_NOISE'):
            sigma=0. if kind=='IDENTITY' else .03
            tag='IDENTITY' if kind=='IDENTITY' else 'CORNER_SIGMA003'
            validation.append(instance(src,kind,sigma,SEED+200000+i,OUT/'validation'/(src['parent_id']+'__'+tag+'.npz')))
    csvwrite(OUT/'source_mask_summary.csv',mask_rows);csvwrite(OUT/'training_events.csv',event_rows)
    csvwrite(OUT/'parent_kind_exposure.csv',[dict(parent_id=p,kind=k,count=v) for (p,k),v in sorted(kind_exposure.items())])
    pages=render(sources)
    oldgeo=ROOT/'experiments/r2_bench_geo_only_acceptance_v1/manifest.json'
    old=read(oldgeo);protected[str(oldgeo)]=sha(oldgeo)
    raw_rows=[]
    for task in old['tasks']:
        path=task['original']['raw_path'];protected[path]=sha(path)
        raw_rows.append(dict(request_id=task['request_id'],row_id=task['request_id'],path=path,sha256=sha(path),
            parent_id=task['parent_id'],case_path=task['case_path'],request=task['request']))
    assert len(raw_rows)==30
    artifacts={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    m=dict(schema='BENCH_BACKEND_SHAPE_PRESERVATION_DATA_V1',status='READY',seed=SEED,denoiser_scale=SCALE,
        train_parents=sorted(exposure),validation_parents=sorted(s['parent_id'] for s in held),parent_overlap=[],
        train_rows=train_rows,stream=stream,validation_rows=validation,raw_rows=raw_rows,parent_exposure=dict(exposure),
        source_rows=mask_rows,protected_inputs=protected,artifact_hashes=artifacts,data_code_sha256=sha(__file__),
        supervised_examples=2000,updates=500,batch=4,identity_examples=1000,independent_noise_examples=500,vertex_correlated_noise_examples=500,
        perturbation='Sigma0.01/0.03/0.05 in official denoiser coordinate units; divide by .95*.3762 to model units; C untouched; same source vertex shares noise across free corners only.',
        split_policy='All15 existing Coverage evaluation parents excluded from trusted USERv2 target pool25, leaving16; source versions match complete target audit hashes and actual recorded AI_PROXY acceptance.',
        selection='One original reviewed condition per TRAIN parent: SEAT_SURFACE priority, then part_type and condition_id lexical. No generated-result-based selection.',
        objective='Per-instance MSE on qualified free coordinates only, 2identity+1independentcorner+1sourcevertexcorrelated per batch; parent balanced125 each. No C loss and no separate weighting.',
        independent_generalization='Held out from this denoiser adaptation only; official pretraining overlap unknown; development panel repeatedly inspected.',
        genuine_RAW_repair_pairs=0,generator_calls=0,teacher_calls=0,training_calls=0,
        limitations=['No real generated-RAW trusted-repair labels. Actual Coverage RAW is transfer evaluation only.',
            'Training targets passed finite-pair, edge-topology/winding audits and prior AI_PROXY view; four TRAIN parents have one known nonmanifold vertex each. Legal point contacts are not automatically masked. Not a universal semantic or manifold certification.',
            'Validation source versions are exact current panel versions; source defects are masked and separately counted, never repaired.',
            'No source identities are passed to the denoiser; they only define corruption and supervision/masks.'],
        gallery_pages=pages,seconds=time.perf_counter()-tick)
    verify(protected);write(done,m);print('DATA_READY',len(train_rows),len(validation),m['seconds'],flush=True);return m

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify'],nargs='?',default='prepare');a=p.parse_args()
    if a.command=='verify' and not (OUT/'manifest.json').exists():raise FileNotFoundError('Data not prepared')
    prepare()
