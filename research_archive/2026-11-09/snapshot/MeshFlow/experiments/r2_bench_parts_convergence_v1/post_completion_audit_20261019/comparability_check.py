"""Read-only CPU coordinate/config/probe decomposition; no model or OT calls."""
import collections, csv, io, json, os, tarfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import numpy as np

ROOT=Path('/mnt/e/MeshFlow'); EXP=ROOT/'experiments/r2_bench_parts_convergence_v1'; OUT=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,x):
    with (OUT/p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def csvout(name,rows):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
old=read(ROOT/'experiments/r2_bench_control_transfer_v1/data/manifest.json')
man=read(EXP/'data/manifest.json'); cases={}; parentrows=[]
with tarfile.open(ROOT/'downloaded_data/archives/shapenet-cls.tar.gz') as archive:
    for r in old['train']+old['dev']:
        a=arrays(r['case_path']);cases[r['parent_id']]=a
        with np.load(io.BytesIO(archive.extractfile(r['source_archive_member']).read()),allow_pickle=False) as z:
            rawv=z['vertices'];rawf=z['faces']
        v=rawv.astype(np.float32);f=rawf.astype(np.int64)
        bounds=np.array([v.min(0),v.max(0)]);nv=v-(bounds[0]+bounds[1])[None,:]/2;nv=nv/(bounds[1]-bounds[0]).max()
        bounds=np.array([nv.min(0),nv.max(0)]);nv=nv-(bounds[0]+bounds[1])[None,:]/2
        bounds=np.array([nv.min(0),nv.max(0)]);nv=nv-(bounds[0]+bounds[1])[None,:]/2;nv=(nv/(bounds[1]-bounds[0]).max()).clip(-.5,.5)
        order=np.lexsort((nv[:,0],nv[:,2],nv[:,1]));rank=np.argsort(order)
        perms=np.argsort(rank[f],axis=1,kind='stable');tri=(nv/np.float32(.3762))[np.take_along_axis(f,perms,axis=1)]
        first=np.lexsort((tri[:,:,2],tri[:,:,1],tri[:,:,0]),axis=1)[:,0]
        perms=np.take_along_axis(perms,(first[:,None]+np.arange(3)[None,:])%3,axis=1)
        expected=(nv/np.float32(.3762))[np.take_along_axis(f,perms,axis=1)]*np.float32(2)
        parentrows.append(dict(parent_id=r['parent_id'],split=r['group'],official_split=r['official_split'],N=len(f),
            source_vertices_equal=bool(np.array_equal(v,a['source_vertices'])),source_faces_equal=bool(np.array_equal(f,a['source_faces'])),
            normalized_equal=bool(np.array_equal(nv,a['normalized_vertices'])),target_canonical_equal=bool(np.array_equal(expected,a['full_target'])),
            normalized_max_span=float(np.ptp(nv,axis=0).max()),model_max_span=float(np.ptp(expected.reshape(-1,3),axis=0).max()),
            target_scalar_rms=float(np.sqrt(np.mean(expected.astype(float)**2))),target_min=float(expected.min()),target_max=float(expected.max())))
checks=[]
for r in man['rows']:
    a=arrays(r['case_path']);full=a['full_target'].reshape(-1,3,3);k=int(a['K']);n=int(a['N'])
    checks.append(dict(condition_id=r['condition_id'],parent_id=r['parent_id'],split=r['group'],official_split=r['official_split'],part=r['part_type'],N=n,K=k,
        closed_C=int(r['boundary_edges'])==0,full_source_equal=bool(np.array_equal(full,cases[r['parent_id']]['full_target'])),
        pre_times_two_equal=bool(np.array_equal(a['pre_ot_full']*np.float32(2),full)),
        C_source_equal=bool(np.array_equal(a['C'],full[a['known_ids']])),Y_source_equal=bool(np.array_equal(a['Y'],full[a['free_ids']])),
        C_index_equal=bool(np.array_equal(a['C_vertices'][a['C_faces_model']],a['C'])),
        partition_equal=bool(np.array_equal(np.sort(np.r_[a['known_ids'],a['free_ids']]),np.arange(n))),
        C_scalar_rms=float(np.sqrt(np.mean(a['C'].astype(float)**2))),Y_scalar_rms=float(np.sqrt(np.mean(a['Y'].astype(float)**2))),
        C_max_span=float(np.ptp(a['C'].reshape(-1,3),axis=0).max())))
csvout('coordinate_parent_checks.csv',parentrows);csvout('coordinate_condition_checks.csv',checks)
probe=read(EXP/'training/COMMON/fixed_loss/step0.json');pm=read(EXP/'fixed_loss_probe/manifest.json')
lookup={r['probe_id']:r for r in pm['rows']};clook={r['condition_id']:r for r in checks}
prows=[]
for r in probe['rows']:
    q=lookup[r['probe_id']];c=clook[r['condition_id']];z=arrays(q['noise_npz'])
    noise=z.get('z',z.get('raw_noise'));eps=z.get('epsilon',z.get('eps'))
    prows.append(dict(condition_id=r['condition_id'],parent_id=r['parent_id'],split=r['split'],official_split=q['official_split'],part=r['part'],t=r['t'],N=c['N'],K=c['K'],
        mse=r['all_free_velocity_sse']/r['all_free_scalar_count'],sse=r['all_free_velocity_sse'],count=r['all_free_scalar_count'],
        C_scalar_rms=c['C_scalar_rms'],Y_scalar_rms=c['Y_scalar_rms'],noise_shape=list(noise.shape),noise_rms=float(np.sqrt(np.mean(noise.astype(float)**2))),
        epsilon_rms=float(np.sqrt(np.mean(eps.astype(float)**2))) if eps is not None else None))
csvout('step0_probe_rows.csv',prows)
groups={}
for fields in [('split',),('split','t'),('split','part'),('split','official_split'),('split','parent_id')]:
    vals=collections.defaultdict(lambda:[0.,0,0])
    for r in prows:
        key=tuple(r[f] for f in fields);vals[key][0]+=r['sse'];vals[key][1]+=r['count'];vals[key][2]+=1
    groups['+'.join(fields)]=[dict(zip(fields,key),sse=x[0],count=x[1],rows=x[2],mse=x[0]/x[1]) for key,x in sorted(vals.items())]
write('step0_probe_decomposition.json',dict(aggregation='sum SSE / sum scalar count, existing saved forwards only',groups=groups))
summary=dict(parent_count=len(parentrows),condition_count=len(checks),models_loaded=0,model_forwards=0,new_OT_calls=0,new_noise_arrays=0,
    parent_failure_counts={k:sum(not r[k] for r in parentrows) for k in ('source_vertices_equal','source_faces_equal','normalized_equal','target_canonical_equal')},
    condition_failure_counts={k:sum(not r[k] for r in checks) for k in ('full_source_equal','pre_times_two_equal','C_source_equal','Y_source_equal','C_index_equal','partition_equal')},
    parent_stats={split:{k:dict(min=min(r[k] for r in parentrows if r['split']==split),median=float(np.median([r[k] for r in parentrows if r['split']==split])),max=max(r[k] for r in parentrows if r['split']==split))
        for k in ('N','normalized_max_span','model_max_span','target_scalar_rms')} for split in ('TRAIN','DEV')},
    official_split_counts={split:dict(collections.Counter(r['official_split'] for r in parentrows if r['split']==split)) for split in ('TRAIN','DEV')},
    probe_noise_rms_range=[min(r['noise_rms'] for r in prows),max(r['noise_rms'] for r in prows)],
    probe_C_rms_by_split={split:dict(mean=float(np.mean([r['C_scalar_rms'] for r in prows if r['split']==split])),min=min(r['C_scalar_rms'] for r in prows if r['split']==split),max=max(r['C_scalar_rms'] for r in prows if r['split']==split)) for split in ('TRAIN','DEV')},
    probe_Y_rms_by_split={split:dict(mean=float(np.mean([r['Y_scalar_rms'] for r in prows if r['split']==split])),min=min(r['Y_scalar_rms'] for r in prows if r['split']==split),max=max(r['Y_scalar_rms'] for r in prows if r['split']==split)) for split in ('TRAIN','DEV')})
write('coordinate_probe_summary.json',summary)
print(json.dumps(summary,indent=2));print(json.dumps(groups,indent=2))
