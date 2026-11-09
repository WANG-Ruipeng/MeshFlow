"""Read-only derived summary of saved arrays; no extraction or parameter changes."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'post_analysis/ESTIMATE_DIAGNOSTIC_SUMMARY.json'
if OUTPUT.exists(): raise FileExistsError(OUTPUT)
inputs={}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    inputs[str(p)]=sha(p)
    return json.loads(Path(p).read_text())
def arrays(p):
    inputs[str(p)]=sha(p)
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def stats(x):
    x=np.asarray(x).reshape(-1); f=x[np.isfinite(x)]
    return dict(count=int(len(x)),nonfinite=int((~np.isfinite(x)).sum()),
        minimum=float(f.min()) if len(f) else None,median=float(np.median(f)) if len(f) else None,
        mean=float(f.mean()) if len(f) else None,p95=float(np.percentile(f,95)) if len(f) else None,
        maximum=float(f.max()) if len(f) else None)
def difference(a,b):
    if a.shape!=b.shape: return dict(same_shape=False)
    d=a-b; denom=np.linalg.norm(b)
    return dict(same_shape=True,byteidentical=a.dtype==b.dtype and a.tobytes()==b.tobytes(),
        unequal_entries=int(np.count_nonzero(a!=b)),elements=int(a.size),max_absolute=float(np.max(abs(d))) if d.size else 0.,
        relative_frobenius=float(np.linalg.norm(d)/denom) if denom else None)
protocol=read(ROOT/'FROZEN_PROTOCOL.json'); factorial=Path(protocol['config']['factorial_run'])
ledger=read(ROOT/'LEDGER.json'); diagnostic=read(ROOT/'audit/J_ESTIMATION_DIAGNOSTICS.json')
diag={(r['object'],r['state']):r for r in diagnostic['rows']}
rows=[]; group_arrays={}; neighborhoods={}
for row in ledger:
    obj,arm,source,frame=row['object'],row['state'],row['source_object'],row['frame']
    rd=ROOT/'runs'/f'{obj}__{arm}'; old=factorial/'runs'/f'{obj}__q_only'
    p=arrays(rd/'parameters.npz'); q=arrays(old/'parameters.npz')
    est=arrays(rd/'estimated_motion.npz'); tr=arrays(rd/'extractor_trace.npz'); ot=arrays(old/'extractor_trace.npz')
    error=np.asarray(diag[(obj,arm)]['per_element_used_error'])
    result=dict(object=obj,source=source,state=arm,frame=frame,
        J_error=stats(error),raw_J_condition=stats(est['raw_condition']),raw_J_determinant=stats(est['raw_determinant']),
        fallback_count=int(est['used_fallback'].sum()),rank_deficient_count=int((est['rank']<3).sum()),
        Q_vs_analytic=difference(p['Q'],q['Q']),eigs_vs_analytic=difference(p['eigs'],q['eigs']),
        normalized_Q_vs_analytic=difference(p['Q']/p['eigs'][:,0,None,None],q['Q']/q['eigs'][:,0,None,None]),
        mesh_byteidentical_to_analytic=sha(rd/'mesh.obj')==sha(old/'mesh.obj'))
    inputs[str(rd/'mesh.obj')]=sha(rd/'mesh.obj'); inputs[str(old/'mesh.obj')]=sha(old/'mesh.obj')
    for key in ['candidate_faces','graph_edges','graph_terminals','precut_labels','postcut_labels']:
        result[key+'_identical']=tr[key].dtype==ot[key].dtype and tr[key].shape==ot[key].shape and tr[key].tobytes()==ot[key].tobytes()
    for key in ['candidate_scores','graph_capacities']:
        result[key+'_vs_analytic']=difference(tr[key],ot[key])
    rows.append(result)
    if frame>0:
        group_arrays.setdefault((source,arm),[]).append((frame,error,est))
    if source not in neighborhoods:
        hood=arrays(ROOT/'motion_inputs'/source/'neighborhood.npz'); singular=hood['weighted_singular_values']
        neighborhoods[source]=dict(weighted_design_condition=stats(singular[:,0]/singular[:,-1]),
            rank_counts={str(k):int((hood['rank']==k).sum()) for k in range(4)},h=stats(hood['h']))
groups=[]
for (source,arm),vals in group_arrays.items():
    selected=[r for r in rows if r['source']==source and r['state']==arm and r['frame']>0]
    last=max(vals,key=lambda x:x[0]); lastrow=next(r for r in selected if r['frame']==last[0])
    groups.append(dict(source=source,state=arm,non_t0_frames=len(vals),
        J_error_non_t0=stats(np.concatenate([v[1] for v in vals])),J_error_final=stats(last[1]),
        raw_J_condition_non_t0=stats(np.concatenate([v[2]['raw_condition'] for v in vals])),
        raw_J_determinant_non_t0=stats(np.concatenate([v[2]['raw_determinant'] for v in vals])),
        raw_J_condition_final=stats(last[2]['raw_condition']),raw_J_determinant_final=stats(last[2]['raw_determinant']),
        Q_byte_different_non_t0=sum(not r['Q_vs_analytic']['byteidentical'] for r in selected),
        candidate_scores_byte_different_non_t0=sum(not r['candidate_scores_vs_analytic']['byteidentical'] for r in selected),
        graph_capacities_byte_different_non_t0=sum(not r['graph_capacities_vs_analytic']['byteidentical'] for r in selected),
        mesh_identical_non_t0=sum(r['mesh_byteidentical_to_analytic'] for r in selected),final_frame=lastrow))
result=dict(scope='Post-only saved array comparison; no new extraction, cuts, sampling, tuning or source changes.',
    caveat='Frames are correlated development observations. Parameter/energy differences with identical meshes do not imply general robustness or equal timing.',
    neighborhoods=neighborhoods,groups=groups,rows=rows,
    counts=dict(rows=len(rows),mesh_identical=sum(r['mesh_byteidentical_to_analytic'] for r in rows),
        Q_byte_different=sum(not r['Q_vs_analytic']['byteidentical'] for r in rows),
        score_byte_different=sum(not r['candidate_scores_vs_analytic']['byteidentical'] for r in rows),
        capacity_byte_different=sum(not r['graph_capacities_vs_analytic']['byteidentical'] for r in rows)),
    input_sha256=inputs,script_sha256=sha(__file__))
with OUTPUT.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(counts=result['counts'],neighborhoods=neighborhoods,
    groups=[{k:v for k,v in g.items() if k!='final_frame'} for g in groups]),indent=2))