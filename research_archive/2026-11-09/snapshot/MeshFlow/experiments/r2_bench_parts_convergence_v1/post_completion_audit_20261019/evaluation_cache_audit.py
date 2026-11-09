"""Read-only NumPy checks of saved inference wiring; no model or rescoring."""
from pathlib import Path
import collections, hashlib, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/r2_bench_parts_convergence_v1'
OUT=Path(__file__).resolve().parent
def path(p):
    return Path(str(p).replace('/mnt/e/MeshFlow','E:/MeshFlow'))
def read(p): return json.loads(path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(path(p).read_bytes()).hexdigest()
sources={}
def arr(p):
    p=path(p);sources[str(p)]=sha(p)
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
manifest=read(EXP/'evaluation_inputs/manifest.json')
sources[str(EXP/'evaluation_inputs/manifest.json')]=sha(EXP/'evaluation_inputs/manifest.json')
rows=[]
for label in ['NATIVE_S21000','SEAM_S21000']:
  for q in manifest['requests']:
    d=EXP/'evaluation'/label/q['request_id'];r=d/'raw';p=d/'post'
    inp=arr(q['inference_npz']);source=arr(q['source_npz']);noise=arr(q['noise_npz'])['z']
    raw=arr(r/'raw.npz');traj=arr(r/'trajectory.npz')['state'];post=arr(p/'post.npz')
    pre=arr(p/'pre_geo.npz')['output'];proposal=arr(p/'proposal.npz')['official_unconstrained_proposal']
    layout=arr(EXP/'evaluation'/label/'layouts'/q['condition_id']/'layout.npz')
    n,k=q['N'],q['K'];anchor=raw['raw_anchor_ids'];fixed=anchor>=0
    expected=np.zeros((n,3,3),np.float32);expected[k:]=noise;expected[fixed]=inp['C_vertices'][anchor[fixed]]
    restore=proposal.copy();restore[fixed]=inp['C_vertices'][anchor[fixed]]
    corner=raw['source_corner_ids'];ids=post['source_face_ids'];pc=post['source_corner_ids']
    checks={
      'input_N_K_shape':int(inp['N'])==n and int(inp['K'])==k and raw['raw'].shape==(n,3,3),
      'input_original_C_vs_source':same(inp['C_vertices'][inp['C_faces_original']],source['C_original']),
      'raw_C_canonical':same(raw['raw'][:k],inp['C_vertices'][inp['C_faces_model']]),
      'trajectory_51_states':traj.shape==(51,n,3,3),
      'trajectory_initial_saved_noise':same(traj[0],expected),
      'trajectory_final_saved_RAW':same(traj[-1],raw['raw']),
      'every_state_all_anchors_exact':all(same(x[fixed],expected[fixed]) for x in traj),
      'raw_output_corner_transport':same(raw['output'],raw['raw'][np.arange(n)[:,None],corner]),
      'raw_indexed_roundtrip':same(raw['vertices'][raw['faces']],raw['output']),
      'raw_export_source_C':same(raw['C'],source['C_original']),
      'layout_anchor_identity':same(layout['anchor_ids'],anchor),
      'pre_geo_exact_anchor_restore_only':same(pre,restore),
      'post_indexed_roundtrip':same(post['vertices'][post['faces']],post['output']),
      'post_export_source_C':same(post['C'],source['C_original']),
      'post_RAW_face_ids_unique':len(set(ids.tolist()))==len(ids) and np.array_equal(ids[:k],np.arange(k)),
      'post_RAW_role_transport':np.array_equal(post['retained_seam_mask'],raw['raw_seam_slot_mask'][ids]),
      'post_anchor_transport':np.array_equal(post['output_anchor_ids'],anchor[ids[:,None],pc]),
      'post_corner_ID_domain':bool(np.all((pc>=0)&(pc<3))),
      'source_bundle_no_Y_or_owner':not any('owner' in z or z in ('Y','full_target','occupied_labels') for z in source),
    }
    anchor_map={}
    for ci,pi in zip(inp['C_faces_original'].flat,post['faces'][:k].flat):
        if int(ci) in anchor_map and anchor_map[int(ci)]!=int(pi):raise ValueError('C identity split')
        anchor_map[int(ci)]=int(pi)
    checks['post_C_vertex_bijection']=len(set(anchor_map.values()))==len(inp['C_vertices'])
    pa=post['output_anchor_ids'];pf=pa>=0
    checks['post_anchor_index_reference']=all(int(post['faces'][i,j])==anchor_map[int(pa[i,j])] for i,j in np.argwhere(pf))
    g=read(p/'geometry.json');a=read(p/'anchor_identity.json');pr=read(p/'proposal_receipt.json');rr=read(r/'receipt.json')
    checks['reported_calls_Euler50_D0one']=rr['generator_forwards']==50 and pr['D0_forwards']==1
    checks['post_radius_original']=g['stats']['radius_model']==.015/(.95*.3762)
    checks['D0_Bench_SHA']=pr['official_checkpoint_sha256']=='67765c728c503b61c21624c05997ea026c9de486d1529052fafa134be3336b14'
    for s in (r/'receipt.json',p/'proposal_receipt.json',p/'geometry.json',p/'anchor_identity.json'):sources[str(s)]=sha(s)
    rows.append(dict(label=label,request_id=q['request_id'],split=q['split'],part=q['part_type'],N=n,K=k,
       source_bundle_keys=sorted(source),checks=checks,all_pass=all(checks.values()),
       input_C_absmax=float(np.abs(inp['C_vertices']).max()),RAW_absmax=float(np.abs(raw['raw']).max()),
       D0_absmax=float(np.abs(proposal).max()),POST_absmax=float(np.abs(post['output']).max()),
       source_C_extent=np.ptp(inp['C_vertices'],axis=0).tolist(),
       deleted_by_reason=g['stats']['deleted_by_reason'],deleted_free_faces=g['stats']['deleted_free_faces']))
result=dict(schema='BENCH_FINAL_SAVED_WIRING_AUDIT_V1',status='PASS' if all(x['all_pass'] for x in rows) else 'FAIL',
 outputs=len(rows),checks_per_output=len(rows[0]['checks']),failed=[(x['label'],x['request_id'],k) for x in rows for k,v in x['checks'].items() if not v],
 model_calls=0,GPU_calls=0,geometry_scans=0,new_noise_arrays=0,scope='Existing cache integrity and array identity; no metric recomputation',
 rows=rows,source_hashes=sources,script_sha256=sha(__file__),numpy_version=np.__version__)
(OUT/'evaluation_cache_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('status','outputs','checks_per_output','failed','model_calls','geometry_scans')}))
