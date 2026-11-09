from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path.cwd().resolve()
RUN=ROOT/'output/a_fm_tail_20260925_03'
OLD=ROOT/'output/a_ae_fm_closed16_20260925_01'
OUT=ROOT/'output/a_fm_tail_20260925_03_review_01'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def close(a,b):assert np.isclose(a,b,rtol=1e-13,atol=1e-15),(a,b)
metrics=read(RUN/'artifacts/metrics/RESULTS.json')['rows'];mi={(r['seed'],r['arm'],r['sample_id']):r for r in metrics}
status=read(RUN/'artifacts/metrics/DISTRIBUTION_STATUS.json');assert len(status)==2 and {x['seed'] for x in status}=={10,11}
comparisons=[];bound={};failures=[];group_count=0
for ss in status:
 seed=ss['seed'];p=RUN/f'artifacts/metrics/distribution/seed{seed}';op=OLD/f'seed{seed}/distribution'
 assert ss['returncode']==0 and ss['euler_steps']==32 and ss['auxiliary_excluded']
 assert ss['original_evaluate_sha256']==sha(ROOT/'experiments/a_radial/evaluate.py')
 assert ss['sample_manifest_sha256']==sha(RUN/f'artifacts/seed{seed}/samples/SAMPLES.json')
 assert ss['generation_identity']['FM_step']==64000 and ss['generation_identity']['AE_step']==3200
 for arm in ['fixed','a','qem']:assert ss['outlet_manifest_sha256'][arm]==sha(RUN/f'artifacts/seed{seed}/exports/{arm}_MANIFEST.json')
 assert ss['command'][-3:]==['--exact','--points','2048']
 rows=read(p/'PER_SAMPLE.json');groups=read(p/'DISTRIBUTION_GROUPS.json');matrix=read(p/'MATRIX.json');old={x['row']:x for x in read(op/'MATRIX.json')}
 assert len(rows)==192 and len(groups)==12 and len(matrix)==3
 expected={(f'ours_{a}',f'{i:05d}') for a in ['fixed','a','qem'] for i in range(64)}
 assert {(r['row'],r['sample_id']) for r in rows}==expected and len(expected)==len(rows)
 assert {(r['row'],r['offset']) for r in groups}=={(a,o) for a in ['ours_fixed','ours_a','ours_qem'] for o in [0,16,32,48]}
 for r in rows:
  arm=r['row'][5:];m=mi[(seed,arm,r['sample_id'])]
  assert r['status']=='ok' and r.get('metric_status')!='failed'
  assert r['actual_faces']==512 and r['budget_match']
  assert r['topology']==m['audit']['topology'] and r['exact']==m['audit']['exact']
  assert r['joint_valid_exact']==m['audit']['valid']
  if not r['joint_valid_exact']:failures.append({'seed':seed,'arm':arm,'sample_id':r['sample_id'],'actual_faces':r['actual_faces'],'topology_pass':r['topology']['closed_oriented_g0'],'exact_intersections':r['exact']['pair_count']})
 for g in groups:
  assert g['status']=='ok' and g['n_each']==16 and g['closed_set_not_holdout']
  gp=p/f"{g['row']}_group{g['offset']:04d}.npz";bound[str(gp.relative_to(ROOT))]=sha(gp)
  with np.load(gp,allow_pickle=False) as z:C=z['cross_distance'];D=z['all_distance']
  assert C.shape==(16,16) and D.shape==(32,32)
  assert np.isfinite(C).all() and (C>=0).all()
  assert np.isposinf(np.diag(D)).all() and np.isfinite(D[~np.eye(32,dtype=bool)]).all()
  assert np.array_equal(D,D.T) and np.array_equal(C,D[:16,16:])
  labels=np.r_[np.zeros(16,dtype=int),np.ones(16,dtype=int)]
  calculated={'MMD_CD':float(np.mean(np.min(C,axis=0))),'coverage':float(len(np.unique(np.argmin(C,axis=1)))/16),'one_NNA':float(np.mean(labels[np.argmin(D,axis=1)]==labels)),'nearest_train_distance_mean':float(np.mean(np.min(C,axis=1)))}
  for k,v in calculated.items():close(v,g[k])
  group_count+=1
 for row in matrix:
  rs=[x for x in rows if x['row']==row['row']];gs=[x for x in groups if x['row']==row['row']]
  assert row['attempts']==64 and row['metrics_readable']==64 and row['budget512_count']==64
  assert row['joint_valid_exact_count']==sum(x['joint_valid_exact'] for x in rs)
  assert row['valid_metric_groups']==4 and row['not_heldout'] and not row['full_publication_ranking_qualified']
  for k in ['MMD_CD','coverage','one_NNA']:close(row[k],np.mean([x[k] for x in gs]))
  o=old[row['row']]
  comparisons.append({'seed':seed,'arm':row['row'][5:],'attempts_each':64,'all_readable_meshes_included_regardless_geometry_validity':True,'exact_valid_8k_64k':[o['joint_valid_exact_count'],row['joint_valid_exact_count']],**{k:{'8k':o[k],'64k':row[k],'delta_64k_minus_8k':row[k]-o[k]} for k in ['MMD_CD','coverage','one_NNA']},'relative_MMD_CD_reduction_percent':100*(1-row['MMD_CD']/o['MMD_CD'])})
 for f in p.iterdir():
  if f.is_file():bound[str(f.relative_to(ROOT))]=sha(f)
 bound[str((op/'MATRIX.json').relative_to(ROOT))]=sha(op/'MATRIX.json')
 bound[str((RUN/f'artifacts/metrics/distribution/seed{seed}_INVOCATION.json').relative_to(ROOT))]=sha(RUN/f'artifacts/metrics/distribution/seed{seed}_INVOCATION.json')
for p in [RUN/'artifacts/metrics/DISTRIBUTION_STATUS.json',RUN/'artifacts/metrics/RESULTS.json',ROOT/'experiments/a_radial/evaluate.py',RUN/'RUN_PROTOCOL.json',RUN/'ARTIFACT_PROTOCOL.json']:
 if p.exists():bound[str(p.relative_to(ROOT))]=sha(p)
report={'status':'DISTRIBUTION_SAVED_MATRIX_AND_DENOMINATOR_REVIEW_PASS','source_sha256':sha(__file__),'scope':'CPU read-only independent recomputation from saved distance matrices; no new cloud sampling, geometry audit, model forward or GPU','seed_count':2,'planned_primary_samples':128,'planned_outlet_attempts':384,'per_sample_rows':384,'group_matrices_recomputed':group_count,'group_n_each':16,'points_per_surface':2048,'MC_surface_RMS_8192_is_a_separate_metric':True,'comparison':comparisons,'geometric_failures_retained_in_distribution':failures,'no_aux_E128_in_primary':True,'no_failed_mesh_filtering':True,'one_NNA_not_monotonically_lower_is_better':True,'limits':['8k to64k intervention combines added training budget with declared constant-tail learning rate; not a pure budget ablation','16 training references; four generation blocks are not independent training seeds or a heldout test','two independent training seeds; descriptive changes are not significance or generalization evidence','saved matrix statistic replication does not independently reproduce mesh sampling or Chamfer distance computation'],'bound_files_sha256':bound}
(OUT/'DISTRIBUTION_REVIEW.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='bound_files_sha256'},indent=2))
