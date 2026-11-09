from core import *
import shutil,ast

def register():
 OUT.mkdir(parents=True,exist_ok=True)
 resources('registration')
 paths=set()
 # All original run evidence except large weights, images, model/teacher tensor payloads.
 for p in SRC.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.csv','.md','.py','.diff','.npz') and 'teacher_raw' not in p.parts:
   paths.add(p)
 for p in (ROOT/'experiments/mf_h1_local0/chair_domain_control_v1').glob('*.py'):paths.add(p)
 protocol=js(SRC/'protocol.json')
 for p in protocol['source_scorer_identity']:paths.add(ROOT/p)
 for name in ('AGENTS.md','WORKSPACE.md'):paths.add(ROOT/name)
 for base in ('native_t1','models'):
  for p in (ROOT/base).glob('*.py'):paths.add(p)
 prompt=Path('/mnt/c/Users/warpwang/Downloads/Chair_Control_Attribution_Codex_Prompt.md')
 if prompt.exists():paths.add(prompt)
 old_followup=ROOT/'native_t1/analysis/chair_control_followup_v1'
 for p in old_followup.rglob('*.md'):paths.add(p)
 # Historical frozen code.diff is retained and compared to current scoring file contents.
 entries=[];schemas=[]
 for p in sorted(paths):
  item=dict(path=str(p),relative_path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else None,bytes=p.stat().st_size,sha256=digest(p),status='PRESENT')
  if p.suffix=='.csv':
   with p.open(encoding='utf-8-sig') as f:
    rr=csv.reader(f);fields=next(rr);n=sum(1 for _ in rr)
   item.update(rows=n,fields=fields);schemas.append(item)
  entries.append(item)
 current=dict(schema='chair_control_attribution_source_manifest_v1',sources=entries,excluded=['checkpoint tensors: metadata only, never deserialized','teacher_raw arrays: teacher CSV only','old figures: not numeric truth'])
 path=OUT/'source_manifest.json'
 if path.exists():
  before={x['path']:x['sha256'] for x in js(path)['sources']}
  if before!={x['path']:x['sha256'] for x in entries}:raise RuntimeError('Source manifest changed; refuse silent reentry')
 else:savej(path,current)
 savej(OUT/'source_schema.json',schemas)
 checks=[]
 for p,h in protocol['source_scorer_identity'].items():checks.append(dict(path=p,recorded=h,actual=digest(ROOT/p),match=h==digest(ROOT/p)))
 if not all(x['match'] for x in checks):raise RuntimeError('Original scorer mismatch')
 diff=(SRC/'code.diff').read_text()
 original=ROOT/'experiments/mf_h1_local0/chair_domain_control_v1/width_analytics.py'
 rel=str(original.relative_to(ROOT));needle='+++ b/'+rel
 block=diff.split(needle,1)[1].split('\n--- /dev/null',1)[0]
 extracted='\n'.join(s[1:] for s in block.splitlines() if s.startswith('+') and not s.startswith('+++'))+'\n'
 checks.append(dict(path=rel,frozen_source='original code.diff added file',match=extracted==original.read_text()))
 if not checks[-1]['match']:raise RuntimeError('Frozen width scorer differs')
 savej(OUT/'original_source_identity_check.json',checks)
 srcsmall=OUT/'original_scoring_code';srcsmall.mkdir(exist_ok=True)
 srcpaths=set(ROOT/p for p in protocol['source_scorer_identity'])
 srcpaths.update(ROOT/'experiments/mf_h1_local0/chair_domain_control_v1'/n for n in ('width_analytics.py','reporting.py','evaluation.py','prepare.py','conclusions.py'))
 srcpaths.add(Path(__import__('trimesh').triangles.__file__))
 sourcehash=[]
 for p in sorted(srcpaths):
  rel=p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path('dependency')/p.name
  dest=srcsmall/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  sourcehash.append(dict(source=str(p),packaged=str(dest.relative_to(OUT)),sha256=digest(p),bytes=p.stat().st_size))
 savej(OUT/'source_code_hashes.json',sourcehash)
 od=OUT/'original_tables';od.mkdir(exist_ok=True)
 names=['B_width_response.csv','B_cross_condition_scores.csv','B_analytic_baselines.csv','A_context_per_case.csv','teacher_per_case.csv','shared_input_generation_curves.csv','B_width_summary.csv','teacher_summary.csv','C_holdout_paired_summary.csv','C_holdout_paired_per_parent.csv','failure_classification.csv','task_manifest.json','protocol.json','checkpoint_manifest.json','panels.json']
 for n in names:shutil.copyfile(SRC/n,od/n)
 shutil.copyfile(prompt,OUT/'REQUEST.md')
 contract=dict(schema='metric_contract_v1',units={'coordinates':'original model coordinates','e_base_response':'fraction of L0','squared_terms':'L0 squared normalized','legacy_bbox_pct':'100*distance/L_alpha','coverage':'fraction; multiply once by 100 for percentage points'},
 width=dict(formula='q95(X)-q05(X)',population='all finite free triangle corners, equal corner weight; copied C excluded',dtype='raw float32 converted to float64 BEFORE np.quantile',method='numpy linear',source='width_analytics.py:extent',small_p='p<=1e-8*L0 -> relative response undefined'),
 decomposition=dict(e='(p_a-alpha*g)/L0',base='alpha*(p-g)/L0',response='(p_a-alpha*p)/L0',cross='2*base*response',actual_GT_error='(p_a-g_alpha)/L0=e+(alpha*g-g_alpha)/L0; preserve float32 target transform residual',slope_self='(p_1.1-p_.9)/(.2*p)',slope_GT='(p_1.1-p_.9)/(.2*g)'),
 transform=dict(origin='original fixed origin, X only',source='prepare.py:load_task_sample',source_dtype='float32 multiply by np.float32(alpha)',pullback='float64(F_alpha); X /= float(np.float32(alpha)); invert actual saved transform factor; nominal alpha separately recorded',no_alignment=True),
 cross=dict(formula='E[alpha_in,beta_target]=sqrt(mean(distance(Q_beta,F_alpha)^2))/L0',queries='frozen Q1 points transported by float32(alpha) factor in float64',direction='rows=input condition, columns=target',column_advantage='mean(E[wrong,b])-E[b,b]',row_preference='mean(E[a,wrong])-E[a,a]',old_field='wrong_mean_minus_diagonal_RMS_L0 is ROW; old diagonal wins also ROW',rank_ties='raw sign strict retained; tolerance ties atol=1e-8+1e-6*abs(diagonal) separately',reuse='648 original cells + original JSON independent mapping; no rerun when agree'),
 coverage=dict(points='original reference coverage_points; uniform per sampled point, Q=512',threshold='d<=.01*L for original A case',partition='both_in,enter,leave,both_out',contribution='sum_group((d_new/L)^2-(d_old/L)^2)/Q',original_R12='uses DIFFERENT pairwise_target_points; fixed_query_MSE never relabeled original R12 RMS',source='eval_scoring.py:score_free'),
 pullback=dict(candidates=96,types=2,symmetric='512 area samples independently on each surface via original sampler RandomState(9911); sum of two directional means /2, sqrt /L0',GT='frozen B Q1 -> canonical pulled free, coverage<=.01 L0',sampling='finite offline surface uniform draws only, NO Gaussian; not generated meshes',auxiliary='same-slot coordinate RMS is auxiliary, no GT slot matching',chunk=32),
 aggregation='curve first; within parent average noise/alpha; parents equal weight. Phase comparison common n0 only. A20/A40 separate complements.',
 reconciliation=TOL,fixture_families=8,budgets=dict(B_outputs=216,curves=72,cross_cells=648,pullback_candidates=96,A_vectors=270,teacher_rows=288),
 forbidden=dict(model_loads=0,forwards=0,backwards=0,optimizer_updates=0,OT=0,new_Gaussians=0,new_generations=0,CUDA_initializations=0),
 visual_registration=dict(raw_examples='lexicographically first UID in each B split; noise0; all three arms; raw/pullback/GT; fixed view elev20 azim-60',fixed_global_heatmap=True),
 source_code_hashes=sourcehash)
 savej(OUT/'metric_contract.json',contract)
 line_locations=[]
 for p in srcpaths:
  for i,line in enumerate(p.read_text().splitlines(),1):
   if any(s in line for s in ('def extent','def response_metrics','matching =','off =','wrong_mean_minus_diagonal','def point_diagnostic','def score_free','def surface_distance_squared','def sample_mesh_surface','coverage_points','pairwise_target_points','def load_task_sample')):
    line_locations.append(dict(file=str(p),line=i,source=line))
 writecsv('metric_source_locations.csv',line_locations)
 acc=js(SRC/'execution_accounting.json');gen=generated()
 identity=dict(complete_generations=acc['complete_generations'],phase_table_counts={str(k):len(v) for k,v in groups([r for r in gen if r['status']=='PASS'],['phase_step','table']).items()},checkpoint_identity_source='metadata/manifest only',protocol_NFE=protocol['NFE'],precision=protocol['precision'],main_updates_per_arm=protocol['main_updates_per_arm'])
 savej(OUT/'source_identity.json',identity)
 return dict(files=len(entries),source_scorer_matches=all(x['match'] for x in checks),identity=identity),['source_manifest.json','source_schema.json','original_source_identity_check.json','source_code_hashes.json','metric_contract.json','metric_source_locations.csv','source_identity.json']
