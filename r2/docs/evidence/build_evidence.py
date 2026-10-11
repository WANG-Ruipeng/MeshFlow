"""Export small, auditable public tables from completed local studies; zero scoring/model work.

Run with --artifact-root pointing to the separate experiment archive. Sources are
read-only; writes are confined to this evidence directory. All retained CSV values
remain exact source strings. Only a summary explicitly marked DERIVED is computed.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,re,statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
OUT=Path(__file__).resolve().parent
IDENTITY='evidence_phase model_instance_id request_id label method denoiser stage parent_id uid condition_id candidate noise_candidate ratio condition_cohort panel region_type tier N K status reason auxiliary_status score_execution'.split()
QUALITY='free_face_denominator CF_proper_affected_free_fraction CF_illegal_affected_free_fraction CF_proper_pair_count CF_coplanar_pair_count CF_unresolved_or_degenerate_pair_count FF_illegal_affected_free_fraction FF_illegal_pair_count FF_proper_pair_count FF_coplanar_pair_count FF_unresolved_or_degenerate_pair_count ConflictCov_C ConflictCov_C_cross ConflictCov_C_dup ConflictCov_C_cross_and_dup ConflictCov_C_uncertain_fraction ConflictCov_C_status free_minimum_angle_lt5_fraction free_angle_p05_degrees free_q_gt10_fraction free_near_degenerate_fraction free_exact_zero_area_fraction free_exact_duplicate_excess_fraction free_total_surface_area EXACT_COORD_CF_shared_edges C_output_bitwise C_all51_bitwise N_preserved finite FP32'.split()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,body):
 p=Path(p).resolve()
 if not p.is_relative_to(OUT.resolve()) or p==OUT.resolve():raise ValueError('Outside owned evidence directory')
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(body,encoding='utf-8',newline='');tmp.replace(p)
def writecsv(name,rows,fields):
 f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows);atomic(OUT/name,f.getvalue())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repository-root',type=Path,default=Path(__file__).resolve().parents[3]);ap.add_argument('--artifact-root',type=Path,required=True);ap.add_argument('--legacy-source-root',type=Path);args=ap.parse_args()
 roots={'repository':args.repository_root.resolve(),'artifact':args.artifact_root.resolve(),'legacy':(args.legacy_source_root or args.repository_root.parent/'MeshFlow-Control').resolve()};records=[];datasets={}
 def register(alias,rel,kind):
  p=roots[alias]/rel;entry=dict(source_id=alias+'/'+rel,kind=kind,sha256=sha(p),bytes=p.stat().st_size);records.append(entry);return p,entry
 def export(name,alias,rel,mode='quality',expected=None):
  p,entry=register(alias,rel,'CSV');f=p.open(encoding='utf-8-sig',newline='');reader=csv.DictReader(f);fields=reader.fieldnames;data=list(reader);f.close()
  if expected is not None:assert len(data)==expected,(name,len(data),expected)
  kept=([k for k in IDENTITY+QUALITY if k in fields] if mode=='quality' else fields if mode=='all' else [k for k in fields if k not in mode])
  if name=='student_fidelity_per_output.csv':kept=[k for k in kept if not k.startswith(('pre_GEO_d_','pre_GEO_L_'))]
  if name=='student_fidelity_summary.csv':kept=[k for k in kept if not k.startswith(('pre_GEO_d_','pre_GEO_L_'))]
  assert len(kept)==len(set(kept));result=[{k:r[k] for k in kept} for r in data]
  for r in result:
   for v in r.values():assert not re.search(r'(?i)(?:[A-Z]:[\\/]|/mnt/|/home/|\\Users\\)',v),(name,'private path')
  writecsv(name,result,kept);entry.update(export=name,rows=len(result),retained_columns=kept,omitted_columns=[k for k in fields if k not in kept],row_filter='NONE',numeric_transform='NONE; retained source strings unchanged',export_sha256=sha(OUT/name));datasets[name]=result
 specs=[
 ('r2_recipe_per_output.csv','artifact','experiments/chair_condition_recipe_factorial_v1/analysis/recipe_factorial_delivery_tables_v1/complete_per_output_640.csv','quality',640),
 ('user_finetune_per_output.csv','artifact','experiments/r2_user_condition_finetune_pilot_v1/evaluation/per_output.csv','quality',84),
 ('user_finetune_summary.csv','artifact','experiments/r2_user_condition_finetune_pilot_v1/evaluation/summary_equal_parent.csv','all',6),
 ('user_mix_per_output.csv','repository','experiments/r2_user_mix_from_official_v1/evaluation/per_output.csv','quality',48),
 ('user_mix_summary.csv','repository','experiments/r2_user_mix_from_official_v1/evaluation/summary_equal_parent.csv','all',4),
 ('user_mix_regions.csv','repository','experiments/r2_user_mix_from_official_v1/evaluation/summary_by_region.csv','all',8),
 ('user_mix_AI_PROXY.csv','repository','experiments/r2_user_mix_from_official_v1/review/AI_PROXY_per_output.csv',['colored_image','neutral_image'],48),
 ('official_post_per_output.csv','repository','experiments/r2_postprocess_avmr_pmr_v1/quality_per_output.csv','quality',96),
 ('official_post_paired_summary.csv','repository','experiments/r2_postprocess_avmr_pmr_v1/quality_paired_summary.csv','all',72),
 ('official_post_match_all8.csv','repository','experiments/r2_postprocess_avmr_pmr_v1/per_output.csv',['condition_id','tier','N','K'],768),
 ('official_multistep_per_output.csv','repository','experiments/r2_multistep_denoise_v1/quality_per_output.csv','quality',48),
 ('student_post_per_output.csv','repository','experiments/r2_single_step_denoiser_distillation_v1/evaluation/quality_per_output.csv','quality',144),
 ('student_fidelity_per_output.csv','repository','experiments/r2_single_step_denoiser_distillation_v1/evaluation/distillation_per_output.csv',['teacher_path','student_path'],66),
 ('student_fidelity_summary.csv','repository','experiments/r2_single_step_denoiser_distillation_v1/evaluation/distillation_summary.csv','all',None),
 ('teacher_benchmark.csv','repository','experiments/r2_single_step_denoiser_distillation_v1/benchmark/D0_OFFICIAL/summary.csv','all',14),
 ('student_benchmark.csv','repository','experiments/r2_single_step_denoiser_distillation_v1/benchmark/S8_W512_KD/summary.csv','all',14),
 ('student_multistep_per_output.csv','repository','experiments/r2_student_multistep_denoise_v1/quality_per_output.csv','quality',60),
 ('rbf_feedback_paired.csv','repository','experiments/r2_c_restore_coupling_v1/decision_per_parent_noise.csv','all',36),
 ('bounded_restore_per_output.csv','repository','experiments/r2_c_restore_bounded_v1/decision_per_output.csv','all',36),
 ('bounded_restore_summary.csv','repository','experiments/r2_c_restore_bounded_v1/decision_summary.csv','all',33),
 ('source_winding_per_output.csv','repository','experiments/r2_source_winding_export_v1/paired_per_parent_noise.csv','all',36),
 ('source_winding_topology.csv','repository','experiments/r2_source_winding_export_v1/topology_per_output.csv','all',108),
 ('source_winding_summary.csv','repository','experiments/r2_source_winding_export_v1/topology_summary.csv','all',99),
 ('source_winding_inherited_quality_summary.csv','repository','experiments/r2_source_winding_export_v1/review/quality_inherited_summary.csv','all',3),
 ('denoise_migration_per_case.csv','repository','experiments/r2_denoise_publish_v1/migration/validation_per_case.csv',['condition_id','seconds','maxrss_KiB'],60),
 ('denoise_wheel_arrays.csv','repository','experiments/r2_denoise_publish_v1/package/array_validation.csv','all',128),
 ('source_C_conditions.csv','repository','experiments/r2_source_winding_export_v1/review/source_restoration_summary.csv','all',6),
 ]
 for spec in specs:export(*spec)
 # Trace names, not coincidental numerical equality, to their source definitions.
 cf_contracts=[]
 for name,data in sorted(datasets.items()):
  fields=list(data[0])
  proper='CF_proper_affected_free_fraction' if 'CF_proper_affected_free_fraction' in fields else 'CF' if name.startswith('bounded_restore_') else ''
  illegal='CF_illegal_affected_free_fraction' if 'CF_illegal_affected_free_fraction' in fields else 'CF_illegal' if name.startswith('bounded_restore_') else ''
  long_form=('metric' in fields and any(r['metric'].startswith('CF_') for r in data))
  if not (proper or illegal or long_form):continue
  compared=[r for r in data if proper and illegal and r[proper] and r[illegal]]
  cf_contracts.append(dict(export=name,rows=len(data),proper_column=proper,proper_definition='free faces affected by proper_transverse_penetration only' if proper else 'NOT_PRESENT',illegal_column=illegal,illegal_definition='union of free faces affected by proper_transverse_penetration or coplanar_positive_area_overlap' if illegal else 'NOT_PRESENT',alias_definition='CF = proper; CF_illegal = union' if name.startswith('bounded_restore_') else 'explicit original field names; long-form metric values retain original name',denominator='actual free faces at recorded RAW/POST stage; pair counts are separate',compared_rows=len(compared),different_rows_tolerance_1e_15=sum(abs(float(r[proper])-float(r[illegal]))>1e-15 for r in compared),numerical_equality_does_not_equate_definitions=True))
 writecsv('CF_metric_contracts.csv',cf_contracts,list(cf_contracts[0]));records.append(dict(kind='DERIVED_DEFINITION_AUDIT',export='CF_metric_contracts.csv',rows=len(cf_contracts),export_sha256=sha(OUT/'CF_metric_contracts.csv'),definition='Column trace and arithmetic equality check only; no geometry classification or new scoring.'))
 # Reproduce equal-parent RAW summary from all 640 registered rows, retaining phase identity.
 grouped=defaultdict(list)
 for row in datasets['r2_recipe_per_output.csv']:grouped[row['evidence_phase'],row['label'],row['ratio']].append(row)
 summary=[];metrics=['CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction']
 for (phase,label,ratio),group in sorted(grouped.items()):
  parents=sorted({r['parent_id'] for r in group});assert len(parents)==8 and len(group)==32
  row=dict(evidence_phase=phase,label=label,ratio=ratio,parents=8,outputs=32,aggregation='DERIVED: mean noise within parent then equal mean over parents')
  for metric in metrics:row[metric]=statistics.mean(statistics.mean(float(r[metric]) for r in group if r['parent_id']==parent) for parent in parents)
  summary.append(row)
 writecsv('r2_recipe_summary.csv',summary,list(summary[0]));records.append(dict(kind='DERIVED_CSV',export='r2_recipe_summary.csv',source_export='r2_recipe_per_output.csv',rows=len(summary),export_sha256=sha(OUT/'r2_recipe_summary.csv'),definition='All phase/label/ratio groups; first average four noises within parent, then average all eight parents. No result-dependent subset.'))
 p,entry=register('repository','experiments/r2_source_winding_export_v1/registration.json','SOURCE_REQUEST_MANIFEST')
 data=json.loads(p.read_text());requests={r['request_id']:r for r in data['records']};keys='request_id parent_id condition_id candidate noise_seed region_type tier N K C_sha256 Gaussian_sha256 condition_source_sha256 checkpoint_sha256'.split();r=[{k:v[k] for k in keys} for _,v in sorted(requests.items())];assert len(r)==12
 writecsv('user_panel_requests.csv',r,keys);entry.update(export='user_panel_requests.csv',rows=12,export_sha256=sha(OUT/'user_panel_requests.csv'),selection='All unique requests across identical D0/RBF/BLEND identities; paths omitted')
 reports=[('artifact','chair_condition_recipe_factorial_v1/EXPERIMENT_REPORT.md'),('artifact','chair_r2_user_conditions_v1/DATA_PREPARATION_REPORT.md'),('artifact','r2_user_condition_finetune_pilot_v1/R2_USER_CONDITION_FINETUNE_REPORT.md'),('repository','r2_user_mix_from_official_v1/R2_USER_MIX_REPORT.md'),('repository','r2_postprocess_avmr_pmr_v1/REPORT.md'),('repository','r2_multistep_denoise_v1/REPORT.md'),('repository','r2_anchor_aware_denoiser_v1/REPORT.md'),('repository','r2_single_step_denoiser_distillation_v1/REPORT.md'),('repository','r2_student_multistep_denoise_v1/REPORT.md'),('repository','r2_c_restore_coupling_v1/REPORT.md'),('repository','r2_c_restore_bounded_v1/REPORT.md'),('repository','r2_source_winding_export_v1/REPORT.md')]
 for alias,rel in reports:register(alias,'experiments/'+rel,'LOCAL_REPORT_NOT_EMBEDDED')
 for rel in ['native_t1/docs/control_metrics.md','r2/tools/c_restore_bounded_v1/orientation.py','r2/tools/source_winding_export_v1/restoration.py','r2/tools/c_restore_bounded_v1/evaluation.py','r2/tools/source_winding_export_v1/report.py','r2/tools/postprocess_avmr_pmr_v1/quality.py','r2/tools/postprocess_avmr_pmr_v1/finalize_reports.py']:
  register('repository',rel,'DEFINITION_SOURCE')
 p,entry=register('repository','experiments/r2_user_mix_from_official_v1/evaluation/registration.json','FROZEN_SCORER_REGISTRATION');reg=json.loads(p.read_text())
 for source,expected in reg['scorer_sources'].items():
  source=Path(source);assert sha(source)==expected
  assert source.is_relative_to(roots['legacy']);register('legacy',source.relative_to(roots['legacy']).as_posix(),'FROZEN_DEFINITION_SOURCE')
 for rel,key in [('r2/pilots/user_mix_from_official_v1/evaluation.py','adapter_sha256'),('r2/pilots/user_condition_finetune_v1/evaluation.py','old_adapter_sha256')]:
  source,entry=register('repository',rel,'FROZEN_ADAPTER_SOURCE');assert entry['sha256']==reg[key]
 p,entry=register('repository','experiments/r2_denoise_publish_v1/migration/completion.json','MIGRATION_COMPLETION_RECEIPT')
 receipt=json.loads(p.read_text());assert receipt['status']=='PASS' and receipt['cases']==60 and receipt['arrays_compared']==4320 and receipt['protected_files']==199
 for name,h in receipt['code_hashes'].items():assert sha(roots['repository']/'r2/src/meshflow_control'/name)==h
 atomic(OUT/'denoise_migration_completion.json',p.read_text());entry.update(export='denoise_migration_completion.json',export_sha256=sha(OUT/'denoise_migration_completion.json'),numeric_transform='NONE; original receipt bytes preserved')
 register('repository','experiments/r2_denoise_publish_v1/migration/protected_hashes.csv','LOCAL_PROTECTED_HASH_LIST_NOT_EMBEDDED')
 def small_receipt(name,rel,kind,convert):
  p,entry=register('repository','experiments/r2_denoise_publish_v1/'+rel,kind)
  value=convert(p);value['source_receipt_id']=entry['source_id'];value['source_receipt_sha256']=entry['sha256']
  atomic(OUT/name,json.dumps(value,ensure_ascii=False,indent=2)+'\n')
  entry.update(export=name,export_sha256=sha(OUT/name),transformation='Selected verified fields; private machine paths, host and process IDs omitted. Original source SHA retained.')
 def cpu_receipt(p):
  suite=ET.parse(p).getroot().find('testsuite');assert suite is not None
  result={k:int(suite.attrib[k]) for k in ('tests','errors','failures','skipped')};assert result==dict(tests=115,errors=0,failures=0,skipped=0)
  return dict(status='PASS',**result,junit_suite_seconds=float(suite.attrib['time']),scope='Full public CPU test suite; no new research scoring. Initial run had six existing outside-repository temp-path guard failures; external temporary root then passed without core-code changes.')
 small_receipt('denoise_cpu_tests.json','cpu_tests_pass.xml','PUBLIC_CPU_TEST_RECEIPT',cpu_receipt)
 register('repository','experiments/r2_denoise_publish_v1/cpu_tests.xml','INITIAL_CPU_TEST_RECEIPT_HASH_ONLY')
 rp,re_entry=register('repository','experiments/r2_denoise_publish_v1/gpu_smoke/resources_after_forward.json','GPU_RESOURCE_RECEIPT');resource=json.loads(rp.read_text())
 dp,de=register('repository','experiments/r2_denoise_publish_v1/gpu_smoke/denoise.json','DENOISE_EXECUTION_RECEIPT');denoise=json.loads(dp.read_text())
 def gpu_receipt(p):
  d=json.loads(p.read_text());assert d['status']=='PASS' and d['model_forwards']==1 and d['operation_counts']['D0_FORWARD_STARTED']==d['operation_counts']['D0_FORWARD_RETURNED']==1
  assert d['proposal']['bytes_equal'] and d['proposal']['max_abs_error']==0.0 and d['canonical']['all_equal'] and d['final']['all_equal']
  for name,h in d['implementation_after'].items():assert sha(roots['repository']/'r2/src/meshflow_control'/name)==h
  checked_stages={k:dict(array_count=len(d[k]['arrays']),arrays=d[k]['arrays'],all_equal=d[k]['all_equal'],missing_keys=d[k]['missing_keys'],extra_keys=d[k]['extra_keys']) for k in ('canonical','final')}
  return dict(status='PASS',request_id=d['request_id'],N=d['proposal']['shape'][0],K=denoise['identity']['known_faces'],proposal=d['proposal'],stages=checked_stages,operation_counts=d['operation_counts'],model_forwards=d['model_forwards'],generator_forwards=d['generator_forwards'],training_updates=d['training_updates'],new_noise=d['new_noise'],full_quality_scores=denoise['full_quality_scores'],GEO_calls=denoise['GEO_calls'],orientation_solver_calls=denoise['source_export']['orientation_solver_calls'],checkpoint_sha256=d['checkpoint_sha256'],implementation_hashes=d['implementation_after'],protected_and_implementation_unchanged=d['protected_and_implementation_unchanged'],elapsed_seconds_including_load_hash_and_GEO=d['seconds'],timing_is_benchmark=False,peak_allocated_bytes=resource['peak_allocated_bytes'],peak_reserved_bytes=resource['peak_reserved_bytes'],peak_scope=resource['scope'],gpu_after_snapshot=resource['snapshot']['gpu']['stdout'].strip(),gpu_snapshot_fields='name,total_MiB,used_MiB,free_MiB,utilization_percent',process_VRAM='N/A under WSL; total GPU usage is not worker process peak',resources_source_sha256=re_entry['sha256'],denoise_execution_source_sha256=de['sha256'],warning='Read-only NumPy-to-torch view warning observed; no in-place tensor write, protected inputs unchanged; no retry.',scope='One fixed real-D0 integration smoke; no quality generalization claim.')
 small_receipt('denoise_real_D0.json','gpu_smoke_validation/verification.json','REAL_D0_VERIFICATION',gpu_receipt)
 mp,me=register('repository','experiments/r2_denoise_publish_v1/package/migration_source_match.json','WHEEL_IMPLEMENTATION_MATCH');match=json.loads(mp.read_text());assert match['status']=='PASS' and len(match['rows'])==16
 for row in match['rows']:assert row['migration_sha256']==row['wheel_sha256']==sha(roots['repository']/'r2/src/meshflow_control'/row['path'])
 def wheel_receipt(p):
  d=json.loads(p.read_text());assert d['status']=='PASS' and d['array_comparisons']==128 and d['all_old_arrays_bitwise_equal'] and not d['source_imports_outside_target'] and not d['CUDA_initialized'] and d['original_before']==d['original_after']
  assert all(x['returncode']==0 and not x['torch_imported'] for x in d['commands'])
  keys='status wheel_sha256 wheel_bytes counts model_loads CUDA_initialized new_scores new_noise array_comparisons pre_GEO_output_C_equal all_old_arrays_bitwise_equal capture_canonical_C_exact source_identity required_package_data source_imports_outside_target seconds default_venv_unchanged note'.split()
  value={k:d[k] for k in keys};value.update(wheel_name=Path(d['wheel']).name,commands=[{k:x[k] for k in ('name','returncode','seconds','torch_imported')} for x in d['commands']],installed_implementation_matches_migration=True,matched_implementation_files=16,implementation_match_source_sha256=me['sha256'],protected_sources_unchanged=True,scope='Isolated target install and temporary working directory; CPU integration test separate from the 60-case migration and historical research budgets.')
  return value
 small_receipt('denoise_wheel_validation.json','package/completion.json','ISOLATED_WHEEL_VERIFICATION',wheel_receipt)
 for q in records:
  if q.get('source_id')=='repository/native_t1/docs/control_metrics.md':q['availability']='LOCAL_POLICY_HASH_ONLY_NOT_PUBLISHED';q['public_definition']='Self-contained research_results.md section 1 and CF_metric_contracts.csv'
 private=re.compile(r'(?i)(?:[A-Z]:[\\/]|/mnt/|/home/|\\Users\\)')
 for p in list(OUT.glob('*.csv'))+list(OUT.glob('denoise_*.json')):assert not private.search(p.read_text(encoding='utf-8'))
 manifest=dict(schema='public_research_evidence_v1',as_of='2026-10-11',source_roots=dict(repository='This repository root; local experiment outputs are not committed',artifact='Separate local experiment archive, not committed',legacy='Original standalone source archive, not committed; source hashes checked against frozen scorer registration'),policy='Every exported CSV preserves all rows; columns are explicitly recorded. Numerical strings are unchanged except the named derived recipe summary. This is a compact evidence subset, not the full 35-family metric archive. No training, model call, rendering or re-scoring.',sources=records,export_script_sha256=sha(__file__),validation=dict(source_row_counts_verified=True,retained_fields_unchanged=True,absolute_private_paths_removed=True))
 atomic(OUT/'source_manifest.json',json.dumps(manifest,ensure_ascii=False,separators=(',',':'))+'\n')
 files=[p for p in OUT.iterdir() if p.is_file()];print(json.dumps(dict(status='PASS',CSV_files=len(list(OUT.glob('*.csv'))),files=len(files),bytes=sum(p.stat().st_size for p in files),scientific_calls=0)))
if __name__=='__main__':main()
