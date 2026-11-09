"""Freeze only a source-order export diagnostic on the already saved36 POSTs."""
from pathlib import Path
from . import common as C

def registration():
 reg=C.read(C.OUT/'registration.json');C.verify(reg['runner_hashes']);return reg

def prepare():
 target=C.OUT/'registration.json'
 if target.exists():
  reg=registration();C.verify(reg['protected']);return reg
 from . import restoration
 old=C.read(C.OLD/'registration.json');prior=C.read(C.OLD/'review/source_C_orientation_receipt.json')
 protected=dict(old['protected']);protected.update(old['runner_hashes']);protected.update(prior['protected_before'])
 C.verify(protected)
 for base in (C.OLD,C.ROOT/'r2/tools/c_restore_bounded_v1'):
  for p in base.rglob('*'):
   if p.is_file() and '__pycache__' not in str(p):protected[str(p)]=C.sha(p)
 refs={x['condition_id']:x for x in C.rows(C.OLD/'review/source_C_orientation_conditions.csv')}
 records=C.read(C.OLD/'post_index.json');assert len(records)==36
 for r in records:
  cr=refs[r['condition_id']];r['parent_prepared_path']=cr['parent_prepared_path'];r['parent_prepared_sha256']=cr['parent_prepared_sha256']
  assert C.sha(r['condition_source'])==cr['condition_source_sha256']
  assert C.sha(r['parent_prepared_path'])==r['parent_prepared_sha256']
  diag=C.read(r['diagnostic_path']);assert diag['status']=='PASS'
  assert diag['quality']['identity']['processed_sha256']==C.sha(r['post_path'])
  C.verify(diag['quality']['files'])
  for p in (r['post_path'],r['diagnostic_path'],r['condition_source'],r['parent_prepared_path'],r['denoised_path'],*diag['quality']['files']):protected[str(p)]=C.sha(p)
 source=C.ROOT/'r2/tools/c_restore_bounded_v1/orientation.py'
 assert C.sha(source)==old['runner_hashes'][str(source)]
 request=Path('/mnt/c/Users/warpwang/.codex/attachments/d6dcfba3-a619-4f05-9ff9-6417cb3bab06/Pasted text.txt')
 protected[str(request)]=C.sha(request)
 C.OUT.mkdir(parents=True,exist_ok=True)
 C.textfile(C.OUT/'USER_REQUEST.md',request.read_text(encoding='utf-8-sig'))
 reg=dict(version=C.VERSION,status='REGISTERED',records=records,conditions=list(refs.values()),
  protected=protected,runner_hashes={str(C.CODE/n):C.sha(C.CODE/n) for n in ('common.py','prepare.py','run.py','restoration.py','restoration_checks.py')},
  orientation_solver_path=str(source),orientation_solver_sha256=C.sha(source),restoration_definition=restoration.definition(),
  parent_registration_sha256=C.sha(C.OLD/'registration.json'),settings=old['settings'],fixed_L=old['fixed_L'],
  budget=dict(existing_POST_outputs=36,source_C_restorations=36,free_orientation_solves=36,CPU_synthetic_fixtures=8,
   model_loads=0,model_forwards=0,training_updates=0,new_noise=0,new_generation=0,new_GEO_C=0,new_full_quality_scores=0,new_images=0),
  stages=['MODEL_POST','RESTORED_C_ONLY','ORIENTED'],
  protocol=dict(methods=['D0_BASE','RBF','BLEND'],identical_mapping_for_all=True,
   C_target='Original source face corner order in the SAME model-coordinate FP32 values; inverse of saved normalization permutation; source selection face order retained',
   solver='Existing full indexed two-incidence nondegenerate XOR solver; freeze restored C. Satisfiable components repaired; conflict components unchanged; no deletion/splitting/tuning',
   geometry='All vertex bytes, unordered indexed triangles, vertex IDs, face IDs/count, connectivity, free vertex sets unchanged',
   attributes='Output corner mappings permuted; old input-domain N-corner audit arrays retained under input_ names, gathered output_ views by source_face_ids/final source_corner_ids. No UV/normal/color attributes present in this panel.',
   old_contract='Canonical C ordered bytes preserved at MODEL_POST only; new exports obey separate source-order C contract',
   quality='All full CF/FF/I/angle/AVMR/PMR/distance values inherited only after geometry equivalence; cached pair witnesses remain in original corner domain; chosen nearest corner witness remapped explicitly',
   partials='Nonmanifold/degenerate/parity/fixed-source-C conflicts reported; no automatic repair beyond registered reversal',
   replay='Completed receipt hashes reused; incomplete attempted solve blocks automatic replay',
   concurrency='Measure first authorized complete case, then resource-based at most2 CPU workers, two threads per worker; GPU0',
   interpretation='Export representation correction, not a new model or new geometric connection',
   stop='Fixed36 export/audit/report/package; D0 default unchanged, no automatic commit/push'))
 C.write(C.OUT/'resources_before_experiment.json',C.resources())
 C.write(target,reg);C.event('REGISTERED',registration_sha256=C.sha(target),budget=reg['budget'])
 C.textfile(C.OUT/'README_RESUME.md','CPU only, from WSL /mnt/e/MeshFlow:\n\n`/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.tools.source_winding_export_v1.run`\n\nSuccessful receipt/hash matches are reused; an incomplete attempted case blocks implicit replay. Source mapping is frozen independently of the generated result. No model, GEO, original full-quality scoring or default changes. Raw scientific stages and final independent audit/report/package have separate receipts.\n')
 return reg
