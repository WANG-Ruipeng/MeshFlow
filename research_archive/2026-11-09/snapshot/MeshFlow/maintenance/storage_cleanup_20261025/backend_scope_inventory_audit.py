"""Read-only Bench weight classification; writes only this cleanup audit directory."""
from pathlib import Path
import csv, json, datetime, os

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def rel(path):
    s = str(path).replace('\\', '/')
    if s.lower().startswith('e:/meshflow/'):
        return s[len('e:/meshflow/'):]
    raise ValueError(s)

keep = {x['path']: x['role'] for x in read(ROOT/'r2/maintenance/bench_reference_20261023/retained_assets.json')['preserved_assets']}
keep.update({
 'checkpoints/official_bench_denoiser/0021000.pt': 'OFFICIAL_D0_PROTECTED',
 'output/120m-ot-v-bench/checkpoints/01000000.pt': 'OFFICIAL_GENERATOR_PROTECTED',
 'experiments/r2_bench_parts_convergence_v1/training/COMMON/step12000.pt': 'RETAINED_COMMON_SOURCE_WITH_MATCHED_OPTIMIZER_RNG',
 'experiments/r2_bench_parts_convergence_v1/training/COMMON/generator_step12000.pt': 'RETAINED_COMMON_SOURCE_INFERENCE_REFERENCE',
 'experiments/r2_bench_seat_coverage_v1/training/NATIVE/checkpoint_step1500.pt': 'CURRENT_COVERAGE_RAW_POSITIVE_CONTINUATION_STATE',
 'experiments/r2_bench_seat_coverage_v1/training/NATIVE/generator_step1500.pt': 'CURRENT_COVERAGE_RAW_POSITIVE_GENERATOR',
 'experiments/r2_bench_backend_shape_preservation_v1/training/candidate_denoiser.pt': 'SHAPE500_BOUNDED_POSITIVE_BACKEND_NOT_DEFAULT',
 'experiments/r2_bench_backend_shape_preservation_v1/training/checkpoints/step0500.pt': 'SHAPE500_MATCHED_RESTORE_STATE_FOR_RETAINED_CANDIDATE',
})
retired_runs = {
 'r2_bench_newdata_joint_v1': 'No promotion: old layout recall and DEV FF/I costs; latest complete-seat and Coverage endpoints supersede continued use.',
 'r2_bench_state_schedule_v1': 'PERT/ASYNC both completed without overall upgrade; retain fixed outputs and full reports.',
 'r2_bench_joint_shift_v1': 'SHIFT follow-up gate closed; adoption evidence insufficient across DEV parents and geometry costs remain.',
 'r2_bench_early_fm_v1': 'No new adoption identities or DEV upgrade; closed time-weighting candidate.',
 'r2_bench_coverage_expansion_v1': 'Historical multipart COVERAGE_SEAM, distinct from retained complete-seat Coverage; not promoted.',
 'r2_bench_relation_training_v1': 'Historical relation-data candidate not promoted; preserve original data/evidence, retire old generator branch.',
 'r2_bench_adaptation_strength_v1': 'A/B/C1500 historical comparisons complete, no accepted overall upgrade; later current references kept.',
 'r2_bench_clean_multipart_seam_v1': 'Historical multipart candidate no DEV body join; superseded research branch.',
 'r2_bench_user_v2_training_v1': 'Historical USER_V2 candidate no overall DEV upgrade; preserves local positive evidence in existing outputs.',
}
rows=[]
with (OUT/'weight_inventory.csv').open(encoding='utf-8-sig', newline='') as f:
    for x in csv.DictReader(f):
        if 'bench' not in x['Path'].lower():
            continue
        p=rel(x['Path']); actual=ROOT/p; n=int(x['Bytes'])
        action='KEEP_SMALL_TEST_OR_EVIDENCE'; reason='Tiny head/fixture evidence; no meaningful disk saving and not a production checkpoint.'
        if p in keep:
            action='KEEP'; reason=keep[p]
        elif n >= 1<<20:
            run=p.split('/')[1] if p.startswith('experiments/') else ''
            if run=='r2_bench_backend_relative_deformation_v1':
                action='DELETE_CANDIDATE'; reason='RELDEF500 did not improve strict visual success over SHAPE and worsened DEV FF; all fixed proposals/metrics/evidence remain usable without weights.'
            elif run=='r2_bench_backend_shape_preservation_v1' and '/training/checkpoints/' in p:
                action='DELETE_CANDIDATE'; reason='Intermediate SHAPE recovery snapshot; final complete step500 and inference export retained. Old step100 boundary replay retires.'
            elif run=='r2_bench_full_seat_curriculum_v1' and '/training/' in p:
                action='DELETE_CANDIDATE'; reason='Intermediate/initial/rolling curriculum snapshot; both final Native6000 and SEAM3000 full checkpoints plus exports retained. COMMON12000 source retained.'
            elif run=='r2_bench_seat_coverage_v1' and '/training/' in p:
                action='DELETE_CANDIDATE'; reason='Initial/rolling Coverage snapshot; final step1500 checkpoint plus export and original Native6000 source retained.'
            elif run=='r2_bench_parts_convergence_v1' and '/training/' in p and ('/SEAM/' in p or '/PLAIN/' in p):
                action='DELETE_CANDIDATE'; reason='Historical multipart Plain/SEAM15000/21000 endpoints superseded by explicitly retained complete-seat reference; head exported independently and kept.'
            elif run=='r2_bench_parts_convergence_v1' and '/audits/worker_write_race_' in p:
                action='DELETE_CANDIDATE'; reason='Historical interrupted checkpoint from completed recovery engineering audit; incident receipt/source evidence remains, COMMON12000 final retained.'
            elif run in retired_runs:
                action='DELETE_CANDIDATE'; reason=retired_runs[run]
            else:
                action='HOLD_UNCLASSIFIED'; reason='No exact current retention/retirement evidence in this scoped audit.'
        rows.append(dict(path=p,absolute_path=str(actual),bytes=n,gib=n/(1<<30),action=action,reason=reason,
                         exists=actual.is_file(),size_matches_inventory=actual.is_file() and actual.stat().st_size==n,
                         large_file_hash_scope='NOT_REHASHED; source receipts preserved',original_sidecar_policy='KEEP'))

datasets=[]
for name in ['r2_bench_backend_shape_preservation_v1','r2_bench_backend_relative_deformation_v1','r2_bench_seat_coverage_v1']:
    p=ROOT/'experiments'/name/'data'; fs=[f for f in p.rglob('*') if f.is_file()]
    datasets.append(dict(path=p.relative_to(ROOT).as_posix(),files=len(fs),bytes=sum(f.stat().st_size for f in fs),
        action='KEEP',reason='Small source arrays/identity maps/masks/cache/manifest; active retained models or cross-experiment provenance and CPU recomputation depend on them.'))

summary={a:dict(files=sum(r['action']==a for r in rows),bytes=sum(r['bytes'] for r in rows if r['action']==a)) for a in sorted(set(r['action'] for r in rows))}
audit=dict(schema='BENCH_CLEANUP_READONLY_RECOMMENDATIONS_V1',status='RECOMMENDATIONS_NOT_DELETION',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_inventory=str(OUT/'weight_inventory.csv'),
    authority='User current unused weight/data cleanup; parent owns all mutations.',
    summary=summary,weights=rows,dataset_recommendations=datasets,
    dependency_findings=[
        'Current Native6000 depends historically on COMMON12000; Coverage depends historically on Native6000. Both sources and final matching AdamW/RNG states are retained.',
        'SHAPE/RELDEF both initialize from official D0. RELDEF does not initialize from SHAPE500; their frozen real evaluation proposals now exist.',
        'Latest local-repair completed with zero models and creates no weights. Its protected files refer to SHAPE/RELDEF proposals, meshes, maps, pairs, scores, not neural checkpoint tensors.',
        'Do not delete old experiment trees, RAW/POST/proposal/target NPZ, source metadata, fixed noise/query arrays, scores, galleries, logs, CPU mappings, original code/receipts, or small occupancy heads.',
        'Retiring weights changes historical full-chain replay availability. Preserve original COMPLETE.json and its hashes unchanged; add separate retirement manifest listing each removed path/hash/size and affected resume/evaluate entrypoints.',
        'A verifier that requires now-retired checkpoint tensors cannot remain advertised as runnable; CPU analysis from preserved saved outputs remains available. No automatic redownload/retrain to satisfy archival verifier.',
        'checkpoint_latest/rolling are not assumed byte-identical to endpoints; they are removed as unused historical states, never claimed deduplicated unless hash proves equality.',
    ],
    activity=dict(model_loads=0,model_forwards=0,GPU_calls=0,scoring_calls=0,deleted_files=0,large_weight_hashes=0),
    process_observation=dict(windows_detail='CIM access denied; basic Get-Process did not expose project Python',
        wsl='Read-only ps showed python PID485 running python -B -; no train/evaluate named process. Parent must identify PID485 before mutations; no process killed.',
        limitation='Point-in-time read-only process list, not open-file liveness proof.'))

for target,obj in [('backend_scope_checkpoint_audit.json',audit)]:
    with (OUT/target).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
with (OUT/'backend_scope_checkpoint_decisions.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(summary=summary,datasets=datasets,all_exist_and_size_match=all(r['exists'] and r['size_matches_inventory'] for r in rows)),indent=2))
