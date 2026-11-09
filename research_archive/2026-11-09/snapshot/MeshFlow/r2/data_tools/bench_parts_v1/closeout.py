"""CPU-only delivery accounting and source snapshot; no new data selection."""
from __future__ import annotations
import datetime, json, shutil
from pathlib import Path
from .publish import OUT, ROOT, read, sha, write, immutable_bytes
from .run import verify_final


def main():
    check=verify_final();assert check['status']=='PASS',check
    independent=read(OUT/'independent_export_verification.json');assert independent['status']=='PASS'
    assert independent['manifest_sha256']==sha(OUT/'data/manifest.json')
    manifest=read(OUT/'data/manifest.json');summary=manifest['summary']
    renderer=read(OUT/'gallery/render_execution.json')
    audits=[read(OUT/p) for p in ('quality/summary.json','quality/supplement_3461/summary.json','quality/combinations/summary.json')]
    ledger=dict(status='COMPLETE',closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        start_resources=read(OUT/'resources_initial.json'),
        GPU_model_workers=0,model_forward_calls=0,training_updates=0,new_completion_samples=0,new_noise_arrays=0,
        source_mesh_changes=0,optimizer_or_weight_writes=0,commit_or_push=0,
        dispatch='Independent CPU preparation/audit/render processes; one numerical thread per process. Audit and rendering used two overlapping workers; publication, independent verification and final rendering were dispatched separately with a cap of three CPU workers. Exact overlap duration was not profiled. AI_PROXY image review was dispatched separately.',
        candidates=342,active_conditions=316,excluded=26,summary=summary,
        phases=dict(base_construction=read(OUT/'candidates/execution.json'),
            geometry_audits=[dict(candidates=q['candidates'],counts=q['counts'],wall_seconds=q['wall_seconds'],ledger=q['ledger']) for q in audits],
            rendering=renderer,independent_export_check=dict(status=independent['status'],conditions=independent['conditions_checked'],wall_seconds=independent['wall_seconds']),
            publication=dict(status='COMPLETE',wall_seconds='NA_NOT_RECORDED_AS_AN_ISOLATED_PHASE',
                attempts=2,first_attempt_failure='Final semantic CSV duplicate keyword after arrays/manifest were saved',
                continuation='Reused and checked all previously written arrays; finished CSV. No new candidate/geometry scan/model call.')),
        visual_review=dict(candidate_rows=342,reviewer_type='AI_PROXY',blinding='NONE',candidate_parent_sheets=71+1+62,generated_output_reviews=0),
        final_verification=check,
        timing_note='Wall times include file IO and may include parallel contention. They are not isolated benchmarks.')
    write(OUT/'execution_ledger.json',ledger)
    snapshots=OUT/'code_snapshot';snapshots.mkdir(exist_ok=True)
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.is_file() and p.suffix in ('.py','.md'):
            dest=snapshots/p.name
            if dest.exists():assert sha(dest)==sha(p)
            else:shutil.copyfile(p,dest)
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    index=f'''# Bench multi-part data delivery

- [Report](REPORT.md)
- [Filterable gallery](gallery/index.html)
- [316 condition rows](data/conditions.csv)
- [TRAIN277 / DEV39 manifests](data/manifest.json)
- [Per-parent part coverage](data/parent_coverage.csv)
- [342 actual AI_PROXY reviews](data/semantic_review.csv)
- [26 exclusions](data/excluded.csv)
- [255 current SEAM-compatible conditions](data/current_seam_compatible.json)
- [61 valid conditions needing entry/layout extension](data/native_requires_entry_extension.json)
- [Independent exact-array verification](independent_export_verification.json)
- [Execution ledger](execution_ledger.json)
- [CPU resume/verify entry](../../r2/data_tools/bench_parts_v1/run.py)
- [Source snapshot](code_snapshot/README.md)

`data/inputs/` contains C-only NPZ + OBJ. `data/paired/` contains C/Y supervision.
The complete source mesh and source owner/alias labels are never in the inference pack.
72 parents, unchanged TRAIN64 / DEV8; additional conditions are not additional parents.

Local artifact size at closeout: {total:,} bytes ({total/2**20:.2f} MiB), excluding this small index/hash inventory.
No weights or original complete source meshes are copied into this delivery.
No training, generator/denoiser calls, new noise, commit or push.
'''
    immutable_bytes(OUT/'DELIVERY_INDEX.md',index.encode('utf-8'))
    files={str(p.relative_to(OUT)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
    write(OUT/'artifact_hashes.json',dict(files=files,source_code={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file() and p.suffix in ('.py','.md')}))
    print(json.dumps(dict(status='COMPLETE',active_conditions=summary['active_conditions'],artifact_count=len(files),artifact_bytes=sum(v['bytes'] for v in files.values()),model_calls=0),indent=2))

if __name__=='__main__':main()
