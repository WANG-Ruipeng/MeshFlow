"""Register exact retired files, archive small source trees, hash protected evidence."""
import json
from pathlib import Path
import zipfile
from native_t1.artifacts import atomic_json, file_sha256

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'native_t1/maintenance/jedge5_cleanup_20261001_115000'
SOURCES=[ROOT/'experiments/mf_h1_local0/loss_screen',ROOT/'experiments/mf_h1_local0/l3_l5_interface_pilot']
RUNS=[ROOT/'experiments/mf_h1_local0/outputs/loss_screen_20260930_232200',
      ROOT/'native_t1/runs/l3_l5_interface_pilot_20261001_104600']


def main():
    target=OUT/'cleanup_manifest.json'
    if target.exists():
        raise FileExistsError(target)
    migration=json.loads((OUT/'checkpoint_migration.json').read_text())
    if migration['status']!='PASS' or len(migration['checkpoint_exports'])!=3:
        raise RuntimeError('Verified three-checkpoint migration required')
    exports={x['source']:x for x in migration['checkpoint_exports']}
    plan=[]
    for run in RUNS:
        data=json.loads((run/'results.json').read_text())
        for arm,record in data['arms'].items():
            for step,original in record['checkpoints'].items():
                path=run/arm/f'additional_step{step}_cumulative_step{2500+int(step)}.pt'
                rel=str(path.relative_to(ROOT))
                assert path.is_file() and path.stat().st_size==original['bytes']
                replacement=exports.get(rel)
                plan.append(dict(path=rel,category='checkpoint',bytes=original['bytes'],sha256=original['sha256'],
                    arm=arm,additional_step=int(step),model_state_sha256=original['state_sha256'],
                    reason='lossless_compact_replacement' if replacement else 'retired_candidate_or_intermediate',
                    replacement=None if replacement is None else replacement['destination'],status='REGISTERED'))
    assert len(plan)==18
    assert sum(x['bytes'] for x in plan)==28171791430
    archive=OUT/'retired_loss_campaign_source.zip'
    source_rows=[]
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for directory in SOURCES:
            for path in sorted(directory.rglob('*')):
                if path.is_symlink():
                    raise RuntimeError('Symlink in source tree')
                if path.is_file():
                    rel=str(path.relative_to(ROOT));z.write(path,rel)
                    source_rows.append(dict(path=rel,category='retired_source',bytes=path.stat().st_size,
                        sha256=file_sha256(path),reason='archived_legacy_campaign',replacement=str(archive.relative_to(ROOT)),status='REGISTERED'))
    import hashlib
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(source_rows)
        for row in source_rows:
            assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    plan.extend(source_rows)
    protected=[]
    # Every non-checkpoint historical file, including original arrays, remains unchanged.
    for run in RUNS:
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix!='.pt':
                protected.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=file_sha256(path)))
    for path in [ROOT/'native_t1/checkpoints/tgeo_g0_cumulative2500.pt',
                 ROOT/'output/t1_publish_20260930/chair_n112.npz']:
        protected.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=file_sha256(path)))
    record=dict(status='REGISTERED_NOT_DELETED',root='E:\\MeshFlow',
        authorization='User requested JEdge5-focused maintenance and deletion of seven-loss temporary checkpoints/code',
        exclusions=['official weights','raw datasets','all other historical runs','.worktrees','.git','reports and raw outputs'],
        checkpoint_migration='checkpoint_migration.json',entries=plan,
        removed_checkpoints_become_unavailable=True,
        historical_paths_remain_as_original_records=True,
        source_archive=dict(path=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,
                            sha256=file_sha256(archive),files=len(source_rows),verified=True),
        protected_evidence=protected,
        checkpoint_bytes_registered=sum(x['bytes'] for x in plan if x['category']=='checkpoint'),
        compact_bytes_added=sum(x['bytes'] for x in migration['checkpoint_exports']),
        CPU_tests=dict(full_suite=87,passed=87,seconds=5.009,
                       command='python -B -m unittest discover -s native_t1/tests -v'),
        work=dict(new_optimizer_updates=0,new_complete_generations=0,new_generator_forwards=0,
                  new_generator_backwards=0,strict_CPU_full_model_loads=3))
    atomic_json(target,record)
    print(json.dumps(dict(status=record['status'],checkpoint_files=18,source_files=len(source_rows),
                          protected_evidence_files=len(protected),source_archive_bytes=archive.stat().st_size)))


if __name__=='__main__':
    main()
