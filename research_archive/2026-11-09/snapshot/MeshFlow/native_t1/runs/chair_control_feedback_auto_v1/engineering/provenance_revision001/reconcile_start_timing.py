"""Read-only numeric-score provenance correction; never invoke scoring/model."""
from pathlib import Path
import shutil
import statistics
from experiments.mf_h1_local0.chair_control_feedback_auto_v1 import assets as a
from experiments.mf_h1_local0.chair_control_usability_quick_v1 import scoring as q


def run(out=a.OUT):
    out=a.resolve(out);dest=out/'engineering/provenance_revision001';dest.mkdir(parents=True,exist_ok=True)
    rows=[];protected={}
    for pool in ('TRAIN_FEEDBACK','DEV'):
        folder=out/'evaluation/START'/pool
        for name in ('per_output.json','summary.json','per_output_metrics.csv','per_parent_metrics.csv','per_noise_metrics.csv'):
            path=folder/name;protected[str(path)]=a.digest(path)
            original=dest/'originals'/pool/name;original.parent.mkdir(parents=True,exist_ok=True)
            if original.exists():assert a.digest(original)==a.digest(path)
            else:shutil.copyfile(path,original)
        metrics=a.read_json(folder/'per_output.json')
        for row in metrics:
            rid=row['request_id'];raw=out/'runtime/START'/rid/'raw.npz';rp=raw.with_name('result.json')
            assert raw.resolve()==a.resolve(row['raw_path']).resolve()
            result=a.read_json(rp)
            assert result['status']=='PASS' and result['label']=='START' and result['split']==pool
            assert a.digest(raw)==result['raw_sha256']==row['raw_sha256']
            assert a.digest(rp)==row['result_sha256']
            assert result['metadata']['accepted_states']==51 and result['metadata']['nfe']==50
            seconds=result['metadata'].get('seconds')
            assert isinstance(seconds,(int,float)) and seconds>=0
            rows.append(dict(request_id=rid,uid=row['uid'],candidate=row['candidate'],pool=pool,label='START',
                actual_generation_status='NEW_GENERATED',generation_time_source='actual_new_generation',
                generation_seconds=seconds,old_generation_status=row.get('generation_status'),
                old_generation_seconds=row.get('generation_seconds'),
                old_generation_time_source=row.get('generation_time_source'),
                raw_path=str(raw),raw_sha256=a.digest(raw),result_path=str(rp),result_sha256=a.digest(rp),
                recorded_rollout_nfe=result['metadata']['nfe'],scoring_rerun=False,model_calls_added=0))
    assert len(rows)==32 and len({r['request_id'] for r in rows})==32
    summary={}
    for pool in ('TRAIN_FEEDBACK','DEV'):
        group=[r for r in rows if r['pool']==pool];values=[r['generation_seconds'] for r in group]
        summary[pool]=dict(n=len(group),mean=statistics.mean(values),min=min(values),max=max(values),sum=sum(values),
            statuses={'NEW_GENERATED':len(group)},misclassified_original_rows=sum(r['old_generation_status']=='CACHED' for r in group))
    q.write_csv(dest/'actual_start_generation_per_output.csv',rows)
    q.write_json(dest/'actual_start_generation.json',dict(status='PASS',rows=rows,summary=summary))
    assert all(a.digest(Path(p))==sha for p,sha in protected.items())
    receipt=dict(status='PASS',version='presentation_provenance_revision001',script_sha256=a.digest(__file__),
        reason='Frozen adapter selected CACHED from presence of the optional historical raw_cache request metadata instead of the actual source raw path. All 32 scored raws are current runtime outputs; only DEV16 display provenance/timing were affected.',
        actual_new_rollouts=32,misclassified_original_rows=16,original_metrics_unchanged=True,
        source_identity_unchanged=True,geometry_unchanged=True,raw_unchanged=True,
        model_calls_added=0,scoring_calls_added=0,budget_calls_changed=False,
        recorded_total_nfe=sum(r['recorded_rollout_nfe'] for r in rows),
        budget_authority='Original root model-call ledger; this table does not create or replace budget events.',
        original_files_sha256=protected,outputs_sha256={p.name:a.digest(p) for p in
            (dest/'actual_start_generation_per_output.csv',dest/'actual_start_generation.json')},summary=summary)
    q.write_json(dest/'receipt.json',receipt)
    (dest/'README.md').write_text(
        'The frozen adapter misclassified the new START DEV outputs as historical CACHED because their requests retain a historical raw_cache pointer. '
        'The scorer actually read all 32 new runtime raws, verified their source hashes, full 51-state trajectories, and binary exports. '
        'This isolated derived table uses each actual result.json metadata.seconds. It preserves all original score JSON/CSV files and scientific values. '
        'No model, geometry scoring, budget events, checkpoints, or frozen code was changed. Missing actual timing would remain NA; all 32 timings are present.\n',encoding='utf-8')
    print(summary)
    return receipt


if __name__=='__main__':run()
