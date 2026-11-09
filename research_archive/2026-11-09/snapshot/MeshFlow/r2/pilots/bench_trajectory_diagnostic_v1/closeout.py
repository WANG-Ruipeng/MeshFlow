"""CPU-only final delivery and read-only recovery verification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
from .final_tables import rows
import argparse,shutil,difflib,zipfile
import numpy as np

def put_text(p,text):
    if p.exists():assert p.read_text(encoding='utf-8')==text,p
    else:p.write_text(text,encoding='utf-8')

def verify_complete():
    if not (OUT/'COMPLETE.json').exists():
        raise RuntimeError('Final closeout missing. No model dispatch; complete pending CPU review/report explicitly.')
    c=read(OUT/'COMPLETE.json');verify(c['files'])
    delivery=read(OUT/'delivery_hashes.json');verify(delivery['files'])
    verify(manifest()['files'])
    print(json.dumps(dict(status='COMPLETE_VERIFIED',generator_returned=c['generator_returned'],D0_calls=c['D0_calls'],
        training_updates=0,new_model_calls=0,verified_delivery_files=len(delivery['files']),zip_bytes=c['zip_bytes'])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only or (OUT/'COMPLETE.json').exists():verify_complete();return
    reg=manifest();verify(reg['files']);verify(read(OUT/'layouts/complete.json')['files'])
    numeric=read(OUT/'analysis_full/numeric_complete.json');verify(numeric['files'])
    assert numeric['generator_returned']==2382 and numeric['D0_calls']==32 and numeric['model_updates']==0
    assert numeric['max_endpoint_identity_error']<1e-6
    metrics=rows(OUT/'analysis_full/per_output_all_metrics.csv')
    assert len(metrics)==96 and all(r['C_preserved_exact']=='True' for r in metrics)
    review=read(OUT/'visual_review/complete.json');verify(review['files'])
    assert review['main_cells']==64 and review['total_distinct_reviewed_pages']==40
    stage=read(OUT/'stage_audit/complete.json');verify(stage['files'])
    functional=read(OUT/'weights/functional/complete.json');verify(functional['files'])
    assert len(rows(OUT/'weights/functional/delta_W_h.csv'))==1320
    requests=read(OUT/'evaluation_inputs/manifest.json')['requests'];records=[]
    d0_calls=0
    for label in LABELS:
        assert read(OUT/'execution'/label/'complete.json')['status']=='COMPLETE'
        for q in requests:
            root=OUT/'evaluation'/label/q['request_id'];post=read(root/'post/receipt.json')
            verify(post['files']);assert post['D0_forwards']==1
            assert sha(root/'raw/raw.npz')==post['raw_sha256']
            assert (OUT/'cpu'/label/q['request_id']/'CLEAN_FINAL_POST/post.npz').exists()
            d0_calls+=post['D0_forwards']
            tr=arrays(root/'raw/trajectory.npz')
            assert tr['states'].shape[0]==51 and tr['velocities'].shape[0]==50
            layout=arrays(OUT/'layouts'/q['condition_id']/'layout.npz');fixed=layout['anchor_ids']>=0
            assert np.all(tr['states'][:,fixed]==tr['states'][0,fixed][None])
            records.append(dict(label=label,request_id=q['request_id'],trajectory_states=51,velocities=50,
                C_and_anchor_all_states_exact=True,raw_sha256=post['raw_sha256'],post_sha256=post['post_sha256']))
    assert d0_calls==32
    csvwrite(OUT/'analysis_full/final_trajectory_integrity.csv',records)
    # Snapshot recovery-era source separately; do not overwrite original frozen snapshots.
    source=Path(__file__).parent;frozen=read(OUT/'source_hashes.json')['files'];changes=[]
    for name,old in frozen.items():
        new=sha(name)
        if old!=new:
            assert Path(name).is_relative_to(source),(name,'protected external dependency changed')
            if Path(name).name=='gpu.py':
                assert read(OUT/'recovery_dispatch.json')['source_hashes'][name]==new
                reason='Already recorded before recovered dispatch: exception-only partial trajectory saving; integration/forward unchanged'
            else:
                assert Path(name).name=='resume.py',(name,'scientific implementation changed')
                reason='Read-only guard for completed task; no change to inference implementation'
            changes.append(dict(path=name,before=old,after=new,reason=reason))
    save(OUT/'rollback_gate.json',dict(status='NOT_TRIGGERED',calls=0,
        reason='C sensitivity and static/local functional weight changes do not isolate a specific adverse module; no module sweep'))
    save(OUT/'final_integrity.json',dict(status='PASS',protected_registered_files=len(reg['files']),
        trajectories=32,three_stage_scores=96,C_all_scored_outputs_exact=True,source_changes=changes,
        generator_returned=2382,D0_calls=32,head_calls=4,training_updates=0,
        original_CUDA_failure='Preserved; cause UNKNOWN; explicit2-call replay counted',
        I_status_counts={s:sum(r['ConflictCov_C_status']==s for r in metrics) for s in ('PASS','LOW_SUPPORT')}))
    oldreport=OUT/'REPORT.md';history=OUT/'REPORT_PARTIAL_20261022.md';final=OUT/'REPORT_FINAL.md'
    if not history.exists():shutil.copy2(oldreport,history)
    shutil.copyfile(final,oldreport)
    snapshot=OUT/'source_snapshot_final';diff=[]
    for f in sorted(source.iterdir()):
        if f.suffix not in ('.py','.ps1'):continue
        rel=f.relative_to(ROOT);dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(f)==sha(dest)
        else:shutil.copy2(f,dest)
        old=OUT/'source_snapshot'/f.name
        before=old.read_text(encoding='utf-8').splitlines(True) if old.exists() else []
        diff.extend(difflib.unified_diff(before,f.read_text(encoding='utf-8').splitlines(True),
            fromfile=str(rel)+' (initial snapshot)' if old.exists() else '/dev/null',tofile=str(rel)+' (final)'))
    put_text(OUT/'code.diff',''.join(diff))
    events=OUT/'engineering_events.jsonl'
    if 'CPU_CLOSEOUT_CORRECTIONS' not in events.read_text(encoding='utf-8'):
        event(events,kind='CPU_CLOSEOUT_CORRECTIONS',pandas_missing='one failed table invocation; replaced by stdlib/numpy; no installation',
            source_guard='One closeout stopped on known recovery-era exception saver. Verified exact recovery hash and inspected diff before allowing; no model replay.',
            resource_metadata='inherited GPU_computation:false is not execution truth; ledgers record actual GPU calls',
            missing_backbone_MLP_activations='NA hook name mismatch; no new forward',new_model_calls=0)
    if 'MINI_PACKAGE_SCOPE_CORRECTION' not in events.read_text(encoding='utf-8'):
        event(events,kind='MINI_PACKAGE_SCOPE_CORRECTION',first_zip_bytes=11313871,
            action='Preserved incomplete packaging attempt; exclude verbose pair detector JSON caches from mini ZIP only. All CSV and local source pair evidence retained.',new_model_calls=0)
    # Hash human-readable deliverables and exact source snapshots. Heavy arrays stay in
    # existing input/receipt inventories; no duplicate model weights are packaged.
    small=[]
    for folder in ('analysis_full','visual_review','layouts','weights','source_reference','stage_audit','source_snapshot_final','scoring_source_snapshot'):
        for f in (OUT/folder).rglob('*'):
            if f.is_file() and f.suffix in ('.csv','.json','.py','.ps1','.md','.png'):
                small.append(f)
    for folder in ('main',):small.extend((OUT/'gallery_full'/folder).glob('*.png'))
    small.extend(OUT/'gallery_full'/f for f in ('index.html','complete_main_COMMON_USER.json','complete_oracle_continuation.json'))
    small.extend(OUT/f for f in ('REPORT.md','REPORT_FINAL.md','DELIVERY_INDEX.md','registration.json','selection.json',
        'source_hashes.json','recovery_authorization.json','recovery_dispatch.json','engineering_events.jsonl','rollback_gate.json','final_integrity.json','code.diff'))
    small=sorted(set(small));assert all(f.exists() for f in small)
    save(OUT/'delivery_hashes.json',dict(status='FROZEN',files={str(f):sha(f) for f in small},
        heavy_assets='Original checkpoint and input hashes in registration; RAW/D0/POST files in per-request receipts; no weights in mini package'))
    zpath=OUT/'BENCH_TRAJECTORY_DIAGNOSTIC_REVIEW.zip'
    packaged=[f for f in small if not (f.name=='pairs.json' or f.name.endswith('_pairs.json'))]
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in packaged+[OUT/'delivery_hashes.json']:z.write(f,f.relative_to(OUT).as_posix())
        z.writestr('PACKAGE_SCOPE.txt','Small evidence package: report, complete numeric CSVs, actual review,16 main pages and code. Verbose pair-detector JSON caches, large raw arrays, checkpoints, oracle/continuation/five-stage images remain at local paths. Full gallery index has local-only links to those pages. Delivery hash list also covers local-only evidence. No model weights included.\n')
    assert zpath.stat().st_size<10_000_000,('review zip exceeds10MB',zpath.stat().st_size)
    save(OUT/'COMPLETE.json',dict(status='COMPLETE',experiment='R2_BENCH_TRAJECTORY_DIAGNOSTIC_V1',
        generator_returned=2382,core_useful_calls=2380,replayed_returned_calls=2,D0_calls=32,head_calls=4,training_updates=0,
        raw_outputs=32,score_rows=96,rendered_pages=64,actual_reviewed_pages=40,module_rollback='NOT_TRIGGERED',
        zip_bytes=zpath.stat().st_size,files={str(f):sha(f) for f in (OUT/'delivery_hashes.json',zpath,OUT/'final_integrity.json')},
        lifecycle='Fixed diagnosis complete; no automatic training, new sampling, cleanup, default switch, commit or push'))
    verify_complete()

if __name__=='__main__':main()
