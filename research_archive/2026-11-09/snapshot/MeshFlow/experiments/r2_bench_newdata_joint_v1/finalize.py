"""Close the completed experiment from immutable receipts, no scientific calls."""
from pathlib import Path
import json,csv,hashlib,datetime
O=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    audit=read(O/'final_audit.json');assert audit['status']=='PASS'
    independent=read(O/'results/key_metrics_independent_audit.json');assert independent['status']=='PASS'
    visual=O/'visual_review/aggregates/5cc5f7e7ba6271d6/summary.json'
    vr=read(visual);assert vr['reviewed_cells']==168 and vr['reviewed_pages']==84 and vr['missing_cells']==0
    stages=['train','raw-baseline-v2','raw-new','d0-baseline','d0-new','score-raw','score-raw-baseline','score-post','collect','gallery']
    statuses={n:read(O/'workflow'/n/'status.json') for n in stages}
    assert all(d['status']=='COMPLETE' for d in statuses.values())
    assert read(O/'workflow/raw-baseline/status.json')['exit_code']==1
    required=['REPORT.md','results/METRIC_TABLES.md','results/per_output.csv','results/paired_per_condition.csv','results/paired_per_parent.csv',
        'results/training_summary.json','results/loss_curves.png','results/head_per_edge.csv','results/key_metrics_summary.json',
        'results/key_metrics_independent_audit.json','final_audit.json','RUNBOOK.md','registration.json','eval_manifest.json',
        'code_evidence/source_hashes.json','code_evidence/code.diff','gallery/index.html','visual_review/aggregates/5cc5f7e7ba6271d6/summary.json']
    hashes={rel:sha(O/rel) for rel in required}
    text=(O/'REPORT.md').read_text(encoding='utf-8')
    assert '\ufffd' not in text and '本版不晋级' in text
    records=list(csv.DictReader((O/'results/per_output.csv').open(encoding='utf-8-sig')))
    assert len(records)==168 and all(r['C_preserved_exact']=='True' for r in records)
    result=dict(schema='R2_BENCH_NEWDATA_JOINT_COMPLETION_V1',status='COMPLETE',
        experiment='R2_BENCH_NEWDATA_JOINT_V1',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        decision='MIXED_LOCAL_SIGNALS_NOT_PROMOTED',data=dict(training_parents=27,training_conditions=92,added_parents=1,added_conditions=1),
        training=audit['training'],sampling=audit['sampling']['counts'],metric_rows=168,metric_fields=177,
        visual_review=dict(status=vr['status'],pages=84,cells=168,kind='AI_PROXY',blind=False,human_ratings=0,receipt=str(visual)),
        historical_failure=dict(stage='raw-baseline',actual_neural_calls=0,extra_generator_loads=1,
            resolution='Separate raw-baseline-v2 completed; original failure preserved'),
        stage_seconds={n:d['seconds'] for n,d in statuses.items()},files=hashes,
        code_sha256=sha(Path(__file__)),default_changed=False,commit=False,push=False,deleted_weights=False,
        further_training=False,further_experiments=False,
        scope='One fixed 1500-update arm, fixed 28 requests/model, no best selection; one added parent and joint head adaptation, not pure data-only effect.')
    target=O/'COMPLETE.json'
    if target.exists():
        prev=read(target);assert prev['files']==hashes and prev['status']=='COMPLETE'
        print(json.dumps({'status':'ALREADY_COMPLETE','path':str(target)}));return
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'report_sha256':hashes['REPORT.md'],'path':str(target)}))
if __name__=='__main__':run()
