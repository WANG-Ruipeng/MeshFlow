"""Close the fixed task only after all real evidence and full independent audit pass."""
import argparse,json,time
from pathlib import Path
from .protocol import ROOT,OUT,ARMS,WAVE1,WAVE2
from r2.pilots.seam_abc_v1 import generate as G

def run(audit_path):
    target=OUT/'completion.json'
    if target.exists():
        old=G.read(target)
        for p,d in old['evidence_hashes'].items():
            if G.sha(p)!=d:raise ValueError('Closed evidence changed: '+p)
        print('COMPLETE_VERIFIED');return old
    audit_path=Path(audit_path).resolve();audit=G.read(audit_path)
    if audit['status']!='PASS' or audit['scope']!='FULL':raise ValueError('Full independent PASS required')
    required=[OUT/'REPORT.md',OUT/'WAVE1_RESULTS.md',OUT/'analysis_policy.json',OUT/'TASK.md',
        OUT/'protocol.json',OUT/'cleanup/completion.json',OUT/'visual_review.csv',OUT/'visual_summary.csv',
        OUT/'visual_completion.json',OUT/'training_summary/completion.json',OUT/'training_summary/concurrency_phases.json',
        OUT/'results/final/completion.json',OUT/'results/final/KEY_TABLES.md',OUT/'results/final/quality_trajectory.png',
        OUT/'EXECUTION_NOTES.md',OUT/'report_preparation/report_review.json',
        OUT/'report_preparation/closeout_first_attempt.json',audit_path]
    if G.read(OUT/'results/final/completion.json')['rows']!=672:raise ValueError('Incomplete final matrix')
    if G.read(OUT/'visual_completion.json')['new_POST']!=140:raise ValueError('Incomplete actual review')
    for a in ARMS:
        for stage in ('raw','post'):
            p=OUT/'evaluation'/a/(stage+'_complete.json');required.append(p)
            if G.read(p)['outputs']!=28:raise ValueError('Incomplete evaluation')
    weights=[]
    for arm in ('CLEAN_BULK_FM','SURGICAL_LATE4'):
        p=OUT/'training'/arm/'status.json';required.append(p);s=G.read(p)
        if s['status']!='COMPLETE' or s['local_step']!=1000 or s['global_step']!=14000:raise ValueError('Training incomplete')
        for key in ('checkpoint','inference_export'):
            metadata=dict(s[key])
            if metadata.get('arm',arm)!=arm:raise ValueError('Weight arm identity mismatch')
            metadata.update(arm=arm,kind=key);weights.append(metadata)
    for label in WAVE1:
        p=OUT/'inference'/(label+'_generator.json');required.append(p)
    code_dir=ROOT/'r2/pilots/seam_quality_wave1_wave2_v1'
    source_files=sorted(p for p in code_dir.iterdir() if p.is_file() and p.suffix in ('.py','.ps1','.md'))
    source_hashes={str(p):G.sha(p) for p in source_files}
    G.write(OUT/'source_hashes.json',dict(scope='All isolated experiment implementation files at closure',files=source_hashes))
    required.append(OUT/'source_hashes.json')
    text=(OUT/'REPORT.md').read_text(encoding='utf-8')
    if '\ufffd' in text or '????' in text or 'PENDING' in text or '待完成' in text:raise ValueError('Report has incomplete/encoding marker')
    result=dict(status='COMPLETE',task='R2_SEAM_QUALITY_WAVE1_WAVE2_V1',closed_unix=time.time(),
        execution=dict(new_RAW=140,generator_sampling_forwards=7000,D0_forwards=140,new_BOUND015=140,
            new_score_cells=420,training_arms=2,updates_each=1000,training_forward_each=8000,training_backward_each=8000,
            head_forward_calls=0,new_OT_calls=0,baseline_regeneration=0,final_metric_rows=672,numeric_fields=124,
            actual_new_POST_visual_reviews=140),
        cleanup=G.read(OUT/'cleanup/completion.json'),retained_new_weights=weights,
        independent_audit=str(audit_path),independent_audit_status='PASS',
        evidence_hashes={str(p):G.sha(p) for p in required},
        stopping_contract='Fixed task complete. No automatic additional steps, combinations, alpha sweep, default change, commit or push.',
        recovery='python -B -m r2.pilots.seam_quality_wave1_wave2_v1.resume status')
    G.write(target,result)
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(stage='CLOSEOUT',status='COMPLETE',time_unix=time.time(),completion_sha256=G.sha(target)))+'\n')
    print(json.dumps(dict(status='COMPLETE',completion=str(target),evidence_files=len(required))));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);run(p.parse_args().audit)
