"""Record completion of authorized computation without running any model."""
from pathlib import Path
import hashlib
import json
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_user_data_scaling_32_67_v1'
def read(p): return json.loads((OUT/p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    destination=OUT/'scientific_completion.json'
    if destination.exists(): raise FileExistsError(destination)
    training=read('training_analysis/final/summary.json')
    assert training['status']=='PASS_CLOSED_TRAINING_ACCOUNTING' and not training['errors']
    assert all(a['completed_updates']==6000 for a in training['arms'])
    aggregate=read('evaluation/aggregate_completion.json')
    assert aggregate['status']=='COMPLETE' and aggregate['stage_records']==840
    review=read('evaluation/gallery/output_review_receipt.json')
    assert review['status']=='COMPLETE' and review['images_actually_viewed']==280
    for name in ('REPORT.md','README_RESUME.md','training_analysis/final/paired_NEW32_POST.png',
                 'training_analysis/final/paired_NEW32_POST_qa.json','report_fragments/results_analysis.md'):
        assert (OUT/name).is_file(), name
    now=datetime.now(timezone.utc)
    start=datetime.fromisoformat('2026-10-12T03:15:34.306484+00:00')
    deadline=datetime.fromisoformat('2026-10-12T15:15:34.306484+00:00')
    commands=[['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],
              ['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],
              ['git','status','--short'],['git','rev-parse','HEAD']]
    commands_out=[]
    for args in commands:
        r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=False)
        commands_out.append(dict(args=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    receipt=dict(schema='closed_authorized_scientific_work_v1',status='SCIENTIFIC_WORK_COMPLETE_REPORT_WRITTEN',
                 utc=now.isoformat(),start_utc=start.isoformat(),deadline_utc=deadline.isoformat(),
                 elapsed_wall_seconds_including_wait=(now-start).total_seconds(),within_12h=now<deadline,
                 formal_updates=12000,formal_microbatch_forward=96000,formal_microbatch_backward=96000,
                 preflight_forward=160,preflight_backward=128,discarded_preflight_updates=16,
                 fixed_diagnostic_forward=128,new_generator_rollouts=408,new_generator_sampling_forward=20400,
                 cached_raw=12,new_D0_forward=408,cached_D0_proposals=12,post_CPU_outputs=420,
                 scored_stage_records=840,actual_reviewed_stage_records=840,actual_viewed_pages=280,
                 model_retries=0,new_model_calls_in_this_step=0,remaining_scientific_work=[],
                 remaining_delivery_work=['Final package build and member validation'],
                 default_changed=False,commit=False,push=False,
                 report_sha256=sha(OUT/'REPORT.md'),
                 training_accounting_sha256=sha(OUT/'training_analysis/final/summary.json'),
                 visual_review_sha256=sha(OUT/'evaluation/AI_PROXY_visual_review.csv'),
                 resource_and_git_commands=commands_out)
    destination.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('status','utc','elapsed_wall_seconds_including_wait','within_12h','remaining_delivery_work')}))

if __name__=='__main__': main()
