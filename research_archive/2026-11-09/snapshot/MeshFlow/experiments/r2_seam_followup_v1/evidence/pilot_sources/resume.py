"""Read-only status by default; explicit stages never silently retry attempts."""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_followup_v1'

def status():
    stages={}
    for name,path in [('layout_RAW',OUT/'layout/raw_complete.json'),('layout_POST',OUT/'layout/post_complete.json'),
                      ('layout_scores',OUT/'layout/evaluation/score_completion.json'),('binding',OUT/'binding/completion.json'),
                      ('final',OUT/'completion.json')]:
        stages[name]=json.loads(path.read_text(encoding='utf-8-sig')).get('status','RECORDED') if path.exists() else 'NOT_COMPLETE'
    print(json.dumps(dict(stages=stages,new_model_calls=0),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['status','layout-raw','layout-post','layout-prepare','layout-score','layout-summary','binding'],default='status')
    a=p.parse_args()
    if a.stage=='status':status();return
    if (OUT/'completion.json').exists():
        raise SystemExit('Fixed experiment complete: no automatic further execution; use --stage status.')
    commands={
        'layout-raw':['r2.pilots.seam_followup_v1.layout','raw'],
        'layout-post':['r2.pilots.seam_followup_v1.layout','post'],
        'layout-prepare':['r2.pilots.seam_followup_v1.layout_score','prepare'],
        'layout-score':['r2.pilots.seam_followup_v1.layout_score','score'],
        'layout-summary':['r2.pilots.seam_followup_v1.layout_score','summarize'],
        'binding':['r2.pilots.seam_followup_v1.binding','--stage','run'],
    }
    raise SystemExit(subprocess.call([sys.executable,'-B','-m',*commands[a.stage]],cwd=ROOT))

if __name__=='__main__':main()
