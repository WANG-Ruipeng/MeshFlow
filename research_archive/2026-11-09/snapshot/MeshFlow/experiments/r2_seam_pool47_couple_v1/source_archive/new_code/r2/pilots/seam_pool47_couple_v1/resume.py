"""Explicit resumable stage entrypoint. Default is read-only status."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_pool47_couple_v1'
PREFIX='r2.pilots.seam_pool47_couple_v1.'

def status():
    paths={'protocol':'protocol.json','shared_data':'shared_data_manifest.json',
      'PLAIN3000':'training/PLAIN3000/status.json','COUPLE3000':'training/COUPLE3000/status.json',
      'cpu_binding':'cpu_binding/completion.json','corner_rule':'corner_rule/export_recovery/completion.json',
      'generation':'generation_registration.json','final':'completion.json'}
    result={}
    for key,rel in paths.items():
        p=OUT/rel
        if p.exists():
            v=json.loads(p.read_text(encoding='utf-8-sig'))
            result[key]={k:v[k] for k in ('status','local_step','global_step','cells','samples') if k in v}
        else:result[key]={'status':'NOT_COMPLETE'}
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',nargs='?',default='status',choices=['status','data-prepare','data-cache','train','generation-prepare','raw','post','cpu-binding','dev-diagnosis','score','score-summary','endpoint-binding','results','gallery-manifest','training-summary'])
    p.add_argument('--arm',choices=['PLAIN3000','COUPLE3000']);a=p.parse_args()
    if a.stage=='status':status();return
    if (OUT/'completion.json').exists():raise SystemExit('Fixed experiment is closed. Status only; no automatic extension.')
    commands={'data-prepare':['data','prepare'],'data-cache':['data','cache'],
      'generation-prepare':['generation','prepare'],'cpu-binding':['cpu_binding','--stage','run'],
      'dev-diagnosis':['cpu_binding','--stage','diagnose'],
      'score':['endpoint_scoring','score'],'score-summary':['endpoint_scoring','summarize'],
      'endpoint-binding':['endpoint_binding','run'],'results':['results','summarize'],
      'gallery-manifest':['results','gallery'],'training-summary':['training_summary']}
    if a.stage in ('train','raw','post'):
        if not a.arm:p.error('--arm required')
        cmd=['training','--arm',a.arm] if a.stage=='train' else ['generation',a.stage,'--arm',a.arm]
    else:cmd=commands[a.stage]
    raise SystemExit(subprocess.call([sys.executable,'-B','-m',PREFIX+cmd[0],*cmd[1:]],cwd=ROOT))

if __name__=='__main__':main()
