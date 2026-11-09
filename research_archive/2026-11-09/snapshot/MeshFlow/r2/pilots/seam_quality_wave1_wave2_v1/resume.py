"""Recoverable explicit stage entrypoint; completed tasks only verify/status."""
import argparse,json,subprocess,sys
from .protocol import ROOT,OUT,ARMS

def status():
    result={}
    for p in [OUT/'cleanup/completion.json',OUT/'protocol.json',OUT/'wave1_exports.json',
              OUT/'wave2_inputs/preparation_completion.json',OUT/'completion.json',
              *[OUT/'evaluation'/a/(s+'_complete.json') for a in ARMS for s in ('raw','post')],
              *[OUT/'training'/a/'status.json' for a in ('CLEAN_BULK_FM','SURGICAL_LATE4')]]:
        if p.exists():
            v=json.loads(p.read_text(encoding='utf-8-sig'));result[str(p.relative_to(OUT))]={k:v[k] for k in ('status','local_step','global_step','outputs','wall_seconds') if k in v}
        else:result[str(p.relative_to(OUT))]={'status':'NOT_COMPLETE'}
    print(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',nargs='?',default='status',choices=['status','exports','generation','training','training_prepare','scoring','binding','aggregation','gallery','audit'])
    args,extra=p.parse_known_args()
    if args.stage=='status':status();return
    if (OUT/'completion.json').exists():raise SystemExit('Fixed task closed; status only, no implicit extension')
    raise SystemExit(subprocess.call([sys.executable,'-B','-m','r2.pilots.seam_quality_wave1_wave2_v1.'+args.stage,*extra],cwd=ROOT))

if __name__=='__main__':main()
