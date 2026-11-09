import argparse,json,time
from r2.pilots.bench_high_joint_coverage_v1.common import OUT,resources,read,write_new

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage');a=parser.parse_args()
    if not a.stage.replace('_','').isalnum():raise ValueError('safe stage name required')
    result=dict(stage=a.stage,time_unix=time.time(),resources=resources(),jobs={})
    p=OUT/'training/WIDE_HIGH/status.json'
    if p.exists():
        s=read(p);result['training']={k:s[k] for k in ('status','stage_step','global_step','elapsed_seconds','last') if k in s}
    for p in (OUT/'workflow').glob('*/status.json'):
        s=read(p);result['jobs'][p.parent.name]={k:s[k] for k in ('status','seconds','exit_code','pid','child_pid','error') if k in s}
    dest=OUT/'dispatch'/(a.stage+'_'+str(time.time_ns())+'.json');write_new(dest,result)
    print(json.dumps(dict(path=str(dest),training=result.get('training'),jobs=result['jobs'],
        GPU=result['resources']['gpu']['stdout'],RAM_available=result['resources']['RAM_available']),ensure_ascii=False))

if __name__=='__main__':main()
