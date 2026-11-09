"""Serial, lock-yielding dispatch of already registered case groups."""
import argparse,json,subprocess,sys,time
from .registration import OUT,resource_snapshot
from experiments.mf_h1_local0.jedge5_diagnosis.common import read_json,write_json

def run(stage,start):
    log=OUT/f'dispatch/{stage}.json'
    if log.exists():raise RuntimeError('Dispatch already registered')
    records=[];write_json(log,dict(stage=stage,records=records,concurrency=1))
    for i in range(start,6):
        existing=list((OUT/'workers').glob('*.json'))
        used=sum(read_json(p)['counts']['model_forward_attempts'] for p in existing)
        reserve=148 if stage=='local' else 318
        if used+reserve>4500:raise RuntimeError('Global hard4500 budget pre-dispatch')
        rec=dict(case_id=i,resources=resource_snapshot(),actual_prior_calls=used,maximum_group_calls=reserve);records.append(rec)
        write_json(log,dict(stage=stage,records=records,concurrency=1))
        tick=time.perf_counter();p=subprocess.run([sys.executable,'-B','-m','experiments.mf_h1_local0.jedge5_local_error_B.worker',stage,'--case',str(i)])
        rec.update(returncode=p.returncode,subprocess_wall_seconds=time.perf_counter()-tick,resources_after_exit=resource_snapshot())
        write_json(log,dict(stage=stage,records=records,concurrency=1))
        if p.returncode or read_json(OUT/f'workers/{stage}_case{i}.json')['status']!='PASS':raise RuntimeError('Worker stopped; no automatic retry')
    subprocess.run([sys.executable,'-B','-m','experiments.mf_h1_local0.jedge5_local_error_B.worker','aggregate_'+stage],check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['local','suffix']);p.add_argument('--start',type=int,default=0);a=p.parse_args();run(a.stage,a.start)
