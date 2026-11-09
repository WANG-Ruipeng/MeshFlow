"""Read-only training state / rolling paired-error snapshot; no framework import."""
from .common import *
import json

def main():
    status=read(OUT/'training'/ARM/'status.json');n=status['local_step'];lo=max(1,n-249);rows=[]
    for label,root in [('USER',OLD_TRAIN/'training/USER_V2_SEAM'),('EARLY',OUT/'training'/ARM)]:
        lines=(root/'exposure.jsonl').read_text().splitlines()
        data=[]
        for line in lines:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue  # A concurrently appended final line is not yet complete.
            if lo<=r['local_step']<=n:data.append(r)
        for half in ('early','late'):
            rr=[r for r in data if (r['t']<.5)==(half=='early')]
            rows.append(dict(label=label,half=half,instances=len(rr),unweighted_MSE=sum(r['all_free_velocity_sse'] for r in rr)/sum(r['all_free_scalar_count'] for r in rr)))
    result=dict(status=status['status'],local_step=n,window_start=lo,training_only_not_generation_quality=True,rows=rows)
    save(OUT/'analysis'/('progress_step'+str(n)+'.json'),result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
