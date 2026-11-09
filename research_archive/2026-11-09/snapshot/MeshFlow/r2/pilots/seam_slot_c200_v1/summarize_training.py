"""Aggregate saved training logs; no torch/model/geometry execution."""
from collections import defaultdict
import json
from . import common as C

def records(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def run():
    rows=[];diagnostics=[];occupancy=[];curves=[]
    for arm in C.ARMS:
        folder=C.OUT/'training'/arm
        if not (folder/'status.json').exists():continue
        status=C.read(folder/'status.json')
        train=records(folder/'training_log.jsonl') if (folder/'training_log.jsonl').exists() else []
        ds=records(folder/'geometry_diagnostics.jsonl') if (folder/'geometry_diagnostics.jsonl').exists() else []
        rows.extend(dict(arm=arm,**r) for r in train);diagnostics.extend(dict(arm=arm,**r) for r in ds)
        if (folder/'occupancy_training.jsonl').exists():occupancy.extend(dict(arm=arm,**r) for r in records(folder/'occupancy_training.jsonl'))
        bins=defaultdict(list)
        for r in ds:bins[(1 if r['local_step']<=50 else 51 if r['local_step']<=100 else 101 if r['local_step']<=150 else 151,r['time_bin'])].append(r)
        for (start,binid),group in sorted(bins.items()):
            row=dict(arm=arm,step_first=start,step_last=start+49,time_bin=binid,samples=len(group))
            for kind in ('source_interface_third','source_interface_other_free','bulk'):
                count=sum(r[kind+'_scalar_count'] for r in group)
                for tag in ('velocity','endpoint'):
                    sse=sum(r[kind+'_'+tag+'_sse'] for r in group)
                    row[kind+'_'+tag+'_mse']=sse/count if count else None
                row[kind+'_scalar_count']=count
            row['normal_cosine_face_weighted']=sum(r['source_winding_normal_cosine_mean']*r['normal_comparable_faces'] for r in group if r['normal_comparable_faces'])/sum(r['normal_comparable_faces'] for r in group)
            curves.append(row)
    for name,value in [('training_per_update',rows),('training_per_micro',diagnostics),('occupancy_training_per_micro',occupancy),('learning_curve_time_buckets',curves)]:C.csvwrite(C.OUT/'statistics'/(name+'.csv'),value,replace=True)
    return dict(updates=len(rows),microbatches=len(diagnostics),head_microbatches=len(occupancy),curve_rows=len(curves))
if __name__=='__main__':print(json.dumps(run()))
