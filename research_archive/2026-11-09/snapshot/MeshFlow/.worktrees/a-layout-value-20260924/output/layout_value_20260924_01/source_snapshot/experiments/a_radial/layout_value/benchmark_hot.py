"""Separate hot timings from each original cold/uncontrolled-start attempt."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from ..exporter import MeshExporter
from ..io_utils import write_json,array_hash
from .geometry import mesh_quality

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    protocol=json.loads((run/'RUN_PROTOCOL.json').read_text())
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    started=time.perf_counter();exporter=MeshExporter(a_seed=10)
    setup=time.perf_counter()-started
    # Explicit warmup on fixed first UID. No output is selected by quality.
    with np.load(run/'inputs'/f"{protocol['uids'][0]}.npz",allow_pickle=False) as z:warm=z['logr']
    warm_start=time.perf_counter()
    for arm in ['fixed','a','qem']:exporter.export(warm,512,arm)
    warmup=time.perf_counter()-warm_start
    rows=[]
    for uid in protocol['uids']:
        with np.load(run/'inputs'/f'{uid}.npz',allow_pickle=False) as z:y=z['logr']
        common=[]
        for repeat in range(3):
            t=time.perf_counter();fx,_=exporter.export(y,512,'fixed');q=mesh_quality(fx['vertices'],fx['faces']);exporter.support.mesh(y)
            common.append(time.perf_counter()-t)
        for arm in ['fixed','a','vtk_qem']:
            path=run/'attempts'/uid/arm/'result.json'
            if not path.exists():continue
            original=json.loads(path.read_text());times=[];same=[]
            for repeat in range(3):
                t=time.perf_counter();mesh,_=exporter.export(y,512,{'vtk_qem':'qem'}.get(arm,arm));times.append(time.perf_counter()-t)
                same.append(array_hash(mesh['vertices'],mesh['faces'])==original['mesh_hash'])
            rows.append({'uid':uid,'arm':arm,'hot_seconds':times,'median_hot_seconds':float(np.median(times)),
                'same_mesh_hash_all_repeats':all(same),'original_attempt_seconds':original['algorithm_seconds'],
                'common_quality_floor_and_reference_setup_seconds':common})
    write_json(run/'HOT_TIMING.json',{'exporter_startup_seconds':setup,'warmup_seconds':warmup,'rows':rows,
        'scope':'Separate measurement pass without optimization or CGAL running; 3 repeats; original geometry remains selected',
        'original_algorithm_times_include_first_call_overhead':True,
        'quality_floor_setup':'Reported separately: derived from Fixed and shared by comparison contract',
        'optimizer_not_retimed':'Single bounded original trajectories; no claim of asymptotic optimized implementation speed'})
    print(json.dumps({'hash_match_all':all(r['same_mesh_hash_all_repeats'] for r in rows),'rows':len(rows),
      'median_by_arm':{a:float(np.median([r['median_hot_seconds'] for r in rows if r['arm']==a])) for a in ['fixed','a','vtk_qem']}}))
if __name__=='__main__':main()

