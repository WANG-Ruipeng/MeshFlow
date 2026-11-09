"""Record the pre-forward package-path repair, without model construction."""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_hinge_quality_v1'

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('before','after'));a=p.parse_args()
    package=importlib.util.find_spec('meshflow_control')
    module=importlib.util.find_spec('meshflow_control.geo_io')
    root=OUT/'generation_engineering';root.mkdir(exist_ok=True)
    r=dict(stage=a.stage,unix=time.time(),package_origin=package.origin if package else None,
           geo_io_origin=module.origin if module else None,sys_path=sys.path,
           neural_calls_in_this_check=0,model_constructions=0)
    if a.stage=='before':
        ar=OUT/'evaluation/CLEAN_EDGE_COH'
        assert not list(ar.glob('*/raw/calls.jsonl'))
        assert not list(ar.glob('*/raw/attempt.json'))
        r.update(failed_launch='ModuleNotFoundError: meshflow_control.geo_io',
                 failure_location='owner_layout_v1.generation.raw import, before _load_generator and before any request directory',
                 consumed_generator_forwards=0,consumed_requests=0,
                 attempts_preserved=[str(x) for x in ar.glob('generation_attempt_*')],
                 repair='Explicit PYTHONPATH=/mnt/e/MeshFlow/r2/src for generation and D0 workers; frozen code and definitions unchanged')
    else:
        assert module and Path(module.origin).resolve()==ROOT/'r2/src/meshflow_control/geo_io.py'
        assert package and Path(package.origin).resolve()==ROOT/'r2/src/meshflow_control/__init__.py'
        r['geo_io_sha256']=hashlib.sha256(Path(module.origin).read_bytes()).hexdigest()
        r['status']='PASS'
    with (root/(a.stage+'.json')).open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
    print(json.dumps({k:v for k,v in r.items() if k!='sys_path'}))

if __name__=='__main__':main()
