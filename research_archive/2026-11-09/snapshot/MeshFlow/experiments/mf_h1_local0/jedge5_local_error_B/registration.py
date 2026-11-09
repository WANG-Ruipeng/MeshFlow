"""Frozen registration and protected-input ledger for window B only."""
from native_t1 import runtime as _runtime
from pathlib import Path
import subprocess, time, json
import numpy as np
import torch
from experiments.mf_h1_local0.jedge5_diagnosis.common import ROOT,ASSETS,EVALUATION,load_cases,load_model,read_json,write_json,file_sha256,array_hash
from .math_ops import RESOLUTION_RULE
ARCHIVE=ROOT/'native_t1/runs/jedge5_diagnosis_20261002_102215'
OUT=ROOT/'native_t1/runs/jedge5_local_error_B_20261002_114200'
CASES=[(0,0,10401),(1,0,10401),(0,4,10401),(0,4,10402),(1,2,10401),(1,2,10402)]
INDICES=[0,10,20,35,49,50,51,65,75,85,95,98]
CUTS=[20,50,80]

def resource_snapshot():
    def run(args):
        p=subprocess.run(args,capture_output=True,text=True);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    return dict(time=time.time(),gpu=run(['nvidia-smi']),gpu_csv=run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),processes=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']),cpu_cores=run(['nproc']),memory=run(['free','-b']),load=run(['uptime']))

def bit_equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint32),b.view(np.uint32))

def register():
    dest=OUT/'registration/protocol.json'
    if dest.exists():raise RuntimeError('Registration exists; do not overwrite')
    OUT.mkdir(parents=True,exist_ok=True)
    old=read_json(ARCHIVE/'registration/protocol.json'); assert old['generation_seeds']==[10401,10402]
    assert old['sampler']=='direct FP32 t=k/NFE, FP32 v*dt Euler; C untouched'
    cases=load_cases(); identities=[]; records=[]
    noises_path=ARCHIVE/'registration/noises.npz'
    assert file_sha256(noises_path)==old['noise_sha256']
    with np.load(noises_path,allow_pickle=False) as z: noises={s:z[f'seed_{s}'].copy() for s in [10401,10402]}
    for model in ('B','J'):
        obj,identity=load_model(model,device='cpu');identities.append(dict(label=model,**identity));del obj
        assert not torch.cuda.is_initialized()
        historic=next(v for v in old['checkpoint_identities'] if v['label']==model)
        assert identity['state_sha256']==historic['state_sha256'] and identity['checkpoint_sha256']==historic['checkpoint_sha256']
        previous=read_json(ARCHIVE/f'A/{model}/results.json');assert previous['status']=='PASS'
        for cid,(p,j,s) in enumerate(CASES):
            selected=[r for r in previous['rows'] if (r['parent'],r['patch'],r['seed'],r['precision'],r['nfe'])==(p,j,s,'fp32',100)]
            assert len(selected)==1;r=selected[0];path=ARCHIVE/r['path'];assert file_sha256(path)==r['sha256']
            with np.load(path,allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
            assert a['states'].shape==(101,112,9) and a['states'].dtype==np.float32
            assert np.array_equal(a['state_indices'],np.arange(101)) and bit_equal(a['raw'],a['states'][-1])
            C=cases[p,j]['C'].astype(np.float32);assert bit_equal(a['C'],C)
            assert bit_equal(a['states'][:,:12],np.broadcast_to(C.reshape(12,9),(101,12,9)).copy())
            assert bit_equal(a['states'][0,12:],noises[s][0,12:]) and array_hash(torch.from_numpy(noises[s]))==r['audit']['noise_sha256']
            audit=r['audit'];assert audit['known_slots']==list(range(12)) and audit['free_slots']==list(range(12,112))
            assert audit['total_faces']==audit['valid_faces']==112 and audit['known_all_states_bitwise']
            records.append(dict(case_id=cid,model=model,parent=p,patch=j,seed=s,path=str(path),sha256=r['sha256'],bbox=cases[p,j]['bbox'],known_slots=audit['known_slots'],free_slots=audit['free_slots'],valid_slots=list(range(audit['valid_faces'])),noise_sha256=audit['noise_sha256'],C_sha256=array_hash(C),all_101_states_saved=True,mask_source='Reconstructed from explicit archived audited known/free slot lists and valid_faces; not re-inferred from geometry'))
    paths=[ROOT/'AGENTS.md',ROOT/'WORKSPACE.md',EVALUATION,noises_path,ARCHIVE/'registration/protocol.json',ARCHIVE/'stage_c/teacher_context_time.csv',ARCHIVE/'stage_c/parameter_velocity.csv']
    paths += [ASSETS[m]['path'] for m in ('B','J')]
    paths += [Path(r['path']) for r in records]
    paths += list((ROOT/'native_t1').glob('*.py'))+list((ROOT/'native_t1/losses').glob('*.py'))+list((ROOT/'models').glob('*.py'))
    paths += list((ROOT/'experiments/mf_h1_local0/jedge5_diagnosis').glob('*.py'))
    paths += [ROOT/'configs/snet/base-120m-ot-v-chair.yaml']
    protected={str(p):file_sha256(p) for p in paths}
    protocol=dict(status='FROZEN',archive=str(ARCHIVE),case_order=CASES,indices=INDICES,cuts=CUTS,models=identities,trajectories=records,noise_path=str(noises_path),precision='FP32 all',h=.01,etas=[1e-4,5e-5],resolution_rule=RESOLUTION_RULE,growth_indices='all absolute nodes from cut+1 through100',supplemental_t08=dict(forwards=24,reason='Required .8 direction and suffix cuts absent from primary12 local times; minimal prerequisite within4500; no replacement of original times'),budget=dict(local_primary=864,t08_supplement=24,directions_max=144,suffix=1764,total_planned_max=2796,hard_limit=4500,full_replay_planned=0,updates=0,OT=0,backward=0),dispatch=dict(gpu_workers=1,lock='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock',groups='one case with both models; release before next case',reason='Explicit window B single-worker/exclusive shared lock contract overrides otherwise possible VRAM concurrency',headroom_MiB=2048),resources=resource_snapshot(),source_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protected=protected)
    write_json(dest,protocol);print(json.dumps(dict(status='PASS',trajectories=len(records),CUDA_initialized=torch.cuda.is_initialized(),run=str(OUT))))

def check_protected():
    protocol=read_json(OUT/'registration/protocol.json');changed=[p for p,s in protocol['protected'].items() if file_sha256(p)!=s]
    result=dict(status='PASS' if not changed else 'FAIL',files=len(protocol['protected']),changed=changed);write_json(OUT/'registration/protected_after.json',result);return result
if __name__=='__main__':register()

