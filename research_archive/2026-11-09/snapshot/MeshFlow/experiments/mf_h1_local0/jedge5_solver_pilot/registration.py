"""Single-model four-case solver comparison, fixed before model calls."""
from pathlib import Path
import subprocess,time
import numpy as np
from experiments.mf_h1_local0.jedge5_diagnosis.common import ROOT,ASSETS,load_cases,noise_bank,read_json,write_json,file_sha256,array_hash
OLD=ROOT/'native_t1/runs/jedge5_diagnosis_20261002_102215'
BLACKBOX=ROOT/'native_t1/runs/jedge5_blackbox_20261002_113600'
OUT=ROOT/'native_t1/runs/jedge5_solver_pilot_20261002_121000'
GPU_LOCK='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'
CASES=[dict(case_id=i,parent=0,patch=p,seed=s) for i,(p,s) in enumerate([(0,10401),(0,10402),(4,10401),(4,10402)])]
METHODS={'E_quad50':dict(nfe=50,intervals=50,grid='t[k]=(k/50)^2'),'Midpoint25':dict(nfe=50,intervals=25,grid='uniform'),'Heun25':dict(nfe=50,intervals=25,grid='uniform'),'Heun50':dict(nfe=100,intervals=50,grid='uniform')}

def main():
    dest=OUT/'registration/protocol.json'
    if dest.exists():raise FileExistsError(dest)
    cases=load_cases();noises=noise_bank(OLD);items=[]
    for item in CASES:
        c=cases[item['parent'],item['patch']]
        oldkey=f"J_P0_patch{item['patch']}_seed{item['seed']}_bf16_Euler50"
        observer=BLACKBOX/'trajectories/J'/(oldkey+'.npz')
        with np.load(observer) as z:expected_v_hash=array_hash(z['velocities'][0]);initial_hash=array_hash(z['states'][0])
        items.append(dict(**item,C_sha256=array_hash(c['C']),noise_sha256=array_hash(noises[item['seed']]),GT_free_sha256=array_hash(c['free']),
            first_velocity_source=str(observer),first_velocity_source_sha256=file_sha256(observer),first_velocity_sha256=expected_v_hash,initial_state_sha256=initial_hash))
    public=[ROOT/'native_t1'/p for p in ['model.py','context_geometry.py','sampling.py','runtime.py','objectives.py','losses/edge.py']]
    protocol=dict(status='FROZEN',registered_timestamp=time.time(),user_scope='Zero training solver verification; at most16 new full trajectories, no guidance/CFG/training arms.',
        model='J',checkpoint=str(ASSETS['J']['path']),checkpoint_file_sha256=file_sha256(ASSETS['J']['path']),model_state_sha256=ASSETS['J']['state'],
        cases=items,methods=METHODS,precision='bf16_forward_FP32_parameters_and_integration',expected_actual_model_forwards=1000,
        max_new_full_trajectories=16,new_optimizer_updates=0,extra_full_replays=0,seed_changes=0,
        C_policy='Known coordinates clamped in every accepted and every stage state; raw known-slot model velocity not integrated.',
        endpoint_policy='Heun evaluates t=1 for final corrector, no endpoint clipping or dropped stage. Midpoint25 latest t=.98. All finite checks fail closed.',
        initial_velocity_gate='Each new rollout first model call at same original noise/C/t0 must equal archived BF16 raw velocity bitwise; no extra call.',
        comparisons=['E_quad50/Midpoint25/Heun25 versus archived BF16 Euler50 at50 NFE','Heun50 versus archived BF16 Euler100 at100 NFE'],
        decision='Report per-case remainder symmetric/bidirectional RMS, Gamma RMS/p95/coverage, FF/CF, area and angle quality together; no adaptive tuning after results, no universal superiority claim from four development cases.',
        protected={str(p.relative_to(ROOT)):file_sha256(p) for p in public},source_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        gpu_dispatch=dict(lock=GPU_LOCK,workers=1,scope='CUDA init before exclusive shared flock is forbidden; one process per case, all4 solvers, release at process exit.',headroom_MiB=2048,conservative_increment_MiB=1536),
        other_windows=dict(A='FP32 RK4/Dopri5 reference, not these BF16 methods',B='FP32 local Euler step doubling and Euler100 suffixes, not these full methods'),
        no_GT_to_sampler=True,postprocessing=False,loss_computation_at_inference=False,baseline_archive=str(OLD))
    write_json(dest,protocol);print('REGISTERED 16 full / 1000 F / 0 updates',flush=True)
if __name__=='__main__':main()
