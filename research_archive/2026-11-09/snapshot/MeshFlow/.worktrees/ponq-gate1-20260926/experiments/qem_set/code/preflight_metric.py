"""Historical B0 provenance and one archived-noise rollout replay, no extraction."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,platform,sys,time,subprocess
import numpy as np
import scipy
import torch
from common import ROOT,dump,sha
from fm_model import total_velocity
from interaction_model import InteractionVelocity
from single_state_common import arrays

def read(p):return json.loads(p.read_text())
def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    checks=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    p=read(ROOT/'RUN_PROTOCOL.json');previous=__import__('pathlib').Path(p['previous_run']);parent_run=__import__('pathlib').Path(p['parent_run'])
    out=ROOT/'preflight';out.mkdir(exist_ok=False)
    env=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
        torch=str(torch.__version__),torch_cuda=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(),device_capability=list(torch.cuda.get_device_capability()),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,threads=torch.get_num_threads())
    try:env['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    except Exception as exc:env['nvidia_smi_error']=repr(exc)
    baseline=torch.load(ROOT/'control/final.pt',map_location='cpu',weights_only=False)
    parent=torch.load(ROOT/'parent/final.pt',map_location='cpu',weights_only=False)
    data=arrays(ROOT/'data.npz')
    check('control_final_9000',baseline['completed_steps']==9000 and baseline['new_updates']==6000 and baseline['status']=='COMPLETED')
    check('parent_3000',parent['completed_steps']==3000 and parent['status']=='COMPLETED')
    check('control_torch_version',str(baseline['protocol']['torch_version'])==str(torch.__version__))
    check('parent_torch_version',str(parent['protocol']['torch_version'])==str(torch.__version__))
    check('parent_numpy_version',parent['protocol']['numpy_version']==np.__version__)
    check('previous_python_version',read(previous/'RUN_PROTOCOL.json')['environment']['python']==sys.version)
    check('copied_data_exact',sha(ROOT/'data.npz')==sha(previous/'data.npz')==parent['protocol']['data_sha256']==baseline['protocol']['data_sha256'])
    check('baseline_checkpoint_exact',sha(ROOT/'control/final.pt')==sha(previous/'training/B0/final.pt'))
    check('parent_checkpoint_exact',sha(ROOT/'parent/final.pt')==sha(parent_run/'training/direct/final.pt'))
    check('baseline_training_stream_exact',sha(ROOT/'control/TRAIN_TRACE.jsonl')==sha(previous/'training/B0/TRAIN_TRACE.jsonl'))
    unchanged=['fm_model.py','fm_state.py','interaction_model.py','train_fm_pilot.py','train_interaction.py','single_state_common.py',
        'interaction_semantics.py','sample_single_state.py','common.py','safe_extract.py','safe_extract_domain.py','exact_audit.py']
    for name in unchanged:check('source_exact:'+name,sha(ROOT/'code'/name)==sha(previous/'code'/name))
    model=InteractionVelocity(arm='B0',width=128).cuda();model.load_state_dict(baseline['model_state_dict']);model.eval()
    check('parameter_count',sum(x.numel() for x in model.parameters())==88214)
    mean=torch.tensor(data['mean'],dtype=torch.float32,device='cuda');scale=torch.tensor(data['scale'],dtype=torch.float32,device='cuda')
    noise=arrays(ROOT/'control/archived_sample_noise.npz')['noise'][0:1]
    reference=arrays(ROOT/'control/archived_endpoints.npz')['states'][0]
    z=torch.tensor(noise,device='cuda');torch.cuda.synchronize();begin=time.perf_counter()
    with torch.no_grad():
        for step in range(64):
            velocity,_=total_velocity(model,z,torch.full((1,),step/64,device='cuda'),mean,scale,'direct');z=z+velocity/64
    torch.cuda.synchronize();duration=time.perf_counter()-begin
    actual=(z*scale+mean)[0].cpu().numpy().astype(np.float64)
    difference=float(np.max(abs(actual-reference)))
    check('archived_B0_noise0_rollout_replay',np.isfinite(actual).all() and difference<=5e-6,dict(max_abs=difference,tolerance=5e-6,byteidentical=bool(np.array_equal(actual,reference))))
    np.savez_compressed(out/'B0_ARCHIVED_REPLAY.npz',actual=actual,reference=reference)
    dump(out/'ENVIRONMENT.json',env)
    result=dict(passed=all(r['passed'] for r in checks),check_count=len(checks),checks=checks,
        replay_nfe=64,replay_count=1,replay_seconds=duration,new_free_evaluation_count=0,geometry_extractions=0,
        environment_comparability='Same current venv and matching recorded Python/Torch/NumPy/core-source/data/checkpoints; archived B0 endpoint replay verified. Historical full driver/CUDA-runtime/scipy/cuDNN versions were not all recorded, so full historical-runtime identity is NOT asserted.',
        cost_note='Historical B0 training time is context only, not a contemporaneous speed comparison; both fresh samplers measured now.')
    dump(out/'RESULT.json',result);print('PREFLIGHT_METRIC',result['passed'],difference,flush=True)
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()