"""Real CUDA correctness smoke: final12 equals interrupted5 + resumed7.

Runs only training-UID data. It never evaluates or selects a student model.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path('/mnt/e/MeshFlow/.worktrees/a-layout-value-20260924')
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from experiments.a_radial.layout_student.train import train_run, file_sha256, model_sha256

SOURCE = REPO/'output/a_taskaligned_20260924_01'
OUT = REPO/'output/a_taskaligned_cuda_smoke_20260924_01'


def exact_same(a, b, trail='root'):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b), trail
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), trail
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), trail
        for k in a:
            exact_same(a[k], b[k], trail+'.'+str(k))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), trail
        for i, (x, y) in enumerate(zip(a, b)):
            exact_same(x, y, trail+'.'+str(i))
    else:
        assert a == b, trail


def main():
    original_protocol_hash = file_sha256(SOURCE/'RUN_PROTOCOL.json')
    original_labels_hash = file_sha256(SOURCE/'train_labels.npz')
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE/'train_labels.npz', OUT/'train_labels.npz')
    shutil.copy2(Path(__file__), OUT/'GPU_SMOKE_TEST.py')
    protocol = json.loads((SOURCE/'RUN_PROTOCOL.json').read_text())
    protocol['experiment'] = 'A-TASKALIGNED-CUDA-SMOKE-ONLY'
    protocol['scope'] = 'SMOKE_ONLY: actual CUDA correctness, no scientific result or checkpoint selection.'
    protocol['smoke_parent_protocol_sha256'] = original_protocol_hash
    protocol['training']['updates'] = 12
    protocol['training']['checkpoint_selection'] = 'final_update_12_smoke_only_never_deploy'
    protocol['hashes']['config_sha256'] = hashlib.sha256(json.dumps(protocol['training'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (OUT/'RUN_PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    assert file_sha256(OUT/'train_labels.npz') == original_labels_hash
    assert torch.cuda.is_available()
    gradient_records=[]
    orig_step = torch.optim.AdamW.step
    def audited_step(self, *args, **kwargs):
        params=[p for group in self.param_groups for p in group['params']]
        assert all(p.device.type=='cuda' and p.dtype==torch.float32 for p in params)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        gradient_norm=float(torch.sqrt(sum(p.grad.double().square().sum() for p in params)).cpu())
        assert gradient_norm>0
        gradient_records.append({'update_call':len(gradient_records)+1,'parameter_count':sum(p.numel() for p in params),
            'device':str(params[0].device),'dtype':str(params[0].dtype),'gradient_l2':gradient_norm,
            'all_gradients_finite':True,'all_gradients_present':True})
        return orig_step(self,*args,**kwargs)
    torch.optim.AdamW.step = audited_step
    np_load = np.load
    opened=[]
    def whitelist_load(path,*args,**kwargs):
        assert Path(path).resolve()==(OUT/'train_labels.npz').resolve(), str(path)
        opened.append(str(path))
        return np_load(path,*args,**kwargs)
    np.load=whitelist_load
    started=time.perf_counter()
    try:
        whole=train_run(OUT,10,12,device='cuda',checkpoint_every=5,output_dir=OUT/'whole')
        first=train_run(OUT,10,12,device='cuda',checkpoint_every=5,stop_after=5,output_dir=OUT/'first')
        resumed=train_run(OUT,10,12,Path(first['checkpoint']),device='cuda',checkpoint_every=5,output_dir=OUT/'resumed')
    finally:
        torch.optim.AdamW.step=orig_step
        np.load=np_load
    a=torch.load(whole['checkpoint'],map_location='cpu',weights_only=False)
    b=torch.load(resumed['checkpoint'],map_location='cpu',weights_only=False)
    checked=['model','optimizer','scheduler','rng','sampler','config','trace']
    for key in checked:
        exact_same(a[key],b[key],key)
    assert whole['model_sha256']==resumed['model_sha256']==model_sha256(a['model'])
    assert len(gradient_records)==24
    assert len(opened)==3
    assert file_sha256(SOURCE/'RUN_PROTOCOL.json')==original_protocol_hash
    assert file_sha256(SOURCE/'train_labels.npz')==original_labels_hash
    report={'status':'PASS','scope':'SMOKE_ONLY','cuda_device':torch.cuda.get_device_name(0),
        'compute_capability':list(torch.cuda.get_device_capability(0)),
        'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,
        'tested_updates':{'continuous':12,'interrupted':5,'resumed_additional':7,'actual_total':24},
        'exactly_equal_keys':checked,'end_model_sha256':whole['model_sha256'],
        'all_gradients_cuda_float32_finite_nonzero':True,'gradient_records':gradient_records,
        'max_peak_gpu_allocated_bytes':max(r['peak_gpu_allocated_bytes'] for r in (whole,first,resumed)),
        'max_peak_gpu_reserved_bytes':max(r['peak_gpu_reserved_bytes'] for r in (whole,first,resumed)),
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'training_archive_read_count':len(opened),'internal_check_teacher_reads':0,
        'formal_protocol_unchanged':True,'formal_train_archive_unchanged':True,
        'formal_protocol_sha256':original_protocol_hash,'training_labels_sha256':original_labels_hash,
        'smoke_source_sha256':file_sha256(OUT/'GPU_SMOKE_TEST.py'),
        'train_source_sha256':file_sha256(REPO/'experiments/a_radial/layout_student/train.py'),
        'elapsed_wall_seconds':time.perf_counter()-started,
        'runs':{'whole':whole,'first':first,'resumed':resumed}}
    (OUT/'GPU_SMOKE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('gradient_records','runs')},indent=2),flush=True)


if __name__=='__main__':
    main()
