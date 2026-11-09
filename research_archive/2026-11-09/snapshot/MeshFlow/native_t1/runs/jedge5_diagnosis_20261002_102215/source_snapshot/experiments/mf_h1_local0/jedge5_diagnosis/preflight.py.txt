"""Read-only asset/seed checks and one-time registration; no neural forwards."""
from .common import *
import argparse
import re
import time
import platform
import subprocess
import mmap


def historical_seed_check(out,seeds):
    roots=[ROOT/'native_t1/runs',ROOT/'experiments/mf_h1_local0/outputs',ROOT/'output']
    field=re.compile(rb'"[^"\r\n]*seed[^"\r\n]*"\s*:\s*(?:\[[^\]]{0,4096}\]|-?\d+)',re.I)
    hits=[];files=0;name_checks=0
    for base in roots:
        if not base.exists():continue
        for p in base.rglob('*'):
            if not p.is_file() or Path(out) in p.parents:continue
            name_checks+=1
            for seed in seeds:
                if re.search(r'(?:seed|noise)[_-]?'+str(seed)+r'(?:\D|$)',p.name,re.I):hits.append(dict(path=str(p.relative_to(ROOT)),seed=seed,source='filename'))
            if p.suffix.lower() not in ('.json','.jsonl') or p.stat().st_size==0:continue
            files+=1
            with p.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
                for match in field.finditer(data):
                    for seed in seeds:
                        if re.search(rb'(?<!\d)'+str(seed).encode()+rb'(?!\d)',match.group()):
                            hits.append(dict(path=str(p.relative_to(ROOT)),seed=seed,source='seed_field',field=match.group().decode('utf-8',errors='replace')[:500]))
    return dict(seeds=seeds,files_scanned=files,filenames_scanned=name_checks,hits=hits,unused=not hits)


def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError('Registration requires new run directory')
    (out/'registration').mkdir(parents=True)
    tick=time.perf_counter();records=[]
    for label,spec in ASSETS.items():
        model,audit=load_model(label,device='cpu')
        records.append(dict(label=label,**audit));del model
    cases=load_cases()
    if file_sha256(TRAIN)!=TRAIN_SHA:raise ValueError('Training data hash mismatch')
    trials=[]
    for first in range(10401,10501,10):
        audit=historical_seed_check(out,[first,first+1]);trials.append(audit)
        if audit['unused']:break
    else:raise RuntimeError('No registered unused pair')
    seeds=trials[-1]['seeds'];write_json(out/'registration/seed_audit.json',trials)
    from native_t1.sampling import make_noise
    noises={f'seed_{s}':make_noise(s).numpy() for s in seeds}
    np.savez_compressed(out/'registration/noises.npz',**noises)
    task=Path('/mnt/c/Users/warpwang/.codex/attachments/a453f759-e4e6-4888-9a5d-50a2b6a7132a/Pasted text.txt')
    protocol=dict(schema='jedge5_diagnosis_v1',status='FROZEN',generation_seeds=seeds,seeds=seeds,
        teacher_seed=20261002,teacher_times=[.1,.3,.5,.75,.9],parameter_probe_precision='BF16',
        case_order=[dict(parent=p,patch=j,seed=s) for p in (0,1) for j in range(5) for s in seeds],
        primary_order=['A','B','C','D'],models=['B','J'],N=112,K=12,free=100,
        A=dict(configurations=[['BF16',50],['BF16',100],['FP32',50],['FP32',100]],complete=168,reference_patches=[0,4],reference_seed=seeds[0],reference_nfe=200,forward_budget=13600),
        B=dict(patches=[0,2,4],times=[.2,.5,.8],indices=[20,50,80],precision='FP32',nfe=100,cross_suffixes=72,same_model_suffix_checks=6,forward_budget=3900),
        C=dict(teacher_states=100,teacher_forwards=400,parameter_probe_forwards=8,velocity_VJPs=1200,parameter_VJPs=24),
        D=dict(arms=['R_B','R_J'],updates_per_arm=500,cumulative_start=2500,cumulative_end=3000,half_life=100,beta=2**(-1/100),initialization_weight_after500=1/32,complete=60,teacher_curve_forwards=140),
        lambda_edge=4.071385484299878,lambda_surface=.3266104383520167,gate=.5,ramp_updates=50,
        extra_full_replays_max=2,planned_extra_full_replays=1,extra_replay_reason='B first case BF16 Euler50 production-sampler bitwise parity',
        complete_total_planned=229,complete_total_ceiling=230,parameter_count_changes=0,
        sampler='direct FP32 t=k/NFE, FP32 v*dt Euler; C untouched',
        original_sampler_unchanged=True,gt_OT_generation=False,pair_or_stage2=False,
        no_hyperparameter_selection=True,task_sha256=file_sha256(task),
        HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        inputs=dict(training=dict(path=str(TRAIN),sha256=TRAIN_SHA),evaluation=dict(path=str(EVALUATION),sha256=EVALUATION_SHA)),
        checkpoint_identities=records,noise_sha256=file_sha256(out/'registration/noises.npz'))
    write_json(out/'registration/protocol.json',protocol)
    environment=dict(python=platform.python_version(),torch=torch.__version__,CUDA=torch.version.cuda,
        numpy=np.__version__,cpu_count=os.cpu_count(),cpu_memory=Path('/proc/meminfo').read_text().splitlines()[:3],
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv'],text=True),
        disk=subprocess.check_output(['df','-B1','/mnt/e'],text=True),
        CUDA_initialized=torch.cuda.is_initialized(),AGENTS_sha256=file_sha256(ROOT/'AGENTS.md'))
    write_json(out/'registration/environment.json',environment)
    result=dict(status='PASS',strict_CPU_checkpoint_loads=3,model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,
        cases=len(cases),generation_seeds=seeds,seconds=time.perf_counter()-tick)
    write_json(out/'registration/preflight.json',result)
    print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
