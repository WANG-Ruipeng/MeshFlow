"""CPU-only identity checks and immutable registration for 120 equal-NFE requests."""
from native_t1 import runtime as _runtime
from pathlib import Path
import argparse, csv, hashlib, json, os, platform, re, shutil, subprocess, time
import numpy as np
import torch
from native_t1.sampling import make_noise
from experiments.mf_h1_local0.jedge5_diagnosis.common import (
    ROOT, ASSETS, EVALUATION, EVALUATION_SHA, TRAIN, TRAIN_SHA,
    load_model, load_cases, array_hash, file_sha256, write_json, read_json)

HISTORICAL = ROOT/'native_t1/runs/jedge5_diagnosis_20261002_102215'
PILOT = ROOT/'native_t1/runs/jedge5_solver_pilot_20261002_121000'
SOLVERS = ['Euler50','Midpoint25','E_quad50']
FILE_SHA = {'B':'0f22ce498ca2c27dc74b43890fb8aad20f0f3e7153aa74ee6d3c14ea445a2660',
            'J':'cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9'}

def resource_snapshot():
    def cmd(args):
        p=subprocess.run(args,capture_output=True,text=True)
        return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
    return dict(timestamp=time.time(),gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        cpu_cores=os.cpu_count(),cpu_affinity=len(os.sched_getaffinity(0)),load=os.getloadavg(),
        memory={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:','SwapFree:'))},
        cpu_processes=cmd(['ps','-eo','pid,comm,pcpu,pmem,args','--sort=-pcpu']),
        disk=cmd(['df','-B1',str(ROOT)]))

def seed_audit(out):
    roots=[ROOT/'native_t1/runs',ROOT/'native_t1/maintenance',ROOT/'experiments/mf_h1_local0/outputs',ROOT/'output']
    used={};files=documents=csvs=0
    pattern=re.compile(r'"([^"\r\n]*seed[^"\r\n]*)"\s*:\s*(\[[^\]]{0,8192}\]|-?\d+)',re.I)
    def add(seed,path,field):
        if type(seed) is int and seed>=10501:
            used.setdefault(seed,[]).append(dict(path=str(path.relative_to(ROOT)),field=field))
    for base in roots:
        if not base.exists():continue
        for p in base.rglob('*'):
            if not p.is_file() or out in p.parents:continue
            files+=1
            for m in re.finditer(r'(?:seed|noise)[_-]?(\d+)',p.name,re.I):add(int(m.group(1)),p,'filename')
            if p.suffix.lower() in ('.json','.jsonl'):
                documents+=1
                content=p.read_text(encoding='utf-8-sig',errors='replace')
                for m in pattern.finditer(content):
                    try:v=json.loads(m.group(2))
                    except ValueError:continue
                    for s in v if isinstance(v,list) else [v]:add(s,p,m.group(1))
            elif p.suffix.lower()=='.csv':
                csvs+=1
                try:
                    with p.open(encoding='utf-8-sig',errors='replace',newline='') as f:
                        reader=csv.DictReader(f)
                        fields=[k for k in (reader.fieldnames or []) if k and 'seed' in k.lower()]
                        if fields:
                            for row in reader:
                                for k in fields:
                                    if re.fullmatch(r'\d+',str(row.get(k,''))):add(int(row[k]),p,k)
                except (csv.Error,ValueError):continue
    trials=[];first=10501
    while True:
        pair=[first,first+1];hits=[dict(seed=s,**h) for s in pair for h in used.get(s,[])]
        trials.append(dict(seeds=pair,hits=hits,unused=not hits))
        if not hits:break
        first+=2
    return dict(seeds=pair,trials=trials,files_scanned=files,json_documents_scanned=documents,csv_documents_scanned=csvs,
        roots=[str(p) for p in roots],scope='Semantic JSON seed fields, CSV seed columns and seed/noise filenames in retained local records. Does not certify unavailable/deleted/unrecorded history.',
        exclusion=str(out))

def task(kind,model,solver,parent,patch,seed,**more):
    return dict(task_id=f'{kind}_{model}_{solver}_P{parent}_patch{patch}_seed{seed}',kind=kind,
        model=model,solver=solver,parent=parent,patch=patch,seed=seed,**more)

def protection_manifest(out,refs):
    # Existing files are protected; newly created files in other authorized runs
    # are allowed. Hash model/data/public code and selected history; metadata all old runs.
    files=set(p for p in (ROOT/'native_t1').rglob('*.py') if 'runs' not in p.parts and '__pycache__' not in p.parts)
    files.update((ROOT/'models').rglob('*.py'))
    files.update((ROOT/'native_t1/checkpoints').rglob('*.pt'))
    files.update([EVALUATION,TRAIN,ROOT/'AGENTS.md',ROOT/'WORKSPACE.md'])
    files.update(Path(p) for p in refs)
    files.update((ROOT/'experiments/mf_h1_local0/jedge5_solver_pilot').glob('*.py'))
    files.update((ROOT/'experiments/mf_h1_local0/jedge5_diagnosis').glob('*.py'))
    for p in (ROOT/'experiments/mf_h1_local0').glob('*.py'):files.add(p)
    hashed={str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(files) if p.is_file()}
    metadata={}
    for base in ('native_t1/runs','native_t1/maintenance','datasets','downloaded_data','assets'):
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and out not in p.parents:
                st=p.stat();metadata[str(p.relative_to(ROOT))]=[st.st_size,st.st_mtime_ns]
    return dict(hashes=hashed,prior_file_metadata=metadata,policy='All hashed files must remain identical; missing/modified old files reported individually, additions outside this run allowed.')

def run(out):
    out=Path(out).resolve()
    if out.parent!=ROOT/'native_t1/runs' or not out.name.startswith('equal_nfe_confirmation_'):raise ValueError('Isolated run required')
    if out.exists():raise FileExistsError(out)
    (out/'registration').mkdir(parents=True);started=time.perf_counter()
    snapshot=resource_snapshot();write_json(out/'registration/resources_before.json',snapshot)
    torch.set_num_threads(1);_runtime.configure_stable_runtime()
    identities={}
    for name in ('B','J'):
        if file_sha256(ASSETS[name]['path'])!=FILE_SHA[name]:raise ValueError('Checkpoint file mismatch '+name)
        model,identity=load_model(name,device='cpu')
        identities[name]=dict(identity,parameter_count=sum(p.numel() for p in model.parameters()),
            parameter_schema_sha256=hashlib.sha256(json.dumps([(n,list(p.shape),str(p.dtype)) for n,p in model.named_parameters()]).encode()).hexdigest())
        if identity['cumulative_updates']!=3000:raise ValueError('Requires raw cumulative3000')
        del model
    if identities['B']['parameter_schema_sha256']!=identities['J']['parameter_schema_sha256']:raise ValueError('Model architectures differ')
    cases=load_cases()
    if len(cases)!=10 or file_sha256(TRAIN)!=TRAIN_SHA:raise ValueError('Input identity differs')
    audit=seed_audit(out);seeds=audit['seeds'];write_json(out/'registration/seed_audit.json',audit)
    noises={f'seed_{s}':make_noise(s).numpy() for s in seeds}
    np.savez_compressed(out/'registration/noises.npz',**noises)
    free={k:v[:,12:].copy() for k,v in noises.items()}
    np.savez_compressed(out/'registration/free_noises.npz',**free)
    inputs=[]
    for (p,j),c in sorted(cases.items()):
        if c['C'].reshape(12,9).tobytes()!=c['full'].reshape(112,9)[c['source_face_ids']].tobytes():raise ValueError('Known face slots/source mismatch')
        if c['free'].reshape(100,9).tobytes()!=c['full'].reshape(112,9)[c['free_source_ids']].tobytes():raise ValueError('Free complement mismatch')
        inputs.append(dict(parent=p,patch=j,condition_class='finetuning_C' if j==0 else 'other_previously_observed_C',
            known_slots=list(range(12)),free_slots=list(range(12,112)),valid_slots=list(range(112)),
            source_face_ids=c['source_face_ids'].tolist(),free_source_ids=c['free_source_ids'].tolist(),
            arrays={k:array_hash(c[k]) for k in ('C','full','free','boundary_points','boundary_weights','coverage_points','pairwise_target_points')},bbox=c['bbox']))
    write_json(out/'registration/input_identity.json',dict(cases=inputs,noise_protocol='native_t1.sampling.make_noise(seed): CPU torch.Generator, randn FP32 [1,112,9]; checked free slots12:112 saved as [1,100,9].',
        noises={str(s):dict(full_sha256=array_hash(noises[f'seed_{s}']),free_sha256=array_hash(free[f'seed_{s}'])) for s in seeds}))
    regression=[]
    for solver in SOLVERS:
        ref=HISTORICAL/'A/J/J_P0_patch4_seed10401_bf16_Euler50.npz' if solver=='Euler50' else PILOT/f'raw/case2/{solver}.npz'
        regression.append(task('regression','J',solver,0,4,10401,noise_source='historical',reference_npz=str(ref),reference_sha256=file_sha256(ref)))
    ref=HISTORICAL/'A/B/B_P0_patch0_seed10401_bf16_Euler50.npz'
    regression.append(task('regression','B','Euler50',0,0,10401,noise_source='historical',reference_npz=str(ref),reference_sha256=file_sha256(ref)))
    groups=[dict(group_id='regression',tasks=regression)]
    rng=np.random.RandomState(2026100215);configs=[(m,s) for m in ('B','J') for s in SOLVERS]
    for p in (0,1):
        for j in range(5):
            for seed in seeds:
                tasks=[task('formal',*configs[i],p,j,seed,noise_source='new') for i in rng.permutation(6)]
                groups.append(dict(group_id=f'block{len(groups)-1:02d}',parent=p,patch=j,seed=seed,tasks=tasks))
    repeats=[]
    for m,s in (('B','Euler50'),('J','Midpoint25')):
        original=task('formal',m,s,0,0,seeds[0])['task_id']
        repeats.append(task('repeat',m,s,0,0,seeds[0],noise_source='new',repeat_of=original))
    groups.append(dict(group_id='repeats',tasks=repeats))
    cache=HISTORICAL/'registration/static_context_audit.json'
    shutil.copyfile(cache,out/'registration/static_context_audit.json')
    from experiments.mf_h1_local0.jedge5_diagnosis.scoring import banks_with_context_audit,REFERENCE
    banks=banks_with_context_audit(out)
    if set(banks)!=set(range(5)):raise ValueError('Must score all5 patches')
    refs=[t['reference_npz'] for t in regression]+[cache,REFERENCE/'evaluation.py',REFERENCE/'edge_metric.py']
    protect=protection_manifest(out,refs);write_json(out/'registration/protection_before.json',protect)
    source_dir=Path(__file__).parent
    sources={str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(source_dir.glob('*.py'))}
    protocol=dict(schema='equal_nfe_confirmation_v1',status='FROZEN',created_timestamp=time.time(),
        HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        models=['B','J'],solvers=SOLVERS,primary_candidate='Midpoint25',secondary_candidate='E_quad50',
        seeds=seeds,generation_seeds=seeds,groups=groups,historical_run=str(HISTORICAL),
        budget=dict(formal_requests=120,formal_forwards=6000,regression_requests=4,regression_forwards=200,repeat_requests=2,repeat_forwards=100,warmup_forwards_max=20,total_forward_attempts_max=6320,planned_forwards=6300,planned_warmup_forwards=0),
        input_archive=dict(path=str(EVALUATION),sha256=EVALUATION_SHA),checkpoint_identities=identities,
        checkpoint_files={m:dict(path=str(ASSETS[m]['path']),sha256=FILE_SHA[m],state_sha256=ASSETS[m]['state'],raw=True) for m in ('B','J')},
        noise_archive_sha256=file_sha256(out/'registration/noises.npz'),free_noise_archive_sha256=file_sha256(out/'registration/free_noises.npz'),
        source_sha256=sources,metric_sources={str(p.relative_to(ROOT)):file_sha256(p) for p in refs[-2:]},
        input_identity_sha256=file_sha256(out/'registration/input_identity.json'),known_slots=list(range(12)),free_slots=list(range(12,112)),
        schedule_rng=dict(engine='numpy.RandomState',seed=2026100215,scope='six configurations independently permuted within each parent/patch/seed block; generation RNG separate'),
        numeric=dict(Euler50='Production k/50 t; constant Python1/50; FP32 state + FP32 velocity * dt',
            Midpoint25='Historical Python FP64 ideal k/25 grid; dt=t1-t0; midpoint=(t0+t1)*.5; independent FP32 casts; update from original state',
            E_quad50='Historical Python FP64 (k/50)^2 grid; dt=t1-t0 before FP32 cast; front dense back sparse',
            runtime='CUBLAS :4096:8; deterministic warn_only=False; TF32 off; SDPA math; BF16 forwards, FP32 params/states/integration; eval/no_grad'),
        dispatch=dict(gpu_lock='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock',maximum_GPU_workers=1,release='process exit after each group',
            headroom='max(2048MiB,10%total)',CPU_threads_per_worker=1,CPU_workers='chosen from current available CPU/RAM and load before dispatch',
            initial_GPU_peak='Observe authorized regression group; no probe forward; process VRAM plus allocated/reserved independently'),
        grouping=dict(parents=[0,1],training_C_patches=[0],other_previously_observed_C_patches=[1,2,3,4],both_parents_seen_in_training=True,patch3_historical_overlap='11/12',patch4='lower historical overlap'),
        statistics=dict(paired_before_summary=True,macro='mean two seeds per parent/patch then equal ten units',
            missing='Difference requires all operands finite; cell mean requires both seeds; full macro requires all requested cells. Incomplete observed summaries separately labeled with valid/requested counts; never zero fill.',
            sign='lower errors/counts better, higher coverage better, area descriptive only',success='No weighted score or new numerical success threshold; primary candidate interpretation based on all fixed metrics and strata'),
        binding=dict(groups=60,scoring_cells=240,new_generation=0,row_target_bbox=True,free_only=True,save_both_directions=True,numerical_rule='inherit original cross_binding, raw differences retained'),
        plots=dict(elev=22,azim=-55,vertical_axis='y',bbox='each parent original GT bounding cube center; radius=.60*max_axis_span',
            coverage_color=dict(units='distance percent bbox',vmin=0,vmax=10,cmap='viridis',clip=True),seed='first_registered'),
        failure_contract='No automatic retries, extra seeds, replacement outputs or unregistered forward calls. Stop dispatch after numerical/OOM/identity/regression failure; preserve attempted NFE and all120 requested rows.',
        baseline_inheritance='Inherited BF16 face-permutation FAIL remains FAIL at its original tolerance; not retested or repaired.',
        topology='Existing metrics retained; global watertight/exact-predicate certification NOT_RUN',
        no_training=True,no_optimizer=True,no_EMA=True,no_default_change=True)
    write_json(out/'registration/protocol.json',protocol)
    (out/'registration/protocol.sha256').write_text(file_sha256(out/'registration/protocol.json')+'\n')
    write_json(out/'registration/environment.json',dict(python=platform.python_version(),executable=os.sys.executable,torch=torch.__version__,CUDA=torch.version.cuda,numpy=np.__version__,CUDA_initialized=torch.cuda.is_initialized()))
    if torch.cuda.is_initialized():raise RuntimeError('Preflight initialized CUDA')
    result=dict(status='PASS',model_forwards=0,strict_CPU_checkpoint_loads=2,parameter_count=identities['B']['parameter_count'],cases=len(cases),seeds=seeds,seconds=time.perf_counter()-started)
    write_json(out/'registration/preflight.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
