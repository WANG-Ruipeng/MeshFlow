"""CPU-only window A preregistration. No CUDA initialization, Gaussian draw, or OT."""
from .assets import *
import argparse
import importlib.metadata
import importlib.util
import platform
import subprocess
import time


def _bits(a,b):
    a=np.ascontiguousarray(a); b=np.ascontiguousarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def _run_command(args):
    try:
        result=subprocess.run(args,capture_output=True,text=True,check=False)
        return dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    except OSError as exc:
        return dict(error=str(exc))


def run(out):
    out=Path(out).resolve()
    if out.parent != ROOT/'native_t1/runs' or not out.name.startswith('jedge5_numeric_reference_A_'):
        raise ValueError('Window A may only write its own named run directory')
    registration=out/'registration'
    registration.mkdir(parents=True,exist_ok=False)
    tick=time.perf_counter(); protected={}
    def remember(path,kind):
        path=Path(path).resolve()
        record=dict(path=str(path),sha256=file_sha256(path),bytes=path.stat().st_size,kind=kind)
        protected[str(path)]=record
        return record
    def read(path,kind='read_metadata'):
        remember(path,kind)
        return read_json(path)
    before_cuda=torch.cuda.is_initialized()
    if before_cuda: raise RuntimeError('CPU preflight must run before any CUDA initialization')
    start=dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
               git_status_note='Informational only; unrelated window files are not protected assets.')
    write_json(registration/'start_state.json',start)
    archive,discovery=discover_archive()
    old_protocol=read(archive/'registration/protocol.json')
    remember(archive/'JEDGE5_DIAGNOSIS_REPORT.md','archive_identity')
    noise_path=archive/'registration/noises.npz'
    if remember(noise_path,'Gaussian_archive')['sha256']!=old_protocol['noise_sha256']:
        raise ValueError('Original Gaussian archive checksum mismatch')
    with np.load(noise_path,allow_pickle=False) as z:
        noises={int(k.removeprefix('seed_')):z[k].copy() for k in z.files}
    if set(noises)!={10401,10402}:
        raise ValueError('Original Gaussian bank differs from registered seeds')
    for noise in noises.values():
        if noise.shape!=(1,112,9) or noise.dtype!=np.float32 or not np.isfinite(noise).all():
            raise ValueError('Missing original FP32 Gaussian values')
    if remember(EVALUATION,'evaluation_inputs')['sha256']!=EVALUATION_SHA:
        raise ValueError('Evaluation input hash mismatch')
    # This loads and validates original source/complement, C, boundary and scoring arrays;
    # it does not run optimal transport or draw model Gaussian noise.
    from experiments.mf_h1_local0.jedge5_diagnosis.common import load_cases
    all_cases=load_cases()
    old_rows={};old_scores={}
    for label,spec in ASSETS.items():
        replay=read(archive/'stage_d'/('R_'+label)/'replay.json')
        if replay['raw500_state_sha256']!=spec['state'] or not replay['raw500_bitwise_reference_equal']:
            raise ValueError('Raw replay does not recover prescribed model')
        matrix=read(archive/'A'/label/'results.json')
        old_rows[label]={(r['parent'],r['patch'],r['seed'],r['nfe']):r for r in matrix['rows'] if r['precision']=='fp32'}
        score_path=archive/'scores'/('A_'+label)/'results.json'
        if score_path.is_file():
            score=read(score_path)
            old_scores[label]={(r['parent'],r['patch'],r['seed'],r['nfe']):r for r in score['rows'] if r['precision']=='fp32'}
        else: old_scores[label]={}
    registered={};case_records=[];euler_rows=[]
    for p,j,seed in CASE_SPECS:
        ident=case_id(p,j,seed);case=all_cases[p,j]
        C=np.asarray(case['C'],dtype=np.float32).copy();noise=noises[seed]
        source_evidence=[]
        for label in ASSETS:
            for nfe in (50,100,200):
                row=old_rows[label].get((p,j,seed,nfe))
                desc=dict(case_id=ident,model=label,parent=p,patch=j,seed=seed,precision='fp32',nfe=nfe)
                if row is None:
                    euler_rows.append(dict(desc,status='NOT_AVAILABLE',reason='Not in original Euler matrix; no additional Euler run authorized'))
                    continue
                raw_path=archive/row['path']
                if not raw_path.exists():
                    euler_rows.append(dict(desc,status='NOT_AVAILABLE',reason='Original raw archive missing'))
                    continue
                if remember(raw_path,'old_Euler_raw')['sha256']!=row['sha256']:
                    raise ValueError('Old Euler raw checksum mismatch')
                a=row['audit']
                expected_known=list(range(12));expected_free=list(range(12,112))
                if (a['known_slots']!=expected_known or a['free_slots']!=expected_free
                    or a['total_faces']!=112 or a['valid_faces']!=112
                    or a['integration_dtype']!='torch.float32' or a['precision']!='fp32'
                    or not a['known_all_states_bitwise'] or not a['full_from_noise']):
                    raise ValueError('Archived slot/precision/C contract differs')
                if a['noise_sha256']!=array_hash(torch.from_numpy(noise)):
                    raise ValueError('Actual Gaussian disagrees with archived forward input hash')
                if a['condition_sha256']!=array_hash(torch.from_numpy(C.reshape(12,9))):
                    raise ValueError('Actual C disagrees with archived input hash')
                with np.load(raw_path,allow_pickle=False) as z:
                    raw=z['raw'].copy();states=z['states'].copy();indices=z['state_indices'].copy();archived_C=z['C'].copy()
                if not _bits(C,archived_C.reshape(12,3,3)):
                    raise ValueError('Euler archive C values differ')
                if (raw.dtype!=np.float32 or raw.shape!=(112,9) or states.dtype!=np.float32
                    or states.shape[1:]!=(112,9) or indices[0]!=0 or indices[-1]!=nfe
                    or not _bits(states[-1],raw) or not _bits(states[0,12:],noise[0,12:])):
                    raise ValueError('Archived Euler initial/free/final values differ')
                if not _bits(states[:,:12],np.broadcast_to(C.reshape(1,12,9),(len(states),12,9))):
                    raise ValueError('C not bitwise fixed in archived trajectory')
                score=old_scores[label].get((p,j,seed,nfe))
                score_record=dict(status='NOT_AVAILABLE',reason='Old score absent; raw comparison remains available')
                if score is not None:
                    if score['sha256']!=row['sha256'] or score['path']!=row['path']:
                        raise ValueError('Old score does not belong to this raw trajectory')
                    score_file=archive/score['score_path']
                    if score_file.is_file():
                        rec=remember(score_file,'old_Euler_score')
                        score_record=dict(status='AVAILABLE',path=str(score_file),sha256=rec['sha256'],scalars=score['scalars'])
                record=dict(desc,status='AVAILABLE',path=str(raw_path),sha256=row['sha256'],score=score_record,
                            state_indices=indices.tolist(),raw_array_sha256=array_hash(raw),
                            initial_array_sha256=array_hash(states[0]),known_slots=expected_known,free_slots=expected_free)
                euler_rows.append(record);source_evidence.append(dict(model=label,nfe=nfe,path=str(raw_path),sha256=row['sha256']))
        # At least one true old initial state is mandatory for each model/case.
        if set(r['model'] for r in source_evidence)!={'B','J'}:
            raise FileNotFoundError('Missing original Gaussian/C/state evidence for '+ident)
        known=np.zeros((1,112),dtype=np.bool_);known[:,:12]=True
        valid=np.ones_like(known);y=np.array([112],dtype=np.int64)
        initial=noise.copy();initial[:,:12]=C.reshape(1,12,9)
        arrays=dict(noise=noise.copy(),C=C,known=known,valid=valid,y=y,
                    known_slots=np.arange(12,dtype=np.int64),free_slots=np.arange(12,112,dtype=np.int64),
                    initial_full=initial,initial_free=initial[:,12:].copy())
        for name,a in arrays.items():registered[ident+'__'+name]=a
        case_records.append(dict(id=ident,parent=p,patch=j,seed=seed,N=112,K=12,free_faces=100,bbox=case['bbox'],
            arrays_source='Continuous arrays copied from verified original Gaussian/Euler/C archives',
            mask_y_provenance='Exact reconstruction from archived known/free slot lists, valid/total face counts, and unchanged original sampler y=[112]; mask/y were not serialized as separate raw tensors',
            source_evidence=source_evidence,array_hashes={k:array_hash(v) for k,v in arrays.items()}))
    with (registration/'inputs.npz').open('xb') as stream:
        np.savez_compressed(stream,**registered)
    identities=[]
    for label,spec in ASSETS.items():
        remember(spec['path'],'frozen_raw_checkpoint')
        model,audit=load_model(label,device='cpu')
        identities.append(dict(label=label,**audit));del model
    try:
        td_spec=importlib.util.find_spec('torchdiffeq')
        td=dict(available=td_spec is not None,version=importlib.metadata.version('torchdiffeq') if td_spec else None,
                origin=td_spec.origin if td_spec else None)
    except importlib.metadata.PackageNotFoundError:
        td=dict(available=True,version='UNKNOWN',origin=td_spec.origin)
    from .solvers import dependency_info
    verified_dependency=dependency_info()
    fixture_path=Path(__file__).resolve().parent/'tests/cpu_fixture_final.json'
    fixture=read_json(fixture_path)
    if (fixture.get('status')!='PASS' or fixture.get('actual_neural_model_forwards')!=0
        or fixture.get('cuda_initialized') is not False
        or fixture.get('dependency')!=verified_dependency):
        raise ValueError('Existing CPU fixture does not match current solver dependency')
    (registration/'cpu_fixture.json').write_bytes(fixture_path.read_bytes())
    write_json(registration/'dependency_info.json',verified_dependency)
    for path,expected in verified_dependency.get('sources_sha256',{}).items():
        if remember(Path(path),'verified_solver_dependency')['sha256']!=expected:
            raise ValueError('Solver dependency changed after fixture')
    for row in actually_imported_sources():protected[row['path']]=row
    # The original sampler is read as part of the exact mask/face-condition contract.
    for relative in ('experiments/mf_h1_local0/jedge5_diagnosis/sampling.py',
                     'experiments/mf_h1_local0/jedge5_diagnosis/stages_ab.py',
                     'configs/snet/base-120m-ot-v-chair.yaml'):
        remember(ROOT/relative,'read_source_contract')
    task=Path('/mnt/c/Users/warpwang/.codex/attachments/b09d5c04-eff9-45b7-b5cd-a099a4cad5ce/Pasted text.txt')
    task_sha=file_sha256(task)
    (registration/'request.txt').write_bytes(task.read_bytes())
    protocol=dict(schema='jedge5_numeric_reference_A_v1',status='FROZEN',window='A',source_archive=str(archive),
        task_sha256=task_sha,generation_seeds=[10401,10402],selection='purposefully selected observed diagnostic cases, not a new confirmation set',
        cases=case_records,models=identities,registered_inputs_sha256=file_sha256(registration/'inputs.npz'),
        evaluation_path=str(EVALUATION),evaluation_sha256=EVALUATION_SHA,
        ODE_state='only FP32 free coordinates [1,100,9]',model_autocast=False,parameter_dtype='FP32',velocity_dtype='FP32',
        N=112,K=12,free=100,known_time=1,free_time='actual controller time converted to FP32',
        GPU_lock='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock',GPU_workers_per_window=1,
        R1=dict(method='classical_RK4',steps=100,dt=.01,NFE=400),
        R2=dict(method='classical_RK4',steps=200,dt=.005,NFE=800),
        D1=dict(method='Dormand_Prince_5_4',rtol=1e-4,atol=1e-6,first_step=.01,maximum_NFE=1200),
        D2=dict(method='Dormand_Prince_5_4',rtol=1e-5,atol=1e-7,first_step=.01,maximum_NFE=1200),
        budgets=dict(formal_trajectories_max=32,extra_repeat_trajectories_max=2,RK4_NFE=9600,adaptive_NFE_max=19200,total_model_forward_hard_limit=30000,
                     training_updates=0,backward_calls=0,OT_calls=0),
        adaptive_dependency=verified_dependency,production_changes=False,read_only_Euler_reuse=True,HEAD=start['HEAD'],
        planned_extra_repeat=dict(model='B',case_id='P0_patch0_seed10401',solver='R1',NFE=400,maximum_trajectories=1),
        planned_GPU_preflight=dict(model_forward_calls=12),
        total_model_forward_planned_maximum=29212,
        CPU_fixture=dict(path='registration/cpu_fixture.json',sha256=file_sha256(registration/'cpu_fixture.json'),
                         copied_existing_result=True,reexecuted=False,counts=fixture['counts']))
    write_json(registration/'protocol.json',protocol)
    write_json(registration/'discovery.json',discovery)
    write_json(registration/'old_euler_index.json',dict(source_archive=str(archive),rows=euler_rows))
    write_json(registration/'protected_assets.json',dict(scope='Actual read assets/sources only; not the entire workspace',files=list(protected.values())))
    environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,CUDA_build=torch.version.cuda,
        CUDA_initialized=torch.cuda.is_initialized(),torchdiffeq=td,cpu_count=os.cpu_count(),cpu_memory=Path('/proc/meminfo').read_text().splitlines()[:3],
        GPU=_run_command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
        GPU_processes=_run_command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        CPU_load=Path('/proc/loadavg').read_text().strip(),disk=_run_command(['df','-B1','/mnt/e']))
    write_json(registration/'environment.json',environment)
    if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
    result=dict(status='PASS',CPU_strict_checkpoint_loads=2,model_forwards=0,backwards=0,optimizer_updates=0,OT_calls=0,
        original_Gaussian_reused=True,cases=4,old_Euler_available=sum(r['status']=='AVAILABLE' for r in euler_rows),
        old_Euler_missing=[r for r in euler_rows if r['status']!='AVAILABLE'],
        model_input_contract='C/continuous values verified against actual arrays; exact discrete masks/y recovered from saved slot audit',
        torchdiffeq=td,CUDA_initialized=False,seconds=time.perf_counter()-tick)
    write_json(registration/'preflight.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    run(p.parse_args().out)
