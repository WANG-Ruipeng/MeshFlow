"""Bounded six-input neural and complete POST timing; no model work on import.

Each model runs in a separate process and owns one Phase, exactly 228 forwards.
The timed boundary excludes disk/ledger I/O, quality scoring and rendering.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import common as C
from .model import effective_proposal, load_teacher, load_student
from native_t1 import postprocessing as GEO
from native_t1 import postprocess_cli as PC

BASE = C.OUT / 'benchmark'
MODELS = ('D0_OFFICIAL', C.STUDENT)
REPEATS = {'neural': (5, 20), 'full_post': (3, 10)}


def rows_and_arrays():
    manifest = C.read(C.OUT/'manifest/data.json')
    rows = {r['sample_id']: r for r in manifest['dev_real']}
    ids = manifest['benchmark_ids']
    if len(ids) != 6 or len(set(ids)) != 6 or not set(ids) <= set(rows):
        raise ValueError('Exactly six frozen real DEV benchmark inputs required')
    result = []
    for sid in ids:
        r = rows[sid]
        if C.sha(r['input_path']) != r['input_sha256']:
            raise ValueError('Benchmark input changed')
        with np.load(r['input_path'], allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        n, k = r['N'], r['K']
        for name in ('input_model','input_d'):
            if a[name].dtype != np.float32 or a[name].shape != (n,3,3) or not np.isfinite(a[name]).all():
                raise ValueError('Invalid benchmark coordinates')
        if a['C'].tobytes() != a['input_model'][:k].tobytes():
            raise ValueError('Benchmark RAW differs from original C')
        if not np.array_equal(a['known_mask'], np.arange(n)<k):
            raise ValueError('Benchmark C must be original prefix')
        result.append((r,a))
    return result


def stats(values):
    a = np.asarray(values, np.float64)
    if len(a) == 0 or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError('Invalid measured times')
    return dict(timed_repetitions=len(a), median_seconds=float(np.median(a)),
                p95_seconds=float(np.quantile(a,.95)), minimum_seconds=float(a.min()),
                maximum_seconds=float(a.max()), mean_seconds=float(a.mean()))


def source_identity():
    return {str(p): C.sha(p) for p in [Path(__file__), C.CODE/'model.py', C.CODE/'common.py',
             Path(GEO.__file__), Path(PC.__file__)]}


def verify_complete(path):
    d = C.read(path)
    if d['status'] != 'COMPLETE' or d['forwards'] != 228:
        raise ValueError('Incomplete benchmark receipt')
    phase = C.read(C.OUT/'gpu_phases'/('benchmark_'+d['model'].lower()+'.json'))
    kind = 'teacher_benchmark_F' if d['model']=='D0_OFFICIAL' else 'student_benchmark_F'
    if phase['status'] != 'COMPLETE' or phase['counts'].get(kind) != 228:
        raise ValueError('Benchmark model phase not cleanly completed')
    C.verify(d['artifacts'])
    C.verify(d['source_hashes'])
    if C.sha(C.OUT/'manifest/data.json') != d['data_manifest_sha256']:
        raise ValueError('Benchmark registration changed')
    return d


def proposal(model, x, count):
    # Same actual PE/projection/decoder and FP32 residual add/sub as inference.
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        return effective_proposal(model,x,count)


def full_post(model, raw, original_C):
    """CPU RAW to final arrays; every call transfers the original untouched RAW."""
    n, k = len(raw), len(original_C)
    x = torch.from_numpy(raw).to(device='cuda',dtype=torch.float32).reshape(1,n,9)*C.SCALE
    count = torch.tensor([n],device='cuda',dtype=torch.long)
    pd, residual = proposal(model,x,count)
    with torch.inference_mode(), torch.autocast('cuda',enabled=False):
        proposed_model = (pd.float()/C.SCALE).reshape(n,3,3).cpu().numpy().copy()
    proposed_model[:k] = original_C
    result = GEO.process(proposed_model,k)
    # This is the actual PC.run array contract, with filesystem operations
    # removed. No second geometry process, quantization or altered radius.
    arrays = PC._validate_result(proposed_model,k,result)
    return arrays


def execute_one(label):
    C.runtime(threads=4)
    dest = BASE/label
    done = dest/'completion.json'
    if done.exists():
        verify_complete(done);print('SKIP_VERIFIED_COMPLETE benchmark',label);return
    selected = rows_and_arrays()
    registered = C.read(C.OUT/'registration.json')
    if registered['teacher']['sha256'] != C.EXPECTED or registered['teacher']['scale'] != C.SCALE:
        raise ValueError('Teacher identity/scale differs')
    if GEO.FIXED_CONFIG != C.read(C.OLD/'registration.json')['postprocess_config']:
        raise ValueError('GEO configuration changed')
    if (BASE/label/'timings.csv').exists():
        raise RuntimeError('Partial benchmark evidence; no automatic replay')
    source_hashes = source_identity()
    phase = C.Phase('benchmark_'+label.lower(),estimated_mib=2500 if label=='D0_OFFICIAL' else 1500)
    rows, peaks, summaries = [], [], []
    call_kind = 'teacher_benchmark_F' if label=='D0_OFFICIAL' else 'student_benchmark_F'
    try:
        # Cold-load means one model load into this fresh process, with no
        # independent filesystem-cache reset or repeat; report descriptively.
        if label != 'D0_OFFICIAL':
            checkpoint = C.OUT/'checkpoints/student_inference.pt'
            cr = C.read(checkpoint.with_suffix('.json'))
            if cr['status'] != 'PASS' or cr['step'] != 4000 or C.sha(checkpoint) != cr['sha256']:
                raise ValueError('Student endpoint/hash required')
        tick = time.perf_counter()
        if label == 'D0_OFFICIAL':
            # Registration already checked the full checkpoint SHA. Avoid a
            # redundant 7.29GB digest within the cold model loading boundary.
            model, spec = load_teacher(verify_checkpoint_hash=False)
            checkpoint, checkpoint_sha = C.ASSET, C.EXPECTED
        else:
            checkpoint = C.OUT/'checkpoints/student_inference.pt'
            checkpoint_sha = cr['sha256']
            model, spec, payload = load_student(checkpoint,require_step=4000)
            del payload
        model.to('cuda').eval().requires_grad_(False)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter()-tick
        parameters = sum(p.numel() for p in model.parameters())
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('Benchmark parameters must be FP32')
        parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
        C.write(dest/'load.json',dict(model=label,cold_load_seconds=load_seconds,independent_process_pid=os.getpid(),
            cold_load_scope='One CPU checkpoint/model construction + move to CUDA; includes first CUDA context when needed; no disk cache reset or repeated independent cold load',
            cold_load_descriptive_only=True,parameters=parameters,FP32_parameter_bytes=parameter_bytes,
            checkpoint_path=str(checkpoint),checkpoint_sha256=checkpoint_sha,checkpoint_file_bytes=checkpoint.stat().st_size,
            weight_comparison='Pure parameter tensor bytes at identical FP32; teacher full training checkpoint is not the compression denominator',
            teacher_weight_export_written=False,teacher_present_with_student=False,spec=spec,
            resident_before_benchmark=C.memory(),resources=C.resources()))
        all_rng_before = C.rng_hash()
        last_hashes = {}
        for r,a in selected:
            n, k, sid = r['N'],r['K'],r['sample_id']
            for mode,(warm,timed) in REPEATS.items():
                # No transfer is inside the neural-only timing; full POST
                # deliberately begins from CPU RAW anew on every repetition.
                x = torch.from_numpy(a['input_d']).cuda().reshape(1,n,9) if mode=='neural' else None
                count = torch.tensor([n],device='cuda',dtype=torch.long) if mode=='neural' else None
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                observations = []
                for iteration in range(warm+timed):
                    identity = dict(sample_id=sid,mode=mode,iteration=iteration,warmup=iteration<warm)
                    phase.begin(call_kind,identity)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if mode=='neural':
                        output = proposal(model,x,count)
                    else:
                        output = full_post(model,a['input_model'],a['C'])
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-start
                    # Model/ledger accounting is outside both timed intervals.
                    phase.end()
                    if mode=='neural':
                        pd, rd = output
                        if not torch.isfinite(pd).all():raise FloatingPointError('Nonfinite benchmark proposal')
                        digest = C.ah(pd.detach().cpu().numpy())
                        del pd,rd
                    else:
                        if output['output'][:k].tobytes()!=a['C'].tobytes():
                            raise AssertionError('Benchmark POST changed C')
                        digest = C.ah(output['output'])
                    key = (sid,mode)
                    if key in last_hashes and digest != last_hashes[key]:
                        raise AssertionError('Repeated independent input does not produce identical result')
                    last_hashes[key] = digest
                    rec = dict(model=label,sample_id=sid,N=n,K=k,region_type=r['region_type'],tier=r['tier'],
                        mode=mode,iteration=iteration,warmup=iteration<warm,seconds=seconds,
                        output_array_sha256=digest,input_sha256=r['input_sha256'],independent_original_input=True)
                    rows.append(rec);observations.append(rec)
                    del output
                scope = dict(model=label,sample_id=sid,mode=mode,N=n,K=k,
                             includes_resident_model=True,includes_warmups=True,**C.memory())
                peaks.append(scope)
                summaries.append(dict(model=label,sample_id=sid,N=n,K=k,region_type=r['region_type'],
                     mode=mode,warmups=warm,**stats([z['seconds'] for z in observations if not z['warmup']])))
                del x,count
            C.write(dest/'progress.json',dict(completed_inputs=len(summaries)//2,forwards=len(rows),resources=C.resources()))
        if C.rng_hash() != all_rng_before:
            raise AssertionError('Benchmark changed RNG state')
        if len(rows)!=228 or phase.counts.get(call_kind)!=228:
            raise AssertionError('Wrong bounded benchmark call count')
        for mode in REPEATS:
            summaries.append(dict(model=label,sample_id='ALL_SIX_DESCRIPTIVE_POOL',N=None,K=None,region_type='seat_and_back',
                mode=mode,warmups=sum(z['warmup'] for z in rows if z['mode']==mode),
                **stats([z['seconds'] for z in rows if z['mode']==mode and not z['warmup']])))
        C.csvwrite(dest/'timings.csv',rows);C.csvwrite(dest/'summary.csv',summaries);C.csvwrite(dest/'memory_scopes.csv',peaks)
        load = C.read(dest/'load.json')
        receipt = dict(status='COMPLETE',model=label,forwards=228,neural_forwards=150,full_post_forwards=78,
            neural_timed=120,full_post_timed=60,cold_load_seconds=load_seconds,parameters=parameters,
            FP32_parameter_bytes=parameter_bytes,checkpoint_sha256=checkpoint_sha,
            pure_neural_scope='GPU-resident input_d/count, one BF16 PE/projection/decoder plus FP32 effective residual add/sub; synchronized host wall clock; no H2D/D2H or GEO or ledger I/O',
            full_POST_scope='CPU original RAW -> H2D -> FP32 scale -> same BF16 model call -> FP32 inverse scale -> D2H -> restore original C -> unchanged GEO_C -> final array validation; no disk IO, scoring, rendering',
            allocated_reserved_note='Separate peaks; never added. Fresh model process; resets include only each resident-model measurement scope.',
            process_VRAM='See nvidia-smi resource snapshots; if WDDM per-process bytes unavailable report NA, not device total',
            concurrency_note='Exactly one benchmark model process; other system GPU processes/CPU load recorded; no claim of exclusive hardware beyond evidence',
            repeated_input='Each call starts from the identical original input, never previous output',
            CPU_threads=4,precision='BF16 forward/FP32 parameters and residual add/sub',no_TF32=True,SDPA='math',
            no_rng_consumed=True,no_Euler50_benchmark=True,no_new_quality_outputs=True,
            max_peak_allocated_bytes=max(p['peak_allocated_bytes'] for p in peaks),
            max_peak_reserved_bytes=max(p['peak_reserved_bytes'] for p in peaks),
            source_hashes=source_hashes,data_manifest_sha256=C.sha(C.OUT/'manifest/data.json'),
            artifacts={str(dest/name):C.sha(dest/name) for name in ('load.json','timings.csv','summary.csv','memory_scopes.csv')})
        C.write(done,receipt)
        phase.finish()
        del model;gc.collect();torch.cuda.empty_cache()
        print(json.dumps(dict(status='COMPLETE',stage='benchmark',model=label,forwards=228)),flush=True)
    except BaseException as exc:
        C.csvwrite(dest/'partial_timings.csv',rows) if rows else None
        C.write(dest/'failure.json',dict(error=repr(exc),completed_forward_receipts=len(rows),no_automatic_retry=True))
        phase.finish('FAILED',repr(exc));raise


def aggregate():
    import csv
    def loadcsv(p):
        with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
    receipts = {label:verify_complete(BASE/label/'completion.json') for label in MODELS}
    summaries = {label:{(r['sample_id'],r['mode']):r for r in loadcsv(BASE/label/'summary.csv')} for label in MODELS}
    paired = []
    for key,teacher in summaries[MODELS[0]].items():
        student = summaries[MODELS[1]][key]
        paired.append(dict(sample_id=key[0],mode=key[1],N=teacher['N'],K=teacher['K'],
            teacher_median_seconds=float(teacher['median_seconds']),student_median_seconds=float(student['median_seconds']),
            teacher_over_student_median_speedup=float(teacher['median_seconds'])/float(student['median_seconds']),
            teacher_p95_seconds=float(teacher['p95_seconds']),student_p95_seconds=float(student['p95_seconds'])))
    C.csvwrite(BASE/'paired_speed.csv',paired)
    C.write(BASE/'completion.json',dict(status='COMPLETE',models=list(MODELS),forwards_per_model=228,total_forwards=456,
        pure_FP32_parameter_compression=receipts[MODELS[0]]['FP32_parameter_bytes']/receipts[MODELS[1]]['FP32_parameter_bytes'],
        comparisons_path=str(BASE/'paired_speed.csv'),comparisons_sha256=C.sha(BASE/'paired_speed.csv'),
        model_receipts={str(BASE/x/'completion.json'):C.sha(BASE/x/'completion.json') for x in MODELS},
        no_full_generator_pipeline_speedup_claim=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS);a=p.parse_args()
    if a.model:
        execute_one(a.model);return
    done=BASE/'completion.json'
    if done.exists():
        d=C.read(done);C.verify(d['model_receipts'])
        for label in MODELS:verify_complete(BASE/label/'completion.json')
        if C.sha(d['comparisons_path'])!=d['comparisons_sha256']:raise ValueError('Benchmark summary changed')
        print('SKIP_VERIFIED_COMPLETE benchmark_all');return
    # Fresh subprocesses eliminate teacher allocator/context occupancy from the
    # student's memory scope. Child Phase independently measures resources.
    for label in MODELS:
        completed=subprocess.run([sys.executable,'-B','-m',__package__+'.benchmark','--model',label],cwd=C.ROOT)
        if completed.returncode:raise SystemExit(completed.returncode)
    aggregate()


if __name__=='__main__':main()
