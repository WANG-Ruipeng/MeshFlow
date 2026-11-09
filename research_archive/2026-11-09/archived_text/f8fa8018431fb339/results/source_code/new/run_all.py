"""Resumable entry: register, freeze, sample, score, gallery, report. Zero training."""
import argparse
from pathlib import Path
import json
import os
import sys
from .quick_common import ROOT, OUT, CODE, read, write, sha, event


def freeze(out):
    path=out/'protocol.json'
    if path.exists(): return read(path)
    requests=read(out/'requests.json')
    definitions=dict(
        schema='CHAIR_CONTROL_USABILITY_QUICK_PILOT_V1', status='FROZEN_BEFORE_GENERATION',
        checkpoints=read(out/'registration/checkpoints.json'),
        main_parents=sorted({r['uid'] for r in requests if r['panel']=='main'}),
        main_requests=64, nested_requests=24, overlapping_requests=8, unique_limit=80,
        candidates=[0,1,2,3], nested_candidates=[0,1], alpha=1,
        training_updates=0, backward_calls=0, sampler='original_Euler50', nfe=50,
        precision='BF16 forward; FP32 parameters/state/integration; math SDPA; TF32 off; strict deterministic',
        gpu_workers=1, gpu_lock='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock', cpu_threads_per_worker=2,
        scheduling_reason='One GPU worker is explicitly mandated; independent bounded CPU preparation/scoring/rendering may overlap.',
        sampler_limits=dict(complete_attempts=82, identity_replays=2, sampling_forwards=4100, extra_precheck_forwards=12, total_forwards=4112),
        old_queries=512, surface_queries=2048, sensitivity_queries=8192,
        sensitivity='First two lexicographic A20 validation parents, candidate0, both models: four outputs only',
        surface_thresholds=[.005,.01,.02,.05], interface_points_per_edge=32, interface_sensitivity_points=128,
        interface_thresholds=[.001,.003,.005,.01,.02,.05],
        D_sym2048='(mean d(Q_Y,F)^2 + mean d(Q_F,Y)^2)/(2*L0^2); free finite closed triangles; no alignment',
        oracle='Choose minimum D_sym2048 among fixed 0/1 or 0/1/2/3; all other selected metrics from that same candidate',
        diversity='Mean six unordered pair symmetric free surface MSE / L0^2; full value NA on any invalid output',
        failures='Attempted model failures: distance +inf, Fscore0; unattempted assets/completion gaps: NA; full request denominator retained',
        interface='True GT known/free interface only; 32 equal-length midpoint samples per edge weighted by edge length/32; finite free edges and surfaces separate',
        common_region='GT source faces outside C3, one fixed2048 sample; directed distance to current free only, never copied C',
        geometry='No ICP, re-scaling, welding, face deletion, smoothing, repair, or target inputs',
        inference_input_allowlist=['z','C','known_mask','valid_mask','N'],
        human_usable='NA', user_study='NOT_RUN', human_rule='C preservation plus all three real-human A/B/C scores >=2',
        purpose='Low-poly candidates for shape ideation and later selection/editing, not manufacturing or physics-ready meshes',
        aggregation='candidate-within-condition first; eight parents equally weighted; paired parent differences, no candidate pseudoreplication',
        requests_sha256=sha(out/'requests.json'), construction_protocol_sha256=sha(out/'registration/construction_protocol.json'),
        evidence_limits='Old development parents, two frozen models, new metric definitions and noise; exploratory, no independent confirmation or human-use claim')
    if (out/'task_contract.txt').exists():definitions['task_contract_sha256']=sha(out/'task_contract.txt')
    write(path,definitions)
    science=['native_t1/model.py','native_t1/context_geometry.py','native_t1/runtime.py',
             'models/equidit.py','models/attention.py','configs/snet/base-120m-ot-v-chair.yaml',
             'experiments/mf_h1_local0/control_mvp48/model_mvp.py',
             'experiments/mf_h1_local0/control_mvp48/eval_runtime.py',
             'experiments/mf_h1_local0/chair_hybrid_coupling_v1/sampling.py']
    write(out/'registration/generation_sources_before.json',{p:sha(ROOT/p) for p in science})
    write(out/'registration/new_code_before_generation.json',{str(p.relative_to(ROOT)):sha(p) for p in CODE.glob('*.py')})
    return definitions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['register','freeze','sample','score','gallery','report','all'],default='all')
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
    stages=['register','freeze','sample','score','gallery','report'] if args.stage=='all' else [args.stage]
    for stage in stages:
        event(out/'ledger/stages.jsonl',dict(event='STAGE_BEGIN',stage=stage))
        try:
            if stage=='register':
                from .registration import prepare
                result=prepare(out)
            elif stage=='freeze':result=freeze(out)
            elif stage=='sample':
                freeze(out)
                from .sampling_worker import run
                result=run(out)
            elif stage=='score':
                from .scoring import run
                from .aggregation import finalize
                result=run(out);finalize(out)
            elif stage=='gallery':
                from .gallery import run
                result=run(out)
            else:
                from .reporting import run
                result=run(out)
            event(out/'ledger/stages.jsonl',dict(event='STAGE_COMPLETE',stage=stage))
            print(stage, json.dumps(result,ensure_ascii=False,default=str)[:3000],flush=True)
        except BaseException as exc:
            event(out/'ledger/stages.jsonl',dict(event='STAGE_FAILED',stage=stage,error=repr(exc)))
            raise


if __name__=='__main__':main()
