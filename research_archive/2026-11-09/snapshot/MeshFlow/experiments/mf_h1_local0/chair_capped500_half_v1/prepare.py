"""Register one fixed alpha and reuse all existing conditions/noises. No model calls."""
from .common import *

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protected={};manifests={}
    for panel in ('OLD','FRESH'):
        source=PREVIOUS/'manifests'/f'{panel}.json';data=read_json(source)
        assert len(data['rows'])==16 and len({r['uid'] for r in data['rows']})==8
        protected[str(source)]=digest(source)
        data.update(labels=list(LABELS),source_manifest=str(source),source_manifest_sha256=digest(source),
            interpretation='Previously observed eight DEV parents and OLD/FRESH noises; not independent confirmation.')
        path=OUT/'manifests'/f'{panel}.json';freeze_json(path,data);manifests[panel]=digest(path)
        for r in data['rows']:
            for key,hkey in [('bank_npz','bank_sha256'),('condition_npz','condition_sha256')]:
                p=Path(r[key]);actual=digest(p);assert actual==r[hkey];protected[str(p)]=actual
    source=PREVIOUS/'registration/conditions.json';target=OUT/'registration/conditions.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==source.read_bytes()
    else:target.write_bytes(source.read_bytes())
    protected[str(source)]=digest(source)
    for p in (ROOT/'native_t1/runtime.py',ROOT/'experiments/mf_h1_local0/chair_hybrid_coupling_v1/sampling.py',
              ROOT/'experiments/mf_h1_local0/chair_hybrid_coupling_v1/checkpoint.py',
              ROOT/'experiments/mf_h1_local0/chair_control_feedback_auto_v1/worker.py',
              ROOT/'experiments/mf_h1_local0/chair_control_feedback_auto_v1/assets.py'):
        protected[str(p)]=digest(p)
    for p in Path(__file__).parent.glob('*.py'):
        if p.name in ('common.py','prepare.py','blend.py','ledger.py','worker.py'):protected[str(p)]=digest(p)
    from experiments.mf_h1_local0.chair_control_feedback_auto_v1 import evaluation as old
    for p in (Path(old.__file__),Path(old.control.__file__),Path(old.quality.__file__)):
        protected[str(p)]=digest(p)
    protocol=dict(task='CHAIR_CAPPED500_HALF_AMPLITUDE_PILOT_V1',status='FROZEN',alpha=.5,
        formula='FP64 arithmetic midpoint of corresponding FP32 parameters, rounded once to FP32. Buffers must be equal and are copied unchanged.',
        labels=list(LABELS),new_label=LABEL,start_checkpoint=str(START),start_checkpoint_sha256=START_SHA,
        capped_checkpoint=str(CAPPED),capped_checkpoint_sha256=CAPPED_SHA,manifest_sha256=manifests,
        protected_files=protected,original_scorer_identity=old.identities(),budget=dict(training_updates=0,backward_calls=0,model_forward_calls=1600,new_Euler50_attempts=32,
        solver_steps_per_output=50,additional_seeds=0,additional_alphas=0,baseline_generated=0),
        evaluation=dict(parents=8,OLD_candidates=[0,1],FRESH_candidates=[2,3],baseline_cells_reused=64,total_cells=96,
        references='Original v1r1 S/I/E plus unchanged full quality; same frozen queries and thresholds. All parents/noises retained.',
        comparisons=['HALF_CAPPED500 minus START','HALF_CAPPED500 minus HIST_CAPPED500','HIST_CAPPED500 minus START'],
        interpretation='Descriptive salvage test on already observed DEV panels; no new independent confirmation or promotion threshold.'),
        hypothesis='The historical continuation direction may retain some positive seam response at half amplitude while reducing shape/tail cost.',
        success_interpretation='A useful compromise needs seam benefit over START and reduced CAPPED costs; mere return to START is not retained control benefit. No alpha selection after results.',
        runtime='Original Euler50; BF16 forward, FP32 parameters/integration, deterministic strict, TF32 off, math SDPA, C hard preserved; no repairs/guidance.',
        GPU_plan='One resident model under maintained exclusive GPU lock; independent CPU work uses two threads/process. No inference probe or training.',
        stop='Complete this alpha and fixed requests then stop. Failed attempted requests are not resampled automatically. No optimizer/default/Git changes.')
    freeze_json(OUT/'protocol.json',protocol)
    return dict(status='PASS',requests=32,alpha=.5,training_updates=0,model_calls=0,protocol_sha256=digest(OUT/'protocol.json'))

if __name__=='__main__':
    import json
    print(json.dumps(run()))
