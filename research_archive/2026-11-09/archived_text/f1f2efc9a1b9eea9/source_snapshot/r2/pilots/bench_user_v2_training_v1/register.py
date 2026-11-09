"""Freeze the realized data pool and one authorized coverage experiment."""
from pathlib import Path
import argparse
import shutil
from .common import ROOT, OUT, ARMS, EXPERIMENT, BASELINE_ROOT, read, resolve, sha, write_new, resources

def prepare_eval():
    source = BASELINE_ROOT / 'eval_manifest.json'
    dest = OUT / 'eval_manifest.json'
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    if sha(source) != sha(dest):
        raise ValueError('Evaluation manifest must be byte-identical to A')
    return dest

def freeze():
    if (OUT / 'training_registration.json').exists():
        raise FileExistsError('Already frozen; use training --check-cpu')
    evaluation = prepare_eval()
    old = read(BASELINE_ROOT / 'training_registration.json')
    start = BASELINE_ROOT / 'training/A_COMMON_LR1E5/checkpoint_step0.pt'
    start_receipt = read(start.with_suffix('.json'))
    if sha(start) != start_receipt['sha256']:
        raise ValueError('Protected A initial snapshot changed')
    manifest_path, schedule_path = OUT / 'data/manifest.json', OUT / 'data/schedule.json'
    manifest = read(manifest_path)
    data_counts = dict(rows=len(manifest['rows']), train=len(manifest['train']), dev=len(manifest['dev']),
                       train_parents=len({r['parent_id'] for r in manifest['train']}))
    gate = read(OUT / 'cpu_gate.json')
    if gate['status'] != 'PASS' or gate['neural_forwards'] != 0:
        raise ValueError('Successful CPU gate required before freeze')
    paths = [resolve(p) for p in old['training_code_hashes']]
    for p, h in old['training_code_hashes'].items():
        if sha(p) != h:
            raise ValueError('Inherited source changed: ' + p)
    paths += sorted(Path(__file__).parent.glob('*.py'))
    source_hashes = {str(p):sha(p) for p in paths}
    reg = dict(schema='BENCH_USER_V2_TRAINING_REGISTRATION_V1', status='FROZEN', experiment=EXPERIMENT,
        arms=list(ARMS), updates=1500, batch=8, microbatch=1, hybrid_lambda=.25,
        source_checkpoint=str(start), source_checkpoint_sha256=start_receipt['sha256'],
        source_generator_sha256=start_receipt['generator_state_sha256'],
        source_optimizer_sha256=start_receipt['optimizer_state_sha256'], source_rng_sha256=start_receipt['rng_sha256'],
        source_global_step=12000, final_global_step=13500, training_seed=20261020,
        model_config=old['model_config'], training_code_hashes=source_hashes,
        manifest=str(manifest_path), manifest_sha256=sha(manifest_path), data_counts=data_counts,
        input_schedule=str(schedule_path), input_schedule_sha256=sha(schedule_path),
        eval_manifest=str(evaluation), eval_manifest_sha256=sha(evaluation),
        evaluation_manifest=str(evaluation), evaluation_manifest_sha256=sha(evaluation),
        protocol_sha256=sha(OUT/'PROTOCOL.md'), cpu_gate_sha256=sha(OUT/'cpu_gate.json'),
        optimizer='Exact A step0 fresh AdamW, all groups1e-5,beta(.9,.95),eps1e-8,wd0,clip1',
        new_parameters=0, fixed_probe_forwards=0, fresh_training_forwards_per_arm=12000,
        fresh_backwards_per_arm=12000, fresh_updates_per_arm=1500,
        precision='BF16forward FP32parameters/integration; deterministic SDPA math TF32off',
        failure_contract='No silent replay or scientific setting changes; exact completed cursor resume allowed',
        expected_new_raw=48, expected_new_d0=48, expected_new_score_rows=144,
        generator_forward_sampling=2400, final_selection='1500only',
        evaluation_labels={'USER_V2_S1500':dict(arm='USER_V2_SEAM',subset='full',local_step=1500,global_step=13500,
            generator_path=str(OUT/'training/USER_V2_SEAM/generator_step1500.pt'))},
        reused_baseline=dict(root=str(BASELINE_ROOT),label='A_S1500',eval_manifest_sha256=sha(evaluation)),
        no_commit_push_default_change=True)
    write_new(OUT/'resource_initial.json', resources())
    write_new(OUT/'registration.json', reg)
    write_new(OUT/'training_registration.json', reg)
    return dict(status='FROZEN', data_counts=data_counts, updates=1500,
                registration_sha256=sha(OUT/'training_registration.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepare-eval',action='store_true')
    args=p.parse_args()
    print(prepare_eval() if args.prepare_eval else freeze())
