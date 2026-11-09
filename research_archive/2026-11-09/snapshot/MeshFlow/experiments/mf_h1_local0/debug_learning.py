"""Stage D: gated, zero-update diagnosis of the existing Q/G step200 states.

No optimizer is constructed, no step is allowed, and no training is resumed.
The stable Stage C gate is checked before Harness initializes a CUDA model.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from .debug_repeatability import Harness, load_case, OLD, PROJECT, REPO, tensor_hash, metadata
from .prompt_dataset import RealChairStream, collate_samples, collate_prompts, sample_input_hash
from .sample import euler_sample
from .evaluate import guidance_metrics, raw_quality, completion_rms

ARMS = ('Q_center', 'G_closest')
KINDS = ('single_edge', 'single_face', 'mixed_2edges_1face')
SEEDS = (4101, 4102)
MODES = ('off', 'correct', 'cyclic_mismatch')
TEACHER_HASH = '294bb081c6bc6a206a5febbcf7a8cda53cc9bdd6d0b5cccf25c317d03290bdc8'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_gate(path, output):
    path = Path(path).resolve()
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') not in ('PASS', 'PASS_EXECUTION'):
        raise RuntimeError('Stage C has not passed; Stage D is forbidden')
    gate = result.get('gate', {})
    required = ('stage_d_allowed', 'bitwise_repeatability', 'same_process',
                'fresh_process', 'all_prompt_types')
    if any(gate.get(key) is not True for key in required):
        raise RuntimeError('Stage C needs explicit strict repeatability gates: '+', '.join(required))
    settings = result.get('settings', {})
    if (settings.get('precision') != 'bf16' or settings.get('deterministic_algorithms') is not True
            or settings.get('sdpa_backend') != 'math'):
        raise RuntimeError('Gate does not validate the selected BF16, deterministic, SDPA math path')
    if settings.get('deterministic_warn_only', False) is not False:
        raise RuntimeError('warn-only deterministic execution is not a stable gate')
    if settings.get('tf32', False) or settings.get('tf32_matmul', False) or settings.get('tf32_cudnn', False):
        raise RuntimeError('Stage D requires both TF32 settings disabled')
    if not Path(output).resolve().is_relative_to((PROJECT/'outputs').resolve()):
        raise ValueError('New debug output must stay below the experiment outputs directory')
    return dict(path=str(path), sha256=file_hash(path), validated=result)


def case_hashes(case):
    return {key: tensor_hash(value) for key, value in
            dict(z=case['z'], y=case['y'], mask=case['mask'], **case['prompts']).items()}


def batch_hashes(batch):
    return {key: tensor_hash(value) for key, value in
            dict(tokens=batch['tokens'], noise=batch['noise'], y=batch['num_faces'],
                 mask=batch['masks'], t=batch['t'], **batch['prompts']).items()}


def valid_values(tensor, mask):
    return tensor.detach().float()[mask]


def square_sum(tensor):
    return float(tensor.double().square().sum().item())


class ResidualDiagnostics:
    """Summaries only; no hidden activation arrays are retained."""
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.mode = None
        self.mask = None
        self.rows = {mode: [dict(feature_square_sum=0., residual_square_sum=0.,
                          elements=0, observed_forwards=0) for _ in range(4)] for mode in MODES}
        self.handles = []
        for index, layer in enumerate(wrapper.adapter.layers):
            self.handles.append(layer.register_forward_hook(self.adapter_hook(index)))
        for index, layer_number in enumerate((3, 6, 9, 12)):
            self.handles.append(wrapper.backbone.layers[layer_number-1].register_forward_hook(self.base_hook(index)))

    def observe(self, index, feature, residual=None):
        values = valid_values(feature, self.mask)
        if not torch.isfinite(values).all():
            raise RuntimeError('Nonfinite teacher hidden feature')
        row = self.rows[self.mode][index]
        row['feature_square_sum'] += square_sum(values)
        row['elements'] += values.numel()
        row['observed_forwards'] += 1
        if residual is not None:
            values = valid_values(residual, self.mask)
            if not torch.isfinite(values).all():
                raise RuntimeError('Nonfinite teacher control residual')
            row['residual_square_sum'] += square_sum(values)

    def adapter_hook(self, index):
        def hook(module, args, output):
            if self.mode in ('correct', 'cyclic_mismatch'):
                self.observe(index, args[0], output)
        return hook

    def base_hook(self, index):
        def hook(module, args, output):
            if self.mode == 'off':
                self.observe(index, output)
        return hook

    def summary(self):
        result = {}
        for mode, rows in self.rows.items():
            result[mode] = []
            for layer, row in zip((3, 6, 9, 12), rows):
                count = row['elements']
                feature = math.sqrt(row['feature_square_sum']/count) if count else None
                residual = math.sqrt(row['residual_square_sum']/count) if count else None
                result[mode].append(dict(layer_one_based=layer, feature_rms=feature,
                    residual_rms=residual, residual_to_feature_rms=residual/feature if feature else None,
                    valid_hidden_elements=count, observed_forwards=row['observed_forwards']))
        return result

    def close(self):
        self.mode = None
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def teacher_diagnosis(harness):
    manifest = json.loads((OLD/'split_manifest.json').read_text())
    samples = RealChairStream(manifest['train'], seed=9010).next_effective_batch(8)
    digest = sample_input_hash(samples)
    denominator = sum(int(sample['num_faces'])*9 for sample in samples)
    if digest != TEACHER_HASH or denominator != 40635:
        raise RuntimeError('Historical fixed teacher reconstruction changed; stop before diagnosis')
    report = dict(source='Historical seed9010 first effective batch; reconstructed without training',
        seed=9010, rng_seeds=dict(data=19010, prompt=29010, time=39010),
        input_hash=digest, parent_uids=[sample['parent_uid'] for sample in samples],
        face_counts=[int(sample['num_faces']) for sample in samples],
        original_t=[sample['t'] for sample in samples], effective_batch=8, microbatch=2,
        valid_coordinate_denominator=denominator,
        mismatch_rule='Sample j receives the unchanged prompt of sample (j+1) modulo 8; Xt/t/velocity target are unchanged',
        additional_t_strata='NOT_RUN', arms={})
    harness.report['teacher'] = report
    for arm in ARMS:
        wrapper = harness.wrapper(arm)
        harness.report.setdefault('initial_adapter_state_hashes', {})[arm] = harness.report['adapter_states'][arm]['sha256']
        diagnostics = ResidualDiagnostics(wrapper)
        losses = {mode: 0. for mode in MODES}
        per_sample = {mode: [] for mode in MODES}
        delta = {name: dict(square_sum=0., max_abs=0.) for name in ('correct_minus_off', 'correct_minus_mismatch')}
        microbatch_rows = []
        try:
            for start in range(0, 8, 2):
                batch = collate_samples(samples[start:start+2], 'cuda')
                before = batch_hashes(batch)
                t, x0, x1, mask = batch['t'], batch['noise'], batch['tokens'], batch['masks']
                xt = t[:, None, None]*x1 + (1-t[:, None, None])*x0
                velocity_target = x1-x0
                # Full targets stay outside prompts, encoded states and the sampler.
                mismatched = collate_prompts([samples[(j+1)%8]['prompt'] for j in range(start,start+2)], 'cuda')
                mismatch_before = {key: tensor_hash(value) for key,value in mismatched.items()}
                predictions = {}
                diagnostics.mask = mask.repeat_interleave(3, dim=1)
                for mode in MODES:
                    diagnostics.mode = mode
                    prompts = None if mode == 'off' else batch['prompts'] if mode == 'correct' else mismatched
                    with harness.autocast():
                        pred = wrapper(xt, t, y=batch['num_faces'], mask=mask,
                                       prompts=prompts, control_scale=0. if mode == 'off' else 1.)
                    harness.counts['teacher_forwards'] += 1
                    if not torch.isfinite(pred).all():
                        raise RuntimeError('Nonfinite teacher velocity')
                    predictions[mode] = pred.float()
                    error = (pred.float()-velocity_target).square()
                    losses[mode] += float((error*mask.unsqueeze(-1)).sum().item())
                    for j in range(2):
                        n = int(batch['num_faces'][j])
                        per_sample[mode].append(float(error[j,:n].sum().item()/(n*9)))
                    harness.arrays[f'teacher_{arm}_mb{start//2}_{mode}_velocity'] = pred.float().cpu().numpy()
                diagnostics.mode = None
                for name, other in [('correct_minus_off','off'),('correct_minus_mismatch','cyclic_mismatch')]:
                    change = valid_values(predictions['correct']-predictions[other], mask)
                    delta[name]['square_sum'] += square_sum(change)
                    delta[name]['max_abs'] = max(delta[name]['max_abs'],float(change.abs().max().item()))
                if batch_hashes(batch) != before or {key:tensor_hash(value) for key,value in mismatched.items()} != mismatch_before:
                    raise RuntimeError('Teacher inputs were modified in place')
                if arm == ARMS[0]:
                    for key,value in [('xt',xt),('target_velocity',velocity_target),('t',t),('mask',mask)]:
                        harness.arrays[f'teacher_mb{start//2}_{key}'] = value.cpu().numpy()
                microbatch_rows.append(dict(sample_indices=[start,start+1],input_hashes=before,
                    xt=metadata(xt),target_velocity=metadata(velocity_target),inputs_unchanged=True))
            mse = {mode: value/denominator for mode,value in losses.items()}
            report['arms'][arm] = dict(mse=mse, mse_per_sample=per_sample,
                correct_mse_reduction_from_off=mse['off']-mse['correct'],
                correct_mse_reduction_from_mismatch=mse['cyclic_mismatch']-mse['correct'],
                velocity_changes={key:dict(rms=math.sqrt(value['square_sum']/denominator),max_abs=value['max_abs'])
                                  for key,value in delta.items()},
                layer_residuals=diagnostics.summary(),microbatches=microbatch_rows,
                wrapper_counts=dict(wrapper.counts))
            harness.write()
            print(json.dumps(dict(stage=harness.label,teacher_arm=arm,mse=mse,counts=harness.counts)),flush=True)
        finally:
            diagnostics.close()
    if sample_input_hash(samples) != digest:
        raise RuntimeError('Historical teacher source arrays changed during diagnosis')
    report['inputs_unchanged'] = True


def generate_unique_outputs(harness):
    outputs, cases = {}, {}
    noise_hashes = {}
    for seed in SEEDS:
        for kind in KINDS:
            for parent in range(4):
                case = load_case(parent, kind, seed)
                digest = tensor_hash(case['z'])
                if seed in noise_hashes and noise_hashes[seed] != digest:
                    raise RuntimeError('Same-seed archived noise differs across parent/type; paired cross-scoring is forbidden')
                noise_hashes[seed] = digest
                cases[(parent, kind, seed)] = case
    report = dict(planned_unique_adapted_condition_inputs=48,planned_independent_base_draws=2,
                  actual_rollouts=[],condition_reuse='Cyclic mismatch reuses the neighboring condition output; no extra generation',
                  condition_input_hashes={},same_noise_verified=True,noise_hashes_by_seed=noise_hashes)
    harness.report['generation'] = report
    # Base has no geometry condition; the two archived noises are used once each.
    for wrapper in harness.wrappers:
        wrapper.close()
    harness.wrappers = []
    for seed in SEEDS:
        case = cases[(0, KINDS[0], seed)]
        before = case_hashes(case)
        generated, stats = euler_sample(harness.base,case['z'].clone(),case['y'],case['mask'],precision=harness.precision)
        harness.counts['complete_rollouts'] += 1
        harness.counts['base_rollouts'] += 1
        harness.counts['independent_base_draws'] += 1
        harness.counts['sampler_calls'] += 1
        outputs[('base',None,None,seed)] = generated[0].cpu().numpy()
        harness.arrays[f'base_seed{seed}'] = outputs[('base',None,None,seed)]
        report['actual_rollouts'].append(dict(arm='base',seed=seed,stats=stats))
        if stats['status'] != 'PASS' or case_hashes(case) != before:
            raise RuntimeError('Base sampling failed or changed its archived input')
    for arm in ARMS:
        wrapper = harness.wrapper(arm)
        if harness.report['adapter_states'][arm]['sha256'] != harness.report['initial_adapter_state_hashes'][arm]:
            raise RuntimeError('Adapter state changed between teacher diagnosis and generation')
        for kind in KINDS:
            for seed in SEEDS:
                for parent in range(4):
                    case = cases[(parent,kind,seed)]
                    before = case_hashes(case)
                    cases[(parent,kind,seed)] = case
                    generated,stats = euler_sample(wrapper,case['z'].clone(),case['y'],case['mask'],
                                                   prompts=case['prompts'],precision=harness.precision)
                    harness.counts['complete_rollouts'] += 1
                    harness.counts['adapted_rollouts'] += 1
                    harness.counts['unique_adapted_condition_inputs'] += 1
                    harness.counts['sampler_calls'] += 1
                    key = arm,parent,kind,seed
                    outputs[key] = generated[0].cpu().numpy()
                    name = f'{arm}_parent{parent}_{kind}_seed{seed}'
                    harness.arrays[name] = outputs[key]
                    report['condition_input_hashes'][name] = before
                    report['actual_rollouts'].append(dict(arm=arm,parent_index=parent,kind=kind,seed=seed,stats=stats))
                    if stats['status'] != 'PASS' or case_hashes(case) != before:
                        raise RuntimeError('Adapted sampling failed or changed archived input')
                    if stats['call_delta'].get('wrapper_forward') != 50 or stats['call_delta'].get('backbone_forward') != 50 or stats['call_delta'].get('adapter_layer') != 200:
                        raise RuntimeError('Actual adapter/backbone call counts differ from the 50-step protocol')
                    harness.write()
                    print(json.dumps(dict(stage=harness.label,generated=name,counts=harness.counts)),flush=True)
    return outputs,cases


def cross_score(harness, outputs, cases):
    cache, own_rows, cross_rows = {}, [], []
    def guide(arm,mesh_parent,condition_parent,kind,seed):
        key = arm,mesh_parent,condition_parent,kind,seed
        if key not in cache:
            output_key = (arm,mesh_parent,kind,seed) if arm != 'base' else ('base',None,None,seed)
            case = cases[(condition_parent,kind,seed)]
            cache[key] = guidance_metrics(outputs[output_key],case['arrays']['check_points'],case['record']['bbox_diagonal'])
            harness.counts['guide_scores'] += 1
        else:
            harness.counts['guide_score_cache_hits'] += 1
        return cache[key]
    quality_cache = {}
    for key, value in outputs.items():
        quality_cache[key] = raw_quality(value)
        harness.counts['quality_scores'] += 1
    for arm in (*ARMS,'base'):
        for kind in KINDS:
            for seed in SEEDS:
                for parent in range(4):
                    case = cases[(parent,kind,seed)]
                    output_key = (arm,parent,kind,seed) if arm != 'base' else ('base',None,None,seed)
                    score = guide(arm,parent if arm != 'base' else None,parent,kind,seed)
                    completion = completion_rms(outputs[output_key],case['arrays']['target'],case['record']['bbox_diagonal'])
                    harness.counts['completion_scores'] += 1
                    own_rows.append(dict(arm=arm,parent_index=parent,parent_uid=case['record']['parent_uid'],
                        kind=kind,noise_seed=seed,guide=score,quality=quality_cache[output_key],
                        completion_symmetric_rms_bbox_pct=completion))
    for arm in ARMS:
        for kind in KINDS:
            for seed in SEEDS:
                for a in range(4):
                    b = (a+1)%4
                    eaa = guide(arm,a,a,kind,seed)['guide_rms_bbox_pct']
                    eab = guide(arm,b,a,kind,seed)['guide_rms_bbox_pct']
                    ebb = guide(arm,b,b,kind,seed)['guide_rms_bbox_pct']
                    eba = guide(arm,a,b,kind,seed)['guide_rms_bbox_pct']
                    ma,mb = outputs[(arm,a,kind,seed)],outputs[(arm,b,kind,seed)]
                    change = ma.astype(np.float64)-mb.astype(np.float64)
                    cross_rows.append(dict(arm=arm,kind=kind,noise_seed=seed,parent_a=a,parent_b=b,
                        E_Ca_Ma=eaa,E_Ca_Mb=eab,E_Cb_Mb=ebb,E_Cb_Ma=eba,
                        a_own_condition_improvement=eab-eaa,b_own_condition_improvement=eba-ebb,
                        mean_own_binding_improvement=((eab-eaa)+(eba-ebb))/2,
                        output_change=dict(bitwise_equal=bool(np.array_equal(ma,mb)),
                            rms=float(np.sqrt(np.mean(change*change))),max_abs=float(np.max(np.abs(change))))))
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for kind in KINDS:
            own = [r for r in own_rows if r['arm']==arm and r['kind']==kind]
            cross = [r for r in cross_rows if r['arm']==arm and r['kind']==kind]
            baseline = [r for r in own_rows if r['arm']=='base' and r['kind']==kind]
            summary[arm][kind] = dict(own_guide_mean=float(np.mean([r['guide']['guide_rms_bbox_pct'] for r in own])),
                base_guide_mean=float(np.mean([r['guide']['guide_rms_bbox_pct'] for r in baseline])),
                mean_binding_improvement=float(np.mean([r['mean_own_binding_improvement'] for r in cross])),
                nonzero_condition_change_pairs=sum(not r['output_change']['bitwise_equal'] for r in cross),
                cycle_pairs=len(cross),both_conditions_improve_pairs=sum(r['a_own_condition_improvement']>0 and r['b_own_condition_improvement']>0 for r in cross),
                mean_min_angle_lt5_fraction=float(np.mean([r['quality']['min_angle_lt5_fraction'] for r in own])))
    harness.report['learning_scores'] = dict(own_condition_rows=own_rows,cyclic_cross_score_rows=cross_rows,
        summary=summary,guide_evaluations=len(cache),unique_quality_evaluations=len(quality_cache),
        response_metric='Coordinate RMS at common face-slot/corner indices under the same initial noise; not a surface-distance diversity metric',
        interpretation='Nonzero condition response, FM MSE improvement and guide improvement are distinct. Four parents and one adapter seed do not establish statistical significance, generalization or architecture advantage.',
        stable_repeatability_source=harness.report['stable_gate']['path'])


def run(output, gate):
    output = Path(output).resolve()
    gate_record = validate_gate(gate,output)
    label = 'stage_d_learning'
    for name in (label+'.json',label+'_arrays.npz',label+'_partial_arrays.npz'):
        if (output/name).exists():
            raise FileExistsError('Never overwrite prior Stage D evidence: '+str(output/name))
    if not output.is_dir():
        raise FileNotFoundError('Existing Stage C output directory is required')
    adapter_paths = {arm:OLD/arm/'checkpoints/adapter_step0200.pt' for arm in ARMS}
    file_before = {arm:file_hash(path) for arm,path in adapter_paths.items()}
    for arm,path in adapter_paths.items():
        saved = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if saved['step'] != 200 or saved['config']['arm'] != arm:
            raise ValueError('Only the historical matching step200 state may be loaded')
        del saved
    harness = Harness(output,label,precision='bf16',deterministic=True)
    harness.report['stable_gate'] = gate_record
    harness.report['adapter_file_hashes_before'] = file_before
    harness.report['stage_d_policy'] = dict(optimizer_construction=0,optimizer_step_forbidden=True,
        new_training=False,old_step200_only=True,phase3=False,parameter_and_integrator_dtype='float32',
        sdpa_backend='math',face_cfg=1,geometry_cfg=1,euler_updates=50,raw_face_slots=512,
        no_retraining_to_correct_historical_bf16_training=True)
    harness.counts.update(teacher_forwards=0,official_examples=0,base_rollouts=0,adapted_rollouts=0,
        sampler_calls=0,quality_scores=0,completion_scores=0,guide_score_cache_hits=0,
        unique_adapted_condition_inputs=0,independent_base_draws=0)
    example_hook = harness.base.register_forward_pre_hook(
        lambda module,args:harness.counts.__setitem__('official_examples',harness.counts['official_examples']+int(args[0].shape[0])))
    try:
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            teacher_diagnosis(harness)
            outputs,cases = generate_unique_outputs(harness)
            cross_score(harness,outputs,cases)
        expected = dict(official_forwards=2524,official_examples=2548,teacher_forwards=24,
            complete_rollouts=50,base_rollouts=2,adapted_rollouts=48,sampler_calls=50,
            condition_encodes=64,guide_scores=168,quality_scores=50,completion_scores=72,
            unique_adapted_condition_inputs=48,independent_base_draws=2,
            optimizer_updates=0,backbone_updates=0,phase3_updates=0)
        harness.report['expected_counts'] = expected
        if any(harness.counts[key] != value for key,value in expected.items()):
            raise RuntimeError('Stage D execution count differs from the explicit bounded plan')
        file_after = {arm:file_hash(path) for arm,path in adapter_paths.items()}
        harness.report['adapter_file_hashes_after'] = file_after
        if file_before != file_after:
            raise RuntimeError('A historical adapter file changed during diagnosis')
        example_hook.remove()
        harness.finish()
        initial = harness.report['initial_adapter_state_hashes']
        if initial != harness.report['adapters_after']:
            raise RuntimeError('Loaded adapter parameters changed during zero-update diagnosis')
        if harness.report['actual_attention_dispatch']['external_flash']:
            raise RuntimeError('External flash attention entered the declared SDPA math path')
        harness.report['stage_d_gate'] = 'PASS_EXECUTION_ONLY'
        harness.write()
        return harness.report
    except BaseException as error:
        harness.report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        np.savez_compressed(output/(label+'_partial_arrays.npz'),**harness.arrays)
        harness.write()
        raise
    finally:
        example_hook.remove()
        for wrapper in harness.wrappers:
            wrapper.close()
        import models.attention as attention
        for name,original in harness.original_dispatch.items():
            setattr(attention,name,original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--gate',required=True,help='Explicit Stage C stable_gate.json; must pass before model loading')
    args = parser.parse_args()
    report = run(args.output,args.gate)
    print(json.dumps(dict(stage='D',status=report['status'],counts=report['counts'])),flush=True)


if __name__ == '__main__':
    main()
