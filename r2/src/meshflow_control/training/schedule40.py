"""Independent official-EMA schedule experiment; controller owns GPU dispatch."""
from __future__ import annotations
from .. import runtime
from contextlib import nullcontext
import copy, csv, hashlib, os, random, signal, time, traceback
from pathlib import Path
import numpy as np
import torch
from ..config import MODEL_CONFIG
from ..checkpoints import state_hash, rng_state, restore_rng
from ..data.io import atomic_json, append_event, digest, json_hash, read_json
from ..data.dataset import collate, model_inputs
from ..models.native import build_model, native_execution
from .context_alignment import AlignmentHead, AttentionCapture, alignment_loss, select_messages
from .context_contract import HEAD_CONFIG, HEAD_PARAMETERS
from .context_checkpoint import _cpu_copy, _check_generator
from .losses import masked_fm_loss

EXPERIMENT = 'CHAIR_EARLY_ALIGNMENT_SCHEDULE_40_V1'
SCHEMA = 'chair_early_alignment_schedule40_training_v1'
INFERENCE_SCHEMA = 'chair_early_alignment_schedule40_generator_v1'
OFFICIAL_FILE_SHA256 = 'bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d'
OFFICIAL_BACKBONE_SHA256 = 'b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222'
SEGMENTS = {'COMMON': (0, 5000), 'EARLY': (0, 6000), 'FM_ONLY': (5000, 6000), 'LATE': (5000, 6000)}
MODEL_SEED = 20261008
PROTOCOL = dict(experiment=EXPERIMENT, total_logical_updates=6000, physical_updates_without_replay=13000,
    known_ratio=.4, hybrid_lambda=.25, batch=8, microbatch=1, clean_per_batch=4,
    width_augmentation=[.9, 1.1], generator_lr=1e-5, aligner_lr=1e-4,
    betas=[.9, .95], weight_decay=0., generator_clip=1., aligner_clip=1.,
    head=HEAD_CONFIG, model=MODEL_CONFIG, model_seed=MODEL_SEED,
    checkpoint_interval=100, precision='BF16 forward; FP32 parameters and integration',
    FM_denominator='whole effective batch valid free scalar count',
    early_active_inclusive=[1, 1000], late_active_inclusive=[5001, 6000],
    replay_contract='one dual-to-serial downgrade; at most one OOM resume per affected segment; <=99 completed updates replayed',
    official_file_sha256=OFFICIAL_FILE_SHA256, official_backbone_sha256=OFFICIAL_BACKBONE_SHA256)


def active_alignment(segment, step):
    if segment not in SEGMENTS or type(step) is not int or not 1 <= step <= 6000:
        raise ValueError('Invalid schedule identity/update')
    return (segment == 'EARLY' and step <= 1000) or (segment == 'LATE' and step > 5000)


def head_steps(segment, step):
    if segment not in SEGMENTS or type(step) is not int or not 0 <= step <= 6000:
        raise ValueError('Invalid completed update cursor')
    return min(step, 1000) if segment == 'EARLY' else max(0, step - 5000) if segment == 'LATE' else 0


def tree_hash(value):
    h = hashlib.sha256()
    def add(v):
        if isinstance(v, torch.Tensor):
            x = v.detach().cpu().contiguous()
            h.update(str(('tensor', str(x.dtype), tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for key in sorted(v, key=lambda k: (type(k).__name__, repr(k))):
                add(key); add(v[key])
        elif isinstance(v, (list, tuple)):
            h.update(str((type(v).__name__, len(v))).encode())
            for item in v: add(item)
        else:
            h.update(str((type(v).__name__, repr(v))).encode())
    add(value)
    return h.hexdigest()


def optimizer_names(model, head):
    names = [['generator.' + n for n, _ in model.named_parameters()]]
    if head is not None: names.append(['aligner.' + n for n, _ in head.named_parameters()])
    return names


def make_optimizer(model, head):
    groups = [dict(params=list(model.parameters()), lr=1e-5)]
    if head is not None: groups.append(dict(params=list(head.parameters()), lr=1e-4))
    return torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=0.)


def validate_optimizer(model, head, optimizer, segment, step):
    if not isinstance(optimizer, torch.optim.AdamW) or model.training or model.readout_mode != 'none':
        raise ValueError('Eval-mode Native generator and AdamW required')
    if (head is not None) != (segment in ('EARLY', 'LATE')):
        raise ValueError('Head ownership differs from schedule')
    modules = [model] + ([] if head is None else [head])
    if len(optimizer.param_groups) != len(modules): raise ValueError('Optimizer group count differs')
    for index, (module, group) in enumerate(zip(modules, optimizer.param_groups)):
        expected_step = step if index == 0 else head_steps(segment, step)
        if module.training or [id(p) for p in module.parameters()] != [id(p) for p in group['params']]:
            raise ValueError('Optimizer parameter identity/order differs')
        if (group['lr'], tuple(group['betas']), group['weight_decay']) != (1e-5 if index == 0 else 1e-4, (.9, .95), 0.):
            raise ValueError('Optimizer hyperparameters differ')
        for name, p in module.named_parameters():
            if p.dtype != torch.float32 or not p.requires_grad: raise ValueError('Trainable FP32 required: ' + name)
            state = optimizer.state.get(p, {})
            if expected_step == 0 and not state: continue
            if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != expected_step:
                raise ValueError('AdamW cursor differs: ' + name)
            if any(state[k].shape != p.shape or state[k].dtype != torch.float32 for k in ('exp_avg', 'exp_avg_sq')):
                raise ValueError('AdamW moment identity differs: ' + name)
    return True


def initialize_official(path, segment, device='cpu'):
    if segment not in ('COMMON', 'EARLY'): raise ValueError('FM_ONLY/LATE require complete COMMON5000 fork')
    runtime.configure_stable_runtime()
    actual_file = digest(path)
    if actual_file != OFFICIAL_FILE_SHA256: raise ValueError('Official chair file hash differs')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    ema = payload.get('ema')
    if not isinstance(ema, dict) or not ema: raise ValueError('Official EMA dictionary required')
    prefixed = [key.startswith('module.') for key in ema]
    if any(prefixed) and not all(prefixed): raise ValueError('Mixed official EMA key prefixes')
    state = {key[7:] if all(prefixed) else key: value for key, value in ema.items()}
    if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('Official backbone must be finite FP32')
    if state_hash(state) != OFFICIAL_BACKBONE_SHA256: raise ValueError('Official EMA backbone state differs')
    model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode='none', trainable=True).eval()
    model.backbone.load_state_dict(state, strict=True)
    if state_hash(model.backbone.state_dict()) != OFFICIAL_BACKBONE_SHA256: raise ValueError('EMA inheritance not exact')
    if torch.count_nonzero(model.role_embedding) or torch.count_nonzero(model.context_encoder.W_out.weight):
        raise ValueError('Role and Geo exit must initialize at exact zero')
    _check_generator(model.state_dict())
    head = AlignmentHead().eval() if segment == 'EARLY' else None
    audit = dict(selected_state='ema', official_file_sha256=actual_file,
        official_backbone_sha256=OFFICIAL_BACKBONE_SHA256,
        generator_initial_sha256=state_hash(model.state_dict()),
        head_initial_sha256=None if head is None else state_hash(head.state_dict()),
        backbone_tensors=len(state), generator_tensors=len(model.state_dict()),
        generator_parameters=sum(p.numel() for p in model.parameters()),
        role_nonzero=0, geo_exit_nonzero=0, Geo_seed=1010, fresh_AdamW=True,
        new_optimizer_step=0, no_conditioned_output_equivalence_claim=True)
    del state, ema, payload
    model.to(device)
    if head is not None: head.to(device)
    optimizer = make_optimizer(model, head)
    validate_optimizer(model, head, optimizer, segment, 0)
    return model, head, optimizer, audit


def _payload(model, head, optimizer, *, segment, step, stream_state, identities, lineage, receipt):
    validate_optimizer(model, head, optimizer, segment, step)
    if stream_state['batch_index'] != step: raise ValueError('Incomplete input/update checkpoint')
    generator = _cpu_copy(model.state_dict()); _check_generator(generator)
    aligner = None if head is None else _cpu_copy(head.state_dict())
    opt, rng = _cpu_copy(optimizer.state_dict()), rng_state()
    return dict(schema=SCHEMA, experiment=EXPERIMENT, segment=segment, global_step=step,
        head_completed_updates=head_steps(segment, step), update_in_progress=False,
        protocol=copy.deepcopy(PROTOCOL), protocol_sha256=json_hash(PROTOCOL),
        model_config=copy.deepcopy(MODEL_CONFIG), generator_state=generator,
        generator_state_sha256=state_hash(generator), aligner_state=aligner,
        aligner_state_sha256=None if aligner is None else state_hash(aligner),
        optimizer=opt, optimizer_state_sha256=tree_hash(opt),
        optimizer_parameter_names=optimizer_names(model, head), rng=rng, rng_sha256=tree_hash(rng),
        stream_state=copy.deepcopy(stream_state), stream_state_sha256=json_hash(stream_state),
        identities=copy.deepcopy(identities), lineage=copy.deepcopy(lineage), execution_receipt=copy.deepcopy(receipt))


def _write_checkpoint(path, payload, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists(): raise FileExistsError('Immutable checkpoint exists: ' + str(path))
    temporary = path.with_name(path.name + '.partial.' + str(os.getpid()))
    with temporary.open('xb') as handle:
        torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    return dict(path=str(path.resolve()), sha256=digest(path), schema=payload['schema'],
        segment=payload['segment'], global_step=payload['global_step'],
        generator_state_sha256=payload['generator_state_sha256'], aligner_state_sha256=payload.get('aligner_state_sha256'),
        optimizer_state_sha256=payload.get('optimizer_state_sha256'), rng_sha256=payload.get('rng_sha256'),
        stream_state_sha256=payload.get('stream_state_sha256'))


def validate_payload(payload):
    if payload.get('schema') != SCHEMA or payload.get('experiment') != EXPERIMENT or payload.get('update_in_progress') is not False:
        raise ValueError('Not a completed schedule40 checkpoint')
    if payload['protocol'] != PROTOCOL or payload['protocol_sha256'] != json_hash(PROTOCOL) or payload['model_config'] != MODEL_CONFIG:
        raise ValueError('Checkpoint protocol identity differs')
    segment, step = payload['segment'], payload['global_step']
    if segment not in SEGMENTS or type(step) is not int or not SEGMENTS[segment][0] <= step <= SEGMENTS[segment][1]:
        raise ValueError('Invalid checkpoint segment/cursor')
    if payload['stream_state']['batch_index'] != step or payload['head_completed_updates'] != head_steps(segment, step):
        raise ValueError('Checkpoint stream/head cursor differs')
    _check_generator(payload['generator_state'])
    for value, expected, hash_fn in (
        (payload['generator_state'], payload['generator_state_sha256'], state_hash),
        (payload['optimizer'], payload['optimizer_state_sha256'], tree_hash),
        (payload['rng'], payload['rng_sha256'], tree_hash),
        (payload['stream_state'], payload['stream_state_sha256'], json_hash)):
        if hash_fn(value) != expected: raise ValueError('Checkpoint internal state hash differs')
    head_state = payload['aligner_state']
    if segment in ('EARLY', 'LATE'):
        if not isinstance(head_state, dict) or state_hash(head_state) != payload['aligner_state_sha256']:
            raise ValueError('Checkpoint head state differs')
        if sum(p.numel() for p in head_state.values()) != HEAD_PARAMETERS: raise ValueError('Checkpoint head capacity differs')
    elif head_state is not None or payload['aligner_state_sha256'] is not None:
        raise ValueError('FM checkpoint unexpectedly contains a head')
    return payload


def load_checkpoint(path, segment=None, device='cpu', fork=False):
    runtime.configure_stable_runtime()
    payload = validate_payload(torch.load(path, map_location='cpu', weights_only=True, mmap=True))
    source = payload['segment']; segment = source if segment is None else segment
    if fork:
        if source != 'COMMON' or payload['global_step'] != 5000 or segment not in ('FM_ONLY', 'LATE'):
            raise ValueError('Fork requires complete COMMON5000 and FM_ONLY/LATE destination')
    elif segment != source: raise ValueError('Resume segment identity differs')
    model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode='none', trainable=True).eval()
    model.load_state_dict(payload['generator_state'], strict=True); model.to(device)
    head = AlignmentHead().eval() if segment in ('EARLY', 'LATE') else None
    if head is not None:
        if not fork: head.load_state_dict(payload['aligner_state'], strict=True)
        head.to(device)
    source_head = None if fork else head
    optimizer = make_optimizer(model, source_head)
    if payload['optimizer_parameter_names'] != optimizer_names(model, source_head): raise ValueError('Optimizer mapping differs')
    optimizer.load_state_dict(payload['optimizer'])
    if fork and head is not None:
        optimizer.add_param_group(dict(params=list(head.parameters()), lr=1e-4, betas=(.9, .95), weight_decay=0.))
    validate_optimizer(model, head, optimizer, segment, payload['global_step'])
    return model, head, optimizer, payload


def export_generator(checkpoint, out):
    payload = validate_payload(torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True))
    if payload['global_step'] != 6000 or payload['segment'] == 'COMMON': raise ValueError('Only final endpoints export')
    result = dict(schema=INFERENCE_SCHEMA, experiment=EXPERIMENT,
        segment=payload['segment'], global_step=6000, model_config=payload['model_config'],
        generator_state=payload['generator_state'], generator_state_sha256=payload['generator_state_sha256'],
        protocol_sha256=payload['protocol_sha256'], identities=payload['identities'], lineage=payload['lineage'],
        training_checkpoint_sha256=digest(checkpoint), inference_contract=dict(readout_mode='none',
        teacher=False, aligner=False, capture=False, forward='BF16', parameters='FP32', integration='FP32'))
    return _write_checkpoint(out, result, immutable=True)


def load_generator(path, device='cpu', expected_sha256=None):
    runtime.configure_stable_runtime()
    if expected_sha256 is not None and digest(path) != expected_sha256: raise ValueError('Generator export file differs')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('schema') != INFERENCE_SCHEMA or payload.get('experiment') != EXPERIMENT:
        raise ValueError('Not a schedule40 generator export')
    if payload['model_config'] != MODEL_CONFIG or payload['protocol_sha256'] != json_hash(PROTOCOL) or payload['global_step'] != 6000:
        raise ValueError('Generator export protocol differs')
    _check_generator(payload['generator_state'])
    if state_hash(payload['generator_state']) != payload['generator_state_sha256']: raise ValueError('Export tensor hash differs')
    model = build_model(copy.deepcopy(MODEL_CONFIG), readout_mode='none', trainable=False).eval()
    model.load_state_dict(payload['generator_state'], strict=True); model.to(device).set_trainable(False)
    return model, {k: v for k, v in payload.items() if k != 'generator_state'}


class CallLedger:
    """One writer per attempt; failed calls retained, never auto-retried."""
    def __init__(self, path, segment, attempt_id):
        self.path, self.segment, self.attempt_id = Path(path), segment, attempt_id
        self.counts, self.index = {}, 0
    def call(self, kind, function, context):
        self.index += 1; ident = self.index
        append_event(self.path, dict(event='CALL_BEGIN', call_id=ident, kind=kind,
            segment=self.segment, attempt_id=self.attempt_id, **context))
        started = time.perf_counter()
        try:
            result = function(); torch.cuda.synchronize()
        except BaseException as error:
            append_event(self.path, dict(event='CALL_FAILED', call_id=ident, kind=kind,
                segment=self.segment, attempt_id=self.attempt_id, error=repr(error), **context))
            raise
        seconds = time.perf_counter() - started
        self.counts[kind] = self.counts.get(kind, 0) + 1
        append_event(self.path, dict(event='CALL_DONE', call_id=ident, kind=kind,
            segment=self.segment, attempt_id=self.attempt_id, seconds=seconds, **context))
        return result, seconds


def update(model, head, optimizer, samples, targets, ledger, segment, step):
    if len(samples) != 8 or len({s['object_id'] for s in samples}) != 8 or sum(s['alpha'] == 1. for s in samples) != 4:
        raise ValueError('Require eight distinct parents and four exact clean samples')
    validate_optimizer(model, head, optimizer, segment, step - 1)
    device = next(model.parameters()).device
    if device.type != 'cuda': raise ValueError('Formal training requires an owned CUDA worker')
    active = active_alignment(segment, step)
    if active and (head is None or targets is None): raise ValueError('Active schedule requires registered head and cached teacher')
    denominator = sum(s['free_coordinate_count'] for s in samples)
    if denominator != 9 * sum(int((s['valid_mask'] & ~s['known_mask']).sum()) for s in samples):
        raise ValueError('Whole-batch FM denominator differs')
    torch.cuda.reset_peak_memory_stats(device); optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    result = dict(FM=0., raw_align=0., weighted_align=0., total_loss=0.,
        alignment_active=active, aligned_samples=0, free_coordinate_denominator=denominator,
        alignment_denominator=4, forward_seconds=0., aligner_forward_seconds=0.,
        backward_seconds=0., optimizer_seconds=0., sample_losses=[])
    for micro, sample in enumerate(samples):
        batch = collate([sample], device)
        context = dict(step=step, microbatch=micro, sample_index=sample['sample_index'],
            input_bytehash=sample['input_bytehash'], label_bytehash=sample['label_bytehash'])
        eligible = active and sample['alpha'] == 1.
        capture = AttentionCapture(model) if eligible else nullcontext(None)
        with capture as captured, native_execution(model, coordinates=(batch['xt'], batch['t']), autocast=True):
            velocity, seconds = ledger.call('train_forward', lambda: model(*model_inputs(batch)).float(), context)
            result['forward_seconds'] += seconds
            fm = masked_fm_loss(velocity, batch['u'], batch['valid_mask'], batch['known_mask'], denominator=denominator)
            raw, statistics = None, {}
            if eligible:
                messages = captured.get()
                tokens, selected = select_messages(messages, batch['known_mask'], batch['valid_mask'], 'free')
                teacher = torch.as_tensor(targets.for_sample(sample), dtype=torch.float32, device=device)
                prediction, seconds = ledger.call('aligner_forward', lambda: head(tokens[0]), context)
                result['aligner_forward_seconds'] += seconds
                raw, statistics = alignment_loss(prediction, teacher)
                total = fm + (.1 / 4.) * raw
                result['aligned_samples'] += 1
                result['raw_align'] += float(raw.detach()) / 4.
                result['weighted_align'] += .025 * float(raw.detach())
            else: total = fm
            if not bool(torch.isfinite(total)): raise FloatingPointError('Nonfinite combined training loss')
            fm_value, total_value = float(fm.detach()), float(total.detach())
            _, seconds = ledger.call('train_backward', total.backward, dict(context, combined_alignment_backward=eligible))
            result['backward_seconds'] += seconds
        result['FM'] += fm_value; result['total_loss'] += total_value
        result['sample_losses'].append(dict(sample_index=sample['sample_index'], alpha=sample['alpha'],
            FM_contribution=fm_value, raw_align=None if raw is None else float(raw.detach()),
            alignment_eligible=eligible, batch_free_denominator=denominator, **statistics))
        if eligible: del messages, tokens, selected, teacher, prediction, raw
        del velocity, fm, total, batch
    if result['aligned_samples'] != (4 if active else 0): raise AssertionError('Auxiliary eligible count differs')
    generators = list(model.parameters())
    if any(p.grad is None for p in generators): raise RuntimeError('Unused generator parameter')
    if head is not None and any((p.grad is None) != (not active) for p in head.parameters()):
        raise RuntimeError('Head gradients do not match active schedule')
    result['generator_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(generators, 1., error_if_nonfinite=True))
    result['aligner_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1., error_if_nonfinite=True)) if active else 0.
    _, result['optimizer_seconds'] = ledger.call('optimizer_update', optimizer.step,
        dict(step=step, alignment_active=active, expected_generator_adamw_step=step, expected_head_adamw_step=head_steps(segment, step)))
    if any(not bool(torch.isfinite(p).all()) for p in generators): raise FloatingPointError('Nonfinite generator after update')
    if head is not None and any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
        raise FloatingPointError('Nonfinite head after update')
    validate_optimizer(model, head, optimizer, segment, step)
    result.update(generator_adamw_step=step, head_adamw_step=head_steps(segment, step),
        update_seconds=time.perf_counter() - start, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


LOG_COLUMNS = ('segment', 'attempt_id', 'step', 'alignment_active', 'FM', 'raw_align', 'weighted_align', 'total_loss',
    'free_coordinate_denominator', 'alignment_denominator', 'aligned_samples', 'generator_gradient_norm', 'aligner_gradient_norm',
    'generator_adamw_step', 'head_adamw_step', 'shared_batch_sha256', 'input_batch_sha256', 'label_batch_sha256',
    'input_seconds', 'forward_seconds', 'aligner_forward_seconds', 'backward_seconds', 'optimizer_seconds', 'update_seconds',
    'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')
EXPOSURE_COLUMNS = ('segment', 'attempt_id', 'step', 'sample_index', 'uid', 'task_id', 'N', 'K', 'ratio', 'alpha', 't',
    'alignment_active', 'alignment_eligible', 'free_coordinates', 'shared_input_sha256', 'input_bytehash', 'label_bytehash', 'ot_cache_key', 'ot_cache_status')


def _seed_worker():
    random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_initialized(): torch.cuda.manual_seed_all(MODEL_SEED)


def train_segment(segment, official, data_manifest, stream_root, teacher_registration, out,
                  resume=None, fork=None, device='cuda', attempt_id='initial'):
    """Execute one registered physical segment. No internal retries."""
    from ..data.schedule40 import Schedule40Stream, Schedule40TargetStore
    if segment not in SEGMENTS or (resume is not None and fork is not None): raise ValueError('Invalid segment/recovery source')
    if not attempt_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in attempt_id):
        raise ValueError('Safe explicit attempt_id required')
    if not str(device).startswith('cuda'): raise ValueError('Training worker requires CUDA')
    torch.set_num_threads(2); runtime.configure_stable_runtime()
    out, data_manifest = Path(out).resolve(), Path(data_manifest).resolve()
    stream_root, teacher_registration = Path(stream_root).resolve(), Path(teacher_registration).resolve()
    if teacher_registration.is_dir(): teacher_registration = teacher_registration / 'teacher_manifest.json'
    folder = out / 'training' / segment
    attempt = folder / 'attempts' / attempt_id; attempt.mkdir(parents=True, exist_ok=False)
    status_path, events = folder / 'status.json', attempt / 'events.jsonl'
    ledger = CallLedger(attempt / 'calls.jsonl', segment, attempt_id)
    stream = Schedule40Stream(data_manifest.parent, stream_root, namespace=segment)
    paired_path = stream_root / 'paired_inputs.json'; paired = read_json(paired_path)
    if paired['plan_sha256'] != stream.plan_sha256: raise ValueError('Paired stream plan identity differs')
    expected_inputs = {r['sample_index']: r for r in paired['records']}
    identities = dict(data_manifest_sha256=digest(data_manifest), plan_sha256=stream.plan_sha256,
        paired_inputs_sha256=digest(paired_path), teacher_registration_sha256=digest(teacher_registration))
    targets = Schedule40TargetStore(teacher_registration.parent) if segment in ('EARLY', 'LATE') else None
    step, stop_step = SEGMENTS[segment]
    state = dict(status='LOADING', segment=segment, attempt_id=attempt_id, pid=os.getpid(),
        step=step, stop_step=stop_step, identities=identities, protocol_sha256=json_hash(PROTOCOL),
        checkpoint=None, physical_optimizer_updates_this_attempt=0,
        torch_peak_scope='reset before each effective update; 8 microbatch F/B and optimizer included; excludes context/desktop')
    atomic_json(status_path, state); started = time.perf_counter()
    signal_state = {'number': None}
    def interrupt(signum, frame): signal_state['number'] = int(signum)
    handlers = {s: signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGINT)}
    first_step = step
    try:
        _seed_worker()
        if resume or fork:
            source = resume or fork
            model, head, optimizer, payload = load_checkpoint(source, segment=segment, device=device, fork=fork is not None)
            if payload['identities'] != identities: raise ValueError('Resume data/stream/teacher identities differ')
            stream.load_state_dict(payload['stream_state']); step = payload['global_step']; first_step = step
            restore_rng(payload['rng']); lineage = copy.deepcopy(payload['lineage'])
            if fork:
                lineage.update(common_checkpoint_sha256=digest(source), common_generator_sha256=payload['generator_state_sha256'],
                    common_optimizer_sha256=payload['optimizer_state_sha256'], common_rng_sha256=payload['rng_sha256'],
                    fork_step=5000, fork_destination=segment, generator_optimizer_reset=False,
                    head_initial_sha256=None if head is None else state_hash(head.state_dict()))
            append_event(events, dict(event='FORK' if fork else 'RESUME', source=str(source), source_sha256=digest(source), step=step))
            # Release mmap/CPU optimizer copies once owned CUDA tensors are restored.
            del payload
        else:
            model, head, optimizer, lineage = initialize_official(official, segment, device)
            atomic_json(folder / 'initialization_audit.json', lineage); _seed_worker()
        if stream.batch_index != step: raise ValueError('Restored stream cursor differs')
        validate_optimizer(model, head, optimizer, segment, step)
        state.update(status='TRAINING', step=step, inherited_updates=first_step, lineage=lineage)
        atomic_json(status_path, state)
        def save(current, final=False):
            append_event(events, dict(event='CHECKPOINT_BEGIN', step=current))
            completed = dict(attempt_id=attempt_id, segment=segment, first_step=first_step,
                completed_updates=current-first_step, successful_calls=dict(ledger.counts), events_path=str(events))
            payload = _payload(model, head, optimizer, segment=segment, step=current, stream_state=stream.state_dict(),
                identities=identities, lineage=lineage, receipt=completed)
            receipt = _write_checkpoint(folder / 'latest.pt', payload); atomic_json(folder / 'latest.json', receipt)
            if final:
                immutable = out / 'checkpoints' / f'{segment}_step{current:04d}.pt'; immutable.parent.mkdir(parents=True, exist_ok=True)
                if immutable.exists(): raise FileExistsError('Final/prefix checkpoint already exists')
                os.link(folder / 'latest.pt', immutable)
                receipt = dict(receipt, path=str(immutable.resolve())); atomic_json(immutable.with_suffix('.json'), receipt)
            append_event(events, dict(event='CHECKPOINT_DONE', step=current, checkpoint=receipt))
            state['checkpoint'] = receipt
            return receipt
        with (attempt / 'training_log.csv').open('x', newline='', encoding='utf-8') as logfile, (attempt / 'exposure.csv').open('x', newline='', encoding='utf-8') as exposure:
            writer = csv.DictWriter(logfile, fieldnames=LOG_COLUMNS); writer.writeheader()
            ew = csv.DictWriter(exposure, fieldnames=EXPOSURE_COLUMNS); ew.writeheader()
            for current in range(step + 1, stop_step + 1):
                if signal_state['number'] is not None:
                    save(step); state.update(status='INTERRUPTED_SAFE', signal=signal_state['number']); break
                append_event(events, dict(event='INPUT_BEGIN', step=current)); tick = time.perf_counter()
                samples = stream.next_effective_batch(); input_seconds = time.perf_counter() - tick
                for sample in samples:
                    for key in ('shared_input_sha256', 'input_bytehash', 'label_bytehash', 'ot_cache_key', 'free_coordinate_count'):
                        if sample[key] != expected_inputs[sample['sample_index']][key]: raise ValueError('Paired input differs: ' + key)
                audit = stream.last_audit; append_event(events, dict(event='INPUT_DONE', step=current, **audit))
                for s in samples:
                    ew.writerow(dict(segment=segment, attempt_id=attempt_id, step=current, sample_index=s['sample_index'],
                        uid=s['object_id'], task_id=s['task_id'], N=int(s['y']), K=s['K'], ratio=s['ratio'], alpha=s['alpha'],
                        t=float(s['t']), alignment_active=active_alignment(segment, current),
                        alignment_eligible=active_alignment(segment, current) and s['alpha'] == 1., free_coordinates=s['free_coordinate_count'],
                        **{k: s[k] for k in ('shared_input_sha256', 'input_bytehash', 'label_bytehash', 'ot_cache_key', 'ot_cache_status')}))
                exposure.flush(); os.fsync(exposure.fileno())
                append_event(events, dict(event='UPDATE_BEGIN', step=current, shared_batch_sha256=audit['shared_batch_sha256']))
                result = update(model, head, optimizer, samples, targets, ledger, segment, current); step = current
                append_event(events, dict(event='UPDATE_DONE', step=step, generator_adamw_step=step,
                    head_adamw_step=result['head_adamw_step'], optimizer_updates=1, shared_batch_sha256=audit['shared_batch_sha256']))
                append_event(attempt / 'diagnostics.jsonl', dict(event='SAMPLE_LOSSES', step=step, values=result['sample_losses']))
                writer.writerow(dict(segment=segment, attempt_id=attempt_id, step=step, input_seconds=input_seconds,
                    **{k: audit[k] for k in ('shared_batch_sha256', 'input_batch_sha256', 'label_batch_sha256')},
                    **{k: result[k] for k in LOG_COLUMNS if k in result}))
                logfile.flush(); os.fsync(logfile.fileno()); free, total = torch.cuda.mem_get_info()
                state.update(step=step, physical_optimizer_updates_this_attempt=ledger.counts.get('optimizer_update', 0),
                    updates_this_attempt=ledger.counts.get('optimizer_update', 0),
                    last_FM=result['FM'], last_raw_align=result['raw_align'], alignment_active=result['alignment_active'],
                    update_seconds=result['update_seconds'], cuda_free_bytes=free, cuda_total_bytes=total,
                    cuda_peak_allocated_bytes=result['cuda_peak_allocated_bytes'], cuda_peak_reserved_bytes=result['cuda_peak_reserved_bytes'],
                    cuda_current_allocated_bytes=torch.cuda.memory_allocated(), cuda_current_reserved_bytes=torch.cuda.memory_reserved(),
                    successful_calls=dict(ledger.counts), wall_seconds=time.perf_counter()-started)
                if step % 100 == 0 or step == stop_step: save(step, final=step == stop_step)
                atomic_json(status_path, state)
                atomic_json(folder / 'progress.json', state)
                if step == first_step + 1 or step % 25 == 0:
                    print(segment, step, 'FM', result['FM'], 'align', result['raw_align'], flush=True)
            else: state['status'] = 'COMPLETE'
        if state['status'] == 'COMPLETE' and segment != 'COMMON':
            state['inference_export'] = export_generator(state['checkpoint']['path'], out / 'inference' / f'{segment}_generator_step6000.pt')
    except BaseException as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or 'CUDA out of memory' in str(error)
        state.update(status='OOM' if is_oom else 'BLOCKED', error=repr(error), traceback=traceback.format_exc(),
            step=step, stream_cursor=stream.batch_index, recoverable_under_controller_contract=is_oom,
            successful_calls=dict(ledger.counts), physical_optimizer_updates_this_attempt=ledger.counts.get('optimizer_update', 0))
        append_event(events, dict(event='TRAIN_FAILED', step=step, status=state['status'], error=repr(error), successful_calls=dict(ledger.counts)))
        atomic_json(attempt / 'failure.json', state)
        raise
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        state.update(wall_seconds=time.perf_counter()-started, step=step)
        atomic_json(attempt / 'final_status.json', state); atomic_json(status_path, state)
        atomic_json(folder / 'progress.json', state)
        if state['status'] == 'COMPLETE': atomic_json(folder / 'completed.json', state)
    return state
