"""CPU-testable state/ledger contract for a new Native branch from global18000."""
from __future__ import annotations
from meshflow_control.checkpoints import state_hash
from meshflow_control.training.context_checkpoint import _check_generator
from r2.pilots.bench_adaptation_strength_v1 import training as io
from r2.pilots.bench_full_seat_curriculum_v1.training_state import (
    validate_optimizer, validate_head_state, statistics, verify_sample, HEAD_CONFIG, HEAD_STATE_SHA256)
from .common import BASE_STEP, SOURCE_IDENTITIES, STAGE

SCHEMA = 'bench_seat_coverage_training_v1'
INFERENCE_SCHEMA = 'bench_seat_coverage_generator_v1'


def stage_spec(reg):
    updates = reg['updates']
    if type(updates) is not int or updates <= 0 or reg['source_global_step'] != BASE_STEP:
        raise ValueError('Finite branch budget / global start differs')
    full = reg['full_steps']
    if full != sorted(set(full)) or full[0] != 0 or full[-1] != updates or any(type(x) is not int or not 0 <= x <= updates for x in full):
        raise ValueError('Immutable checkpoints must include local0 and fixed endpoint')
    if reg['final_global_step'] != BASE_STEP + updates:
        raise ValueError('Global endpoint differs')
    return dict(updates=updates, full_steps=full, global_start=BASE_STEP)


def cursor(reg, step):
    spec = stage_spec(reg)
    if type(step) is not int or not 0 <= step <= spec['updates']:
        raise ValueError('Local cursor outside finite budget')
    return dict(stage=STAGE, stage_step=step, local_step=step, curriculum_step=step,
                global_step=BASE_STEP+step, native_total_step=6000+step, seam_total_step=0)


def limits(reg):
    updates = stage_spec(reg)['updates']
    return dict(train_forward=8*updates, train_backward=8*updates, optimizer_update=updates,
                head_forward=0, head_backward=0, head_optimizer_update=0, probe_forward=0)


def require_cursor(ledger, reg, step):
    cursor(reg, step)
    if ledger.pending or ledger.failed:
        raise RuntimeError('Pending/failed calls prohibit automatic replay')
    expected = dict(train_forward=8*step, train_backward=8*step, optimizer_update=step,
                    head_forward=0, head_backward=0, head_optimizer_update=0, probe_forward=0)
    if ledger.limits != limits(reg) or any(ledger.attempted[k] != n or ledger.successful[k] != n for k,n in expected.items()):
        raise RuntimeError('Exact saved cursor differs from attempted/completed calls')
    starts = [r for r in ledger.rows if r['event']=='CALL_BEGIN']
    for kind in ('train_forward','train_backward','optimizer_update'):
        rows = [r['context'] for r in starts if r['kind']==kind]
        wanted = [(s,m) for s in range(1,step+1) for m in (range(8) if kind!='optimizer_update' else (None,))]
        if [(r['stage_step'],r.get('microbatch')) for r in rows] != wanted or any(r['stage'] != STAGE or r['global_step'] != BASE_STEP+r['stage_step'] for r in rows):
            raise RuntimeError('Exact update/microbatch/global sequence differs')


def validate_payload(p, reg, step, reg_sha):
    if any(p.get(k)!=v for k,v in cursor(reg,step).items()):
        raise ValueError('Saved state cursor differs')
    if (p.get('schema'),p.get('arm'),p.get('registration_sha256'),p.get('EMA'),p.get('update_in_progress')) != (SCHEMA,STAGE,reg_sha,False,False):
        raise ValueError('Saved state identity differs')
    if p['stream_state'] != dict(batch_index=step,stage_batch_index=step,next_sample_index=8*step):
        raise ValueError('Saved new-branch stream cursor differs')
    for key,hkey,fn in (('generator_state','generator_state_sha256',state_hash),('optimizer','optimizer_state_sha256',io.base.tree_hash),('rng','rng_sha256',io.base.tree_hash)):
        if fn(p[key]) != p[hkey]:
            raise ValueError('Checkpoint content/hash differs: '+key)
    _check_generator(p['generator_state'])
    validate_head_state(p['head_state'],p['head_state_sha256'])
    if p['head_optimizer'] is not None or p['head_step'] != 0 or p['head_calls'] != 0 or p['head_config'] != HEAD_CONFIG:
        raise ValueError('Unused retained head must have zero training activity')
    if p['model_config'] != reg['model_config'] or p['source_global_step'] != BASE_STEP:
        raise ValueError('Model/source identity changed')
