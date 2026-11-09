"""Reference the exact completed W stream; precompute count-only loss gains.

No samples, noise, times, augmentations or OT are generated here. Narrow B is
used only for its registered free-coordinate counts, never predictions/losses.
"""
from __future__ import annotations
import json
from . import common as C
from r2.pilots.bench_high_joint_coverage_v1 import stream as inherited

ARMS = C.ARMS
DEFAULT_SEED = inherited.DEFAULT_SEED
UPDATES = C.UPDATES
SCHEMA = 'BENCH_GROUP_WEIGHT_MATCH_COEFFICIENTS_V1'


def validate_manifest(manifest):
    return inherited.validate_manifest(manifest)


def validate(schedule, train, updates=UPDATES, arm='W_MATCHED'):
    if arm not in ARMS or updates != UPDATES:
        raise ValueError('Only the registered W_MATCHED1500 branch')
    # The referenced JSON keeps its original arm and exact original bytes.
    return inherited.validate(schedule, train, updates, 'WIDE_HIGH')


def validate_frozen_narrow(schedule, train):
    return inherited.validate_frozen_narrow(schedule, train)


def _scalar_counts(entry, train):
    by_id = {r['condition_id']:r for r in train}
    result = []
    for row in entry['samples']:
        source = by_id[row['condition_id']]
        if source['parent_id'] != row['parent_id'] or source['replay_group'] != row['replay_group']:
            raise ValueError('Stream/source group identity differs')
        count = 9 * (int(source['N']) - int(source['K']))
        if count <= 0:raise ValueError('No effective free scalar coordinates')
        result.append(count)
    return result


def build_coefficients(wide, train, narrow, narrow_train):
    paired = inherited.validate_narrow(wide, train, narrow, narrow_train)
    result = []
    for ew, eb in zip(wide['steps'], narrow['steps']):
        groups = [r['replay_group'] for r in ew['samples']]
        cw, cb = _scalar_counts(ew, train), _scalar_counts(eb, narrow_train)
        sums = lambda counts,g:sum(c for c,x in zip(counts,groups) if x==g)
        mo, mf, hw, hb = sums(cw,'OLD'),sums(cw,'FOCUS'),sums(cw,'HIGH'),sums(cb,'HIGH')
        if mo != sums(cb,'OLD') or mf != sums(cb,'FOCUS'):
            raise ValueError('OLD/FOCUS free-coordinate counts changed')
        branch = 'HIGH' if hw else 'FOCUS'
        dw, db = sum(cw),sum(cb)
        if branch == 'HIGH':
            if mf or not hb or groups != ['OLD']*6+['HIGH']*2:
                raise ValueError('Expected six OLD plus two HIGH microbatches')
            rho = hb/hw
            base_gain = dw/db
            gains = [base_gain*(rho if g=='HIGH' else 1.) for g in groups]
        else:
            if hb or dw != db or cw != cb or groups != ['OLD']*6+['FOCUS']*2:
                raise ValueError('FOCUS update must preserve the exact original objective')
            rho, base_gain, gains = None,1.,[1.]*8
        result.append(dict(stage_step=ew['stage_step'],branch=branch,
            m_old=mo,m_focus=mf,m_high_B=hb,m_high_W=hw,D_B=db,D_W=dw,
            rho=rho,base_gain=base_gain,micro_gains=gains,micro_groups=groups,
            micro_free_scalar_counts=cw,reference_micro_free_scalar_counts=cb))
    return dict(schema=SCHEMA,status='FROZEN_COUNT_ONLY',updates=UPDATES,arm='W_MATCHED',
        original_schedule_arm='WIDE_HIGH',loss_normalization=C.LOSS_NORMALIZATION,
        formula='(SSE_OLD + (m_HIGH_B/m_HIGH_W)*SSE_HIGH)/(m_OLD+m_HIGH_B); FOCUS unchanged',
        gain_implementation='original SSE_i/D_W multiplied by D_W/D_B, and additionally rho only for HIGH',
        count_units='valid nonfixed scalar coordinates; Native9*(N-K)',
        semantics='Matches group coordinate-MSE coefficients, not observed loss or gradient shares',
        paired_stream=paired,steps=result)


def prepare():
    paths = dict(manifest=C.MANIFEST,wide_schedule=C.SCHEDULE,
                 narrow_manifest=C.NARROW_MANIFEST,narrow_schedule=C.NARROW_SCHEDULE)
    hashes = {str(p):C.sha(p) for p in paths.values()}
    m,w,bm,b = (C.read(paths[k]) for k in ('manifest','wide_schedule','narrow_manifest','narrow_schedule'))
    validate_manifest(m);validate(w,m['train'])
    result = build_coefficients(w,m['train'],b,bm['train'])
    result['referenced_inputs']=hashes
    C.write_new(C.COEFFICIENTS,result)
    receipt = dict(status='PASS',schema='bench_group_weight_match_stream_gate_v1',
        referenced_inputs=hashes,coefficient_path=str(C.COEFFICIENTS),
        coefficient_sha256=C.sha(C.COEFFICIENTS),steps=UPDATES,
        focus_steps=sum(s['branch']=='FOCUS' for s in result['steps']),
        high_steps=sum(s['branch']=='HIGH' for s in result['steps']),
        exact_original_W_schedule=True,original_W_schedule_sha256=C.sha(C.SCHEDULE),
        original_W_manifest_sha256=C.sha(C.MANIFEST),paired_stream=result['paired_stream'],
        code_sha256=C.sha(__file__),model_calls=0,OT_calls=0,new_noise=0,
        source_weight_loads=0,training_budget_authorized_by_this_file=False)
    C.write_new(C.OUT/'data/stream_gate.json',receipt)
    return receipt


def checked_coefficients(path, schedule, manifest):
    result=C.read(path)
    expected=build_coefficients(schedule,manifest['train'],C.read(C.NARROW_SCHEDULE),C.read(C.NARROW_MANIFEST)['train'])
    expected['referenced_inputs']={str(p):C.sha(p) for p in (C.MANIFEST,C.SCHEDULE,C.NARROW_MANIFEST,C.NARROW_SCHEDULE)}
    if result != expected:raise ValueError('Frozen coefficient rows or original references changed')
    return result['steps']


if __name__=='__main__':
    r=prepare();print(json.dumps({k:r[k] for k in ('status','steps','focus_steps','high_steps','OT_calls')}))
