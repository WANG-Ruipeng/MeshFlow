"""One-variable joint C/Y translation on the frozen RELATION sample stream.

The standard Gaussian and HYBRID epsilon are unchanged. Production Euclidean
OT is recomputed from the translated half-scale target on every formal sample.
Source arrays remain read-only; all runtime model-coordinate duplicates agree.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from r2.pilots.bench_relation_training_v1 import data as reference

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/r2_bench_joint_shift_v1'
REFERENCE = ROOT/'experiments/r2_bench_relation_training_v1'
SEED, UPDATES = reference.SEED, reference.UPDATES
L = float(np.sqrt(3.)*2./.3762)
TRANSLATION_AXIS_FRACTION = .04
TRANSLATION_AXIS_MAX = TRANSLATION_AXIS_FRACTION*L
TRANSLATION_DOMAIN = 'R2_BENCH_JOINT_SHIFT_V1_TRANSLATION_ONLY'
TRANSLATION_KEYS = frozenset(('translation_seed', 'translation_xyz'))
resolve, read, sha, arrays, load_case, write_new = (
    reference.resolve, reference.read, reference.sha, reference.arrays,
    reference.load_case, reference.write_new)


def translation_fields(step, row):
    """Independent RNG: never consume parent, scale, Gaussian, epsilon or t RNG."""
    token = f'{TRANSLATION_DOMAIN}|{int(step)}|{int(row["sample_index"])}|{row["condition_id"]}'
    seed = int.from_bytes(hashlib.sha256(token.encode('utf-8')).digest()[:4], 'little')
    shift = ([0., 0., 0.] if row['augmentation'] == 'CLEAN' else
             list(map(float, np.random.RandomState(seed).uniform(
                 -TRANSLATION_AXIS_MAX, TRANSLATION_AXIS_MAX, 3))))
    return dict(translation_seed=seed, translation_xyz=shift)


def transformed_case(case, scale_xyz, translation_xyz):
    scale = np.asarray(scale_xyz, np.float32)
    shift = np.asarray(translation_xyz, np.float32)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid positive XYZ scale')
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError('Invalid translation')
    if np.max(np.abs(shift.astype(np.float64))) > TRANSLATION_AXIS_MAX + 1e-7:
        raise ValueError('Translation exceeds frozen per-axis model-unit bound')
    out = {k: v.copy() for k, v in case.items()}
    # Apply the same FP32 expression to the shared model vertex array once.
    # Preserve all source identities/corner permutations, including winding.
    out['model_vertices'] = np.asarray(case['model_vertices'], np.float32)*scale
    if np.any(shift != 0):
        out['model_vertices'] = out['model_vertices'] + shift
    out['full_target'] = out['model_vertices'][out['source_face_vertex_ids']]
    out['pre_ot_full'] = out['full_target']*np.float32(.5)
    out['C_vertices'] = out['model_vertices'][out['C_source_vertex_ids']]
    for name, ids in (('C', out['known_ids']), ('Y', out['free_ids'])):
        out[name] = out['full_target'][ids]
        out[name+'_pre_ot'] = out['pre_ot_full'][ids]
    # source_vertices/normalized_vertices/transform_json remain provenance, not
    # runtime model coordinates. Explicitly record the second affine transform.
    out['runtime_augmentation_json'] = np.asarray(json.dumps(dict(
        schema='MODEL_SPACE_XYZ_THEN_JOINT_SHIFT_V1', scale_xyz=scale.tolist(),
        translation_xyz=shift.tolist(), translation_unit='model', fixed_L=L,
        source_normalization_fields='unchanged source provenance',
        model_coordinate_fields=['model_vertices','full_target','C','Y','C_vertices'],
        no_recenter_no_clamp=True), sort_keys=True))
    return out


def scaled_case(case, scale_xyz):
    """Compatibility helper; zero shift retains production inputs bit-for-bit."""
    return transformed_case(case, scale_xyz, [0., 0., 0.])


def sample_for_training(case, row):
    prepared = transformed_case(case, row['scale_xyz'], row['translation_xyz'])
    n, k = int(prepared['N']), int(prepared['K'])
    z = np.random.RandomState(int(row['noise_seed'])).standard_normal((n-k,3,3)).astype(np.float32)
    eps = np.random.RandomState(int(row['epsilon_seed'])).standard_normal(z.shape).astype(np.float32)
    normal = float(np.random.RandomState(int(row['time_seed'])).standard_normal())
    t = float(1/(1+np.exp(-normal)))
    return reference.old.supervised_sample(prepared, z, eps, t, int(row['permutation_seed']), 'SEAM')


def sample_for_probe(case, z, eps, t, permutation_seed):
    return reference.old.supervised_sample(case, z, eps, t, permutation_seed, 'SEAM')


def make_schedule(train):
    old_manifest = read(REFERENCE/'data/manifest.json')
    if train != old_manifest['train']:
        raise ValueError('Joint-shift must use exactly the frozen RELATION training rows/order')
    schedule = copy.deepcopy(read(REFERENCE/'data/schedule.json'))
    for entry in schedule['steps']:
        for row in entry['samples']:
            row.update(translation_fields(entry['step'], row))
    schedule.update(schema='BENCH_JOINT_SHIFT_PARENT_PART_XYZ_STREAM_V1',
        reference_schedule=str(REFERENCE/'data/schedule.json'),
        reference_schedule_sha256=sha(REFERENCE/'data/schedule.json'),
        translation=dict(fixed_L=L, per_axis_fraction=TRANSLATION_AXIS_FRACTION,
            per_axis_model_bound=TRANSLATION_AXIS_MAX,
            vector_norm_max=float(np.sqrt(3.)*TRANSLATION_AXIS_MAX),
            independent_seed_domain=TRANSLATION_DOMAIN,
            policy='CLEAN:zero; ANISOTROPIC:independent uniform per axis after same XYZ scale; standard Gaussian and epsilon unchanged'),
        policy='Exact RELATION stream plus only joint target/C translation on second half; production OT recomputed')
    validate_schedule(schedule['steps'])
    return schedule


def validate_schedule(steps):
    old_steps = read(REFERENCE/'data/schedule.json')['steps']
    if len(steps) != UPDATES or len(old_steps) != UPDATES:
        raise ValueError('Expected exactly 1500 paired updates')
    zero = shifted = 0
    for new, old in zip(steps, old_steps):
        if {k:v for k,v in new.items() if k != 'samples'} != {k:v for k,v in old.items() if k != 'samples'}:
            raise ValueError('Original step fields changed')
        if len(new['samples']) != 8:
            raise ValueError('Expected batch8')
        for row, original in zip(new['samples'], old['samples']):
            if set(row) != set(original)|TRANSLATION_KEYS:
                raise ValueError('Unexpected new/missing schedule fields')
            if {k:v for k,v in row.items() if k not in TRANSLATION_KEYS} != original:
                raise ValueError('Original parent/condition/scale/RNG stream changed')
            if any(row[k] != v for k,v in translation_fields(new['step'], original).items()):
                raise ValueError('Translation not reproducible from its independent seed')
            t = np.asarray(row['translation_xyz'], np.float64)
            if t.shape != (3,) or not np.isfinite(t).all() or np.max(np.abs(t)) > TRANSLATION_AXIS_MAX:
                raise ValueError('Translation bounds invalid')
            if original['augmentation'] == 'CLEAN':
                if np.any(t != 0) or original['scale_xyz'] != [1.,1.,1.]:
                    raise ValueError('Identity half changed')
                zero += 1
            elif original['augmentation'] == 'ANISOTROPIC':
                shifted += 1
            else:
                raise ValueError('Unknown original augmentation')
    if (zero, shifted) != (6000, 6000):
        raise ValueError('Exactly 6000 identity + 6000 translated instances required')
    return dict(status='PASS', original_sample_fields_identical=12000,
        identity_samples=zero, translated_samples=shifted,
        independent_translation_stream=True, no_original_RNG_consumption=True,
        reference_schedule_sha256=sha(REFERENCE/'data/schedule.json'))
