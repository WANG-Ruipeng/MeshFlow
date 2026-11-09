"""Dormant H3 helper; does not register a recipe or dispatch model work.

The cap limits the detached auxiliary LOSS value, not its gradient norm.
Inactive multiplication must remain exactly ``.1 * raw`` in the trainer.
"""
import math
from pathlib import Path
from .common import read_json, digest, START_SHA

ARM = 'TRAJECTORY_REPLAY_CAPPED'


def coefficient(primary_fm, auxiliary_mse):
    primary_fm, auxiliary_mse = float(primary_fm), float(auxiliary_mse)
    if not (math.isfinite(primary_fm) and math.isfinite(auxiliary_mse)):
        raise FloatingPointError('Nonfinite loss balancing inputs')
    if primary_fm < 0 or auxiliary_mse < 0:
        raise ValueError('Negative MSE input')
    denominator = max(auxiliary_mse, 1e-30)
    ratio = primary_fm / denominator
    weight = min(.1, ratio)
    return weight, dict(primary_fm_python_sum=primary_fm,
        auxiliary_mse_fp32_to_python=auxiliary_mse, denominator=denominator,
        detached_ratio=ratio, effective_weight=weight,
        effective_weight_hex=weight.hex(), cap_active=weight < .1,
        loss_value_cap_only=True)


def frozen_recipe(out):
    out = Path(out)
    path = out/'decisions/H3_recipe.json'
    recipe = read_json(path)
    assert recipe['status'] == 'FROZEN' and recipe['arm'] == ARM
    assert recipe['START_sha256'] == START_SHA
    assert recipe['parent_protocol_sha256'] == digest(out/'protocol.json')
    assert recipe['feedback_pool_sha256'] == digest(out/'feedback/pool_v1/manifest.json')
    proof = recipe['exploration_matched_control_equivalence']
    assert digest(Path(proof['path'])) == proof['sha256']
    receipt = read_json(Path(proof['path']))
    assert receipt['status'] == 'PASS_CAP_INACTIVE_1000'
    assert receipt['completed_teacher_steps'] == receipt['safe_inactive_steps'] == 1000
    assert not receipt['cap_active_steps'] and not receipt['boundary_inconclusive_steps']
    assert receipt['minimum_positive_margin'] > 0
    assert receipt['actual_marker_complete'] and not receipt['errors'] and not receipt['missing']
    side_path = out/'checkpoints/explore_REPLAY_FM_CONTROL_1000.json'
    for evidence in (out/'protocol.json', out/'feedback/pool_v1/manifest.json',
                     out/'streams/explore/train_plan.json', side_path):
        assert receipt['source_and_evidence_hashes'][str(evidence)] == digest(evidence)
    side = read_json(side_path)
    assert Path(receipt['endpoint']['path']).resolve() == (out/'checkpoints/explore_REPLAY_FM_CONTROL_1000.pt').resolve()
    assert receipt['endpoint']['checkpoint_sha256'] == side['checkpoint_sha256']
    assert side['metadata']['source_checkpoint_sha256'] == START_SHA
    for key in ('model_state_sha256', 'optimizer_state_sha256', 'rng_sha256', 'stream_state_sha256'):
        assert receipt['endpoint'][key] == side['metadata'][key]
    assert recipe['coefficient_formula'] == 'min(0.1, detached_python_FM_sum / max(detached_FP32_aux_MSE, 1e-30))'
    assert recipe['primary_stream_arm_alias'] == 'TRAJECTORY_REPLAY'
    assert recipe['auxiliary_arm_alias'] == 'TRAJECTORY_REPLAY'
    assert recipe['loss_balance_source_sha256'] == digest(Path(__file__))
    for name, expected in recipe['sources'].items():
        source = Path(__file__).with_name(name)
        assert source.parent == Path(__file__).parent and digest(source) == expected
    return recipe, digest(path)
