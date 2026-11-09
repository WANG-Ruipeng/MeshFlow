"""CPU semantic cache checks, precise diff, and unchanged old-gallery receipt."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT/'native_t1/runs/chair_control_feedback_auto_v1'
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'experiments/mf_h1_local0/chair_control_feedback_auto_v1/visualization.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cause = json.loads((HERE/'cause.json').read_text())
    before = (HERE/'original/visualization.py').read_text()
    after = SOURCE.read_text()
    expected = before.replace("assert saved['identity']==identity", "assert saved['identity']==json.loads(json.dumps(identity,sort_keys=True,allow_nan=False))")
    assert after == expected, 'Exactly one assertion expression may change'
    tree = ast.parse(after)
    checks = [n for n in ast.walk(tree) if isinstance(n, ast.Assert) and "saved['identity']" in ast.unparse(n.test)]
    assert len(checks) == 1
    expression = compile(ast.Expression(checks[0].test), '<actual-fixed-cache-expression>', 'eval')
    original_manifest = Path(cause['original_cache_manifest'])
    saved = json.loads(original_manifest.read_text())
    identity = copy.deepcopy(saved['identity'])
    identity['views'] = [tuple(v) for v in identity['views']]
    def test(value):
        return eval(expression, {'json': json}, dict(saved=saved, identity=value))
    assert test(identity)
    cases = [('JSON_list_vs_tuple_equivalent', True)]
    for field in ['labels', 'views', 'source_files', 'availability', 'panel_size', 'split']:
        wrong = copy.deepcopy(identity)
        if field == 'labels': wrong[field][0] = 'changed'
        elif field == 'views': wrong[field][0] = ('front_3q', 46., 22.)
        elif field in ('source_files', 'availability'): wrong[field][next(iter(wrong[field]))] = 'changed'
        elif field == 'panel_size': wrong[field] = 401
        elif field == 'split': wrong[field] = 'changed'
        assert not test(wrong), field
        cases.append((field+'_mismatch_rejected', True))
    assert sha(original_manifest) == cause['original_cache_manifest_sha256']
    reviewed = OUT/'visuals/reviews/explore_TRAJECTORY_REPLAY_500_vs_explore_REPLAY_FM_CONTROL_500.json'
    assert sha(reviewed) == '89f1751105c054ea61992e604c03fd737f7604eafe4b04e01e6481bb04a0f2d8'
    diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='original/visualization.py', tofile='visualization.py'))
    (HERE/'code.diff').write_text(diff)
    receipt = dict(status='PASS', reason='Only five VIEWS tuple/list type differences existed after JSON round-trip; all values and the full canonical identity hash were equal.',
                   scientific_changes=False, changed_behavior='Validate saved cache identity using JSON semantics consistent with the existing identity hash.',
                   before_sha256=cause['before_sha256'], after_sha256=sha(SOURCE),
                   exact_one_assertion_changed=True, tests=cases,
                   preserved_old_gallery_manifest=dict(path=str(original_manifest), sha256=sha(original_manifest)),
                   preserved_complete_500_review=dict(path=str(reviewed), sha256=sha(reviewed)),
                   diff_path=str(HERE/'code.diff'), diff_sha256=sha(HERE/'code.diff'),
                   verifier_source_sha256=sha(__file__),
                   model_calls=0, render_calls=0, new_metric_calls=0, CONFIRM_read=False,
                   next_gallery='A new combined1000 directory includes the changed visualization source hash. No 500 rerender is required or performed.')
    (HERE/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
