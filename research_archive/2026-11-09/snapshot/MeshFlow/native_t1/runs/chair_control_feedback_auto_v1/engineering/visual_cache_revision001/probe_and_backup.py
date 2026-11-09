"""Bounded read-only cache-identity probe; stops before any render dispatch."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'native_t1/runs/chair_control_feedback_auto_v1'
SOURCE = ROOT / 'experiments/mf_h1_local0/chair_control_feedback_auto_v1/visualization.py'
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def differences(a, b, p='$'):
    result = []
    if type(a) is not type(b):
        result.append(dict(path=p, saved_type=type(a).__name__, live_type=type(b).__name__,
                           saved=a, live=b))
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(a.keys() | b.keys()):
            if k not in a or k not in b:
                result.append(dict(path=p+'.'+k, missing_side='saved' if k not in a else 'live'))
            else:
                result.extend(differences(a[k], b[k], p+'.'+k))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            result.append(dict(path=p, saved_length=len(a), live_length=len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                result.extend(differences(x, y, p+f'[{i}]'))
    elif a != b:
        result.append(dict(path=p, saved=a, live=b))
    return result


def main():
    backup = HERE/'original/visualization.py'
    backup.parent.mkdir(exist_ok=True)
    assert not backup.exists()
    shutil.copy2(SOURCE, backup)
    from experiments.mf_h1_local0.chair_control_feedback_auto_v1 import visualization as module
    module._renderer = lambda: (_ for _ in ()).throw(RuntimeError('Forbidden render in identity probe'))
    captured = {}
    class Captured(Exception):
        pass
    def trace(frame, event, arg):
        if event == 'line' and frame.f_code is module.generate.__code__ and frame.f_lineno == 196:
            captured.update(frame.f_locals)
            raise Captured()
        return trace
    sys.settrace(trace)
    try:
        module.generate(OUT, ['START', 'explore_FM_CONTROL_500',
                             'explore_REPLAY_FM_CONTROL_500', 'explore_TRAJECTORY_REPLAY_500'])
    except Captured:
        pass
    finally:
        sys.settrace(None)
    assert captured
    saved, live = captured['saved']['identity'], captured['identity']
    diffs = differences(saved, live)
    normalized_equal = saved == json.loads(json.dumps(live, sort_keys=True, allow_nan=False))
    assert normalized_equal and len(diffs) == 5
    assert {d['path'] for d in diffs} == {f'$.views[{i}]' for i in range(5)}
    result = dict(status='CAUSE_CONFIRMED', source_path=str(SOURCE),
                  before_sha256=sha(SOURCE), backup_path=str(backup), backup_sha256=sha(backup),
                  original_cache_manifest=str(captured['dest']/'manifest.json'),
                  original_cache_manifest_sha256=sha(captured['dest']/'manifest.json'),
                  literal_python_equality=saved == live,
                  canonical_json_identity_equal=normalized_equal,
                  canonical_hash_saved=module._hash(saved), canonical_hash_live=module._hash(live),
                  exact_differences=diffs, render_calls=0, model_calls=0,
                  new_metric_calls=0, CONFIRM_read=False, probe_script_sha256=sha(__file__))
    (HERE/'cause.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
