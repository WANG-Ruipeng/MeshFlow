"""Create isolated single-arm wrappers from the completed fixed-panel experiment."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'r2/pilots/bench_state_schedule_v1'
DST = ROOT / 'r2/pilots/bench_coverage_expansion_v1'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def replace(s, old, new, count=1):
    actual = s.count(old)
    if actual != count:
        raise ValueError((old, actual, count))
    return s.replace(old, new)

def main():
    records = []
    for name in ('eval_plan.py', 'sample.py', 'post_d0.py', 'evaluation.py', 'collect.py', 'gallery.py'):
        s = (SRC / name).read_text(encoding='utf-8')
        if name == 'eval_plan.py':
            s = replace(s, 'LABELS = ("PERTURBED_STATE_S1500", "ASYNC_LEAD_SEAM_S1500")', 'LABELS = ("COVERAGE_S1500",)')
            s = replace(s, 'ARM_BY_LABEL = {\n    "PERTURBED_STATE_S1500": "PERTURBED_STATE",\n    "ASYNC_LEAD_SEAM_S1500": "ASYNC_LEAD_SEAM",\n}', 'ARM_BY_LABEL = {"COVERAGE_S1500": "COVERAGE_SEAM"}')
            s = s.replace('Exactly two newly generated endpoint labels required', 'Exactly one registered coverage endpoint required')
            s = replace(s, 'if len(receipts) != 96:', 'if len(receipts) != 48:')
            s = s.replace('All 96 new RAW', 'All 48 new RAW')
            s = s.replace('new_outputs=96', 'new_outputs=48')
            s = s.replace('two new final labels', 'one new final label')
        elif name == 'sample.py':
            s = replace(s, '    if spec["arm"] == "ASYNC_LEAD_SEAM":\n        from . import async_core\n        code.append(Path(async_core.__file__))\n', '')
            s = replace(s, '            if spec["arm"] == "PERTURBED_STATE":\n                raw, trajectory, audit = S.rollout(\n                    model, lay, arrays(q["noise_npz"])["z"], "SEAM", dest / "calls.jsonl")\n            else:\n                from .async_core import rollout\n                raw, trajectory, audit = rollout(\n                    model, lay, arrays(q["noise_npz"])["z"], "SEAM", dest / "calls.jsonl")',
                '            raw, trajectory, audit = S.rollout(\n                model, lay, arrays(q["noise_npz"])["z"], "SEAM", dest / "calls.jsonl")')
            s = s.replace('Two registered endpoint samplers', 'One registered coverage endpoint sampler')
        elif name == 'post_d0.py':
            s = s.replace('all 96 new completed RAW', 'all 48 new completed RAW')
        elif name == 'collect.py':
            s = s.replace('96 new outputs plus 48', '48 new outputs plus 48')
            s = s.replace('len(planned) != 432', 'len(planned) != 288')
            s = s.replace('Expected 3 models x 48', 'Expected 2 models x 48')
            s = s.replace('expected_rows=432', 'expected_rows=288')
            s = replace(s, 'rows=432, new_outputs=96, reused_outputs=48, new_score_rows=288, reused_score_rows=144',
                        'rows=288, new_outputs=48, reused_outputs=48, new_score_rows=144, reused_score_rows=144')
            s = s.replace('BENCH_STATE_SCHEDULE_COLLECTION_V1', 'BENCH_COVERAGE_EXPANSION_COLLECTION_V1')
        elif name == 'gallery.py':
            s = s.replace('Bench state/schedule: historical A and two fixed endpoints',
                          'Bench coverage: historical A and the fixed coverage endpoint')
        target = DST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='\n') as f:
            f.write(s)
        records.append(dict(source=str(SRC / name), source_sha256=digest(SRC / name),
                            target=str(target), target_sha256=digest(target)))
    with (Path(__file__).parent / 'evaluation_wrapper_sources.json').open('x', encoding='utf-8') as f:
        json.dump(dict(status='CREATED', rows=records, scientific_scoring_changed=False,
                       sampler='ordinary inherited Euler50;48fixedrequests;singleD0'), f, indent=2)
    print(json.dumps(dict(status='CREATED', files=len(records))))

if __name__=='__main__':
    main()
