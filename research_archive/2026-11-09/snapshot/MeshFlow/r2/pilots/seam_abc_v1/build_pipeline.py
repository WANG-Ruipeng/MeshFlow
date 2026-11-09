"""Copy reviewed runners into this experiment, retaining original sources."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'seam_quality_four_v1'

def main():
    origins = {}
    for name in ('eval_common.py', 'generate.py', 'postprocess.py', 'evaluate.py'):
        p = OLD / name
        s = p.read_text(encoding='utf-8')
        origins[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        s = s.replace('seam_quality_four', 'seam_abc')
        s = s.replace("('LONG','ANTI_COVER','LOCAL_NOISE')", "('LONG','A_COUPLE','B_POOL47','C_SHARED_TIP')")
        s = s.replace("('LONG', 'ANTI_COVER', 'LOCAL_NOISE')", "('LONG', 'A_COUPLE', 'B_POOL47', 'C_SHARED_TIP')")
        if name == 'generate.py':
            s = s.replace("len(requests) != 16 or len({q['request_id'] for q in requests}) != 16", "len(requests) != 28 or len({q['request_id'] for q in requests}) != 28")
            s = s.replace('Exactly sixteen fixed request identities required', 'Exactly twenty-eight fixed request identities required')
            s = s.replace('len(conditions) != 4 or', 'len(conditions) != 7 or')
            s = s.replace('Four conditions with four noises each required', 'Seven conditions with four noises each required')
            s = s.replace("    path = root / 'inference' / (arm + '_generator.pt')", "    modelroot = REPOSITORY / 'experiments/r2_seam_quality_four_v1' if arm == 'LONG' else root\n    path = modelroot / 'inference' / (arm + '_generator.pt')")
            s = s.replace("payload['schema'] != 'r2_seam_abc_generator_v1'", "payload['schema'] != ('r2_seam_quality_four_generator_v1' if arm == 'LONG' else 'r2_seam_abc_generator_v1')")
            s = s.replace("        hp = root / 'inference' / (arm + '_occupancy.pt');", "        modelroot = REPOSITORY / 'experiments/r2_seam_quality_four_v1' if arm == 'LONG' else root\n        hp = modelroot / 'inference' / (arm + '_occupancy.pt');")
            s = s.replace("payload['schema'] != 'r2_seam_abc_occupancy_v1'", "payload['schema'] != ('r2_seam_quality_four_occupancy_v1' if arm == 'LONG' else 'r2_seam_abc_occupancy_v1')")
            begin = s.index('            # Common saved Gaussian; only occupied third-corner mean changes.')
            end = s.index('            raw, trajectory, audit = rollout', begin)
            s = s[:begin] + s[end:]
            line = "            audit['noise_policy'] = ('saved standard Gaussian plus C edge midpoint at occupied seam third corners; sigma=1; bulk unchanged' if arm == 'LOCAL_NOISE' else 'saved standard Gaussian; no HYBRID sampling noise')"
            s = s.replace(line, "            audit['noise_policy'] = 'saved standard Gaussian; C arm applies predicted shared-tip references; no local mean/scale change'")
            s = s.replace('outputs=16,', 'outputs=len(manifest[\'requests\']),')
        elif name == 'postprocess.py':
            s = s.replace('All 48 RAW receipts', 'All 112 RAW receipts (16 reused)')
            s = s.replace("result['outputs'] != 16", "result['outputs'] != len(manifest['requests'])")
            s = s.replace('all sixteen RAW outputs', 'all twenty-eight RAW outputs')
            s = s.replace('len(records) != 48', "len(records) != len(G.ARMS)*len(manifest['requests'])")
            s = s.replace('RAW=48,', "RAW=len(barrier),")
            s = s.replace('outputs=16,', "outputs=len(manifest['requests']),")
        elif name == 'evaluate.py':
            s = s.replace("len(value['conditions']) != 4 or len(value['requests']) != 16", "len(value['conditions']) != 7 or len(value['requests']) != 28")
            s = s.replace('four TRAIN C, sixteen requests and three registered arms', 'seven C, twenty-eight requests and four registered arms')
            s = s.replace("len({q['request_id'] for q in value['requests']}) != 16", "len({q['request_id'] for q in value['requests']}) != 28")
            s = s.replace("        if condition['split'] != 'TRAIN':\n            raise ValueError('This is a TRAIN-only local pilot')", "        if condition['split'] not in ('TRAIN', 'DEV'):\n            raise ValueError('Unregistered split')")
            s = s.replace('cells=96, registered_requests=16,', 'cells=224, registered_requests=28,')
            s = s.replace('registered=96', 'registered=224').replace('cells=96', 'cells=224')
            s = s.replace("occupancy='Four C-only layouts, not sixteen independent trials. Compare to all-positive 62/66, positive recall 1, Null recall 0, layouts 2/4.'", "occupancy='Seven C-only layouts per arm, not 28 independent trials; owner-slot labels separately audited; no source layout used in inference.'")
        target = HERE / name
        if target.exists():
            raise FileExistsError(target)
        target.write_text(s, encoding='utf-8', newline='\n')
    (HERE / 'adapter_origin.json').write_text(json.dumps(origins, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    main()
