"""Create independent endpoint adapters from the preserved, tested C200 runners.

This maintenance script never calls a model. Resulting source files are frozen
before execution; the original runners and installed SEAM core are read-only.
"""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'seam_slot_c200_v1'


def replace_once(s, old, new):
    if s.count(old) != 1:
        raise ValueError((old, s.count(old)))
    return s.replace(old, new)


def main():
    provenance = {}
    for name in ('common.py', 'generate.py', 'postprocess.py', 'evaluate.py'):
        p = OLD / name
        s = p.read_text(encoding='utf-8')
        provenance[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        if name == 'common.py':
            s = s.replace("OUT=ROOT/'experiments/r2_seam_slot_c200_v1'", "OUT=ROOT/'experiments/r2_seam_quality_four_v1'")
            s = s.replace("ARMS=('FULL4_C200','SEAM_SLOT_C200')", "ARMS=('LONG','ANTI_COVER','LOCAL_NOISE')")
            target = 'eval_common.py'
        elif name == 'generate.py':
            s = s.replace("ARMS = ('FULL4_C200', 'SEAM_SLOT_C200')", "ARMS = ('LONG', 'ANTI_COVER', 'LOCAL_NOISE')")
            s = s.replace('r2_seam_slot_c200_endpoint_inference_v1', 'r2_seam_quality_four_endpoint_inference_v1')
            s = s.replace("'experiments/r2_seam_slot_c200_v1'", "'experiments/r2_seam_quality_four_v1'")
            s = s.replace('owns only experiments/r2_seam_slot_c200_v1', 'owns only experiments/r2_seam_quality_four_v1')
            s = s.replace('r2_seam_slot_c200_generator_v1', 'r2_seam_quality_four_generator_v1')
            s = s.replace("int(payload['local_step']) != 200 or int(payload['global_step']) != 12200", "int(payload['local_step']) != 800 or int(payload['seam_total_step']) != 1000 or int(payload['global_step']) != 13000")
            s = s.replace("if arm == 'SEAM_SLOT_C200':", "if arm in ARMS:")
            s = s.replace("hp = root / 'inference/SEAM_SLOT_C200_occupancy.pt'", "hp = root / 'inference' / (arm + '_occupancy.pt')")
            s = s.replace('r2_seam_slot_c200_occupancy_v1', 'r2_seam_quality_four_occupancy_v1')
            s = s.replace("payload['local_step'] != 200", "payload['local_step'] != 800 or payload['seam_total_step'] != 1000 or payload['global_step'] != 13000 or payload['arm'] != arm")
            s = replace_once(s, "            z = read_noise(q)\n            raw, trajectory, audit = rollout", "            z = read_noise(q)\n            # Common saved Gaussian; only occupied third-corner mean changes.\n            if arm == 'LOCAL_NOISE':\n                z = z.copy()\n                for slot in layout['seam_slots']:\n                    ids = layout['anchor_ids'][int(slot), :2]\n                    midpoint = layout['C_vertices'][ids].mean(axis=0, dtype=np.float32)\n                    z[int(slot)-int(layout['K']), 2] += midpoint\n            raw, trajectory, audit = rollout")
            s = replace_once(s, "            arrays = raw_arrays(raw, layout, logits, source)", "            audit['noise_policy'] = ('saved standard Gaussian plus C edge midpoint at occupied seam third corners; sigma=1; bulk unchanged' if arm == 'LOCAL_NOISE' else 'saved standard Gaussian; no HYBRID sampling noise')\n            arrays = raw_arrays(raw, layout, logits, source)")
            s = s.replace('"""Fixed endpoint Euler50 for FULL4_C200 and C-only SEAM_SLOT_C200.', '"""Fixed endpoint Euler50 for three SEAM quality continuations.')
            target = name
        elif name == 'postprocess.py':
            s = s.replace('r2_seam_slot_c200', 'r2_seam_quality_four')
            s = s.replace('RAW=32', 'RAW=48').replace('outputs must exist', 'outputs must exist')
            target = name
        else:
            s = s.replace('from . import common as C', 'from . import eval_common as C')
            s = s.replace('seam_slot_c200_original_quality_free_only_graph_v1', 'seam_quality_four_original_quality_free_only_graph_v1')
            s = s.replace('two registered arms', 'three registered arms')
            s = s.replace("arm=='SEAM_SLOT_C200'", 'arm in C.ARMS')
            s = s.replace('cells=64', 'cells=96').replace('registered=64', 'registered=96')
            # Retain only the unchanged scorer and bounded dispatcher. New paired
            # summaries are written by summarize.py, not the old two-arm routine.
            s = s[:s.index('\ndef occupancy(root,m):')]
            s += "\n\ndef main():\n    p=argparse.ArgumentParser()\n    p.add_argument('--stage', choices=('prepare','score'), required=True)\n    p.add_argument('--root', default=str(C.OUT))\n    p.add_argument('--workers',type=int,choices=(1,2),default=2)\n    a=p.parse_args()\n    result=prepare(a.root) if a.stage=='prepare' else run(a.root,a.workers)\n    print(json.dumps(dict(status=result['status'],stage=a.stage)),flush=True)\n\nif __name__=='__main__':main()\n"
            target = name
        dest = HERE / target
        if dest.exists():
            raise FileExistsError(dest)
        dest.write_text(s, encoding='utf-8', newline='\n')
    (HERE / 'adapter_origin.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
