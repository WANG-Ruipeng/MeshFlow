"""CPU-only H3 DEV gallery workflow. Does not train, sample, or score.

Only step500/1000 and DEV are accepted. Missing completed scoring produces
PENDING without placeholders. Manual AI review is a separate, explicit step.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LP_NUM_THREADS']:
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'

OUT = Path('/mnt/e/MeshFlow/native_t1/runs/chair_control_feedback_auto_v1')
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(step, generate=False):
    assert step in (500, 1000)
    labels = ['START'] + ['explore_'+a+'_'+str(step) for a in [
        'FM_CONTROL', 'REPLAY_FM_CONTROL', 'TRAJECTORY_REPLAY', 'TRAJECTORY_REPLAY_CAPPED']]
    registration = OUT/'manifests/DEV.json'
    requests = json.loads(registration.read_text())['rows']
    expected_ids = {r['id'] for r in requests}
    assert len(expected_ids) == 16
    completed_sources, missing = {}, []
    for label in labels:
        folder = OUT/'evaluation'/label/'DEV'
        for name in ['summary.json', 'per_output.json']:
            p = folder/name
            if not p.exists():
                missing.append(str(p))
            else:
                completed_sources[str(p)] = sha(p)
        if (folder/'per_output.json').exists():
            rows = json.loads((folder/'per_output.json').read_text())
            assert len(rows) == 16 and {r['request_id'] for r in rows} == expected_ids
            # Identity-only read: no scalar metrics are used to choose figures.
            for rid in sorted(expected_ids):
                for p in [folder/'scoring/joined'/(rid+'.json'), folder/'scoring/control'/(rid+'.npz'),
                          OUT/'runtime'/label/rid/'raw.npz', OUT/'runtime'/label/rid/'result.json']:
                    if not p.exists():
                        missing.append(str(p))
    state = dict(status='PENDING' if missing else 'READY', step=step, labels=labels,
        columns=['C']+labels+['GT: C + free'], split='DEV', expected_requests=16,
        expected_parents=8, expected_unique_mesh_panels=480, expected_overlay_panels=240,
        expected_page_sizes=dict(mesh=[3003,4251], feedback=[2187,2619]),
        assigned_review_parents=['86ec9c796a6c1789e792598ed58dd1cc','bdd29e651e5f6fb2b079317292bdc5d4',
                                 'c1b8c2f0c4f3cc43aff7c08b060f5ed6','f59627a077dcc2566c63b8b48495261a'],
        candidate_label=labels[-1], matched_label=labels[2], ablation_reference_label=labels[3],
        AI_VISUAL_REVIEW='PENDING', human_usability='NOT_RUN',
        missing_files=missing, completed_evaluation_sources=completed_sources,
        DEV_manifest_sha256=sha(registration), source_script_sha256=sha(__file__),
        CPU_threads=2, GL_workers=1, software_GL=True,
        frozen_scientific_source_changes=0, model_calls=0, new_metric_calls=0, CONFIRM_read=False)
    if not missing and generate:
        from experiments.mf_h1_local0.chair_control_feedback_auto_v1.visualization import generate as render
        result = render(OUT, labels, split='DEV')
        assert result['status'] == 'COMPLETE'
        assert result['unique_cached_mesh_panels'] == 480 and result['overlay_panels'] == 240
        gallery = Path(result['index']).parent
        state.update(status='COMPLETE', gallery=str(gallery), index=result['index'],
                     contact_sheet=result['contact_sheet'],
                     manifest_sha256=sha(gallery/'manifest.json'),
                     gallery_wall_seconds=result['wall_seconds'],
                     geometry_unchanged=result['all_geometry_unchanged'],
                     all_vertices_inside=result['all_vertices_inside'])
    (HERE/f'step{step}.json').write_text(json.dumps(state, indent=2)+'\n')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, choices=[500,1000], required=True)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    result = run(args.step, args.generate)
    print(json.dumps({k:v for k,v in result.items() if k != 'completed_evaluation_sources'}, indent=2))
