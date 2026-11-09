"""Hash-bind actual manually entered H3 observations. Never judges or renders.

Input notes must be written after viewing all assigned saved pages. This helper
only validates identities and records both the teacher and original-TR comparison.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from PIL import Image

OUT = Path('/mnt/e/MeshFlow/native_t1/runs/chair_control_feedback_auto_v1')
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bind(step, notes_path):
    assert step in (500,1000)
    state = json.loads((HERE/f'step{step}.json').read_text())
    assert state['status'] == 'COMPLETE'
    gallery = Path(state['gallery'])
    manifest_path = gallery/'manifest.json'
    assert sha(manifest_path) == state['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['all_vertices_inside'] and manifest['all_geometry_unchanged']
    pages = {Path(p['path']).name: p for p in manifest['parent_pages']}
    notes_path = Path(notes_path)
    notes = json.loads(notes_path.read_text())
    assert notes['actually_viewed'] is True and notes['step'] == step
    assert notes['candidate_label'] == state['candidate_label']
    assert notes['matched_label'] == state['matched_label']
    assert notes['ablation_reference_label'] == state['ablation_reference_label']
    records = notes['rows']
    expected = {(u,n) for u in state['assigned_review_parents'] for n in (0,1)}
    assert len(records) == 8 and {(r['uid'],r['noise_index']) for r in records} == expected
    registration = json.loads((OUT/'manifests/DEV.json').read_text())['rows']
    registered = {(r['uid'],r['noise_index']):r for r in registration}
    images, rows, ablation_rows = {}, [], []
    for note in records:
        assert note['actually_viewed'] is True
        for k in ['judgment','versus_TR_judgment','versus_START_judgment']:
            assert note[k] in ['BETTER','WORSE','MIXED','SIMILAR']
        for k in ['geometry_vs_teacher','geometry_vs_TR','geometry_vs_START','condition_observation','feedback_observation']:
            assert isinstance(note[k],str) and len(note[k]) > 20
        paths = [str(gallery/(note['uid']+s+'.png')) for s in ['', '_feedback']]
        for p in paths:
            if p not in images:
                page = pages[Path(p).name]
                assert sha(p) == page['sha256']
                with Image.open(p) as im:
                    assert list(im.size) == page['size']
                images[p] = dict(path=p,sha256=page['sha256'],size=page['size'],actually_viewed=True)
        row = dict(note)
        row.update(request_id=registered[(note['uid'],note['noise_index'])]['id'],
            review_type='AI_VISUAL_REVIEW',reviewer='AI agent /root/coupling',
            candidate_label=state['candidate_label'],matched_label=state['matched_label'],
            ablation_reference_label=state['ablation_reference_label'],reference_label='START',
            image_paths=paths,image_sha256={p:images[p]['sha256'] for p in paths},
            scope_status='COMPLETE',mesh_views=['front_3q','back_3q','front','side','wire'],
            feedback_views=['S','I','E'],human_usability='NOT_RUN')
        row['status']='FAIL' if row['new_obvious_main_body_collapse_vs_matched'] else 'PASS'
        rows.append(row)
        ablation=dict(row, matched_label=state['ablation_reference_label'],
                      judgment=row['versus_TR_judgment'],
                      geometry_observation=row['geometry_vs_TR'],
                      new_obvious_main_body_collapse_vs_matched=row['new_obvious_main_body_collapse_vs_TR'],
                      status='FAIL' if row['new_obvious_main_body_collapse_vs_TR'] else 'PASS')
        ablation_rows.append(ablation)
    assigned_failures=[r['request_id'] for r in rows if r['status']=='FAIL']
    result=dict(schema='chair_h3_matched_dev_partial_visual_review_v1',
        status='FAIL' if assigned_failures else 'INCOMPLETE',
        scope_status='PARTIAL_FULL_COHORT',assigned_scope_status='COMPLETE',
        created_utc=datetime.now(timezone.utc).isoformat(),review_type='AI_VISUAL_REVIEW',
        reviewer='AI agent /root/coupling',candidate_label=state['candidate_label'],
        matched_label=state['matched_label'],ablation_reference_label=state['ablation_reference_label'],
        human_usability='NOT_RUN',
        gate=dict(status='FAIL' if assigned_failures else 'INCOMPLETE',
                  assigned_rows_status='FAIL' if assigned_failures else 'PASS',
                  failing_request_ids=assigned_failures,
                  interpretation='Only new obvious main-body collapse. Local defects and WORSE judgments remain material; individual PASS is not a usability or clean-mesh pass.',
                  full_cohort_requires='Merge actual root review of first four DEV parents.'),
        scope=dict(split='DEV',assigned_parents=4,assigned_requests=8,full_cohort_requests=16,
                   viewed_columns=state['columns'],four_shaded_views_plus_wire=True,
                   all_three_feedback_overlays=True,
                   display_regions=[[0,0,3003,2162],[0,2162,3003,4251]],
                   display_method='Original full-resolution pages shown in two adjacent lossless regions covering all pixels; full feedback pages viewed. Files unchanged.',
                   model_calls=0,new_metric_calls=0,CONFIRM_read=False),
        descriptive_counts_vs_teacher=dict(Counter(r['judgment'] for r in rows)),
        descriptive_counts_vs_TR=dict(Counter(r['versus_TR_judgment'] for r in rows)),
        limitations=['AI unblinded image assessment, not human usability.',
                     'Saved x-ray feedback point density is not an area-weighted metric.',
                     'Style change alone is not judged as improvement or deterioration.',
                     'No exact attachment, manifoldness or physical stability certification from images.'],
        source_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)),
        source_images=list(images.values()),source_notes=dict(path=str(notes_path),sha256=sha(notes_path)),
        source_script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        rows=rows,records=rows,ablation_records=ablation_rows)
    stem=state['candidate_label']+'_vs_'+state['matched_label']+'_coupling_partial'
    path=OUT/'visuals/reviews'/(stem+'.json')
    assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n')
    lines=[f'# H3 DEV{step} assigned-four-parent visual review','',
           f"Assigned scope COMPLETE (8 requests); overall status {result['status']}. Human usability NOT_RUN.",'',
           '| Parent | Noise | Cap vs teacher | Cap vs TR | Cap vs START | Teacher comparison | TR comparison |',
           '|---|---:|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['uid'][:6]} | {r['noise_index']} | {r['judgment']} | {r['versus_TR_judgment']} | {r['versus_START_judgment']} | {r['geometry_vs_teacher']} | {r['geometry_vs_TR']} |")
    lines+=['',f'Full observations and original image hashes: `{path}`.',
            'First four parents are reviewed by root; this file alone does not establish full-cohort PASS. No CONFIRM, model or scoring calls.','']
    path.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=result['status'],assigned_scope_status='COMPLETE',path=str(path),sha256=sha(path)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--step',type=int,choices=[500,1000],required=True)
    parser.add_argument('--notes',required=True)
    args=parser.parse_args()
    bind(args.step,args.notes)
