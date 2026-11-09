"""Assemble actual agent image observations, never infer ratings from metrics."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/e/MeshFlow/experiments/r2_bench_layout_oracle_v1')


def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def translate(p): return Path(str(p).replace('\\', '/').replace('E:/', '/mnt/e/'))


def main():
    supplied = json.loads((ROOT/'visual_review/part_02_actual_notes.json').read_text(encoding='utf-8'))
    labels = ['EXPOSURE', 'PLAIN1000', 'LAYOUT1000', 'LAYOUT_NULL']
    axes = ['overall_shape', 'C_use', 'support_retention', 'visible_fragment_or_gap', 'layering_or_fold']
    rows = []
    for page in supplied['notes']:
        page_id = page['stage'] + '__' + page['request_id']
        directory = ROOT/'gallery'/page_id
        image = directory/'comparison.png'
        receipt = directory/'comparison.json'
        image_sha = h(image)
        identity = json.loads(receipt.read_text(encoding='utf-8'))
        assert identity['image_sha256'] == image_sha
        assert identity['identity']['request_id'] == page['request_id']
        events = [e for e in supplied['events'] if translate(e['path']) == image]
        assert len(events) == 1
        event = events[0]
        for label, values in zip(labels, page['ratings']):
            row = dict(request_id=page['request_id'], label=label, stage=page['stage'],
                actually_viewed=True, views_and_sections_checked=True,
                views_checked=['azimuth=-60', 'azimuth=120', 'front finite section', 'side finite section'],
                image_path=str(image), image_sha256=image_sha, image_receipt_path=str(receipt),
                image_receipt_sha256=h(receipt), tool='view_image', view_event_id=event['id'],
                viewed_at_utc=event['viewed_at_utc'], note=values[5])
            row.update(dict(zip(axes, values[:5])))
            if label == 'EXPOSURE' and page['stage'] == 'BASE_POST' and page['request_id'].startswith('459') and page['request_id'].endswith('n1'):
                crop = supplied['crop']
                row.update(crop_paths=[crop['path']], crop_sha256=[h(translate(crop['path']))],
                    crop_view_events=[crop], crop_policy='Original-pixel rectangle only; no rescaling or camera changes')
            rows.append(row)
    assert len(rows) == 48
    assert len({(r['request_id'],r['stage'],r['label']) for r in rows}) == 48
    report = dict(schema='bench_layout_oracle_actual_review_rows_v1',status='COMPLETE',
        assignment_id='review_part_02',reviewer='/root/layout_model_impl',reviewer_kind='AI_PROXY',blind=False,
        scope='Actual viewing of 12 original comparison pages, each with 4 methods, two oblique views and front/side finite sections; one native-pixel crop inspected',
        observation_source='Direct view_image inspection only; no numeric score or historical label substitution',
        view_event_id_policy='Agent-assigned explicit call identifiers emitted in tool results and stored with source-image paths',
        rows=rows,
        limitations=['AI_PROXY and not blind; not independent human certification',
            'A section-line endpoint alone does not prove a hole',
            'Tiny remaining support strips in 459/n1 EXPOSURE POST cannot be confidently certified as intact supports'],
        notes_sha256=h(ROOT/'visual_review/part_02_actual_notes.json'))
    target=ROOT/'visual_review/completed/review_part_02.json'
    target.parent.mkdir(exist_ok=True)
    with target.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(path=str(target),rows=len(rows),sha256=h(target))))


if __name__=='__main__': main()
