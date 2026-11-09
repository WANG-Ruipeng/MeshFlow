"""Record actual assistant image observations; verify completion. CPU/files only."""
from pathlib import Path
import csv
import hashlib
import html
import io
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_user_v2_training_v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve(p):
    p = str(p).replace('\\', '/')
    if p.startswith('/mnt/e/MeshFlow/'):
        return ROOT / p[len('/mnt/e/MeshFlow/'):]
    return Path(p)


def write(p, text):
    data = text.encode('utf-8')
    if p.exists():
        assert p.read_bytes() == data, p
    else:
        p.write_bytes(data)


def write_json(p, obj):
    write(p, json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def write_csv(p, rows):
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    write(p, '\ufeff' + f.getvalue())


def main():
    if (OUT / 'CLOSEOUT.json').exists():
        rec = read(OUT / 'CLOSEOUT.json')
        for rel, expected in rec['files'].items():
            assert digest(OUT / rel) == expected, rel
        assert digest(Path(__file__)) == rec['finalize_code_sha256']
        print('PASS: existing closeout unchanged; no model calls or scoring')
        return
    assert read(OUT / 'analysis/audit.json')['status'] == 'PASS'
    notes = read(OUT / 'visual_review_notes.json')
    manifest = read(OUT / 'analysis/contact_sheets/manifest.json')['records']
    requests = read(OUT / 'eval_manifest.json')['requests']
    assert set(notes['parents']) == {r['parent_id'] for r in manifest}
    full = set(notes['full_resolution_pages'])
    pages = []
    reqrows = []
    sections = []
    for record in manifest:
        parent = record['parent_id']
        contact = resolve(record['path'])
        assert digest(contact) == record['sha256']
        qs = sorted([q for q in requests if q['parent_id'] == parent], key=lambda q: q['request_id'])
        assert len(qs) == len(notes['parents'][parent]['observations']) == 4
        by_request = dict(zip([q['request_id'] for q in qs], notes['parents'][parent]['observations']))
        for q in qs:
            reqrows.append(dict(request_id=q['request_id'], parent_id=parent,
                reviewer='AI_PROXY', blind=False, stages_seen='RAW|BASE_POST|CLEAN_FINAL_POST',
                cross_stage_observation=by_request[q['request_id']],
                stage_changes=notes['parents'][parent]['stages'],
                topology_visual_status='UNKNOWN_NOT_INFERRED_FROM_SECTION_ENDPOINTS'))
        for page in record['pages']:
            source = resolve(page['source'])
            assert digest(source) == page['source_sha256']
            stage, request = page['page_id'].split('__', 1)
            assert request in by_request
            pages.append(dict(page_id=page['page_id'], request_id=request, stage=stage,
                reviewer='AI_PROXY', blind=False, actually_viewed_in_contact_sheet=True,
                also_viewed_original_resolution=page['page_id'] in full,
                source=str(source.relative_to(OUT)), source_sha256=page['source_sha256'],
                contact_sheet=str(contact.relative_to(OUT)), contact_sha256=record['sha256'],
                cross_stage_observation=by_request[request],
                cross_stage_changes=notes['parents'][parent]['stages'],
                independent_human_rating=False, numerical_adoption_label='NOT_ASSIGNED',
                limitation='Small cracks/topology UNKNOWN; nonblind paired inspection; inherited clipping retained'))
        link = '../' + str(contact.relative_to(OUT)).replace('\\', '/')
        observations = ''.join('<li>' + html.escape(x) + '</li>' for x in notes['parents'][parent]['observations'])
        sections.append(f'<h2>{parent}</h2><p>{html.escape(notes["parents"][parent]["stages"])}</p><ol>{observations}</ol><a href="{link}"><img src="{link}" alt="{parent}"></a>')
    assert len(pages) == 144 and len(reqrows) == 48
    assert len({p['page_id'] for p in pages}) == 144
    assert full <= {p['page_id'] for p in pages} and len(full) == 5
    dest = OUT / 'visual_review'
    dest.mkdir(exist_ok=True)
    write_csv(dest / 'actual_review.csv', pages)
    write_csv(dest / 'per_request_notes.csv', reqrows)
    write_json(dest / 'actual_review.json', dict(status='ACTUALLY_REVIEWED_AI_PROXY',
        method=notes['method'], contact_sheets=12, paired_pages_covered=144, requests=48,
        additional_full_resolution_pages=sorted(full), independent_human_ratings=0,
        adoption_success_count='NOT_ASSIGNED', source_notes_sha256=digest(OUT / 'visual_review_notes.json'),
        contact_manifest_sha256=digest(OUT / 'analysis/contact_sheets/manifest.json'),
        source_note='Prepared manifest is historical; this receipt records subsequent actual viewing.'))
    write(dest / 'index.html', '<!doctype html><meta charset="utf-8"><title>USER v2 actual review</title>'
        '<style>body{font:16px sans-serif;margin:32px;max-width:1500px}img{width:100%;height:auto}li{margin:10px 0}</style>'
        '<h1>Bench USER v2 — 实际 AI_PROXY 图审</h1><p>每组12页；行RAW/BASE/FINAL，列4请求，每格左A右USER v2。'
        '实际查看联系图；五页另看原分辨率。不是人类盲评，不从截面端点认证裂缝/拓扑。</p>' + ''.join(sections))
    files = [OUT / 'REPORT.md', OUT / 'visual_review_notes.json']
    files += list(dest.iterdir())
    files += [p for p in (OUT / 'analysis').rglob('*') if p.is_file()]
    files += [OUT / 'training_registration.json', OUT / 'results/completion.json', OUT / 'gallery/completion.json']
    write_json(OUT / 'CLOSEOUT.json', dict(status='COMPLETE_WITH_NO_OVERALL_PROMOTION',
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        final_label='USER_V2_S1500', updates=1500, new_raw=48, new_D0=48, new_score_rows=144,
        actual_review=dict(reviewer='AI_PROXY', contact_sheets=12, paired_pages=144, original_resolution_pages=5),
        current_review_model_calls=0, current_review_rescoring=0,
        default_changed=False, weights_deleted=False, git_commit_or_push=False,
        audit=str((OUT / 'analysis/audit.json').relative_to(OUT)),
        files={str(p.relative_to(OUT)).replace('\\', '/'): digest(p) for p in sorted(files)},
        finalize_code_sha256=digest(Path(__file__))))
    print('COMPLETE: 48 request notes, 144 paired pages recorded, 5 full-resolution checks; no model/scoring calls')


if __name__ == '__main__':
    main()
