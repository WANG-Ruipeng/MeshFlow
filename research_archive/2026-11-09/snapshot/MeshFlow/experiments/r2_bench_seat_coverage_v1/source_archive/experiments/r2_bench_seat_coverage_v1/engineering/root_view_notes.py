"""Save root's actual supplementary image observations, not scored ratings."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_seat_coverage_v1'
records = []
for parent, note in (
    ('a8e987a6a6486b962d6c8e51365a5a87', 'Both candidate RAW noises show a recognizable four-leg slatted bench, with gray slat underside sections following the blue slats more closely. Candidate final views retain mostly the blue seat and thin remnants of the gray legs. In n0 BASE_POST this support loss is already visible; it cannot be assigned specifically to HINGE/ALPHA/HOLE. Native n1 visibly has gray seat material interrupting/overlying the blue slat structure. These are visible observations, not a topology or unique-source-style judgment.'),
    ('27eb79ddf26fd7ac7898580cb9d16864', 'Both candidate RAW noises show a coherent curved blue seat with gray back and slender end support/arm frames. Native also forms a recognizable bench using this seat, with broad end panels. Candidate final views lose most of the slender support frames and show a detached-looking top rail; the n0 BASE_POST already shows the main support loss. This is not evidence that the RAW generator never produced the support.'),
    ('df6fd36040a5719662701a18243b45d3', 'Both noises of both models retain the broad blue twin-wing seat while the gray completion concentrates in the central gap/end structures, with visible broken or folded strips. Candidate does not visibly resolve this organization in the inspected views. Final gray structure is sparser than RAW. This one held-out parent does not demonstrate broader transfer.'),
):
    for noise in (0, 1):
        request = parent + '__COMPLETE_SEAT_CANDIDATE__COVERAGE_V1__n' + str(noise)
        stages = ['RAW', 'CLEAN_FINAL_POST']
        if noise == 0 and not parent.startswith('df6'):
            stages.append('BASE_POST')
        images = []
        for stage in stages:
            p = OUT / 'gallery' / (stage + '__' + request + '.png')
            images.append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'stage': stage})
        records.append({'request_id': request, 'actually_viewed': True, 'views_and_sections_checked': True,
                        'actual_view_source': 'functions.exec -> tools.view_image; root tool outputs during 2026-10-23 09:31-09:35 UTC',
                        'images': images, 'note': note})
result = {'schema': 'ROOT_SUPPLEMENTARY_VIEW_NOTES_V1', 'reviewer': '/root', 'reviewer_kind': 'AI_PROXY',
          'blind': False, 'recorded_utc': datetime.now(timezone.utc).isoformat(),
          'scope': '14 actually opened pages, 6 requests. Supplementary observations only; excluded from the registered 252-cell rating aggregation. Root had read numerical summaries before these views.',
          'records': records, 'model_calls': 0, 'ratings_generated': 0}
p = OUT / 'visual_review/root_supplementary_notes.json'
if p.exists():
    raise RuntimeError('Do not overwrite actual review notes')
p.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print({'status': 'SAVED_ACTUAL_SUPPLEMENTARY_OBSERVATIONS', 'pages': sum(len(r['images']) for r in records)})
