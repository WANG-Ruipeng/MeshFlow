"""Submit only the 30 pages actually viewed by coverage_assets; no inferred ratings."""
import json
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(root))
from r2.pilots.bench_seat_coverage_v1.visual_review import record_page

submission=json.loads(Path(__file__).with_name('review_submission.json').read_text('utf-8'))
assert len(submission['pages'])==30
assert sum(len(p['ratings']) for p in submission['pages'])==84
for i,page in enumerate(submission['pages'],1):
    observations={label:dict(zip(submission['axes'],values)) for label,values in zip(submission['labels'],page['ratings'])}
    result=record_page(submission['review_group'],page['page_id'],observations,
        reviewer=submission['reviewer'],view_event_id=f'COVERAGE_ASSETS_B_PAGE_{i:02d}_20261023',
        viewed_at_utc=page['viewed_at_utc'],
        actual_view_source='functions.exec tools.view_image full saved comparison PNG; two oblique views and both registered front/side sections inspected; page identity and UTC view time recorded in review_submission.json',
        actually_viewed=True,views_and_sections_checked=True)
    print(json.dumps(result))
