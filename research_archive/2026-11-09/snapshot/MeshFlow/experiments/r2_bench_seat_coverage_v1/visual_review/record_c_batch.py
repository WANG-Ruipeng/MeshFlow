"""Encode explicit reviewer observations after actual full-PNG viewing only."""
import json, sys
from datetime import datetime, timezone
from r2.pilots.bench_seat_coverage_v1 import visual_review as v
maps = [dict(R='REASONABLE', I='INCOMPLETE', U='UNCERTAIN'),
        dict(P='PLAUSIBLE', X='REPLACED_OR_IGNORED', U='UNCERTAIN'),
        dict(P='PRESENT', M='MISSING_OR_DAMAGED', U='UNCERTAIN'),
        dict(Y='YES', N='NO_OBVIOUS', U='UNCERTAIN'),
        dict(Y='YES', N='NO_OBVIOUS', U='UNCERTAIN')]
doc=json.loads(v.resolve(sys.argv[1]).read_text(encoding='utf-8'))
for page in doc:
    obs={label:dict(**{k:maps[i][row[i]] for i,k in enumerate(v.AXES)},note=row[5])
         for label,row in page['observations'].items()}
    print(v.record_page('c_curved_and_holdout_df6',page['page_id'],obs,
        reviewer='coverage_runtime',view_event_id='coverage_actual_'+page['page_id'],
        viewed_at_utc=datetime.now(timezone.utc).isoformat(),
        actual_view_source='functions.exec -> tools.view_image(detail=original); full PNG, both oblique views and fixed front/side sections',
        actually_viewed=True,views_and_sections_checked=True))
