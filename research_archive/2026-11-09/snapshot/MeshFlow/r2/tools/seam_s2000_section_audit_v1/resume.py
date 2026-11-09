"""Read-only status for the saved-geometry section comparison."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / 'experiments/r2_seam_s2000_section_audit_v1'
names = ('protocol.json', 'topology/completion.json', 'gallery/completion.json',
         'visual_train.csv', 'visual_dev.csv', 'case_trace.json', 'REPORT.md', 'completion.json')
present = {name: (OUT / name).exists() for name in names}
print(json.dumps({'state': 'COMPLETE' if present['completion.json'] else 'IN_PROGRESS',
                  'artifacts': present, 'automatic_dispatch': False,
                  'recovery': 'Inspect existing per-stage receipts. Do not rerun a model or overwrite historical geometry.'}, indent=2))
