"""Read-only recovery/status entry for the saved-S2000 CPU diagnostic."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_s2000_fragment_audit_v1'
def main():
    present={name:(OUT/name).exists() for name in ('protocol.json','numeric/summary.json',
        'gallery/index.html','visual_train.csv','visual_dev.csv','REPORT.md','independent_audit.json','completion.json')}
    print(json.dumps(dict(output=str(OUT),artifacts=present,
        state='COMPLETE' if present['completion.json'] else 'IN_PROGRESS',
        model_calls=0,automatic_dispatch=False,
        recovery='Read protocol.json and per-stage receipts before explicitly invoking analyze.py/render.py; never run a model or overwrite original assets.'),indent=2))
if __name__=='__main__': main()
