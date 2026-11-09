"""Materialize pixel-exact final-gallery inspection sheets; no rendering."""
from pathlib import Path
import base64
import json
from experiments.mf_h1_local0.chair_capped500_gradbudget_v1.inspect_gallery import inspect, OUT

DEST = OUT / 'visuals/reviews/coupling_inspection'
UIDS = ['866e7248434968051cc7750d0afca796', '86ec9c796a6c1789e792598ed58dd1cc']
DEST.mkdir(parents=True, exist_ok=True)
records = []
for panel, candidates in [('OLD', [0, 1]), ('FRESH', [2, 3])]:
    for uid in UIDS:
        for candidate in candidates:
            for group in ['history', 'primary', 'stability']:
                for overlay in [False, True]:
                    result = inspect(OUT, panel, uid, candidate, group, overlay)
                    stem = f'{panel}_{uid[:4]}_n{candidate}_{group}' + ('_overlay' if overlay else '')
                    png = DEST / (stem + '.png')
                    meta = DEST / (stem + '.json')
                    png.write_bytes(base64.b64decode(result['png_base64']))
                    meta.write_text(json.dumps(result['metadata'], indent=2) + '\n')
                    records.append({'png': str(png), 'metadata': str(meta)})
(DEST / 'index.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps({'materialized': len(records), 'model_calls': 0, 'scorer_calls': 0, 'render_calls': 0}))
