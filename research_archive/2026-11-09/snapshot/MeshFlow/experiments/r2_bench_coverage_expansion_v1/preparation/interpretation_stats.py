"""Read frozen metadata only; supplementary descriptive tables, no geometry scan."""
from pathlib import Path
import json,collections,hashlib
ROOT=Path('/mnt/e/MeshFlow');OUT=ROOT/'experiments/r2_bench_coverage_expansion_v1'
paths=[ROOT/'experiments/r2_bench_clean_multipart_seam_v1/data/manifest.json',OUT/'data/manifest.json']
old,new=[json.loads(p.read_text()) for p in paths]
def parts(rows):
    out={}
    for part in sorted({r['part_type'] for r in rows}):
        group=[r for r in rows if r['part_type']==part]
        out[part]=dict(conditions=len(group),parents=len({r['parent_id'] for r in group}))
    return out
newparents={r['parent_id']:r['source_target_risks']['source_exact_duplicate_ID_faces'] for r in new['new_train']}
stats=dict(old=parts(old['train']),new=parts(new['train']),new_parent_duplicate_excess=newparents,
    new_parents_with_ID_duplicates=sum(x>0 for x in newparents.values()),new_parent_duplicate_excess_sum=sum(newparents.values()),
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    note='Uses existing recorded source duplicate metadata. No new triangle-pair geometry or model evaluation.')
(OUT/'preparation/interpretation_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats,indent=2))
