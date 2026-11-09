"""Planned exposure accounting from the frozen expanded and comparator streams."""
import csv,json,hashlib,collections
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/e/MeshFlow');OUT=ROOT/'experiments/r2_bench_coverage_expansion_v1'
def read(p):return json.loads(Path(p).read_text())
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
mp=OUT/'data/manifest.json';sp=OUT/'data/schedule.json'
oldmp=ROOT/'experiments/r2_bench_clean_multipart_seam_v1/data/manifest.json'
oldsp=ROOT/'experiments/r2_bench_clean_multipart_seam_v1/data/schedule.json'
m,s,old,olds=map(read,(mp,sp,oldmp,oldsp))
oldids={r['condition_id'] for r in old['train']};newids={r['condition_id'] for r in m['new_train']}
cnt=collections.Counter(r['condition_id'] for step in s['steps'] for r in step['samples'])
oldcnt=collections.Counter(r['condition_id'] for step in olds['steps'] for r in step['samples'])
def topology(r):
    return 'CLOSED_C' if r['boundary_edges']==0 else ('OPEN_NO_OWNER' if r['source_owner_count']==0 else 'OPEN_OWNER')
def source(r):return 'NEW_PARENT_CONDITION' if r['condition_id'] in newids else ('RESTORED_ORIGINAL' if r['condition_id'] not in oldids else 'PREVIOUS_ACTIVE')
rows=[]
for r in m['train']:
    cid=r['condition_id']
    rows.append(dict(condition_id=cid,parent_id=r['parent_id'],part_type=r['part_type'],extent_description=r['review'].get('extent_description',''),
        origin=source(r),topology=topology(r),N=r['N'],K=r['K'],area_fraction=r['area_fraction'],components=r['source_edge_components'],
        owner_count=r['source_owner_count'],boundary_edges=r['boundary_edges'],planned_exposures=cnt[cid],comparator_exposures=oldcnt[cid]))
def writecsv(p,rs):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
writecsv(OUT/'data/condition_exposure.csv',rows)
parents=[]
for pid in sorted({r['parent_id'] for r in rows}):
    pr=[r for r in rows if r['parent_id']==pid]
    parents.append(dict(parent_id=pid,conditions=len(pr),parts=';'.join(sorted({r['part_type'] for r in pr})),
        planned_exposures=sum(r['planned_exposures'] for r in pr),comparator_exposures=sum(r['comparator_exposures'] for r in pr)))
writecsv(OUT/'data/parent_exposure.csv',parents)
by=[]
for field in ('part_type','origin','topology'):
    for value in sorted({r[field] for r in rows}):
        group=[r for r in rows if r[field]==value]
        by.append(dict(dimension=field,value=value,conditions=len(group),parents=len({r['parent_id'] for r in group}),
            planned_exposures=sum(r['planned_exposures'] for r in group),comparator_exposures=sum(r['comparator_exposures'] for r in group)))
writecsv(OUT/'data/group_exposure.csv',by)
stats=lambda x:dict(min=min(x),median=float(np.median(x)),max=max(x))
summary=dict(scope='Planned frozen stream before training, not a claim of executed updates.',
    samples=sum(cnt.values()),comparator_samples=sum(oldcnt.values()),parents=len(parents),conditions=len(rows),
    per_parent=stats([r['planned_exposures'] for r in parents]),per_condition=stats([r['planned_exposures'] for r in rows]),
    groups=by,protected_hashes={str(p):h(p) for p in (mp,sp,oldmp,oldsp)},
    limitations=['Coarse part labels and review descriptions do not imply complete parts.',
      'New source targets were not repaired or certified free of all C/Y or Y/Y intersections.',
      'Source-owner supervision is defined from source completion; inference remains C-only.',
      'Expanded coverage and restored closed/noowner examples jointly change; this one arm cannot isolate them.',
      'Training exposure per parent falls because the fixed 12000 instances now cover 79 rather than 58 parents.'])
(OUT/'data/coverage_exposure_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
