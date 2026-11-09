"""Seal CPU-only backend source snapshot and derived accounting, no scoring."""
from __future__ import annotations
import collections,csv,json
from pathlib import Path
from . import geo_only as C

def main():
    out=C.OUT;m=C.prepare();complete=C.read(out/'COMPLETE.json');C.verify(complete['files'])
    for t in m['tasks']:
        for f in ('geometry/receipt.json','cascade/complete.json'):C.verify(C.read(C.resolve(t['output_dir'])/f)['files'])
    root=out/'source_archive';files={}
    source=dict(m['code'])
    for name in ('geo_only.py','geo_report.py','geo_review.py','geo_archive.py'):
        p=Path(__file__).parent/name;source[str(p)]=C.sha(p)
    for name in ('PROTOCOL.md','manifest.json'):
        p=out/name;source[str(p)]=C.sha(p)
    C.verify(source)
    for name,h in source.items():
        p=C.resolve(name)
        try:relative=p.relative_to(C.ROOT)
        except ValueError:relative=Path('external')/p.parent.parent.name/p.parent.name/p.name
        dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(p.read_bytes())
        assert C.sha(dest)==h;files[str(dest)]=h
    C.write(root/'manifest.json',dict(status='SOURCE_SNAPSHOT_ONLY',source_files=source,archived_files=files,
        original_production_code_modified=False,weights_copied=False,model_calls=0,scoring_calls=0))
    C.textnew(out/'RUNBOOK.md','''# Verify / resume

This four-request CPU backend experiment is COMPLETE. Do not run any model.

`python -B -m r2.pilots.bench_high_joint_coverage_v1.geo_only verify`

checks protected inputs, frozen geometry/scoring code, complete receipts and all
geometry/cascade files. Completed `run` is verification-only; partial science
attempts stop for explicit review rather than silently replaying.

`geo_report collect` and `geo_report render` reuse sealed outputs. The literal
review in `geo_review.py` records actual view_image events, not generated labels;
do not rerun it over the immutable review JSON. No additional experiment is needed.

REPORT table FF, CF and small-angle columns are percentages; other columns use
their original count denominators. I remains primary NA (zero/2048 interior
queries; minimum32), with legacy numeric fields preserved only for traceability.

No D0 or model was called. Four original GEO/source exports, four HINGE_OPP015,
four ALPHA, four HOLE passes and eight expanded original metric evaluations ran.
The old RAW/BASE/FINAL twelve scores were reused read-only. CPU workers2, threads1.
No failed science call, restart, replay, source mutation, checkpoint or new noise.
''')
    with (out/'analysis/area_stages.csv').open(encoding='utf-8-sig') as f:areas=list(csv.DictReader(f))
    by=collections.defaultdict(list)
    for r in areas:by[r['stage']].append(r)
    fields=('surviving_original_free_faces','deleted_original_free_faces','summed_original_free_triangle_area',
            'original_RAW_free_area','free_corner_max_displacement_model','free_corner_mean_displacement_model')
    summary=[]
    for stage,rr in by.items():
        summary.append(dict(stage=stage,requests=len(rr),**{k:sum(float(r[k]) for r in rr)/len(rr) for k in fields}))
    C.csvwrite(out/'analysis/area_stage_macro.csv',summary)
    counts=collections.defaultdict(int)
    with (out/'analysis/deletion_reasons.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):counts[(r['chain'],r['reason'])]+=int(r['count'])
    C.write(out/'execution_and_area_accounting.json',dict(status='VERIFIED',counts=C.read(out/'compute_complete.json'),
        actual_view_events=8,actual_review_cells=16,expanded_score_rows_new=8,expanded_score_rows_reused=12,
        scientific_failures=0,scientific_replay=0,archive_manifest_sha256=C.sha(root/'manifest.json'),
        code_sha256=C.sha(__file__),area_stage_means=summary,
        deletion_reason_totals=[dict(chain=c,reason=r,count=n) for (c,r),n in counts.items()],
        limits='Maximum displacement column is mean of per-request maxima. Area counts all triangle multiplicity; no semantic missing-body inference.',
        protected_files_unchanged=True))
    print(json.dumps(dict(status='VERIFIED',archived_files=len(files),deletion_totals=[dict(chain=c,reason=r,count=n) for (c,r),n in counts.items()]),ensure_ascii=False))

if __name__=='__main__':main()
