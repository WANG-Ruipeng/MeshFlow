import pathlib,json,shutil,datetime
root=pathlib.Path(__file__).resolve().parent
dest=root/'engineering_attempt_00';dest.mkdir(exist_ok=False)
src=root.parents[1]/'r2/pilots/unified_rescue_post_v1'
shutil.copytree(src,dest/'source')
cells=list((root/'cells').glob('*/*'))
rows=[]
for p in cells:
    rows.append(dict(cell=str(p.relative_to(root)),complete=(p/'complete.json').exists(),
        saved_stages=[d.name for d in p.iterdir() if d.is_dir() and (d/'post.npz').exists()],
        score_completes=[str(x.relative_to(p)) for x in p.glob('*/scores/complete.json')]))
for name in ('cells','registration.json','registration_pre_execution.json','dispatch.json','resources_before.json','synthetic_unit_tests.json','input_audit.csv'):
    p=root/name
    if p.exists():
        assert p.resolve().is_relative_to(root.resolve())
        shutil.move(str(p),str(dest/name))
record=dict(status='CPU_ENGINEERING_REVISION_REQUIRED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    reason='Prospective protected corners must reserve exact current coordinates; earlier corner cannot merge into coordinate held by later same-face corner',
    requests_started=len(rows),requests_complete=sum(r['complete'] for r in rows),cells=rows,
    retry_scope='Exactly 140 fixed requests in corrected execution; prior CPU work separately counted; no model retry/no new D0/no new noise')
(dest/'recovery_receipt.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
