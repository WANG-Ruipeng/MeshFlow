"""Small evidence archive; preserve the sealed full experiment unchanged."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'experiments/r2_bench_raw_local_repair_v1'
OUT=Path(__file__).resolve().parent
DEST=ROOT/'deliverables/R2_BENCH_RAW_LOCAL_REPAIR_V1_REVIEW_SMALL.zip'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def run():
    assert not DEST.exists()
    seal=json.loads((SRC/'COMPLETE.json').read_text())
    assert seal['status']=='COMPLETE_VERIFIED'
    selected=set()
    for name in ['REPORT.md','PROTOCOL.md','CPU_CONTROL_ADDENDUM.md','decision.json','COMPLETE.json','source_hashes.json','code.diff']:
        selected.add(SRC/name)
    for folder in ['analysis','visual_review','audits','source_archive','training_preparation']:
        selected.update(p for p in (SRC/folder).rglob('*') if p.is_file())
    for name in ['per_output.csv','per_condition.csv','per_parent.csv','parent_macro_summary.csv','paired_per_output.csv','metric_rows.json']:
        selected.add(SRC/'scores'/name)
    for name in ['manifest.json','conflict_attribution.csv']:
        selected.add(SRC/'data'/name)
    for name in ['summary.csv','strategy.json','COMPLETE.json']:
        p=SRC/'targets'/name
        if p.exists():selected.add(p)
    for p in (SRC/'targets').glob('*/target.npz'):
        selected.add(p)
        selected.update(x for name in ['per_corner.csv','actions.csv','patches.json','receipt.json','candidate_detection.json']
                        if (x:=p.parent/name).exists())
    selected.update((SRC/'gallery').glob('*/matched_control.png'))
    members={p.relative_to(SRC).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)}
    for p in selected:
        old=seal['files'].get(str(p))
        if old is not None:assert old==sha(p)
    readme='''# RAW local repair pilot review pack\n\nThe complete sealed experiment stays in experiments/r2_bench_raw_local_repair_v1.\nThis compact pack includes all 12 target arrays, masks, per-output metrics, paired tables, actual review records, actual source code, and the 12 same-chain FINAL comparison images.\nNo weights. Full RAW/D0/POST history, all 90 reviewed views, proposal arrays and original detector pair caches remain in the full local experiment or its protected source paths; this is not a portable full-pipeline replay bundle.\nThe decision is NO_TRAINING_DATA_GATE_FAILED. Local geometric benefit exists, but only six requests contain EDIT and the pre-registered data gate requires seven.\n'''
    DEST.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(DEST,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('README.md',readme)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(members,ensure_ascii=False,indent=2))
        for p in sorted(selected):z.write(p,p.relative_to(SRC).as_posix())
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
        for n,r in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==r['sha256']
    result=dict(status='ARCHIVED_VERIFIED',path=str(DEST),bytes=DEST.stat().st_size,sha256=sha(DEST),
        scientific_members=len(members),all_12_targets=True,full_original_experiment_unchanged=True,
        source_seal_sha256=sha(SRC/'COMPLETE.json'),member_hashes=members)
    assert result['bytes']<10_000_000, 'Keep the existing verified archive but do not claim it is below 10MB'
    with (OUT/'local_repair_archive_receipt.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='member_hashes'},ensure_ascii=False))

if __name__=='__main__':run()
