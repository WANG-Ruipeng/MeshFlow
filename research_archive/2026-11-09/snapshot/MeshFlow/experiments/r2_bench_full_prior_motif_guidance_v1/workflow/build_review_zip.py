"""Package the completed evidence; preserve original scientific outputs."""
from pathlib import Path
import hashlib,json,zipfile

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
DEST=OUT/'R2_BENCH_FULL_PRIOR_MOTIF_GUIDANCE_REVIEW.zip'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if DEST.exists():raise FileExistsError(DEST)
    close=json.loads((OUT/'CLOSEOUT.json').read_text())
    if close['status']!='COMPLETE':raise ValueError('Full closure is required')
    for name,digest in close['files'].items():
        if sha(OUT/name)!=digest:raise ValueError('Closeout identity mismatch: '+name)
    files=set()
    for name in ('REPORT.md','RUNBOOK.md','PROTOCOL.md','registration.json','tasks.json','baseline.json','lock.json',
                 'generation_code.json','generation_code_v2.json','generation_code_v3.json','evaluation_code.json',
                 'SOURCE_HASHES.json','code.diff','CLOSEOUT.json'):
        files.add(OUT/name)
    for folder in ('analysis/final','analysis_numeric/final','archive','audits','calibration','visual_review'):
        for p in (OUT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.md','.csv','.json','.diff') and p.name!='complete.json':files.add(p)
    for p in (OUT/'inputs').glob('*.json'):files.add(p)
    for p in (OUT/'engineering_incidents').rglob('*'):
        if p.is_file() and p.suffix in ('.json','.md','.py','.log'):files.add(p)
    for p in (OUT/'workflow').glob('*.py'):files.add(p)
    for p in (OUT/'workflow').glob('*.json'):files.add(p)
    for p in (OUT/'workflow/attempts').rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.log'):files.add(p)
    for p in (OUT/'workflow/post_four_incremental').rglob('*'):
        if p.is_file() and p.suffix in ('.md','.json','.jsonl') and p.name not in ('initial_verification.json',):files.add(p)
    files.add(OUT/'workflow/ledger.jsonl')
    pairs=json.loads((OUT/'archive/review_image_selection.json').read_text())['request_ids']
    images=[]
    for rid in pairs:
        for stage in ('RAW','CLEAN_FINAL_POST'):
            base=OUT/'gallery/final'/(stage+'__'+rid)
            images.extend((base/'comparison.png',base/'comparison.json'))
    for p in files|set(images):
        if not p.is_file():raise FileNotFoundError(p)
    note=('This compact analysis package contains all per-output/per-parent numeric tables, all actual visual observations, '
          'actual scoring and pilot source, hashes, protocol, calls and engineering records. '
          'Illustration pages are an explicitly selected subset; every RAW/BASE/FINAL mesh, 51-state trajectory, full gallery '
          'and original receipt remains in the local experiment directory. Large weights and complete meshes are excluded. '
          'The original source dependencies are included as byte-identical evidence, not a standalone replacement environment.\n')
    with zipfile.ZipFile(DEST,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(files|set(images)):z.write(p,str(p.relative_to(OUT)))
        z.writestr('PACKAGE_README.txt',note)
        z.writestr('PACKAGE_FILE_SHA256.json',json.dumps(
            {str(p.relative_to(OUT)):sha(p) for p in sorted(files|set(images))},indent=2))
    with zipfile.ZipFile(DEST) as z:
        bad=z.testzip()
        if bad:raise ValueError(bad)
        hashes=json.loads(z.read('PACKAGE_FILE_SHA256.json'))
        for name,digest in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError(name)
        if any(n.endswith(('.pt','.pth','.ckpt')) for n in z.namelist()):raise ValueError('Unexpected weights')
    receipt=dict(status='COMPLETE',zip=str(DEST),bytes=DEST.stat().st_size,sha256=sha(DEST),
                 files=len(files|set(images))+2,illustrative_requests=pairs,full_local_assets_preserved=True,
                 CRC_check='PASS',every_stored_file_SHA256='PASS',large_weights_included=False,
                 omitted='Large weights, complete meshes/trajectories and unselected illustration pages only; all metric/review rows retained')
    with (OUT/'PACKAGE_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))
if __name__=='__main__':main()
