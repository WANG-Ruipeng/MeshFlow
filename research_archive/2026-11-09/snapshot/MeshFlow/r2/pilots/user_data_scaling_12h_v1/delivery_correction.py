"""One-time audited correction of asset-vs-parent wording and review archive."""
from pathlib import Path
import datetime, difflib, hashlib, json, shutil, zipfile, os
ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_user_data_scaling_12h_v1'
CODE=ROOT/'r2/pilots/user_data_scaling_12h_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    rev=OUT/'delivery_revisions'/'v1'
    rev.mkdir(parents=True,exist_ok=True)
    for name in ['REPORT.md','delivery_receipt.json','review_file_manifest.json','source_hashes.json','code.diff']:
        dst=rev/name
        if not dst.exists(): shutil.copyfile(OUT/name,dst)
    report=OUT/'REPORT.md'
    text=report.read_text('utf-8-sig')
    old='| 官方train内N=128..256原始候选父 | 462 |'
    new='| 官方train内N=128..256原始候选网格条目 | 462（已知同源组417） |'
    assert old in text
    report.write_text(text.replace(old,new),encoding='utf-8-sig')
    event=dict(event='DELIVERY_WORDING_CORRECTION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='462 is a count of asset entries, not independent parents; inventory reports 417 known groups. No data, qualification, model, split or scientific result changed.',prior_delivery='delivery_revisions/v1/delivery_receipt.json')
    put(OUT/'delivery_correction.json',event)
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    hashes=[];diff=[]
    for p in sorted(CODE.iterdir()):
        if not p.is_file() or p.suffix not in ('.py','.md'):continue
        dst=OUT/'code_snapshot'/p.name
        if not dst.exists():shutil.copyfile(p,dst)
        assert sha(p)==sha(dst)
        hashes.append(dict(path=str(p),sha256=sha(p),snapshot=str(dst)))
        diff+=difflib.unified_diff([],p.read_text('utf-8-sig').splitlines(True),fromfile='/dev/null',tofile='b/'+p.relative_to(ROOT).as_posix())
    sources=json.loads((OUT/'source_hashes.json').read_text('utf-8'))
    sources['new_code']=hashes
    put(OUT/'source_hashes.json',sources)
    (OUT/'code.diff').write_text(''.join(diff),encoding='utf-8')
    include=[];exclude=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(OUT)
        if rel.as_posix() in ['R2_USER_DATA_SCALING_12H_V1_REVIEW.zip','review_file_manifest.json','delivery_receipt.json'] or '.tmp.' in p.name:continue
        omit=rel.parts[0]=='assets' or rel.parts[:2]==('data','parents') or (rel.suffix=='.png' and rel.parts[:2]==('data','review_sheets') and not rel.name.startswith('v2_'))
        item=dict(path=str(p),relative_path=rel.as_posix(),bytes=p.stat().st_size,sha256=sha(p),embedded_in_zip=not omit)
        (exclude if omit else include).append(item)
    put(OUT/'review_file_manifest.json',dict(status='DATA_SCALE_BLOCKED',included=include,excluded=exclude,weights='weights_manifest.json',standalone_training_package=False,revision=2))
    target=OUT/'R2_USER_DATA_SCALING_12H_V1_REVIEW.zip'
    tmp=OUT/'REVIEW.tmp.zip'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for r in include:z.write(r['path'],r['relative_path'])
        z.write(OUT/'review_file_manifest.json','review_file_manifest.json')
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        assert new in z.read('REPORT.md').decode('utf-8-sig')
        assert len([n for n in z.namelist() if n.startswith('data/conditions/') and n.endswith('.npz')])==286
    os.replace(tmp,target)
    receipt=dict(status='PASS_ZIP_CRC_AND_REQUIRED_CONTENTS',revision=2,zip=str(target),bytes=target.stat().st_size,sha256=sha(target),file_count=len(include)+1)
    put(OUT/'delivery_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
