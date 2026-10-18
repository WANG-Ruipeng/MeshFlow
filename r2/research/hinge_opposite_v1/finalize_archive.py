"""Verify and package the two explicitly authorized publication directories."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
import verify_archive

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
OUT=REPO/'experiments/r2_hinge_opposite_v1'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    result=verify_archive.run(check_import=True)
    assert result['status'].startswith('PASS'),result
    receipt=json.loads((HERE/'validation/portable_decode_receipt.json').read_text(encoding='utf-8'))
    assert receipt['status']=='PASS'
    assert len(receipt['original_array_comparison'])==63
    assert all(r['byte_equal'] for r in receipt['original_array_comparison'])
    result['portable_array_count']=len(receipt['original_array_comparison'])
    result['portable_module_count']=len(receipt['scientific_modules'])
    dump(HERE/'validation/archive_verification.json',result)
    support=[p for p in HERE.iterdir() if p.is_file() and p.name!='support_manifest.json']
    dump(HERE/'support_manifest.json',dict(files=[dict(path=p.name,sha256=digest(p),bytes=p.stat().st_size)
                                                for p in sorted(support)]))
    files=[]
    for folder in (REPO/'r2/tools/hinge_opposite_v1',HERE):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
                assert p.suffix not in ('.pt','.pth','.ckpt','.npz','.npy'),p
                files.append(p)
    target=OUT/'HINGE_OPPOSITE_V1_REVIEW.zip'
    if target.exists():
        raise RuntimeError('REFUSE_EXISTING_FINAL_ZIP: '+str(target))
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            info=zipfile.ZipInfo(p.relative_to(REPO).as_posix(),date_time=(2026,10,18,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes(),compresslevel=9)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)
        for p in files:
            assert z.read(p.relative_to(REPO).as_posix())==p.read_bytes(),p
    result=dict(status='PASS_FINAL_ARCHIVE',files=len(files),
                uncompressed_bytes=sum(p.stat().st_size for p in files),
                zip=str(target),zip_bytes=target.stat().st_size,zip_sha256=digest(target),
                verification=result,git_operations=0,model_calls=0,
                publication_directories=['r2/tools/hinge_opposite_v1','r2/research/hinge_opposite_v1'])
    dump(OUT/'archive_delivery_receipt.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='verification'},indent=2))


if __name__=='__main__':
    main()
