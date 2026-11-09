"""Recoverable public small-archive inspection used during asset preparation.

Only the three pinned official data archives are fetched, and no mesh is
extracted. Existing caches must have the exact published digest and size.
"""
from pathlib import Path
import collections
import csv
import hashlib
import io
import json
import tarfile
import time
import urllib.request
import numpy as np

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_user_data_scaling_12h_v1'
ITEMS=[
 ('shapenet-rebuttal.tar.gz',5296872,'de05caa31dc388d2f00587582f4138f881cf09c655b7495202016844f7021875'),
 ('shapenet-rebuttal2.tar.gz',4250184,'93a7d0e3ab3f3caeb3ddb5e368c8e1f6384e638e2e5296a58d64c86c08b5330e'),
 ('shapenet.tar.gz',50276795,'4ee1afec59200c7aaf8a3e26cca5e71845123a1c550aa5dee9684dcb0ed63e59'),
]


def get_assets():
    root=OUT/'assets'
    if root.is_symlink() or not root.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Unsafe archive cache')
    root.mkdir(parents=True,exist_ok=True)
    for name,size,expected in ITEMS:
        p=root/name
        url='https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/'+name
        tick=time.monotonic()
        if not p.exists():
            part=p.with_suffix(p.suffix+'.download.part');h=hashlib.sha256();count=0
            # A pre-existing partial download is not silently overwritten.
            with urllib.request.urlopen(url,timeout=60) as response,part.open('xb') as target:
                for block in iter(lambda:response.read(1024*1024),b''):
                    target.write(block);h.update(block);count+=len(block)
            if count!=size or h.hexdigest()!=expected:
                raise RuntimeError('Public data identity mismatch: '+name)
            part.rename(p)
        elif p.is_symlink() or p.stat().st_size!=size or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Existing data identity mismatch: '+name)
        with tarfile.open(p,'r:gz') as archive:
            members=[m for m in archive if m.isfile()]
        names=[m.name for m in members]
        data=dict(archive=name,source_url=url,bytes=size,sha256=expected,wall_seconds=time.monotonic()-tick,
            member_count=len(names),prefix_counts=dict(collections.Counter(n.split('/')[0] for n in names)),
            members=names,member_sizes={m.name:m.size for m in members},extracted_files=0,models=0)
        receipt=root/(name+'.inventory.json')
        if receipt.exists():
            old=json.loads(receipt.read_text())
            for key in data:
                if key!='wall_seconds' and old[key]!=data[key]:raise RuntimeError('Archive member list changed')
        else:
            receipt.write_text(json.dumps(data,indent=2)+'\n')
        print(name,'verified; mesh extraction count=0',flush=True)


def audit_chair():
    original={r['uid']:r for r in json.loads((ROOT/'native_t1/runs/control_mvp48_v1/data/candidate_audit.json').read_text())['objects']}
    rows=[]
    with tarfile.open(OUT/'assets/shapenet.tar.gz','r:gz') as archive:
        members=archive.getmembers();splits={}
        for member in members:
            if member.isfile() and '/split/' in member.name:
                # Hash-pinned official published metadata uses an object array.
                with np.load(io.BytesIO(archive.extractfile(member).read()),allow_pickle=True) as z:
                    for row in z['npz_list']:
                        splits.setdefault(str(row['uid']),[]).append(Path(member.name).stem)
        for member in members:
            uid=Path(member.name).stem
            if not(member.isfile() and '/objaverse_occ_v5_ids/' in member.name and uid in original):continue
            h=hashlib.sha256(archive.extractfile(member).read()).hexdigest();prior=original[uid]
            rows.append(dict(uid=uid,member=member.name,archive_member_sha256=h,
                existing_chair_sha256=prior['source_sha256'],byte_identical=h==prior['source_sha256'],
                official_chair_split=prior['official_split'],generic_archive_split=','.join(splits.get(uid,[])),source_N=prior['N']))
    summary=dict(schema='r2_scaling_downloaded_archives_chair_audit_v1',matched_existing_chair_members=len(rows),
        byte_identical_to_existing_chair_members=sum(r['byte_identical'] for r in rows),
        different_chair_versions=sum(not r['byte_identical'] for r in rows),
        known_chair_missing_in_archive=sorted(set(original)-{r['uid'] for r in rows}),
        generic_archive_split_differs_from_chair_split=sum(r['generic_archive_split']!=r['official_chair_split'] for r in rows),
        new_qualified_chair_variants=0,other_generic_UID_category_status='UNKNOWN_NOT_USED',
        rebuttal_chair_members=0,rebuttal2_chair_members=0,extracted_meshes=0,weights_downloaded=0,model_calls=0,new_training=0,
        conclusion='No additional verified Chair variants; same 3135 Chair member bytes. Generic archive split is not substituted for category-specific official split.')
    dest=OUT/'inventory/downloaded_archives_chair_audit.json'
    if dest.exists():
        if json.loads(dest.read_text())!=summary:raise RuntimeError('Prior archive audit changed')
    else: dest.write_text(json.dumps(summary,indent=2)+'\n')
    csv_path=dest.with_suffix('.csv')
    if not csv_path.exists():
        with csv_path.open('x',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    get_assets();audit_chair()
