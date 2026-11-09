"""Create the compact, fully indexed review ZIP only after explicit --build.

Read-only artifact collection: no model, metric computation, mesh modification,
rendering, image recompression or large-checkpoint rehash. All CSV rows and all
32 original JPEG pages are retained. Missing completion gates prevent a build.
"""
from __future__ import annotations
import argparse, csv, difflib, hashlib, io, json, os, re, time, urllib.parse, zipfile
from pathlib import Path, PurePosixPath
try:
    from . import common as C
except ImportError:
    import common as C

ZIP_NAME='R2_CONDITION_WIDTH_RESPONSE_V1_REVIEW.zip'
TARGET_BYTES=10_000_000
SKIP_SCORE_NAMES={'pairs.json','shape_per_face.json','orientation.json'}
REQUIRED=[
 'REPORT.md','README_RESUME.md','registration.json','selection.json',
 'analysis/definition.json','analysis/response_completion.json',
 'analysis/per_output_geometry.csv','analysis/per_curve_response.csv',
 'analysis/per_fixed_state_response.csv','analysis/per_free_face_velocity_response.csv',
 'analysis/per_trajectory_response.csv','evaluation/raw_index.json','evaluation/post_index.json',
 'evaluation/score_completion.json','evaluation/gallery/manifest.json',
 'evaluation/gallery/index.html','evaluation/gallery/output_review_receipt.json',
 'evaluation/gallery/AI_PROXY_visual_review.csv','audit/baseline_reproduction.json',
 'audit/protected_asset_hashes.json','execution/generator_complete.json','execution/D0_complete.json']

def hbytes(b):return hashlib.sha256(b).hexdigest()
def js(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
def csvbytes(rows):
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=keys);w.writeheader()
    for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    return s.getvalue().encode('utf-8-sig')

def output_files():
    excluded_roots={'delivery_review_contents','delivery_drafts','__pycache__'}
    for p in C.OUT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(C.OUT)
        if any(x in excluded_roots for x in rel.parts):continue
        if p.name==ZIP_NAME or p.name.startswith(('delivery_receipt','package_preflight')) or p.suffix in ('.tmp','.zip','.pyc'):continue
        yield p,rel.as_posix()

def include_file(p,rel):
    # Full row-bearing tables are never filtered by value, method or outcome.
    if p.suffix.lower()=='.csv':return True
    # Compact review projection: full scalar rows remain in the unfiltered CSVs.
    # Duplicated per-cell JSON/call logs stay local with hashes and full paths.
    if rel.startswith(('evaluation/raw/','evaluation/post/','evaluation/scores/')) and p.suffix in ('.json','.jsonl'):return False
    if rel.startswith(('inputs/','sources/')) and p.suffix=='.npz':return True
    if p.suffix in ('.npz','.npy','.ply','.pt','.pth','.lock'):return False
    if '/scores/' in '/'+rel and p.name in SKIP_SCORE_NAMES:return False
    return p.suffix.lower() in ('.json','.jsonl','.md','.txt','.html','.jpg','.jpeg','.png','.py','.diff','.patch','.log')

def current_jobs():
    found=[]
    if not Path('/proc').is_dir():return found
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            pid=int(p.parent.name)
            if pid==os.getpid():continue
            args=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if ('python' in args and 'condition_width_response_v1' in args
                and any(x in args for x in ('.gpu ', '/gpu.py ','.evaluate ', '/evaluate.py ','.rendering ', '/rendering.py ','.analyze ', '/analyze.py '))):
                found.append(dict(pid=pid,command=args))
        except (OSError,ValueError):pass
    return found

def preflight():
    missing=[x for x in REQUIRED if not (C.OUT/x).is_file()]
    jpgs=sorted((C.OUT/'evaluation/gallery').glob('*/*.jpg'))
    failures=[]
    if len(jpgs)!=32:failures.append('Expected all32 existing RAW/POST JPEG pages; found '+str(len(jpgs)))
    review=C.OUT/'evaluation/gallery/AI_PROXY_visual_review.csv'
    if review.is_file():
        with review.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        if len(rows)!=96 or any(r.get('review_status')!='REVIEWED_AI_PROXY' or not r.get('reviewed_image_sha256') for r in rows):
            failures.append('Need all96 actually reviewed output rows and image identities')
    for relative,n in [('evaluation/raw_index.json',48),('evaluation/post_index.json',48)]:
        p=C.OUT/relative
        if p.exists():
            a=C.read(p)
            if a.get('status')!='COMPLETE' or len(a.get('rows',[]))!=n:failures.append(relative+' is not COMPLETE48')
    running=current_jobs()
    if running:failures.append('An experiment scientific/analysis writer is still active')
    chosen=[(p,r) for p,r in output_files() if include_file(p,r)]
    return dict(status='READY' if not missing and not failures else 'NOT_READY',missing=missing,failures=failures,
        active_jobs=running,selected_existing_files=len(chosen),selected_existing_uncompressed_bytes=sum(p.stat().st_size for p,_ in chosen),
        JPEG_pages=len(jpgs),JPEG_original_bytes=sum(p.stat().st_size for p in jpgs),
        target_bytes=TARGET_BYTES,code_and_manifests_additional=True,
        expected_size_note='Final size is authoritative; all CSV rows and original JPEGs retained; redundant per-cell JSON/call logs indexed locally',
        final_packaging_invoked=False)

def source_snapshot():
    expected={};origins={}
    def add(mapping,receipt):
        for path,h in mapping.items():
            p=C.resolve(path).resolve()
            if not p.is_file():raise FileNotFoundError('Required actual source missing: '+str(p))
            k=str(p)
            if k in expected and expected[k]!=h:raise ValueError('Conflicting frozen code identity for '+k)
            expected[k]=h;origins.setdefault(k,[]).append(receipt)
    reg=C.read(C.OUT/'registration.json');add(reg['preparation_source_hashes'],'registration.preparation_source_hashes')
    freeze=C.read(C.OUT/'execution/gpu_code_freeze.json');add(freeze['files'],'execution/gpu_code_freeze.json')
    impl=C.read(C.OUT/'evaluation/implementation_ready.json')
    add(impl['scoring_source_hashes'],'evaluation/implementation_ready.scoring_source_hashes')
    add(impl['rendering_source_hashes'],'evaluation/implementation_ready.rendering_source_hashes')
    # New engineering/analysis helpers are collected at final build time.
    for p in C.CODE.glob('*.py'):
        key=str(p.resolve())
        if key not in expected:expected[key]=None;origins[key]=['actual final new helper']
    payload={};rows=[];newdiff=[]
    for path in sorted(expected):
        p=Path(path);b=p.read_bytes();h=hbytes(b)
        if expected[path] is not None and h!=expected[path]:raise ValueError('Frozen source changed before delivery: '+path)
        if p.is_relative_to(C.ROOT):rel=p.relative_to(C.ROOT).as_posix();member='code/MeshFlow/'+rel
        else:
            anchor=Path('/mnt/e/MeshFlow-Control')
            if p.is_relative_to(anchor):member='code/MeshFlow-Control/'+p.relative_to(anchor).as_posix()
            else:member='code/external/'+h[:12]+'_'+p.name
        if member in payload and payload[member]!=b:raise ValueError('Code snapshot collision')
        payload[member]=b;rows.append(dict(source_path=path,package_path=member,bytes=len(b),sha256=h,
            expected_frozen_sha256=expected[path] or 'NEW_FINAL_HELPER',hash_match=True,provenance=origins[path]))
        if p.parent==C.CODE:
            text=b.decode('utf-8-sig');newdiff.extend(difflib.unified_diff([],text.splitlines(keepends=True),fromfile='/dev/null',tofile=member,n=3))
    payload['code.diff']=''.join(newdiff).encode('utf-8')
    payload['source_hashes.csv']=csvbytes(rows)
    payload['code/README.md']=('实际执行的新工具与冻结依赖的小源码快照。code.diff 将本轮新脚本表示为新增文件；不声称修改了旧评分器。\n'
        '旧评分、模型、D0/GEO/定向代码保持冻结定义；工程失败版和修正记录另在 engineering/。\n'
        '此审阅包不含模型权重、大RAW/probe/查询数组，也不是可独立训练的仓库。\n').encode('utf-8')
    return payload,rows

def reference_hashes(doc,known):
    if isinstance(doc,dict):
        for k,v in doc.items():
            if isinstance(k,str) and (k.startswith('/mnt/') or re.match('^[A-Za-z]:',k)) and isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v):
                known[str(C.resolve(k).resolve())]=v
            if isinstance(k,str) and k.endswith('_path') and isinstance(v,str):
                h=doc.get(k[:-5]+'_sha256')
                if isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h):known[str(C.resolve(v).resolve())]=h
        if isinstance(doc.get('path'),str):
            h=doc.get('sha256') or doc.get('actual_sha256')
            if isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h):known[str(C.resolve(doc['path']).resolve())]=h
        for v in doc.values():reference_hashes(v,known)
    elif isinstance(doc,list):
        for v in doc:reference_hashes(v,known)

def local_assets(files):
    known={};rows=[]
    # Parse only receipt/metadata documents, not huge pair or face diagnostics.
    for p,rel in files:
        if p.suffix=='.json' and p.name not in SKIP_SCORE_NAMES and p.stat().st_size<1_500_000:
            try:reference_hashes(C.read(p),known)
            except (ValueError,UnicodeError):pass
    targets=[]
    for p,rel in files:
        if not include_file(p,rel):targets.append((p,rel,'OMITTED_LOCAL_ARRAY_OR_LARGE_DIAGNOSTIC'))
    reg=C.read(C.OUT/'registration.json')
    for key in ('checkpoint','d0_checkpoint'):
        p=C.resolve(reg[key]['path']);targets.append((p,None,'PROTECTED_MODEL_'+key));known[str(p.resolve())]=reg[key]['sha256']
    for q in reg['requests']:
        for pathfield,hashfield in [('original_input_path','original_input_sha256'),('original_condition_source','original_condition_source_sha256')]:
            p=C.resolve(q[pathfield]);targets.append((p,None,'READ_ONLY_INHERITED_SOURCE'));known[str(p.resolve())]=q[hashfield]
    unique={str(p.resolve()):(p,rel,reason) for p,rel,reason in targets}
    for key,(p,rel,reason) in sorted(unique.items()):
        if not p.is_file():raise FileNotFoundError('Local indexed asset absent: '+str(p))
        actual=known.get(key)
        if actual:mode='EXISTENCE_SIZE_NOW_AND_RECORDED_RECEIPT_SHA256'
        else:
            if p.stat().st_size>1_000_000_000:raise ValueError('No recorded hash for a large asset; refusing redundant full scan')
            actual=C.sha(p);mode='ACTUAL_CPU_HASH_AT_PACKAGE_ONCE'
        rows.append(dict(path=key,out_relative_path=rel or '',bytes=p.stat().st_size,sha256=actual,hash_evidence=mode,reason=reason,in_zip=False))
    return rows

def verify_gallery_links(payload):
    name='evaluation/gallery/index.html';text=payload[name].decode('utf-8-sig');links=[]
    for link in re.findall(r'(?:href|src)=[\"\']([^\"\']+)',text):
        if '://' in link or link.startswith(('data:','#','mailto:')):continue
        local=urllib.parse.unquote(link.split('#')[0].split('?')[0])
        # Gallery uses ordinary relative links; reject escaping the package.
        target=(PurePosixPath(name).parent/local).as_posix()
        if '..' in PurePosixPath(target).parts:raise ValueError('Unexpected gallery parent traversal: '+link)
        if target not in payload:raise ValueError('Missing gallery target: '+target)
        links.append(target)
    jpgs=[k for k in payload if k.startswith('evaluation/gallery/') and k.endswith('.jpg')]
    if len(jpgs)!=32 or not set(jpgs)<=set(links):raise ValueError('Gallery must link all32 original JPEG pages')
    return dict(status='PASS',relative_links_checked=len(links),unique_targets=len(set(links)),original_JPEG_pages=len(jpgs))

def build():
    started=time.perf_counter();pre=preflight()
    if pre['status']!='READY':raise RuntimeError('REVIEW_PACKAGE_NOT_READY: '+json.dumps(pre,ensure_ascii=False))
    final=C.checked(C.OUT/ZIP_NAME)
    if final.exists():raise FileExistsError('Final review ZIP already exists; no silent overwrite')
    files=list(output_files());payload={};sourcepaths={}
    for p,rel in files:
        if include_file(p,rel):payload[rel]=p.read_bytes();sourcepaths[rel]=str(p)
    code,code_rows=source_snapshot()
    for k,v in code.items():
        if k in payload:raise ValueError('Member collision: '+k)
        payload[k]=v
    assets=local_assets(files);payload['local_assets_manifest.csv']=csvbytes(assets)
    payload['PACKAGE_README.md']=('本包用于审阅，不是独立训练/采样环境。\n\n'
        '包含最终报告、全部既有CSV行、完整32页原JPEG图集与逐输出实际AI_PROXY记录、定义/注册、实际源码/冻结依赖及审计收据。\n'
        '打开 evaluation/gallery/index.html 可浏览全部图；JPEG未重绘、未裁剪、未重编码。\n'
        '大权重、RAW/POST/probe数组、冻结查询与巨大面对/逐面/方向细节留本地，见 local_assets_manifest.csv。部分小输入与来源映射NPZ保留于包内。\n'
        '本地索引优先沿用已验证收据哈希；当前核对存在性和大小，不再次读取7GB去噪权重。\n'
        'MANIFEST.sha256 覆盖除自身以外所有ZIP成员；其自身不可能递归自哈希。SHA256只记录完整性，不替代模型或评分验证。\n'
        'CSV未删行，源码未按结果筛选。原始文件绝对路径保留追溯意义；包内HTML图链接可独立浏览。\n').encode('utf-8')
    links=verify_gallery_links(payload)
    report_links=[]
    for link in re.findall(r'\]\(([^)]+)\)', payload['REPORT.md'].decode('utf-8-sig')):
        link=link.strip().strip('<>')
        if '://' in link or link.startswith(('#','mailto:')): continue
        target=urllib.parse.unquote(link.split('#')[0].split('?')[0])
        if target not in payload: raise ValueError('Missing report package-relative target: '+target)
        report_links.append(target)
    report_link_check=dict(status='PASS',relative_links_checked=len(report_links),targets=report_links)
    manifest=[dict(member=k,bytes=len(b),sha256=hbytes(b),source_path=sourcepaths.get(k,'GENERATED_BY_PACKAGE_OR_CODE_SNAPSHOT')) for k,b in sorted(payload.items())]
    payload['package_file_manifest.csv']=csvbytes(manifest)
    sha_lines=[hbytes(b)+'  '+k for k,b in sorted(payload.items())]
    payload['MANIFEST.sha256']=('\n'.join(sha_lines)+'\n').encode('utf-8')
    tmp=C.checked(final.with_name(final.name+'.tmp'))
    with zipfile.ZipFile(tmp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9,allowZip64=True) as z:
        for name,b in sorted(payload.items()):
            method=zipfile.ZIP_STORED if name.lower().endswith(('.jpg','.jpeg','.png','.npz')) else zipfile.ZIP_DEFLATED
            z.writestr(name,b,compress_type=method,compresslevel=None if method==zipfile.ZIP_STORED else 9)
    os.replace(tmp,final)
    # One complete member verification pass, and one final compressed-file hash.
    with zipfile.ZipFile(final) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(payload):raise ValueError('ZIP member set mismatch')
        for name in names:
            if hbytes(z.read(name))!=hbytes(payload[name]):raise ValueError('ZIP member SHA mismatch: '+name)
    n=final.stat().st_size;ziphash=C.sha(final)
    receipt=dict(status='COMPLETE' if n<TARGET_BYTES else 'COMPLETE_OVER_SIZE_TARGET',utc=C.utc(),path=str(final),bytes=n,MB=n/1e6,MiB=n/(1024**2),
        sha256=ziphash,members=len(payload),CSV_members=sum(k.endswith('.csv') for k in payload),original_JPEG_pages=32,
        actual_source_files=len(code_rows),local_asset_index_rows=len(assets),local_asset_bytes=sum(a['bytes'] for a in assets),
        all_member_SHA256='PASS_SINGLE_COMPLETE_VERIFICATION',CRC='PASS_ALL_MEMBERS_READ_BY_ZIPFILE',gallery_links=links,report_links=report_link_check,weight_files_in_zip=0,new_model_calls=0,new_metric_calls=0,new_renders=0,
        size_target_bytes=TARGET_BYTES,within_target=n<TARGET_BYTES,seconds=time.perf_counter()-started,
        implementation_sha256=C.sha(__file__),no_training_or_sampling_environment_claim=True)
    C.write(C.OUT/'delivery_receipt.json',receipt)
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');a=p.parse_args()
    print(json.dumps(build() if a.build else preflight(),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
