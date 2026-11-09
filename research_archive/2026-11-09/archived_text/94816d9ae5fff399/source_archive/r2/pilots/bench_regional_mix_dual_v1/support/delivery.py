"""Offline gallery index and immutable evidence/source review package.

index may run before report completion. preflight/seal require the final report
review receipt. No model, geometry, detector, scoring or visual-label calls.
"""
from .. import runtime as R
import argparse
import difflib
import hashlib
import html
import io
import json
import os
from pathlib import Path
from urllib.parse import quote
import zipfile

CODE=R.ROOT/'r2/pilots/bench_regional_mix_dual_v1'
TEXT_SUFFIXES={'.json','.jsonl','.csv','.tsv','.md','.txt','.log','.html'}
LIMIT=10_000_000


def encoded(value):
    return (json.dumps(R.clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')


def digest(value):return hashlib.sha256(value).hexdigest()


def index():
    reg,rh,jobs=R.load_jobs(R.OUT/'jobs.json');render=R.read(R.OUT/'render_queue/complete.json')
    if render.get('status')!='COMPLETE' or render['pages']!=24 or render['cells']!=96:
        raise ValueError('All 24 exact render pages required')
    R.verify(render['files']);folder=R.OUT/'gallery';folder.mkdir(exist_ok=True)
    evidence=[];sections=[]
    for reference in reg['references']:
        q=reference['request'];rid=q['request_id'];cards=[]
        for label in ('ASSIST10','ASSIST01','ASSIST11','AUTO_COVER'):
            job=next(j for j in reg['jobs'] if j['request']['request_id']==rid and j['label']==label)
            receipt_path=R.job_dir(job['job_id'])/'gallery/CLEAN_FINAL_POST.json'
            receipt=R.read(receipt_path);image=R.resolve(receipt['image_path'])
            if receipt['registration_sha256']!=rh or R.sha(image)!=receipt['image_sha256']:
                raise ValueError('Rendered page identity differs')
            link=quote(Path(os.path.relpath(image,folder)).as_posix(),safe='/')
            meta=quote(Path(os.path.relpath(receipt_path,folder)).as_posix(),safe='/')
            cards.append(f'<article><h3>{html.escape(label)}</h3><a href="{link}"><img loading="lazy" src="{link}" alt="{html.escape(job["job_id"])}"></a><p><a href="{meta}">Image and column identity receipt</a></p></article>')
            evidence.append(dict(job_id=job['job_id'],request_id=rid,label=label,
                image=str(image),image_sha256=R.sha(image),receipt=str(receipt_path),receipt_sha256=R.sha(receipt_path)))
        sections.append('<section><h2>'+html.escape(rid)+'</h2><div class="cards">'+''.join(cards)+'</div></section>')
    content=('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bench regional mix: all 24 FINAL comparisons</title><style>
body{font:15px system-ui,sans-serif;margin:24px;color:#18202a;background:#f4f6f8}h1{font-size:24px}h2{font-size:16px;overflow-wrap:anywhere}h3{font-size:16px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}article{background:white;padding:12px;border:1px solid #cad2db;border-radius:6px}img{width:100%;height:auto}section{margin:28px 0}a{color:#1359a0}p{line-height:1.5}@media(max-width:900px){.cards{grid-template-columns:1fr}}</style>
<h1>Bench regional mixing: all 24 FINAL comparison pages</h1>
<p>Six fixed development requests; four candidates for every request. Each image has four columns: original RAW, original M_D0 chain, whole-RAW chain, and the named regional candidate. Each column includes two fixed views and two inherited-position sections. Click any image for its full original resolution.</p>
<p>ASSIST10 / ASSIST01 / ASSIST11 use the two preregistered assisted regions. AUTO_COVER is the fixed coverage heuristic. These are all candidates, without selecting a best result. Actual AI_PROXY review and limitations are preserved separately in <a href="../visual_review/PROTOCOL.md">the review protocol</a>.</p>
'''+''.join(sections)+'</html>\n').encode('utf-8')
    target=folder/'index.html';receipt=folder/'index.json'
    if target.exists():
        if target.read_bytes()!=content:raise ValueError('Existing gallery index differs; do not overwrite')
        old=R.read(receipt)
        if old['index_sha256']!=digest(content):raise ValueError('Index receipt differs')
        return old
    with target.open('xb') as handle:handle.write(content)
    result=dict(status='COMPLETE',registration_sha256=rh,pages=24,cells=96,
        index_path=str(target),index_sha256=digest(content),relative_paths_same_local_and_ZIP=True,
        pages_in_fixed_order=evidence,render_calls=0,scientific_calls=0,new_visual_labels=0)
    R.write_new(receipt,result);return result


def checked_readiness(review_path):
    reg,rh,jobs=R.load_jobs(R.OUT/'jobs.json');report=R.OUT/'REPORT.md';review=R.read(review_path)
    if review.get('status')!='PASS' or review.get('report_sha256')!=R.sha(report):
        raise ValueError('A PASS final review bound to current REPORT is required')
    required={
        'dispatch/complete.json':'COMPLETE','results/complete.json':'COMPLETE',
        'analysis/COMPLETE.json':'COMPLETE','render_queue/complete.json':'COMPLETE',
        'audits/execution/complete.json':'PASS','gallery/index.json':'COMPLETE'}
    for rel,status in required.items():
        record=R.read(R.OUT/rel)
        if record.get('status')!=status:raise ValueError('Missing complete stage: '+rel)
        if 'files' in record:R.verify(record['files'])
    for j in reg['jobs']:R.verify_result(R.read(R.job_dir(j['job_id'])/'complete.json'))
    for rel in ('REPORT.md','PROTOCOL.md','RUNBOOK.md','visual_review/PROTOCOL.md'):
        if not (R.OUT/rel).is_file():raise ValueError('Required delivery missing: '+rel)
    return reg,rh,review


def source_entries(reg):
    expected=dict(reg['source_hashes']);R.verify(expected)
    expected={str(R.resolve(p)):value for p,value in expected.items()}
    files={};records=[];seen={};new=[]
    paths={R.resolve(p) for p in expected}|set(CODE.rglob('*.py'))|{R.ROOT/'native_t1/docs/control_metrics.md'}
    for source in sorted(paths,key=str):
        if source.suffix not in ('.py','.md','.json','.txt'):
            raise ValueError('Unexpected frozen source type: '+str(source))
        data=source.read_bytes();h=digest(data)
        if str(source) in expected and h!=expected[str(source)]:raise ValueError('Frozen source changed')
        if source.is_relative_to(R.ROOT):relative=source.relative_to(R.ROOT).as_posix()
        else:relative='_external_sources/'+hashlib.sha256(str(source).encode()).hexdigest()[:16]+'/'+source.name
        arc='source_archive/'+relative
        if arc in seen and seen[arc]!=h:raise ValueError('Source archive collision')
        seen[arc]=h;files[arc]=data
        records.append(dict(original_path=str(source),archive_path=arc,sha256=h,bytes=len(data),
            frozen_scientific_source=str(source) in expected,current_experiment_source=source.is_relative_to(CODE)))
        if source.is_relative_to(CODE):
            rel=source.relative_to(R.ROOT).as_posix()
            lines=data.decode('utf-8-sig').splitlines(keepends=True)
            new.append('diff --git a/'+rel+' b/'+rel+'\nnew file mode 100644\n'+''.join(
                difflib.unified_diff([],lines,fromfile='/dev/null',tofile='b/'+rel)))
    files['source_archive/source_hashes.json']=encoded(dict(status='SOURCE_SNAPSHOT',files=records,
        policy='Frozen source hashes verified before copying. Support/analysis helpers are current non-scientific additions. No original model/scorer source was edited.'))
    files['source_archive/code.diff']=(''.join(new)).encode('utf-8')
    return files,records


def assemble(review_path):
    reg,rh,review=checked_readiness(review_path);files={};source_files,source_records=source_entries(reg)
    files.update(source_files);inventory=[];excluded=[]
    for p in sorted(R.OUT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(R.OUT).as_posix()
        if rel.startswith(('delivery/','source_archive/')):continue
        if p.suffix in ('.pt','.pth','.ckpt','.safetensors'):
            raise ValueError('Unexpected weight in CPU-only experiment; never archive it')
        data=p.read_bytes();h=digest(data)
        included=p.suffix.lower() in TEXT_SUFFIXES or p.suffix.lower()=='.png'
        inventory.append(dict(local_path=str(p),relative_path=rel,bytes=len(data),sha256=h,included_in_review=included))
        if included:files[rel]=data
        else:excluded.append(dict(path=str(p),sha256=h,bytes=len(data),reason='Array/binary asset retained locally; not duplicated in text-and-image review package'))
    # Include exact preserved ledgers of external RAW_ALL / D0 aliases.
    external=[]
    for j in reg['jobs']:
        receipt=R.read(R.job_dir(j['job_id'])/'complete.json')
        for key,p in receipt['artifacts'].items():
            source=R.resolve(p)
            if not source.is_file() or source.is_relative_to(R.OUT) or source.suffix not in TEXT_SUFFIXES:continue
            data=source.read_bytes();h=digest(data)
            name='external_evidence/'+hashlib.sha256(str(source).encode()).hexdigest()[:16]+'/'+source.name
            files[name]=data;external.append(dict(job_id=j['job_id'],role=key,original_path=str(source),archive_path=name,sha256=h,bytes=len(data)))
    files['external_evidence/index.json']=encoded(external)
    files['delivery/final_artifact_manifest.json']=encoded(dict(status='READ_ONLY_FINAL_SNAPSHOT',
        registration_sha256=rh,report_sha256=R.sha(R.OUT/'REPORT.md'),report_review_path=str(R.resolve(review_path)),
        report_review_sha256=R.sha(review_path),files=inventory,excluded_binary_assets=excluded,
        scientific_calls=0,new_visual_labels=0))
    files['README_REVIEW.txt']=('Open gallery/index.html for all 24 original-resolution comparison pages.\n'
        'REPORT.md and all original177/parent/request/slot/selection/connectivity tables are included.\n'
        'source_archive contains exact frozen source plus current support/analysis code; code.diff describes only this experiment additions.\n'
        'Original absolute paths inside scientific receipts are provenance references and were not rewritten. External reused JSON ledgers have an explicit external_evidence/index.json mapping.\n'
        'NPZ geometry/noise arrays remain locally with hashes in delivery/final_artifact_manifest.json. No weights or new model results are included.\n').encode()
    scientific_json={
        'report_sha256':R.sha(R.OUT/'REPORT.md'),'report_review_sha256':R.sha(review_path),
        'jobs_sha256':rh,'original_png_pages':24,'source_files':len(source_records),
        'members':[{ 'path':name,'bytes':len(data),'sha256':digest(data)} for name,data in sorted(files.items())],
        'policy':'All text tables/ledgers and actual review records included. Original binary arrays are indexed, not copied. No visual or numerical reselection.'}
    files['PACKAGE_MANIFEST.json']=encoded(scientific_json)
    return files,scientific_json


def compress(files,method):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=method,compresslevel=9 if method==zipfile.ZIP_DEFLATED else None) as archive:
        for name,data in sorted(files.items()):
            item=zipfile.ZipInfo(name,date_time=(2026,10,28,0,0,0));item.compress_type=method
            item.external_attr=0o100644<<16
            archive.writestr(item,data,compress_type=method,compresslevel=9 if method==zipfile.ZIP_DEFLATED else None)
    result=out.getvalue()
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC failed')
        if set(archive.namelist())!=set(files):raise ValueError('ZIP membership differs')
        for name,data in files.items():
            if digest(archive.read(name))!=digest(data):raise ValueError('ZIP member bytes differ: '+name)
    return result


def package(review_path,seal=False):
    files,manifest=assemble(review_path);data=compress(files,zipfile.ZIP_DEFLATED);method='DEFLATE'
    if len(data)>=LIMIT:
        alternative=compress(files,zipfile.ZIP_LZMA)
        if len(alternative)<len(data):data=alternative;method='LZMA'
    # Keep all original24 pages and full evidence even if the preferred cap
    # cannot be met. The honest byte count is returned; nothing is silently omitted.
    result=dict(status='PREFLIGHT_PASS',bytes=len(data),decimal_MB=len(data)/1_000_000,
        below_10MB=len(data)<LIMIT,compression=method,members=len(files),
        report_sha256=manifest['report_sha256'],report_review_sha256=manifest['report_review_sha256'],
        source_files=manifest['source_files'],original_gallery_pages=24,
        science_calls=0,new_visual_labels=0,zip_sha256=digest(data))
    if not seal:return result
    dest=R.OUT/'delivery';done=dest/'complete.json'
    if done.exists():
        old=R.read(done)
        if old['zip_sha256']!=result['zip_sha256']:raise ValueError('Sealed delivery differs; no overwrite')
        R.verify(old['files']);return old
    if (dest/'REVIEW.zip').exists() or (R.OUT/'source_archive').exists():
        raise ValueError('Partial archive retained for explicit review')
    # No original files are rewritten; materialize only the new deliverables.
    for name,blob in files.items():
        if name.startswith('source_archive/') or name=='delivery/final_artifact_manifest.json':
            target=R.OUT/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as handle:handle.write(blob)
    target=dest/'REVIEW.zip'
    with target.open('xb') as handle:handle.write(data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise ValueError('Written ZIP CRC failed')
        for name,blob in files.items():
            if digest(archive.read(name))!=digest(blob):raise ValueError('Written ZIP member hash differs')
    result.update(status='COMPLETE',utc=R.utc(),zip_path=str(target),
        files={str(p):R.sha(p) for folder in (dest,R.OUT/'source_archive') for p in folder.rglob('*') if p.is_file()})
    R.write_new(done,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('index','preflight','seal'))
    parser.add_argument('--report-review',default=str(R.OUT/'audits/report_review.json'))
    args=parser.parse_args()
    result=index() if args.action=='index' else package(args.report_review,args.action=='seal')
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','pages_in_fixed_order')},ensure_ascii=False))
