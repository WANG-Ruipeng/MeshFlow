"""Build the small review package after the completed report and PASS audit.

No models, scoring, raw arrays or weights are loaded. All 56 gallery figures
are included as disclosed review copies; full originals remain unchanged.
"""
from __future__ import annotations
import argparse,csv,difflib,hashlib,io,json,subprocess,zipfile
from pathlib import Path
from . import common as C

OUT=C.OUT
ZIP=OUT/'R2_C_REFERENCE_FRAME_V1_REVIEW_SMALL.zip'
RECEIPT=OUT/'PACKAGE_RECEIPT.json'
MAX_BYTES=10_000_000
IMAGE_WIDTH=1800
IMAGE_QUALITY=60
TEXT_SUFFIXES={'.csv','.json','.jsonl','.md','.txt','.html'}
EXCLUDED_ROOT={'PACKAGE_RECEIPT.json','PACKAGE_SIZE_BLOCKED.json'}


def digest_bytes(value):return hashlib.sha256(value).hexdigest()


def verify(mapping):
    for p,h in mapping.items():
        if C.sha(p)!=h:raise ValueError('Package source changed: '+str(p))


def file_rows(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as file:return list(csv.DictReader(file))


def prerequisites():
    for p in (OUT/'REPORT.md',OUT/'README_RESUME.md',OUT/'COMPLETE.json'):
        if not p.is_file():raise RuntimeError('Await final delivery prerequisite: '+str(p))
    completed=C.read(OUT/'COMPLETE.json')
    if completed.get('status')!='COMPLETE':raise RuntimeError('Experiment is not COMPLETE')
    audit=C.read(OUT/'independent_audit/final_audit.json')
    if audit.get('status')!='PASS':raise RuntimeError('Independent final audit is not PASS')
    accounting=C.read(OUT/'call_accounting.json')
    if accounting.get('status')!='PASS':raise RuntimeError('Actual call accounting is not PASS')
    gallery=C.read(OUT/'evaluation/gallery/manifest.json')
    if (gallery['request_pages'],gallery['POST_noise_sheets'],gallery['review_rows'])!=(48,8,384):
        raise ValueError('Gallery budget mismatch')
    verify(gallery['output_hashes'])
    # Every registered anonymous row must have an actual saved judgment.
    reviewed={}
    paths=sorted((OUT/'evaluation/gallery').glob('review_*.csv'))
    for p in paths:
        for row in file_rows(p):
            if not {'request_id','anonymous_label','stage'}<=set(row):continue
            key=(row['request_id'],row['anonymous_label'],row['stage'])
            if row.get('review_status')!='REVIEWED':raise ValueError('Unreviewed visual row')
            if not row.get('reviewed_image_sha256') or row['reviewed_image_sha256']!=row['image_sha256']:
                raise ValueError('Missing actual reviewed-image identity')
            if C.sha(row['image_path'])!=row['reviewed_image_sha256']:raise ValueError('Reviewed master image changed')
            if key in reviewed:raise ValueError('Duplicate visual review row: '+str(key))
            judgments=('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer','whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')
            if any(row.get(field) not in ('YES','NO','UNCERTAIN') for field in judgments):raise ValueError('Invalid actual visual judgment')
            reviewed[key]=row
    expected={(r['request_id'],r['anonymous_label'],r['stage']) for r in file_rows(OUT/'evaluation/gallery/AI_PROXY_review_template.csv')}
    if set(reviewed)!=expected or len(reviewed)!=384:raise ValueError('Not all 384 anonymous visual rows are saved')
    style={}
    for p in sorted((OUT/'evaluation/gallery').glob('style_*.csv')):
        for row in file_rows(p):
            if not {'condition_id','anonymous_label','stage'}<=set(row):continue
            key=(row['condition_id'],row['anonymous_label'],row['stage'])
            if key in style:raise ValueError('Duplicate style review row')
            if row.get('review_status')!='REVIEWED' or row.get('near_identical_across_noises') not in ('YES','NO','UNCERTAIN'):raise ValueError('Unreviewed noise style row')
            if not row.get('coarse_style_variation') or row['coarse_style_variation']=='UNASSESSED':raise ValueError('Missing style description')
            if row.get('reviewed_image_sha256')!=row['image_sha256'] or C.sha(row['image_path'])!=row['image_sha256']:raise ValueError('Style image identity mismatch')
            style[key]=row
    expected_style={(r['condition_id'],r['anonymous_label'],r['stage']) for r in file_rows(OUT/'evaluation/gallery/AI_PROXY_noise_style_template.csv')}
    if set(style)!=expected_style or len(style)!=32:raise ValueError('Not all 32 style judgments saved')
    return gallery


def source_destination(path):
    p=C.resolve(path).resolve()
    if p.is_relative_to(C.ROOT.resolve()):return 'code/MeshFlow/'+p.relative_to(C.ROOT.resolve()).as_posix()
    # Preserve the external scorer's package layout instead of flattening names.
    for parent in p.parents:
        if parent.name in ('MeshFlow-Control','MeshFlow-Control-artifacts'):
            return 'code/external/'+parent.name+'/'+p.relative_to(parent).as_posix()
    raise ValueError('Unexpected source-code root: '+str(p))


def newline_diff(paths):
    chunks=[]
    for p in sorted(paths):
        rel=p.relative_to(C.ROOT).as_posix()
        tracked=subprocess.run(['git','ls-files','--error-unmatch',rel],cwd=C.ROOT,capture_output=True)
        if tracked.returncode==0:raise ValueError('Pilot source unexpectedly tracked; use an actual baseline diff: '+rel)
        chunks.append('diff --git a/'+rel+' b/'+rel+'\nnew file mode 100644\n')
        chunks.extend(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(True),fromfile='/dev/null',tofile='b/'+rel))
    return ''.join(chunks).encode('utf8')


def build():
    gallery=prerequisites()
    if RECEIPT.exists():
        prior=C.read(RECEIPT);verify(prior['source_hashes']);verify({ZIP:prior['zip_sha256']})
        if ZIP.stat().st_size>=MAX_BYTES:raise ValueError('Existing package exceeds size limit')
        return prior
    entries={};source_hashes={};provenance=[]
    def put(name,data):
        if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe archive member')
        if name in entries and entries[name]!=data:raise ValueError('Archive collision: '+name)
        entries[name]=data
    def include(p,name=None):
        p=C.resolve(p);name=name or p.relative_to(OUT).as_posix();data=p.read_bytes()
        put(name,data);h=digest_bytes(data);source_hashes[str(p)]=h
        provenance.append(dict(original_path=str(p),archive_path=name,sha256=h,kind='BYTE_IDENTICAL'))
    # Full small tables and manifests; exclude per-cell mesh/trajectory directories.
    selected=[p for p in OUT.iterdir() if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES and p.name not in EXCLUDED_ROOT]
    for relative in ('inputs','evaluation','evaluation/gallery','evaluation/blinding','execution'):
        folder=OUT/relative
        if folder.is_dir():selected += [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES]
    for relative in ('independent_audit','analysis','reviews','results','accounting','preparation','plans','training','resources','evaluation/execution'):
        folder=OUT/relative
        if folder.is_dir():selected += [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES]
    engineering=OUT/'engineering'
    if engineering.is_dir():selected += [p for p in engineering.rglob('*') if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES|{'.py','.diff'}]
    for p in sorted(set(selected)):include(p)
    registration=C.read(OUT/'evaluation/registration.json')
    completion=C.read(OUT/'evaluation/score_completion.json')
    training=C.read(OUT/'registration.json')
    training_sources={p:h for p,h in training['source_hashes'].items() if Path(p).suffix=='.py'}
    from .accounting import effective_evaluation_hashes
    evaluation_sources,amendments=effective_evaluation_hashes(registration)
    frozen_sources={**training_sources,**evaluation_sources,**completion['definition_hashes'],**gallery['identity']['definition_hashes']}
    verify(frozen_sources)
    pilot_sources=sorted(C.CODE.glob('*.py'))
    code_sources={C.resolve(p) for p in frozen_sources}|set(pilot_sources)
    snapshot_rows=[]
    for p in sorted(code_sources):
        archive_path=source_destination(p);include(p,archive_path)
        snapshot=OUT/'source_code'/archive_path.removeprefix('code/')
        data=p.read_bytes();C._write_bytes(snapshot,data)
        snapshot_rows.append(dict(original_path=str(p),snapshot_path=str(snapshot),archive_path=archive_path,sha256=digest_bytes(data)))
    diff_payload=newline_diff(pilot_sources)
    C._write_bytes(OUT/'code.diff',diff_payload);put('code.diff',diff_payload)
    source_index=dict(status='FROZEN_SNAPSHOT',files=snapshot_rows,
                      code_diff=dict(path=str(OUT/'code.diff'),sha256=digest_bytes(diff_payload)),
                      meaning='Original scoring implementations and this new pilot; all source bytes preserved. code.diff is the added pilot files against /dev/null.')
    C.write(OUT/'source_file_hashes.json',source_index)
    include(OUT/'source_file_hashes.json')
    from PIL import Image
    images=file_rows(OUT/'evaluation/gallery/images.csv')+file_rows(OUT/'evaluation/gallery/POST_contact_sheets.csv')
    if len(images)!=56 or len({row['image_path'] for row in images})!=56:raise ValueError('Not exactly 56 unique images')
    copy_rows=[]
    for row in images:
        p=C.resolve(row['image_path']);original_sha=C.sha(p)
        if original_sha!=row['image_sha256']:raise ValueError('Image source differs from gallery')
        with Image.open(p) as opened:
            image=opened.convert('RGB');original_size=image.size
            if image.width>IMAGE_WIDTH:
                image=image.resize((IMAGE_WIDTH,round(image.height*IMAGE_WIDTH/image.width)),Image.Resampling.LANCZOS)
            buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=IMAGE_QUALITY,optimize=True)
            payload=buffer.getvalue();copy_size=image.size
        name=p.relative_to(OUT).as_posix();put(name,payload);source_hashes[str(p)]=original_sha
        copy_rows.append(dict(original_path=str(p),archive_path=name,original_sha256=original_sha,copy_sha256=digest_bytes(payload),
                              original_width=original_size[0],original_height=original_size[1],
                              copy_width=copy_size[0],copy_height=copy_size[1],JPEG_quality=IMAGE_QUALITY))
    image_stream=io.StringIO(newline='');writer=csv.DictWriter(image_stream,fieldnames=list(copy_rows[0]))
    writer.writeheader();writer.writerows(copy_rows);put('IMAGE_COPY_MANIFEST.csv',image_stream.getvalue().encode('utf-8-sig'))
    source_stream=io.StringIO(newline='');writer=csv.DictWriter(source_stream,fieldnames=list(provenance[0]))
    writer.writeheader();writer.writerows(provenance);put('SOURCE_FILES_SHA256.csv',source_stream.getvalue().encode('utf-8-sig'))
    notice="""# Small review evidence package

All 48 request pages and all 8 POST noise contact sheets are present, without image selection.
The image files in this archive are review copies: maximum width 1800 pixels, JPEG quality 60.
Full-resolution master images remain unchanged in the local experiment. The saved AI_PROXY judgments
were made on those masters, not on these compressed copies. Original and copy hashes and dimensions
are separately recorded in IMAGE_COPY_MANIFEST.csv. Existing CSV image hashes identify originals;
they were not silently rewritten to identify the compressed copies.

Tables, registration, review records, report, audit and included code are byte-identical copies.
SOURCE_FILES_SHA256.csv records their source paths and hashes. The actual original metric/scoring
implementations are included under code/external and code/MeshFlow, alongside the new pilot code.
code.diff describes these newly added pilot files against /dev/null, not a modification of old scorers.

Weights, RAW/POST geometry arrays, saved Gaussian inputs, trajectories and per-point arrays are excluded.
The full local files remain hash-addressed by the manifests. This is a review/re-aggregation package;
it cannot independently rerun geometric scoring without those excluded arrays and the local dependencies.
The recovery entry remains README_RESUME.md; it is not a promise that this small ZIP installs the model.

The model key is included separately under evaluation/blinding, after the anonymous reviews were saved.
Open REPORT.md for interpretation and material costs; use evaluation/gallery/index.html for figures.
"""
    put('PACKAGE_README.md',notice.encode('utf8'))
    put('PACKAGE_CONTENTS_SHA256.json',(json.dumps({name:digest_bytes(data) for name,data in sorted(entries.items())},indent=2,sort_keys=True)+'\n').encode())
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(entries.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,13,0,0,0))
            info.compress_type=zipfile.ZIP_STORED if name.endswith('.jpg') else zipfile.ZIP_DEFLATED
            z.writestr(info,data,compress_type=info.compress_type,compresslevel=9)
    payload=buffer.getvalue()
    if len(payload)>=MAX_BYTES:
        C.write(OUT/'PACKAGE_SIZE_BLOCKED.json',dict(status='BLOCKED_SIZE',bytes=len(payload),limit=MAX_BYTES,images=56,profile=dict(width=IMAGE_WIDTH,quality=IMAGE_QUALITY)))
        raise RuntimeError('Fixed all-image package exceeds 10 MB; no partial ZIP written')
    verify(source_hashes)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        bad=z.testzip()
        if bad is not None:raise ValueError('Archive CRC failed: '+bad)
        if len(z.namelist())!=len(entries):raise ValueError('Archive member count differs')
    C._write_bytes(ZIP,payload)
    result=dict(status='COMPLETE',path=str(ZIP),zip_sha256=digest_bytes(payload),bytes=len(payload),decimal_MB=len(payload)/1e6,
                images=56,members=len(entries),image_profile=dict(max_width=IMAGE_WIDTH,JPEG_quality=IMAGE_QUALITY),
                source_hashes=source_hashes,model_calls=0,training_updates=0,quality_calls=0,CRC='PASS')
    C.write(RECEIPT,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('check','build'),default='build');a=p.parse_args()
    if a.stage=='check':
        prerequisites();result=dict(status='READY',model_calls=0,quality_calls=0)
    else:result=build()
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
