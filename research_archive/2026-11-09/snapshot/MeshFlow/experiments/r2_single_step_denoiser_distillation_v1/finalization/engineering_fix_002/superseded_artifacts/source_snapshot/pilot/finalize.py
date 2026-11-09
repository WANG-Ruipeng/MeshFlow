"""Verify completed evidence and create a portable review archive on CPU only.

No model import, inference, training, rescoring, or scientific report generation.
REPORT/METHODS/README_RESUME are supplied by the experiment investigator and
are never overwritten. Large-file hashes come from existing audited receipts.
"""
from __future__ import annotations
import argparse
import ast
import csv
import difflib
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import zipfile

try:
    from . import common as C
except ImportError:
    import common as C

OUT, CODE = C.OUT, C.CODE
PACKAGE = OUT / 'review_package'
FINAL = OUT / 'finalization'
ZIP = OUT / 'R2_SINGLE_STEP_DENOISER_DISTILLATION_REVIEW.zip'
VERSION = 'r2_single_step_distillation_finalization_v1'
METHODS = ('IDENTITY_GEO', 'D0_GEO', 'STUDENT_GEO')
IMAGES = 58
BINARY = {'.pt', '.pth', '.npz', '.npy', '.ply', '.safetensors', '.bin'}
HEX = re.compile(r'^[0-9a-f]{64}$')


class Incomplete(RuntimeError):
    pass


def need(test, message):
    if not test:
        raise Incomplete(message)


def existing(path):
    path = Path(path)
    need(path.is_file() and path.stat().st_size > 0, 'Required artifact missing/empty: ' + str(path))
    return path


def read_required(path, status=None):
    d = C.read(existing(path))
    if status is not None:
        need(d.get('status') == status, f'{path}: expected status {status}, got {d.get("status")}')
    return d


def rows(path):
    with existing(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def same_destination(path, data):
    path = C.checked(path)
    if path.exists():
        need(path.read_bytes() == data, 'Existing finalization artifact differs; inspect before replacing: ' + str(path))
        return
    C.atomic(path, lambda f: f.write(data), True)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def json_new(path, value):
    same_destination(path, json_bytes(value))


def csv_new(path, data):
    need(bool(data), 'Cannot write empty CSV without schema')
    buf = io.StringIO(newline='')
    fields = list(dict.fromkeys(k for row in data for k in row))
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader(); writer.writerows(data)
    same_destination(path, buf.getvalue().encode('utf-8'))



def prerequisites():
    checks = {}
    def check(path, status=None):
        d=read_required(path,status);checks[str(path)]=C.sha(path);return d
    reg=check(OUT/'registration.json','REGISTERED')
    for name in ('REPORT.md','METHODS.md','README_RESUME.md'):
        path=existing(OUT/name);checks[str(path)]=C.sha(path)
    prep=check(OUT/'manifest/preparation_completion.json','COMPLETE')
    need(prep['counts']['train_conditions']==112 and prep['counts']['train_parents']==25,'Wrong data population')
    gen=check(OUT/'generation/completion.json','COMPLETE')
    need(gen['rollouts']==249 and gen['forwards']==12450 and len(gen['rows'])==249,'Incomplete fixed RAW generation')
    targets=check(OUT/'teacher_cache/completion.json','COMPLETE')
    need(targets['new_teacher_forwards']==1193 and targets['exact_synthetic_reuses']==18 and len(targets['rows'])==1211,'Incomplete teacher cache')
    for name in ('teacher_completion.json','student_completion.json'):check(OUT/'preflight'/name,'PASS')
    train=check(OUT/'training/completion.json','COMPLETE')
    for name,value in [('updates',4000),('train_forwards',32000),('train_backwards',32000),('diagnostic_forwards',165)]:
        need(train.get(name)==value,'Incomplete student '+name)
    need(train['monitor_steps']==[0,500,1000,2000,4000],'Missing fixed monitoring checkpoint')
    for key in ('final_full','inference'):
        existing(train[key]['path'])
        need(HEX.fullmatch(train[key]['sha256']) is not None,'Final checkpoint receipt lacks SHA256')
    need(len(rows(OUT/'training/learning_curve.csv'))==4000,'Training curve missing updates')
    for step in (0,500,1000,2000,4000):need(len(rows(OUT/'monitor'/f'step{step:04d}.csv'))==33,'Missing monitor observations')
    inf=check(OUT/'inference/completion.json','COMPLETE')
    need(inf['student_forwards']==66 and len(inf['rows'])==66,'Incomplete student outputs')
    ev=check(OUT/'evaluation/completion.json','PASS')
    for name,value in [('primary_POST',144),('new_quality',96),('new_GEO',96),('D0_reused',48),('primary_AVMR_rows',2304),('boundary_vertices',2976)]:
        need(ev.get(name)==value,'Wrong evaluation count '+name)
    check(OUT/'evaluation/fidelity_completion.json','PASS')
    for name,count in [('post_index.csv',144),('quality_per_output.csv',144),('topology_per_output.csv',144),('per_output.csv',2304),('per_boundary_vertex.csv',2976),('distillation_per_output.csv',66)]:
        need(len(rows(OUT/'evaluation'/name))==count,'Wrong CSV row count '+name)
    bench=check(OUT/'benchmark/completion.json','COMPLETE')
    need(bench['forwards_per_model']==228 and bench['total_forwards']==456,'Incomplete benchmark')
    for name in ('D0_OFFICIAL',C.STUDENT):
        b=check(OUT/'benchmark'/name/'completion.json','COMPLETE')
        need(b['forwards']==228 and b['neural_forwards']==150 and b['full_post_forwards']==78,'Wrong benchmark calls')
        need(len(rows(OUT/'benchmark'/name/'timings.csv'))==228,'Missing individual timing repeats')
    audit=check(OUT/'independent_audit/final_REPORT.json','PASS')
    need(audit['source_sha256']==C.sha(CODE/'audit.py'),'Final audit source changed')
    for name in ('data','targets','preflight','training','evaluation','fidelity','checkpoints','budget','protected'):
        need(audit['results'].get(name,{}).get('status')=='PASS','Final audit incomplete: '+name)
    need(audit['results']['budget']['unclosed_operations']==0,'Unclosed scientific call')
    need(audit['results']['budget']['GPU_seconds']<=14400,'GPU time budget exceeded')
    allowed={r['phase'] for r in audit['results']['budget'].get('engineering_exceptions',[])}
    for path in (OUT/'gpu_phases').glob('*.json'):
        phase=C.read(path)
        if phase['status']!='COMPLETE':
            need(phase['name'] in allowed and phase['counts']=={'generator_F':0} and phase['pending'] is None,'Failed scientific phase: '+phase['name'])
        checks[str(path)]=C.sha(path)
    gallery=check(OUT/'gallery/gallery_manifest.json','COMPLETE')
    need((gallery['input_requests'],gallery['POST_outputs'],gallery['images'],gallery['local_sheets'])==(48,144,58,10),'Incomplete gallery')
    need(gallery['code_sha256']==C.sha(CODE/'gallery.py'),'Gallery source changed')
    need(gallery['post_index_sha256']==C.sha(OUT/'evaluation/post_index.csv'),'Gallery POST index changed')
    expected={(r['method'],r['label'],r['request_id']) for r in rows(OUT/'evaluation/post_index.csv')}
    need(len(expected)==144 and {x[0] for x in expected}==set(METHODS),'POST identity duplicates/method mismatch')
    review_path=OUT/'gallery/AI_PROXY_per_output_review.csv';review=rows(review_path)
    need(len(review)==144 and {(r['denoiser'],r['generator'],r['request_id']) for r in review}==expected,'Review coverage differs')
    for r in review:
        need(r['review_kind']=='AI_PROXY' and r['review_status']=='COMPLETE','Visual review not complete')
        for key in ('C_use','boundary_continuation','overall_quality','overall_structure_vs_D0','C_use_and_boundary_vs_D0','new_thin_structure_damage_vs_D0','near_fold_sharp_layer_change','uncertainty'):
            need(r.get(key,'').strip().upper() not in {'','PENDING','NOT_REVIEWED','TODO'},'Unfilled review field '+key)
        need(r.get('blind','').lower() in ('false','0','no'),'Review must not claim blind assessment')
    viewed=check(OUT/'gallery/AI_PROXY_review_receipt.json','COMPLETE')
    need(viewed.get('reviewed_POST_outputs')==144 and viewed.get('reviewed_images')==58,'Incomplete actual image viewing')
    expected_pngs={str(Path(r['png']).resolve()):r['png_sha256'] for r in gallery['receipts']};seen=set()
    for item in viewed['images']:
        path=str(Path(item['path']).resolve())
        need(item.get('actually_viewed') is True and path in expected_pngs and path not in seen,'Unexpected/duplicate/unviewed image')
        need(item['sha256']==expected_pngs[path]==C.sha(path),'Viewed image changed');seen.add(path)
    need(len(seen)==58,'Missing actual image views')
    for r in gallery['receipts']:
        for k,h in [('png','png_sha256'),('jpg','jpg_sha256'),('receipt_path','receipt_sha256')]:need(C.sha(r[k])==r[h],'Render artifact changed')
        q=r['identity'];need(C.sha(q['render_arrays_path'])==q['render_arrays_sha256'],'Render arrays changed')
    for p in [review_path,OUT/'gallery/AI_PROXY_REVIEW.md',OUT/'manifest/data.json',OUT/'manifest/train_stream.json',OUT/'gpu_calls.jsonl',OUT/'training/updates.jsonl',OUT/'generation/ledger.jsonl']:
        existing(p);checks[str(p)]=C.sha(p)
    return reg,audit,gallery,checks


def snapshot_sources(reg):
    entries={p.resolve():('pilot',None) for p in CODE.glob('*.py')}
    er=read_required(OUT/'evaluation/registration.json')
    for mapping in (reg.get('protected',{}),er.get('protected',{}),er.get('quality_source_hashes',{})):
        for path,h in mapping.items():
            p=Path(path)
            if p.suffix=='.py':entries.setdefault(p.resolve(),('inherited_readonly',h))
    official=C.OLD/'source_snapshot/official_denoiser_db86bc3';om=C.read(C.OLD/'asset_audit/official_denoiser_source_manifest.json')
    need(om['commit']=='db86bc32423c316c00bcc2f645a26243e8010f20','Wrong official architecture revision')
    for r in om['files']:entries[(official/r['path']).resolve()]=('official_readonly',r['sha256'])
    for p in [C.ROOT/'LICENSE',C.ROOT/'r2/LICENSE',C.ROOT/'r2/THIRD_PARTY_NOTICES.md',C.ROOT/'r2/pyproject.toml',C.ROOT/'native_t1/requirements.txt']:
        entries[p.resolve()]=('license_dependency',None)
    for p in OUT.glob('audit/**/*.py'):entries[p.resolve()]=('engineering_provenance',None)
    result=[]
    for p,(kind,h) in sorted(entries.items(),key=lambda x:str(x[0])):
        need(p.is_file(),'Source/license missing: '+str(p));value=C.sha(p)
        need(h is None or value==h,'Registered source changed: '+str(p));need(p.stat().st_size<4<<20,'Oversized source')
        if kind=='pilot':dest=OUT/'source_snapshot/pilot'/p.name
        elif kind=='official_readonly':dest=OUT/'source_snapshot/official'/p.relative_to(official)
        elif p.is_relative_to(C.ROOT):dest=OUT/'source_snapshot/repository'/p.relative_to(C.ROOT)
        else:dest=OUT/'source_snapshot/external'/f'{value[:12]}_{p.name}'
        same_destination(dest,p.read_bytes())
        result.append(dict(original_path=str(p),snapshot_path=str(dest.relative_to(OUT)),role=kind,bytes=p.stat().st_size,sha256=value))
    source=(official/'models/rmsnorm.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='RMSNorm')
    fragments={'RMSNorm.py':('models/rmsnorm.py','import torch\nfrom torch import nn\n\n'+ast.get_source_segment(source,node)+'\n')}
    source=(official/'models/utils.py').read_text();wanted={'get_2d_sincos_pos_embed','get_2d_sincos_pos_embed_from_grid','get_1d_sincos_pos_embed_from_grid'}
    parts=[ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    need(len(parts)==3,'Missing runtime helpers');fragments['positional_helpers.py']=('models/utils.py','import numpy as np\n\n'+'\n\n'.join(parts)+'\n')
    for name,(src,text) in fragments.items():
        dest=OUT/'source_snapshot/actual_runtime_fragments'/name;same_destination(dest,text.encode())
        result.append(dict(original_path=str(official/src)+'#AST_runtime_fragment',snapshot_path=str(dest.relative_to(OUT)),role='actual_runtime_fragment',bytes=dest.stat().st_size,sha256=C.sha(dest)))
    for name in ('intersections.py','pair_audit.py','shape.py','equivae.py'):
        need(any(Path(x['original_path']).name==name for x in result),'Missing actual source '+name)
    csv_new(OUT/'source_hashes.csv',result)
    diff=['# New isolated pilot files; original model/scoring sources unchanged.\n']
    for p in sorted(CODE.glob('*.py')):
        diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(keepends=True),fromfile='/dev/null',tofile='b/r2/pilots/single_step_denoiser_distillation_v1/'+p.name))
    same_destination(OUT/'code.diff',''.join(diff).encode());same_destination(OUT/'source_snapshot/unchanged_original_sources.diff',b'')
    json_new(OUT/'source_snapshot/official_source_manifest.json',om)
    return result


def copy_source_evidence(reg,audit):
    er=C.read(OUT/'evaluation/registration.json');index=[]
    paths={Path(p):h for p,h in er['protected'].items() if Path(p).suffix=='.csv'};olddata=C.ANCHOR/'data_audit'
    for name in ('training_whitelist.json','dev_conditions.json','condition_audit.csv','exclusions.csv','exclusions.json','parents.csv','SOURCE_SUPPORTED_INTERFACE.csv','summary.json'):
        p=olddata/name;paths[p]=C.sha(p)
    data=C.read(OUT/'manifest/data.json')
    for r in data['train_inputs']+data['dev_synthetic']:
        p=Path(r['source_interface_json']);paths[p]=C.sha(p)
    for p,h in sorted(paths.items(),key=lambda x:str(x[0])):
        need(C.sha(p)==h,'Reused data/table changed')
        folder='source_data_contract' if p.is_relative_to(olddata) else 'reused_D0_tables'
        dest=OUT/'referenced_evidence'/folder/f'{h[:12]}_{p.name}';same_destination(dest,p.read_bytes())
        index.append(dict(original_path=str(p),snapshot_path=str(dest.relative_to(OUT)),sha256=h,bytes=p.stat().st_size,role='READONLY_SOURCE_DATA_OR_REUSED_CURRENT_D0_EVIDENCE'))
    csv_new(OUT/'referenced_evidence/ORIGIN_INDEX.csv',index)
    json_new(OUT/'protected_before_after.json',dict(status='PASS',expected_before=reg['protected'],after_verified_equal=True,
        verification_receipt='independent_audit/final_REPORT.json',receipt_sha256=C.sha(OUT/'independent_audit/final_REPORT.json'),audit_source_sha256=audit['source_sha256'],
        actual_verification='audit.py final runs C.verify(registration.protected); imported result, not a new weight read'))
    return index


def detailed_saved_geometry_tables():
    contacts=[];face_rows=[];headers=[]
    for r in rows(OUT/'evaluation/post_index.csv'):
        d=C.read(r['diagnostic_path']);q=d['quality'];pair=Path(q['pair_audit_path']);shape=Path(q['shape_details_path'])
        for p in (pair,shape):need(C.sha(p)==q['files'][str(p)],'Saved pair/face audit changed')
        meta={k:r[k] for k in ('method','label','request_id','parent_id','condition_id','K','N')}
        for domain,section in C.read(pair).items():
            records=section.get('contacts',[])
            headers.append(dict(**meta,domain=domain,kind='PAIR_AUDIT',source_path=str(pair),source_sha256=C.sha(pair),records=len(records),header_json=json.dumps({k:v for k,v in section.items() if k!='contacts'},separators=(',',':'),ensure_ascii=False)))
            for i,item in enumerate(records):
                contacts.append(dict(**meta,domain=domain,contact_index=i,pair_i=item['pair'][0],pair_j=item['pair'][1],classification=item['classification'],contact_json=json.dumps(item,separators=(',',':'),ensure_ascii=False)))
        for domain,section in C.read(shape).items():
            headers.append(dict(**meta,domain=domain,kind='SHAPE_AUDIT',source_path=str(shape),source_sha256=C.sha(shape),records=len(section['per_face']),header_json=json.dumps({k:v for k,v in section.items() if k!='per_face'},separators=(',',':'),ensure_ascii=False)))
            for item in section['per_face']:face_rows.append(dict(**meta,domain=domain,**item))
    need(bool(contacts) and bool(face_rows),'Saved geometry detail unexpectedly empty')
    csv_new(OUT/'review_details/all_saved_pair_contacts.csv',contacts);csv_new(OUT/'review_details/all_saved_shape_faces.csv',face_rows);csv_new(OUT/'review_details/saved_geometry_audit_headers.csv',headers)
    return dict(pair_contacts=len(contacts),shape_face_rows=len(face_rows),audit_headers=len(headers),new_scores=0,original_JSON_preserved_local=True)


def receipt_hashes():
    saved={}
    def add(path,h,origin):
        if not isinstance(path,str) or not path.startswith('/mnt/') or not isinstance(h,str) or not HEX.fullmatch(h):return
        p=Path(path)
        if p.suffix.lower() not in BINARY:return
        # Historical closed-step receipts describe obsolete rotating latest.pt.
        if p.name=='latest.pt' and origin!='checkpoints/latest.json':return
        key=str(p.resolve())
        if key in saved:need(saved[key]['sha256']==h,'Conflicting active binary receipts: '+key)
        else:saved[key]=dict(sha256=h,receipt=origin)
    def walk(value,origin,directory):
        if isinstance(value,dict):
            for k,v in value.items():add(k,v,origin)
            for pk,hk in [('path','sha256'),('npz','sha256'),('output_path','output_sha256'),('source_npz','source_npz_sha256')]:add(value.get(pk),value.get(hk),origin)
            for k,v in value.items():
                if k.endswith('_path'):add(v,value.get(k[:-5]+'_sha256'),origin)
            fs=value.get('files')
            if isinstance(fs,dict):
                for name,r in fs.items():
                    if isinstance(r,dict) and 'sha256' in r:add(str(Path(name) if Path(name).is_absolute() else directory/name),r['sha256'],origin)
            for v in value.values():walk(v,origin,directory)
        elif isinstance(value,list):
            for v in value:walk(v,origin,directory)
    for p in sorted(OUT.rglob('*.json')):
        if any(x in p.relative_to(OUT).parts for x in ('review_package','finalization','source_snapshot')):continue
        walk(C.read(p),str(p.relative_to(OUT)),p.parent)
    return saved


def local_binary_manifest():
    saved=receipt_hashes()
    files={p.resolve() for p in OUT.rglob('*') if p.is_file() and p.suffix.lower() in BINARY and 'review_package' not in p.parts}
    files.update(Path(p) for p in saved if Path(p).is_file());result=[]
    for p in sorted(files,key=str):
        key=str(p.resolve())
        if key in saved:
            value=saved[key];h=value['sha256'];origin=value['receipt'];basis='IMPORTED_EXISTING_RECEIPT_AFTER_FINAL_AUDIT_PASS_NOT_REHASHED'
        else:
            need(p.stat().st_size<=64<<20,'Large binary missing active hash receipt: '+str(p))
            h=C.sha(p);origin='finalizer direct small-array hash';basis='DIRECT_SMALL_BINARY_SHA256'
        result.append(dict(path=str(p),bytes=p.stat().st_size,sha256=h,hash_origin_receipt=origin,
            verification_basis=basis,artifact_location='CURRENT_PILOT' if p.is_relative_to(OUT) else 'REFERENCED_READONLY_SOURCE',embedded_in_review_zip=False))
    csv_new(OUT/'local_binary_artifacts.csv',result);return result


def omitted_json_reason(p):
    rel=p.relative_to(OUT);parts=rel.parts;name=rel.as_posix()
    if p.suffix!='.json':return None
    if name=='manifest/preparation_completion.json':return 'Duplicate complete input/source hash inventory; manifest, input CSV and independent data audit retained.'
    if parts[0]=='teacher_cache' and p.name!='completion.json':return 'Individual teacher receipt duplicated in complete teacher_cache/completion.json and binary manifest.'
    if parts[:2]==('generation','receipts'):return 'Individual RAW receipt duplicated in generation/completion.json and binary manifest.'
    if parts[:2] in (('evaluation','scores'),('evaluation','diagnostics')):return 'Original detail JSON remains local; all saved pair contacts/headers, shape faces and all existing per-case CSVs embedded without rescoring.'
    if parts[:2]==('evaluation','outputs') and p.name=='postprocess.json':return 'Original complete corner/face map kept locally; all per-face GEO change CSV and original processing code retained.'
    if parts[0] in ('inference','synthetic_results') and p.name!='completion.json':return 'Individual inference receipt duplicated in inference completion and binary manifest; original remains local.'
    if parts[0]=='gallery' and len(parts)==2 and '__' in p.name:return 'Image sidecar duplicated completely in gallery_manifest.json.'
    return None


def selected(p):
    rel=p.relative_to(OUT)
    if any(x in rel.parts for x in ('review_package','finalization','__pycache__')):return False
    if p==OUT/'completion.json' or p.name=='review_pending.csv':return False
    if p.suffix.lower() in BINARY|{'.png','.zip','.tmp','.pyc'}:return False
    if omitted_json_reason(p):return False
    if rel.parts[0]=='source_snapshot':return True
    if p.suffix in ('.jpg','.jpeg'):return rel.parts[:2]==('gallery','review_jpg')
    if p.suffix=='.html':return p==OUT/'gallery/index_portable.html'
    return p.suffix.lower() in ('.csv','.json','.jsonl','.md','.diff','.txt','.py','.yaml','.toml')


def omitted_and_small_manifests():
    omitted=[];small=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or any(x in p.relative_to(OUT).parts for x in ('review_package','finalization','__pycache__')):continue
        if p.suffix.lower() in BINARY|{'.zip','.tmp','.pyc'} or p in (OUT/'small_file_hashes.csv',OUT/'OMITTED_JSON.csv',OUT/'completion.json'):continue
        reason=omitted_json_reason(p);h=C.sha(p);rel=str(p.relative_to(OUT))
        if reason:omitted.append(dict(path=rel,absolute_local_path=str(p),bytes=p.stat().st_size,sha256=h,reason=reason,local_original_preserved=True,embedded_in_review_zip=False))
        small.append(dict(path=rel,bytes=p.stat().st_size,sha256=h,hash_basis='DIRECT_FINALIZER_SHA256',embedded_in_review_zip=selected(p)))
    csv_new(OUT/'OMITTED_JSON.csv',omitted);csv_new(OUT/'small_file_hashes.csv',small)
    return omitted,small


def build_package(gallery,checks,source_rows,binaries,details):
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and selected(p):same_destination(PACKAGE/p.relative_to(OUT),p.read_bytes())
    same_destination(PACKAGE/'gallery/index.html',(OUT/'gallery/index_portable.html').read_bytes())
    same_destination(PACKAGE/'index.html',b'<!doctype html><meta charset="utf-8"><h1>R2 single-step denoiser distillation</h1><ul><li><a href="REPORT.md">Report</a></li><li><a href="METHODS.md">Methods</a></li><li><a href="gallery/index.html">All 48 four-path comparisons</a></li><li><a href="evaluation/distillation_per_output.csv">66 per-input distillation results</a></li><li><a href="README_PACKAGE.md">Archive scope</a></li></ul>')
    text="""# Review package

Supplied REPORT, METHODS and README_RESUME are copied unchanged. This archive contains all current per-example CSVs and all reused D0 CSVs referenced by the current evaluation, including eight-threshold tables, pair-contact/shape-face records, 58 unchanged JPEG sheets, 144 actual AI_PROXY reviews, manifests, actual source snapshots/code.diff/hash lists, source data whitelist/exclusions/interfaces, and all formal resource, scientific call and update ledgers. It contains no old experiment report as new evidence.

`review_details/` losslessly serializes existing saved pair-contact and per-face JSON records to CSV, including complete contact JSON fields and original audit headers. No geometric metric or model was rerun. The original JSON files remain local and are indexed in `OMITTED_JSON.csv`. `referenced_evidence/ORIGIN_INDEX.csv` identifies readonly source data and D0 evidence copied from their actual original paths.

Checkpoints, NPZ/NPY arrays, PLY meshes and full PNGs remain local. `local_binary_artifacts.csv` lists their paths, bytes, hashes and the existing receipt/audit supporting each hash; the finalizer does not reread multi-GB weights. `small_file_hashes.csv` lists local small text/image evidence. `PACKAGE_MANIFEST.csv` verifies actual package members.

CSV and JSONL members use lossless ZIP-LZMA (method 14); other members use DEFLATE. File and JPEG bytes are unchanged. Use an extractor supporting ZIP-LZMA, such as Python zipfile or 7-Zip; older Windows Explorer versions may not support it. The target is below 10,000,000 bytes; actual size and compression methods are recorded in completion.json outside the package. Required detail/images are never removed merely to meet a byte target.

`gallery/index.html` works entirely with relative JPEG paths. PNGs, actual rendering arrays and original meshes stay local. AI_PROXY is not blind human review. Student export remains local and independent loading code is included; no teacher or student weights are embedded.

The source snapshot preserves actual architecture/scorer dependencies, repository licenses, third-party notices and requirements. `code.diff` presents the isolated pilot as additions from /dev/null. `source_snapshot/unchanged_original_sources.diff` is empty because inherited scientific sources were not edited. Engineering failure/correction receipts remain explicit and do not imply extra successful model calls.

The archive freezes evidence just before finalization finishes. Subsequent controller completion/resume bookkeeping is outside this snapshot and cannot authorize another scientific call.
"""
    same_destination(PACKAGE/'README_PACKAGE.md',text.encode())
    json_new(PACKAGE/'PACKAGE_PROVENANCE.json',dict(schema=VERSION,source_files=len(source_rows),binary_artifacts_local=len(binaries),prerequisites=checks,required_review_POST=144,JPEG_sheets=58,model_calls=0,weights_loaded=0,reports_modified=False,saved_detail_conversion=details,excluded_extensions=sorted(BINARY|{'.png'})))
    members=[]
    for p in sorted(PACKAGE.rglob('*')):
        if p.is_file() and p.name!='PACKAGE_MANIFEST.csv':members.append(dict(path=p.relative_to(PACKAGE).as_posix(),bytes=p.stat().st_size,sha256=C.sha(p)))
    csv_new(PACKAGE/'PACKAGE_MANIFEST.csv',members);paths=sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    need(sum(p.suffix.lower()=='.jpg' for p in paths)==58,'Missing unchanged JPEGs')
    need(not any(p.suffix.lower() in BINARY|{'.png'} for p in paths),'Array/weight embedded unexpectedly')
    if not ZIP.exists():
        def archive(f):
            with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
                for p in paths:
                    method=zipfile.ZIP_LZMA if p.suffix.lower() in ('.csv','.jsonl') else zipfile.ZIP_DEFLATED
                    info=zipfile.ZipInfo(p.relative_to(PACKAGE).as_posix(),date_time=(2026,10,11,0,0,0));info.compress_type=method;info.external_attr=0o644<<16
                    z.writestr(info,p.read_bytes(),compress_type=method,compresslevel=9)
        C.atomic(ZIP,archive,True)
    with zipfile.ZipFile(ZIP) as z:
        need(z.testzip() is None,'ZIP CRC failed')
        need(set(z.namelist())=={p.relative_to(PACKAGE).as_posix() for p in paths} and len(z.namelist())==len(paths),'ZIP members differ')
        for p in paths:need(hashlib.sha256(z.read(p.relative_to(PACKAGE).as_posix())).hexdigest()==C.sha(p),'ZIP member bytes differ')
        methods=sorted({i.compress_type for i in z.infolist()})
    for filename in ('gallery/index.html','gallery/index_portable.html','index.html'):
        p=PACKAGE/filename
        for link in re.findall(r'(?:href|src)="([^"]+)"',p.read_text()):
            target=(p.parent/html.unescape(link)).resolve()
            need(target.is_relative_to(PACKAGE.resolve()) and target.is_file(),'Portable link missing '+link)
    return dict(path=str(ZIP),bytes=ZIP.stat().st_size,sha256=C.sha(ZIP),members=len(paths),source_files=len(source_rows),CSV_members=sum(p.suffix=='.csv' for p in paths),JPEG_members=58,CRC='PASS',compression_methods=methods,under_10MB=ZIP.stat().st_size<10_000_000,size_target_bytes=10_000_000,package_manifest_sha256=C.sha(PACKAGE/'PACKAGE_MANIFEST.csv'),portable_links='PASS')


def verify_finished():
    done=read_required(OUT/'completion.json','COMPLETE')
    need(done['source_code_sha256']==C.sha(__file__),'Finalizer source changed')
    need(C.sha(ZIP)==done['archive']['sha256'],'Archive changed');C.verify(done['prerequisite_hashes']);C.verify(done['scientific_source_hashes'])
    need(C.sha(PACKAGE/'PACKAGE_MANIFEST.csv')==done['archive']['package_manifest_sha256'],'Local package manifest changed')
    with zipfile.ZipFile(ZIP) as z:
        need(z.testzip() is None,'Archive CRC changed');need(len(z.infolist())==done['archive']['members'],'Archive count changed')
        for r in rows(PACKAGE/'PACKAGE_MANIFEST.csv'):need(hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'],'Archive member changed')
    print(json.dumps(dict(status='SKIP_VERIFIED_COMPLETE',archive=done['archive'],model_calls=0,new_scores=0)));return done


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.check_only:
        ast.parse(Path(__file__).read_text());print(json.dumps(dict(status='PASS_STATIC_ONLY',model_calls=0,artifact_writes=0)));return
    if (OUT/'completion.json').exists():verify_finished();return
    if args.verify:raise Incomplete('No completed archive to verify')
    reg,audit,gallery,checks=prerequisites();source_rows=snapshot_sources(reg);copy_source_evidence(reg,audit)
    details=detailed_saved_geometry_tables();binaries=local_binary_manifest();omitted,small=omitted_and_small_manifests()
    archive=build_package(gallery,checks,source_rows,binaries,details);C.verify(checks)
    result=dict(schema=VERSION,status='COMPLETE',archive=archive,prerequisite_hashes=checks,scientific_source_hashes={str(p):C.sha(p) for p in CODE.glob('*.py')},source_code_sha256=C.sha(__file__),source_files=len(source_rows),binary_artifacts_listed=len(binaries),small_artifacts_hashed=len(small),omitted_redundant_JSON=len(omitted),actual_AI_PROXY_POST_rows=144,actual_reviewed_images=58,model_calls=0,new_scores=0,large_weight_rehashes=0,scientific_reports_modified=False,default_models_changed=False,git_commit_push=False)
    json_new(OUT/'completion.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except (Incomplete,FileNotFoundError) as exc:
        print(json.dumps(dict(status='FINALIZATION_BLOCKED_OR_INCOMPLETE',reason=str(exc),model_calls=0,retries_or_replays=0),ensure_ascii=False));raise SystemExit(2)
