"""Independent stdlib-only verification of the completed small evidence ZIP.

Reads the ZIP and small linked evidence only. No imports of experiment/scoring
code, no model/weight reads, and no writes unless --record is explicitly used.
A hash-bound AI_PROXY row is verified as a saved review record, not as proof of
human review or independent certification of the visual judgment itself.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, os, re, sys, zipfile
from pathlib import Path, PurePosixPath

SCRIPT = Path(__file__).resolve()
OUT = SCRIPT.parents[2]
ROOT = SCRIPT.parents[4]
MAIN = ('FULL_S0000','FULL_S1000','FULL_S3000','FULL_S6000')
REFS = ('SMALL_C200','CREF_C200')
STAGES = ('RAW','POST')
FIELDS = ('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer',
          'whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')
LIMIT = 10_000_000


def resolve(value):
    s = str(value).replace('\\','/')
    if os.name != 'nt' and re.match(r'^[A-Za-z]:/',s): s='/mnt/'+s[0].lower()+s[2:]
    if os.name == 'nt' and re.match(r'^/mnt/[a-z]/',s): s=s[5].upper()+':/'+s[7:]
    p=Path(s)
    return p if p.is_absolute() else ROOT/p


def sha_bytes(value): return hashlib.sha256(value).hexdigest()
def sha(path): return sha_bytes(Path(path).read_bytes())
def js(value): return json.loads(value.decode('utf-8-sig'))
def csvrows(value): return list(csv.DictReader(io.StringIO(value.decode('utf-8-sig'),newline='')))
def require(ok, message):
    if not ok: raise AssertionError(message)


def run():
    pointer_path=OUT/'archive/latest.json'
    if not pointer_path.is_file():
        return {'status':'WAITING_ARCHIVE','missing':str(pointer_path),'writes':0,'model_calls':0,'weight_reads':0}
    pointer_bytes=pointer_path.read_bytes(); pointer=js(pointer_bytes)
    complete_path=resolve(pointer['path'])
    require(complete_path.is_relative_to(OUT/'archive'),'Completion path outside this experiment archive')
    complete_bytes=complete_path.read_bytes()
    require(sha_bytes(complete_bytes)==pointer['sha256'],'Completion pointer SHA256 mismatch')
    complete=js(complete_bytes)
    require(pointer.get('status')=='COMPLETE' and complete.get('status')=='COMPLETE','Archive is not COMPLETE')
    zpath=resolve(complete['zip_path'])
    require(zpath.is_relative_to(OUT/'archive'),'ZIP path outside this experiment archive')
    data=zpath.read_bytes(); digest=sha_bytes(data)
    require(0<len(data)<LIMIT,'ZIP must be strictly smaller than 10,000,000 bytes')
    require(len(data)==complete['zip_bytes'] and digest==complete['zip_sha256'],'ZIP completion identity differs')
    verified_sources=[]; definition_keys=set(); local_images={}
    with zipfile.ZipFile(io.BytesIO(data),'r') as z:
        names=z.namelist(); name_set=set(names)
        require(len(names)==len(name_set),'Duplicate ZIP members')
        require(all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and '\\' not in n for n in names),'Unsafe member path')
        require(z.testzip() is None,'ZIP CRC check failed')
        require('MANIFEST.json' in name_set,'MANIFEST missing')
        manifest=js(z.read('MANIFEST.json'))
        require(manifest.get('status')=='COMPLETE','ZIP manifest not COMPLETE')
        member_hashes=manifest['file_hashes']
        require(set(member_hashes)|{'MANIFEST.json'}==name_set,'MANIFEST does not index exactly all other members')
        for name,h in member_hashes.items(): require(sha_bytes(z.read(name))==h,'Member SHA256 mismatch: '+name)
        for name in names:
            require(PurePosixPath(name).suffix.lower() not in {'.pt','.pth','.ckpt','.bin','.safetensors'},'Large weights unexpectedly packed: '+name)
        report=z.read('evidence/REPORT.md')
        require(report==(OUT/'REPORT.md').read_bytes(),'Packaged REPORT differs byte-for-byte from current REPORT.md')
        regbytes=z.read('evidence/registration.json')
        require(regbytes==(OUT/'registration.json').read_bytes(),'Packaged frozen registration differs')
        reg=js(regbytes); requests={q['request_id']:q for q in reg['requests']}
        require(len(requests)==24 and tuple(reg['evaluation_labels'])==MAIN,'Registered panel differs')
        expected={(label,rid,stage) for label in MAIN for rid in requests for stage in STAGES}
        review_bytes=z.read('evidence/evaluation/review/AI_PROXY_reviews.csv')
        require(review_bytes==(OUT/'evaluation/review/AI_PROXY_reviews.csv').read_bytes(),'Packaged formal review differs from current CSV')
        reviews=csvrows(review_bytes); ids=[(r['label'],r['request_id'],r['stage']) for r in reviews]
        require(len(reviews)==192 and len(set(ids))==192 and set(ids)==expected,'Main review identity set differs from registered192')
        for row in reviews:
            q=requests[row['request_id']]
            require(row['parent_id']==q['parent_id'] and int(row['candidate'])==q['candidate'],'Review parent/noise identity differs')
            require(int(row['N'])==q['N'] and int(row['K'])==q['K'] and row['split']==q['split'],'Review N/K/split differs')
            require(row.get('reviewer_kind')=='AI_PROXY' and row.get('review_status')=='REVIEWED','Actual review kind/status invalid')
            require(bool(row.get('reviewer','').strip()) and bool(row.get('notes','').strip()),'Reviewer/observations missing')
            require(row.get('blinded','').lower() in ('false','0'),'Nonblind identity missing')
            require(all(row.get(f) in ('YES','NO','UNCERTAIN') for f in FIELDS),'Invalid visual judgment value')
            for prefix in ('image','section_image'):
                p=resolve(row[prefix+'_path']); h=row[prefix+'_sha256']
                require(p.is_relative_to(OUT/'evaluation') and p.suffix.lower() in ('.png','.jpg','.jpeg'),'Review image path outside expected small evidence')
                if str(p) not in local_images:
                    require(p.is_file(),'Actual reviewed image missing: '+str(p))
                    local_images[str(p)]=sha(p)
                require(local_images[str(p)]==h,'Actual reviewed image hash differs: '+str(p))
        main_cells=0;reference_cells=0
        for label in MAIN+REFS:
            name='evidence/evaluation/tables/'+label+'/per_output.csv'
            tablebytes=z.read(name); table=csvrows(tablebytes)
            require(tablebytes==(OUT/'evaluation/tables'/label/'per_output.csv').read_bytes(),'Packaged per-output table differs: '+label)
            ident=[(r['label'],r['request_id'],r['stage']) for r in table]
            want={(label,rid,stage) for rid in requests for stage in STAGES}
            require(len(table)==48 and len(set(ident))==48 and set(ident)==want,'Per-output table incomplete: '+label)
            require(all(k in table[0] for k in ('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','C_F_valid_edge_count','C_F_valid_edge_total_length')),'Required original per-output metrics absent: '+label)
            if label in MAIN:main_cells+=len(table)
            else:reference_cells+=len(table)
        source_rows=csvrows(z.read('source_hashes.csv'))
        source_index={str(resolve(r['path'])):r for r in source_rows}
        require(len(source_index)==len(source_rows) and len(source_rows)>0,'Source path hashes missing or duplicate')
        for row in source_rows:
            content=z.read(row['archive_path'])
            require(sha_bytes(content)==row['sha256'] and len(content)==int(row['bytes']),'Actual source copy hash/size differs')
        for label in MAIN+REFS:
            for category in ('score_completion','boundary_completion'):
                receipt=js(z.read('evidence/evaluation/'+category+'/'+label+'.json'))
                require(receipt['status']=='COMPLETE' and len(receipt['receipts'])==48,'Completion receipts incomplete: '+label+'/'+category)
                definitions=[receipt['definition_hashes']] if category=='score_completion' else [r['identity']['definition_hashes'] for r in receipt['receipts']]
                for mapping in definitions:
                    for original,h in mapping.items():
                        p=str(resolve(original)); require(p in source_index,'Actual scoring source missing: '+original)
                        row=source_index[p]
                        require(row['sha256']==h,'Actual scoring source disagrees with saved definition: '+original)
                        definition_keys.add((p,h))
        diff=z.read('code.diff').decode('utf-8')
        require(len(diff)>10000 and '+def ' in diff,'code.diff does not contain actual added implementation')
        for name in ('train.py','evaluate.py','comparison.py','review_summary.py','result_plot.py'):
            relative='r2/pilots/full_seat_learning_curve_v1/'+name
            require('+++ b/'+relative in diff,'Actual new source content absent from code.diff: '+relative)
            require(str(ROOT/relative) in source_index,'New experiment source missing from source_hashes: '+relative)
        require('artifact_inventory.csv' in name_set and 'declared_path_hashes.csv' in name_set,'Local-only artifact/hash inventory missing')
        require('RESUME_AND_CONTENTS.md' in name_set,'Recovery/readme missing')
        previews=js(z.read('preview/manifest.json'))['images']
        require(len(previews)==48,'Expected48 uniformly selected previews')
        for group in ('gallery','sections_gallery'):
            sub=[r for r in previews if r['archive_path'].startswith('preview/'+group+'/')]
            require(len(sub)==24 and {r['request_id'] for r in sub}==set(requests),'Incomplete preview request set: '+group)
            for row in sub:
                b=z.read(row['archive_path'])
                require(sha_bytes(b)==row['preview_sha256'] and b.startswith(b'\xff\xd8\xff'),'Preview hash or JPEG format differs')
        require(len({r['archive_path'] for r in previews})==48,'Duplicate preview names')
        # Validate the latest packaged full192 derived-boundary snapshot.
        boundptr=js(z.read('evidence/evaluation/boundary_components_v1/latest.json'))
        boundlocal=resolve(boundptr['path'])
        boundname='evidence/'+boundlocal.relative_to(OUT).as_posix()
        boundbytes=z.read(boundname); require(sha_bytes(boundbytes)==boundptr['sha256'],'Boundary latest pointer hash differs')
        boundary=js(boundbytes)
        require(boundary['status']=='COMPLETE' and boundary['available_cells']==boundary['expected_cells']==192 and boundary['pending_cells']==0,'Main derived boundary evidence incomplete')
        boundtable=[p for p in boundary['files'] if Path(p).name=='per_output.csv']
        require(len(boundtable)==1,'Boundary per-output table identity ambiguous')
        bn='evidence/'+resolve(boundtable[0]).relative_to(OUT).as_posix(); brows=csvrows(z.read(bn))
        bids=[(r['label'],r['request_id'],r['stage']) for r in brows]
        require(len(bids)==192 and set(bids)==expected and all(r['status']=='AVAILABLE' for r in brows),'Boundary per-output identity set differs')
        require(pointer_path.read_bytes()==pointer_bytes,'Archive latest pointer changed during verification')
        result=dict(status='PASS',zip_path=str(zpath),zip_sha256=digest,zip_bytes=len(data),limit_bytes=LIMIT,
            crc='PASS',manifest_member_hashes_checked=len(member_hashes),zip_members=len(names),report_byte_exact=True,
            report_sha256=sha_bytes(report),registration_sha256=sha_bytes(regbytes),actual_review_rows=192,
            review_csv_sha256=sha_bytes(review_bytes),actual_review_image_files_hashed=len(local_images),main_model_count=4,
            main_per_output_cells=main_cells,separate_reference_cells=reference_cells,actual_source_files=len(source_rows),
            original_scoring_definition_path_hashes_verified=len(definition_keys),code_diff_bytes=len(diff.encode()),
            source_hashes_sha256=sha_bytes(z.read('source_hashes.csv')),preview_images=48,derived_boundary_cells=192,
            reviewer_kind='AI_PROXY',review_observations_not_independently_rejudged=True,
            model_calls=0,weight_reads=0,new_scores=0,new_visual_reviews=0,
            verifier_path=str(SCRIPT),verifier_sha256=sha(SCRIPT))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',action='store_true',help='Save a new immutable audit receipt keyed by ZIP and verifier SHA; never overwrite')
    args=p.parse_args()
    try: result=run()
    except Exception as exc:
        result=dict(status='FAIL',error_type=type(exc).__name__,error=str(exc),model_calls=0,weight_reads=0,new_scores=0)
    if args.record and result['status']=='PASS':
        target=SCRIPT.parent/('archive_verification_'+result['zip_sha256'][:16]+'_'+result['verifier_sha256'][:12]+'.json')
        payload=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8')
        if target.exists():require(target.read_bytes()==payload,'Existing verification evidence differs; will not overwrite')
        else:
            with target.open('xb') as f:f.write(payload)
        result=dict(result,receipt_path=str(target),receipt_sha256=sha(target))
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))
    return 1 if result['status']=='FAIL' else 2 if result['status']=='WAITING_ARCHIVE' else 0


if __name__=='__main__':sys.exit(main())