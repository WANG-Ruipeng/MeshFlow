"""Build and verify a small evidence ZIP. No geometry/model imports or old writes.

Intended for the workspace curator after all component extractions are closed.
Portable table recalculation is separately provided by recompute.py.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse, csv, hashlib, json, subprocess, sys, zipfile

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text('utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def normalized(v):
    return str(v).replace('E:\\MeshFlow','/mnt/e/MeshFlow').replace('E:/MeshFlow','/mnt/e/MeshFlow').replace('\\','/')
def writejson(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original',type=Path,required=True);a=ap.parse_args()
    old=a.original.resolve()
    assert not ROOT.is_relative_to(old) and ROOT!=old
    required={'metrics_per_output.csv':840,'AVMR_PMR_all8.csv':6720,'visual_review_840.csv':840,
              'parents.csv':105,'conditions.csv':264,'source_quality_and_interfaces.csv':264,
              'exposure_by_condition.csv':285,'training_steps.csv':12000,'fixed_fixtures.csv':16,'fixed_FM_diagnostics.csv':128}
    checks={};tables={}
    for name,n in required.items():
        tables[name]=rows(ROOT/name);checks[name]={'rows':len(tables[name]),'expected':n}
        assert len(tables[name])==n,(name,len(tables[name]),n)
    for name in ['ANSWERS.md','MISSING.md','README.md','metric_dictionary.json','derived/summary.json','derived/training_quality_exposure.csv']:
        assert (ROOT/name).is_file(),name
    ms=tables['metrics_per_output.csv'];vs=tables['visual_review_840.csv']
    k=lambda r:(r['request_id'],r.get('label',r.get('model_label')),r['stage'])
    assert len({k(r) for r in ms})==840 and {k(r) for r in ms}=={k(r) for r in vs}
    assert Counter(r['panel'] for r in ms)==Counter(NEW32=768,OLD6=72)
    expected_metric_fields=[r['metric'] for r in rows(ROOT/'protocol_and_code/registered_metric_fields.csv')]
    assert len(set(expected_metric_fields))==96
    original_metrics=rows(old/'evaluation/per_output.csv');omi={k(r):r for r in original_metrics}
    original_visual=rows(old/'evaluation/AI_PROXY_visual_review.csv');ovi={k(r):r for r in original_visual}
    scalar_checks=0;visual_value_checks=0
    for r in ms:
        for field,value in omi[k(r)].items():
            assert r[field]==value,(k(r),field,'original string changed')
            scalar_checks+=1
    for r in vs:
        for field,value in ovi[k(r)].items():
            assert r[field]==value,(k(r),field,'visual original changed')
            visual_value_checks+=1
    assert sha(ROOT/'AVMR_PMR_all8.csv')==sha(old/'evaluation/AVMR_PMR_all8.csv')
    av=tables['AVMR_PMR_all8.csv'];byoutput=defaultdict(set)
    for r in av:byoutput[k(r)].add(r['setting'])
    assert len(byoutput)==840 and all(len(v)==8 for v in byoutput.values())
    checks.update(original_metric_string_checks=scalar_checks,original_visual_string_checks=visual_value_checks,
                  AVMR_original_byte_copy=True,primary_keys_unique=True,metric_visual_keys_equal=True)

    parent_rows=tables['parents.csv'];condition_rows=tables['conditions.csv']
    assert Counter(r['parent_role'] for r in parent_rows)==Counter(COMMON_SMALL32=32,EXP_ONLY35=35,NEW_DEV32=32,OLD6=6)
    assert len({(r['parent_uid'],r['condition_id']) for r in condition_rows})==264
    assert Counter(r['parent_role'] for r in condition_rows)==Counter(COMMON_SMALL32=91,EXP_ONLY35=103,NEW_DEV32=64,OLD6=6)
    train_groups={r['group_id'] for r in parent_rows if r['parent_role'] in {'COMMON_SMALL32','EXP_ONLY35'}}
    dev_groups={r['group_id'] for r in parent_rows if r['parent_role']=='NEW_DEV32'}
    assert not train_groups&dev_groups
    agg=rows(ROOT/'derived/aggregate_recomputed.csv')
    meddiff=max(abs(float(r['original_parent_median'])-float(r['recomputed_parent_median'])) for r in agg)
    assert meddiff<=1e-12
    checks.update(parent_roles_and_condition_keys_pass=True,train_dev_group_disjoint=True,
                  aggregate_median_comparisons=len(agg),maximum_absolute_median_difference=meddiff)

    # Merge provenance as declared by each independent extractor; preserve IDs such as T005.
    provenance={};prior=load(old/'delivery/R2_USER_DATA_SCALING_32_67_REVIEW/PACKAGE_MANIFEST.json')
    known={normalized(r['source_path']):r['sha256'] for r in prior['files']}
    enriched=[]
    for p in sorted((ROOT/'_work').glob('*provenance.json')):
        part=load(p);source_rows=part.get('sources',[])
        if isinstance(source_rows,dict):source_rows=list(source_rows.values())
        for s in source_rows:
            path=s.get('source_path',s.get('path'))
            if path and normalized(path) in known:
                prior_sha=known[normalized(path)]
                present=s.get('existing_sha256')
                if not present or str(present).startswith('NA'):
                    s['additional_existing_sha256']=prior_sha
                    s['additional_existing_hash_reference']='Prior full review PACKAGE_MANIFEST.json; enriched without source reread'
                    enriched.append(path)
        provenance[p.stem]=part
    # Include root-owned extraction/analysis provenance even if metadata formatting differs by component.
    provenance['root_analysis']=dict(
        sources=['metrics_per_output.csv','visual_review_840.csv','protocol_and_code/original_aggregate_reference.csv','protocol_and_code/registered_metric_fields.csv'],
        outputs={p.relative_to(ROOT).as_posix():dict(DERIVED=True,projection='Arithmetic only; exact pairing and hierarchy implemented in protocol_and_code/recompute.py')
                 for p in sorted((ROOT/'derived').glob('*')) if p.is_file() and p.name!='training_quality_exposure.csv'},
        original_metric_sha256=sha(old/'evaluation/per_output.csv'),original_review_sha256=sha(old/'evaluation/AI_PROXY_visual_review.csv'),
        original_files_modified=False,model_calls=0,geometry_calls=0,image_reads=0)
    writejson(ROOT/'provenance.json',provenance)

    # CSV parent/split policy is also verified by the data extractor, retained as an independent receipt.
    receipts=sorted(set((ROOT/'_work').glob('*verification*.json'))|set((ROOT/'_work').glob('*validation*.json')))
    verification={p.name:load(p) for p in receipts}
    verification['root_final_checks']=checks
    verification['scope']=dict(no_model_load=True,no_forward=True,no_new_noise=True,no_mesh_or_geometry_score=True,no_render_or_review=True,
                               no_weight_rehash=True,old_output_read_only=True,prior_hash_enrichments=len(enriched))
    writejson(ROOT/'VALIDATION.json',verification)

    excluded={'_work','__pycache__'}
    duplicate_exclusions={'protocol_and_code/source_quality_rule_functions.txt':'Complete original functions included in original/source_quality_scan.py and original/source_data_freeze.py'}
    forbidden={'.pt','.pth','.ckpt','.npz','.npy','.ply','.obj','.png','.jpg','.jpeg','.webp','.gif','.mp4','.html','.pyc'}
    selected=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if not p.is_file() or set(rel.parts)&excluded or p.name in {'MANIFEST.json','FILE_INDEX.csv'} or rel.as_posix() in duplicate_exclusions:continue
        assert p.suffix.lower() not in forbidden,(rel,'forbidden payload')
        assert not p.is_symlink()
        selected.append(p)
    metadata={}
    for namespace,part in provenance.items():
        output=part.get('outputs',{})
        if isinstance(output,list):output={r.get('file',r.get('path','')):r for r in output}
        for name,record in output.items():metadata[name]={**record,'provenance_namespace':namespace}
    metadata['derived/training_quality_exposure.csv']=dict(DERIVED=True,
        provenance_namespace='exposure_quality_provenance',source_refs=['conditions.csv','source_quality_and_interfaces.csv','exposure_by_condition.csv'],
        projection='Join identities; sum observed integer counts, Decimal ratios; original source labels preserved')
    for name in ['parents.csv','conditions.csv','source_quality_and_interfaces.csv']:
        metadata[name]['DERIVED']='MIXED: inherited values plus explicitly named DERIVED columns and documented arithmetic fields'
        metadata[name]['source_refs']='provenance.json/data_provenance/sources; row source IDs resolve within that namespace'
        metadata[name]['projection']=provenance['data_provenance']['projections'][Path(name).stem]
    metadata['source_events.json']['DERIVED']='PROJECTION_ONLY: saved events, no new geometry'
    metadata['source_events.json']['source_refs']='provenance.json/data_provenance/sources; each event record retains origin'
    metadata['source_events.json']['projection']=provenance['data_provenance']['projections']['source_events']
    listing=[]
    for p in selected:
        rel=p.relative_to(ROOT).as_posix();rec=metadata.get(rel,{})
        rr=rows(p) if p.suffix=='.csv' else None
        derived=rec.get('DERIVED',rec.get('derived',rel.startswith('derived/') or p.name in {'ANSWERS.md','MISSING.md','README.md','VALIDATION.json','provenance.json'}))
        if not rec:
            rec=dict(source_refs=['Current MINI extraction/recalculation code and preserved provenance.json'],
                     transformation='NEW_DOCUMENTATION_OR_EXTRACTION_CODE' if p.suffix in {'.py','.md'} else 'CONSOLIDATED_METADATA',DERIVED=derived,existing_source_hash='NA_NEW_MINI_FILE',source_path=rel)
        listing.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),rows=(len(rr) if rr is not None else None),
                            fields=(list(rr[0]) if rr else [] if rr==[] else None),DERIVED=derived,provenance=rec))
    manifest=dict(package=ROOT.name,created_utc=datetime.now(timezone.utc).isoformat(),source_experiment=str(old),
                  source_paths_are_references_not_runtime_dependencies=True,files=listing,
                  source_records='provenance.json; source IDs retained per extractor namespace',
                  self_hash_policy='MANIFEST and FILE_INDEX are indexed by final external ZIP receipt; manifest cannot hash itself',
                  exclusions=['_work','all weights/arrays/meshes/images','old complete ZIP'],duplicate_exclusions=duplicate_exclusions,
                  numeric_precision='Original CSV strings retained; derived floating arithmetic not report-rounded',
                  derived_tag_policy='Explicit DERIVED in table rows/columns or per-file manifest; inherited states not flattened')
    writejson(ROOT/'MANIFEST.json',manifest)
    with (ROOT/'FILE_INDEX.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,['file','bytes','rows','sha256','DERIVED']);w.writeheader()
        for r in listing:w.writerow(dict(file=r['path'],bytes=r['bytes'],rows=r['rows'],sha256=r['sha256'],DERIVED=r['DERIVED']))
    selected += [ROOT/'MANIFEST.json',ROOT/'FILE_INDEX.csv']
    archive=ROOT.with_suffix('.zip')
    if archive.exists():raise FileExistsError('Do not replace a delivered archive: '+str(archive))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in selected:z.write(p,p.relative_to(ROOT).as_posix())
    if archive.stat().st_size>10*1024*1024:
        raise RuntimeError('ZIP exceeds 10MiB: preserve full tables and split with manifests; do not deliver silently')
    # Actual ZIP read/extract/recalculation, no reliance on absolute workspace paths.
    verify=ROOT/'_work/unzip_validation'
    if verify.exists():raise FileExistsError(verify)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in listing:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'],r['path']
        for name in z.namelist():
            assert not name.startswith(('/', '\\')) and '..' not in Path(name).parts
        z.extractall(verify)
    derived_check=ROOT/'_work/recomputed_from_unzip'
    result=subprocess.run([sys.executable,'-B',str(verify/'protocol_and_code/recompute.py'),'--out',str(derived_check)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    for p in derived_check.iterdir():
        assert sha(p)==sha(verify/'derived'/p.name),('portable recalculation differs',p.name)
    exposure_check=ROOT/'_work/exposure_from_unzip'
    result=subprocess.run([sys.executable,'-B',str(verify/'protocol_and_code/summarize_quality_exposure.py'),'--out',str(exposure_check)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert sha(exposure_check/'training_quality_exposure.csv')==sha(verify/'derived/training_quality_exposure.csv')
    receipt=dict(status='MINI_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),archive=str(archive),bytes=archive.stat().st_size,
                 MiB=archive.stat().st_size/(1024**2),sha256=sha(archive),members=len(selected),target_5MiB_met=archive.stat().st_size<=5*1024**2,
                 under_10MiB=True,zip_crc_pass=True,all_member_hashes_pass=True,portable_unzip_recompute_byte_identical=True,portable_exposure_recompute_byte_identical=True,
                 model_load=0,forward=0,noise=0,geometry_score=0,render=0,visual_review=0,weight_bytes_rehashed=0,
                 old_outputs_modified=False,commit=False,push=False)
    writejson(ROOT.parent/(ROOT.name+'_DELIVERY.json'),receipt)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
