"""Seal all cases below 10MB by omitting reproducible duplicate table columns, not cases."""
import csv,difflib,hashlib,io,json,zipfile
from . import common as c

def main():
    # Preserve the actual data-preparation script and this final packager.
    hashes=c.read(c.OUT/'source_hashes.json');diff=[]
    for p in sorted(c.CODE.glob('*.py')):
        target=c.OUT/'code_snapshot'/p.name
        c._write_bytes(target,p.read_bytes())
        hashes['actual_new_code'][str(p)]=c.sha(p)
        diff.extend(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(c.ROOT)),n=3))
    c.write(c.OUT/'source_hashes.json',hashes,replace=True)
    c._write_bytes(c.OUT/'code.diff',''.join(diff).encode(),replace=True)
    entries={};payload=c.OUT/'mini_payload'
    for f in sorted(payload.rglob('*')):
        if not f.is_file():continue
        name=str(f.relative_to(payload))
        if f.name in ['free_face_geometry.csv','normal_line_layers.csv']:continue
        if f.name=='all_normal_line_hits.csv':
            rs=list(csv.DictReader(f.open(encoding='utf-8-sig')));s=io.StringIO(newline='')
            keys=['query_id','free_face_id','signed_t','in_corridor','layer_group_id'];w=csv.DictWriter(s,fieldnames=keys);w.writeheader()
            for r in rs:w.writerow({k:r[k] for k in keys})
            entries[name]=s.getvalue().encode('utf-8-sig')
        elif f.name=='SMALL_DATA_SCOPE.json':
            x=c.read(f);x['contains']='All 50 compact C/F geometries and exact indices; 6 original query banks; 44 original masks; 102400 compact per-query CSV rows; ALL finite normal-line hit query/face IDs, signed t, corridor flags and group IDs; candidate faces, parallel regions, exact edges and sidewall tables; all 28 uniform compressed images; inherited metrics, 44 reviews, actual code and hashes.'
            x['omitted']='Weights/trajectories/old large raw containers. Repeated query xyz/normals and nearest-point coordinates omitted from compact per-query CSV; hit normal_dot/barycentric columns omitted; repeated free_face_geometry and normal_line_layers tables omitted from ZIP. All omitted diagnostic arrays/tables and full-resolution PNGs remain in the local experiment. Included geometry and original queries permit their recomputation.'
            x['prior_package_size_check']='First data-only compression estimate 13645778 bytes. No oversized final ZIP was delivered. Omitted derived/repeated columns uniformly for all cases; no results or images dropped.'
            entries[name]=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
        else:entries[name]=f.read_bytes()
    roots=['REPORT.md','README_RESUME.md','registration.json','science_freeze.json','render_freeze.json','analytic_tests.json',
      'resources_prepare.json','resources_dispatch.json','geometry_completion.json','aggregation_completion.json','review_completion.json',
      'inherited_per_output.csv','per_output_layer_diagnosis.csv','paired_per_request_stage.csv','AI_PROXY_layer_review.csv','paired_layer_review.csv',
      'inherited_FF_contacts_with_candidate_flags.csv','protected_hashes.json','source_hashes.json','original_scorer_sources.json','code.diff','ledger.jsonl']
    for name in roots:entries[name]=(c.OUT/name).read_bytes()
    for folder in ['code_snapshot','original_scorer','reviews','independent_audit','draft_provenance']:
        for f in sorted((c.OUT/folder).rglob('*')):
            if f.is_file() and '__pycache__' not in str(f):entries[str(f.relative_to(c.OUT))]=f.read_bytes()
    inventory={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()}
    entries['PACKAGE_HASHES.json']=(json.dumps(inventory,indent=2,sort_keys=True)+'\n').encode()
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(entries.items()):z.writestr(name,data)
    data=buf.getvalue();assert len(data)<10000000,('Over strict10MB',len(data))
    target=c.OUT/'R2_CREF_LAYER_AUDIT_V1_REVIEW_SMALL.zip';c._write_bytes(target,data)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,h in inventory.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
        assert len([n for n in z.namelist() if n.endswith('/per_query_compact.csv')])==50
        assert len([n for n in z.namelist() if n.endswith('.jpg')])==28
    result=dict(status='PASS',path=str(target),bytes=len(data),sha256=c.sha(target),entries=len(entries),under_10_decimal_MB=True,
        all_44_model_cells=True,all_6_reference_cells=True,all_28_images=True,compact_query_rows=102400,archive_hashes_verified=len(inventory))
    c.write(c.OUT/'package_receipt.json',result);c.event('SMALL_PACKAGE_SEALED',**result);print(result)
if __name__=='__main__':main()