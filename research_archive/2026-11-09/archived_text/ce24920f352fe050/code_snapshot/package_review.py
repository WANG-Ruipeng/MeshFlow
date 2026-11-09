"""Evidence-only compact package. No model, scoring or geometry recalculation."""
import argparse,csv,difflib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from . import common as c
P=c.OUT/'mini_payload'

def copy(src,dest):return c._write_bytes(dest,c.resolve(src).read_bytes())
def csvrows(path):return list(csv.DictReader(c.resolve(path).open(encoding='utf-8-sig')))

def data():
    reg=c.read(c.OUT/'registration.json');oldhash={}
    for cell in reg['cells']:
        for p,h in c.read(cell['score_path'])['source_hashes'].items():
            if p in oldhash:assert oldhash[p]==h,('old version conflict',p)
            oldhash[p]=h
    sources=[]
    for p,h in sorted(oldhash.items()):
        assert c.sha(p)==h,p
        dest=c.OUT/'original_scorer'/c.resolve(p).relative_to(Path('/mnt/e'))
        copy(p,dest);sources.append(dict(path=p,sha256=h,snapshot=str(dest),status='MATCHES_EXECUTED_OLD_SCORER_HASH'))
    c.write(c.OUT/'original_scorer_sources.json',sources)
    source_hash={};diff=[]
    for p in sorted(c.CODE.glob('*.py')):
        dest=c.OUT/'code_snapshot'/p.name;copy(p,dest);source_hash[str(p)]=c.sha(p)
        diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(c.ROOT)),n=3))
    for p in (c.OUT/'independent_audit').glob('*.py'):
        copy(p,c.OUT/'code_snapshot/independent_audit'/p.name);source_hash[str(p)]=c.sha(p)
    c._write_bytes(c.OUT/'code.diff',''.join(diff).encode('utf-8'))
    c.write(c.OUT/'source_hashes.json',dict(actual_new_code=source_hash,old_scoring_sources=oldhash,changed_old_sources=0))
    idx=[];querysaved=set();image_map=[]
    for cell in reg['cells']+reg['references']:
        g=c.load_geometry(cell);d=c.OUT/'diagnostics'/cell['cell_id'];dest=P/'cells'/cell['cell_id']
        c.frozen_npz(dest/'geometry.npz',{k:g[k] for k in ('C','F','C_original','vertices','faces','known_mask')})
        if cell['condition_id'] not in querysaved:
            querysaved.add(cell['condition_id']);copy(cell['query_path'],P/'queries'/f"{cell['condition_id']}.npz")
        if cell.get('vectors_path'):
            old=c.arrays(cell['vectors_path']);c.frozen_npz(dest/'inherited_masks.npz',{k:old[k] for k in ('conflict_dup','conflict_cross','conflict_union')})
        # Compact CSV retains every registered query row, no sampling.
        keep=['query_id','C_face_row','weight','nearest_F_face_id','nearest_F_distance','signed_nearest_point_offset',
              'nearest_parallel_F_face_id','nearest_parallel_F_distance','signed_nearest_parallel_point_offset',
              'old_interior','old_dup','reconstructed_dup','all_isolated_normal_hits','corridor_isolated_normal_hits','corridor_layer_groups','coplanar_normal_line_intervals']
        c.csvwrite(dest/'per_query_compact.csv',[{k:r[k] for k in keep} for r in csvrows(d/'per_query.csv')])
        hits=c.arrays(d/'normal_line_hits.npz');cols=['query_id','free_face_id','signed_t','normal_dot','in_corridor','layer_group_id','barycentric_strict']
        c.csvwrite(dest/'all_normal_line_hits.csv',[{k:hits[k][j] for k in cols} for j in range(len(hits['query_id']))])
        for name in ['summary.json','candidate_faces.csv','parallel_regions.csv','sidewall_adjacency.csv','exact_edges.csv','normal_line_layers.csv','free_face_geometry.csv','near_parallel_witnesses.npz','coplanar_line_intervals.npz']:
            copy(d/name,dest/name)
        idx.append(dict(cell_id=cell['cell_id'],geometry=str((dest/'geometry.npz').relative_to(P)),
            query=str(Path('queries')/f"{cell['condition_id']}.npz"),source_geometry_path=cell['geometry_path'],source_geometry_sha256=cell['geometry_sha256'],
            source_query_sha256=cell['query_sha256'],stage=cell['stage'],label=cell['label'],graph_identity=g['graph_identity']))
    c.write(P/'portable_inputs.json',dict(cells=idx,L=c.L,config=c.read(c.OUT/'science_freeze.json')['config']))
    pictures=csvrows(c.OUT/'gallery/images.csv')+csvrows(c.OUT/'gallery/source_reference_images.csv')
    for r in pictures:
        source=c.resolve(r['image_path']);assert c.sha(source)==r['image_sha256']
        rel=source.relative_to(c.OUT/'gallery').with_suffix('.jpg');target=P/'gallery'/rel
        with Image.open(source) as im:
            im=im.convert('RGB');im.thumbnail((1400,1400),Image.Resampling.LANCZOS)
            buf=io.BytesIO();im.save(buf,format='JPEG',quality=65,optimize=True)
        c._write_bytes(target,buf.getvalue());image_map.append(dict(source_path=str(source),source_sha256=r['image_sha256'],package_path=str(target.relative_to(P)),package_sha256=c.sha(target)))
    c.csvwrite(P/'image_map.csv',image_map)
    html='<!doctype html><meta charset="utf-8"><title>CREF layer audit</title><h1>All 28 registered pages</h1><p>Uniform compressed copies. Full-resolution originals remain local. Nonblind AI_PROXY; source shapes are structural references, not targets.</p>'
    for r in image_map:
        html+='<p>'+Path(r['package_path']).stem+'</p><a href="'+r['package_path']+'"><img style="max-width:1400px;width:100%" src="'+r['package_path']+'"></a>'
    c._write_bytes(P/'index.html',html.encode('utf-8'))
    c.write(P/'SMALL_DATA_SCOPE.json',dict(status='ALL_REGISTERED_CASES_NO_SELECTION',main_cells=44,source_references=6,query_rows=102400,
        contains='All 50 compact C/F geometries, exact indices, 6 frozen query banks, 44 original masks, full per-query compact CSV, all normal hits with IDs/t/sign/groups, local graph/sidewall/face tables, all 28 uniformly compressed images, all reviews, inherited metrics, actual source and hashes.',
        omitted='Weights, latent/trajectory arrays, old heavy raw containers, repeated query xyz/normals and closest-point coordinates in per-query compact CSV; hit barycentric arrays omitted from CSV. Full original 50 diagnostic folders and full-resolution PNGs remain local.',
        reconstruction='Included compact C/F and query arrays permit recomputing new geometry explanations. Old CF/FF quality results are inherited, not recomputed by this package.',
        image_transform=dict(max_dimension=1400,JPEG_quality=65,all_pages_same_settings=True),new_model_calls=0))
    print('Data ready',len(idx),len(image_map),'images')

def package():
    roots=['REPORT.md','README_RESUME.md','registration.json','science_freeze.json','render_freeze.json','analytic_tests.json',
       'resources_prepare.json','resources_dispatch.json','geometry_completion.json','aggregation_completion.json','review_completion.json',
       'inherited_per_output.csv','per_output_layer_diagnosis.csv','paired_per_request_stage.csv','AI_PROXY_layer_review.csv','paired_layer_review.csv',
       'inherited_FF_contacts_with_candidate_flags.csv','protected_hashes.json','source_hashes.json','original_scorer_sources.json','code.diff','ledger.jsonl']
    for name in roots:copy(c.OUT/name,P/name)
    for folder in ['code_snapshot','original_scorer','reviews','independent_audit']:
        for f in sorted((c.OUT/folder).rglob('*')):
            if f.is_file() and '__pycache__' not in str(f):copy(f,P/f.relative_to(c.OUT))
    inventory={str(f.relative_to(P)):c.sha(f) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='PACKAGE_HASHES.json'}
    c.write(P/'PACKAGE_HASHES.json',inventory)
    target=c.OUT/'R2_CREF_LAYER_AUDIT_V1_REVIEW_SMALL.zip'
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for f in sorted(P.rglob('*')):
            if f.is_file():z.write(f,str(f.relative_to(P)))
    payload=buf.getvalue();assert len(payload)<10000000,('Package exceeds strict 10MB',len(payload))
    c._write_bytes(target,payload)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert {n:z.read(n) for n in []}=={}
        for n,h in inventory.items():assert c.hashlib.sha256(z.read(n)).hexdigest()==h,n
    c.write(c.OUT/'package_receipt.json',dict(status='PASS',path=str(target),bytes=len(payload),sha256=c.sha(target),entries=len(inventory)+1,
        under_10_decimal_MB=True,all_44_model_cells=True,all_6_reference_cells=True,all_28_images=True))
    print(c.read(c.OUT/'package_receipt.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['data','zip'],required=True);a=p.parse_args()
    data() if a.stage=='data' else package()