"""Small complete-tabular review pack; original meshes and full images stay local."""
from .bootstrap import bind,ROOT,OUT
D,E=bind()
import csv,hashlib,io,json,zipfile,difflib
from pathlib import Path
from PIL import Image

def build():
    assert D.read(OUT/'completion.json')['status']=='COMPLETE'
    dest=OUT/'evidence_pack';dest.mkdir(exist_ok=True)
    members={};external=[]
    def add(p,name=None):
        p=D.resolve(p);members[name or str(p.relative_to(OUT))]=p.read_bytes()
    for name in ['REPORT.md','README_RESUME.md','completion.json','registration.json','preflight.json','analysis_source_lock.json','execution_source_lock.json','inputs/manifest.json','evaluation/review_seal_before_unblinding.json','evaluation/reviews_actual.json','evaluation/review_rubric.json','evaluation/review_assignment.json','evaluation/review_gate.json','evaluation/blind_gallery/reviews_anonymous_actual.json','evaluation/blind_gallery/view_receipts.json']:
        add(OUT/name)
    for folder in ['evaluation/analysis','audit','numeric_crosscheck']:
        for q in sorted((OUT/folder).rglob('*')):
            if q.is_file() and q.suffix in ('.json','.csv','.md','.py'):
                add(q)
    for q in sorted((OUT/'inputs/requests').glob('*.npz')):add(q)
    imgs=D.read(OUT/'evaluation/blind_gallery/manifest.json')['images'];previews=[]
    for r in imgs:
        original=D.resolve(r['image_path']);im=Image.open(original).convert('RGB');im.thumbnail((1440,1440))
        b=io.BytesIO();im.save(b,format='JPEG',quality=82,optimize=True)
        name='gallery/images/'+original.name;members[name]=b.getvalue()
        previews.append(dict(request_id=r['request_id'],preview_member=name,preview_sha256=hashlib.sha256(b.getvalue()).hexdigest(),original_path=str(original),original_sha256=r['image_sha256'],original_size=r['pixels'],preview_size=list(im.size)))
    members['gallery/preview_provenance.json']=(json.dumps(dict(all32_included=True,uniform_resize_only=True,used_for_actual_review=False,actual_reviews_used_full_originals=True,rows=previews),indent=2)+'\n').encode()
    links=''.join('<h3>'+r['request_id']+'</h3><img style="max-width:100%" src="images/'+Path(r['preview_member']).name+'">' for r in previews)
    members['gallery/index.html']=('<meta charset="utf-8"><p>All32 paired pages; uniformly reduced previews. Actual reviews used original full resolution.</p>'+links).encode()
    compact=[]
    for q in sorted((OUT/'evaluation').glob('*/scores/*/*/score.json')):
        a=D.read(q)
        compact.append(dict(score_path=str(q),score_sha256=D.sha(q),identity=a['identity'],metrics=a['metrics'],free_face_denominator=a['free_face_denominator'],C_preserved_exact=a['C_preserved_exact'],topology={k:v for k,v in a['topology'].items() if not isinstance(v,(list,dict))}))
    assert len(compact)==192
    members['scores_compact.json']=(json.dumps({'rows':compact,'original_geometric_detail_arrays_external':True},separators=(',',':'))+'\n').encode()
    sources={}
    for q in (ROOT/'r2/pilots/nft_training_c_frozen_v1').glob('*'):
        if q.is_file() and q.suffix in ('.py','.ps1'):sources[str(q)]=D.sha(q)
    sources.update(D.read(OUT/'execution_source_lock.json')['hashes'])
    sources.update(compact[0]['identity']['definition_hashes'])
    source_rows=[];diff=''
    for i,(path,h) in enumerate(sorted(sources.items())):
        q=D.resolve(path);assert D.sha(q)==h
        rel=str(q.relative_to(ROOT)) if q.is_relative_to(ROOT) else 'external/'+str(i)+'_'+q.name
        name='source/'+rel;add(q,name);source_rows.append({'path':str(q),'sha256':h,'member':name})
        if q.parent==ROOT/'r2/pilots/nft_training_c_frozen_v1':diff+=''.join(difflib.unified_diff([],q.read_text(encoding='utf-8-sig').splitlines(True),fromfile='/dev/null',tofile=rel))
    members['source/code.diff']=diff.encode()
    members['source/source_hashes.json']=(json.dumps(source_rows,indent=2)+'\n').encode()
    manifest=D.read(OUT/'inputs/manifest.json')
    for label in ['BASE','NFT','SELF_FM']:
        for q in manifest['evaluation_requests']:
            rid=q['request_id'];base=OUT/'evaluation'/label
            for path in [base/'raw'/rid/'raw.npz',base/'post'/rid/'post.npz',base/'proposal'/rid/'proposal.npz']:
                external.append(dict(path=str(path),bytes=path.stat().st_size,sha256=D.sha(path),reason='Full mesh/trajectory remains local; numerical tables and all image previews included'))
    members['external_mesh_assets.json']=(json.dumps(external,indent=2)+'\n').encode()
    members['PACK_SCOPE.txt']=b'All96 output rows, all192 RAW/POST metric rows, all32 paired image previews, actual review receipts and source code are included. Original full-resolution images, full geometry and detailed scorer arrays remain at their recorded local paths; this ZIP supports numeric aggregation review, not independent geometric rescoring without external meshes. Frozen model weights are unchanged and are external.\n'
    checks={name:hashlib.sha256(data).hexdigest() for name,data in members.items()}
    members['checksums.json']=(json.dumps(checks,indent=2)+'\n').encode()
    target=dest/'NFT_TRAINING_C_FROZEN_REVIEW.zip'
    if target.exists():raise RuntimeError('Pack already exists; no overwrite')
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(members.items()):z.writestr(name,data)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,h in checks.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    result=dict(status='COMPLETE',path=str(target),bytes=target.stat().st_size,sha256=D.sha(target),members=len(members),all_member_hashes_verified=True,CRC_PASS=True,all96_outputs_included=True,all32_preview_pages=True,negative_and_unknown_rows_retained=True,model_calls=0)
    D.write(dest/'receipt.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':build()
