"""Small complete-case review ZIP; original full-resolution results stay local."""
from pathlib import Path
import csv,difflib,hashlib,html,io,json,zipfile
import numpy as np
from PIL import Image
from . import common as C

def archive(dry_run=False):
    r=C.read(C.OUT/'registration.json');done=C.read(C.OUT/'evaluation/score_completion.json');gallery=C.read(C.OUT/'evaluation/gallery/manifest.json')
    if not (C.OUT/'REPORT.md').exists() or not (C.OUT/'evaluation/gallery/AI_PROXY_reviews.csv').exists():raise ValueError('Report and actual reviews required')
    content={};sources={};meshrefs=[];imagerefs=[]
    def add(name,data):
        data=data if isinstance(data,bytes) else data.encode('utf-8')
        if name in content and content[name]!=data:raise ValueError('Archive name conflict '+name)
        content[name]=data
    def file(p,name=None):
        p=C.resolve(p);add(name or str(p.relative_to(C.OUT)).replace('\\','/'),p.read_bytes());sources[str(p)]=C.sha(p)
    for name in ['REPORT.md','registration.json','ledger.jsonl','resources_start.json','inputs/manifest.json','preparation/condition_audit.json','preparation/input_review_record.json','evaluation/scoring_registration.json','evaluation/score_completion.json','evaluation/collection_complete.json','evaluation/raw_index.json','evaluation/post_index.json','evaluation/cache_reuse.json']:
        p=C.OUT/name
        if p.exists():file(p)
    for folder in ['results','results_v2','independent_audit','historical_context']:
        for p in sorted((C.OUT/folder).glob('*')):
            if p.is_file() and p.suffix in ['.json','.csv','.md','.py']:file(p)
    for p in (C.OUT/'evaluation').glob('*.csv'):file(p)
    for p in (C.OUT/'evaluation/gallery').glob('*.csv'):file(p)
    for p in (C.OUT/'evaluation/gallery').glob('*receipt*.json'):file(p)
    for p in (C.OUT/'evaluation/gallery').glob('review*.json'):file(p)
    for p in (C.OUT/'evaluation/execution').rglob('*.json'):file(p)
    for q in r['requests']:
        file(q['input_path'],'inputs/requests/'+q['request_id']+'.npz')
    for q in r['queries']:file(q['path'],'inputs/queries/'+q['condition_id']+'.npz')
    for c in r['conditions']:file(c['condition_path'],'inputs/sources/'+c['condition_id']+'.npz')
    for stage in ['raw','post']:
        for rec in C.read(C.OUT/'evaluation'/f'{stage}_index.json')['rows']:
            p=rec[stage+'_path'];a=C.arrays(p)
            data={'C':a['C'] if 'C' in a else a['output'][:int(rec['K'])],'mesh':a['raw'] if stage=='raw' else a['output'],'K':np.array(rec['K']),'N_requested':np.array(rec['N'])}
            if stage=='post':data.update({k:a[k] for k in ['vertices','faces'] if k in a})
            b=io.BytesIO();np.savez_compressed(b,**data);name='meshes/'+rec['label']+'/'+rec['request_id']+'__'+stage.upper()+'.npz';add(name,b.getvalue())
            meshrefs.append(dict(archive_path=name,original_path=p,original_sha256=rec[stage+'_sha256'],request_id=rec['request_id'],label=rec['label'],stage=stage.upper(),array_hashes={k:C.array_sha(v) for k,v in data.items()}))
            file(C.OUT/'evaluation'/stage/rec['label']/rec['request_id']/'receipt.json')
    for ref in done['receipts']:
        file(ref['path'],'scores/'+ref['label']+'/'+ref['request_id']+'/'+ref['stage']+'/score.json')
    for i,(p,h) in enumerate(gallery['images'].items()):
        if C.sha(p)!=h:raise ValueError('Image changed')
        img=Image.open(C.resolve(p)).convert('RGB');img.thumbnail((1720,1467));buf=io.BytesIO();img.save(buf,format='JPEG',quality=76,subsampling=0,optimize=True)
        name='gallery/'+Path(p).stem+'.jpg';add(name,buf.getvalue());imagerefs.append(dict(archive_path=name,original_path=p,original_sha256=h,resolution=list(img.size),encoding='JPEG quality76 4:4:4 at max1720x1467; full original PNG local'))
    for c in r['conditions']:
        img=Image.open(C.resolve(c['candidate_image'])).convert('RGB');img.thumbnail((1300,1000));b=io.BytesIO();img.save(b,format='JPEG',quality=78,optimize=True);add('inputs/images/'+c['condition_id']+'.jpg',b.getvalue())
    cards=''.join('<h2>'+html.escape(Path(x['archive_path']).stem)+'</h2><img loading="lazy" style="width:100%" src="'+html.escape(Path(x['archive_path']).name)+'">' for x in imagerefs)
    add('gallery/index.html','<!doctype html><meta charset="utf-8"><title>All regular-patch cases</title>'+cards)
    for p in sorted((C.OUT/'evaluation/depth_gallery').glob('*')):
        if p.is_file() and p.suffix in ['.png','.json','.html']:file(p)
    code=dict(r['code_hashes']);code.update(done['definition_hashes']);code.update(gallery['definition_hashes'])
    for p in C.CODE.glob('*.py'):code[str(p)]=C.sha(p)
    for p,h in code.items():
        p=C.resolve(p)
        if C.sha(p)!=h:raise ValueError('Code changed '+str(p))
        parts=p.parts;start=next((i for i,v in enumerate(parts) if v in ['MeshFlow','MeshFlow-Control']),None)
        name='code/'+('/'.join(parts[start:]) if start is not None else p.name);file(p,name)
    diffs=[]
    for p in sorted(C.CODE.glob('*.py')):
        lines=p.read_text(encoding='utf-8').splitlines(keepends=True);diffs.extend(difflib.unified_diff([],lines,fromfile='/dev/null',tofile='b/'+str(p.relative_to(C.ROOT)),lineterm='\n'))
    add('code.diff',''.join(diffs));add('code_hashes.json',json.dumps(code,indent=2))
    add('mesh_manifest.json',json.dumps(meshrefs,indent=2));add('image_manifest.json',json.dumps(imagerefs,indent=2))
    add('README.txt','R2_REGULAR_PATCH_CONTROL_V1. Read REPORT.md, evaluation/per_output.csv, results/paired_visual.csv, and gallery/index.html. All 96 model/stage final meshes and actual source/query inputs included. Weights and full 51-state trajectories stay at bound local paths; no raw experiment outputs deleted. All 24 output pages included uniformly as JPEG previews; full PNG and per-point/pair score vectors remain local. Historical scores are immutable reused originals, with explicit receipts. Code snapshot includes actual numerical scoring implementations. Three-letter field I is not adoption or watertightness. No training, default model switch, commit or push.\n')
    manifest={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in content.items()}
    add('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2));b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(content.items()):z.writestr(name,data)
    payload=b.getvalue()
    if dry_run:
        print(dict(status='SIZE_PREFLIGHT',bytes=len(payload),entries=len(content),meshes=len(meshrefs),output_images=len(imagerefs),depth_images=24));return
    if len(payload)>=10000000:raise RuntimeError('Package exceeds decimal 10MB: '+str(len(payload)))
    dest=C.OUT/'R2_REGULAR_PATCH_CONTROL_V1_REVIEW_SMALL.zip';C._write_bytes(dest,payload)
    with zipfile.ZipFile(dest) as z:
        if z.testzip():raise ValueError('ZIP CRC')
        for name,meta in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=meta['sha256']:raise ValueError('ZIP content hash')
    result=dict(status='PASS',path=str(dest),bytes=len(payload),sha256=C.sha(dest),entries=len(content),meshes=len(meshrefs),output_images=len(imagerefs),depth_supplement_images=24,original_sources_unchanged=all(C.sha(p)==h for p,h in sources.items()))
    C.write(C.OUT/'package_receipt.json',result);print(result)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');archive(p.parse_args().dry_run)