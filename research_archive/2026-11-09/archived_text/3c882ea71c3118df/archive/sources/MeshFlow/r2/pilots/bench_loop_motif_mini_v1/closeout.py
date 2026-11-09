"""Archive only completed fixed experiment; no generation or scoring."""
import sys,json,csv,collections,statistics,hashlib,difflib,shutil,zipfile,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'r2/src')]
from r2.pilots.bench_loop_motif_mini_v1 import run as X
from r2.pilots.bench_loop_motif_mini_v1.analyze import savecsv
O=X.OUT
def readcsv(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def reviews():
    notes=X.read(O/'analysis/actual_visual_notes.json');out=[]
    prefix=lambda p: next(s for s in ('6b','a8','28b9','bcb') if p.startswith(s))
    for q in X.read(O/'inputs/requests.json'):
        for stage in ('RAW','CLEAN_FINAL_POST'):
            v=notes[stage][prefix(q['parent_id'])+'/'+str(q['noise_index'])]
            png=O/'gallery/loop_new'/(stage+'__'+q['request_id'])/'comparison.png'
            for method in ('L0','L3'):
                r=dict(method=method,stage=stage,request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],reviewer='AI_PROXY',method_labels_hidden=False,review_status='ACTUALLY_VIEWED',actually_viewed=True,source_image=str(png),source_image_sha256=X.sha(png),observation=v[5])
                r.update(dict(zip(('body','adoption','support','fragment_evidence','folding'),v[:5])))
                X.E._normalize_review(r);r['J']=int(r['body']=='REASONABLE' and r['adoption']=='PLAUSIBLE' and r['support']=='PRESENT');out.append(r)
    assert len(out)==32
    savecsv('actual_visual_new32.csv',out)
    ids={q['request_id'] for q in X.read(O/'inputs/requests.json')}
    oldfile=X.PREV/'analysis_numeric/final/visual/actual_review_per_output.csv'
    old=[r for r in readcsv(oldfile) if r['request_id'] in ids and r['method'] in ('R','G0','G1','G3') and r['main_panel']=='True' and r['stage'] in ('RAW','CLEAN_FINAL_POST')]
    assert len(old)==64 and len({(r['method'],r['stage'],r['request_id']) for r in old})==64
    for r in old:r.update(origin='PREVIOUS_ACTUAL_REVIEW_UNCHANGED',source_review=str(oldfile),source_review_sha256=X.sha(oldfile))
    savecsv('actual_visual_all96.csv',old+out)
    combined=old+out;s=[];paired=[]
    for stage in ('RAW','CLEAN_FINAL_POST'):
        for method in ('R','G0','G1','G3','L0','L3'):
            rs=[r for r in combined if r['stage']==stage and r['method']==method]
            s.append(dict(stage=stage,method=method,outputs=len(rs),J=sum(int(r['J']) for r in rs),body_reasonable=sum(r['body']=='REASONABLE' for r in rs),adoption_plausible=sum(r['adoption']=='PLAUSIBLE' for r in rs),support_present=sum(r['support']=='PRESENT' for r in rs)))
        for a,b in [('L0','G0'),('L3','G3'),('L3','L0'),('L3','G1'),('L3','R')]:
            for rid in sorted(ids):
                ar=next(r for r in combined if r['method']==a and r['stage']==stage and r['request_id']==rid)
                br=next(r for r in combined if r['method']==b and r['stage']==stage and r['request_id']==rid)
                paired.append(dict(comparison=a+'-'+b,stage=stage,request_id=rid,parent_id=ar['parent_id'],J_a=int(ar['J']),J_b=int(br['J']),J_delta=int(ar['J'])-int(br['J']),adoption_a=ar['adoption'],adoption_b=br['adoption'],body_a=ar['body'],body_b=br['body'],support_a=ar['support'],support_b=br['support']))
    savecsv('visual_summary.csv',s);savecsv('visual_request_transitions.csv',paired)
    print(json.dumps(s),flush=True)
def archive():
    X.G.verify(X.read(O/'registration.json')['protected'])
    assert len(list(O.glob('cpu/*/complete.json')))==16
    sources={Path(p) for p in X.read(O/'evaluation_code.json')['source_hashes']}
    sources.update(Path(p) for p in X.read(O/'registration.json')['protected'] if p.endswith('.py'))
    sources.update(Path(__file__).parent.glob('*.py'))
    sources.update([ROOT/'r2/pilots/bench_c_edgeflip_v1/candidate_post.py',ROOT/'r2/pilots/bench_full_prior_motif_guidance_v1/guidance.py',ROOT/'r2/pilots/bench_full_prior_motif_guidance_v1/run.py'])
    hashes=[];diff=[]
    for p in sorted(sources):
        relative=str(p).replace('/mnt/e/','').replace(':','');dest=O/'archive/sources'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        hashes.append(dict(source=str(p),archived=str(dest.relative_to(O)),sha256=X.sha(p),bytes=p.stat().st_size))
        if p.parent==Path(__file__).parent:diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(keepends=True),fromfile='/dev/null',tofile=str(p.relative_to(ROOT))))
    write(O/'archive/source_hashes.json',hashes);(O/'archive/code.diff').write_text(''.join(diff),encoding='utf-8')
    receipts=[]
    for p in O.glob('workflow/attempts/*/completion.json'):receipts.append(dict(attempt=p.parent.name,**X.read(p)))
    savecsv('actual_phase_resources.csv',receipts)
    call=collections.Counter()
    for t in X.tasks():
        a=X.read(Path(t['output_dir'])/'audit.json')
        for key in ('generator_forwards','input_VJP','training_updates'):call[key]+=a[key]
        for line in (Path(t['output_dir'])/'calls.jsonl').read_text().splitlines():
            d=json.loads(line)
            if d['event'] in ('CALL_BEGIN','CALL_END'):call[d['event']+'__'+d['kind']]+=1
    assert call['generator_forwards']==800 and call['input_VJP']==360 and call['training_updates']==0
    assert call['CALL_END__vector_field']==800 and call['CALL_END__input_VJP']==360
    assert all(X.read(p)['D0_forwards']==1 for p in O.glob('post/*/receipt.json'))
    write(O/'analysis/final_budget_validation.json',dict(status='PASS',sampling=dict(call),gate=X.read(O/'checks/r1/complete.json'),D0=16,score_rows=144,new_score_rows=48,reused_score_rows=96,new_actual_visual=32,reused_actual_visual=64,parameter_updates=0))
    body=['<!doctype html><meta charset="utf-8"><title>Loop motif fixed comparison</title><h1>RAW / FINAL · R, G0, G1, G3, L0, L3</h1><p>固定视角及截面；AI_PROXY图审另见CSV。点击图片查看原图。</p>']
    for q in X.read(O/'inputs/requests.json'):
        body.append('<h2>'+html.escape(q['request_id'])+'</h2>')
        for stage in ('RAW','CLEAN_FINAL_POST'):
            rel='loop_development/'+stage+'__'+q['request_id']+'/comparison.png'
            body.append(f'<h3>{stage}</h3><a href="{rel}"><img width="100%" src="{rel}"></a>')
    (O/'gallery/index.html').write_text('\n'.join(body),encoding='utf-8')
    files=[O/'REPORT.md',O/'PROTOCOL.md',O/'RECOVERY.md',O/'registration.json',O/'tasks.json',O/'baseline.json',O/'evaluation_code.json']
    for name in ('analysis','archive','gallery','inputs','checks','workflow'):
        files.extend(p for p in (O/name).rglob('*') if p.is_file() and p.suffix not in ('.pyc',))
    for t in X.tasks():
        files.extend([Path(t['raw_path']),Path(t['raw_receipt_path']),Path(t['output_dir'])/'audit.json'])
        files.extend([O/'post'/t['cell_id']/'post.npz',O/'post'/t['cell_id']/'receipt.json',O/'cpu'/t['cell_id']/'CLEAN_FINAL_POST/post.npz'])
    files=sorted(set(files));manifest=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=X.sha(p)) for p in files]
    write(O/'DELIVERY_MANIFEST.json',manifest)
    zipfile_path=O/'R2_BENCH_LOOP_MOTIF_MINI_REVIEW.zip'
    with zipfile.ZipFile(zipfile_path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files+[O/'DELIVERY_MANIFEST.json']:z.write(p,p.relative_to(O))
    with zipfile.ZipFile(zipfile_path) as z:assert z.testzip() is None
    write(O/'COMPLETE.json',dict(status='COMPLETE',zip_bytes=zipfile_path.stat().st_size,zip_sha256=X.sha(zipfile_path),new_generation=16,D0=16,training=0,automatic_followup=False,report_sha256=X.sha(O/'REPORT.md')))
    print(json.dumps(X.read(O/'COMPLETE.json')),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='reviews':reviews()
    elif sys.argv[1]=='archive':archive()
