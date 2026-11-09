"""Matched CPU depth views of all retained outcomes, including blocked targets."""
from .common import *
import io, html
from PIL import Image,ImageDraw
from r2.data_tools.bench_user_v2.render import cell,basis,shared_bounds,font,text_lines
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'

def write_image(p,im,meta):
    p.parent.mkdir(parents=True,exist_ok=True);buf=io.BytesIO();im.save(buf,format='PNG');data=buf.getvalue()
    if p.exists():assert p.read_bytes()==data
    else:p.write_bytes(data)
    receipt=dict(image_path=str(p),image_sha256=sha(p),review_status='NOT_REVIEWED',model_calls=0,CPU_raster=True,**meta)
    save(p.with_suffix('.json'),receipt);return receipt

def parent(r):
    a=load(r['source_parent']['clean_case_path']);z=load(r['target_case_path']);t=a['model_vertices'][a['source_faces']];tt=z['model_vertices'][z['source_faces']]
    views=[('oblique front',-60,22),('oblique back',120,22),('front',-90,0),('side',0,0)]
    size=620;rowheight=620;head=100
    im=Image.new('RGB',(size*2,head+len(views)*rowheight),'white');d=ImageDraw.Draw(im);records=[]
    text_lines(d,(15,8),r['parent_id']+' | '+r['method']+' | '+r['outcome'],size*2-30,19,2)
    d.text((15,64),'Original M | N='+str(len(t)),font=font(21),fill='black');d.text((size+15,64),'Proposed M\u2032 | N='+str(len(tt)),font=font(21),fill='black')
    for row,(name,az,el) in enumerate(views):
        B=basis(az,el);center,span=shared_bounds([t,tt],[B])
        for col,tri in enumerate([t,tt]):
            image,stats=cell(tri,0,B,center,span,size=size)
            im.paste(image,(col*size,head+row*rowheight))
            records.append(dict(view=name,column=col,center=center.tolist(),span=span,visible_pixels=stats))
        d.text((14,head+row*rowheight+6),name,font=font(18),fill='#333333')
    return write_image(OUT/'gallery'/('target_'+r['parent_id']+'_fullframe.png'),im,dict(parent_id=r['parent_id'],kind='COMPLETE_TARGET',views=records,source_sha256=sha(r['source_parent']['clean_case_path']),target_sha256=r['target_case_sha256'],
      note='Per-view common source/target scale. All source topology outcomes included. Pixel render does not certify finite gaps.'))

def condition(r):
    p=load(r['paired_path']);a=load(r['source_case_path']);t=a['model_vertices'][a['source_faces']];k=int(p['K'])
    C=p['C_vertices'][p['C_faces_original']];Y=a['model_vertices'][a['source_faces'][p['free_ids']]]
    groups=[('Complete M\u2032',t,0),('Input C\u2032',C,len(C)),('Complement Y\u2032',Y,0),('C\u2032 + Y\u2032',np.concatenate([C,Y]),len(C))]
    size=440;head=130;views=[('front oblique',-60,22),('back oblique',120,22)]
    im=Image.new('RGB',(size*4,head+size*len(views)),'white');d=ImageDraw.Draw(im);records=[]
    text_lines(d,(12,5),r['condition_id']+' | '+r['proposed_region'],size*4-24,19,2)
    text_lines(d,(12,57),f"N={int(p['N'])} K={k} | BLUE=input C | {r.get('status','PENDING')} | No model generation",size*4-24,17,2)
    for j,(label,tri,known) in enumerate(groups):d.text((j*size+12,103),label,font=font(19),fill='black')
    for row,(name,az,el) in enumerate(views):
        B=basis(az,el);center,span=shared_bounds([t],[B])
        for col,(_,tri,known) in enumerate(groups):
            cellim,stats=cell(tri,known,B,center,span,size=size);im.paste(cellim,(col*size,head+row*size))
            records.append(dict(view=name,column=col,center=center.tolist(),span=span,visibility=stats))
    return write_image(OUT/'gallery'/('condition_'+r['condition_id']+'.png'),im,dict(condition_id=r['condition_id'],parent_id=r['parent_id'],kind='CONDITION',views=records,
      paired_sha256=sha(r['paired_path']),target_sha256=sha(r['source_case_path'])))

def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    from r2.pilots.bench_adaptation_strength_v1.common import resources
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['parents','conditions']);args=ap.parse_args();event('CPU_RENDER_DISPATCH',workers=2,threads=1,resources=resources())
    rows=read(OUT/'selected_targets.json')['rows'] if args.stage=='parents' else read(OUT/'tasks/manifest.json')['rows']
    fn=parent if args.stage=='parents' else condition
    with ProcessPoolExecutor(max_workers=2) as pool:receipts=list(pool.map(fn,rows))
    save(OUT/'gallery'/(args.stage+('_fullframe' if args.stage=='parents' else '')+'_manifest.json'),dict(rows=receipts,review_status='NOT_REVIEWED'))
    images=sorted((OUT/'gallery').glob('*.png'))
    doc='<meta charset="utf-8"><title>Bench three family targets</title><h1>Three fixed families: target and C/Y review</h1><p>CPU data construction. No generated model outputs. Blocked outcomes retained. Actual reviews in reviews/.</p>'+''.join('<h2>'+html.escape(p.stem)+'</h2><img style="max-width:100%" src="'+p.name+'">' for p in images)
    # Gallery index is navigation only and can include newly finished stage pages.
    (OUT/'gallery/index.html').write_text(doc,'utf-8');print('RENDERED',args.stage,len(receipts))

if __name__=='__main__':main()
