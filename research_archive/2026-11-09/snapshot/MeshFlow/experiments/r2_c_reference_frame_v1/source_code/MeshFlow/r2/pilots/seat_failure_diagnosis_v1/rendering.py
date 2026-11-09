"""Fixed 24-page CPU gallery: complete meshes plus C-centered exact cross sections.

Reads only frozen registration and RAW/POST indices, never quality rankings.
The 80-row review template remains NOT_REVIEWED until actual visual inspection.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,html,json,time
from pathlib import Path
import numpy as np
from . import common as C

GALLERY=C.OUT/'evaluation/gallery'
REVIEW_FIELDS=('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer',
               'whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')
META=('request_id','pair_id','condition_id','base_condition_id','parent_id','panel','split',
      'condition_type','region_type','tier','N','K','variant','candidate')
SCHEMA='r2_seat_failure_fixed_sections_v1'
CAMERA=(22.,-60.)
SECTION_EPS=1e-12*C.L
DEFINITION=dict(schema=SCHEMA,expected_pages=24,expected_review_rows=80,
    base_pages='8 pairs x 2 noises; TRAIN and DEV groups, RAW/POST rows',
    retessellation_pages='4 conditions x 2 noises; base and sub4 groups, RAW/POST rows',
    each_group_views=['whole mesh orthographic oblique','front XY section at area-weighted C center Z','side ZY section at area-weighted C center X'],
    global_bounds='Union of all displayed meshes on page, including every fragment; identical RAW/POST and both columns.',
    local_bounds='Union C bbox on page plus .075L X/Z margin, .10L Y margin; fixed from C only, never fitted to free outputs.',
    sections='Triangle/plane intersection segments, all triangles considered; coplanar triangles contribute edges. Thin pale blue projected C outlines show its overall extent, thick blue segments are actual section intersections. Gray segments are F intersections.',
    coordinate_frame='Model Y up, plot whole=(X,Z,Y). Local 2D axes explicitly model coordinates.',
    slice_plane='Per-condition area-weighted C centroid; same RAW/POST. Base and exact-surface sub4 centroid should coincide to floating precision.',
    no_geometry_change=True,no_smoothing=True,no_quality_ranking=True,
    limitations='A central slice can miss off-plane parts or attachments; whole view retains them. Rendering cannot certify actual index connectivity. Labels disclose TRAIN/DEV and variants; review nonblind AI_PROXY.')

def verify(hashes):
    for p,h in hashes.items():
        if C.sha(p)!=h:raise ValueError('Render source changed: '+p)

def layout(reg):
    req=reg['requests'];assert len(req)==40 and len({q['request_id'] for q in req})==40
    base=collections.defaultdict(dict);extra=collections.defaultdict(dict)
    for q in req:
        if q['variant']=='base':base[q['pair_id'],int(q['candidate'])][q['split']]=q
    for q in req:
        if q['variant']!='base':
            assert q['variant']=='sub4'
            key=(q['pair_id'],q['split'],int(q['candidate']))
            extra[key]['sub4']=q;extra[key]['base']=base[q['pair_id'],int(q['candidate'])][q['split']]
    pages=[]
    for (pair,n),qs in sorted(base.items()):
        assert set(qs)=={'TRAIN','DEV'}
        pages.append(dict(page_id=f'{pair}__n{n}__TRAIN_DEV',kind='TRAIN_DEV',pair_id=pair,candidate=n,
                          titles=['TRAIN','DEV'],requests=[qs['TRAIN'],qs['DEV']]))
    for (pair,split,n),qs in sorted(extra.items()):
        assert set(qs)=={'base','sub4'}
        pages.append(dict(page_id=f'{pair}__{split}__n{n}__BASE_SUB4',kind='BASE_SUB4',pair_id=pair,candidate=n,
                          titles=[split+' base',split+' sub4'],requests=[qs['base'],qs['sub4']]))
    assert len(pages)==24
    return pages

def get_indices(reg):
    result={}
    for stage in ('raw','post'):
        obj=C.read(C.OUT/'evaluation'/f'{stage}_index.json');rs=obj['rows']
        if obj.get('status')!='COMPLETE' or len(rs)!=40:raise ValueError('Incomplete '+stage+' index')
        mapping={r['request_id']:r for r in rs}
        if set(mapping)!={q['request_id'] for q in reg['requests']}:raise ValueError('Index request set differs')
        result[stage.upper()]=mapping
    return result

def center_of_c(tri):
    a=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    return np.average(tri.mean(1),axis=0,weights=a)

def section_segments(tri,axis,value):
    result=[]
    for face in np.asarray(tri,dtype=np.float64):
        delta=face[:,axis]-value
        if np.all(np.abs(delta)<=SECTION_EPS):
            result.extend([face[[a,b]] for a,b in ((0,1),(1,2),(2,0))]);continue
        points=[]
        for a,b in ((0,1),(1,2),(2,0)):
            if abs(delta[a])<=SECTION_EPS:points.append(face[a])
            if delta[a]*delta[b]<0:points.append(face[a]+(face[b]-face[a])*(-delta[a]/(delta[b]-delta[a])))
        unique=[]
        for p in points:
            if not any(np.linalg.norm(p-v)<=SECTION_EPS for v in unique):unique.append(p)
        if len(unique)>=2:
            best=max(((a,b) for a in range(len(unique)) for b in range(a+1,len(unique))),key=lambda ab:np.linalg.norm(unique[ab[0]]-unique[ab[1]]))
            result.append(np.array([unique[best[0]],unique[best[1]]]))
    return np.asarray(result,dtype=float).reshape(-1,2,3)

def section_plot(ax,mesh,k,center,bounds,view):
    from matplotlib.collections import LineCollection
    axis,project=(2,(0,1)) if view=='front' else (0,(2,1))
    horizontal='X' if view=='front' else 'Z';cut='Z' if view=='front' else 'X'
    # Projected C edge outline is contextual, not a claim the full C lies in cut.
    edges=np.concatenate([mesh[:k,[a,b]] for a,b in ((0,1),(1,2),(2,0))])
    ax.add_collection(LineCollection(edges[:,:,project],colors='#a6d9ef',linewidths=.6,alpha=.65,zorder=1))
    for tri,color,lw,z in ((mesh[k:],'#52585d',.75,2),(mesh[:k],'#057bbd',2.,3)):
        segments=section_segments(tri,axis,float(center[axis]))
        if len(segments):ax.add_collection(LineCollection(segments[:,:,project],colors=color,linewidths=lw,alpha=1,zorder=z))
    lo=np.array(bounds['local_low']);hi=np.array(bounds['local_high'])
    ax.set_xlim(lo[project[0]],hi[project[0]]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.14,linewidth=.5);ax.tick_params(labelsize=6,length=2)
    ax.set_xlabel(horizontal+' (model)',fontsize=7);ax.set_ylabel('Y height',fontsize=7)
    ax.set_title(f'{view} local cut {cut}={center[axis]:.3f}',fontsize=8)

def definitions():
    paths=[Path(__file__),C.CODE/'common.py',C.ROOT/'r2/tools/student_multistep_denoise_v1/gallery.py']
    return {str(p):C.sha(p) for p in paths}

def render():
    reg=C.load_registration();pages=layout(reg);indices=get_indices(reg);defs=definitions()
    frozen=dict(registration_sha256=C.sha(C.OUT/'registration.json'),index_hashes={str(C.OUT/'evaluation'/f'{s}_index.json'):C.sha(C.OUT/'evaluation'/f'{s}_index.json') for s in ('raw','post')},definition_hashes=defs,definition=DEFINITION)
    manifest=GALLERY/'manifest.json'
    if manifest.exists():
        prior=C.read(manifest)
        if prior['identity']!=frozen:raise ValueError('Frozen gallery identity changed')
        verify(prior['output_hashes']);return prior
    C.write(GALLERY/'definition.json',DEFINITION)
    if not (GALLERY/'resources_before.json').exists():C.resources(GALLERY/'resources_before.json')
    G=C.module('_seat_old_draw',C.ROOT/'r2/tools/student_multistep_denoise_v1/gallery.py')
    import matplotlib.pyplot as plt
    from PIL import Image
    receipts=[];reviews={};images=[];cards=[];tick=time.perf_counter()
    for page in pages:
        meshes={};Cs={};centers={};hashes={};items=page['requests']
        for q in items:
            rid=q['request_id'];k=int(q['K']);inp=C.arrays(q['input_path']);src=C.arrays(q['condition_source'])
            # Inputs are frozen; C key has no hidden source completion.
            cc=inp['C'].reshape(k,3,3);original=src['C_original'].reshape(k,3,3)
            Cs[rid]=cc;centers[rid]=center_of_c(cc.astype(float))
            hashes[q['input_path']]=q['input_sha256'];hashes[q['condition_source']]=q['condition_source_sha256']
            for stage in ('RAW','POST'):
                r=indices[stage][rid];pk='raw_path' if stage=='RAW' else 'post_path';hk='raw_sha256' if stage=='RAW' else 'post_sha256'
                p=r[pk];arrays=C.arrays(p);mesh=arrays['raw' if stage=='RAW' else 'output'].reshape(-1,3,3)
                if mesh.dtype!=np.float32 or not np.isfinite(mesh).all():raise ValueError('Bad render array')
                if not C.same(mesh[:k],cc if stage=='RAW' else original):raise ValueError('Rendered C identity differs')
                if stage=='POST' and not C.same(arrays['vertices'][arrays['faces']],mesh):raise ValueError('POST indices differ')
                meshes[rid,stage]=mesh;hashes[p]=r[hk]
        verify(hashes)
        allxyz=np.concatenate([m.reshape(-1,3) for m in meshes.values()]).astype(float);lo=allxyz.min(0);hi=allxyz.max(0)
        allc=np.concatenate([x.reshape(-1,3) for x in Cs.values()]).astype(float);clo=allc.min(0);chi=allc.max(0)
        margins=np.array([.075,.10,.075])*C.L
        bounds=dict(center_model_xyz=((lo+hi)/2).tolist(),span=float((hi-lo).max()*1.10),local_low=(clo-margins).tolist(),local_high=(chi+margins).tolist(),C_section_centers={k:v.tolist() for k,v in centers.items()})
        ident=C.clean(dict(**page,hashes=hashes,bounds=bounds,identity=frozen,dimensions=[1950,980]))
        dest=GALLERY/'images'/(page['page_id']+'.jpg');side=dest.with_suffix('.json')
        if side.exists():
            receipt=C.read(side)
            if receipt['identity']!=ident or C.sha(dest)!=receipt['image_sha256']:raise ValueError('Existing page differs')
        else:
            if dest.exists():raise RuntimeError('Incomplete image requires explicit engineering recovery')
            fig=plt.figure(figsize=(15,7.538461538461538),dpi=130,facecolor='white')
            for i,stage in enumerate(('RAW','POST')):
                for j,q in enumerate(items):
                    rid=q['request_id'];k=int(q['K']);m=meshes[rid,stage]
                    ax=fig.add_subplot(2,6,i*6+j*3+1,projection='3d');G.draw(ax,m,k,bounds,CAMERA)
                    ax.set_title(page['titles'][j]+' | '+stage+f' | K={k} F={len(m)-k}\n'+q['parent_id'][:8],fontsize=8)
                    for v,view in enumerate(('front','side'),2):
                        ax=fig.add_subplot(2,6,i*6+j*3+v);section_plot(ax,m,k,centers[rid],bounds,view)
            warning=' | PAIR04: source geometry near-duplicate caution' if page['pair_id']=='PAIR04' else ''
            fig.suptitle(page['page_id']+' | '+items[0]['condition_type']+' | EXP67 frozen'+warning,fontsize=12)
            fig.text(.5,.039,'BLUE: C. GRAY: free. Whole views include all faces; local cuts intentionally show only plane intersections.',ha='center',fontsize=9)
            fig.text(.5,.020,'Pale blue: projected C edges. Thick blue: cut intersects C. Y is vertical. Fixed C-only local bounds; same whole scale per page.',ha='center',fontsize=8)
            fig.text(.5,.004,'Nonblind AI_PROXY review; images and slice contacts do not certify shared-edge topology. No smoothing or mesh edits.',ha='center',fontsize=8)
            fig.subplots_adjust(left=.015,right=.99,top=.88,bottom=.115,wspace=.27,hspace=.33)
            dest=C.checked(dest);dest.parent.mkdir(parents=True,exist_ok=True);tmp=C.checked(dest.with_name(dest.name+'.tmp'))
            with tmp.open('xb') as f:fig.savefig(f,format='jpg',dpi=130,pil_kwargs={'quality':91,'optimize':True})
            plt.close(fig);os.replace(tmp,dest)
            with Image.open(dest) as im:im.verify()
            verify(hashes);receipt=dict(status='RENDERED_NOT_REVIEWED',identity=ident,image_path=str(dest),image_sha256=C.sha(dest));C.write(side,receipt)
        receipts.append(receipt);images.append(dict(page_id=page['page_id'],kind=page['kind'],pair_id=page['pair_id'],candidate=page['candidate'],image_path=str(dest),image_sha256=receipt['image_sha256'],requests=[q['request_id'] for q in items]))
        for q in items:
            for stage in ('RAW','POST'):
                key=(q['request_id'],stage)
                if key not in reviews:reviews[key]=dict(**{k:q[k] for k in META},stage=stage,label='EXP67_FROZEN',review_status='NOT_REVIEWED',reviewer='AI_PROXY',blind=False,image_paths=[],image_sha256s=[],reviewed_image_sha256='',**{k:'UNKNOWN' for k in REVIEW_FIELDS},notes='')
                reviews[key]['image_paths'].append(str(dest));reviews[key]['image_sha256s'].append(receipt['image_sha256'])
        relative='images/'+dest.name
        cards.append('<section><h2>'+html.escape(page['page_id'])+'</h2><a href="'+relative+'"><img style="width:100%;max-width:1950px" loading="lazy" src="'+relative+'"></a></section>')
        C.write(GALLERY/'progress.json',dict(status='RENDERING',images=len(receipts),expected=24),replace=True)
        print(json.dumps(dict(page=page['page_id'],images=len(receipts),expected=24)),flush=True)
    assert len(reviews)==80 and len(receipts)==24
    template=GALLERY/'AI_PROXY_review_template.csv';C.csvwrite(template,[reviews[k] for k in sorted(reviews)])
    C.csvwrite(GALLERY/'images.csv',images)
    text='<!doctype html><meta charset="utf-8"><title>EXP67 seat failure diagnosis</title><h1>TRAIN / DEV and exact-surface subdivision</h1><p>24 fixed pages. Whole orthographic view plus two C-centered cross sections, RAW above POST. Do not interpret cut-plane contact as exact topology. PAIR04 source shapes are highly similar; assess separately for generalization.</p>'+''.join(cards)
    C._write_bytes(GALLERY/'index.html',text.encode('utf8'))
    outputhash={str(p):C.sha(p) for p in sorted(GALLERY.rglob('*')) if p.is_file() and p.name not in ('manifest.json','progress.json')}
    result=dict(status='RENDERED_NOT_REVIEWED',identity=frozen,images=24,review_rows=80,seconds=time.perf_counter()-tick,receipts=receipts,output_hashes=outputhash,actual_visual_review='NOT_RUN_BY_RENDERER',model_calls=0,quality_calls=0)
    C.write(manifest,result);C.write(GALLERY/'progress.json',dict(status='RENDERED_NOT_REVIEWED',images=24,expected=24),replace=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.check_only:
        r=C.load_registration();p=layout(r);print(json.dumps(dict(status='RENDER_LAYOUT_CHECK_PASS',requests=len(r['requests']),pages=len(p),expected_review_rows=80,actual_renders=0,model_calls=0)))
    else:
        result=render();print(json.dumps({k:v for k,v in result.items() if k not in ('receipts','output_hashes','identity')}))
if __name__=='__main__':main()
