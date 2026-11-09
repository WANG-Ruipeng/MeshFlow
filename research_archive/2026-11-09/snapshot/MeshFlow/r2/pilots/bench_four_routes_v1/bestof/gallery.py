"""All four RAW candidates and the two fixed/selected POST references; CPU only."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,html,io,json,resource,time
import numpy as np
from .common import *

GALLERY=OUT/'gallery'
AXES={'overall_shape':['REASONABLE','INCOMPLETE','UNCERTAIN'],
      'C_use':['PLAUSIBLE','REPLACED_OR_IGNORED','UNCERTAIN'],
      'support_retention':['PRESENT','MISSING_OR_DAMAGED','UNCERTAIN'],
      'visible_fragment_or_gap':['YES','NO_OBVIOUS','UNCERTAIN'],
      'layering_or_fold':['YES','NO_OBVIOUS','UNCERTAIN']}


def cell(q,label,stage):
    if stage=='RAW':mesh=raw_dir(q)/'raw.npz';receipt=raw_dir(q)/'receipt.json'
    elif stage=='BASE_POST':mesh=post_dir(q)/'post.npz';receipt=post_dir(q)/'receipt.json'
    elif stage=='CLEAN_FINAL_POST':
        folder=score_dir(q,stage).parent;mesh=folder/'post.npz';receipt=folder.parent/'complete.json'
    else:raise ValueError(stage)
    return dict(model=label,label=label+' '+stage,stage=stage,path=str(mesh),receipt=str(receipt),
        request_id=q['request_id'],noise_index=q['noise_index'],noise_sha256=q['noise_sha256'])


def prepare(kind):
    registered()
    if kind not in ('raw','post'):raise ValueError(kind)
    path=GALLERY/(kind+'_manifest.json')
    if path.exists():r=read(path);verify(r['protected']);return r
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    by=collections.defaultdict(list)
    for q in candidates():by[q['condition_id']].append(q)
    winner={}
    if kind=='post':
        from .run import selected
        winner={q['condition_id']:q for q in selected()}
        if len(winner)!=23:raise ValueError('Explicit failed-condition display needed; no silent gallery omission')
    pages=[]
    for cid,part in sorted(by.items()):
        part.sort(key=lambda q:q['noise_index']);base=part[0]
        stages=('RAW',) if kind=='raw' else ('BASE_POST','CLEAN_FINAL_POST')
        for stage in stages:
            cells=([cell(q,'CANDIDATE_N'+str(q['noise_index']),stage) for q in part] if kind=='raw' else
                   [cell(part[0],'FIXED_N0',stage),cell(part[1],'FIXED_N1',stage),cell(winner[cid],'SELECTED',stage)])
            pages.append(dict(condition_id=cid,parent_id=base['parent_id'],request_id=cid,
                report_group=base['report_group'],part_type=base['part_type'],split=base['split'],K=base['K'],N=base['N'],
                closed_C=base['closed_C'],open_interface_status='NA_CLOSED_C' if base['closed_C'] else 'DESCRIPTIVE_NOT_REQUIRED_INTERFACE',
                page_id=stage+'__'+cid,sheet=stage,cells=cells,inference_npz=base['inference_npz'],
                bounds=base['gallery_bounds'],depth_bounds=base['gallery_depth_bounds']))
    protected={str(p):sha(p) for p in (MANIFEST,Path(__file__),Path(G.__file__),Path(G.D.__file__),Path(G.H.__file__))}
    if kind=='post':protected[str(OUT/'selection/winners.json')]=sha(OUT/'selection/winners.json')
    plan=dict(status='FROZEN',version='bestof_complete_candidate_gallery_v2',kind=kind,pages=pages,
        sheets=len(pages),cells=sum(len(p['cells']) for p in pages),protected=protected,
        cameras='Inherited azimuth -60/120 elevation23; original center/span; no fit to candidates',
        sections='Inherited central front/side finite planes, original bounds; endpoints alone do not certify gaps',
        scope='All92 RAW candidates; fixedn0/fixedn1/SELECTED BASE/FINAL; no scores/ranks/winner marks in RAW sheets',
        interpretation='Nonblind AI_PROXY V2; no unique-source target; unchanged B weights',
        GT_display=False,model_calls=0,scoring_calls=0,geometry_edits=0)
    immutable(path,plan);return plan


def one(page,definition_sha):
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    D,H=G.D,G.H;png=GALLERY/(page['page_id']+'.png');receipt=png.with_suffix('.json')
    if receipt.exists():
        r=read(receipt)
        if r['definition_sha256']!=definition_sha or r['image_sha256']!=sha(png):raise ValueError('Frozen image changed')
        return r
    if png.exists():raise RuntimeError('Unsealed image retained; requires explicit recovery')
    tick=time.perf_counter();plt=H._mpl();cols=len(page['cells'])
    fig=plt.figure(figsize=(5.3*cols,13.5),dpi=110,facecolor='white')
    grid=fig.add_gridspec(4,cols,left=.025,right=.99,bottom=.055,top=.91,wspace=.15,hspace=.23)
    inp=arrays(page['inference_npz']);original=inp['C_vertices'][inp['C_faces_original']]
    K,bounds,depth=page['K'],page['bounds'],page['depth_bounds'];display=[]
    try:
        for col,info in enumerate(page['cells']):
            tri,role=G.roles(arrays(info['path']),K)
            g=dict(C=tri[:K],F=tri[K:],C_original=original,tri=tri)
            facts=dict(label=info['model'],request_id=info['request_id'],noise_index=info['noise_index'],views=[])
            for row,az in enumerate((-60,120)):
                raster,scope=G.role_raster(tri,role,K,D.basis(az,23),np.asarray(depth['center']),depth['span'])
                ax=fig.add_subplot(grid[row,col]);ax.imshow(raster);ax.set_axis_off()
                ax.set_title(info['model']+' | n'+str(info['noise_index'])+'\n'+info['stage']+' | '+str(az),fontsize=9)
                facts['views'].append(dict(azimuth=az,**scope))
                if scope['projected_vertices_outside_inherited_frame']:
                    ax.text(.01,.01,'Inherited-frame clipping: see receipt',transform=ax.transAxes,fontsize=7,color='#b34b27')
            for row,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[row,col]);H.draw_section(ax,g,np.flatnonzero(role[K:]=='SEAM'),bounds,G.C.L,view,1)
                from matplotlib.collections import LineCollection
                axis,project=(2,(0,1)) if view=='front' else (0,(2,1))
                pos=bounds['front_positions' if view=='front' else 'side_positions'][1]
                ff,ids=H.finite_sections(g['F'],axis,pos,1e-12*G.C.L);bm=role[K:][ids]=='BRIDGE'
                if bm.any():ax.add_collection(LineCollection(ff[bm][:,:,project],colors='#41a46c',linewidths=1.6,zorder=3))
                ax.set_title(info['model']+' | '+view+f'={pos:.5f}',fontsize=8)
            display.append(facts)
        closed=' | closed C: no required open seam' if page['closed_C'] else ''
        fig.suptitle(page['condition_id']+' | '+page['report_group']+f' | K={K} N={page["N"]}'+closed+'\n'
            +page['sheet']+' | Same frozen narrow B; separate preregistered noises | Blue C / gray BULK / green BRIDGE',fontsize=11)
        fig.text(.5,.018,'Actual nonblind V2 review separate. No score-based labels or unique-source-style grading. Section endpoints alone do not prove cracks.',ha='center',fontsize=9)
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110)
        with png.open('xb') as f:f.write(buffer.getvalue())
    finally:plt.close(fig)
    r=dict(status='PENDING_ACTUAL_VIEW',page_id=page['page_id'],request_id=page['request_id'],condition_id=page['condition_id'],
        sheet=page['sheet'],report_group=page['report_group'],path=str(png),image_sha256=sha(png),definition_sha256=definition_sha,
        cells=page['cells'],display=display,closed_C=page['closed_C'],open_interface_status=page['open_interface_status'],
        seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_calls=0,scoring_calls=0)
    immutable(receipt,r);print('RENDER',page['page_id'],flush=True);return r


def render(kind):
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    plan=prepare(kind);done=GALLERY/(kind+'_completion.json')
    if done.exists():r=read(done);verify(r['files']);return r
    sources={}
    for page in plan['pages']:
        for c in page['cells']:sources.update(G.authenticate(c))
    definition=GALLERY/(kind+'_definition.json')
    immutable(definition,dict(manifest_sha256=sha(GALLERY/(kind+'_manifest.json')),sources=sources,protected=plan['protected']))
    before=G.C.resources();tick=time.perf_counter();images=[]
    # Exactly one CPU renderer may overlap the authorized <=3 scoring workers.
    for page in plan['pages']:images.append(one(page,sha(definition)))
    verify(plan['protected']);verify(sources)
    html_text='<meta charset="utf-8"><h1>Frozen B best-of-four: '+html.escape(kind)+'</h1><p>Nonblind AI_PROXY; all candidates retained. No quality labels are generated here.</p>'
    for im in images:
        name=Path(im['path']).name;html_text+='<h2>'+html.escape(im['page_id'])+'</h2><img loading="lazy" style="width:100%" src="'+name+'">'
    index=GALLERY/(kind+'_index.html')
    if index.exists():
        if index.read_text()!=html_text:raise ValueError('Gallery index changed')
    else:index.write_text(html_text,encoding='utf-8')
    files={str(p):sha(p) for im in images for p in (Path(im['path']),Path(im['path']).with_suffix('.json'))}
    files[str(index)]=sha(index)
    result=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',kind=kind,pages=len(images),cells=plan['cells'],workers=1,
        seconds=time.perf_counter()-tick,resources_before=before,resources_after=G.C.resources(),files=files,
        model_calls=0,scoring_calls=0,ratings_generated=0)
    immutable(done,result);return result


def protocol():
    raw=prepare('raw');post=prepare('post')
    source=ROOT/'experiments/r2_bench_group_weight_match_v1/visual_review/v2_definition.json';prior=read(source)
    definition=dict(schema='BENCH_BESTOF_VISUAL_CRITERIA_V2',status='FROZEN',interpretation_version=2,
        source_definition_path=str(source),source_definition_sha256=sha(source),definitions=prior['definitions'],anchors=prior['anchors'],
        scope='All four B RAW candidates; fixedn0/fixedn1/SELECTED BASE and FINAL. Same model weights, different sampling/selection budget.',
        rules=['Actually view every cell, both oblique views and both original sections.',
               'Five axes independent; concrete per-cell note; uncertainty preserved.',
               'Do not read scores or copy historical ratings to assign labels.',
               'Closed C does not require an open seam; no unique-source-style comparison.',
               'Selection identity is not an adoption or overall-quality label.'])
    immutable(OUT/'visual_review/v2_definition.json',definition)
    units=[]
    for page in raw['pages']+post['pages']:
        for c in page['cells']:
            units.append(dict(page_id=page['page_id'],request_id=c['request_id'],condition_id=page['condition_id'],parent_id=page['parent_id'],
                report_group=page['report_group'],noise_index=c['noise_index'],label=c['model'],stage=c['stage']))
    if len(units)!=230:raise ValueError('Complete candidate+selected comparison coverage')
    result=dict(schema='bestof_all_candidate_actual_view_v2',status='FROZEN',interpretation_version=2,axes=AXES,
        pages=69,cells=230,expected_units=units,definition_sha256=sha(OUT/'visual_review/v2_definition.json'),
        per_page_labels=True,reviewer_kind='AI_PROXY',blind=False,metrics_read=False,historical_label_pooling=False,
        raw_labels=['CANDIDATE_N0','CANDIDATE_N1','CANDIDATE_N2','CANDIDATE_N3'],post_labels=['FIXED_N0','FIXED_N1','SELECTED'],
        protected={str(p):sha(p) for p in (GALLERY/'raw_manifest.json',GALLERY/'post_manifest.json',OUT/'visual_review/v2_definition.json')})
    immutable(OUT/'visual_review/protocol.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare-raw','prepare-post','render-raw','render-post','protocol']);a=p.parse_args()
    if a.action=='protocol':r=protocol()
    elif a.action.startswith('prepare'):r=prepare(a.action.split('-')[1])
    else:r=render(a.action.split('-')[1])
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','pages','cells','protected','expected_units')},indent=2))


if __name__=='__main__':main()
