"""Offline raw-triangle gallery with the three registered control overlays."""
from pathlib import Path
import argparse
import html
import json
import os
import time
import numpy as np
from .common import OUT, LABELS, read_json, resolve, digest, atomic_json
from .registration import _arrays, bits
from .evaluation import load_condition
from experiments.mf_h1_local0.chair_control_usability_quick_v1.gallery import _render, _frame, _display, _page

VERSION='chair_boundary_attachment_gallery_v1'
LABEL_TEXT={'START_HYBRID':'START HYBRID','A_FM_END':'A: FM','B_BOUNDARY_END':'B: Boundary marker',
    'C_SEAM_END':'C: Seam loss','D_BOUNDARY_SEAM_END':'D: Boundary + seam'}


def _overlay(path,C,free,frame,reference,vectors,metrics):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    fig=plt.figure(figsize=(15,5),dpi=130,facecolor='#f6f7f8')
    center=_display(np.asarray(frame['center']));half=frame['half_extent']
    for i,name in enumerate(('Seam coverage','C conflict','Reference exposure retention')):
        ax=fig.add_subplot(1,3,i+1,projection='3d');ax.set_proj_type('ortho')
        # These are original triangles; alpha is only a display property.
        ax.add_collection3d(Poly3DCollection(_display(free),facecolors='#b7bec7',edgecolors='none',alpha=.08,zsort='average'))
        edges=_display(C[:,[[0,1],[1,2],[2,0]],:].reshape(-1,2,3))
        ax.add_collection3d(Line3DCollection(edges,colors='#218ba5',linewidths=.5,alpha=.45))
        handles=[]
        def points(q,mask,color,label,size=3):
            mask=np.asarray(mask,bool)
            if mask.any():
                a=_display(np.asarray(q)[mask]);ax.scatter(a[:,0],a[:,1],a[:,2],s=size,c=color,depthshade=False,alpha=.90)
            handles.append(Line2D([0],[0],marker='o',color='none',markerfacecolor=color,markersize=5,label=label))
        if i==0:
            if 'gamma_points' in reference and 'seam_covered_0.01' in vectors:
                q=reference['gamma_points'];covered=vectors['seam_covered_0.01'];unknown=vectors['seam_uncertain']
                points(q,covered,'#7b8794','Covered Gamma',3)
                points(q,~covered&~unknown,'#d93532','Uncovered Gamma',7)
                points(q,unknown,'#eab308','Uncertain Gamma',7)
                ids=np.unique(vectors['seam_witness_free_edge']);ids=ids[ids>=0]
                if len(ids):
                    e=_display(vectors['seam_free_edges'][ids])
                    ax.add_collection3d(Line3DCollection(e,colors='#1c9c56',linewidths=2.2))
                handles.append(Line2D([0],[0],color='#1c9c56',lw=2,label='Qualifying free edges'))
            else:ax.text2D(.05,.95,'Whole Gamma NA',transform=ax.transAxes,color='#c33')
            score=metrics.get('SeamCov');status=metrics.get('SeamCov_status')
        elif i==1:
            q=reference['C_points'];cross=vectors['conflict_cross'];dup=vectors['conflict_dup']
            points(q,~(cross|dup),'#7ca7b4','Other C points',2)
            points(q,cross&~dup,'#d93532','Cross only',6)
            points(q,dup&~cross,'#b02fc2','Duplicate only',6)
            points(q,cross&dup,'#612ca5','Cross + duplicate',6)
            points(q,vectors['conflict_unknown'],'#eab308','Uncertain',7)
            score=metrics.get('ConflictCov_C');status=metrics.get('ConflictCov_C_status')
        else:
            q=reference['C_points'];eligible=vectors['exposure_reference_exposed']
            points(q,~eligible,'#c4c7cd','Not reference-exposed',2)
            points(q,vectors['exposure_retained_lower'],'#1c9c56','Retained',4)
            points(q,vectors['exposure_occluded'],'#d93532','Occluded',5)
            points(q,vectors['exposure_uncertain'],'#eab308','Output uncertain',7)
            points(q,vectors['exposure_reference_uncertain'],'#945bb7','Reference uncertain',7)
            score=metrics.get('ExposedRetain');status=metrics.get('ExposedRetain_status')
        score_text='NA' if score is None else f'{score:.4f}'
        ax.set_title(name+'\n'+score_text+' | '+str(status),fontsize=10)
        ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half);ax.set_zlim(center[2]-half,center[2]+half)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=45);ax.set_axis_off()
        ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.15),frameon=False,fontsize=7,ncol=2)
    fig.subplots_adjust(left=.005,right=.995,top=.89,bottom=.14,wspace=.01)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.png')
    fig.savefig(tmp,metadata={'Software':VERSION});plt.close(fig);os.replace(tmp,path)


def run(out=OUT,*,allow_partial=False,include_P5=True):
    from PIL import Image
    out=Path(out);tick=time.perf_counter();requests=read_json(out/'requests.json')
    selected=[r for r in requests if r['panel']=='P1' or (include_P5 and r['panel']=='P5')]
    p1=[r for r in selected if r['panel']=='P1']
    assert len(p1)==160
    joined={}
    for r in selected:
        p=out/'scoring/joined'/(r['request_id']+'.json')
        if p.exists():joined[r['request_id']]=read_json(p)
    missing=[r['request_id'] for r in p1 if r['request_id'] not in joined]
    if missing and not allow_partial:
        value=dict(status='BLOCKED',reason='All 160 P1 raw and control vectors must be available',
            missing=missing,expected_P1=160,available_P1=160-len(missing),model_forwards=0)
        atomic_json(out/'gallery/summary.json',value);return value
    conditions={};meshes={};records=[]
    for r in selected:
        row=joined.get(r['request_id'])
        if row is None:continue
        if row['status']!='PASS':continue
        c=conditions.setdefault(r['condition_id'],load_condition(out,r))
        path=resolve(row['raw_path']);raw=_arrays(path)['raw'];n,k=int(c['N']),int(c['K'])
        if digest(path)!=row['raw_sha256']:raise ValueError('Raw changed before gallery')
        assert raw.shape==(n,9) and raw.dtype==np.float32 and np.isfinite(raw).all()
        assert bits(raw[:k],c['C'].reshape(k,9))
        meshes[r['request_id']]=raw[k:].reshape(-1,3,3)
    frames={}
    for uid in sorted({r['uid'] for r in selected}):
        parent=[r for r in selected if r['uid']==uid and r['request_id'] in meshes]
        if parent:
            frames[uid]=_frame([conditions[r['condition_id']]['GT_full'] for r in parent],[meshes[r['request_id']] for r in parent])
    gallery=out/'gallery';assets=gallery/'assets';assets.mkdir(parents=True,exist_ok=True)
    atomic_json(gallery/'frames.json',dict(frames=frames,policy='One union cube per parent across ALL available labels/candidates/registered P1 and P5 contexts; no raw clipping',complete_P1=not missing))
    reference_paths={}
    for cid,c in conditions.items():
        uid=c['uid'];p=assets/('GT__'+cid+'.png');_render(p,c['C'],c['GT_free'],frames[uid]);reference_paths[cid]=p
    for r in selected:
        rid=r['request_id']
        if rid not in meshes:continue
        c=conditions[r['condition_id']];row=joined[rid];free=meshes[rid]
        geometry=assets/(rid+'__geometry.png');overlay=assets/(rid+'__overlays.png')
        _render(geometry,c['C'],free,frames[r['uid']])
        vp=resolve(row['control_vectors_path']);rp=resolve(row['control_reference_path'])
        if digest(vp)!=row['control_vectors_sha256'] or digest(rp)!=row['control_reference_sha256']:
            raise ValueError('Control overlay source changed')
        _overlay(overlay,c['C'],free,frames[r['uid']],_arrays(rp),_arrays(vp),row)
        records.append(dict(request_id=rid,label=r['label'],panel=r['panel'],uid=r['uid'],
            condition_id=r['condition_id'],candidate=r['candidate'],geometry_image=str(geometry),overlay_image=str(overlay),
            raw_path=row['raw_path'],raw_sha256=row['raw_sha256'],geometry_sha256=digest(geometry),overlay_sha256=digest(overlay),
            source_control_vectors_sha256=row['control_vectors_sha256'],source_reference_sha256=row['control_reference_sha256'],
            fixed_views=4,wireframe=True,geometry_repairs=0,C_bitwise=True))
    for panel,page_name in (('P1','index.html'),('P5','nested.html')):
        body=['<p class="note">Technical review only. Original triangles are displayed without repair, welding, rescaling, or simplification. C is cyan; free faces are orange. Four fixed views plus wireframe use one frame per parent across models and candidates.</p>',
            '<p>The three overlays show exact saved metric samples: uncovered Gamma and qualifying free edges; C cross/duplicate conflict; GT-reference exposure retention. Gray free faces in overlays are translucent for inspection only. NA and uncertain samples remain explicit.</p>',
            '<nav><a href="index.html">P1 all 160 outputs</a><a href="nested.html">P5 nested contexts</a><a href="../per_output_metrics.csv">Full per-output CSV</a></nav>']
        for uid in sorted({r['uid'] for r in selected if r['panel']==panel}):
            body.append('<h2>Parent '+html.escape(uid)+'</h2>')
            cids=sorted({r['condition_id'] for r in selected if r['panel']==panel and r['uid']==uid})
            for cid in cids:
                if cid in reference_paths:
                    body.append('<div class="card"><h3>GT reference: '+html.escape(cid)+'</h3><img loading="lazy" src="assets/'+html.escape(reference_paths[cid].name)+'"></div>')
                for candidate in sorted({r['candidate'] for r in selected if r['condition_id']==cid and r['panel']==panel}):
                    body.append('<h3>Candidate '+str(candidate)+'</h3>')
                    for label in LABELS:
                        rr=next((r for r in selected if r['label']==label and r['condition_id']==cid and r['candidate']==candidate and r['panel']==panel),None)
                        rec=next((r for r in records if rr and r['request_id']==rr['request_id']),None)
                        body.append('<div class="card"><h3>'+html.escape(LABEL_TEXT[label])+'</h3>')
                        if rec:
                            for name in ('geometry_image','overlay_image'):body.append('<img loading="lazy" src="assets/'+html.escape(Path(rec[name]).name)+'">')
                        else:body.append('<p class="missing">BLOCKED / result not yet available</p>')
                        body.append('</div>')
        _page(gallery/page_name,'Chair boundary attachment | '+panel,body)
    for p in assets.glob('*.png'):
        with Image.open(p) as im:im.verify()
    p1_count=sum(r['panel']=='P1' for r in records);p5_count=sum(r['panel']=='P5' for r in records)
    summary=dict(status='COMPLETE' if p1_count==160 else 'INCOMPLETE',expected_P1=160,rendered_P1=p1_count,
        rendered_P5=p5_count,generated_images=2*len(records)+len(reference_paths),records=records,
        four_fixed_views_plus_wireframe=True,three_overlay_panels_per_output=True,geometry_repairs=0,model_forwards=0,
        human_ratings='NOT_RUN',AI_PROXY='NOT_PERFORMED_BY_GALLERY',source_sha256=digest(__file__),
        frame_adjusted_parents=[uid for uid,f in frames.items() if f['adjusted_for_outlier']],
        seconds=time.perf_counter()-tick,paths=dict(main=str(gallery/'index.html'),nested=str(gallery/'nested.html')))
    atomic_json(gallery/'summary.json',summary)
    return {k:v for k,v in summary.items() if k!='records'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();print(json.dumps(run(a.out,allow_partial=a.allow_partial)))
