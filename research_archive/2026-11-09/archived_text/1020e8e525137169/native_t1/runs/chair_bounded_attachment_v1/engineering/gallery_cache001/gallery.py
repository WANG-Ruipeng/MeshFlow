"""Full fixed 16-row offline gallery; no model, projection, or scoring calls."""
from pathlib import Path
import argparse
import html
import json
import os
import time
import numpy as np
from .common import OUT,LABELS,read_json,digest,atomic_json,resolve,arrays
from experiments.mf_h1_local0.chair_control_usability_quick_v1.gallery import _render,_frame,_display

VERSION='chair_bounded_attachment_gallery_v1'
COMPLETE_STATUSES=('CACHED','PROJECTED','SAMPLED')


def _load_logs(result):
    items=[]
    for key in ('intervention_log_path','project_log_path'):
        if result.get(key):items.append(result[key])
    for key in ('intervention_log_paths','project_log_paths'):
        items.extend(result.get(key,[]))
    if not items and result.get('log_path'):items.append(result['log_path'])
    records=[]
    for value in dict.fromkeys(str(x) for x in items):
        path=resolve(value);log=read_json(path)
        expected=result.get('intervention_log_sha256') if len(items)==1 else None
        if expected and digest(path)!=expected:raise ValueError('Intervention log identity changed')
        nested=log.get('interventions',log.get('steps'))
        if isinstance(nested,list):
            for row in nested:
                content=row.get('intervention_log',row.get('log',row))
                records.append(dict(step=row.get('step',row.get('k')),log=content))
        else:records.append(dict(step=None,log=log))
    # The trajectory writer may store its intervention records directly.
    for row in result.get('interventions',[]):
        if isinstance(row,dict) and ('log' in row or 'intervention_log' in row):
            records.append(dict(step=row.get('step',row.get('k')),log=row.get('log',row.get('intervention_log'))))
    return records


def _overlay(path,C,base,free,frame,logs,label):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    displacement=free.astype(np.float64)-base.astype(np.float64)
    moved=np.any(free!=base,axis=2)
    segments=np.stack((base[moved],free[moved]),axis=1)
    constraints=[]
    for item in logs:
        for constraint in item['log'].get('accepted_constraints',[]):
            constraints.append((item['step'],constraint))
    fig=plt.figure(figsize=(11,4.7),dpi=130,facecolor='#f6f7f8')
    center=_display(np.asarray(frame['center']));half=frame['half_extent']
    for i,title in enumerate(('Actual final displacement versus RAW','Accepted endpoint constraints')):
        ax=fig.add_subplot(1,2,i+1,projection='3d');ax.set_proj_type('ortho')
        ax.add_collection3d(Poly3DCollection(_display(free),facecolors='#b6bdc5',edgecolors='none',alpha=.16))
        ax.add_collection3d(Line3DCollection(_display(C[:,[[0,1],[1,2],[2,0]],:].reshape(-1,2,3)),colors='#218ba5',linewidths=.7,alpha=.6))
        handles=[Line2D([0],[0],color='#218ba5',lw=1,label='Unchanged known C')]
        if i==0:
            if moved.any():
                ax.add_collection3d(Line3DCollection(_display(segments),colors='#bd268e',linewidths=1.8))
                points=_display(free[moved]);ax.scatter(points[:,0],points[:,1],points[:,2],s=10,c='#bd268e',depthshade=False)
            handles.append(Line2D([0],[0],marker='o',color='#bd268e',lw=1,label='Moved corners; displacement not amplified'))
            suffix=f'{int(moved.sum())}/{moved.size} corners; max={np.linalg.norm(displacement,axis=2).max(initial=0):.6g}'
        else:
            if constraints:
                anchors=np.asarray([c['known_edge'] for _,c in constraints],np.float64)
                ax.add_collection3d(Line3DCollection(_display(anchors),colors='#15833c',linewidths=2.5))
                start=np.asarray([c['free_edge'] for _,c in constraints],np.float64)
                ax.add_collection3d(Line3DCollection(_display(start),colors='#c75b18',linewidths=1.2,linestyles='dashed'))
            handles.extend([Line2D([0],[0],color='#15833c',lw=2,label='Accepted known edge target'),
                Line2D([0],[0],color='#c75b18',lw=1,linestyle='--',label='Free edge before that intervention')])
            suffix=f'{len(constraints)} accepted constraints'
            if label=='TRAJECTORY':suffix+='; intermediate acceptance is not a final guarantee'
        ax.set_title(title+'\n'+suffix,fontsize=9)
        ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half);ax.set_zlim(center[2]-half,center[2]+half)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=45);ax.set_axis_off()
        ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.15),fontsize=7,frameon=False)
    fig.subplots_adjust(left=.005,right=.995,top=.88,bottom=.14,wspace=.02)
    tmp=p.with_suffix('.tmp.png');fig.savefig(tmp,metadata={'Software':VERSION});plt.close(fig);os.replace(tmp,p)
    return dict(moved_corner_count=int(moved.sum()),total_free_corners=int(moved.size),
        accepted_edges_drawn=len(constraints),actual_displacement_not_amplified=True)


def run(out=OUT):
    from PIL import Image
    out=Path(out);tick=time.perf_counter();bases=read_json(out/'base_requests.json')
    requests=read_json(out/'requests.json');assert len(bases)==16 and len(requests)==48
    conditions={};meshes={};results={};pending=[]
    for base in bases:
        path=resolve(base['condition_npz'])
        assert digest(path)==base['condition_sha256']
        conditions[base['condition_id']]=arrays(path)
    for row in requests:
        path=out/'outputs'/row['request_id']/'result.json'
        result=read_json(path) if path.exists() else dict(status='NOT_RUN')
        results[row['request_id']]=result
        if result['status'] not in COMPLETE_STATUSES:
            if not result['status'].startswith('NOT_RUN_NO_PHASE1_SIGNAL'):pending.append(row['request_id'])
            continue
        p=resolve(result['raw_path']);assert digest(p)==result['raw_sha256']
        raw=arrays(p)['raw'];C=conditions[row['condition_id']]['C'];n,k=int(row['N']),int(row['K'])
        assert raw.dtype==np.float32 and raw.shape==(n,9) and np.isfinite(raw).all()
        assert raw[:k].tobytes()==C.reshape(k,9).tobytes()
        meshes[row['request_id']]=raw[k:].reshape(-1,3,3)
    frames={}
    for uid in sorted({r['uid'] for r in bases}):
        group=[r for r in requests if r['uid']==uid and r['request_id'] in meshes]
        cs=[conditions[r['condition_id']]['GT_full'] for r in bases if r['uid']==uid]
        frames[uid]=_frame(cs,[meshes[r['request_id']] for r in group])
        center=np.asarray(frames[uid]['center']);half=frames[uid]['half_extent']
        assert all(np.max(np.abs(meshes[r['request_id']]-center))<=half for r in group)
    gallery=out/'gallery';assets=gallery/'assets';assets.mkdir(parents=True,exist_ok=True)
    atomic_json(gallery/'frames.json',dict(frames=frames,
        policy='One union frame per parent across two noises, every available label and GT; no clipping, normalization or repairs',
        includes_every_raw_corner=True))
    records=[];by_id={};references={}
    for base in bases:
        cid=base['condition_id'];c=conditions[cid]
        if cid not in references:
            p=assets/('REFERENCE__'+cid+'.png');_render(p,c['C'],c['GT_free'],frames[base['uid']]);references[cid]=p
    for row in requests:
        rid=row['request_id'];result=results[rid]
        rec=dict(request_id=rid,base_id=row['base_id'],label=row['label'],uid=row['uid'],candidate=row['candidate'],status=result['status'])
        if rid in meshes:
            C=conditions[row['condition_id']]['C'];F=meshes[rid]
            geometry=assets/(rid+'__geometry.png');overlay=assets/(rid+'__intervention.png')
            _render(geometry,C,F,frames[row['uid']])
            raw=meshes['RAW__'+row['base_id']]
            observation=_overlay(overlay,C,raw,F,frames[row['uid']],_load_logs(result),row['label'])
            rec.update(geometry_image=str(geometry),geometry_sha256=digest(geometry),overlay_image=str(overlay),
                overlay_sha256=digest(overlay),raw_path=result['raw_path'],raw_sha256=result['raw_sha256'],
                correction_status=result.get('correction_status'),C_bitwise=True,four_fixed_views=True,wireframe=True,**observation)
        records.append(rec);by_id[rid]=rec
    body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Bounded attachment | all 16 fixed comparisons</title><style>body{margin:0;font:15px/1.5 system-ui;background:#edf1f4;color:#202c36}main{padding:20px;max-width:2400px;margin:auto}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border:1px solid #bcc7d0;background:white;padding:8px;vertical-align:top}img{width:100%;height:auto;display:block}th{position:sticky;top:0;z-index:1;background:#e7eef3}.note{padding:14px;background:#fff5d9}.missing{padding:60px 10px;color:#923e27;background:#fff0e8;overflow-wrap:anywhere}.small{font-size:12px;color:#536573}h2{overflow-wrap:anywhere}a{color:#176084}summary{cursor:pointer}</style><main>',
        '<h1>Bounded attachment: all 16 fixed outputs</h1>',
        '<p class="note">固定 8 父对象 × 两个原噪声，完整展示无变化和拒绝例。每父所有方法与噪声共用相机、框和光照，异常点纳入范围。C 为青色，自由面为橙色；四视角与线框均为原三角形，没有平滑、修补、合并或裁剪。点击图片查看原尺寸。</p>',
        '<p>紫色显示最终自由角相对 RAW 的真实移动（未放大）；绿色显示被接受的 C 边锚点，橙色虚线为该次干预前的自由边。TRAJECTORY 中间接受不保证终点仍贴合。图片仅作技术检查；human ratings = NOT_RUN，AI_PROXY 不等于人评。</p>',
        '<p><a href="../BOUNDED_ATTACHMENT_CORRECTION_REPORT.md">报告</a> · <a href="../per_output_metrics.csv">全部逐例指标</a> · <a href="../phase1_gate.json">Phase 1 gate</a> · <a href="frames.json">统一框审计</a></p>']
    prior_uid=None
    for base in bases:
        uid=base['uid'];cid=base['condition_id']
        if uid!=prior_uid:
            body.append('<h2>Parent '+html.escape(uid)+'</h2><details><summary>C + GT free technical reference</summary><a target="_blank" href="assets/'+references[cid].name+'"><img loading="lazy" src="assets/'+references[cid].name+'"></a></details>');prior_uid=uid
        body.append('<h3>Original noise '+str(base['candidate'])+'</h3><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in LABELS)+'</tr></thead><tbody><tr>')
        for label in LABELS:
            rec=by_id[label+'__'+base['base_id']];body.append('<td>')
            if 'geometry_image' not in rec:
                body.append('<div class="missing">'+html.escape(rec['status'])+'</div>')
            else:
                body.append('<p class="small">'+html.escape(str(rec.get('correction_status') or rec['status']))+' · moved corners '+str(rec['moved_corner_count'])+' · accepted constraints '+str(rec['accepted_edges_drawn'])+'</p>')
                for key in ('geometry_image','overlay_image'):
                    name=html.escape(Path(rec[key]).name);body.append('<a target="_blank" href="assets/'+name+'"><img loading="lazy" src="assets/'+name+'"></a>')
            body.append('</td>')
        body.append('</tr></tbody></table>')
    body.append('</main></html>');(gallery/'index.html').write_text('\n'.join(body),encoding='utf-8')
    images=[Path(r[k]) for r in records for k in ('geometry_image','overlay_image') if k in r]+list(references.values())
    for p in images:
        with Image.open(p) as im:im.verify()
    summary=dict(status='COMPLETE' if not pending else 'INCOMPLETE',expected_rows=16,actual_rows=len(bases),
        expected_columns=list(LABELS),registered_cells=48,rendered_cells=len(meshes),
        explicit_not_run_cells=sum(r['status'].startswith('NOT_RUN_NO_PHASE1_SIGNAL') for r in records),
        pending_cells=pending,generated_images=len(images),reference_images=len(references),
        counts_by_label={label:sum(r['label']==label and 'geometry_image' in r for r in records) for label in LABELS},
        frame_adjusted_parents=[uid for uid,f in frames.items() if f['adjusted_for_outlier']],
        frames_include_all_available_corners=True,geometry_repairs=0,model_calls=0,projection_calls=0,scoring_calls=0,
        human_ratings='NOT_RUN',AI_PROXY='NOT_PERFORMED_BY_GALLERY',records=records,
        seconds=time.perf_counter()-tick,source_sha256=digest(__file__),page=str(gallery/'index.html'))
    atomic_json(gallery/'summary.json',summary)
    return {k:v for k,v in summary.items() if k!='records'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));a=p.parse_args()
    print(json.dumps(run(a.out),ensure_ascii=False,indent=2))