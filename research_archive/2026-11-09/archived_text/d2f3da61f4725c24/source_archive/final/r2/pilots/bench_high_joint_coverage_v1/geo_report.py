"""Read-only metrics/slot ledger and actual-review gallery for GEO_ONLY pilot."""
from __future__ import annotations
import argparse,collections,io,json,sys,time
import numpy as np
from pathlib import Path
from . import geo_only as C
OUT=C.OUT
STAGES=('RAW','ORIGINAL_D0_FINAL','GEO_BASE','GEO_FINAL')
AXES={'overall_shape':['REASONABLE','INCOMPLETE','UNCERTAIN'],
      'C_use':['PLAUSIBLE','REPLACED_OR_IGNORED','UNCERTAIN'],
      'support_retention':['PRESENT','MISSING_OR_DAMAGED','UNCERTAIN'],
      'visible_fragment_or_gap':['YES','NO_OBVIOUS','UNCERTAIN'],
      'layering_or_fold':['YES','NO_OBVIOUS','UNCERTAIN']}

def meshpaths(t):
    root=C.resolve(t['output_dir'])
    return dict(RAW=C.resolve(t['paths']['raw']),ORIGINAL_D0_FINAL=C.resolve(t['paths']['old_final']),
        GEO_BASE=root/'geometry/post.npz',GEO_FINAL=root/'cascade/CLEAN_FINAL_POST/post.npz')

def lineage(mesh,n):
    ids=mesh.get('parent_source_raw_slot',mesh.get('raw_slot_ids',mesh.get('source_face_ids')))
    corners=mesh.get('source_corner_ids',mesh.get('base_source_corner_ids'))
    if ids is None or corners is None:raise ValueError('Missing exact slot/corner lineage')
    tri=mesh['output'];out=np.full((n,3,3),np.nan,np.float64);alive=np.zeros(n,bool)
    for i,s in enumerate(ids):
        if int(s)<0:continue
        if i>=len(corners):raise ValueError('Missing original corner lineage')
        if alive[int(s)]:raise ValueError('Repeated source slot')
        out[int(s),corners[i]]=tri[i];alive[int(s)]=True
    return out,alive

def collect():
    m=C.prepare();comp=C.read(OUT/'compute_complete.json');C.verify(comp['files'])
    done=OUT/'analysis/complete.json'
    if done.exists():r=C.read(done);C.verify(r['files']);return r
    metrics=m['metric_fields'];rows=[];slots=[];corners=[];areas=[];deletions=[];references={}
    for t in m['tasks']:
        q=t['request'];rid=q['request_id'];n,k=int(q['N']),int(q['K']);root=C.resolve(t['output_dir'])
        raw=C.arrays(t['paths']['raw'])['raw'];rawarea=C.BC.geometry_quantities(raw)[2]
        for stage,oldstage in [('RAW','RAW'),('ORIGINAL_D0_BASE','BASE_POST'),('ORIGINAL_D0_FINAL','CLEAN_FINAL_POST')]:
            r,score,folder=C.EV.authenticate_score('B',q,oldstage)
            row=dict(r,stage=stage,label='B_NATURAL_HIGH',request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],origin='READ_ONLY_REUSED')
            row['primary_I']=None;row['primary_I_status']='LOW_SUPPORT';rows.append(row)
            for p in (folder/'row.json',folder/'score.json'):references[str(p)]=C.sha(p)
        for stage,folder in [('GEO_BASE',root/'cascade/BASE_POST/scores'),('GEO_FINAL',root/'cascade/CLEAN_FINAL_POST/scores')]:
            r=C.read(folder/'row.json');score=C.read(folder/'score.json')
            assert all(f in r for f in metrics)
            assert r['ConflictCov_C_interior_samples']==0
            rows.append(dict(r,stage=stage,label='B_NATURAL_HIGH',request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],
                origin='NEW_ORIGINAL_SCORER',primary_I=None,primary_I_status='LOW_SUPPORT'))
        newcan=C.arrays(root/'geometry/post_canonical.npz');oldcan=C.arrays(t['paths']['old_canonical'])
        oldpre=C.arrays(t['paths']['old_pre'])['output'];newmerged=C.BC.merged(raw,newcan);oldmerged=C.BC.merged(oldpre,oldcan)
        for chain,can in [('ORIGINAL_D0',oldcan),('GEO_ONLY',newcan)]:
            count=collections.Counter(str(x) for x in can['source_face_reason'][k:][~can['source_face_kept'][k:]])
            for reason,num in count.items():deletions.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],chain=chain,reason=reason,count=num))
        stages=[('RAW',raw,np.ones(n,bool)),('ORIGINAL_PRE_GEO',oldpre,np.ones(n,bool)),
            ('ORIGINAL_MERGED_BEFORE_CLEANUP',oldmerged,np.ones(n,bool)),('GEO_MERGED_BEFORE_CLEANUP',newmerged,np.ones(n,bool))]
        for name,path in meshpaths(t).items():
            if name=='RAW':continue
            tri,alive=lineage(C.arrays(path),n);stages.append((name,tri,alive))
        for stage,tri,alive in stages:
            area=np.zeros(n);area[alive]=C.BC.geometry_quantities(tri[alive])[2]
            disp=np.linalg.norm(tri-raw,axis=-1)
            areas.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],stage=stage,
                original_free_faces=n-k,surviving_original_free_faces=int(alive[k:].sum()),deleted_original_free_faces=int((~alive[k:]).sum()),
                summed_original_free_triangle_area=float(area[k:].sum()),original_RAW_free_area=float(rawarea[k:].sum()),
                free_corner_max_displacement_model=float(np.nanmax(disp[k:])),free_corner_mean_displacement_model=float(np.nanmean(disp[k:])),
                area_interpretation='Triangle multiplicity counted; not semantic completeness; appended bridges excluded'))
            can=oldcan if stage.startswith('ORIGINAL') else newcan
            for s in range(k,n):
                slots.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],stage=stage,raw_slot_id=s,
                    survives=bool(alive[s]),raw_area=float(rawarea[s]),current_area=float(area[s]),
                    GEO_cleanup_reason=str(can['source_face_reason'][s]) if not alive[s] else 'KEPT',
                    maximum_corner_displacement_model=float(disp[s].max()) if alive[s] else None))
                for c in range(3):
                    corners.append(dict(request_id=rid,stage=stage,raw_slot_id=s,raw_corner_id=c,survives=bool(alive[s]),
                        raw_x=float(raw[s,c,0]),raw_y=float(raw[s,c,1]),raw_z=float(raw[s,c,2]),
                        current_x=float(tri[s,c,0]) if alive[s] else None,current_y=float(tri[s,c,1]) if alive[s] else None,
                        current_z=float(tri[s,c,2]) if alive[s] else None,displacement_model=float(disp[s,c]) if alive[s] else None))
    tables=dict(full177_per_output=rows,slot_stage_ledger=slots,corner_displacement_ledger=corners,area_stages=areas,deletion_reasons=deletions)
    paired=[];by={(r['request_id'],r['stage']):r for r in rows}
    for t in m['tasks']:
        q=t['request'];rid=q['request_id'];a=by[rid,'ORIGINAL_D0_FINAL'];b=by[rid,'GEO_FINAL'];r=dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],primary_I=None,primary_I_status='LOW_SUPPORT')
        for key in metrics:
            av,bv=a[key],b[key];r[key+'__original']=av;r[key+'__geo']=bv
            r[key+'__delta']=bv-av if isinstance(av,(int,float)) and not isinstance(av,bool) and isinstance(bv,(int,float)) else None
        paired.append(r)
    tables['paired_full177']=paired
    for name,data in tables.items():C.csvwrite(OUT/'analysis'/(name+'.csv'),data)
    C.write(OUT/'analysis/metric_rows.json',rows)
    C.write(done,dict(status='COMPLETE',rows=len(rows),new_scored_rows=8,reused_rows=12,metric_fields=metrics,
        model_calls=0,scoring_calls=0,source_rows=references,
        files={str(OUT/'analysis'/(name+'.csv')):C.sha(OUT/'analysis'/(name+'.csv')) for name in tables},
        interpretation='I primary NA; numeric legacy raw values untouched. Areas count multiplicity, not missing-body certification.'))
    return C.read(done)

def render():
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    m=C.prepare();done=OUT/'gallery/complete.json'
    if done.exists():r=C.read(done);C.verify(r['files']);return r
    tick=time.perf_counter();out=[]
    for t in m['tasks']:
        q=t['request'];rid=q['request_id'];png=OUT/'gallery'/(rid+'.png');receipt=png.with_suffix('.json')
        if receipt.exists():r=C.read(receipt);assert C.sha(png)==r['image_sha256'];out.append(r);continue
        if png.exists():raise RuntimeError('Partial image requires review')
        paths=meshpaths(t);bounds=q['gallery_bounds'];depth=q['gallery_depth_bounds'];K=int(q['K'])
        inp=C.arrays(q['inference_npz']);original=inp['C_vertices'][inp['C_faces_original']]
        plt=G.H._mpl();fig=plt.figure(figsize=(22,14),dpi=120,facecolor='white')
        grid=fig.add_gridspec(4,4,left=.025,right=.99,bottom=.04,top=.91,wspace=.15,hspace=.24);facts=[]
        try:
            for col,stage in enumerate(STAGES):
                tri,role=G.roles(C.arrays(paths[stage]),K);g=dict(C=tri[:K],F=tri[K:],C_original=original,tri=tri)
                views=[]
                for row,az in enumerate((-60,120)):
                    raster,scope=G.role_raster(tri,role,K,G.D.basis(az,23),np.asarray(depth['center']),depth['span'])
                    ax=fig.add_subplot(grid[row,col]);ax.imshow(raster);ax.set_axis_off();ax.set_title(stage+' | '+str(az),fontsize=10);views.append(scope)
                for row,view in ((2,'front'),(3,'side')):
                    ax=fig.add_subplot(grid[row,col]);G.H.draw_section(ax,g,np.flatnonzero(role[K:]=='SEAM'),bounds,G.C.L,view,1)
                    pos=bounds['front_positions' if view=='front' else 'side_positions'][1]
                    ax.set_title(stage+' | '+view+f'={pos:.5f}',fontsize=9)
                facts.append(dict(stage=stage,path=str(paths[stage]),sha256=C.sha(paths[stage]),views=views))
            fig.suptitle(rid+f'\nHIGH_TRAIN NATURAL_HIGH | N={q["N"]} K={q["K"]} | fixed views/sections | blue C, gray free\nRAW / original D0 FINAL / GEO(RAW) BASE / GEO(RAW) FINAL',fontsize=13)
            fig.text(.5,.012,'Nonblind AI_PROXY actual review recorded separately. No source-style success criterion. I is LOW_SUPPORT.',ha='center',fontsize=10)
            buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120);png.parent.mkdir(parents=True,exist_ok=True)
            with png.open('xb') as f:f.write(buf.getvalue())
        finally:plt.close(fig)
        r=dict(status='PENDING_ACTUAL_VIEW',request_id=rid,path=str(png),image_sha256=C.sha(png),cells=facts,
            original_camera=depth,original_sections=bounds,model_calls=0,scoring_calls=0)
        C.write(receipt,r);out.append(r)
    C.write(OUT/'gallery/review_protocol.json',dict(axes=AXES,reviewer='backend_runtime',reviewer_kind='AI_PROXY',blind=False,
        required='Actually inspect both views and front/side original sections; distinct five axes with concrete visible notes; uncertainty allowed',
        cells=16,requests=4,stages=STAGES))
    C.textnew(OUT/'gallery/index.html','<meta charset="utf-8"><h1>Four fixed high-N GEO(RAW) comparisons</h1>'+''.join('<h2>'+r['request_id']+'</h2><img style="width:100%" src="'+Path(r['path']).name+'">' for r in out))
    r=dict(status='COMPLETE',pages=4,cells=16,seconds=time.perf_counter()-tick,model_calls=0,scoring_calls=0,
        files={str(Path(r['path'])):r['image_sha256'] for r in out})
    C.write(done,r);return r

def report():
    m=C.prepare();collect();v=C.read(OUT/'visual_review.json');rows=C.read(OUT/'analysis/metric_rows.json')
    assert len(v['rows'])==16 and all(r['actually_viewed'] and r['views_and_sections_checked'] for r in v['rows'])
    metric=[('FF','FF_illegal_affected_free_fraction',100),('CF','CF_illegal_affected_free_fraction',100),
        ('free faces','actual_free_faces',1),('legal CF edges','C_F_valid_edge_count',1),('small-angle %','free_minimum_angle_lt5_fraction',100),
        ('isolated 1face','hinge_diag_isolated_1face_edge_components',1),('nonmanifold','nonmanifold_edges',1)]
    lines=['# High-N RAW → GEO_ONLY backend pilot','',
        'Four fixed NATURAL_HIGH HIGH_TRAIN requests; no new model calls, D0, noise, queries or training. This is a backend comparison on two familiar high-N parents, not generalization.',
        '', '| Stage | '+' | '.join(x[0] for x in metric)+' | primary I |','|---|'+'---:|'*(len(metric)+1)]
    for stage in ('RAW','ORIGINAL_D0_BASE','ORIGINAL_D0_FINAL','GEO_BASE','GEO_FINAL'):
        rr=[r for r in rows if r['stage']==stage]
        lines.append('| '+stage+' | '+' | '.join(f'{np.mean([r[key] for r in rr])*scale:.4f}' for _,key,scale in metric)+' | NA: LOW_SUPPORT |')
    lines+=['','Both fixed C query sets have zero interior samples (required at least 32). Raw legacy I numbers are retained in full177 tables; they are not an applicable primary comparison. The four-request mean equals parent-equal mean here because each parent has two noises.',
        '', '| Request | original FINAL free / FF% / CF% | GEO FINAL free / FF% / CF% |', '|---|---:|---:|']
    by={(r['request_id'],r['stage']):r for r in rows}
    for t in m['tasks']:
        q=t['request'];a=by[q['request_id'],'ORIGINAL_D0_FINAL'];b=by[q['request_id'],'GEO_FINAL']
        fmt=lambda r:f"{r['actual_free_faces']} / {100*r['FF_illegal_affected_free_fraction']:.3f} / {100*r['CF_illegal_affected_free_fraction']:.3f}"
        lines.append('| '+q['parent_id'][:6]+'/n'+str(q['noise_index'])+' | '+fmt(a)+' | '+fmt(b)+' |')
    lines+=['', '## Actual visual observations','',v['summary'],'',
        'The labels are nonblind AI_PROXY, not independent human ratings. The comparison is not a unique-source-style reconstruction task.',
        '', '## Accounting and limits','',
        'The slot and corner ledgers preserve original RAW identity across pre-GEO, merged-before-cleanup, BASE and FINAL. Deleted slots remain explicit. Areas sum triangles including overlapping/double-sided multiplicity; neither lower area nor fewer faces alone establishes missing semantic structure.',
        '', 'The original C geometry and vertex identities are preserved. The downstream guard ledgers separately check existing legal edges, newly illegal pairs, deletions, nonmanifold and same-winding additions. D0-free shape retention does not exempt GEO_ONLY from geometric quality costs.',
        '', '[Full 177 paired metrics](analysis/paired_full177.csv) · [Area stages](analysis/area_stages.csv) · [Deleted reasons](analysis/deletion_reasons.csv) · [Slot ledger](analysis/slot_stage_ledger.csv) · [Gallery](gallery/index.html)',
        '', '## Resume','',
        '`python -B -m r2.pilots.bench_high_joint_coverage_v1.geo_only verify` verifies completed science; no replay. `geo_report collect/render` reuse sealed outputs. Unresolved partial directories require explicit engineering review. No default switch, cleanup, commit or push.']
    C.textnew(OUT/'REPORT.md','\n'.join(lines)+'\n')
    C.verify(m['protected']);C.verify(m['code'])
    files={str(p):C.sha(p) for p in (OUT/'manifest.json',OUT/'compute_complete.json',OUT/'analysis/complete.json',OUT/'gallery/complete.json',OUT/'visual_review.json',OUT/'REPORT.md',Path(__file__))}
    C.write(OUT/'COMPLETE.json',dict(status='COMPLETE',science=C.read(OUT/'compute_complete.json'),actual_review_cells=16,
        reviewer_kind='AI_PROXY',blind=False,protected_inputs_unchanged=True,model_calls=0,D0_calls=0,files=files))
    return dict(status='COMPLETE',report=str(OUT/'REPORT.md'),actual_review_cells=16)

def detail():
    """Higher pixel resolution at the exact same camera; shared crop across arms."""
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    m=C.prepare();done=OUT/'gallery/detail_complete.json'
    if done.exists():r=C.read(done);C.verify(r['files']);return r
    files={}
    for t in m['tasks']:
        q=t['request'];rid=q['request_id'];p=OUT/'gallery'/(rid+'__detail.png');rec=p.with_suffix('.json')
        if rec.exists():
            r=C.read(rec);assert C.sha(p)==r['image_sha256'];files[str(p)]=C.sha(p);continue
        if p.exists():raise RuntimeError('Unsealed detail image')
        paths=meshpaths(t);depth=q['gallery_depth_bounds'];imgs=[];crops=[]
        for az in (-60,120):
            rr=[];boxes=[]
            for stage in STAGES:
                tri,role=G.roles(C.arrays(paths[stage]),int(q['K']))
                im,owner=G.D.raster(tri,int(q['K']),G.D.basis(az,23),np.asarray(depth['center']),depth['span'],size=1600)
                rr.append(np.asarray(im));ys,xs=np.nonzero(owner>=0);boxes.append((int(ys.min()),int(ys.max()+1),int(xs.min()),int(xs.max()+1)))
            box=(max(0,min(b[0] for b in boxes)-25),min(1600,max(b[1] for b in boxes)+25),max(0,min(b[2] for b in boxes)-25),min(1600,max(b[3] for b in boxes)+25))
            crops.append(dict(azimuth=az,pixel_box=box));imgs.append([im[box[0]:box[1],box[2]:box[3]] for im in rr])
        plt=G.H._mpl();fig,ax=plt.subplots(2,4,figsize=(24,8),dpi=120,facecolor='white')
        try:
            for i,az in enumerate((-60,120)):
                for j,stage in enumerate(STAGES):ax[i,j].imshow(imgs[i][j]);ax[i,j].set_axis_off();ax[i,j].set_title(stage+' | '+str(az))
            fig.suptitle(rid+'\nExact original camera at 1600px; same union crop for all four stages; supplemental visibility only',fontsize=12)
            fig.tight_layout(rect=(0,0,1,.92));buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120)
            with p.open('xb') as f:f.write(buf.getvalue())
        finally:plt.close(fig)
        r=dict(status='PENDING_ACTUAL_VIEW',request_id=rid,path=str(p),image_sha256=C.sha(p),resolution=1600,
            camera=depth,crops=crops,geometry_changed=False,model_calls=0,scoring_calls=0,
            sources={str(v):C.sha(v) for v in paths.values()})
        C.write(rec,r);files[str(p)]=C.sha(p)
    C.write(done,dict(status='COMPLETE',files=files,pages=4,model_calls=0,scoring_calls=0))
    return dict(status='COMPLETE',pages=4,model_calls=0,scoring_calls=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('collect','render','detail','report'));a=p.parse_args();r=globals()[a.command]()
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','source_rows','metric_fields')},ensure_ascii=False))
