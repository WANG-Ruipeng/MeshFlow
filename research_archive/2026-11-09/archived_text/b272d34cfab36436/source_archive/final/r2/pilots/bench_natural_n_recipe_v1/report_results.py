"""CPU-only frozen-frame gallery and read-only full-metric analysis.

Ratings are accepted only as explicit actual-view records. This module never
creates a rating from a score, mesh statistic, historical label or filename.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,collections,concurrent.futures,csv,html,io,json,multiprocessing,resource,statistics,sys,time
from pathlib import Path
from . import evaluation as E
from r2.pilots.bench_full_seat_curriculum_v1 import summarize_results as S
from r2.pilots.bench_seat_coverage_v1.report_tables import CORE,unit

OUT,ROOT=E.OUT,E.ROOT
GROUPS=('OLD','FOCUS','DEV','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT')

def groups():return tuple(g for g in GROUPS if any(q['report_group']==g for q in E.requests()))
CONTRASTS=(('P0','A'),('P0','B'),('A','B'))
AXES={
 'overall_shape':('REASONABLE','INCOMPLETE','UNCERTAIN'),
 'C_use':('PLAUSIBLE','REPLACED_OR_IGNORED','UNCERTAIN'),
 'support_retention':('PRESENT','MISSING_OR_DAMAGED','UNCERTAIN'),
 'visible_fragment_or_gap':('YES','NO_OBVIOUS','UNCERTAIN'),
 'layering_or_fold':('YES','NO_OBVIOUS','UNCERTAIN'),
}
VERSION='bench_natural_n_recipe_report_v1'
GALLERY=OUT/'gallery'
ANALYSIS=OUT/'results/analysis_v1'


def text_file(path,value):
    path=E.resolve(path);raw=value.encode('utf-8')
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Immutable text differs: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(raw)


def cell(q,label,stage):
    root=E.root_for(label,q);stored=E.storage_label(label,q);rid=q['request_id']
    if stage=='CLEAN_FINAL_POST':
        folder=root/'cpu'/stored/rid;mesh=folder/stage/'post.npz';receipt=folder/'complete.json'
    else:
        kind='raw' if stage=='RAW' else 'post';folder=root/'evaluation'/stored/rid/kind
        mesh=folder/(kind+'.npz');receipt=folder/'receipt.json'
    return dict(model=label,label=label+' '+stage,stage=stage,path=str(mesh),receipt=str(receipt),request_id=rid)


def prepare():
    """Freeze displays and empty review schema before new outputs exist."""
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    E.prepare_panel();pages=[]
    for q in E.requests():
        for stage in E.STAGES:
            pages.append(dict(**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','N','K','split','part_type','report_group')},
                sheet=stage,page_id=stage+'__'+q['request_id'],cells=[cell(q,m,stage) for m in E.LABELS],
                inference_npz=q['inference_npz'],bounds=q['gallery_bounds'],depth_bounds=q['gallery_depth_bounds']))
    protected={str(p):E.sha(p) for p in (Path(__file__),E.MANIFEST,Path(G.__file__),Path(G.D.__file__),Path(G.H.__file__))}
    manifest=dict(status='FROZEN',schema=VERSION,pages=pages,sheets=3*len(E.requests()),cells=9*len(E.requests()),requests=len(E.requests()),
        labels=list(E.LABELS),stages=list(E.STAGES),protected=protected,
        cameras='Inherited exact center/span, azimuth -60/120 elevation23; no fitting to output',
        sections='Original central front/side positions[1], same across all models/stages',
        role_colors='Blue C / gray BULK / green BRIDGE; Native all-free, no SEAM',
        reviewer_kind='AI_PROXY',blind=False,review='PENDING_ACTUAL_VIEW',GT_display=False,model_calls=0)
    E.immutable(GALLERY/'manifest.json',manifest)
    protocol=dict(schema=VERSION+'_actual_view',status='FROZEN',axes={k:list(v) for k,v in AXES.items()},labels=list(E.LABELS),cells=9*len(E.requests()),
        expected_units=[dict(request_id=q['request_id'],parent_id=q['parent_id'],condition_id=q['condition_id'],
            report_group=q['report_group'],noise_index=q['noise_index'],label=label,stage=stage,
            page_id=stage+'__'+q['request_id']) for q in E.requests() for stage in E.STAGES for label in E.LABELS],
        required_fields=['reviewer','reviewer_kind','blind','view_event_id','viewed_at_utc','actually_viewed',
            'views_and_sections_checked','image_sha256','image_receipt_sha256','gallery_definition_sha256','note',*AXES],
        instructions=['Actually open each page and check both views and both original sections.',
            'Five axes independent; uncertainty and missing observations remain explicit.',
            'No source-style target. C preservation or metric score does not establish adoption.',
            'Cross-section endpoints alone do not certify gaps; inspect full views.',
            'Do not copy historic P0 observations; record label drift separately.',
            'No all-panel visual conclusion until all registered actual observations are complete.'],
        denominator='Noise within C, C within parent, equally weighted parent; groups remain separate',
        gallery_manifest_sha256=E.sha(GALLERY/'manifest.json'))
    E.immutable(OUT/'visual_review/protocol.json',protocol)
    return dict(status='FROZEN',sheets=3*len(E.requests()),cells=9*len(E.requests()),model_calls=0,ratings_generated=0)


def one(job):
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    import numpy as np
    C,D,H=G.C,G.D,G.H;page,dh=job;png=GALLERY/(page['page_id']+'.png');receipt=png.with_suffix('.json')
    if receipt.exists():
        r=E.read(receipt)
        if r['definition_sha256']!=dh or E.sha(png)!=r['image_sha256']:raise ValueError('Saved gallery differs')
        return r
    if png.exists():raise RuntimeError('Unsealed gallery image requires explicit recovery')
    tick=time.perf_counter();plt=H._mpl();ncols=3
    fig=plt.figure(figsize=(5.3*ncols,13.5),dpi=110,facecolor='white')
    grid=fig.add_gridspec(4,ncols,left=.025,right=.99,bottom=.055,top=.91,wspace=.15,hspace=.23)
    a=E.arrays(page['inference_npz']);original=a['C_vertices'][a['C_faces_original']]
    K,bounds,depth=page['K'],page['bounds'],page['depth_bounds'];display=[]
    try:
        for col,info in enumerate(page['cells']):
            tri,role=G.roles(E.arrays(info['path']),K)
            g=dict(C=tri[:K],F=tri[K:],C_original=original,tri=tri)
            facts=dict(label=info['label'],role_counts={r:int((role==r).sum()) for r in ('C','SEAM','BULK','BRIDGE')},views=[])
            for row,az in enumerate((-60,120)):
                raster,scope=G.role_raster(tri,role,K,D.basis(az,23),np.asarray(depth['center']),depth['span'])
                ax=fig.add_subplot(grid[row,col]);ax.imshow(raster);ax.set_axis_off()
                ax.set_title(info['model']+'\n'+info['stage']+' | '+str(az),fontsize=9)
                facts['views'].append(dict(azimuth=az,**scope))
                if scope['projected_vertices_outside_inherited_frame']:
                    ax.text(.01,.01,'Inherited-frame clipping: see receipt',transform=ax.transAxes,fontsize=7,color='#b34b27')
            for row,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[row,col]);H.draw_section(ax,g,np.flatnonzero(role[K:]=='SEAM'),bounds,C.L,view,1)
                from matplotlib.collections import LineCollection
                axis,project=(2,(0,1)) if view=='front' else (0,(2,1))
                pos=bounds['front_positions' if view=='front' else 'side_positions'][1]
                ff,ids=H.finite_sections(g['F'],axis,pos,1e-12*C.L);bm=role[K:][ids]=='BRIDGE'
                if bm.any():ax.add_collection(LineCollection(ff[bm][:,:,project],colors='#41a46c',linewidths=1.6,zorder=3))
                ax.set_title(info['model']+' | '+view+f'={pos:.5f}',fontsize=8)
            display.append(facts)
        fig.suptitle(page['request_id']+' | '+page['report_group']+' '+page['part_type']+f' | K={K} N={page["N"]}\n'
            +page['sheet']+' | Blue C / gray BULK / green BRIDGE\nP0 prior B; A CONTROL_OLD; B NATURAL_HIGH. Same frozen noise, frame and sections.',fontsize=12)
        fig.text(.5,.018,'Actual nonblind AI_PROXY review recorded separately. Section endpoints do not certify cracks. No unique-source-style grading.',ha='center',fontsize=9)
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110)
        with png.open('xb') as f:f.write(buffer.getvalue())
    finally:plt.close(fig)
    r=dict(status='PENDING_ACTUAL_VIEW',page_id=page['page_id'],request_id=page['request_id'],condition_id=page['condition_id'],
        sheet=page['sheet'],report_group=page['report_group'],path=str(png),image_sha256=E.sha(png),definition_sha256=dh,
        cells=page['cells'],display=display,seconds=time.perf_counter()-tick,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,scoring_calls=0)
    E.immutable(receipt,r);return r


def render(workers=3):
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
    prepare();plan=E.read(GALLERY/'manifest.json');sources={}
    for page in plan['pages']:
        for c in page['cells']:sources.update(G.authenticate(c))
    E.immutable(GALLERY/'definition.json',dict(manifest_sha256=E.sha(GALLERY/'manifest.json'),sources=sources,protected=plan['protected']))
    done=GALLERY/'completion.json'
    if done.exists():
        r=E.read(done);E.verify(r['files']);return r
    before=G.C.resources();tick=time.perf_counter();jobs=[(p,E.sha(GALLERY/'definition.json')) for p in plan['pages']]
    first=one(jobs[0]);count=worker_count(G.C.resources(),first,workers)
    dispatch=GALLERY/('dispatch_'+str(time.time_ns())+'.json')
    E.immutable(dispatch,dict(workers=count,resources_before=before,resources_after_first=G.C.resources(),
        first_peak_RSS_KiB=first['peak_RSS_KiB'],model_calls=0,scoring_calls=0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        images=[first,*pool.map(one,jobs[1:])]
    E.verify(plan['protected']);E.verify(sources)
    content='<meta charset="utf-8"><h1>Bench natural N: P0 / CONTROL_OLD / NATURAL_HIGH</h1><p>Same frozen panel; HIGH groups separate. All original sections; nonblind AI_PROXY observations separate.</p>'
    for im in images:
        name=Path(im['path']).name;content+='<h2>'+html.escape(im['page_id'])+'</h2><a href="'+name+'"><img loading="lazy" style="width:100%" src="'+name+'"></a>'
    text_file(GALLERY/'index.html',content)
    files={str(p):E.sha(p) for im in images for p in (Path(im['path']),Path(im['path']).with_suffix('.json'))}
    files[str(GALLERY/'index.html')]=E.sha(GALLERY/'index.html')
    r=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',sheets=3*len(E.requests()),cells=9*len(E.requests()),images=images,files=files,
        workers=count,seconds=time.perf_counter()-tick,model_calls=0,scoring_calls=0,ratings_generated=0)
    E.immutable(done,r);return r


def deletion_rows(labels=E.LABELS):
    """Source-slot area accounting; no new detector and no source-GT scoring."""
    import numpy as np
    rows=[];faces=[]
    def areas(tri):return .5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    for label in labels:
        for q in E.requests():
            raw=E.arrays(E.producer(label,q,'raw')/'raw.npz');postdir=E.producer(label,q,'post')
            pre=E.arrays(postdir/'pre_geo.npz')['output'];geo=E.read(postdir/'geometry.json')
            K,N=int(q['K']),int(q['N']);ra=areas(raw['raw']);pa=areas(pre)
            if len(ra)!=N or len(pa)!=N:raise ValueError('Deletion slot domain differs')
            byid={int(r['source_face_id']):r for r in geo['face_records']}
            if set(byid)!=set(range(N)):raise ValueError('Missing GEO face fate')
            for stage in ('BASE_POST','CLEAN_FINAL_POST'):
                m=E.arrays(cell(q,label,stage)['path'])
                ids=m.get('parent_source_raw_slot',m.get('source_face_ids'))
                if ids is None or len(ids)!=len(m['faces']):raise ValueError('Final RAW-slot provenance missing')
                retained={int(i) for i in ids if K<=i<N};deleted=sorted(set(range(K,N))-retained)
                if any(int(i)>=N for i in ids):raise ValueError('Out-of-range original slot')
                if stage=='BASE_POST' and retained!={i for i,r in byid.items() if i>=K and r['kept']}:raise ValueError('GEO deletion receipt differs')
                row=dict(label=label,stage=stage,**{k:q[k] for k in ('request_id','parent_id','condition_id','noise_index','report_group')},
                    raw_free_faces=N-K,retained_original_free_faces=len(retained),deleted_original_free_faces=len(deleted),
                    added_bridge_faces=int((ids<0).sum()),raw_free_area=float(ra[K:].sum()),pre_geo_free_area=float(pa[K:].sum()),
                    deleted_raw_reference_area=float(ra[deleted].sum()),deleted_pre_geo_reference_area=float(pa[deleted].sum()),
                    retained_raw_reference_area=float(ra[sorted(retained)].sum()),actual_final_free_area=float(areas(m['output'])[K:].sum()),
                    deleted_raw_area_fraction=float(ra[deleted].sum()/ra[K:].sum()) if ra[K:].sum()>0 else None,
                    deletion_identity='generated_RAW_slot_not_GT',area_units='model_squared',model_calls=0)
                rows.append(row)
                for i in range(K,N):
                    faces.append(dict(label=label,stage=stage,request_id=q['request_id'],parent_id=q['parent_id'],report_group=q['report_group'],raw_slot=i,
                        retained=i in retained,raw_area=float(ra[i]),pre_geo_area=float(pa[i]),GEO_kept=bool(byid[i]['kept']),GEO_reason=byid[i]['reason']))
    return rows,faces


def tables():
    done=ANALYSIS/'completion.json'
    if done.exists():r=E.read(done);E.verify(r['files']);return r
    completed=E.read(OUT/'results/completion.json');E.verify(completed['files']);E.verify(completed['source_rows'])
    if (completed['rows'],completed['new_score_rows'],completed['reused_score_rows'])!=(E.budget()['total_score_rows'],E.budget()['new_score_rows'],90):raise ValueError('Wrong score universe')
    rows=E.read(OUT/'results/metric_rows.json');metrics=completed['metric_fields'];original=[];primary=[];paired=[];paired_primary=[]
    buckets={(l,s,g):[r for r in rows if (r['label'],r['stage'],r['report_group'])==(l,s,g)] for l in E.LABELS for s in E.STAGES for g in groups()}
    for (label,stage,group),subset in buckets.items():
        meta=dict(label=label,stage=stage,report_group=group)
        original.extend(dict(**meta,unit=unit(m),**S.aggregate(subset,m,'ORIGINAL_FINITE_ONLY')) for m in metrics)
        primary.extend(dict(**meta,unit=unit(m),**S.aggregate(subset,m,'I_PASS_ONLY' if m in S.PASS_ONLY_I else 'ORIGINAL_FINITE_ONLY')) for m in CORE)
    for stage in E.STAGES:
        for group in groups():
            for a,b in CONTRASTS:
                meta=dict(reference=a,candidate=b,stage=stage,report_group=group)
                paired.extend(dict(**meta,unit=unit(m),**S.paired(buckets[a,stage,group],buckets[b,stage,group],m,'ORIGINAL_FINITE_ONLY')) for m in metrics)
                paired_primary.extend(dict(**meta,unit=unit(m),**S.paired(buckets[a,stage,group],buckets[b,stage,group],m,'I_PASS_ONLY' if m in S.PASS_ONLY_I else 'ORIGINAL_FINITE_ONLY')) for m in CORE)
    retention,fates=deletion_rows()
    for name,data in dict(full177_parent_macro=original,primary_parent_macro=primary,paired177_parent_macro=paired,
        paired_primary_parent_macro=paired_primary,deletion_area_per_output=retention,deletion_fate_per_slot=fates).items():
        E.csvwrite(ANALYSIS/(name+'.csv'),data)
    lines=['# Bench natural-N recipe: complete numerical results','',
        'P0 is frozen exposure B: old30 outputs are reused; high-N outputs are newly generated. A/B each add1500 updates. Contrasts: A-P0, B-P0 and B-A.',
        '所有分数沿用原定义，父等权；I主展示只用PASS支持域，完整177字段保留原值与NA。',
        'Fractions use0-1, not percent. OLD/FOCUS/DEV/HOLDOUT/HIGH_TRAIN/HIGH_HOLDOUT remain separate; no automatic promotion.',
        '本文件不产生视觉标签。实际图审及支撑完整性必须另见 visual_review。','']
    for group in groups():
        for stage in E.STAGES:
            lines+=['## '+group+' / '+stage,'','| metric | P0 | A | B | unit |','| --- | ---: | ---: | ---: | --- |']
            for m in CORE:
                cells=[next(r for r in primary if (r['label'],r['stage'],r['report_group'],r['metric'])==(l,stage,group,m)) for l in E.LABELS]
                def fmt(r):return ('NA' if r['mean'] is None else f'{r["mean"]:.6g}')+f' ({r["valid_parents"]}/{r["expected_parents"]}父)'
                lines.append('| '+m+' | '+' | '.join(fmt(r) for r in cells)+' | '+unit(m)+' |')
            lines.append('')
    text_file(ANALYSIS/'METRICS.md','\n'.join(lines))
    files={str(p):E.sha(p) for p in ANALYSIS.iterdir() if p.is_file()}
    result=dict(status='COMPLETE',full_fields=177,groups=list(groups()),rows=E.budget()['total_score_rows'],files=files,model_calls=0,scoring_calls=0,
        derived_area_semantics='Deleted original generated slot measured both at RAW and pre-GEO; added bridges are separate. No GT surface completeness assertion.',
        collector_completion_sha256=E.sha(OUT/'results/completion.json'),report_code_sha256=E.sha(__file__))
    E.immutable(done,result);return result


def visual():
    """Validate submitted shards and summarize explicit observations only."""
    done=OUT/'visual_review/summary_v1/completion.json'
    if done.exists():
        r=E.read(done);E.verify(r['files']);E.verify(r['sources']);return r
    protocol=E.read(OUT/'visual_review/protocol.json');rendered=E.read(GALLERY/'completion.json');E.verify(rendered['files'])
    expected={(r['request_id'],r['label'],r['stage']):r for r in protocol['expected_units']};rows=[];sources={}
    for path in sorted((OUT/'visual_review/shards').glob('*.json')):
        sources[str(path)]=E.sha(path);payload=E.read(path);rows.extend(payload['rows'])
    keys=[(r['request_id'],r['label'],r['stage']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)-set(expected):raise ValueError('Duplicate/unregistered actual review')
    if len(rows)!=9*len(E.requests()):return dict(status='PENDING_ACTUAL_VIEW',actual_observations=len(rows),expected=9*len(E.requests()),missing=sorted(set(expected)-set(keys)),ratings_generated=0)
    for r in rows:
        key=(r['request_id'],r['label'],r['stage']);want=expected[key]
        for k in ('parent_id','condition_id','report_group','noise_index'):
            if r[k]!=want[k]:raise ValueError('Review identity mismatch '+k)
        png=GALLERY/(want['page_id']+'.png');rp=png.with_suffix('.json');rr=E.read(rp)
        if (r.get('reviewer_kind')!='AI_PROXY' or r.get('blind') is not False or r.get('actually_viewed') is not True
            or r.get('views_and_sections_checked') is not True):raise ValueError('Missing explicit actual-view attestation')
        for k in ('reviewer','view_event_id','viewed_at_utc','note'):
            if not str(r.get(k,'')).strip():raise ValueError('Empty actual-view '+k)
        if r['image_sha256']!=E.sha(png) or r['image_receipt_sha256']!=E.sha(rp) or r['gallery_definition_sha256']!=rr['definition_sha256']:
            raise ValueError('Review image identity differs')
        for axis,allowed in AXES.items():
            if r[axis] not in allowed:raise ValueError('Unrecognized observation '+axis)
    summary=[];transitions=[];by={(r['label'],r['stage'],r['request_id']):r for r in rows}
    for label in E.LABELS:
        for stage in E.STAGES:
            for group in groups():
                subset=[r for r in rows if (r['label'],r['stage'],r['report_group'])==(label,stage,group)]
                for axis,allowed in AXES.items():
                    for value in allowed:
                        pv=S.parent_values([(r,float(r[axis]==value)) for r in subset])
                        summary.append(dict(label=label,stage=stage,report_group=group,axis=axis,value=value,
                            count=sum(r[axis]==value for r in subset),requests=len(subset),parents=len(pv),
                            parent_equal_fraction=statistics.fmean(pv.values()),reviewer_kind='AI_PROXY',blind=False))
    for a,b in CONTRASTS:
        for stage in E.STAGES:
            for q in E.requests():
                ar,br=by[a,stage,q['request_id']],by[b,stage,q['request_id']]
                for axis in AXES:transitions.append(dict(reference=a,candidate=b,stage=stage,request_id=q['request_id'],parent_id=q['parent_id'],report_group=q['report_group'],axis=axis,reference_value=ar[axis],candidate_value=br[axis]))
    drift=[];history=E.OLD/'visual_review/summary_v1/actual_observations.csv'
    if history.exists():
        sources[str(history)]=E.sha(history)
        with history.open(encoding='utf-8-sig',newline='') as f:hist=list(csv.DictReader(f))
        for old in hist:
            if old['label']!='B':continue
            now=by.get(('P0',old['stage'],old['request_id']))
            if now is None:continue
            for axis in AXES:drift.append(dict(request_id=old['request_id'],stage=old['stage'],axis=axis,historical_value=old[axis],current_value=now[axis],changed=old[axis]!=now[axis],same_geometry=True))
    dest=OUT/'visual_review/summary_v1'
    for name,rr in dict(actual_observations=rows,parent_macro_counts=summary,paired_transitions=transitions,historical_P0_label_drift=drift).items():E.csvwrite(dest/(name+'.csv'),rr)
    result=dict(status='COMPLETE',actual_observations=9*len(E.requests()),ratings_generated=0,reviewer_kind='AI_PROXY',blind=False,
        protocol_sha256=E.sha(OUT/'visual_review/protocol.json'),sources=sources,files={str(p):E.sha(p) for p in dest.iterdir() if p.is_file()},
        caution='AI_PROXY is not independent human certification; no new success metric or automatic promotion.')
    E.immutable(dest/'completion.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','render','tables','visual'));p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    result={'prepare':prepare,'render':lambda:render(a.workers),'tables':tables,'visual':visual}[a.action]()
    if 'torch' in sys.modules:raise RuntimeError('CPU report imported torch')
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','images','sources','missing')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
