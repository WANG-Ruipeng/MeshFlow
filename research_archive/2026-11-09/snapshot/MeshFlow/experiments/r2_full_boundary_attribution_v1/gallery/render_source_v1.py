"""CPU-only diagnostic figures for the fixed FULL-boundary attribution panel.

Renders saved meshes and saved diagnostic arrays only. Does not classify use,
repair geometry, recompute old quality metrics or call a model.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, csv, html, io, json, math
from pathlib import Path
import numpy as np
from . import common as C
from r2.pilots.regular_patch_control_v1.depth_gallery import raster, basis
from r2.tools.cref_layer_audit_v1.render import finite_sections

VERSION='full_boundary_saved_candidates_depth_sections_v1'
COLS=(('SMALL_C200','RAW'),('CREF_C200','RAW'),('SMALL_C200','POST'),('CREF_C200','POST'))
BLUE='#1686c0'; GRAY='#aeb6bd'; ORANGE='#e18b2c'
LINES=(('surface','#24764c','finite F surface'),('edge','#286bbe','all finite F edges'),
       ('open_edge','#8d48a6','F-open finite edges'),('vertex','#c94442','F vertices'))
DEST=C.OUT/'gallery'
DEFINITION=dict(version=VERSION,pages=11,cells=44,
    columns=[list(x) for x in COLS],depth_views='Whole azimuth -60 / elevation 23; local azimuth 120 / elevation 28',
    depth_method='Inherited CPU finite barycentric z-buffer; only exact/1e-12 depth ties favor earlier C.',
    whole_scale='Union of all registered outputs per parent, all noise/model/stage states; common world scale.',
    local_scale='C bbox plus fixed .075L X/Z and .10L Y margins; all registered states of the same C share scale.',
    sections='Finite triangle intersections with Z=center(C bbox Z) and X=center(C bbox X). Positions depend only on C.',
    candidates='Orange: any F face nearest to at least one saved boundary edge-midpoint sample, regardless of distance; no illegality/semantic label.',
    distance_query='Saved deterministic source-boundary edge midpoints, in source edge-ID order; vertices have zero length weight and are excluded from plotted curves.',
    distance_x='Cumulative source edge length / L, concatenated in sorted source edge ID order. Adjacent plotted intervals need not be spatial neighbors.',
    distance_y='Distance / L * 100, symmetric-log axis with .2 percent linear region; fixed limits across all registered states of the same parent.',
    distance_targets='Finite F surface, all finite F edges, F-open finite edges, F vertices; no C self-match.',
    observation_limit='Raster misses subpixel gaps; exact coincident sections can overprint. Orange nearest faces can be distant or semantically unrelated. These figures are not a legal-connectivity, thickness or adoption certification.',
    new_geometry=0,model_calls=0,old_quality_scoring_calls=0)


def _csv(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def _load(cell):
    p=C.resolve(cell['geometry_path'])
    if cell.get('geometry_sha256') and C.sha(p)!=cell['geometry_sha256']:
        raise ValueError('Frozen geometry changed: '+cell['cell_id'])
    g=C.arrays(p);c=np.asarray(g['C_original']).reshape(-1,3,3);f=np.asarray(g['F']).reshape(-1,3,3)
    if len(c)!=int(cell['K']) or not np.isfinite(c).all() or not np.isfinite(f).all():
        raise ValueError('Wrong/nonfinite C/F geometry')
    d=C.OUT/'results'/cell['cell_id'];summary=C.read(d/'summary.json');v=C.arrays(d/'vectors.npz')
    edge=_csv(d/'edge_summary.csv');edge.sort(key=lambda x:int(x['boundary_edge_id']))
    for field in ('surface_distance','surface_id','edge_distance','open_edge_distance','vertex_distance','sample_kind','points','boundary_edge_id'):
        if field not in v:raise ValueError('Missing saved diagnostic '+field)
    m=v['sample_kind']==0;candidate=set(map(int,v['surface_id'][m]));candidate.discard(-1)
    if any(x>=len(f) for x in candidate):raise ValueError('Candidate face identity mismatch')
    refs={str(x):C.sha(x) for x in (p,d/'summary.json',d/'vectors.npz',d/'edge_summary.csv')}
    return dict(cell=cell,C=c,F=f,tri=np.concatenate([c,f]),summary=summary,vectors=v,edges=edge,
                candidates=candidate,files=refs)


def _projected_span(points,center,B):
    return float(np.ptp((points-center)@B,axis=0)[:2].max())


def _bounds(items):
    groups=collections.defaultdict(list)
    for it in items:groups[it['cell']['parent_id']].append(it)
    result={};B=basis(-60,23);localB=basis(120,28)
    for pid,rr in groups.items():
        cv=rr[0]['C'].reshape(-1,3);lo=cv.min(0);hi=cv.max(0);localcenter=(lo+hi)/2
        cp=sorted(map(tuple,cv))
        if any(sorted(map(tuple,x['C'].reshape(-1,3)))!=cp for x in rr):
            raise ValueError('Parent has multiple distinct C; group by condition instead')
        points=np.concatenate([it['tri'].reshape(-1,3) for it in rr]);center=(points.min(0)+points.max(0))/2
        whole=max(_projected_span(points,center,B),1e-8)
        margins=np.array([.075,.10,.075])*C.L;low=lo-margins;high=hi+margins
        corners=np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
        localspan=max(_projected_span(corners,localcenter,localB),1e-8)
        distances=np.concatenate([it['vectors'][name+'_distance'][it['vectors']['sample_kind']==0] for it in rr for name,_,_ in LINES])
        finite=distances[np.isfinite(distances)]*100/C.L
        ymax=max(float(finite.max())*1.07 if len(finite) else 0.,2.)
        result[pid]=dict(whole_center=center.tolist(),whole_span=whole,local_center=localcenter.tolist(),
                         local_span=localspan,local_low=low.tolist(),local_high=high.tolist(),
                         C_center=((lo+hi)/2).tolist(),distance_ymax_pctL=ymax,
                         whole_basis=B.tolist(),local_basis=localB.tolist())
    return result


def _depth(it,B,center,span,size=460):
    image,owner=raster(it['tri'],len(it['C']),np.asarray(B),np.asarray(center),float(span),size=size)
    pixels=np.asarray(image).copy();orange=np.array([225,139,44],float)
    for fi in it['candidates']:
        take=owner==len(it['C'])+fi
        if np.any(take):
            shade=pixels[take,0].astype(float)/185.
            pixels[take]=np.clip(shade[:,None]*orange,0,255).astype(np.uint8)
    return pixels,dict(visible_C_pixels=int(((owner>=0)&(owner<len(it['C']))).sum()),
                       visible_nearest_candidate_pixels=int(np.isin(owner,list(len(it['C'])+x for x in it['candidates'])).sum()))


def _sections(ax,it,axis,position,projection,low,high):
    from matplotlib.collections import LineCollection
    fs,owners=finite_sections(it['F'],axis,position)
    if len(fs):
        colors=[ORANGE if int(fi) in it['candidates'] else GRAY for fi in owners]
        ax.add_collection(LineCollection(fs[:,:,projection],colors=colors,linewidths=1.25,alpha=.92))
    cs,_=finite_sections(it['C'],axis,position)
    if len(cs):ax.add_collection(LineCollection(cs[:,:,projection],colors=BLUE,linewidths=2.1))
    ax.set_xlim(low[projection[0]],high[projection[0]]);ax.set_ylim(low[projection[1]],high[projection[1]])
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
    ax.tick_params(labelsize=6,pad=1)
    ax.set_xlabel(('X' if projection[0]==0 else 'Z')+' (model units)',fontsize=7,labelpad=1)
    return dict(finite_C_segments=len(cs),finite_F_segments=len(fs),slice_axis=axis,position=position)


def _curve(ax,it,ymax):
    v=it['vectors'];offset=0.;ends=[];labels=[];centers=[]
    for e in it['edges']:
        eid=int(e['boundary_edge_id']);length=float(e['length']);chosen=(v['sample_kind']==0)&(v['boundary_edge_id']==eid)
        a=np.array([float(e['start_'+x]) for x in 'xyz']);b=np.array([float(e['end_'+x]) for x in 'xyz'])
        t=((v['points'][chosen]-a)@(b-a))/np.dot(b-a,b-a)
        order=np.argsort(t);x=(offset+t[order]*length)/C.L
        for name,color,label in LINES:
            y=v[name+'_distance'][chosen][order]*100/C.L
            # Nonfinite targets remain NA in numeric tables; no invented zero.
            ax.plot(x,np.where(np.isfinite(y),y,np.nan),color=color,lw=1.1,label=label if eid==0 else None)
        centers.append((offset+length/2)/C.L);labels.append(str(eid));offset+=length;ends.append(offset/C.L)
    for x in ends[:-1]:ax.axvline(x,color='#bbbbbb',lw=.45,alpha=.6)
    ax.axhline(.1,color='#999999',lw=.7,ls=':');ax.axhline(1.,color='#999999',lw=.7,ls=':')
    ax.set_yscale('symlog',linthresh=.2,linscale=.5);ax.set_ylim(0,ymax);ax.set_xlim(0,offset/C.L)
    ax.tick_params(labelsize=6,pad=1);ax.set_xlabel('concatenated boundary length / L',fontsize=7,labelpad=1)
    top=ax.secondary_xaxis('top');stride=max(1,int(math.ceil(len(labels)/14)))
    top.set_xticks(centers[::stride]);top.set_xticklabels(labels[::stride],fontsize=6);top.tick_params(length=2,pad=1)
    top.set_xlabel('source boundary edge IDs (sorted)',fontsize=6,labelpad=2)
    return dict(source_edge_count=len(ends),total_length_over_L=offset/C.L,
                plotted_samples=int((v['sample_kind']==0).sum()),nonfinite_distance_values={name:int(np.sum(~np.isfinite(v[name+'_distance'][v['sample_kind']==0]))) for name,_,_ in LINES})


def _page(request,items,bounds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ordered={(x['cell']['label'],x['cell']['stage']):x for x in items}
    if set(ordered)!=set(COLS):raise ValueError('Incomplete four-column request')
    cell=items[0]['cell'];b=bounds[cell['parent_id']]
    fig,axes=plt.subplots(5,4,figsize=(16,14),gridspec_kw={'height_ratios':[2.7,2.7,1.25,1.25,2.1]})
    fig.subplots_adjust(left=.055,right=.995,bottom=.065,top=.925,hspace=.53,wspace=.14)
    short=cell['parent_id'][:8]+'_n'+str(cell['candidate'])
    fig.suptitle(short+' | FULL K='+str(cell['K'])+' | saved RAW and POST, identical view scales',fontsize=14,x=.5,y=.99)
    fig.text(.5,.963,'Blue: source C. Orange: nearest finite F faces for boundary samples (not an illegality or adoption label). Gray: other F.',ha='center',fontsize=9)
    records=[]
    for col,key in enumerate(COLS):
        it=ordered[key];info={'cell_id':it['cell']['cell_id'],'label':key[0],'stage':key[1]}
        axes[0,col].set_title(key[0]+' / '+key[1],fontsize=11,pad=7)
        pixels,visibility=_depth(it,b['whole_basis'],b['whole_center'],b['whole_span'])
        axes[0,col].imshow(pixels);axes[0,col].axis('off');info['whole_visibility']=visibility
        pixels,visibility=_depth(it,b['local_basis'],b['local_center'],b['local_span'])
        axes[1,col].imshow(pixels);axes[1,col].axis('off');info['local_visibility']=visibility
        for row,axis,projection,name in ((2,2,[0,1],'front'),(3,0,[2,1],'side')):
            info[name+'_section']=_sections(axes[row,col],it,axis,b['C_center'][axis],projection,b['local_low'],b['local_high'])
            if col==0:axes[row,col].set_ylabel(name+' finite cut\nY (model units)',fontsize=8)
        info['distance_curve']=_curve(axes[4,col],it,b['distance_ymax_pctL'])
        if col==0:axes[4,col].set_ylabel('distance (% fixed L)\nsymlog',fontsize=8)
        records.append(info)
    handles,labels=axes[4,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.023),ncol=4,fontsize=8,frameon=False)
    fig.text(.008,.797,'whole',rotation=90,fontsize=9);fig.text(.008,.592,'near C',rotation=90,fontsize=9)
    fig.text(.5,.007,'Sorted edge-ID intervals need not be neighbors. Finite cuts may miss other regions; coincident lines can overprint. No geometry changes or topology certification.',ha='center',fontsize=8)
    buff=io.BytesIO();fig.savefig(buff,format='jpg',dpi=120,pil_kwargs={'quality':91,'subsampling':0},facecolor='white');plt.close(fig)
    dest=DEST/(short+'.jpg');C._write_bytes(dest,buff.getvalue())
    return dict(request_id=request,parent_id=cell['parent_id'],candidate=cell['candidate'],path=str(dest),sha256=C.sha(dest),
                pixels=[1920,1680],cell_ids=[x['cell_id'] for x in records],columns=records)


def run():
    regpath=C.OUT/'registration.json';r=C.read(regpath)
    if r.get('status')!='FROZEN':raise ValueError('Freeze registration before rendering')
    cells=r['cells']
    if len(cells)!=44 or len({x['cell_id'] for x in cells})!=44:raise ValueError('Expected exactly 44 frozen cells')
    groups=collections.defaultdict(list)
    for cell in cells:groups[cell['request_id']].append(cell)
    if len(groups)!=11:raise ValueError('Expected exactly 11 frozen requests')
    codefiles=[Path(__file__),Path(C.__file__),C.ROOT/'r2/pilots/regular_patch_control_v1/depth_gallery.py',C.ROOT/'r2/tools/cref_layer_audit_v1/render.py']
    sourcehash={str(x):C.sha(x) for x in codefiles};manifest=DEST/'manifest.json'
    if manifest.exists():
        old=C.read(manifest)
        if old['registration_sha256']!=C.sha(regpath) or old['source_hashes']!=sourcehash:raise ValueError('Render identity changed')
        for p,h in old['input_hashes'].items():
            if C.sha(p)!=h:raise ValueError('Rendered source changed')
        for im in old['images']:
            if C.sha(im['path'])!=im['sha256']:raise ValueError('Rendered image changed')
        return old
    data=[_load(c) for c in cells];byid={d['cell']['cell_id']:d for d in data};bounds=_bounds(data);images=[]
    for rid,cc in groups.items():
        images.append(_page(rid,[byid[c['cell_id']] for c in cc],bounds))
        print(json.dumps(dict(render_complete=len(images),registered=11)),flush=True)
    index=('<!doctype html><meta charset="utf-8"><title>FULL boundary attribution</title>'
           '<style>body{font:15px sans-serif;max-width:1920px;margin:auto;padding:20px}img{width:100%;height:auto}p{max-width:1100px}</style>'
           '<h1>FULL boundary attribution: saved geometry</h1><p>'+html.escape(DEFINITION['observation_limit'])+'</p>'
           +''.join('<h2>'+html.escape(x['parent_id'][:8]+'_n'+str(x['candidate']))+'</h2><a href="'+Path(x['path']).name+'"><img src="'+Path(x['path']).name+'" loading="lazy"></a>' for x in images))
    C._write_bytes(DEST/'index.html',index.encode('utf-8'))
    result=dict(status='COMPLETE',version=VERSION,definition=DEFINITION,registration_sha256=C.sha(regpath),
                source_hashes=sourcehash,input_hashes={p:h for d in data for p,h in d['files'].items()},
                bounds=bounds,images=images,html_path=str(DEST/'index.html'),html_sha256=C.sha(DEST/'index.html'),
                new_geometry=0,model_calls=0,old_quality_scoring_calls=0)
    C.write(manifest,result);C.event('OFFLINE_BOUNDARY_RENDER_COMPLETE',images=11,cells=44)
    return result


def main():
    parser=argparse.ArgumentParser();parser.parse_args();r=run()
    print(json.dumps(dict(status=r['status'],images=len(r['images']),version=r['version'])),flush=True)

if __name__=='__main__':main()
