"""Fixed final figures from existing banks/raw/scores only; zero model calls."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from .common import read_json, digest, atomic_json

ARMS=('T1','Geo','Full')
LABELS={'T1':'Native','Geo':'Native+Geo','Full':'Native+Geo+Joint'}
COLORS=dict(known='#da9b38',free='#6b94b4',gt='#a5adb4',edge='#394c59',covered='#289351',missed='#cf443e')
CAMERA=dict(elevation=20,azimuth=-55,projection='ortho',display_coordinate_order=['X','Z','Y'])

def _resolve(out,path):
    p=Path(path);return p if p.is_absolute() else out/p

def _npz(path):
    with np.load(path,allow_pickle=False) as bank:return {k:bank[k].copy() for k in bank.files}

def _bank(out,record):
    path=_resolve(out,record['input_path'])
    if digest(path)!=record['input_sha256']:raise ValueError('Figure input identity changed')
    bank=_npz(path)
    if 'gt_free' not in bank:bank.update(_npz(_resolve(out,record['target_path'])))
    return bank

def _load_case(out,record,index):
    if record['status']!='READY':return dict(record=record,bank=None,outputs={},scores={},dependencies={})
    bank=_bank(out,record);outputs={};scores={};dependencies={record['input_path']:record['input_sha256']}
    for arm in ARMS:
        row=index[(arm,record['case_id'])]
        if row['status']!='PASS':
            outputs[arm]=None;scores[arm]=None;continue
        for field in ('raw','score'):
            path=_resolve(out,row[field+'_path']);expected=row[field+'_sha256']
            if digest(path)!=expected:raise ValueError('Figure '+field+' identity changed')
            dependencies[str(path.relative_to(out))]=expected
        outputs[arm]=_npz(_resolve(out,row['raw_path']))['raw'].reshape(-1,3,3)
        scores[arm]=read_json(_resolve(out,row['score_path']))
        if len(outputs[arm])!=int(bank['N']):raise ValueError('Figure raw N mismatch')
    reference=_npz(_resolve(out,record['ref_npz']));prefix=record.get('reference_key_prefix','')
    if prefix:reference={key[len(prefix):]:value for key,value in reference.items() if key.startswith(prefix)}
    return dict(record=record,bank=bank,outputs=outputs,scores=scores,reference=reference,dependencies=dependencies)

def _extent(cases):
    clouds=[]
    for case in cases:
        if case['bank'] is None:continue
        clouds.append(case['bank']['full_target'].reshape(-1,3))
        for geo in case['outputs'].values():
            if geo is not None and np.isfinite(geo).all():clouds.append(geo.reshape(-1,3))
    if not clouds:return np.zeros(3),1.
    points=np.concatenate(clouds);lo=points.min(0);hi=points.max(0)
    center=(lo+hi)/2;radius=max(float((hi-lo).max())*.55,1e-6)
    return center,radius

def _width(geo):
    return float(np.diff(np.quantile(np.asarray(geo).reshape(-1,3)[:,0],[.05,.95]))[0])

def _number(value,digits=2):
    if value is None:return 'NA'
    try:return f'{float(value):.{digits}f}' if np.isfinite(float(value)) else 'NA'
    except (TypeError,ValueError):return 'NA'

def _metric_label(score):
    if not score:return 'NO SAVED SCORE'
    m=score['metrics']
    coverage=100*m['fixed_R12_coverage_within_1pct_fraction'] if m.get('fixed_R12_coverage_within_1pct_fraction') is not None else None
    small=100*m['free_minimum_angle_lt5_fraction'] if m.get('free_minimum_angle_lt5_fraction') is not None else None
    gamma=m.get('Gamma_to_free_edges_RMS_bbox_pct')
    return ('RMS '+_number(m.get('remainder_symmetric_RMS_bbox_pct'))+'%L | cov '+_number(coverage,1)+'%\n'
        +'edge-Gamma '+_number(gamma)+'%L | angle<5 '+_number(small,1)+'%\n'
        +'CF '+str(m.get('CF_proper_pair_count','NA'))+'/'+str(m.get('CF_coplanar_pair_count','NA'))+' | FF '+str(m.get('FF_illegal_pair_count','NA')))

def _axis(fig,rows,cols,index,title,extent):
    ax=fig.add_subplot(rows,cols,index,projection='3d');ax.set_title(title,fontsize=8,pad=1)
    center,radius=extent
    ax.view_init(CAMERA['elevation'],CAMERA['azimuth']);ax.set_proj_type('ortho')
    for setter,dimension in ((ax.set_xlim,0),(ax.set_ylim,2),(ax.set_zlim,1)):
        setter(float(center[dimension]-radius),float(center[dimension]+radius))
    ax.set_box_aspect((1,1,1));ax.set_axis_off()
    return ax

def _mesh(ax,triangles,color,*,alpha=.78):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if triangles is None:
        ax.text2D(.15,.5,'NOT_RUN / FAILURE',transform=ax.transAxes,fontsize=9);return
    tri=np.asarray(triangles).reshape(-1,3,3)
    if not np.isfinite(tri).all():
        ax.text2D(.15,.5,'NONFINITE RAW',transform=ax.transAxes,fontsize=9);return
    if not len(tri):return
    ax.add_collection3d(Poly3DCollection(tri[:,:,[0,2,1]],facecolors=color,edgecolors=COLORS['edge'],linewidths=.13,alpha=alpha))

def _full(ax,geo,known):
    if geo is None:_mesh(ax,None,COLORS['free']);return
    _mesh(ax,geo[~known],COLORS['free']);_mesh(ax,geo[known],COLORS['known'],alpha=.96)

def _coverage(ax,case,arm,*,with_free=False):
    bank=case['bank'];score=case['scores'][arm]
    if with_free:
        geo=case['outputs'][arm]
        if geo is not None:_mesh(ax,geo[~bank['known_mask'].reshape(-1)],COLORS['free'],alpha=.35)
    points=case['reference']['coverage_points']
    distances=None if score is None else score.get('detail',{}).get('fixed_GT_coverage',{}).get('per_point_distances')
    if distances is None:
        colors='#999999';ax.text2D(.05,.03,'coverage NOT_AVAILABLE',transform=ax.transAxes,fontsize=7)
    else:
        if len(distances)!=len(points):raise ValueError('Saved coverage point/score length mismatch')
        colors=np.where(np.asarray(distances)<=.01*float(bank['L']),COLORS['covered'],COLORS['missed'])
    ax.scatter(points[:,0],points[:,2],points[:,1],c=colors,s=2.8,depthshade=False)

def _save(fig,path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    fig.savefig(tmp,format='png',dpi=115,facecolor='white');plt.close(fig);os.replace(tmp,path)

def _context_figure(case,path):
    import matplotlib.pyplot as plt
    r=case['record'];fig=plt.figure(figsize=(12.8,10.4))
    title=f"Table {r['table']} | {r['split']} | {r['uid']} | C={int(round(r['ratio']*100))}% | first frozen noise | phase3000 / total4000"
    fig.suptitle(title,fontsize=10,y=.99)
    if case['bank'] is None:
        fig.text(.12,.60,'NOT_APPLICABLE',fontsize=22,color='#ad493d')
        fig.text(.12,.50,r.get('reason','No applicable frozen condition'),fontsize=10,wrap=True)
        fig.text(.12,.40,'Registered slot retained. No replacement object, seed or generated output.',fontsize=10)
        _save(fig,path);return dict(case_ids=[r['case_id']],geometry_status='NOT_APPLICABLE',camera=CAMERA,extent=None)
    b=case['bank'];extent=_extent([case]);known=b['known_mask'].reshape(-1)
    # GT source order differs from sampling slots; construct the display explicitly.
    gt=np.concatenate((b['C'].reshape(-1,3,3),b['gt_free'].reshape(-1,3,3)));gtknown=np.arange(len(gt))<len(b['C'])
    ax=_axis(fig,3,4,1,'Ground truth full (orange = fixed C)',extent);_full(ax,gt,gtknown)
    ax=_axis(fig,3,4,5,'Input C only | N='+str(int(b['N']))+' K='+str(int(b['K'])),extent);_mesh(ax,b['C'],COLORS['known'],alpha=.96)
    ax=_axis(fig,3,4,9,'Ground truth free',extent);_mesh(ax,b['gt_free'],COLORS['gt'])
    for col,arm in enumerate(ARMS,1):
        geo=case['outputs'][arm];score=case['scores'][arm]
        ax=_axis(fig,3,4,col+1,LABELS[arm]+' full\n'+_metric_label(score),extent);_full(ax,geo,known)
        ax=_axis(fig,3,4,5+col,LABELS[arm]+' free',extent);_mesh(ax,None if geo is None else geo[~known],COLORS['free'])
        ax=_axis(fig,3,4,9+col,LABELS[arm]+' | saved GT-free coverage',extent);_coverage(ax,case,arm)
    fig.text(.03,.015,'Shared orthographic camera/range. Display axes X/Z/Y. Coverage: green <= 1% L; red > 1% L. CF = proper/coplanar; FF = illegal pairs. Orange C is copied, not a learned quality metric.',fontsize=7.4)
    fig.subplots_adjust(top=.94,bottom=.055,left=.01,right=.99,wspace=.02,hspace=.19)
    _save(fig,path)
    return dict(case_ids=[r['case_id']],geometry_status='READY',camera=CAMERA,extent=dict(center=extent[0].astype(float).tolist(),radius=extent[1]),dependencies=case['dependencies'])

def _width_figure(cases,path):
    import matplotlib.pyplot as plt
    cases=sorted(cases,key=lambda c:c['record']['alpha']);first=cases[0]['record'];extent=_extent(cases)
    fig=plt.figure(figsize=(12.2,19.6));fig.suptitle('Table B | '+first['split']+' | '+first['uid']+' | fixed noise0 | phase3000 / total4000',fontsize=11,y=.995)
    for col,case in enumerate(cases):
        r=case['record'];b=case['bank'];alpha=r['alpha']
        if b is None:
            for row in range(7):
                ax=_axis(fig,7,3,row*3+col+1,'alpha='+str(alpha),extent)
                ax.text2D(.05,.5,'NOT_APPLICABLE\n'+r.get('reason',''),transform=ax.transAxes,fontsize=7,wrap=True)
            continue
        known=b['known_mask'].reshape(-1);gtfree=b['gt_free'].reshape(-1,3,3);Wgt=_width(gtfree)
        gt=np.concatenate((b['C'].reshape(-1,3,3),gtfree));gtknown=np.arange(len(gt))<len(b['C'])
        ax=_axis(fig,7,3,col+1,'alpha='+str(alpha)+' | GT + fixed C\nGT free Wx='+_number(Wgt,4),extent);_full(ax,gt,gtknown)
        for model_index,arm in enumerate(ARMS):
            geo=case['outputs'][arm];score=case['scores'][arm];free=None if geo is None else geo[~known]
            row=1+2*model_index
            ax=_axis(fig,7,3,row*3+col+1,LABELS[arm]+' full\n'+_metric_label(score),extent);_full(ax,geo,known)
            width='NA' if free is None else _number(_width(free),4)
            ax=_axis(fig,7,3,(row+1)*3+col+1,LABELS[arm]+' free + GT coverage\nWx='+width+' | target='+_number(Wgt,4),extent)
            _coverage(ax,case,arm,with_free=True)
    fig.text(.025,.014,'One camera and one coordinate range across all three alpha values and all models. X/Z/Y display; no per-cell scaling. Green/red points: saved corresponding variant GT-free coverage <=/> 1% L_alpha. Free Wx = equal-corner q95-q05.',fontsize=7.1)
    fig.subplots_adjust(top=.967,bottom=.035,left=.01,right=.99,wspace=.025,hspace=.24)
    _save(fig,path)
    deps={key:val for case in cases for key,val in case['dependencies'].items()}
    return dict(case_ids=[c['record']['case_id'] for c in cases],geometry_status='READY' if all(c['bank'] is not None for c in cases) else 'NOT_APPLICABLE',
        camera=CAMERA,extent=dict(center=extent[0].astype(float).tolist(),radius=extent[1]),alpha_shared_extent=True,dependencies=deps)

def generate_fixed_figures(out,panels=None):
    """Return a durable manifest. Requires completed final raw/score artifacts."""
    out=Path(out).resolve();panels=read_json(out/'panels.json') if panels is None else panels
    manifest_path=out/'figures/fixed_manifest.json'
    if manifest_path.exists():
        old=read_json(manifest_path)
        if old['panels_sha256']!=digest(out/'panels.json'):raise ValueError('Figure panel identity changed')
        for name,sha in old['ledger_sha256'].items():
            if digest(out/name)!=sha:raise ValueError('Figure source ledger changed')
        for item in old['figures']:
            if digest(out/item['path'])!=item['sha256']:raise ValueError('Fixed figure changed')
        return old
    import matplotlib
    matplotlib.use('Agg')
    index={};ledgers={}
    for arm in ARMS:
        path=out/'evaluation/phase3000'/arm/'ledger.json';ledger=read_json(path)
        if ledger['status']!='PASS' or ledger['identity']['phase_step']!=3000 or ledger['identity']['arm']!=arm:
            raise ValueError('Final evaluation incomplete; cannot create final figures')
        ledgers[str(path.relative_to(out))]=digest(path)
        for row in ledger['rows']:index[(arm,row['case_id'])]=row
    # Verify all referenced immutable banks once before rendering, including GT refs.
    for name,sha in panels['frozen_files'].items():
        if digest(_resolve(out,name))!=sha:raise ValueError('Frozen figure bank changed: '+name)
    result=dict(status='PASS',schema='chair_domain_fixed_figures_v1',panels_sha256=digest(out/'panels.json'),ledger_sha256=ledgers,
        selection='All table A conditions, all table B parents at all three alpha, all original table C conditions; exactly frozen noise_index=0',
        model_forwards=0,OT_calls=0,sampling_calls=0,independent_geometry_score_calls=0,
        coverage='Read saved per-point distances and corresponding frozen coverage_points; do not recalculate surface distances',figures=[])
    for table in ('A','C'):
        for record in panels[table]:
            if record['noise_index']!=0:continue
            case=_load_case(out,record,index);path=out/'figures/fixed'/table/(record['case_id']+'.png')
            details=_context_figure(case,path)
            result['figures'].append(dict(table=table,path=str(path.relative_to(out)),sha256=digest(path),**details))
    for uid in panels['B_train_parents']+panels['B_validation_parents']:
        records=[r for r in panels['B'] if r['uid']==uid and r['noise_index']==0]
        if len(records)!=3:raise ValueError('Width figure requires three registered alpha slots')
        cases=[_load_case(out,r,index) for r in records];path=out/'figures/fixed/B'/(uid+'.png')
        details=_width_figure(cases,path)
        result['figures'].append(dict(table='B',path=str(path.relative_to(out)),sha256=digest(path),**details))
    if {t:sum(r['table']==t for r in result['figures']) for t in ('A','B','C')}!={'A':32,'B':8,'C':16}:
        raise ValueError('Fixed figure denominator differs')
    result['files']=[dict(path=item['path'],sha256=item['sha256']) for item in result['figures']]
    atomic_json(manifest_path,result)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=generate_fixed_figures(a.out)
    print('Fixed figures:',len(result['figures']))