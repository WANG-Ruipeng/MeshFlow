"""Window A endpoint convergence evidence, frozen GT scores, and fixed-view figures.

These four observed cases are deliberately selected diagnostics. No solver is
selected using GT error; no representative win rate or ODE truth is inferred.
"""
from pathlib import Path
import argparse
import csv
import time
import numpy as np
from .assets import read_json, write_json, file_sha256
from .scoring import own_out
from native_t1.metrics import symmetric_surface_rms

SOLVERS=('R1','R2','D1','D2')
ORDER=('E50','E100','E200')+SOLVERS
LABELS={'E50':'Euler 50','E100':'Euler 100','E200':'Euler 200',
        'R1':'RK4 100 (400 NFE)','R2':'RK4 200 (800 NFE)',
        'D1':'Dopri5 rtol=1e-4','D2':'Dopri5 rtol=1e-5'}
KEYS=('remainder_symmetric_RMS_bbox_pct','free_to_R12_RMS_bbox_pct',
      'R12_to_free_RMS_bbox_pct','fixed_R12_coverage_RMS_bbox_pct',
      'fixed_R12_coverage_within_1pct_fraction','boundary_RMS_bbox_pct',
      'boundary_p95_bbox_pct','boundary_coverage_fraction',
      'Gamma_to_free_edges_RMS_bbox_pct','Gamma_to_free_edges_p95_bbox_pct',
      'CF_proper_affected_free_fraction','CF_proper_pair_count',
      'CF_coplanar_pair_count','CF_unresolved_or_degenerate_pair_count',
      'FF_proper_pair_count','FF_coplanar_pair_count','FF_illegal_pair_count',
      'FF_unresolved_or_degenerate_pair_count','free_total_surface_area',
      'free_near_degenerate_count','free_exact_zero_area_count',
      'free_minimum_angle_lt5_fraction','free_angle_p05_degrees',
      'nan_coordinate_count','inf_coordinate_count','whole_symmetric_RMS_bbox_pct')


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def load_raw(row,out):
    path=Path(row['path'])
    if not path.is_absolute():path=Path(out)/path
    if file_sha256(path)!=row['sha256']:
        raise ValueError('Endpoint raw checksum differs: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        a=z['raw'].copy().reshape(112,3,3)
    if a.dtype!=np.float32 or not np.isfinite(a).all():
        raise ValueError('Endpoint coordinates must be finite FP32')
    return a


def endpoint_difference(left,right,bbox):
    """Compare free coordinates and independent area-sampled surface geometry.

    Coordinate RMS is per coordinate; vertex RMS is also supplied to remove
    convention ambiguity. Surface comparison has no source-slot correspondence.
    Its 512 samples/direction follow the unchanged seed9911 scoring helper.
    """
    left=np.asarray(left).reshape(112,3,3)
    right=np.asarray(right).reshape(112,3,3)
    if left[:12].tobytes()!=right[:12].tobytes():
        raise ValueError('Compared endpoints do not have identical C')
    d=left[12:].astype(np.float64)-right[12:].astype(np.float64)
    vertex=np.linalg.norm(d,axis=-1)
    surface=symmetric_surface_rms(left[12:],right[12:],bbox)
    return dict(coordinate_RMS=float(np.sqrt(np.mean(d*d))),
        coordinate_max_abs=float(np.max(np.abs(d))),
        vertex_RMS=float(np.sqrt(np.mean(vertex*vertex))),
        vertex_max_L2=float(vertex.max()),
        coordinate_RMS_bbox_pct=float(100*np.sqrt(np.mean(d*d))/bbox),
        vertex_RMS_bbox_pct=float(100*np.sqrt(np.mean(vertex*vertex))/bbox),
        coordinate_max_abs_bbox_pct=float(100*np.max(np.abs(d))/bbox),
        vertex_max_L2_bbox_pct=float(100*vertex.max()/bbox),
        surface=surface,
        interpretation='Slot differences are not semantic errors; sampled surface differences are not certified numerical bounds')


def collect(out):
    out=own_out(out);protocol=read_json(out/'registration/protocol.json')
    rows=[]
    for path in sorted((out/'trajectories').glob('*/result.json')):
        row=read_json(path);ident=path.parent.name
        if row.get('repeat') or row.get('is_repeat') or 'repeat' in ident.lower():continue
        record=dict(row,source='new_formal',trajectory_id=ident)
        scorepath=out/'scores'/ident/'result.json'
        if scorepath.exists():
            score=read_json(scorepath)
            if score['status']=='PASS':
                if score['row']['sha256']!=row['sha256']:
                    raise ValueError('Score belongs to a different endpoint')
                record.update(scalars=score['scalars'],score_status='AVAILABLE',score_path=str(scorepath.relative_to(out)))
            else:record.update(scalars={},score_status=score['status'])
        else:record.update(scalars={},score_status='NOT_AVAILABLE')
        rows.append(record)
    for row in read_json(out/'registration/old_euler_index.json')['rows']:
        score=row.get('score',{})
        rows.append(dict(row,source='old_Euler_read_only',solver='E'+str(row['nfe']),
                         complete=row['status']=='AVAILABLE',scalars=score.get('scalars',{}),
                         score_status=score.get('status','NOT_AVAILABLE')))
    return protocol,rows


def fixed_view_figure(out,protocol,rows,free_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    out=own_out(out)
    lookup={(r['case_id'],r['model'],r['solver']):r for r in rows}
    fig=plt.figure(figsize=(21,22))
    for cidx,case in enumerate(protocol['cases']):
        cid=case['id'];available=[]
        for model in ('B','J'):
            for solver in ORDER:
                row=lookup.get((cid,model,solver))
                if row and row.get('complete'):
                    available.append(load_raw(row,out))
        if available:
            joined=np.concatenate(available).reshape(-1,3)
            low=joined.min(0);high=joined.max(0);center=(low+high)/2
            radius=max(float((high-low).max())*.55,1e-3)
        else:center=np.zeros(3);radius=1
        for mi,model in enumerate(('B','J')):
            for si,solver in enumerate(ORDER):
                ax=fig.add_subplot(8,7,(2*cidx+mi)*7+si+1,projection='3d')
                row=lookup.get((cid,model,solver))
                if row and row.get('complete'):
                    a=load_raw(row,out)
                    free=Poly3DCollection(a[12:,:,[0,2,1]],facecolors='#5d91ad',edgecolors='#244a60',linewidths=.12,alpha=.96)
                    ax.add_collection3d(free)
                    if not free_only:
                        known=Poly3DCollection(a[:12,:,[0,2,1]],facecolors='#dca74c',edgecolors='#665028',linewidths=.3,alpha=1)
                        ax.add_collection3d(known)
                    s=row.get('scalars',{}).get('remainder_symmetric_RMS_bbox_pct')
                    detail=('RMS %.4f%%'%s) if s is not None else 'score unavailable'
                else:detail=row.get('status','NOT_AVAILABLE') if row else 'NOT_AVAILABLE'
                display=center[[0,2,1]]
                ax.set_xlim(display[0]-radius,display[0]+radius)
                ax.set_ylim(display[1]-radius,display[1]+radius)
                ax.set_zlim(display[2]-radius,display[2]+radius)
                ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-54);ax.set_axis_off()
                ax.set_title(f"{model} | {LABELS[solver]}\n{detail}",fontsize=8)
                if si==0:
                    ax.text2D(-.10,.55,cid,transform=ax.transAxes,rotation=90,va='center',fontsize=9)
    fig.suptitle('Four deliberately selected observed cases | '+('free surfaces only' if free_only else 'full outputs: known C gold, free blue')+
                 '\nFP32; same view and bounds within each case; absent/incomplete results are not endpoints',fontsize=13)
    fig.subplots_adjust(left=.05,right=.995,top=.965,bottom=.005,wspace=.00,hspace=.10)
    path=out/('fixed_view_free_only.png' if free_only else 'fixed_view_full.png')
    fig.savefig(path,dpi=135);plt.close(fig)
    return dict(path=path.name,sha256=file_sha256(path),free_only=free_only,
                view=dict(elev=24,azim=-54,display_coordinate_order='X,Z,Y',bounds='shared across all methods and B/J within each case'))


def analyze(out,figures=True):
    tick=time.perf_counter();out=own_out(out);protocol,rows=collect(out)
    lookup={(r['case_id'],r['model'],r['solver']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate formal comparison rows')
    pairs=[];missing=[];table=[];jb=[];convergence=[];arrays={}
    def array(row):
        key=(row['case_id'],row['model'],row['solver'])
        if key not in arrays:arrays[key]=load_raw(row,out)
        return arrays[key]
    for row in rows:
        table.append(dict(model=row['model'],case_id=row['case_id'],parent=row['parent'],patch=row['patch'],
                          seed=row['seed'],solver=row['solver'],source=row['source'],status=row['status'],
                          complete=row.get('complete',False),nfe=row.get('nfe'),score_status=row['score_status'],
                          **row.get('scalars',{})))
    for case in protocol['cases']:
        cid=case['id'];bbox=case['bbox']
        for model in ('B','J'):
            comparisons=[('R1','R2'),('D1','D2'),('D2','R2'),('R1','D1')]
            comparisons += [(e,s) for e in ('E50','E100','E200') for s in SOLVERS]
            comparisons += [('E50','E100'),('E100','E200')]
            local={}
            for a,b in comparisons:
                left=lookup.get((cid,model,a));right=lookup.get((cid,model,b))
                info=dict(model=model,case_id=cid,left_solver=a,right_solver=b)
                if not (left and right and left.get('complete') and right.get('complete')):
                    missing.append(dict(info,status='NOT_AVAILABLE',left_status=left.get('status') if left else 'MISSING',right_status=right.get('status') if right else 'MISSING'))
                    continue
                delta=endpoint_difference(array(left),array(right),bbox)
                record=dict(info,status='PASS',**delta,
                            left_own_GT=left.get('scalars',{}),right_own_GT=right.get('scalars',{}))
                pairs.append(record);local[a+'__'+b]=record
            def ratio(n,d,kind):
                x=local.get(n);y=local.get(d)
                if not x or not y:return None
                xv=x['surface']['symmetric_rms_bbox_pct'] if kind=='surface' else x['coordinate_RMS_bbox_pct']
                yv=y['surface']['symmetric_rms_bbox_pct'] if kind=='surface' else y['coordinate_RMS_bbox_pct']
                return xv/yv if yv else None
            convergence.append(dict(model=model,case_id=cid,
                R1_R2_coordinate_vs_E50_E100_ratio=ratio('R1__R2','E50__E100','coordinate'),
                R1_R2_surface_vs_E50_E100_ratio=ratio('R1__R2','E50__E100','surface'),
                D2_R2_coordinate_vs_R1_D1_ratio=ratio('D2__R2','R1__D1','coordinate'),
                D2_R2_surface_vs_R1_D1_ratio=ratio('D2__R2','R1__D1','surface'),
                pair_values={key:dict(coordinate_RMS_bbox_pct=r['coordinate_RMS_bbox_pct'],
                                     surface_RMS_bbox_pct=r['surface']['symmetric_rms_bbox_pct']) for key,r in local.items()},
                automatic_classification='NOT_ASSIGNED: inspect coordinate and surface evidence; no GT-based solver selection'))
        for solver in ORDER:
            b=lookup.get((cid,'B',solver));j=lookup.get((cid,'J',solver))
            if not (b and j and b.get('complete') and j.get('complete') and b['score_status']=='AVAILABLE' and j['score_status']=='AVAILABLE'):
                jb.append(dict(case_id=cid,solver=solver,status='NOT_AVAILABLE'))
                continue
            shared=set(b['scalars'])&set(j['scalars']);deltas={}
            for key in sorted(shared):
                bv=b['scalars'][key];jv=j['scalars'][key]
                if isinstance(bv,(int,float)) and isinstance(jv,(int,float)):
                    deltas[key]=float(jv-bv)
            jb.append(dict(case_id=cid,solver=solver,status='PASS',B=b['scalars'],J=j['scalars'],J_minus_B=deltas,
                           generated_endpoint_difference=endpoint_difference(array(b),array(j),bbox)))
    # Do not pool the intentionally imbalanced four diagnostic cases into a population win rate.
    counts=dict(formal_rows=sum(r['source']=='new_formal' for r in rows),
                formal_complete=sum(r['source']=='new_formal' and r.get('complete',False) for r in rows),
                formal_scored=sum(r['source']=='new_formal' and r['score_status']=='AVAILABLE' for r in rows),
                old_Euler_available=sum(r['source']=='old_Euler_read_only' and r.get('complete',False) for r in rows),
                paired_endpoint_comparisons=len(pairs),B_J_endpoint_comparisons=sum(r['status']=='PASS' for r in jb),
                missing_pair_comparisons=len(missing),model_forwards=0)
    write_csv(out/'per_case.csv',table)
    pairtable=[]
    for r in pairs:
        pairtable.append({**{k:v for k,v in r.items() if not isinstance(v,(dict,list))},
            **{'surface_'+k:v for k,v in r['surface'].items() if isinstance(v,(str,int,float))}})
    write_csv(out/'endpoint_differences.csv',pairtable)
    write_csv(out/'J_minus_B.csv',[dict(case_id=r['case_id'],solver=r['solver'],status=r['status'],**r.get('J_minus_B',{})) for r in jb])
    figure_records=[]
    if figures:
        figure_records=[fixed_view_figure(out,protocol,rows,False),fixed_view_figure(out,protocol,rows,True)]
    result=dict(status='PASS',schema='numeric_reference_A_analysis_v1',counts=counts,
                source_rows=rows,endpoint_differences=pairs,missing_comparisons=missing,
                convergence_evidence=convergence,J_minus_B=jb,figures=figure_records,
                CPU_analysis_seconds=time.perf_counter()-tick,model_forwards=0,backward_calls=0,OT_calls=0,
                surface_comparison=dict(kind='independent generated-free-surface to generated-free-surface',
                    helper='native_t1.metrics.symmetric_surface_rms',points_per_surface=512,seed=9911,
                    denominator='same original parent full bbox diagonal',
                    coordinate_RMS='sqrt(mean across all free face-corner coordinates))',
                    note='Independent surface difference is not same-slot difference, own GT score, semantic label, or certified ODE error'),
                scope='Four intentionally chosen observed cases; no population/generalization estimate; no solver selected by GT quality')
    write_json(out/'analysis.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--no-figures',action='store_true')
    a=p.parse_args(argv);r=analyze(a.out,not a.no_figures)
    print({'status':r['status'],'counts':r['counts'],'CPU_analysis_seconds':r['CPU_analysis_seconds']},flush=True)

if __name__=='__main__':main()
