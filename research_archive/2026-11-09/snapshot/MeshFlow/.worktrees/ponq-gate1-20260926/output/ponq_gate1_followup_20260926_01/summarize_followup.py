"""Read completed immutable studies; write derived tables and static figures."""
import argparse, csv, hashlib, json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

def read(p): return json.loads(p.read_text())
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_write(p, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def geometry_ok(r):
    return bool(r.get('mesh_present') and r.get('geometry_audited') and
                r.get('closed_oriented_2manifold') and r.get('betti_match') and
                r.get('exact_intersection_pairs')==0 and r.get('exact_degenerate_faces')==0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out.resolve();out.mkdir(exist_ok=False)
    roots={s:a.parent/f'ponq_gate1_{s}_20260926_01' for s in ['synthetic','real','isolation']}
    source={s:read(p/'LEDGER.json') for s,p in roots.items()}
    rows=[]
    for study,rs in source.items():
        extracted=[r for r in rs if r.get('extraction_attempts',0)>0]
        assert len(extracted)==dict(synthetic=36,real=36,isolation=20)[study]
        for r in extracted:
            row=dict(study=study,object=r['object'],state=r['state'],budget=r['budget'],
                     original_status=r['status'],geometry_and_topology_ok=geometry_ok(r),
                     all_candidate_scores_finite=r['nonfinite_cut_weights']==0,
                     all_candidate_scores_nonnegative=r['negative_cut_weights']==0)
            for key in ['faces','vertices','rms_pct_bboxdiag','p95_pct_bboxdiag','angle_min_deg','angle_q05_deg',
                        'beta0','beta1','beta2','closed_oriented_2manifold','betti_match','nonmanifold_edges',
                        'nonmanifold_vertices','exact_intersection_pairs','exact_degenerate_faces',
                        'nonfinite_cut_weights','negative_cut_weights','minimum_cut_weight',
                        'position_shift_over_h_mean','position_shift_over_h_max','extraction_seconds',
                        'metric_seconds','exact_audit_seconds','mesh_sha256','reference_clean_mesh_byte_equal']:
                row[key]=r.get(key)
            row['run_path']=str(roots[study]/'runs'/f"{r['object']}__{r['state']}")
            rows.append(row)
    assert len(rows)==92 and len(set((r['study'],r['object'],r['state']) for r in rows))==92
    dump(out/'LEDGER_92.json',rows);csv_write(out/'LEDGER_92.csv',rows)
    grouped=[]
    for study in roots:
        rr=[r for r in rows if r['study']==study]
        for state in ['ALL']+list(dict.fromkeys(r['state'] for r in rr)):
            group=rr if state=='ALL' else [r for r in rr if r['state']==state]
            grouped.append(dict(study=study,state=state,n=len(group),
                geometry_and_topology_ok=sum(r['geometry_and_topology_ok'] for r in group),
                nonfinite_candidate_score_rows=sum(not r['all_candidate_scores_finite'] for r in group),
                negative_candidate_score_rows=sum(not r['all_candidate_scores_nonnegative'] for r in group),
                candidate_finite_nonnegative_rows=sum(r['all_candidate_scores_finite'] and r['all_candidate_scores_nonnegative'] for r in group),
                rms_pct_bboxdiag_median=float(np.median([r['rms_pct_bboxdiag'] for r in group])),
                faces_min=min(r['faces'] for r in group),faces_max=max(r['faces'] for r in group)))
    dump(out/'GROUPS.json',grouped);csv_write(out/'GROUPS.csv',grouped)
    index={(s,r['object'],r['state']):r for s,rs in source.items() for r in rs}
    comparisons=[]
    for obj in ['sphere','box','thin_u','torus']:
        for level in ['light','strong']:
            clean=index['synthetic',obj,'clean256'];full=index['synthetic',obj,'qem_'+level]
            fixed=index['isolation',obj,'fixed_v_metric_'+level]
            pos=index['isolation',obj,'position_matched_q_'+level]
            comparisons.append(dict(object=obj,level=level,clean_rms=clean['rms_pct_bboxdiag'],
                full_qem_rms=full['rms_pct_bboxdiag'],fixed_v_metric_rms=fixed['rms_pct_bboxdiag'],
                matched_position_rms=pos['rms_pct_bboxdiag'],full_qem_geometry_ok=geometry_ok(full),
                fixed_v_metric_geometry_ok=geometry_ok(fixed),matched_position_geometry_ok=geometry_ok(pos),
                clean_faces=clean['faces'],full_qem_faces=full['faces'],fixed_v_metric_faces=fixed['faces'],
                matched_position_faces=pos['faces'],full_qem_shift_h_mean=full['position_shift_over_h_mean'],
                matched_position_shift_h_mean=pos['position_shift_over_h_mean'],
                fixed_v_mesh_byte_equal=bool(fixed['reference_clean_mesh_byte_equal'])))
    csv_write(out/'ISOLATION_COMPARISON.csv',comparisons);dump(out/'ISOLATION_COMPARISON.json',comparisons)
    original=read(roots['synthetic']/'audit/ORIGINAL_COMPARISON.json')
    replication=dict(n=len(original),mesh_byte_equal=sum(r['mesh_byte_equal'] for r in original),
        fields_differing_rows={k:sum(not r['fields'][k]['equal'] for r in original) for k in original[0]['fields']},
        maximum_absolute_rms_difference=max(abs(r['fields']['rms_pct_bboxdiag']['local']-r['fields']['rms_pct_bboxdiag']['original']) for r in original))
    dump(out/'REPLICATION_COMPARISON.json',replication)
    summaries={s:read(p/'SUMMARY.json') for s,p in roots.items()}
    timings={s:dict(fitting_seconds=d['fitting_seconds_total'],extraction_seconds=d['extraction_seconds_total'],
                   metric_seconds=d['metric_seconds_total'],exact_audit_seconds=d['exact_audit_seconds_total'],
                   stages_wall_seconds=sum(t['wall_seconds'] for t in read(roots[s]/'EXECUTION.json')['stages'])) for s,d in summaries.items()}
    dump(out/'TIMINGS.json',timings)
    # Geometric surfaces, same camera and reference-derived scale across each row.
    objects=[(s,obj) for s in ['synthetic','real'] for obj in dict.fromkeys(r['object'] for r in source[s] if r.get('extraction_attempts',0)>0)]
    fig=plt.figure(figsize=(12,23),layout='constrained')
    for i,(study,obj) in enumerate(objects):
        root=roots[study];ref=trimesh.load(root/'inputs'/obj/'reference.obj',process=False)
        center=ref.bounds.mean(0);extent=max(np.ptp(ref.vertices,axis=0))*.64
        for j,state in enumerate(['reference','clean256','qem_strong']):
            ax=fig.add_subplot(8,3,i*3+j+1,projection='3d')
            p=root/'inputs'/obj/'reference.obj' if state=='reference' else root/'runs'/f'{obj}__{state}'/'mesh.obj'
            mesh=trimesh.load(p,process=False);rr=None if state=='reference' else index[study,obj,state]
            color='#8bbac6' if rr is None or geometry_ok(rr) else '#d58376'
            poly=Poly3DCollection(mesh.vertices[mesh.faces],facecolor=color,edgecolor='#3b4750',linewidth=.22,alpha=1)
            ax.add_collection3d(poly)
            for setter,c in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],center):setter(c-extent,c+extent)
            ax.set_box_aspect((1,1,1));ax.view_init(22,35);ax.set_axis_off()
            title=f'{obj} | {state}\nF={len(mesh.faces)}'
            if rr:
                title+=f" | RMS={rr['rms_pct_bboxdiag']:.4f}%"
                if not geometry_ok(rr):title+=' | TOPOLOGY FAIL'
                if rr['nonfinite_cut_weights'] or rr['negative_cut_weights']:title+='\nCANDIDATE SCORE WARNING'
            ax.set_title(title,fontsize=9)
    fig.suptitle('PoNQ Gate 1: reference, clean K=256, full-Q strong perturbation\nRed = topology failure. Warnings are retained; geometry is not a numerical-validity certificate.',fontsize=12)
    fig.savefig(out/'GEOMETRY_GALLERY.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(15,4),layout='constrained')
    for ax,obj in zip(axes,['sphere','box','thin_u','torus']):
        cc=[r for r in comparisons if r['object']==obj]
        clean=cc[0]['clean_rms'];ax.axhline(clean,color='gray',ls='--',label='Clean')
        for key,label,color in [('full_qem_rms','Full-Q original','#c85c45'),('matched_position_rms','Matched position','#3173ac'),('fixed_v_metric_rms','Fixed-v local Q','#418a55')]:
            ax.plot([0,1],[r[key] for r in cc],'o-',color=color,label=label)
        ax.set_xticks([0,1],['Light','Strong']);ax.set_title(obj);ax.set_ylabel('Surface RMS (% bbox diagonal)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Parameter controls: matched positions use the exact original QEM displacement\nFixed-v local-Q doses are a different intervention; curves are not an additive error decomposition.',fontsize=11)
    fig.savefig(out/'ISOLATION_RMS.png',dpi=160);plt.close(fig)
    metadata=dict(studies={s:dict(path=str(p),ledger_sha256=sha(p/'LEDGER.json'),protocol_sha256=sha(p/'FROZEN_PROTOCOL.json')) for s,p in roots.items()},
        actual_extractions=92,original_protocol_extractions=72,additional_controls=20,network_training_updates=0,
        candidate_score_warning='All-candidate face scores, not necessarily graph-edge capacities; see separate numerical diagnostic.',
        statuses_warning='Original run_gate1 checks finiteness; isolation also checks nonnegativity. Preserve raw status, do not pool it as a uniform success metric.',
        geometry_definition='saved exact-audited mesh; closed oriented 2-manifold; expected Betti; zero exact intersections and degeneracies. No RMS/angle acceptance threshold invented.',
        replication=replication,timings=timings)
    dump(out/'RESULT_NUMBERS.json',metadata)
    shutil.copy2(__file__,out/'summarize_followup.py')
    print(json.dumps(dict(out=str(out),rows=len(rows),groups=grouped,replication=replication),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
