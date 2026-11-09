"""Derived tables and numerical closeout, without another geometry score."""
import collections
import csv
import json
from pathlib import Path
from r2.pilots.seam_pool47_couple_v1 import cpu_binding as R, dev_diagnostic as D
C,np=R.C,R.np

def readcsv(path):
    return list(csv.DictReader(C.resolve(path).open(encoding='utf-8-sig')))

def number(value):
    return float(value == 'True') if value in ('True','False') else float(value)

def run():
    done=C.read(R.OUT/'completion.json');reg=C.read(R.OUT/'registration.json')
    R.CORE['verify'](reg['protected']);R.CORE['verify'](reg['code'])
    rows=readcsv(R.OUT/'per_output.csv');old=readcsv(R.BASE/'evaluation/per_output.csv')
    baseline={r['request_id']:r for r in old if r['label']=='B_LONG_LAYOUT' and r['stage']=='POST'}
    paired=readcsv(R.OUT/'paired_metrics.csv');groups=collections.defaultdict(list)
    for r in paired:groups[(r['label'],r['parent_id'],r['split'],r['metric'])].append(r)
    parents=[]
    for key,rr in sorted(groups.items()):
        parents.append(dict(zip(('label','parent_id','split','metric'),key),n=len(rr),
            baseline_mean=float(np.mean([number(r['baseline']) for r in rr])),
            candidate_mean=float(np.mean([number(r['candidate']) for r in rr])),
            delta_mean=float(np.mean([float(r['delta']) for r in rr]))))
    C.csvwrite(R.OUT/'paired_per_parent.csv',parents)
    # Complete the requested side-to-bulk-open-vertex observation independently
    # of the frozen binding. No candidate selection or score is changed.
    sides=readcsv(D.OUT/'lateral_edge_distances.csv');byreq=collections.defaultdict(list)
    for r in sides:byreq[r['request_id']].append(r)
    supplement=[]
    for request,rr in byreq.items():
        mesh=C.arrays(R.BASE/'evaluation/B_LONG_LAYOUT'/request/'post/post.npz')
        v=mesh['vertices'].astype(float);f=mesh['faces'];K=int(mesh['known_faces']);sf=mesh['source_face_ids']
        seam=mesh['raw_seam_slot_mask'][sf];fixed=set(map(int,f[:K].ravel()))
        bulk=[e for e,occ in R.B.incidence(f).items() if len(occ)==1 and occ[0][0]>=K and not seam[occ[0][0]]]
        vertices=sorted({i for e in bulk for i in e if i not in fixed})
        for row in rr:
            edge=json.loads(row['seam_edge']);sx=v[edge];choices=[]
            for vertex in vertices:
                dist,t,foot=D.point_segments(v[vertex],sx[None]);choices.append((float(dist[0]),vertex,float(t[0])))
            near=min(choices) if choices else None
            options=[]
            for be in bulk:
                for target in (be,be[::-1]):
                    displacement=max(float(np.linalg.norm(v[a]-v[b])) for a,b in zip(edge,target))
                    x=sx[1]-sx[0];y=v[be[1]]-v[be[0]];den=np.linalg.norm(x)*np.linalg.norm(y)
                    cos=abs(float(x@y/den)) if den else 0.
                    options.append((displacement,tuple(be),tuple(target),float(np.degrees(np.arccos(np.clip(cos,0,1))))))
            nearest=min(options) if options else None
            supplement.append(dict(request_id=request,parent_id=row['parent_id'],raw_slot_id=int(row['raw_slot_id']),
                edge_id=int(row['edge_id']),tip_state=row['tip_state'],seam_edge=edge,lateral_C_corner=int(row['lateral_C_corner']),
                nearest_free_bulk_open_vertex=near[1] if near else None,side_to_bulk_vertex_L=near[0]/C.L if near else None,
                nearest_point_parameter_on_side=near[2] if near else None,
                minimum_unconstrained_endpoint_dmax_L=nearest[0]/C.L if nearest else None,
                endpoint_nearest_bulk_edge=list(nearest[1]) if nearest else [],endpoint_nearest_mapping=list(nearest[2]) if nearest else [],
                endpoint_nearest_axis_angle_deg=nearest[3] if nearest else None,
                interpretation='Representative endpoint distance only; not the full-member move cap, candidate admission or shared-edge success'))
    C.csvwrite(D.OUT/'lateral_vertex_endpoint_supplement.csv',supplement)
    tables=[]
    for label in sorted({r['label'] for r in rows}):
        for split in ('TRAIN','DEV'):
            rr=[r for r in rows if r['label']==label and r['split']==split]
            out=dict(label=label,split=split,n=len(rr),
                baseline_bulk30=sum(float(baseline[r['request_id']]['free_only_reached_bulk_area_fraction'])>=.3 for r in rr),
                bound_bulk30=sum(float(r['free_only_reached_bulk_area_fraction'])>=.3 for r in rr))
            for key in ('free_only_reached_bulk_area_fraction','FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction',
                'ConflictCov_C','free_minimum_angle_lt5_fraction','C_F_valid_edge_count','actual_free_faces'):
                out[key]=float(np.mean([float(r[key]) for r in rr]));out['delta_'+key]=float(np.mean([float(r[key])-float(baseline[r['request_id']][key]) for r in rr]))
            for key in ('accepted_tip','accepted_edge','CF_lost_legal_edges','FF_lost_legal_edges','FF_new_illegal_pairs','FF_illegal_pairs_removed','deleted_source_faces'):
                out['sum_'+key]=sum(int(r[key]) for r in rr)
            tables.append(out)
    C.csvwrite(R.OUT/'key_results.csv',tables)
    tips=readcsv(D.OUT/'tip_distances.csv');tip_summary=[]
    for parent in sorted({r['parent_id'] for r in tips}):
        for state in ('FREE','C_ANCHORED'):
            rr=[r for r in tips if r['parent_id']==parent and r['tip_state']==state]
            z=dict(parent_id=parent,tip_state=state,n=len(rr),already_same_bulk_vertex=sum(float(r['tip_to_free_bulk_vertex_L'])<1e-12 for r in rr))
            for key in ('tip_to_free_bulk_vertex_L','tip_to_finite_bulk_edge_L','minimum_tip_full_cluster_motion_L'):
                vals=[float(r[key]) for r in rr if r[key]]
                z[key+'_mean']=float(np.mean(vals)) if vals else None
                z[key+'_le015']=sum(x<=.015 for x in vals);z[key+'_le020']=sum(x<=.02 for x in vals)
            tip_summary.append(z)
    C.csvwrite(D.OUT/'tip_summary_by_parent_state.csv',tip_summary)
    funnel=readcsv(D.OUT/'lateral_threshold_funnel.csv');fg=collections.defaultdict(collections.Counter)
    keys=('TIP_C_ANCHORED_INELIGIBLE','SEAM_LATERAL_NOT_OPEN','AXIS_GE30','NO_FREE_IDENTITY_CORRESPONDENCE',
        'REPRESENTATIVE_GAP_GT_2DELTA','ALL_MEMBER_MOTION_OVER_DELTA','GEOMETRIC_CANDIDATE_BEFORE_INCREMENTAL_GUARDS')
    for r in funnel:
        g=fg[(r['parent_id'],r['delta_fraction'])];g['side_rows']+=1;g['pairs']+=int(r['bulk_open_edge_count'])
        for k in keys:g[k]+=int(r[k]) if r.get(k) else 0
    C.csvwrite(D.OUT/'lateral_funnel_by_parent.csv',[dict(parent_id=k[0],delta_fraction=k[1],**v) for k,v in sorted(fg.items())])
    alias=readcsv(D.OUT/'alias_slot_fates.csv')
    text=['# B_LONG_LAYOUT CPU binding: numerical result', '',
        f"Completed {done['cells']}/56 cells in {done['seconds']:.2f} s with {done['workers']} CPU workers; zero model calls, zero new noise/query draws. All {done['protected_verified']} protected inputs verified. Original 124 numeric metric fields retained. Visual review remains pending.", '',
        'The 1.5% and 2.0% caps add local TRAIN bulk connectivity but do not create a DEV output with >=30% reachable bulk. Point aliases alone are not interface success.', '',
        '| Cap / split | Bulk >=30% before→after | Mean bulk reach | ΔFF pp | ΔI pp | Accepted tip / lateral | New / removed illegal FF pairs |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for t in tables:
        text.append(f"| {t['label'].split('_')[-1]} / {t['split']} | {t['baseline_bulk30']}→{t['bound_bulk30']}/{t['n']} | {100*t['free_only_reached_bulk_area_fraction']:.4f}% | {100*t['delta_FF_illegal_affected_free_fraction']:+.4f} | {100*t['delta_ConflictCov_C']:+.4f} | {t['sum_accepted_tip']} / {t['sum_accepted_edge']} | {t['sum_FF_new_illegal_pairs']} / {t['sum_FF_illegal_pairs_removed']} |")
    text += ['', 'All 56 cells preserve every old legal C/F and F/F edge. No additional source face is deleted, and no new illegal C/F pair is introduced. These safeguards do not prohibit new illegal F/F pairs: DEV has 19 and 20 new pairs at the two caps. Mean CF/I improvements elsewhere cannot cancel this cost.', '',
        'TRAIN additions crossing 30% are 73/n0 and 73/n3. 73/n1 remains 21.89%; 85/n0 remains zero. DEV 608 and 610 remain zero reachable bulk; 2d/n0 and n2 have only 0.983% and 3.709%. Both DEV caps accept point aliases only, no complete lateral-edge candidate survives.', '',
        'DEV has 16/4/4 activated and retained seam faces per 2d/610/608 request, not eight per object. All 96 retained tips are currently FREE; zero C_ANCHORED tips is an observed count, not a skipped category.', '',
        '| Parent | Tips | Nearest free bulk vertex ≤2%L | Finite bulk edge ≤2%L | Distinct tip-cluster minimum motion ≤2%L |',
        '|---|---:|---:|---:|---:|']
    for r in tip_summary:
        if r['tip_state']=='FREE':text.append(f"| {r['parent_id'][:8]} | {r['n']} | {r['tip_to_free_bulk_vertex_L_le020']} | {r['tip_to_finite_bulk_edge_L_le020']} | {r['minimum_tip_full_cluster_motion_L_le020']} |")
    text += ['', 'Nearest point distances can be zero because the seam tip already shares a bulk boundary vertex. They do not establish a shared edge. Minimum distinct-cluster motion excludes the already identical vertex and additionally includes every original PRE_GEO cluster member; its denominator therefore answers a different question.', '',
        f"The three 2d source aliases produce {len(alias)} raw slots across four noises. {sum(r['retained']=='True' for r in alias)} survive original POST. Their source-alias designation is not a true Null label. Detailed illegal-pair and legal-free-component records are in dev_diagnostic/alias_slot_fates.csv and alias_illegal_pairs.csv; this association is not evidence that aliases caused failed bulk connection.", '',
        'The full threshold funnel separates closed lateral edges, axis mismatch, C/identity restrictions, representative gap and original-corner displacement. The finite-segment and endpoint distances use finite geometry, not infinite lines. Supplementary side-to-vertex observations add a missing requested diagnostic without changing, rerunning or selecting binding cells.', '',
        'Interpretation: the saved B_LONG_LAYOUT endpoint benefits locally on TRAIN; this fixed binding does not solve DEV seam-to-bulk continuity. Widening from 1.5% to 2.0% adds DEV aliases and FF cost with essentially unchanged reach. No model or default is promoted by this CPU result.', '']
    path=R.OUT/'NUMERICAL_REPORT.md'
    if path.exists():raise ValueError('Refuse overwrite closeout')
    path.write_text('\n'.join(text),encoding='utf-8')
    C.write(R.OUT/'numerical_report_receipt.json',dict(status='NUMERICAL_COMPLETE_VISUAL_PENDING',
        code_sha256=C.sha(__file__),binding_completion_sha256=C.sha(R.OUT/'completion.json'),diagnostic_completion_sha256=C.sha(D.OUT/'completion.json'),
        files={str(p):C.sha(p) for p in (path,R.OUT/'paired_per_parent.csv',R.OUT/'key_results.csv',D.OUT/'lateral_vertex_endpoint_supplement.csv',D.OUT/'tip_summary_by_parent_state.csv',D.OUT/'lateral_funnel_by_parent.csv')},
        model_calls=0,new_binding_cells=0,new_quality_scans=0))
    print('CPU numerical report complete; visual review pending')

if __name__=='__main__':run()
