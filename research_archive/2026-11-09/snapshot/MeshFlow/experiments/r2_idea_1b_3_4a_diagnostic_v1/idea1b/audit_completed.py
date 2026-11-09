"""Read-only numerical consistency audit of completed Idea1B artifacts; no scoring."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,sys
from pathlib import Path
from collections import Counter
import numpy as np
import run as R
A,Q,L,OUT=R.A,R.Q,R.L,R.OUT
records=[];rows=A.csv_rows(OUT/'per_output.csv');optimizers=[];changes=[]
for row in R.rows():
    short=row['short_id'];dest=OUT/'requests'/short;a=A.load_asset(row);k=row['K']
    opt=A.read(dest/'optimization.json');history=A.csv_rows(dest/'objective_history.csv');optimizers.append(opt)
    assert len(history)==opt['actual_nfev']<=300
    assert sum(h['feasible']=='False' for h in history)==sum(opt['rejected_feasibility'].values())
    feasible=[float(h['value_model']) for h in history if h['feasible']=='True']
    assert abs(min(feasible)*100/L-opt['affine_distance_pctL'])<1e-10
    af=A.arrays(dest/'AFFINE/post.npz');bound=A.arrays(dest/'AFFINE_B2/post.npz');pre=A.arrays(dest/'affine_pre_geo.npz')['output']
    assert np.array_equal(af['faces'],a['restored_faces'])
    assert np.array_equal(af['source_face_ids'],a['source_face_ids'])
    assert np.array_equal(af['source_corner_ids'],a['post']['source_corner_ids'])
    cids=np.unique(a['faces'][:k]);fids=np.unique(a['faces'][k:])
    assert af['vertices'][cids].tobytes()==a['vertices'][cids].tobytes()
    assert af['output'][:k].tobytes()==a['C_original'].tobytes()==bound['output'][:k].tobytes()
    assert np.linalg.norm(af['vertices'][fids].astype(float)-a['vertices'][fids],axis=1).max()<=.035*L+1e-10
    assert np.linalg.norm(pre[k:].astype(float)-a['pre_geo'][k:],axis=2).max()<=.035*L+1e-10
    ids=bound['assignment_input_corner_ids'];targets=bound['assignment_target_C_vertex_ids']
    if len(ids):
        assert np.linalg.norm(pre.reshape(-1,3)[ids].astype(float)-bound['vertices'][targets],axis=1).max()<=.02*L+1e-10
        assert np.linalg.norm(a['pre_geo'].reshape(-1,3)[ids].astype(float)-bound['vertices'][targets],axis=1).max()<=.055*L+1e-10
    oldpairs=A.read(A.resolve(row['score_path']).parent/'pairs.json')
    sa=A.read(dest/'AFFINE/score.json');sb=A.read(dest/'AFFINE_B2/score.json')
    oldff=Q.illegal_pair_sets(oldpairs,a['source_face_ids'],k)['FF']
    affff=Q.illegal_pair_sets(sa['pairs'],af['source_face_ids'],k)['FF']
    assert oldff==affff
    inc=A.read(dest/'AFFINE_B2/vs_affine_ledger.json')['summary'];assert inc['CF_new_illegal_pairs']==0
    originalB=A.read(A.OUT/'cells/B_020'/short/'complete.json')['row']
    after=next(r for r in rows if r['short_id']==short and r['label']=='AFFINE_B2')
    changes.append(dict(short_id=short,split=row['split'],raw_adoption_inherited=row['raw_adoption'],
        original_B2_joint=int(originalB['screening_joint']),affine_B2_joint=int(after['screening_joint']),
        new_vs_original_B2=int(after['screening_joint']) and not int(originalB['screening_joint']),
        lost_vs_original_B2=int(originalB['screening_joint']) and not int(after['screening_joint'])))
    records.append(dict(short_id=short,status='PASS',objective_evaluations=len(history),
        preserved_C_bitwise=True,affine_indexed_topology_unchanged=True,FF_illegal_source_pair_set_unchanged=True,
        all_affine_caps_pass=True,B2_transformed_pre_cap_pass=True,B2_total_bound_5p5_cap_pass=True,
        B2_zero_incremental_CF_pass=True))
maxima={key:max(float(r[key]) for r in rows) for key in ('affine_POST_max_pctL','affine_pre_geo_max_pctL','B2_bound_max_from_transformed_pre_pctL','B2_bound_total_from_original_pre_pctL','final_retained_max_from_original_pre_pctL')}
counts=Counter(str(o['status']) for o in optimizers);rejected=Counter()
for o in optimizers:rejected.update(o['rejected_feasibility'])
result=dict(status='PASS',requests=24,rows=records,scoring_calls=0,model_calls=0,GPU_calls=0,
    maximum_objective_evaluations=max(o['actual_nfev'] for o in optimizers),
    total_objective_evaluations=sum(o['actual_nfev'] for o in optimizers),
    optimizer_status_counts=dict(counts),rejected_feasibility=dict(rejected),
    objective_improved_requests=sum(o['affine_distance_pctL']<o['baseline_distance_pctL']-1e-10 for o in optimizers),
    no_affine_dof_requests=sum(o['reduced_dof']==0 for o in optimizers),displacement_maxima=maxima,
    screening_vs_B2=changes,FF_affine_exact_illegal_pair_set_same_24_of24=True,
    limitations='Audit recomputes only array identities/displacements and already-saved discrete pair sets; no independent intersection classifier, new scores or actual image labels.')
R.write(OUT/'consistency_audit.json',result);R.table(OUT/'paired_screening_vs_old_B2.csv',changes)
special=next(r for r in rows if r['label']=='AFFINE_B2' and r['short_id']=='2d08a64e_n1')
text=['# Additional numerical findings (no rescoring)','',
f"All 24 request consistency checks PASS. Total objective calls {result['total_objective_evaluations']}, status counts {dict(counts)}, feasibility rejections {dict(rejected)}.",
f"Objective improved {result['objective_improved_requests']}/24; {result['no_affine_dof_requests']}/24 are fully constrained to identity by existing C/F shared anchors.",
'All eight 608/610 optimizations reach the fixed 300-call budget. They all improve the distance objective but none yields a legal C/F edge; budget exhaustion does not prove optimality.',
'608 n3 creates one in-budget cluster candidate (1.942282%L), rejected by ANGLE_NEW_OR_WORSE on its affected neighborhood. Other seven have zero in-budget unoccupied cluster candidates.',
'',
'Relative to the original B2 output, the sole new screening joint request is DEV 2d n1. Its one new interface is not a zero-cost success: one new illegal CF pair appears during affine motion (offset by another disappearance in CF affected-face fraction), and B2 adds one illegal FF pair. FF affected-free fraction rises 1.612903 pp vs original POST; I drops .097656 pp. New POST semantic review remains pending.',
'',
'Global affine alone preserves all 24 original illegal FF source-pair sets exactly under the actual scorer; no tolerance-driven FF classification change was observed. B2 has separate incremental ledgers and remains zero-new-CF relative to the affine input, not necessarily relative to original POST.',
'',
f'Displacement maxima across requests: {maxima}. Bound-corner total <=5.5%L follows the two separate caps; final retained coordinates relative to pre_geo also contain prior GEO consolidation and must not be mislabeled as B2-only displacement.',
'',
'Fixed selection gallery has 16 pages, each original POST/AFFINE/AFFINE+B2 with common camera and scale. No automatic new adoption/body labels were produced.']
(OUT/'NUMERICAL_FINDINGS.md').write_text('\n'.join(text)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','screening_vs_B2')},indent=2))
