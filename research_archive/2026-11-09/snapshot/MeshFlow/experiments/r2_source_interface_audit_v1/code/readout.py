from pathlib import Path
import csv,json,collections,hashlib,sys
OUT=Path(__file__).resolve().parents[1]
MINI=OUT.parents[1]/'deliverables/R2_32_67_ANALYSIS_MINI_V1'
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rr):
    rr=list(rr)
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
src=[r for r in rows(OUT/'source_interfaces.csv') if r['topology_domain']=='SOURCE_INDEX' and r['direction']=='EXPORT_ORIENTED']
idx={r['condition_id']:r for r in src}
ms=rows(MINI/'metrics_per_output.csv')
for stage in ['RAW','POST']:
    print(stage, {l:sum(float(r['C_F_has_valid_edge'])>0 for r in ms if r['panel']=='NEW32' and r['stage']==stage and r['label']==l) for l in sorted({r['label'] for r in ms})})
comp=[]
for kind in ['SEAT_FULL','SEAT_HALF','BACK_FULL','BACK_HALF']:
    cr=[r for r in src if r['parent_role']=='NEW_DEV32' and r['condition_type']==kind]
    for label in ['USER_SMALL','USER_EXPANDED_67','HIST_USER_MIX_FROZEN']:
        rr=[r for r in ms if r['panel']=='NEW32' and r['stage']=='POST' and r['label']==label and r['region_type']+'_'+r['tier']==kind]
        comp.append(dict(kind='DERIVED',condition_type=kind,label=label,source_conditions=len(cr),
            source_with_valid_interface=sum(float(r['C_F_has_valid_edge'])>0 for r in cr),generated_requests=len(rr),
            generated_with_valid_interface=sum(float(r['C_F_has_valid_edge'])>0 for r in rr),
            source_strict_perfect_conditions=sum(float(r['perfect']) for r in rows(OUT/'AVMR_PMR_all8.csv') if r['condition_id'] in {c['condition_id'] for c in cr} and r['setting']=='raw_tau_1e-6'),
            source_direction_reach_parent_equal=sum(float(r['opposite_winding_reached_free_area_fraction']) for r in cr)/len(cr),
            generated_direction_reach_parent_equal=sum(float(r['opposite_winding_reached_free_area_fraction']) for r in rr)/len(rr)))
writecsv(OUT/'DEV_task_comparison.csv',comp)
incomplete=[]
for r in rows(OUT/'AVMR_PMR_all8.csv'):
    if r['setting']=='raw_tau_1e-6' and r['perfect']=='0':
        s=idx[r['condition_id']]
        incomplete.append(dict(r,source_valid_edges=s['C_F_valid_edge_count'],source_nonmanifold_CF_edges=s['C_F_nonmanifold_edge_count'],
            reference_boundary_not_user_required=True))
writecsv(OUT/'source_strict_incomplete_conditions.csv',incomplete)
stats={}
for role in ['COMMON_SMALL32','EXP_ONLY35','NEW_DEV32']:
    rr=[r for r in src if r['parent_role']==role]
    stats[role]=dict(conditions=len(rr),legal_interface=sum(float(r['C_F_has_valid_edge'])>0 for r in rr),
        total_valid_edges=sum(int(r['C_F_valid_edge_count']) for r in rr),
        nonmanifold_CF_conditions=sum(int(r['C_F_nonmanifold_edge_count'])>0 for r in rr),
        duplicate_rejected_CF_conditions=sum(int(r['C_F_rejected_duplicate_edge_count'])>0 for r in rr),
        whole_nonmanifold_conditions=sum(int(r['nonmanifold_edges'])>0 for r in rr),
        source_strict_incomplete_conditions=sum(r['parent_role']==role for r in incomplete),
        min_direction_reach=min(float(r['opposite_winding_reached_free_area_fraction']) for r in rr))
differences=[]
allr={(r['condition_id'],r['topology_domain'],r['direction']):r for r in rows(OUT/'source_interfaces.csv')}
for cid in idx:
    si=allr[cid,'SOURCE_INDEX','EXPORT_ORIENTED'];ec=allr[cid,'EXACT_COORD_UINT32','EXPORT_ORIENTED']
    for f in ['C_F_valid_edge_count','C_F_shared_edge_count','C_F_nonmanifold_edge_count','reached_free_area_fraction']:
        if float(si[f])!=float(ec[f]):differences.append(dict(condition_id=cid,field=f,source=si[f],coord=ec[f]))
flipcases=[dict(condition_id=r['condition_id'],flips=r['orientation_flips']) for r in src if int(r['orientation_flips'])]
readout=dict(status='DERIVED',groups=stats,source_vs_exact_coordinate_structural_differences=differences,
    source_orientation_changed_conditions=flipcases,strict_incomplete_count=len(incomplete),new_geometry_in_this_script=0)
(OUT/'readout_summary.json').write_text(json.dumps(readout,indent=2)+'\n')
print(json.dumps(readout,indent=2))
print(json.dumps(comp,indent=2))
