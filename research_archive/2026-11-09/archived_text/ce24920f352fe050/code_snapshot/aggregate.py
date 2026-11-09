"""Deterministic explanatory summaries; never replace or rescore inherited metrics."""
import csv,json
import numpy as np
from . import common as c

def rows(path):return list(csv.DictReader(c.resolve(path).open(encoding='utf-8-sig')))
def wquant(x,w,p):
    ok=np.isfinite(x)&np.isfinite(w)&(w>0)
    if not ok.any():return None
    x=x[ok];w=w[ok];order=np.argsort(x,kind='stable');x=x[order];w=w[order]
    return float(x[min(np.searchsorted(np.cumsum(w),p*w.sum(),side='left'),len(x)-1)])
def main():
    reg=c.read(c.OUT/'registration.json');out=[];pair_contacts=[]
    for cell in reg['cells']+reg['references']:
        d=c.OUT/'diagnostics'/cell['cell_id'];s=c.read(d/'summary.json');v=c.arrays(d/'query_geometry.npz');h=c.arrays(d/'normal_line_hits.npz')
        w=v['query_weights'];den=w.sum();dup=v['reconstructed_dup'];gc=v['corridor_layer_count'];signed=v['signed_nearest_parallel_point_offset']/s['L']
        fields=['cell_id','short_id','request_id','parent_id','condition_id','label','stage','selection_reason','K','N']
        r={k:cell.get(k) for k in fields};r.update(dup_parity_status=s['old_dup_comparison']['status'],query_denominator=float(den),
            new_explanatory_dup_proxy_fraction=float(w@dup/den),near_parallel_candidate_F_count=len(s['near_parallel_free_indices']),
            corridor_one_group_fraction=float(w@(gc==1)/den),corridor_two_or_more_groups_fraction=float(w@(gc>=2)/den),
            corridor_no_group_fraction=float(w@(gc==0)/den),dup_and_two_or_more_groups_fraction=float(w@(dup&(gc>=2))/den),
            exact_observed_CF_edges=s['graph']['exact_CF_edges'],exact_observed_free_components=s['graph']['exact_free_components'],
            nearest_parallel_signed_median_L=wquant(signed,w,.5),dup_nearest_parallel_signed_p05_L=wquant(signed,w*dup,.05),
            dup_nearest_parallel_signed_median_L=wquant(signed,w*dup,.5),dup_nearest_parallel_signed_p95_L=wquant(signed,w*dup,.95),
            finite_normal_line_hits=s['isolated_normal_hits'])
        neg=np.zeros(len(w),bool);pos=neg.copy()
        neg[h['query_id'][h['in_corridor']&(h['signed_t']<0)]]=True;pos[h['query_id'][h['in_corridor']&(h['signed_t']>0)]]=True
        r.update(corridor_both_normal_sides_fraction=float(w@(neg&pos)/den))
        walls=rows(d/'sidewall_adjacency.csv')
        r['exact_C_to_parallel_sidewall_candidates']=sum(x['candidate_C_to_parallel']=='True' for x in walls)
        r['exact_parallel_to_parallel_sidewall_candidates']=sum(x['candidate_parallel_to_parallel']=='True' for x in walls)
        if cell.get('score_path'):
            old=c.read(cell['score_path']);m=old['metrics'];t=old['topology']
            for k in ['ConflictCov_C','ConflictCov_C_cross','ConflictCov_C_dup','ConflictCov_C_cross_and_dup','CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','CF_proper_pair_count','FF_proper_pair_count']:
                r['inherited_'+k]=m[k]
            for k in ['C_F_valid_edge_count','C_F_valid_opposite_winding_edge_count','opposite_winding_reached_free_area_fraction','exact_duplicate_faces']:
                r['inherited_'+k]=t[k]
            assert abs(r['inherited_ConflictCov_C_dup']-r['new_explanatory_dup_proxy_fraction'])<1e-12
            contacts=c.read(cell['pairs_path']);near=set(s['near_parallel_free_indices'])
            touch=both=0
            for contact in contacts['FF']['contacts']:
                a,b=map(int,contact['pair']);istouch=a in near or b in near;isboth=a in near and b in near
                proper=contact['classification']=='proper_transverse_penetration'
                if proper:touch+=istouch;both+=isboth
                pair_contacts.append(dict(cell_id=cell['cell_id'],scope='FF',free_face_a=a,free_face_b=b,classification=contact['classification'],
                    a_is_near_parallel_candidate=a in near,b_is_near_parallel_candidate=b in near,proper_pair_touches_candidate=proper and istouch,
                    source_pairs_path=cell['pairs_path'],source_pairs_sha256=cell['pairs_sha256']))
            r.update(inherited_FF_proper_pairs_touching_near_candidate=touch,inherited_FF_proper_pairs_both_near_candidates=both)
        out.append(r)
    c.csvwrite(c.OUT/'per_output_layer_diagnosis.csv',out)
    c.csvwrite(c.OUT/'inherited_FF_contacts_with_candidate_flags.csv',pair_contacts)
    by={(r['short_id'],r['label'],r['stage']):r for r in out if r['stage']!='SOURCE'};paired=[]
    for request in reg['requests']:
        for stage in ['RAW','POST']:
            a=by[(request['short_id'],'SMALL_C200',stage)];b=by[(request['short_id'],'CREF_C200',stage)]
            row=dict(short_id=request['short_id'],request_id=request['request_id'],parent_id=request['parent_id'],stage=stage,selection_reason=request['selection_reason'])
            for k in a:
                if k.startswith(('inherited_','corridor_','dup_and_','near_parallel_','exact_C_','exact_parallel_')):
                    row['SMALL__'+k]=a[k];row['CREF__'+k]=b[k]
                    if isinstance(a[k],(int,float)) and isinstance(b[k],(int,float)):row['delta__'+k]=b[k]-a[k]
            paired.append(row)
    c.csvwrite(c.OUT/'paired_per_request_stage.csv',paired)
    c.write(c.OUT/'aggregation_completion.json',dict(status='COMPLETE',outputs=44,source_refs=6,paired_rows=len(paired),
        FF_contact_rows=len(pair_contacts),old_metric_rescores=0,scope='Selected cases only; no population rate; no automated semantic labels',
        code_sha256=c.sha(c.CODE/'aggregate.py')))
    c.event('AGGREGATION_COMPLETE',rows=len(out),paired=len(paired),FF_contacts=len(pair_contacts))
    print('50 output/reference rows, 22 paired rows;',len(pair_contacts),'inherited FF contacts')
if __name__=='__main__':main()