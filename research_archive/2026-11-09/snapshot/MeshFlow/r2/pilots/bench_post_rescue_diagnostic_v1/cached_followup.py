"""Read saved HINGE decisions and graph ledgers only; no geometry operation."""
import collections,csv,json
from . import hinge as H
C=H.C


def run():
    root=H.OUT;reg=C.read(root/'registration.json');dest=root/'cached_followup';protected={}
    p=root/'tables_v2/per_output.csv';protected[str(p)]=C.sha(p)
    with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    result={};edgerows=[];decisions=[]
    for label in H.LABELS:
        rr=[r for r in rows if r['label']==label and r['stage']=='HINGE_OPP015' and r['free_only_reached_bulk_area_fraction']!='']
        top=sorted(rr,key=lambda r:(-float(r['free_only_reached_bulk_area_fraction']),r['request_id']))[0]
        result[label]=dict(maximum_bulk_reach=dict(request_id=top['request_id'],split=top['split'],part_type=top['part_type'],
            fraction=float(top['free_only_reached_bulk_area_fraction']),percent=100*float(top['free_only_reached_bulk_area_fraction'])),
            applicable=len(rr),closed_C_NA=48-len(rr),new_edge_roles=collections.Counter(),new_edge_lengths=collections.Counter(),
            accepted_by_role=collections.Counter(),candidate_decisions=collections.Counter(),candidate_decisions_by_role=collections.defaultdict(collections.Counter),
            prefilter=collections.Counter(),retained_candidates=0,accepted=0)
    for job in reg['jobs']:
        label,q=job['label'],job['q'];rid=q['request_id'];d=H.mesh_dir(label,rid);x=result[label]
        for name in ('summary.json','candidates.json','paired_ledger.json','post.npz'):protected[str(d/name)]=C.sha(d/name)
        summ=C.read(d/'summary.json');cand=C.read(d/'candidates.json');ledger=C.read(d/'paired_ledger.json')
        mask=C.arrays(d/'post.npz')['raw_seam_slot_mask']
        x['prefilter'].update(summ['prefilter']);x['retained_candidates']+=summ['candidates'];x['accepted']+=summ['accepted']
        x['accepted_by_role'].update(summ['accepted_by_role']);x['candidate_decisions'].update(summ['decisions'])
        for r in cand:
            x['candidate_decisions_by_role'][r['role_pair']][r['decision']]+=1
            decisions.append(dict(label=label,request_id=rid,split=q['split'],closed_C=q['closed_C'],rank=r['rank'],role=r['role_pair'],decision=r['decision'],
                source_raw_slots=r['source_raw_slots'],original_POST_endpoint_gap=r['original_POST_endpoint_gap'],sort_PRE_GEO_max_displacement=r['sort_PRE_GEO_max_displacement']))
        for e in ledger['new_edges']:
            raw=[int(t[0]) for t in e['source_edge_key']]
            if e['kind']=='FF':
                a,b=map(bool,mask[raw]);role='seam-seam' if a and b else 'seam-bulk' if a or b else 'bulk-bulk'
            else:role=e['kind']
            x['new_edge_roles'][role]+=1;x['new_edge_lengths'][role]+=e['length_model']
            edgerows.append(dict(label=label,request_id=rid,split=q['split'],closed_C=q['closed_C'],kind=e['kind'],role=role,
                source_edge_key=e['source_edge_key'],length_model=e['length_model'],winding_opposite=e['winding_opposite']))
    files={}
    for name,value in (('new_legal_edges_by_role',edgerows),('candidate_decisions_by_role',decisions)):
        p=dest/(name+'.csv');C.csvwrite(p,value);files[str(p)]=C.sha(p)
    out=dict(status='COMPLETE',results=result,files=files,protected=protected,code_sha256=C.sha(__file__),
        model_calls=0,decoder_calls=0,scoring_calls=0,new_pair_scans=0,new_candidate_generation=0,
        interpretation='Exact tip-point contact does not imply a pair of open oppositely-wound incident edges with their other endpoints and every real GEO cluster member within1.5%L, nor passage through shape/legality guards. Global prefilter counts are not separately attributable to tip-only contacts or seam/bulk roles; retained candidates retain role-specific rejection counts.')
    C.write(dest/'summary.json',out);print(json.dumps(result));return out


if __name__=='__main__':run()
