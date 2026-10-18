"""Deterministic paired summaries of all fixed endpoint scores; no re-scoring."""
import collections
import json
import numpy as np
from . import eval_common as C
from . import evaluate as E

BASE=C.ROOT/'experiments/r2_seam_slot_c200_v1'
LABELS=('SEAM200','FULL4_C200',*C.ARMS)

def flatten(s,label):
    row={k:s[k] for k in E.META}
    row.update(label=label,stage=s['stage'])
    for field in ('metrics','topology','distance','seam_to_bulk'):
        row.update({k:v for k,v in s[field].items() if not isinstance(v,(dict,list)) and k!='status'})
    row.update(seam_to_bulk_status=s['seam_to_bulk']['status'],C_preserved_exact=s['C_preserved_exact'],
        I_status=s['status_by_metric']['I'],free_face_denominator=s['free_face_denominator'],
        CF_tested_pair_denominator=s['CF_tested_pair_denominator'],FF_tested_pair_denominator=s['FF_tested_pair_denominator'],
        C_query_weight_denominator=s['C_query_weight_denominator'])
    for b in s['boundary']:
        row['A_VMR__'+b['setting']]=b['VMR'];row['PMR__'+b['setting']]=b['perfect']
    return row

def main():
    root=C.OUT;m=C.read(root/'eval_manifest.json');rows=[];roles=[];refs={};occupancy=[];deletions=[]
    for label in LABELS:
        srcroot=BASE if label in ('SEAM200','FULL4_C200') else root
        arm='SEAM_SLOT_C200' if label=='SEAM200' else label
        for q in m['requests']:
            for stage in ('RAW','POST'):
                folder=srcroot/'evaluation/scores'/arm/q['request_id']/stage
                sp=folder/'score.json';s=C.read(sp);E.verify(s['files']);refs[str(sp)]=C.sha(sp)
                row=flatten(s,label);pair=C.read(folder/'pairs.json')
                mesh=C.arrays(s['identity']['mesh_path']);k=int(q['K']);ids=mesh['raw_slot_ids'];seam=mesh['raw_seam_slot_mask'][ids]
                counts=collections.Counter()
                for kind,scan in pair.items():
                    for contact in scan['contacts']:
                        if contact['classification'] not in E.ILLEGAL:continue
                        i,j=map(int,contact['pair'])
                        if kind=='FF':
                            i+=k;j+=k
                            role='seam-seam' if seam[i] and seam[j] else 'seam-bulk' if seam[i] or seam[j] else 'bulk-bulk'
                        else:
                            j+=k;role='C-seam' if seam[j] else 'C-bulk'
                        counts[role]+=1
                        roles.append(dict(label=label,stage=stage,request_id=q['request_id'],kind=kind,role=role,
                            classification=contact['classification'],raw_face_i=int(ids[i]),raw_face_j=int(ids[j])))
                for role in ('seam-seam','seam-bulk','bulk-bulk','C-seam','C-bulk'):row['illegal_pairs_'+role]=counts[role]
                assert sum(counts[r] for r in ('seam-seam','seam-bulk','bulk-bulk'))==row['FF_illegal_pair_count']
                rows.append(row)
                if stage=='POST':
                    for fi in np.flatnonzero(~mesh['source_face_kept']):
                        deletions.append(dict(label=label,request_id=q['request_id'],raw_slot_id=int(fi),
                            role='SEAM' if mesh['raw_seam_slot_mask'][fi] else 'BULK',reason=str(mesh['source_face_reason'][fi]),
                            duplicate_of_raw_slot=int(mesh['duplicate_of_source_face'][fi])))
        if label=='FULL4_C200':continue
        for condition in m['conditions']:
            q=next(q for q in m['requests'] if q['condition_id']==condition['condition_id'])
            raw=C.arrays(srcroot/'evaluation'/arm/q['request_id']/'raw/raw.npz')
            truth=C.arrays(condition['labels_npz'])['occupied_labels'].astype(bool);pred=raw['occupied'].astype(bool)
            for ei in range(len(pred)):
                occupancy.append(dict(label=label,condition_id=q['condition_id'],edge_id=ei,predicted=bool(pred[ei]),
                    source_label_training_audit_only=bool(truth[ei]),logit=float(raw['occupancy_logits'][ei])))
    C.csvwrite(root/'evaluation/per_output.csv',rows)
    C.csvwrite(root/'evaluation/illegal_pairs_by_role.csv',roles)
    C.csvwrite(root/'evaluation/deleted_faces.csv',deletions)
    C.csvwrite(root/'evaluation/occupancy_edges.csv',occupancy)
    numeric=sorted(k for k in set().union(*(r.keys() for r in rows)) if k not in (*E.META,'label','stage') and any(isinstance(r.get(k),(float,int,bool)) for r in rows))
    summary=[]
    for label in LABELS:
        for stage in ('RAW','POST'):
            rr=[r for r in rows if r['label']==label and r['stage']==stage]
            for metric in numeric:
                vals=[r[metric] for r in rr if isinstance(r.get(metric),(int,float,bool))]
                summary.append(dict(label=label,stage=stage,metric=metric,n_registered=16,n_valid=len(vals),
                    mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None,
                    min=float(np.min(vals)) if vals else None,max=float(np.max(vals)) if vals else None,
                    sum=float(np.sum(vals)) if vals else None,status='PASS' if len(vals)==16 else 'NA_OR_PARTIAL'))
    C.csvwrite(root/'evaluation/summary.csv',summary)
    lookup={(r['label'],r['stage'],r['request_id']):r for r in rows};paired=[]
    for base,candidate in (('SEAM200','LONG'),('LONG','ANTI_COVER'),('LONG','LOCAL_NOISE'),('SEAM200','ANTI_COVER'),('SEAM200','LOCAL_NOISE')):
        for q in m['requests']:
            for stage in ('RAW','POST'):
                a,b=lookup[base,stage,q['request_id']],lookup[candidate,stage,q['request_id']]
                for metric in numeric:
                    x,y=a.get(metric),b.get(metric);valid=isinstance(x,(float,int,bool)) and isinstance(y,(float,int,bool))
                    paired.append(dict(baseline=base,candidate=candidate,request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],stage=stage,metric=metric,
                        baseline_value=x,candidate_value=y,delta=float(y)-float(x) if valid else None,status='PASS' if valid else 'NA'))
    C.csvwrite(root/'evaluation/paired_per_noise.csv',paired)
    groups=collections.defaultdict(list)
    for row in paired:groups[row['baseline'],row['candidate'],row['parent_id'],row['stage'],row['metric']].append(row)
    parent=[]
    for key,rr in sorted(groups.items()):
        valid=[r for r in rr if r['status']=='PASS']
        parent.append(dict(zip(('baseline','candidate','parent_id','stage','metric'),key),n_registered=4,n_valid=len(valid),
            **{f:float(np.mean([r[f] for r in valid])) if valid else None for f in ('baseline_value','candidate_value','delta')},
            status='PASS' if len(valid)==4 else 'NA_OR_PARTIAL'))
    C.csvwrite(root/'evaluation/paired_per_parent.csv',parent)
    C.write(root/'evaluation/summary_completion.json',dict(status='COMPLETE_NUMERICAL_PENDING_ACTUAL_REVIEW',source_scores=refs,
        summary_files={str(p):C.sha(p) for p in (root/'evaluation').glob('*.csv')},
        original_scorer_unchanged=True,new_scans=0,cached_SEAM200_FULL4_scores=64,new_endpoint_scores=96))
    print(json.dumps(dict(status='COMPLETE_NUMERICAL',outputs=len(rows),paired_rows=len(paired))))

if __name__=='__main__':main()
