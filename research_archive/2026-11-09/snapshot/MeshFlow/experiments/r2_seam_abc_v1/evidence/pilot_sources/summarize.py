"""Read saved scores only; retain all metrics and per-request costs by split."""
import collections
import json
import numpy as np
from . import eval_common as C
from . import evaluate as E
from r2.pilots.seam_quality_four_v1.summarize import flatten

BASE=C.ROOT/'experiments/r2_seam_slot_c200_v1'

def main():
    root=C.OUT;m=C.read(root/'eval_manifest.json')
    rows=[];roles=[];deletions=[];occupancy=[];refs={}
    for label in (*C.ARMS,'SEAM200'):
        srcroot=BASE if label=='SEAM200' else root
        arm='SEAM_SLOT_C200' if label=='SEAM200' else label
        requests=[q for q in m['requests'] if label!='SEAM200' or q['split']=='TRAIN']
        for q in requests:
            for stage in ('RAW','POST'):
                folder=srcroot/'evaluation/scores'/arm/q['request_id']/stage
                p=folder/'score.json';s=C.read(p);E.verify(s['files']);refs[str(p)]=C.sha(p)
                row=flatten(s,label);row['split']=q['split']
                pair=C.read(folder/'pairs.json');mesh=C.arrays(s['identity']['mesh_path'])
                k=int(q['K']);ids=mesh['raw_slot_ids'];seam=mesh['raw_seam_slot_mask'][ids]
                counts=collections.Counter()
                for kind,scan in pair.items():
                    for contact in scan['contacts']:
                        if contact['classification'] not in E.ILLEGAL:continue
                        i,j=map(int,contact['pair'])
                        if kind=='FF':
                            i+=k;j+=k
                            role='seam-seam' if seam[i] and seam[j] else 'seam-bulk' if seam[i] or seam[j] else 'bulk-bulk'
                        else:j+=k;role='C-seam' if seam[j] else 'C-bulk'
                        counts[role]+=1
                        roles.append(dict(label=label,split=q['split'],stage=stage,request_id=q['request_id'],kind=kind,role=role,
                            classification=contact['classification'],raw_face_i=int(ids[i]),raw_face_j=int(ids[j])))
                for role in ('seam-seam','seam-bulk','bulk-bulk','C-seam','C-bulk'):row['illegal_pairs_'+role]=counts[role]
                if sum(counts[r] for r in ('seam-seam','seam-bulk','bulk-bulk'))!=row['FF_illegal_pair_count']:
                    raise ValueError('FF role partition mismatch')
                rows.append(row)
                if stage=='POST':
                    for fi in np.flatnonzero(~mesh['source_face_kept']):
                        deletions.append(dict(label=label,split=q['split'],request_id=q['request_id'],raw_slot_id=int(fi),
                            role='SEAM' if mesh['raw_seam_slot_mask'][fi] else 'BULK',reason=str(mesh['source_face_reason'][fi]),
                            duplicate_of_raw_slot=int(mesh['duplicate_of_source_face'][fi])))
        for cond in m['conditions']:
            if label=='SEAM200' and cond['split']!='TRAIN':continue
            q=next(q for q in requests if q['condition_id']==cond['condition_id'])
            raw=C.arrays(srcroot/'evaluation'/arm/q['request_id']/'raw/raw.npz')
            truth=C.arrays(cond['labels_npz'])['occupied_labels'].astype(bool);pred=raw['occupied'].astype(bool)
            for ei in range(len(pred)):
                occupancy.append(dict(label=label,split=q['split'],condition_id=q['condition_id'],edge_id=ei,predicted=bool(pred[ei]),
                    source_label_training_audit_only=bool(truth[ei]),logit=float(raw['occupancy_logits'][ei])))
    for name,data in (('per_output',rows),('illegal_pairs_by_role',roles),('deleted_faces',deletions),('occupancy_edges',occupancy)):
        C.csvwrite(root/'evaluation'/(name+'.csv'),data)
    numeric=sorted(k for k in set().union(*(r.keys() for r in rows)) if k not in (*E.META,'label','stage') and any(isinstance(r.get(k),(float,int,bool)) for r in rows))
    summary=[]
    for label in (*C.ARMS,'SEAM200'):
        for split in ('TRAIN','DEV','ALL'):
            if label=='SEAM200' and split!='TRAIN':continue
            for stage in ('RAW','POST'):
                rr=[r for r in rows if r['label']==label and r['stage']==stage and (split=='ALL' or r['split']==split)]
                registered={'TRAIN':16,'DEV':12,'ALL':28}[split]
                for metric in numeric:
                    vals=[r[metric] for r in rr if isinstance(r.get(metric),(int,float,bool))]
                    summary.append(dict(label=label,split=split,stage=stage,metric=metric,n_registered=registered,n_valid=len(vals),
                        mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None,
                        min=float(np.min(vals)) if vals else None,max=float(np.max(vals)) if vals else None,
                        sum=float(np.sum(vals)) if vals else None,status='PASS' if len(vals)==registered else 'NA_OR_PARTIAL'))
    C.csvwrite(root/'evaluation/summary.csv',summary)
    lookup={(r['label'],r['stage'],r['request_id']):r for r in rows};paired=[]
    for base,candidate in [('LONG',a) for a in C.ARMS if a!='LONG']+[('SEAM200',a) for a in C.ARMS]:
        for q in m['requests']:
            if base=='SEAM200' and q['split']!='TRAIN':continue
            for stage in ('RAW','POST'):
                a,b=lookup[base,stage,q['request_id']],lookup[candidate,stage,q['request_id']]
                for metric in numeric:
                    x,y=a.get(metric),b.get(metric);valid=isinstance(x,(float,int,bool)) and isinstance(y,(float,int,bool))
                    paired.append(dict(baseline=base,candidate=candidate,request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],
                        noise_index=q['noise_index'],stage=stage,metric=metric,baseline_value=x,candidate_value=y,
                        delta=float(y)-float(x) if valid else None,status='PASS' if valid else 'NA'))
    C.csvwrite(root/'evaluation/paired_per_noise.csv',paired)
    groups=collections.defaultdict(list)
    fields=('baseline','candidate','parent_id','split','stage','metric')
    for row in paired:groups[tuple(row[f] for f in fields)].append(row)
    parent=[]
    for key,rr in sorted(groups.items()):
        valid=[r for r in rr if r['status']=='PASS']
        parent.append(dict(zip(fields,key),n_registered=4,n_valid=len(valid),
            **{f:float(np.mean([r[f] for r in valid])) if valid else None for f in ('baseline_value','candidate_value','delta')},
            status='PASS' if len(valid)==4 else 'NA_OR_PARTIAL'))
    C.csvwrite(root/'evaluation/paired_per_parent.csv',parent)
    C.write(root/'evaluation/summary_completion.json',dict(status='COMPLETE_NUMERICAL_PENDING_ACTUAL_REVIEW',source_scores=refs,
        summary_files={str(p):C.sha(p) for p in (root/'evaluation').glob('*.csv')},
        original_scorer_unchanged=True,new_scans=0,historical_SEAM200_scores=32,current_panel_scores=224))
    print(json.dumps(dict(status='COMPLETE_NUMERICAL',outputs=len(rows),paired_rows=len(paired))))

if __name__=='__main__':main()
