"""Freeze the common eligible pool, paired schedule and bounded evaluation."""
import collections,copy,json
from . import common as C

def prepare():
    manifest=C.read(C.OUT/'data/manifest.json')
    rows=manifest['train'];by_parent={r['parent_id']:r for r in rows}
    groups={g:sorted([r for r in rows if r['replay_group']==g],key=lambda x:x['parent_id']) for g in ('OLD','FOCUS','HIGH')}
    if not all(groups.values()):raise ValueError('Common pool must contain all original exposure groups; do not silently reweight')
    source=C.read(C.PRIOR/'data/schedule_PLAIN1500.json')
    coefs=C.read(C.PRIOR/'data/coefficients_PLAIN1500.json')
    if isinstance(coefs,dict):coefs=coefs['steps']
    schedule=copy.deepcopy(source);schedule['steps']=schedule['steps'][:1000]
    pointers=collections.Counter();exposure=collections.Counter();changes=[]
    for entry in schedule['steps']:
        local=entry['stage_step'];entry.update(global_step=C.BASE_STEP+local,native_total_step=C.NATIVE_BASE+local)
        used=set()
        for pos in (0,2,4,6):
            original=entry['samples'][pos];g=original['replay_group'];pid=original['parent_id']
            if pid not in by_parent or pid in used:
                candidates=groups[g]
                for _ in candidates:
                    r=candidates[pointers[g]%len(candidates)];pointers[g]+=1
                    if r['parent_id'] not in used:break
                else:raise ValueError('Not enough distinct eligible parents in group')
                changes.append(dict(step=local,position=pos,old_parent=pid,new_parent=r['parent_id']))
            else:r=by_parent[pid]
            used.add(r['parent_id'])
            drop=C.seed('layout_drop',local,pos//2)%5==0
            for mi in (pos,pos+1):
                q=entry['samples'][mi]
                for field in ('parent_id','condition_id','case_id','case_path','case_sha256','target_version_id','N','K','replay_group'):
                    q[field]=r[field]
                q['physical_condition_id']=r['condition_id'];q['layout_drop']=drop
                for field in ('noise_seed','epsilon_seed','time_seed','permutation_seed'):q[field]=C.seed('PAIRED',local,mi,field)
                exposure[r['parent_id']]+=1
    schedule.update(schema=C.EXPERIMENT+'_paired_schedule',experiment=C.EXPERIMENT,updates=1000,
        arm='COMMON_BOTH_ARMS',layout_dropout='SHA256 per CLEAN/X_WIDTH pair modulo5=0; independent of source shape',
        sampling='Original group and parent schedule retained where eligible; deterministic within-group replacement otherwise',
        parent_exposure=dict(exposure),replacements=changes)
    outco=[]
    for e,old in zip(schedule['steps'],coefs):
        counts=[9*(r['N']-r['K']) for r in e['samples']];gg=[r['replay_group'] for r in e['samples']]
        weights=old['reference_group_mse_weights'];denom=sum(counts)
        mass={g:sum(c for c,h in zip(counts,gg) if g==h) for g in weights}
        gains=[weights[g]*denom/mass[g] for g in gg]
        if abs(sum(w for w in weights.values())-1)>1e-10:raise ValueError('Invalid reference group weights')
        outco.append(dict(stage_step=e['stage_step'],reference_group_mse_weights=weights,micro_free_scalar_counts=counts,
            micro_groups=gg,micro_gains=gains,denominator=denom))
    from .training import validate_stream
    validate_stream(schedule['steps'],outco)
    C.write(C.OUT/'data/schedule.json',schedule);C.write(C.OUT/'data/coefficients.json',outco)
    panel=C.read(C.OUT/'data/eval_manifest.json');qs=panel['requests']
    if len(qs)!=12 or len({q['parent_id'] for q in qs})!=6:raise ValueError('Frozen 12 request panel required')
    heldout={q['parent_id'] for q in qs if q['eval_group'] in ('DEV','NEW_PARENT_HOLDOUT','HIGH_HOLDOUT')}
    if heldout&by_parent.keys():raise ValueError('Evaluation parent leakage')
    C.verify({r['case_path']:r['case_sha256'] for r in rows})
    C.write(C.OUT/'data/preparation.json',dict(status='PASS',parents=len(rows),groups={g:len(v) for g,v in groups.items()},
        instances=8000,parent_exposure=dict(exposure),replacement_pairs=len(changes),
        layout_dropped_instances=sum(r['layout_drop'] for e in schedule['steps'] for r in e['samples']),
        model_calls=0,resource=C.register_resource('CPU_PREPARATION')))
    paths=[C.OUT/'data'/x for x in ('manifest.json','layout_records.json','eval_manifest.json','layout_eval_manifest.json','schedule.json','coefficients.json','preparation.json','verification.json','manual_source_review.json')]
    C.write(C.REGISTRATION,dict(status='FROZEN',experiment=C.EXPERIMENT,
        source=str(C.SOURCE),source_sha256=C.sha(C.SOURCE),source_global=25500,
        updates={'PLAIN':1000,'LAYOUT':1000},batch=8,microbatch=1,lr=1e-5,hybrid_lambda=.25,
        information_contract='Coarse source-privileged region boxes; not C-only completion, not a mathematical upper bound',
        layout_features='2 slots BACK_REGION/SUPPORT_REGION: relative center/L, full extent/L, onehot2; unknown=masked',
        architecture='64dim 4head embedding-only free-corner crossattention; zero output; base Native unchanged',
        layout_dropout=.2,training_scope='Same eligible parents, geometry, N, random streams and group FM coefficients',
        evaluation=dict(requests=12,labels=list(C.LABELS),new_RAW=36,new_D0=36,sampling_forwards=1800,
            stages=['RAW','BASE_POST','CLEAN_FINAL_POST'],all_new_raw_before_D0=True,
            metrics='Original177 fields and same physical queries; per-request transitions and AI_PROXY five-axis review',
            main_question='RAW C adoption AND plausible body AND major supports; costs CF/FF/I/fragmentation and posterior losses retained'),
        stopping='Fixed endpoints and panel; no best selection, predictor training, backend changes, automatic extension or retries',
        authorization='User accepted proposed direction and two1000-update paired pilot',
        data={str(p):C.sha(p) for p in paths}))
    print(json.dumps(dict(status='FROZEN',parents=len(rows),new_raw=36,updates=2000)))

if __name__=='__main__':prepare()
