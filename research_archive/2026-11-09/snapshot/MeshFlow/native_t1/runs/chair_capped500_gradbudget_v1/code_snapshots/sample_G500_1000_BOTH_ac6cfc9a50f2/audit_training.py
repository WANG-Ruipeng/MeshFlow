"""CPU-only readback of actual gradient-budget training; live and final modes.

No torch/model/scorer imports, matching, RNG draws, or scientific calls.
The snapshot audit checks logged actual parameter measurements; it never
reconstructs parameter updates from gradient norms or opens weight tensors.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import struct
import time

DEFAULT_OUT = Path(__file__).resolve().parents[3] / 'native_t1/runs/chair_capped500_gradbudget_v1'
ARMS = ('T','L','G')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def f32(x):
    return struct.unpack('f',struct.pack('f',x))[0]


def mean(values):
    v=[float(x) for x in values if x is not None]
    return statistics.mean(v) if v else None


def median(values):
    v=[float(x) for x in values if x is not None]
    return statistics.median(v) if v else None


def output_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def output_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


class ReadAudit:
    def __init__(self,live):
        self.live=live;self.sources={};self.errors=[];self.checks=Counter();self.missing=[];self.partial_tails=[]

    def check(self,ok,name,detail=None):
        self.checks[name]+=1
        if not ok:self.errors.append(dict(check=name,detail=detail))

    def close(self,a,b,name,detail=None):
        self.check(a is not None and b is not None and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-14),name,detail)

    def read(self,path,lines=False,optional=False):
        path=Path(path)
        if not path.exists():
            if not optional:self.missing.append(str(path))
            return [] if lines else None
        data=path.read_bytes();read_data=data
        if lines and data and not data.endswith(b'\n'):
            if self.live:
                read_data=data[:data.rfind(b'\n')+1]
                self.partial_tails.append(dict(path=str(path),ignored_bytes=len(data)-len(read_data)))
            else:self.check(False,'unterminated_final_jsonl',str(path))
        self.sources[str(path)]=dict(sha256=sha(read_data),bytes=len(read_data),whole_file_snapshot=len(read_data)==len(data))
        text=read_data.decode('utf-8-sig')
        return [json.loads(s) for s in text.splitlines() if s.strip()] if lines else json.loads(text)


def run(out=DEFAULT_OUT,live=False):
    out=Path(out).resolve();dest=out/'training_audit'/('live' if live else 'final');dest.mkdir(parents=True,exist_ok=True)
    audit=ReadAudit(live);started=time.time()
    protocol=audit.read(out/'protocol.json')
    queue=audit.read(out/'registration/auxiliary_queue_plan.json')
    plan=audit.read(out/'streams/repeat/train_plan.json')
    schedule=protocol['training']['true_parameter_update_steps'] if protocol else [s for s in range(1,1001) if s==1 or s==501 or s%25==0]
    if protocol:
        audit.check(schedule==[s for s in range(1,1001) if s==1 or s==501 or s%25==0],'exact_42_snapshot_schedule')
        audit.check(protocol['rho']==1. and protocol['training']['updates_per_arm']==1000,'fixed_rho_updates')
        for path,expected in protocol['scientific_sources'].items():
            data=Path(path).read_bytes();audit.sources[path]=dict(sha256=sha(data),bytes=len(data),whole_file_snapshot=True)
            audit.check(sha(data)==expected,'frozen_scientific_source',path)
    metrics={a:audit.read(out/'training'/f'{a}_metrics.jsonl',lines=True,optional=live) for a in ARMS}
    steps=[];snapshots=[];pairs=[];auxpairs=[];phase_rows=[];ledger_rows=[];markers={}
    qby={q['step']:q for q in queue['records']} if queue else {}
    records={r['sample_index']:r for r in plan['records']} if plan else {}
    parameter_sets=set()
    for arm,rows in metrics.items():
        audit.check([r['pilot_step'] for r in rows]==list(range(1,len(rows)+1)),'contiguous_steps',arm)
        audit.check(len(rows)<=1000,'maximum_1000_steps',arm)
        marker=audit.read(out/'training'/f'{arm}_1000_complete.json',optional=live)
        markers[arm]=marker
        if marker:
            audit.check(marker['status']=='PASS' and marker['steps']==1000 and len(rows)==1000,'completed_marker_full1000',arm)
            if protocol:audit.check(marker['seed']==protocol['training']['seed'],'completion_seed',arm)
        state_identity={}
        for r in rows:
            step=int(r['pilot_step']);context=dict(arm=arm,step=step)
            audit.check(r['arm']==arm and r['conditional_total_step']==5000+step,'arm_and_total_step',context)
            micro=r['sample_diagnostics'];den=sum(9*(int(x['N'])-int(x['K'])) for x in micro)
            audit.check(len(micro)==8 and len({x['uid'] for x in micro})==8,'8_distinct_primary_parents',context)
            audit.check(den==r['free_coordinate_denominator']==r['effective_free_coordinates'],'full_primary_free_denominator',context)
            audit.check(sum(x['alpha']==1. for x in micro)==4 and all(.9<=x['alpha']<=1.1 for x in micro),'4unit4uniform',context)
            audit.check(sum(x['FM_contribution'] for x in micro).hex()==float(r['FM']).hex(),'Python_order_FM_sum',context)
            audit.check(sha(''.join(x['shared_input_sha256'] for x in micro).encode())==r['shared_batch_sha256'],'batch_hash_from_all8',context)
            for offset,x in enumerate(micro):
                index=8*(step-1)+offset
                audit.check(x['sample_index']==index,'primary_sample_cursor',context)
                if index in records:
                    expected=records[index]
                    audit.check(all(x[k]==expected[k] for k in ('uid','task_id','alpha')),'registered_primary_plan',dict(**context,index=index))
            nf,na,lf,la=map(float,(r['gF_norm'],r['gA_norm'],r['FM'],r['aux_unweighted']))
            audit.check(all(math.isfinite(v) and v>=0 for v in (nf,na,lf,la)),'finite_losses_and_norms',context)
            ll=min(.1,lf/max(la,1e-30));lg=nf/(na+1e-12);actual=min(ll,lg) if arm=='G' else ll
            if arm=='G' and nf==0:actual=0.
            for key,value in (('lambda_loss',ll),('lambda_grad',lg),('actual_lambda',actual)):
                audit.check(float(r[key]).hex()==value.hex() and r[key+'_hex']==value.hex(),'detached_coefficient_'+key,context)
            audit.check(r['loss_cap_active']==(ll<.1),'loss_cap_trigger',context)
            audit.check(r['gradient_cap_active']==(arm=='G' and lg<ll),'gradient_cap_trigger',context)
            audit.check(r['coefficient_detached'] is True and r['primary_finite'] is True and r['auxiliary_finite'] is True and r['combination_finite'] is True,'finite_detached_carrier',context)
            aw=float(r['weighted_aux_norm']);pre=float(r['combination_preclip_norm'])
            audit.check(math.isfinite(aw) and math.isfinite(pre),'finite_assembled_norms',context)
            audit.close(r['weighted_auxiliary_loss_value'],actual*la,'weighted_loss_diagnostic',context)
            audit.close(r['theoretical_weighted_aux_norm'],actual*na,'theoretical_weighted_gradient',context)
            if arm=='G':
                audit.check(actual*na<=nf,'mathematical_gradient_budget',context)
                audit.close(r['actual_FP32_budget_excess'],max(0.,aw-nf),'actual_FP32_budget_rounding_report',context)
                audit.close(r['theoretical_budget_excess'],max(0.,actual*na-nf),'theoretical_budget_excess_report',context)
            if nf and na:audit.close(r['cosine'],float(r['gF_dot_gA'])/(nf*na),'full_gradient_cosine',context)
            else:audit.check(r['cosine'] is None and r['cosine_status']=='UNDEFINED_ZERO_NORM','zero_norm_undefined_cosine',context)
            if nf:audit.close(r['weighted_aux_over_FM_norm'],aw/nf,'actual_gradient_norm_ratio',context)
            else:audit.check(r['weighted_aux_over_FM_norm'] is None,'zero_F_ratio_NA',context)
            audit.close(r['clip_coefficient'],min(1.,1./(r['grad_norm_before_clip']+1e-6)),'original_global_clip_coefficient',context)
            parameter_sets.add((r['parameter_set_sha256'],r['parameter_count'],r['parameter_elements']))
            for key in ('primary','auxiliary','both'):
                audit.check(len(r[key+'_none_names'])==r[key+'_none_count'],'None_parameter_count_'+key,context)
            a=r['aux_audit'];expected=qby.get(step)
            audit.check(r['aux_free_denominator']==a['denominator'] and a['denominator']>0,'auxiliary_own_free_denominator',context)
            if expected:
                audit.check(all(a[k]==expected[k] for k in ('queue','uid','request_id','candidate','state_index','payload_sha256')),'frozen_auxiliary_queue_identity',context)
                audit.check(float(a['t'])==f32(expected['t']) and r['aux_t']==float(a['t']),'frozen_auxiliary_time',context)
                audit.check(a['training_seed']==expected['teacher_seed'],'frozen_auxiliary_seed',context)
                audit.check(a['pool_manifest_sha256']==queue['source_pool_sha256'],'frozen_feedback_pool',context)
                target_key='teacher_target_sha256' if arm=='T' else 'trajectory_target_sha256'
                audit.check(a['target_sha256']==expected[target_key],'proper_teacher_or_trajectory_target_map',context)
                if arm!='T':audit.check(a['xt_sha256']==expected['trajectory_state_sha256'],'real_START_state_identity',context)
            identity=tuple(a[k] for k in ('xt_sha256','u_sha256','target_sha256'))
            key=(a['request_id'],a['state_index'])
            audit.check(key not in state_identity or state_identity[key]==identity,'fixed_aux_state_target_through_training',context)
            state_identity[key]=identity
            measurement=r.get('actual_parameter_update')
            audit.check((measurement is not None)==(step in schedule),'exact_actual_parameter_snapshot_steps',context)
            if measurement:
                m=measurement
                audit.check(m['status']=='PASS' and m['no_gradient_proxy'] is True and m['released'] is True,'actual_update_measurement_scope',context)
                audit.check(m['elements']==r['parameter_elements'] and len(m['per_parameter'])==r['parameter_count'],'full_parameter_update_coverage',context)
                audit.close(sum(x['actual_update_norm']**2 for x in m['per_parameter']),m['actual_update_norm']**2,'actual_update_parameter_norm_sum',context)
                if m['parameter_before_norm']:
                    audit.close(m['actual_update_relative_norm'],m['actual_update_norm']/m['parameter_before_norm'],'actual_relative_update_norm',context)
                else:audit.check(m['actual_update_relative_norm'] is None,'zero_parameter_norm_relative_NA',context)
                snapshots.append(dict(**context,actual_update_norm=m['actual_update_norm'],actual_update_relative_norm=m['actual_update_relative_norm'],parameter_before_norm=m['parameter_before_norm'],parameter_after_norm=m['parameter_after_norm'],changed_elements=m['changed_elements'],elements=m['elements'],max_abs_update=m['max_abs_update'],cpu_snapshot_bytes=m['cpu_snapshot_bytes'],snapshot_seconds=m['snapshot_seconds'],measure_seconds=m['measure_seconds']))
            steps.append(dict(**context,phase500='1-500' if step<=500 else '501-1000',FM=lf,raw_aux=la,
                weighted_aux_loss_diagnostic=r['weighted_auxiliary_loss_value'],lambda_loss=ll,lambda_grad=lg,actual_lambda=actual,
                loss_cap_active=r['loss_cap_active'],gradient_cap_active=r['gradient_cap_active'],gF_norm=nf,gA_norm=na,
                actual_weighted_aux_norm=aw,actual_aux_FM_ratio=r['weighted_aux_over_FM_norm'],cosine=r['cosine'],
                preclip_norm=pre,torch_clip_pre_norm=r['grad_norm_before_clip'],clip_norm_difference=r['grad_norm_before_clip']-pre,
                clip_coefficient=r['clip_coefficient'],clip_active=r['grad_norm_before_clip']>1,
                FP32_budget_excess=r['actual_FP32_budget_excess'],queue=a['queue'],uid=a['uid'],request_id=a['request_id'],t=a['t'],
                target_sha256=a['target_sha256'],xt_sha256=a['xt_sha256'],u_sha256=a['u_sha256'],
                shared_batch_sha256=r['shared_batch_sha256'],coupled_batch_sha256=r['coupled_batch_sha256'],
                primary_buffer_bytes=r['primary_buffer_bytes'],carrier_seconds=r['carrier_seconds'],
                update_seconds=r['update_total_seconds'],input_seconds=r['OT_and_input_seconds']))
        if len(rows)==1000:
            audit.check(len(state_identity)==24,'exact_24_states_complete',arm)
            audit.check(sum(s['arm']==arm for s in snapshots)==42,'actual_42_update_snapshots_complete',arm)
    audit.check(len(parameter_sets)<=1,'same_full_trainable_parameter_set_all_arms')
    for a,b in itertools.combinations(ARMS,2):
        for x,y in zip(metrics[a],metrics[b]):
            match=(x['shared_batch_sha256']==y['shared_batch_sha256'] and x['coupled_batch_sha256']==y['coupled_batch_sha256'] and x['free_coordinate_denominator']==y['free_coordinate_denominator'])
            audit.check(match,'cross_arm_paired_primary_stream',dict(arms=[a,b],step=x['pilot_step']))
            pairs.append(dict(arm_a=a,arm_b=b,step=x['pilot_step'],identical=match,shared_sha256=x['shared_batch_sha256'],coupled_sha256=x['coupled_batch_sha256']))
    for x,y in zip(metrics['L'],metrics['G']):
        ax,ay=x['aux_audit'],y['aux_audit'];fields=('request_id','queue','uid','candidate','t','state_index','payload_sha256','xt_sha256','u_sha256','target_sha256','C_sha256')
        match=all(ax[k]==ay[k] for k in fields)
        audit.check(match,'L_G_exact_auxiliary_state_target',x['pilot_step'])
        auxpairs.append(dict(step=x['pilot_step'],identical=match,request_id=ax['request_id'],t=ax['t'],xt_sha256=ax['xt_sha256'],u_sha256=ax['u_sha256'],target_sha256=ax['target_sha256']))
    begins={};ends={}
    for path in sorted((out/'ledger').glob('calls_*.jsonl')):
        for row in audit.read(path,lines=True):
            if row['kind'] not in ('training_forward','training_backward','optimizer_update'):continue
            cid=row['call_id'];target=begins if row['event']=='CALL_BEGIN' else ends
            audit.check(cid not in target,'unique_training_call_identity',cid);target[cid]=row
    calls=Counter();call_scopes=defaultdict(list);pending=[];failed=[];future=[]
    for cid,begin in begins.items():
        ctx=begin['context'];arm=ctx.get('arm');step=int(ctx.get('pilot_step',0));end=ends.get(cid)
        if end is None:pending.append(cid)
        elif end['event']=='CALL_FAILED':failed.append(cid)
        if arm not in metrics or step>len(metrics[arm]):future.append(cid);continue
        audit.check(end is not None and end['event']=='CALL_RETURN','completed_metric_has_returned_call',cid)
        if end:audit.check(end['kind']==begin['kind'] and end['context']==ctx,'call_return_matches_begin',cid)
        calls[(arm,step,begin['kind'])]+=1
        call_scopes[(arm,step,begin['kind'])].append(ctx)
    audit.check(set(ends).issubset(begins),'no_orphan_training_call_returns')
    audit.check(not failed,'no_training_call_failures',failed)
    if not live:audit.check(not pending and not future,'no_pending_or_unrecorded_final_training_calls',dict(pending=pending,future=future))
    for arm,rows in metrics.items():
        for r in rows:
            step=r['pilot_step'];counts={kind:calls[(arm,step,kind)] for kind in ('training_forward','training_backward','optimizer_update')}
            audit.check(counts==dict(training_forward=9,training_backward=9,optimizer_update=1),'actual_9F9B1update_each_step',dict(arm=arm,step=step,counts=counts))
            for kind in ('training_forward','training_backward'):
                contexts=call_scopes[(arm,step,kind)]
                audit.check(Counter(x.get('microbatch') for x in contexts)==Counter(list(range(8))+['aux']),'each_primary_and_auxiliary_call_once',dict(arm=arm,step=step,kind=kind))
                for ctx in contexts:
                    if ctx.get('microbatch')=='aux':
                        audit.check(ctx.get('request_id')==r['aux_audit']['request_id'],'actual_auxiliary_call_identity',dict(arm=arm,step=step,kind=kind))
                    elif isinstance(ctx.get('microbatch'),int) and 0<=ctx['microbatch']<8:
                        m=r['sample_diagnostics'][ctx['microbatch']]
                        audit.check(ctx.get('sample_index')==m['sample_index'] and ctx.get('shared_input_sha256')==m['shared_input_sha256'],'actual_primary_call_identity',dict(arm=arm,step=step,kind=kind))
            ledger_rows.append(dict(arm=arm,step=step,**counts))
    for arm in ARMS:
        for half in ('1-500','501-1000'):
            group=[r for r in steps if r['arm']==arm and r['phase500']==half]
            if not group:continue
            measured=[s for s in snapshots if s['arm']==arm and (s['step']<=500)==(half=='1-500')]
            phase_rows.append(dict(arm=arm,phase500=half,completed_steps=len(group),FM_mean=mean(r['FM'] for r in group),raw_aux_mean=mean(r['raw_aux'] for r in group),
                coefficient_mean=mean(r['actual_lambda'] for r in group),gradient_cap_fraction=mean(r['gradient_cap_active'] for r in group),loss_cap_fraction=mean(r['loss_cap_active'] for r in group),
                clip_fraction=mean(r['clip_active'] for r in group),aux_FM_norm_ratio_mean=mean(r['actual_aux_FM_ratio'] for r in group),aux_FM_norm_ratio_median=median(r['actual_aux_FM_ratio'] for r in group),
                cosine_mean=mean(r['cosine'] for r in group),snapshot_count=len(measured),actual_update_norm_median=median(r['actual_update_norm'] for r in measured),actual_relative_update_median=median(r['actual_update_relative_norm'] for r in measured),
                actual_FP32_budget_excess_max=max((r['FP32_budget_excess'] or 0.) for r in group),carrier_seconds_mean=mean(r['carrier_seconds'] for r in group)))
    all_complete=all(len(metrics[a])==1000 and markers[a] and markers[a]['status']=='PASS' for a in ARMS)
    status='FAIL' if audit.errors else 'LIVE_INCOMPLETE' if live and audit.missing else 'LIVE_PASS' if live and steps else 'LIVE_WAITING' if live else 'PASS' if all_complete and not audit.missing else 'INCOMPLETE'
    result=dict(status=status,live=live,checks=sum(audit.checks.values()),checks_by_name=dict(audit.checks),errors=audit.errors,missing=audit.missing,
        completed_steps={a:len(r) for a,r in metrics.items()},actual_snapshot_counts={a:sum(s['arm']==a for s in snapshots) for a in ARMS},
        expected_final_snapshots_per_arm=42,cross_arm_primary_pairs=len(pairs),expected_final_cross_arm_pairs=3000,L_G_auxiliary_pairs=len(auxpairs),
        training_call_pairs=len(begins),pending_training_calls=pending,failed_training_calls=failed,calls_outside_completed_metric_prefix=future,
        phase500_summary=phase_rows,sources=audit.sources,ignored_live_partial_tails=audit.partial_tails,script_sha256=sha(Path(__file__).read_bytes()),
        started_unix=started,finished_unix=time.time(),CPU_only=True,model_calls=0,optimizer_updates=0,checkpoint_tensors_loaded=False,
        limitations=['Norm/cosine audit recomputes scalar relations from logged full-gradient measurements, not gradients from model tensors.',
            'Actual parameter updates are only the 42 registered snapshots per complete arm; no interpolation to unmeasured steps.',
            'G mathematical budget and actual FP32 rounding residuals are distinct; raw residuals are retained without changing rho or coefficients.',
            'Training effects are not proof of OLD/FRESH generated-control gains.'])
    for name,rows in (('per_step_gradient_budget',steps),('actual_parameter_updates',snapshots),('paired_primary_inputs',pairs),('L_G_auxiliary_pairs',auxpairs),('training_call_counts',ledger_rows),('phase500_summary',phase_rows)):
        output_csv(dest/(name+'.csv'),rows)
    output_json(dest/'summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default=str(DEFAULT_OUT));parser.add_argument('--live',action='store_true')
    args=parser.parse_args();result=run(args.out,args.live)
    print(json.dumps({k:result[k] for k in ('status','checks','completed_steps','actual_snapshot_counts','cross_arm_primary_pairs','errors')},indent=2))
    raise SystemExit(1 if result['status']=='FAIL' else 0)
