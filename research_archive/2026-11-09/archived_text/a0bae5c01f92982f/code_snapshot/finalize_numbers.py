"""Read completed quick-pilot logs and CSVs; no model loading or rescoring.

Run only after both arm status.json files and evaluation/aggregation.json are
COMPLETE. Does not import training/evaluation modules or any tensor framework.
"""
from __future__ import annotations
import collections,csv,hashlib,io,json,math,os,statistics
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seat_quick_learnability_v1'
ARMS=('SMALL_C200','FULL_C200')
BASE='EXP67_FROZEN'
METRICS=(
 'CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
 'A_VMR__raw_tau_1e-6','PMR__raw_tau_1e-6','C_F_has_valid_edge','C_F_valid_edge_count',
 'C_F_valid_opposite_winding_edge_count','opposite_winding_reached_free_area_fraction',
 'free_minimum_angle_lt5_fraction','free_near_degenerate_fraction','actual_free_faces',
 'DESC__C_boundary_to_F_surface_mean_model','boundary_distance_mean_model',
)
INPUT_HASHES={}

def resolve(p):
 s=str(p)
 if os.name=='nt' and s.startswith('/mnt/') and len(s)>7:s=s[5].upper()+':/'+s[7:]
 return Path(s)

def digest(p):
 h=hashlib.sha256()
 with resolve(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def protect(p):
 p=resolve(p);h=digest(p);key=str(p)
 if key in INPUT_HASHES and INPUT_HASHES[key]!=h:raise ValueError('Input changed while reading: '+key)
 INPUT_HASHES[key]=h;return h

def read(p):
 protect(p);return json.loads(resolve(p).read_text(encoding='utf-8-sig'))

def lines(p):
 protect(p);return [json.loads(x) for x in resolve(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def csvrows(p):
 protect(p)
 with resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def number(x):
 if x in (None,'','NA','UNKNOWN','BLOCKED'):return None
 if x is True or x=='True':return 1.0
 if x is False or x=='False':return 0.0
 try:v=float(x)
 except (ValueError,TypeError):return None
 return v if math.isfinite(v) else None

def mean(vals):return statistics.mean(vals) if vals else None

def fixed_write(p,content):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=content:raise RuntimeError('Final artifact already exists with different content: '+str(p))
 else:p.write_bytes(content)

def write_json(p,v):fixed_write(p,(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8'))

def write_csv(p,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row));b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=keys);w.writeheader()
 for row in rows:w.writerow(row)
 fixed_write(p,b.getvalue().encode('utf-8-sig'))

def audit_training(arm,reg):
 folder=OUT/'training'/arm;status=read(folder/'status.json')
 if status.get('status')!='COMPLETE':raise RuntimeError('BLOCKED: training not COMPLETE: '+arm)
 steps=int(reg['updates_per_arm']);batch=int(reg['batch']);assert status['local_step']==steps and status['global_step']==reg['endpoint_global_step']
 logs=lines(folder/'training_log.jsonl');calls=lines(folder/'calls.jsonl');init=read(folder/'initialization.json');resources=read(folder/'resources.json')
 assert [x['step'] for x in logs]==list(range(1,steps+1)),arm+' logs replay/skip'
 assert [x['global_step'] for x in logs]==list(range(reg['source_global_step']+1,reg['endpoint_global_step']+1))
 assert init['source_generator_sha256']==reg['source_generator_sha256'] and init['optimizer_restored'] and init['rng_restored']
 begun={};pending={};attempted=collections.Counter();successful=collections.Counter();failed=collections.Counter();seconds=collections.Counter();attempt_ids=set();by_step=collections.defaultdict(collections.Counter)
 for x in calls:
  assert x['arm']==arm;ident=x['call_id'];kind=x['kind'];attempt_ids.add(x['attempt_id'])
  if x['event']=='CALL_BEGIN':
   assert ident not in begun;begun[ident]=x;pending[ident]=x;attempted[kind]+=1
  elif x['event'] in ('CALL_DONE','CALL_FAILED'):
   assert ident in pending and pending[ident]['kind']==kind
   a=pending.pop(ident);assert a['context']==x['context']
   (successful if x['event']=='CALL_DONE' else failed)[kind]+=1;seconds[kind]+=float(x.get('seconds',0))
   if x['event']=='CALL_DONE':by_step[int(x['context']['local_step'])][kind]+=1
  else:raise ValueError('Unknown ledger event '+x['event'])
 expected={'train_forward':steps*batch,'train_backward':steps*batch,'optimizer_update':steps}
 assert not pending and not failed and dict(attempted)==expected and dict(successful)==expected
 for step in range(1,steps+1):assert dict(by_step[step])=={'train_forward':batch,'train_backward':batch,'optimizer_update':1}
 for key in ('attempted','successful'):assert status['calls'][key]==expected
 assert not status['calls']['failed'] and not status['calls']['pending_call_ids']
 # Read receipts, not multi-GB parameter tensors. The trainer already saved/hashes them.
 receipts={}
 for kind in ('checkpoint','inference_export'):
  bound=status[kind];path=resolve(bound['path']);rp=path.with_suffix('.json');rec=read(rp)
  assert rec['local_step']==steps and rec['global_step']==reg['endpoint_global_step']
  assert rec['sha256']==bound['sha256'] and rec['generator_state_sha256']==bound['generator_state_sha256'] and path.exists()
  receipts[kind]={'path':str(path),'receipt_path':str(rp),'file_sha256_from_completed_receipt':rec['sha256'],
                  'bytes':path.stat().st_size,'generator_state_sha256':rec['generator_state_sha256'],'weight_file_hash_recomputed':False,'model_loaded':False}
 device=[];own=[];free=[];util=[]
 for x in resources.get('samples',[]):
  for line in x.get('gpu',{}).get('stdout','').splitlines():
   parts=[z.strip() for z in line.split(',')]
   if len(parts)>=5:
    vals=[number(parts[i]) for i in (2,3,4)]
    if vals[0] is not None:device.append(vals[0])
    if vals[1] is not None:free.append(vals[1])
    if vals[2] is not None:util.append(vals[2])
  for line in x.get('processes',{}).get('stdout','').splitlines():
   parts=[z.strip() for z in line.split(',')]
   if len(parts)>=3 and parts[0]==str(x.get('pid')):
    val=number(parts[-1])
    if val is not None:own.append(val)
 plans=read(OUT/'plans'/(arm+'.json'));assert len(plans['records'])==steps*batch
 used={x['context']['sample_index'] for x in calls if x['event']=='CALL_DONE' and x['kind']=='train_forward'}
 assert used==set(range(steps*batch))
 parent_exposure=collections.Counter(x['uid'] for x in plans['records']);condition_exposure=collections.Counter(x['task_id'] for x in plans['records'])
 first=[float(x['FM']) for x in logs[:25]];last=[float(x['FM']) for x in logs[-25:]]
 source_time=[float(x['time_unix']) for x in calls]
 return {'arm':arm,'status':'PASS_COMPLETE_ACCOUNTING','local_updates':steps,'source_global_step':reg['source_global_step'],'endpoint_global_step':reg['endpoint_global_step'],
         'actual_attempted':dict(attempted),'actual_successful':dict(successful),'failed':dict(failed),'pending':list(pending),'attempt_ids':sorted(attempt_ids),
         'exposures':len(used),'training_parents':len(parent_exposure),'parent_exposure_min':min(parent_exposure.values()),'parent_exposure_max':max(parent_exposure.values()),
         'condition_exposure':dict(condition_exposure),'FM_first25_mean':mean(first),'FM_last25_mean':mean(last),'FM_scope':'Online own-arm training FM; distributions differ across arms, not a common validation loss.',
         'training_wall_seconds':float(status['wall_seconds']) if len(attempt_ids)==1 else None,'last_attempt_status_wall_seconds':float(status['wall_seconds']),
         'training_wall_status':'ONE_COMPLETED_ATTEMPT_INCLUDES_LOAD_OT_CHECKPOINTS' if len(attempt_ids)==1 else 'NA_CUMULATIVE_WALL_MULTIPLE_ATTEMPTS',
         'call_timestamp_span_seconds':max(source_time)-min(source_time),'call_timestamp_span_scope':'Includes pauses between calls; not pure GPU time.',
         'sum_update_seconds':sum(float(x['update_seconds']) for x in logs),'sum_input_wait_seconds':sum(float(x['input_seconds']) for x in logs),
         'sum_call_seconds':dict(seconds),'timing_note':'Call/update times nest within wall time; do not add nested phases.',
         'cuda_allocated_peak_bytes':max(int(x['cuda_peak_allocated_bytes']) for x in logs),'cuda_reserved_peak_bytes':max(int(x['cuda_peak_reserved_bytes']) for x in logs),
         'cuda_peak_scope':'Max over reset-per-update counters including backward/optimizer; excludes CUDA context/external allocations.',
         'sampled_device_used_peak_mib':max(device) if device else None,'sampled_own_process_peak_mib':max(own) if own else None,
         'process_memory_status':'OBSERVED_SAMPLED_PID' if own else 'NA_UNAVAILABLE_NVIDIA_SMI','sampled_device_min_free_mib':min(free) if free else None,
         'sampled_device_utilization_max_percent':max(util) if util else None,'resource_samples':len(resources.get('samples',[])),
         'resource_scope':'Device numbers include unrelated processes; sampled maxima are not continuous peaks. Process memory may be unavailable in WDDM.',
         'checkpoints':receipts}

def finalize():
 reg=read(OUT/'registration.json');assert tuple(reg['arms'])==ARMS
 agg=read(OUT/'evaluation/aggregation.json')
 if agg.get('status')!='COMPLETE' or agg.get('outputs')!=48:raise RuntimeError('BLOCKED: evaluation aggregate is not COMPLETE 48')
 arms=[audit_training(a,reg) for a in ARMS]
 rows=csvrows(OUT/'evaluation/per_output.csv');manifest=read(OUT/'inputs/manifest.json');requests={q['request_id']:q for q in manifest['requests']}
 assert len(rows)==48 and len(requests)==8 and len({q['parent_id'] for q in requests.values()})==4
 lookup={};exact={};statuses=collections.Counter()
 for row in rows:
  key=(row['label'],row['request_id'],row['stage']);assert key not in lookup;lookup[key]=row
  assert row['label'] in (BASE,*ARMS) and row['request_id'] in requests and row['stage'] in ('RAW','POST')
  for field in METRICS:assert field in row,field
  if digest(row['score_path'])!=row['score_sha256']:raise ValueError('Score hash differs')
  score=read(row['score_path'])
  for k,v in score['status_by_metric'].items():statuses[k+'='+v]+=1
  b=next(x for x in score['boundary'] if x['setting']=='raw_tau_1e-6')
  if b['status']=='PASS' and b['boundary_vertex_count']:
   exact[key,'A_VMR__raw_tau_1e-6']=Fraction(b['matched_count'],b['boundary_vertex_count']);exact[key,'PMR__raw_tau_1e-6']=Fraction(b['perfect'])
  assert int(row['N'])==requests[row['request_id']]['N'] and int(row['K'])==requests[row['request_id']]['K']
 macros=[]
 for label in (BASE,*ARMS):
  for stage in ('RAW','POST'):
   rr=[x for x in rows if x['label']==label and x['stage']==stage];assert len(rr)==8
   for metric in METRICS:
    byparent=collections.defaultdict(list)
    for x in rr:byparent[x['parent_id']].append(number(x[metric]))
    assert len(byparent)==4 and all(len(v)==2 for v in byparent.values())
    parent_means=[mean([x for x in v if x is not None]) for v in byparent.values()]
    valid_values=[x for x in parent_means if x is not None];valid_outputs=sum(number(x[metric]) is not None for x in rr)
    macros.append({'label':label,'stage':stage,'metric':metric,'value':mean(valid_values),'parents':4,'valid_parents':len(valid_values),'outputs':8,'valid_outputs':valid_outputs,
                   'status':'PASS' if valid_outputs==8 else 'PARTIAL_OR_NA','aggregation':'two noises within parent, four parents equal'})
 paired=[]
 for before,after in ((BASE,ARMS[0]),(BASE,ARMS[1]),(ARMS[1],ARMS[0])):
  for rid,q in requests.items():
   for stage in ('RAW','POST'):
    ka=(before,rid,stage);kb=(after,rid,stage);a,b=lookup[ka],lookup[kb]
    assert a['boundary_vertex_denominator']==b['boundary_vertex_denominator']
    for metric in METRICS:
     av,bv=number(a[metric]),number(b[metric]);d=bv-av if av is not None and bv is not None else None
     frac=exact[kb,metric]-exact[ka,metric] if (ka,metric) in exact and (kb,metric) in exact else None
     paired.append({'comparison':after+'__minus__'+before,'request_id':rid,'parent_id':q['parent_id'],'condition_id':q['condition_id'],
                    'condition_type':q['condition_type'],'candidate':q['candidate'],'stage':stage,'metric':metric,'baseline_value':av,'candidate_value':bv,'delta':d,
                    'status':'PASS' if d is not None else 'NA','delta_exact_fraction':str(frac) if frac is not None else '',
                    'delta_exact_sign':int(frac>0)-int(frac<0) if frac is not None else '',
                    'boundary_vertex_denominator':int(a['boundary_vertex_denominator']),'C_sha256':q['C_sha256'],'Gaussian_sha256':q['Gaussian_sha256']})
 groups=collections.defaultdict(list)
 for x in paired:groups[x['comparison'],x['parent_id'],x['condition_id'],x['condition_type'],x['stage'],x['metric']].append(x)
 parents=[]
 for key,rr in sorted(groups.items()):
  assert len(rr)==2;dd=[x['delta'] for x in rr if x['delta'] is not None]
  ff=[Fraction(x['delta_exact_fraction']) for x in rr if x['delta_exact_fraction']!='']
  f=sum(ff,Fraction())/2 if len(ff)==2 else None
  parents.append(dict(zip(('comparison','parent_id','condition_id','condition_type','stage','metric'),key),
                  baseline_value=mean([x['baseline_value'] for x in rr if x['baseline_value'] is not None]),
                  candidate_value=mean([x['candidate_value'] for x in rr if x['candidate_value'] is not None]),delta=mean(dd),
                  status='PASS' if len(dd)==2 else 'PARTIAL_OR_NA',delta_exact_fraction=str(f) if f is not None else '',delta_exact_sign=int(f>0)-int(f<0) if f is not None else ''))
 # Verify the already-produced aggregate, without modifying or regenerating it.
 oldpairs={(x['comparison'],x['request_id'],x['stage'],x['metric']):x for x in csvrows(OUT/'evaluation/paired_per_output.csv')}
 maximum_error=0.;checks=0
 for x in paired:
  old=oldpairs[x['comparison'],x['request_id'],x['stage'],x['metric']]
  for field in ('baseline_value','candidate_value','delta'):
   a,b=x[field],number(old[field]);assert (a is None)==(b is None)
   if a is not None:maximum_error=max(maximum_error,abs(a-b));checks+=1
 assert maximum_error<1e-12
 for p,h in INPUT_HASHES.items():assert digest(p)==h,p
 total={kind:sum(a['actual_successful'][kind] for a in arms) for kind in ('train_forward','train_backward','optimizer_update')}
 result={'status':'COMPLETE_NUMERICAL_AUDIT','version':'seat_quick_final_numbers_v1','training':arms,'actual_total_successful':total,
         'key_metrics':list(METRICS),'macro':macros,'paired_parent_means':parents,'paired_noise_rows':len(paired),
         'status_by_metric_counts':dict(statuses),'existing_pair_table_verified_cells':checks,'max_abs_existing_pair_error':maximum_error,
         'registered_budget':{'updates_per_arm':reg['updates_per_arm'],'batch':reg['batch'],'microbatch':reg['microbatch']},
         'model_calls_in_this_script':0,'training_in_this_script':0,'new_quality_scans':0,'bootstrap':0,'visual_review':'NOT_PERFORMED_OR_INFERRED',
         'interpretation':'Four familiar TRAIN conditions, original eval noises. No claim of generalization, promotion or semantic adoption. FM from different arm distributions cannot be compared as a shared validation score.',
         'input_hashes':INPUT_HASHES,'entry_sha256':digest(__file__)}
 write_csv(OUT/'numerical_pairs.csv',paired)
 result['paired_csv_path']=str(OUT/'numerical_pairs.csv');result['paired_csv_sha256']=digest(OUT/'numerical_pairs.csv')
 write_json(OUT/'summary.json',result)
 def fmt(x,scale=1,digits=3):return 'NA' if x is None else f'{x*scale:.{digits}f}'
 mlook={(x['label'],x['stage'],x['metric']):x['value'] for x in macros}
 text=['# 快速座面可学性实验：账本与数值复算','',
       '两臂均完成固定200次更新，起点为同一EXP67的6000步生成器与AdamW状态，终点6200步。本文只整理日志和已有评分，不包含图审判断。','',
       '|训练臂|更新/F/B|实际训练曝光|父数|训练墙钟秒|首25 FM均值|末25 FM均值|',
       '|---|---:|---:|---:|---:|---:|---:|']
 for a in arms:text.append(f"|{a['arm']}|{a['local_updates']}/{a['actual_successful']['train_forward']}/{a['actual_successful']['train_backward']}|{a['exposures']}|{a['training_parents']}|{fmt(a['training_wall_seconds'])}|{fmt(a['FM_first25_mean'],digits=6)}|{fmt(a['FM_last25_mean'],digits=6)}|")
 text+=['','FM是各自训练分布上的在线损失；集中4条件与完整67父分布不同，不能横比两个FM均值判断模型优劣。首末25段也不是固定验证集误差。训练墙钟包含加载、OT等待、保存等；前向/反向/更新时间嵌套在墙钟内，不能相加重复记时。','',
        f"账本独立核对成功调用：训练F={total['train_forward']}、B={total['train_backward']}、更新={total['optimizer_update']}；各步8个F、8个B和1个更新，未闭合调用和失败均为0。",'',
        '|训练臂|allocated峰MiB|reserved峰MiB|采样整卡占用峰MiB|采样本进程峰MiB|',
        '|---|---:|---:|---:|---:|']
 for a in arms:text.append(f"|{a['arm']}|{fmt(a['cuda_allocated_peak_bytes']/2**20)}|{fmt(a['cuda_reserved_peak_bytes']/2**20)}|{fmt(a['sampled_device_used_peak_mib'])}|{fmt(a['sampled_own_process_peak_mib'])}|")
 text+=['','allocated/reserved是逐更新重置后取最大值，不含所有CUDA上下文开销；整卡采样包含其他程序。WDDM未报告本进程显存时保留NA，不能拿整卡峰或allocated冒充进程峰。','',
        '|模型|阶段|CF非法%|FF非法%|I%|严格AVMR%|PMR%|有合法共边输出/8|合法边均数|方向可达free面积%|小角<5°%|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for label in (BASE,*ARMS):
  for stage in ('RAW','POST'):
   def val(k):return mlook[label,stage,k]
   parts=[label,stage,*[fmt(val(k),100) for k in METRICS[:5]],fmt(val('C_F_has_valid_edge'),8,0)+'/8',fmt(val('C_F_valid_edge_count')),fmt(val('opposite_winding_reached_free_area_fraction'),100),fmt(val('free_minimum_angle_lt5_fraction'),100)]
   text.append('|'+'|'.join(parts)+'|')
 text+=['','每行先平均每父两噪声，再对4父等权。原评分分母、失败状态、S/E/唯一GT的NA保留于原score和summary.json；未用GT款式误差判断开放生成。接续、距离、相交和整体质量不可互相替代。','',
        '逐父×噪声×RAW/POST的三种差值（SMALL−起点、FULL−起点、SMALL−FULL）在numerical_pairs.csv；包含CF/FF/I、严格AVMR/PMR、合法共边及方向可达等14项。AVMR/PMR另保存整数匹配分母导出的精确分数差与符号，避免浮点残差改变持平判定。','',
        f'独立核对既有配对表{checks}个数值单元，最大差{maximum_error:.3g}。本脚本模型调用、训练、重新评分、bootstrap均为0。检查点仅核对已完成保存收据及文件存在性，不重新加载或重复hash大权重。','',
        '训练使用熟悉条件；本小面板只能为是否值得继续研究提供局部可学性证据。此文件不代填图审，不作最终晋级判断。']
 fixed_write(OUT/'NUMERICAL_FINDINGS.md',('\n'.join(text)+'\n').encode('utf-8'))
 return {'status':result['status'],'training_calls':total,'macro_rows':len(macros),'paired_rows':len(paired),'paired_checks':checks,'outputs':48,'models':0,'new_scores':0}

if __name__=='__main__':
 print(json.dumps(finalize(),ensure_ascii=False,allow_nan=False),flush=True)