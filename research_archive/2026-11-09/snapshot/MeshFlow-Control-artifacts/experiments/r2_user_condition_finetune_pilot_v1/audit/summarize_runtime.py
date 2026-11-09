"""Read-only runtime accounting. No model imports, scoring, rendering or weight reads."""
import argparse,csv,hashlib,json,re,time
from collections import Counter
from pathlib import Path

def main(out):
    sources=[]
    def read(p):
        p=Path(p);sources.append({'path':str(p),'bytes':p.stat().st_size})
        return json.loads(p.read_text(encoding='utf-8'))
    def lines(p):
        p=Path(p);sources.append({'path':str(p),'bytes':p.stat().st_size})
        with p.open(encoding='utf-8') as f:
            return [json.loads(s) for s in f if s.strip()]
    events=lines(out/'controller_events.jsonl')
    starts={x['stage']:x for x in events if x['event']=='STAGE_BEGIN'}
    ends={x['stage']:x for x in events if x['event']=='STAGE_END'}
    gpu_stages=['preflight','frozen','train-old','train-user','sample-old','sample-user']
    intervals={k:(starts[k]['time_unix'],ends[k]['time_unix']) for k in gpu_stages}
    def concurrency(names):
        marks=[]
        for k in names: marks.extend([(intervals[k][0],1),(intervals[k][1],-1)])
        v=peak=0
        for _,d in sorted(marks): v+=d;peak=max(peak,v)
        return peak
    observations=[]
    def addobs(x,origin):
        raw=x.get('gpu','')
        raw=raw.get('stdout','') if isinstance(raw,dict) else raw
        for ln in raw.splitlines():
            parts=[z.strip() for z in ln.split(',')]
            if not parts or parts[0]=='name' or parts[0].startswith('memory.'):continue
            if len(parts)==5: parts=parts[1:]
            if len(parts)!=4:continue
            try: nums=[float(re.sub(r'[^0-9.\-]','',s)) for s in parts]
            except ValueError:continue
            if any(s<0 for s in nums):continue
            pro=x.get('compute_processes',x.get('processes',x.get('gpu_processes','')))
            pro=pro.get('stdout','') if isinstance(pro,dict) else pro
            ram=x.get('mem_available_kib')
            if ram is None and isinstance(x.get('memory'),dict):
                rr=x['memory'].get('MemAvailable')
                ram=float(rr.split()[0]) if rr else None
            observations.append({'time':x.get('time_unix'),'total_mib':nums[0],'used_mib':nums[1],'free_mib':nums[2],'util_pct':nums[3],'mem_available_kib':ram,'process_text':pro,'origin':origin})
    for p in sorted((out/'resources').glob('*.json')): addobs(read(p),str(p))
    for p in sorted((out/'resources').glob('*.jsonl')):
        for x in lines(p):addobs(x,str(p))
    resources={}
    for arm in ['R2_CONT_OLD','R2_CONT_USER']:
        resources[arm]=[]
        for p in sorted((out/'training'/arm/'attempts').glob('*/resources.json')):
            d=read(p);resources[arm].append(d['summary'])
            for x in d['samples']: addobs(x,str(p))
    def obs_summary(lo=None,hi=None):
        xs=[x for x in observations if x['time'] is not None and (lo is None or lo<=x['time']) and (hi is None or x['time']<=hi)]
        result={'device_observations':len(xs),'device_scope':'Sampled whole-device nvidia-smi; includes concurrent workers/background; may miss between-sample peaks. Not process VRAM.',
                'process_vram_status':'N/A_WDDM_UNAVAILABLE','process_peak_mib':None}
        for field in ['total_mib','used_mib','free_mib','util_pct','mem_available_kib']:
            vs=[x[field] for x in xs if x[field] is not None]
            result[field+'_min']=min(vs) if vs else None
            result[field+'_max']=max(vs) if vs else None
        result['WDDM_NA_observations']=sum('[N/A]' in x['process_text'] for x in xs)
        return result
    reg=read(out/'registration.json');phases=[];details={}
    def basic(stage,label,count,unit,wall,workers,threads,scope):
        row={'stage':stage,'label':label,'wall_seconds':wall,'wall_scope':scope,'work_count':count,'work_unit':unit,
             'throughput_per_second':count/wall if wall else None,'actual_max_concurrent_wave_workers':workers,'owned_worker_processes':1 if stage in gpu_stages else workers,'cpu_threads_per_worker':threads,
             'timing_caveat':'Shared-host concurrent wall time; not isolated compute benchmark.'}
        if stage in starts:
            row['controller_seconds']=ends[stage]['seconds'];row['controller_returncode']=ends[stage]['returncode']
            row.update(obs_summary(starts[stage]['time_unix'],ends[stage]['time_unix']))
        phases.append(row);return row
    ot=read(out/'data/precompute_receipt.json')
    basic('CPU_OT','both',ot['actual_OT_returns_this_run'],'OT_return',ot['elapsed_seconds'],ot['workers'],ot['native_threads_per_worker'],'precompute receipt elapsed_seconds')
    pre=read(out/'preflight/completed.json')
    row=basic('preflight','fixed8',pre['successful_fixed_samples'],'finite_gradient_sample',pre['wall_seconds'],1,2,'preflight worker wall_seconds')
    row['recorded_resource_observation']=pre['resource_observation']
    row['recorded_resource_device_peak_mib']=pre['resource_observation']['total_device_used_peak_mib']
    row.update({'torch_peak_allocated_bytes':pre['cuda_peak_allocated_bytes'],'torch_peak_reserved_bytes':pre['cuda_peak_reserved_bytes'],'torch_peak_scope':pre['preflight_peak_scope']})
    for arm,stage in [('R2_CONT_OLD','train-old'),('R2_CONT_USER','train-user')]:
        d=read(out/'training'/arm/'completed.json')
        logs=[];calls=lines(out/'training'/arm/'calls.jsonl')
        paths=sorted((out/'training'/arm/'attempts').glob('*/training_log.jsonl'))
        for p in paths:logs.extend(lines(p))
        done=[x for x in logs if x.get('event')=='UPDATE_DONE']
        assert len(done)==d['physical_successful_updates']==1000
        assert len({x['local_step'] for x in done})==1000
        row=basic(stage,arm,len(done),'optimizer_update',d['wall_seconds'],concurrency(['train-old','train-user']),2,'worker whole-attempt wall_seconds including initialization/logging/checkpoint work')
        for key in ['cuda_peak_allocated_bytes','cuda_peak_reserved_bytes']:
            row['torch_'+key.removeprefix('cuda_')]=max(x[key] for x in done)
            row['final_progress_'+key]=d[key]
        row['torch_peak_scope']='Maximum over all 1000 recorded per-update resets (eight forward/backwards+AdamW each); excludes allocations outside those scopes; not process VRAM.'
        row['attempt_count']=len(paths);row['update_seconds_sum']=sum(x['update_seconds'] for x in done)
        row['mean_update_seconds']=row['update_seconds_sum']/len(done)
        row['recorded_resource_device_peak_mib']=max(x['total_device_used_peak_mib'] for x in resources[arm])
        details[arm]={'call_event_counts':dict(Counter(x['event'] for x in calls)),'call_event_kind_counts':dict(Counter(x['event']+':'+x.get('kind','') for x in calls)),
                      'completed_call_accounting':d['calls'],'physical_successful_updates':d['physical_successful_updates'],'physical_update_attempts':d['physical_update_attempts'],
                      'attempt_resource_summaries':resources[arm],'torch_peaks_computed_from_all_logs':[str(p) for p in paths]}
    for label,stage in [('R2_FROZEN','frozen'),('R2_CONT_OLD','sample-old'),('R2_CONT_USER','sample-user')]:
        d=read(out/'evaluation/sampling'/label/'completed.json')
        new=[x for x in d['rows'] if x.get('new_rollouts')==1]
        old=[x for x in d['rows'] if x.get('new_rollouts')==0]
        runs=[read(Path(x['result_path'])) for x in new]
        callrows=[]
        for x in new: callrows.extend(lines(Path(x['result_path']).parent/'calls.jsonl'))
        row=basic(stage,label,len(new),'new_rollout',d['seconds'],1 if label=='R2_FROZEN' else concurrency(['sample-old','sample-user']),2,'sampling worker seconds; includes cell verification/export/cache registration')
        row['historical_reuses_excluded_from_new_work']=len(old)
        row['torch_peak_allocated_bytes']=max(x['cuda_peak_allocated_bytes'] for x in runs)
        row['torch_peak_reserved_bytes']=max(x['cuda_peak_reserved_bytes'] for x in runs)
        row['torch_peak_scope']='Maximum over all NEW rollout run.json peaks, reset once per 50-Euler rollout with resident weights; excludes old reused RAW and process/context allocation.'
        row['rollout_integration_seconds_sum']=sum(x['seconds'] for x in runs)
        row['sampling_forward_reserves']=sum(x.get('event')=='RESERVE' and x.get('kind')=='sampling_forward' for x in callrows)
        row['sampling_forward_returns']=sum(x.get('event')=='RETURN' and x.get('kind')=='sampling_forward' for x in callrows)
        row['call_event_counts']=dict(Counter(x['event'] for x in callrows))
        assert row['sampling_forward_reserves']==row['sampling_forward_returns']==50*len(new)
    sc=read(out/'evaluation/parallel_score_receipt.json')
    row=basic('CPU_score','all84',sc['scored'],'scored_output',sc['total_wall_seconds'],sc['workers'],sc['worker_threads'],'parallel_score total including original cached aggregation and D_free')
    row['score_task_dispatch_wall_seconds']=sc['dispatch_wall_seconds']
    gal=read(out/'gallery/manifest.json')
    basic('render','all84',gal['PNG_count'],'PNG',gal['seconds'],1,2,'render_all seconds for original 56 PNG; controller time additionally reported')
    for name,stages,count,unit in [('training_wave',['train-old','train-user'],2000,'optimizer_update'),('endpoint_sampling_wave',['sample-old','sample-user'],56,'new_rollout')]:
        lo=min(intervals[k][0] for k in stages);hi=max(intervals[k][1] for k in stages)
        row=basic(name,'both',count,unit,hi-lo,concurrency(stages),2,'earliest controller STAGE_BEGIN to latest STAGE_END; includes startup and tails')
        row.update(obs_summary(lo,hi));row['worker_controller_seconds_sum']=sum(ends[k]['seconds'] for k in stages)
        a,b=[intervals[k] for k in stages];row['dual_worker_overlap_seconds']=max(0,min(a[1],b[1])-max(a[0],b[0]))
    endpoints_path=out/'audit/execution_endpoints.csv'
    with endpoints_path.open(encoding='utf-8-sig',newline='') as f:endpoints=list(csv.DictReader(f))
    weights=[]
    for e in endpoints:
        p=Path(e['path']);side=p.with_suffix('.json')
        d=read(side) if side.exists() else None
        if d:assert d['sha256']==e['file_sha256']
        weights.append({'arm':e['arm'],'kind':e['kind'],'path':str(p),'exists':p.is_file(),'bytes':p.stat().st_size if p.exists() else None,
                        'existing_receipt_sha256':e['file_sha256'],'sha_provenance':str(endpoints_path),'sidecar':str(side) if d else None,
                        'sidecar_agrees':d['sha256']==e['file_sha256'] if d else None,'weight_rehashed_now':False,'weight_loaded_now':False})
    for p in sorted((out/'checkpoints').glob('*local0500_global6500.json')):
        d=read(p);w=Path(d['path']);weights.append({'arm':d['arm'],'kind':'recovery500_full','path':str(w),'exists':w.is_file(),'bytes':w.stat().st_size if w.exists() else None,
                        'existing_receipt_sha256':d['sha256'],'sha_provenance':str(p),'sidecar':str(p),'sidecar_agrees':True,'weight_rehashed_now':False,'weight_loaded_now':False})
    assert len(weights)==6 and all(x['exists'] for x in weights)
    dispatch=[]
    for p in sorted((out/'dispatch').glob('*.json')):
        d=read(p);dispatch.append({k:v for k,v in d.items() if k!='resource_snapshot'})
    summary={'schema':'r2_user_finetune_runtime_summary_v1','status':'COMPLETE','generated_unix':time.time(),'phases':phases,'training_details':details,
             'all_resource_observations':obs_summary(),'device':'NVIDIA GeForce RTX 5080','available_cpu_cores_recorded':24,
             'gpu_process_vram':'N/A under WDDM; no torch allocator or whole-device number is represented as measured process VRAM.',
             'dispatch_estimates_note':'4200 MiB training dispatch proxy is conservative device-increment+margin estimate, not measured process memory. Original dispatch field name retained.',
             'dispatch_decisions':dispatch,'weight_inventory':weights,'weights_total_bytes':sum(x['bytes'] for x in weights),
             'scope_notes':['No isolated throughput benchmark claimed. Training arms and endpoint samplers overlap; summing arm seconds is not wave wall time.',
                            'Device ranges combine pre/post snapshots and periodic monitors; sampled maxima can miss short peaks and include unrelated workloads.',
                            'All training per-update logs traversed; final progress per-update peak is separately retained, never used as whole-run maximum.',
                            'CPU scoring and rendering overlap; their measured wall times are separately reported.',
                            'Existing endpoint/sidecar hashes reused. No checkpoint tensor read or heavy file hashing.'],
             'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_files':list({x['path']:x for x in sources}.values()),'audit_model_calls':0,'audit_scoring_calls':0,'audit_render_calls':0}
    audit=out/'audit'
    (audit/'runtime_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,data in [('runtime_summary.csv',phases),('runtime_weight_inventory.csv',weights)]:
        fields=list(dict.fromkeys(k for row in data for k in row))
        with (audit/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in data])
    print(json.dumps({'status':'COMPLETE','phases':[{k:x.get(k) for k in ['stage','wall_seconds','actual_max_concurrent_wave_workers','owned_worker_processes','torch_peak_allocated_bytes','torch_peak_reserved_bytes','used_mib_max','free_mib_min']} for x in phases],
                      'weights':weights,'resources':obs_summary()},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();main(a.out)
