"""Summarize actual recorded phase resources; no model imports or new probes."""
from pathlib import Path
import csv
import io
import json
import statistics

OUT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    protocol=read(OUT/'protocol.json');samples=[]
    with (OUT/'training/gpu_samples.jsonl').open(encoding='utf8') as f:
        for line in f:
            if line.strip():samples.append(json.loads(line))
    rows=[]
    for key in ('precheck','initial_replay','train_H_CONT','train_MOMENT_XY','eval_H_CONT','eval_MOMENT_XY'):
        controller=read(OUT/'controller'/f'{key}.json');stage=read(OUT/'stages'/f'{key}.json')
        own=[r for r in samples if r.get('pid')==controller['pid']];good=[];errors=[];process=[]
        for r in own:
            try:
                if r.get('gpu_returncode')!=0:raise ValueError('GPU sample command failed')
                used,free,util=map(float,r['gpu'].split(','));good.append((used,free,util))
                for values in csv.reader(io.StringIO(r.get('processes',''))):
                    if len(values)>=3 and values[0].strip()==str(controller['pid']):
                        try:process.append(float(values[-1]))
                        except ValueError:pass
            except (ValueError,KeyError) as exc:errors.append(dict(time_unix=r.get('time_unix'),error=str(exc)))
        result=stage.get('result',{});training=read(OUT/'training'/f'{key[6:]}_summary.json') if key.startswith('train_') else {}
        rows.append(dict(stage=key,status=stage['status'],pid=controller['pid'],gpu_workers=1,
            worker_wall_seconds=stage.get('wall_seconds'),lock_wait_seconds=stage.get('lock_wait_seconds'),
            training_loop_wall_seconds=training.get('worker_wall_seconds'),training_timing_seconds=training.get('timing_seconds'),
            checkpoint_save_verify_seconds=training.get('save_verify_seconds'),
            updates=training.get('moment_step'),training_microbatch_returns=training.get('forward_attempts_this_worker'),
            torch_peak_allocated_bytes=training.get('cuda_peak_allocated_bytes'),torch_peak_reserved_bytes=training.get('cuda_peak_reserved_bytes'),
            sampled_global_peak_used_mib=max((r[0] for r in good),default=None),
            sampled_global_min_free_mib=min((r[1] for r in good),default=None),
            sampled_process_peak_mib=max(process,default=None),process_peak_status='MEASURED' if process else 'NA_NVIDIA_SMI_PROCESS_MEMORY_UNAVAILABLE',
            valid_samples=len(good),collection_errors=errors,reserve_mib=protocol['cost_gate']['reserve_mib'],
            sampled_reserve_pass=bool(good and min(r[1] for r in good)>=protocol['cost_gate']['reserve_mib'])))
    value=dict(status='PASS' if all(r['sampled_reserve_pass'] and not r['collection_errors'] for r in rows) else 'REVIEW',phases=rows,
        timing_scope='Single GPU worker; independent CPU audits overlapped. Training loop excludes load; worker includes lock/load. Nested F/B/update timers are not additive.',
        memory_scope='Global/process nvidia-smi samples can miss brief peaks. Global used is not process VRAM. Torch peak includes allocator scope recorded by each stage.',
        precision='BF16 model forward; FP32 parameters/optimizer/states/relations. No additional GPU probes.')
    (OUT/'resources/phase_summary.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf8')
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
    with (OUT/'resources/phase_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(value['status'],[(r['stage'],r['worker_wall_seconds'],r['sampled_global_min_free_mib']) for r in rows])

if __name__=='__main__':main()
