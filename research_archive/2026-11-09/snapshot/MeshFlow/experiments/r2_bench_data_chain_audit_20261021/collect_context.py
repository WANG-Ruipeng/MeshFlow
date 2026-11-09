"""Read-only source/context inventory for the authorized Bench data investigation."""
from pathlib import Path
import csv, datetime, hashlib, json, os, subprocess, time

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'experiments/r2_bench_data_chain_audit_20261021'

def dump(name, value):
    p = OUT / name
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    start = time.monotonic()
    mem = dict(x.split(':', 1) for x in Path('/proc/meminfo').read_text().splitlines())
    queries = {}
    for label, cmd in [
        ('gpu', ['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
        ('gpu_processes', ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        ('cpu_processes', ['ps','-eo','pid,comm,pcpu,pmem,args','--sort=-pcpu'])]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            queries[label] = {'returncode':p.returncode,'stdout':p.stdout[:12000],'stderr':p.stderr[:2000]}
        except Exception as exc:
            queries[label] = {'status':'UNAVAILABLE','reason':str(exc)}
    dump('resources.json', {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'cpu_logical':os.cpu_count(), 'load_average':os.getloadavg(),
        'mem_total':mem['MemTotal'].strip(), 'mem_available':mem['MemAvailable'].strip(),
        'queries':queries, 'dispatch':{'gpu_workers':0,'cpu_workers_max':3,
        'threads_per_worker':1,'phases':'Independent geometry counts and coverage joins; root joins/reads existing reviews.',
        'scope':'No neural calls, sampling, training, OT, source mutation, full finite-triangle rescore or deletion.'},
        'windows_cim_status':'UNAVAILABLE_PERMISSION',
        'wsl_initial_attempts':['Sandbox Wsl/E_ACCESSDENIED; escalated read authorized.',
            'Inline resource query had shell-quote syntax failure; no model/data work occurred. Replaced by this file.']})
    sources = [
        'experiments/r2_bench_clean_multipart_seam_v1/data/manifest.json',
        'experiments/r2_bench_clean_multipart_seam_v1/data/schedule.json',
        'experiments/r2_bench_parts_v1/data/manifest.json',
        'experiments/r2_bench_state_schedule_v1/visual_review/per_cell.csv',
        'experiments/r2_bench_state_schedule_v1/results/per_output.csv']
    dump('context_source_hashes.json', {s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
    reviews = list(csv.DictReader((ROOT/sources[3]).open(encoding='utf-8-sig',newline='')))
    final = [r for r in reviews if r['stage']=='CLEAN_FINAL_POST' and r['model']=='A_S1500']
    fields=['request_id','condition_id','parent_id','split','part_type','overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold','notes','image']
    with (OUT/'existing_A_final_reviews.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in final)
    dump('context_receipt.json', {'status':'COMPLETE','existing_review_rows':len(reviews),
        'A_final_rows':len(final),'new_visual_reviews':0,'reviewer_source':'Existing nonblind AI_PROXY records; not new human labels',
        'gpu_calls':0,'model_forwards':0,'training_updates':0,'wall_seconds':time.monotonic()-start})
    print(json.dumps({'status':'COMPLETE','A_final_rows':len(final),'cpu_logical':os.cpu_count(),'mem_available':mem['MemAvailable'].strip()}))

if __name__ == '__main__':
    main()
