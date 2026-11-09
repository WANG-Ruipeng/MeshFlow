"""Freeze saved S2000 stages, inherited central sections and CPU resources."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import hashlib
import json
import subprocess
import time
import psutil

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_seam_s2000_section_audit_v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = ROOT / 'experiments/r2_seam_s2000_fragment_audit_v1/protocol.json'
    old = json.loads(prior.read_text(encoding='utf-8'))
    historical = ROOT / 'experiments/r2_seam_quality_wave1_wave2_v1/gallery/wave1_POST/definition.json'
    hist = json.loads(historical.read_text(encoding='utf-8'))
    def query(args):
        r = subprocess.run(args, capture_output=True, text=True)
        return dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    resources = dict(unix=time.time(), logical_cores=psutil.cpu_count(),
        available_RAM_bytes=psutil.virtual_memory().available, load_average=os.getloadavg(),
        active_python=[{'pid':p.pid,'name':p.info.get('name')} for p in psutil.process_iter(['name'])
                       if 'python' in (p.info.get('name') or '').lower()],
        gpu=query(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
        gpu_processes=query(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv']),
        GPU_scientific_workers=0, plan={'topology_workers':1,'renderer_workers_max':2,'threads_per_worker':1,
        'renderer_peak_policy':'First actual page then resource remeasure; prior comparable NumPy/PIL peak 81 MiB, leave 2 GiB RAM headroom. No GPU jobs.'})
    protocol = dict(version='s2000_sections_saved_stages_v1',status='FROZEN',requests=old['requests'],
        L=old['L'],stages=old['stages'],source_hashes=old['source_hashes'],
        previous_protocol={'path':str(prior),'sha256':sha(prior)},
        historical_definition={'path':str(historical),'sha256':sha(historical)},
        bounds=hist['bounds'],
        sections=[{'name':'front','axis':2,'position_field':'front_positions','position_index':1},
                  {'name':'side','axis':0,'position_field':'side_positions','position_index':1}],
        cutaway={'front':{'axis':2,'keep':'coordinate <= inherited central position','azimuth':90,'elevation':23},
                 'side':{'axis':0,'keep':'coordinate <= inherited central position','azimuth':0,'elevation':23},
                 'algorithm':'Exact polygon halfspace clipping of each finite triangle, no cap, no centroid face deletion. Rendering only; artificial cut rims distinguished.'},
        scope='28 saved requests, four stages, same two historical central sections and paired cutaway views. No resampling, model calls or GEO.',
        identity='POST face index maps to generated RAW slot through source_face_ids. Complete indexed POST defines connectivity; clipped parts never define whole-mesh disconnected status.',
        visual='Actual nonblind AI_PROXY observations, including uncertainty. Isolated section lines are not automatically floating faces; no GT style ranking.',
        unchanged={'model_loads':0,'model_forwards':0,'new_noise':0,'new_generation':0,'GEO_runs':0,'FF_I_rescoring':0,'commit':0,'push':0})
    for name, obj in [('resources_initial.json',resources),('protocol.json',protocol)]:
        path=OUT/name
        if path.exists():
            raise RuntimeError('Existing preparation must be reviewed; no overwrite: '+str(path))
        path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PREPARED','requests':len(old['requests']),'resources':resources},ensure_ascii=False))

if __name__ == '__main__': main()
