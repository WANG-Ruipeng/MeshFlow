"""Read-only resource sampling, no CUDA initialization or model call."""
import os,time,json
from . import common as C

def main():
 start=time.monotonic();samples=0
 while True:
  phases=[C.read(p) for p in (C.OUT/'gpu_phases').glob('*.json')]
  active=[p['name'] for p in phases if p['status']=='RUNNING']
  C.event('RESOURCE_SAMPLE',ledger='resources/runtime_samples.jsonl',elapsed_seconds=time.monotonic()-start,active_phases=active,resource=C.resources(),sampling_interval_seconds=10,process_memory_note='nvidia-smi may report N/A under WDDM; total GPU used is not experiment process memory')
  samples+=1
  if (C.OUT/'benchmark/completion.json').exists():break
  failures=[p for p in phases if p['status'] in ('FAILED','STOPPED_GPU_BUDGET') and p['name']!='generate_training_raw']
  if failures:break
  if time.monotonic()-start>14460:break
  time.sleep(10)
 C.write(C.OUT/'resources/observer_completion.json',dict(status='COMPLETE',samples=samples,wall_seconds=time.monotonic()-start,model_calls=0,scope='Starts during RAW preparation, includes subsequent teacher/student phases; sampled extrema are not guaranteed instantaneous process peaks'))
if __name__=='__main__':main()