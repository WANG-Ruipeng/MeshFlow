"""Independent CSV arithmetic and identity audit, no geometry/model execution."""
import csv,json,math,hashlib,time
from pathlib import Path
BASE=Path('/mnt/e/MeshFlow');OUT=BASE/'experiments/r2_multistep_denoise_v1';PREV=BASE/'experiments/r2_postprocess_avmr_pmr_v1';DEST=OUT/'independent_review'
META={'label','stage','request_id','condition_id','parent_id','region_type','tier','candidate','N','K'}
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open(newline='',encoding='utf-8-sig')))
def check(v,msg):
 if not v:raise AssertionError(msg)
def parse(v):
 if v in ('',None,'null','None'):return None
 if v in ('True','False'):return v=='True'
 try:return float(v)
 except (ValueError,TypeError):return v
def equal(a,b):
 if a is None or b is None:return a is b
 if isinstance(a,(float,int)) and isinstance(b,(float,int)):return abs(a-b)<=1e-10
 return a==b
def average(xs):return math.fsum(xs)/len(xs) if xs else None

def main():
 tick=time.perf_counter();reg=read(OUT/'registration.json');quality=rows(OUT/'quality_per_output.csv')
 oldraw={(r['request_id'],r['stage']):r for r in rows(PREV/'quality_per_output.csv') if r['label']=='R2_USER_MIX'}
 check(len(quality)==48,'48 quality rows');cells=0;source_records={}
 for row in quality:
  rid,stage=row['request_id'],row['stage']
  if stage=='RAW':src={k:parse(v) for k,v in oldraw[(rid,'RAW')].items() if k not in META}
  else:
   path=(PREV/'scores/R2_USER_MIX'/rid/'score.json') if stage=='D1' else (OUT/'scores'/stage/rid/'score.json')
   p=read(path)['quality'];src=dict(p['metrics'],free_face_denominator=p['free_face_denominator'])
  for field,expected in src.items():
   check(equal(parse(row[field]),expected),f'Quality CSV source mismatch {stage}/{rid}/{field}');cells+=1
  source_records[(rid,stage)]=src
 numeric={}
 for k,src in source_records.items():numeric[k]={name:v for name,v in src.items() if v is None or isinstance(v,(float,int))}
 qs=rows(OUT/'quality_summary.csv');check(len(qs)==12,'12 quality summaries');aggregate_cells=0
 for summary in qs:
  stage,group=summary['stage'],summary['group'];selected=[r for r in quality if r['stage']==stage and (group=='ALL' or r['region_type']==group)]
  check(int(summary['outputs'])==len(selected),'Group denominator')
  for field in set.intersection(*(set(numeric[(r['request_id'],stage)]) for r in selected)):
   vals=[numeric[(r['request_id'],stage)][field] for r in selected if numeric[(r['request_id'],stage)][field] is not None]
   check(equal(parse(summary[field]),average(vals)) and int(summary[field+'_valid_outputs'])==len(vals),'Quality macro aggregate '+field);aggregate_cells+=1
 paired=rows(OUT/'quality_paired_vs_D1.csv');check(len(paired)==36,'36 quality paired rows');paired_cells=0
 for row in paired:
  rid,stage=row['request_id'],row['stage'];a,b=numeric[(rid,'D1')],numeric[(rid,stage)]
  for field in set(a)&set(b):
   if a[field] is None or b[field] is None:continue
   for prefix,expected in [('D1_',a[field]),('endpoint_',b[field]),('delta_',b[field]-a[field])]:
    check(equal(parse(row[prefix+field]),expected),'Quality paired arithmetic '+prefix+field);paired_cells+=1
 parents=rows(OUT/'per_condition_noise_quality.csv');check(len(parents)==24,'24 condition-stage paired-noise rows');parent_cells=0
 for row in parents:
  stage,cid=row['stage'],row['condition_id'];selected=[r for r in quality if r['stage']==stage and r['condition_id']==cid]
  check(len(selected)==int(row['noise_outputs'])==2,'Two original noises per condition')
  for field in set.intersection(*(set(numeric[(r['request_id'],stage)]) for r in selected)):
   vals=[numeric[(r['request_id'],stage)][field] for r in selected if numeric[(r['request_id'],stage)][field] is not None]
   check(equal(parse(row[field]),average(vals)) and int(row[field+'_valid_outputs'])==len(vals),'Parent macro aggregate '+field);parent_cells+=1
 # Actual indexed-topology arithmetic independently traced back to independent geometry audit.
 independent=rows(DEST/'topology_independent.csv');it={(r['request_id'],r['stage']):r for r in independent};top=rows(OUT/'topology_summary.csv');topcells=0
 for summary in top:
  stage,group=summary['stage'],summary['group'];selected=[r for r in independent if r['stage']==stage and (group=='ALL' or r['region_type']==group)]
  check(int(summary['outputs_with_C_free_shared_edge'])==sum(float(r['C_open_edges_with_free'])>0 for r in selected),'Actual C/free attached output count')
  for field in set(selected[0])-META:
   vals=[parse(r[field]) for r in selected if parse(r[field]) is not None]
   check(equal(parse(summary[field]),average(vals)) and int(summary[field+'_valid_outputs'])==len(vals),'Topology summary '+field);topcells+=1
 tpaired=rows(OUT/'topology_paired_vs_D1.csv');check(len(tpaired)==36,'36 topology pairs')
 for row in tpaired:
  a,b=it[(row['request_id'],'D1')],it[(row['request_id'],row['stage'])]
  for field in set(a)-META:
   va,vb=parse(a[field]),parse(b[field])
   if va is None or vb is None:continue
   for prefix,expected in [('D1_',va),('endpoint_',vb),('delta_',vb-va)]:check(equal(parse(row[prefix+field]),expected),'Topology paired '+field)
 report=dict(status='PASS',quality_source_rows=48,quality_source_cells=cells,quality_macro_summaries=12,quality_aggregate_cells=aggregate_cells,paired_quality_rows=36,paired_quality_cells=paired_cells,parent_noise_rows=24,parent_noise_cells=parent_cells,topology_macro_rows=12,topology_macro_cells=topcells,topology_paired_rows=36,model_forwards=0,geometry_recomputations=0,wall_seconds=time.perf_counter()-tick,limitations='Checks saved quality metric consistency/arithmetic; does not independently reimplement CF/FF or I geometry scorers.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (DEST/'aggregate_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()