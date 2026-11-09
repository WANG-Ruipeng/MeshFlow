"""Independent stdlib scalar audit. No model/scorer/mesh/mapping import or read."""
import ast,csv,json,hashlib,math,statistics,itertools,collections,time,sys,datetime
from pathlib import Path
RUN=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1')
REPO=Path('/mnt/e/MeshFlow-Control')
OUT=Path(__file__).resolve().parent
CAND,CTRL='C20_40_MIX','R2_MIX_H_ALL'
LABELS=(CAND,CTRL);SCOPES=('ALL','ORIGINAL6','EXTENDED2')
sources={};counts=collections.Counter();exceptions=[];table_counts={};start=time.perf_counter()
def path(p):
 p=str(p).replace('\\','/')
 return Path('/mnt/'+p[0].lower()+p[2:]) if len(p)>2 and p[1]==':' else Path(p)
def sha(p):return hashlib.sha256(path(p).read_bytes()).hexdigest()
def rj(p):
 p=path(p);sources[str(p)]=sha(p)
 return json.loads(p.read_text(encoding='utf-8-sig'))
def decode(v):
 if v=='':return None
 if v in ('True','False'):return v=='True'
 try:return json.loads(v)
 except (ValueError,TypeError):return v
def rc(p):
 p=path(p);sources[str(p)]=sha(p)
 with p.open(encoding='utf-8-sig',newline='') as f:return [{k:decode(v) for k,v in r.items()} for r in csv.DictReader(f)]
def wj(name,obj):
 p=OUT/name
 with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def wc(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def num(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def eq(a,b):
 if num(a) and num(b):return math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-10)
 if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 return type(a)==type(b) and a==b

def check(kind,actual,expected,key):
 counts[kind]+=1
 if not eq(actual,expected):exceptions.append(dict(kind=kind,key=str(key),actual=actual,expected=expected))
def need(ok,msg):
 if not ok:raise ValueError(msg)
def average(vals):return statistics.fmean(vals) if vals and all(num(x) for x in vals) else None
def delta(a,b):return a-b if num(a) and num(b) else None
def quantile(vals,p):
 if not vals:return None
 v=sorted(vals);z=(len(v)-1)*p;a=math.floor(z);b=math.ceil(z)
 return v[a]*(1-(z-a))+v[b]*(z-a)
def dist(vals,expected):
 v=[float(x) for x in vals if num(x)];full=len(v)==expected
 return dict(expected=expected,available=len(v),missing=expected-len(v),status='COMPLETE' if full else 'INCOMPLETE_METRIC_DENOMINATOR',mean=statistics.fmean(v) if full else None,available_mean=statistics.fmean(v) if v else None,median=statistics.median(v) if full else None,minimum=min(v) if full else None,maximum=max(v) if full else None,p05=quantile(v,.05) if full else None,p95=quantile(v,.95) if full else None,population_sd=statistics.pstdev(v) if full else None,negative_count=sum(x<0 for x in v),exact_zero_count=sum(x==0 for x in v),positive_count=sum(x>0 for x in v),sign_counts_are_descriptive_not_significance=True)
# Lightweight arithmetic checks contain no real data.
need(dist([1,None,3,4],4)['mean'] is None,'NA denominator test')
need(dist([1,None,3,4],4)['available_mean']==8/3,'available mean test')
need(not num(True) and quantile([0,10],.05)==.5,'type/quantile test')
need(delta(None,1) is None and average([1,2,3,4])==2.5,'pair helper test')
# No numerical input is read before all three initial freezes are validated.
manifest=rj(RUN/'audit/visual_review/confirm/review_manifest.json')
manifest_sha=sha(RUN/'audit/visual_review/confirm/review_manifest.json')
for reviewer in ('root','schedule40_data','schedule40_train'):
 folder=RUN/'audit/visual_review/confirm'/reviewer
 freeze=rj(folder/'initial_freeze_receipt.json')
 need(freeze['status']=='FROZEN_INITIAL_REVIEW' and freeze['private_mapping_read'] is False and freeze['confirmation_quality_read'] is False,'Initial freeze gate')
 need(freeze['source_review_manifest_sha256']==manifest_sha,'Freeze manifest gate')
 for rel,h in freeze['request_file_sha256'].items():need(sha(folder/rel)==h,'Initial note hash gate')
 if reviewer=='schedule40_train':
  need(freeze['counts']==dict(requests=24,outputs=48,images=48),'Assigned visual coverage')
  for p in (folder/'requests').glob('*.json'):
   doc=rj(p)
   for im in doc['images']:
    need(im['actually_viewed'] and im['all_five_views_examined'] and sha(im['path'])==im['sha256'],'Actual image evidence gate')
    sources[str(path(im['path']))]=im['sha256']
# Frozen metric names/directions only; never import analyzers with I/O.
def literals(p,wanted):
 sources[str(p)]=sha(p);tree=ast.parse(p.read_text(encoding='utf8'))
 return {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in wanted}
x=literals(REPO/'tools/analyze_recipe_confirmation_results.py',{'METRICS','NONNUMERIC'})
METRICS=list(x['METRICS']);numeric=[m for m in METRICS if m not in x['NONNUMERIC']]
DIRECTIONS=literals(REPO/'tools/analyze_recipe_factorial_results.py',{'DIRECTIONS'})['DIRECTIONS']
need(len(METRICS)==49 and len(numeric)==47,'Fixed field count')
ANA=path(rj(RUN/'analysis/confirmation/latest.json')['analysis_dir'])
receipt=rj(ANA/'analysis_receipt.json');need(receipt['status']=='COMPLETE','Analysis not complete')
for name,h in receipt['output_sha256'].items():check('analysis_output_hash',sha(ANA/name),h,name)
prov=rj(ANA/'provenance.json')
for name,item in prov['input_files'].items():check('registered_input_hash',sha(item['path']),item['sha256'],name);sources[str(path(item['path']))]=item['sha256']
for p,h in prov['sources'].items():check('analyzer_source_hash',sha(p),h,p);sources[str(path(p))]=h
regs={phase:rj(RUN/'phases'/phase/'registration.json') for phase in ('main','course','confirm')}
queries={phase:rj(RUN/'evaluation/query_registrations'/(regs[phase]['requests_sha256']+'.json')) for phase in ('main','confirm')}
qsets={phase:rj(regs[phase]['requests_path'])['rows'] for phase in ('main','confirm')}
raw={phase:rj(RUN/'phases'/phase/'evaluation/per_output.json')['rows'] for phase in ('main','course','confirm')}
old=[r for r in raw['main'] if r['label']==CTRL]+[r for r in raw['course'] if r['label']==CAND]
repeat=raw['confirm'];check('preserved_raw',rj(ANA/'all_metrics.json')['rows'],repeat,'repeat');check('preserved_raw',rj(ANA/'first_stage_pair_all_metrics.json')['rows'],old,'first')
cohort={q['parent_id']:q['condition_cohort'] for q in qsets['confirm'] if q['ratio']==.4}
need(collections.Counter(cohort.values())=={'ORIGINAL6':6,'EXTENDED2':2},'Parent strata')
def group_parents(scope):return sorted(p for p in cohort if scope=='ALL' or cohort[p]==scope)
def identity(rows,qs,phase,noises):
 qby={q['request_id']:q for q in qs};queryby={q['condition_id']:q for q in queries[phase]['rows']}
 check('request_count',len(qby),64,phase);check('output_count',len(rows),128,phase)
 check('unique_cells',len({(r['label'],r['request_id']) for r in rows}),128,phase)
 check('cell_coverage',sorted((r['label'],r['request_id']) for r in rows),sorted((l,r) for l in LABELS for r in qby),phase)
 for parent in cohort:
  for ratio in (.2,.4):check('noise_coverage',sorted(q['candidate'] for q in qs if q['parent_id']==parent and q['ratio']==ratio),list(noises),(phase,parent,ratio))
 for r in rows:
  q=qby[r['request_id']];query=queryby[q['condition_id']];key=(phase,r['label'],r['request_id'])
  check('score_status',r['status'],'PASS',key);check('score_contracts',bool(r['contracts']) and all(v is True for v in r['contracts'].values()),True,key)
  for f in ('parent_id','condition_id','candidate','ratio','condition_cohort','N','K'):check('request_identity',r[f],q[f],key+(f,))
  for f,v in [('request_id',q['request_id']),('label',r['label']),('input_sha256',q['input_sha256']),('query_sha256',query['query_sha256'])]:check('inner_identity',r['identity'][f],v,key+(f,))
  for f,v in [('query_sha256',query['query_sha256']),('registered_C_queries',2048),('registered_free_faces',q['N']-q['K']),('query_weight_denominator',query['weight_denominator']),('L0',query['identity']['L0'])]:check('denominator',r[f],v,key+(f,))
  for f in ('condition_id','parent_id','ratio','C_sha256'):check('query_C_identity',query[f],q[f],key+(f,))
  checkpoint_phase='confirm' if phase=='confirm' else 'main' if r['label']==CTRL else 'course'
  binding=rj(RUN/'phases'/checkpoint_phase/'checkpoints'/(r['label']+'.json'))
  check('checkpoint_binding',r['identity']['checkpoint_sha256'],binding['checkpoint_sha256'],key)
  check('checkpoint_registration',binding['registration_sha256'],sha(RUN/'phases'/checkpoint_phase/'registration.json'),key)
  m=r['metrics'];check('I_union',m['ConflictCov_C'],m['ConflictCov_C_cross']+m['ConflictCov_C_dup']-m['ConflictCov_C_cross_and_dup'],key)
  for f in ('ConflictCov_C','ConflictCov_C_upper','ConflictCov_C_cross','ConflictCov_C_dup','ConflictCov_C_cross_and_dup','ConflictCov_C_uncertain_fraction'):
   check('I_range',num(m[f]) and -1e-12<=m[f]<=1+1e-12,True,key+(f,))
  check('I_lower_upper',m['ConflictCov_C']<=m['ConflictCov_C_upper']+1e-12,True,key)
identity(repeat,qsets['confirm'],'confirm',(8,9,10,11));identity(old,qsets['main'],'main',(0,1,6,7))

def verify_table(name,expected,keys):
 actual=rc(ANA/(name+'.csv'));table_counts[name]=dict(actual=len(actual),expected=len(expected));check('table_count',len(actual),len(expected),name)
 tokey=lambda r:tuple(r[k] for k in keys)
 by={tokey(r):r for r in actual};check('table_unique',len(by),len(actual),name)
 check('table_keyset',sorted(by),sorted(tokey(r) for r in expected),name)
 for want in expected:
  key=tokey(want);got=by.get(key,{})
  for f,v in want.items():check(name+':'+f,got.get(f),v,key)
 return actual

def compute_tables(rows,qs,noises):
 by={(r['label'],r['request_id']):r for r in rows};groups=collections.defaultdict(list)
 for q in qs:groups[q['parent_id'],q['ratio']].append(q)
 pvalues={};parents=[];pn=[];pp=[]
 for label in LABELS:
  for (p,ratio),group in sorted(groups.items()):
   group=sorted(group,key=lambda q:q['candidate'])
   for m in numeric:
    v=[by[label,q['request_id']]['metrics'].get(m) for q in group];pvalues[label,p,ratio,m]=v
    parents.append(dict(label=label,parent_id=p,ratio=ratio,metric=m,noises=list(noises),noise_values=v,parent_membership=cohort[p],**dist(v,4)))
 for q in qs:
  a,b=by[CAND,q['request_id']],by[CTRL,q['request_id']]
  for m in numeric:
   av,bv=a['metrics'].get(m),b['metrics'].get(m)
   pn.append(dict(first=CAND,second=CTRL,request_id=q['request_id'],parent_id=q['parent_id'],ratio=q['ratio'],candidate=q['candidate'],parent_membership=cohort[q['parent_id']],metric=m,first_value=av,second_value=bv,delta=delta(av,bv)))
 for p,ratio in sorted(groups):
  for m in numeric:
   av,bv=pvalues[CAND,p,ratio,m],pvalues[CTRL,p,ratio,m];dd=[delta(a,b) for a,b in zip(av,bv)]
   pp.append(dict(first=CAND,second=CTRL,parent_id=p,ratio=ratio,parent_membership=cohort[p],metric=m,first_mean=average(av),second_mean=average(bv),delta=delta(average(av),average(bv)),noise_deltas=dd,**{'noise_'+k:v for k,v in dist(dd,4).items()}))
 pm={(r['label'],r['parent_id'],r['ratio'],r['metric']):r['mean'] for r in parents};pd={(r['parent_id'],r['ratio'],r['metric']):r['delta'] for r in pp}
 models=[];paired=[];nsummary=[];loo=[];tails=[]
 for ratio in (.2,.4):
  for scope in SCOPES:
   ps=group_parents(scope)
   for m in numeric:
    for label in LABELS:models.append(dict(label=label,ratio=ratio,metric=m,parent_stratum=scope,parent_ids=ps,**dist([pm[label,p,ratio,m] for p in ps],len(ps))))
    dd=[pd[p,ratio,m] for p in ps];direction=DIRECTIONS.get(m,'descriptive');row=dict(first=CAND,second=CTRL,ratio=ratio,metric=m,parent_stratum=scope,parent_ids=ps,parent_deltas=dd,nominal_preferred_direction=direction,**dist(dd,len(ps)))
    loov=[]
    for p in ps:
     remaining=[v for v in ps if v!=p];d=dist([pd[v,ratio,m] for v in remaining],len(ps)-1);loov.append(d['mean'])
     loo.append(dict(first=CAND,second=CTRL,ratio=ratio,metric=m,parent_stratum=scope,parent_ids=ps,omitted_parent_id=p,remaining_parent_ids=remaining,**d))
    if all(num(v) for v in dd):
     row.update(minimum_parent_id=ps[dd.index(min(dd))],maximum_parent_id=ps[dd.index(max(dd))],leave_one_parent_out_min=min(loov),leave_one_parent_out_max=max(loov))
     if direction in ('lower','lower_uncertainty','higher'):
      benefits=sorted([max(0,(-1 if direction.startswith('lower') else 1)*v) for v in dd],reverse=True);total=sum(benefits)
      row.update(top1_share_of_gross_improvement=benefits[0]/total if total else None,top2_share_of_gross_improvement=sum(benefits[:2])/total if total else None)
    paired.append(row)
    ns=[r for r in pn if r['ratio']==ratio and r['parent_id'] in ps and r['metric']==m];nds=[r['delta'] for r in ns]
    nr=dict(first=CAND,second=CTRL,ratio=ratio,metric=m,parent_stratum=scope,parent_ids=[r['parent_id'] for r in ns],**dist(nds,len(ns)))
    if all(num(v) for v in nds):nr.update(minimum_request_id=ns[nds.index(min(nds))]['request_id'],maximum_request_id=ns[nds.index(max(nds))]['request_id'])
    nsummary.append(nr)
    lower=direction.startswith('lower');defined=lower or direction=='higher';sign=-1 if lower else 1
    tv=dict(ratio=ratio,parent_stratum=scope,metric=m,direction=direction,candidate_mean=average([pm[CAND,p,ratio,m] for p in ps]),control_mean=average([pm[CTRL,p,ratio,m] for p in ps]),delta=average(dd),parent_count=len(ps),parent_better=sum(sign*v>0 for v in dd if num(v)) if defined else None,parent_worse=sum(sign*v<0 for v in dd if num(v)) if defined else None,**{k:row.get(k) for k in ('leave_one_parent_out_min','leave_one_parent_out_max','top1_share_of_gross_improvement','top2_share_of_gross_improvement')})
    if defined and all(num(v) for v in dd+nds):
     worstp=max(zip(ps,dd),key=lambda x:-sign*x[1]);worstn=max(ns,key=lambda r:-sign*r['delta'])
     tv.update(worst_parent=worstp[0],worst_parent_delta=worstp[1],worst_request=worstn['request_id'],worst_noise_delta=worstn['delta'])
    tails.append(tv)
 return dict(per_parent_metrics=parents,model_summary=models,paired_noise_metrics=pn,paired_noise_summary=nsummary,paired_parent_metrics=pp,paired_parent_summary=paired,leave_one_parent_out=loo),tails

keys=dict(per_parent_metrics=('label','parent_id','ratio','metric'),model_summary=('label','ratio','metric','parent_stratum'),paired_noise_metrics=('request_id','metric'),paired_noise_summary=('ratio','metric','parent_stratum'),paired_parent_metrics=('parent_id','ratio','metric'),paired_parent_summary=('ratio','metric','parent_stratum'),leave_one_parent_out=('ratio','metric','parent_stratum','omitted_parent_id'))
packs={};tails={}
for stage,rows,qs,noises,prefix in [('repeat',repeat,qsets['confirm'],(8,9,10,11),''),('first',old,qsets['main'],(0,1,6,7),'first_stage_')]:
 pack,tail=compute_tables(rows,qs,noises);packs[stage]=pack;tails[stage]=tail
 for name,rr in pack.items():verify_table(prefix+name,rr,keys[name])
 catalog=rc(ANA/(prefix+'metric_catalog.csv'));check('catalog_count',len(catalog),49,stage)
 check('catalog_metric_set',sorted(r['metric'] for r in catalog),sorted(METRICS),stage)
 for r in catalog:
  m=r['metric'];present=[x['metrics'][m] for x in rows if m in x['metrics']]
  check('catalog_numeric',r['numeric_summary'],m in numeric,(stage,m));check('catalog_registered',r['registered_outputs'],128,(stage,m))
  check('catalog_present',r['present'],len(present),(stage,m));check('catalog_missing',r['missing_field'],128-len(present),(stage,m))
  stat=collections.Counter('NUMERIC' if num(v) else 'NA_NULL' if v is None else json.dumps(v,sort_keys=True) for v in present)
  check('catalog_value_status',r['value_status_counts'],dict(stat),(stage,m))
  check('catalog_declared_status',r['declared_status_counts'],dict(collections.Counter(str(x.get('metric_status',{}).get(m,'UNSPECIFIED')) for x in rows)),(stage,m))
 # Every saved flattened metric value remains traceable to original scalar.
 flat=rc(ANA/('first_stage_pair_per_output.csv' if stage=='first' else 'per_output.csv'));check('flat_count',len(flat),128,stage);by={(r['label'],r['request_id']):r for r in rows}
 for r in flat:
  original=by[r['label'],r['request_id']]
  for m in METRICS:check('flat_metric',r.get('metric.'+m),original['metrics'].get(m),(stage,r['label'],r['request_id'],m))
status=[]
for label in LABELS:
 for ratio in (.2,.4):
  for scope in SCOPES:
   group=[r for r in repeat if r['label']==label and r['ratio']==ratio and r['parent_id'] in group_parents(scope)]
   for m in METRICS:
    s=collections.Counter()
    for r in group:
     if m not in r['metrics']:v='MISSING_FIELD'
     elif r['metrics'][m] is None:v='NA_NULL'
     elif num(r['metrics'][m]):v='NUMERIC'
     else:v=json.dumps(r['metrics'][m],sort_keys=True)
     s[v]+=1
    status.append(dict(label=label,ratio=ratio,parent_stratum=scope,metric=m,registered_outputs=len(group),value_status_counts=dict(s),declared_status_counts=dict(collections.Counter(str(r.get('metric_status',{}).get(m,'UNSPECIFIED')) for r in group))))
verify_table('metric_status_summary',status,('label','ratio','parent_stratum','metric'))

def diversity(stage,noises):
 cc=[];pp=[]
 for phase,label in ([('confirm',None)] if stage=='repeat' else [('main',CTRL),('course',CAND)]):
  folder=RUN/'phases'/phase/'evaluation'
  cc.extend(r for r in rc(folder/'diversity_per_condition.csv') if label is None or r['label']==label)
  pp.extend(r for r in rc(folder/'diversity_pairs.csv') if label is None or r['label']==label)
 check('diversity_conditions',len(cc),32,stage);check('diversity_pairs',len(pp),192,stage)
 group=collections.defaultdict(list)
 for r in pp:group[r['label'],r['condition_id']].append(r)
 values={};condmeans=[]
 for r in cc:
  key=r['label'],r['condition_id'];gg=group[key]
  check('diversity_six_pair_set',sorted((x['candidate_a'],x['candidate_b']) for x in gg),list(itertools.combinations(noises,2)),(stage,key))
  good=[x['D_free_pair'] for x in gg if x['status']=='PASS' and num(x['D_free_pair'])]
  mean=average(good) if len(good)==6 else None
  check('diversity_input_mean',r['D_free_mean'],mean,(stage,key));check('diversity_available',r['available_pairs'],len(good),(stage,key));check('diversity_expected',r['expected_pairs'],6,(stage,key))
  values[r['label'],r['parent_id'],r['ratio']]=mean
  condmeans.append(dict(r,independent_D_free_mean=mean))
 parents=[];models=[];summary=[];loo=[];details=[]
 for p in sorted(cohort):
  for ratio in (.2,.4):
   a,b=values[CAND,p,ratio],values[CTRL,p,ratio]
   parents.append(dict(first=CAND,second=CTRL,parent_id=p,ratio=ratio,parent_membership=cohort[p],first_D_free=a,second_D_free=b,delta=delta(a,b),relative_change=a/b-1 if num(a) and num(b) and b else None,direction='variation_not_larger_is_better'))
 by={(r['parent_id'],r['ratio']):r['delta'] for r in parents}
 for ratio in (.2,.4):
  for scope in SCOPES:
   ps=group_parents(scope);dd=[by[p,ratio] for p in ps]
   for label in LABELS:models.append(dict(label=label,ratio=ratio,parent_stratum=scope,parent_ids=ps,metric='D_free_mean',direction='variation_not_larger_is_better',**dist([values[label,p,ratio] for p in ps],len(ps))))
   summary.append(dict(first=CAND,second=CTRL,ratio=ratio,parent_stratum=scope,parent_ids=ps,direction='variation_not_larger_is_better',**dist(dd,len(ps))))
   a,b=average([values[CAND,p,ratio] for p in ps]),average([values[CTRL,p,ratio] for p in ps])
   details.append(dict(stage=stage,ratio=ratio,parent_stratum=scope,candidate_mean=a,control_mean=b,delta=delta(a,b),relative_change_of_group_means=a/b-1 if num(a) and num(b) and b else None,parents_decreased=sum(num(x) and x<0 for x in dd),parents_increased=sum(num(x) and x>0 for x in dd),direction='variation_not_larger_is_better'))
   for p in ps:
    rem=[x for x in ps if x!=p]
    loo.append(dict(first=CAND,second=CTRL,ratio=ratio,metric='D_free_mean',parent_stratum=scope,parent_ids=ps,omitted_parent_id=p,remaining_parent_ids=rem,**dist([by[x,ratio] for x in rem],len(ps)-1)))
 return dict(diversity_per_condition=cc,diversity_pairs=pp,diversity_summary=models,diversity_paired_parents=parents,diversity_paired_summary=summary,diversity_leave_one_parent_out=loo),details,condmeans
D={};DD=[]
dkeys=dict(diversity_per_condition=('label','condition_id'),diversity_pairs=('label','condition_id','candidate_a','candidate_b'),diversity_summary=('label','ratio','parent_stratum'),diversity_paired_parents=('parent_id','ratio'),diversity_paired_summary=('ratio','parent_stratum'),diversity_leave_one_parent_out=('ratio','parent_stratum','omitted_parent_id'))
for stage,noises,prefix in [('repeat',(8,9,10,11),''),('first',(0,1,6,7),'first_stage_')]:
 dp,details,cm=diversity(stage,noises);D[stage]=dp;DD.extend(details)
 for name,rr in dp.items():verify_table(prefix+name,rr,dkeys[name])
 wc(stage+'_independent_diversity_means.csv',cm)

def sign(a,b):
 if not num(a) or not num(b):return 'NA'
 if a==0 or b==0:return 'both_equal_zero' if a==b else 'one_equal_zero'
 return 'both_lower' if a<0 and b<0 else 'both_higher' if a>0 and b>0 else 'direction_reversal'
def cross_table(name,oldrows,newrows,kk,field):
 a={tuple(r[k] for k in kk):r for r in oldrows};b={tuple(r[k] for k in kk):r for r in newrows};need(a.keys()==b.keys(),'Cross effect coverage')
 rr=[]
 for key in sorted(a):
  x,y=a[key],b[key];av,bv=x[field],y[field]
  rr.append(dict(dict(zip(kk,key)),first=CAND,second=CTRL,first_stage_effect=av,independent_repeat_effect=bv,effect_change=delta(bv,av),direction_relation=sign(av,bv),first_stage_noises=[0,1,6,7],repeat_noises=[8,9,10,11],cross_experiment_output_pairing=False,first_stage_expected=x.get('expected',x.get('noise_expected')),first_stage_available=x.get('available',x.get('noise_available')),repeat_expected=y.get('expected',y.get('noise_expected')),repeat_available=y.get('available',y.get('noise_available'))))
 verify_table(name,rr,kk);return rr
cross=[]
cross.extend(cross_table('cross_repeat_parent_effects',packs['first']['paired_parent_metrics'],packs['repeat']['paired_parent_metrics'],('parent_id','ratio','metric'),'delta'))
cross.extend(cross_table('cross_repeat_group_effects',packs['first']['paired_parent_summary'],packs['repeat']['paired_parent_summary'],('ratio','metric','parent_stratum'),'mean'))
cross_table('cross_repeat_diversity_parent_effects',D['first']['diversity_paired_parents'],D['repeat']['diversity_paired_parents'],('parent_id','ratio'),'delta')
cross_table('cross_repeat_diversity_group_effects',D['first']['diversity_paired_summary'],D['repeat']['diversity_paired_summary'],('ratio','parent_stratum'),'mean')
# All independently computed rows, not only favorable columns, remain available.
wc('independent_per_noise.csv',packs['repeat']['paired_noise_metrics'])
wc('independent_per_parent.csv',packs['repeat']['paired_parent_metrics'])
wc('independent_summary_and_tails.csv',tails['repeat'])
wc('first_stage_independent_summary_and_tails.csv',tails['first'])
wc('independent_diversity_summary.csv',DD)
wc('independent_cross_repeat_effects.csv',cross)
core=('CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction')
wj('core_summaries.json',{stage:[r for r in tails[stage] if r['metric'] in core] for stage in ('first','repeat')})
wj('table_counts.json',table_counts)
sources[str(Path(__file__).resolve())]=sha(__file__)
for p,h in list(sources.items()):check('source_unchanged_after',sha(p),h,p)
wc('source_hashes.csv',[dict(path=p,sha256=h) for p,h in sorted(sources.items())])
receipt=dict(status='PASS' if not exceptions else 'FAIL',checks=sum(counts.values()),counts=dict(counts),exceptions=exceptions,source_file_count=len(sources),sources=sources,primary_rows=128,first_stage_selected_rows=128,fixed_fields=49,numeric_fields=47,paired_noise_groups=64,paired_parent_ratio_groups=16,diversity_conditions=32,diversity_pair_rows=192,wall_seconds=time.perf_counter()-start,tool_python=sys.version,torch_imported='torch' in sys.modules,new_model_calls=0,new_scoring_calls=0,new_geometry_calls=0,private_mapping_read=False,approval_rejection='An initial command also requested unnecessary private mapping; auto-review rejected before execution. Removed that read and continued authorized scalar-only audit; no retry or indirect mapping lookup.',limits=['Independent saved-scalar arithmetic only; geometry detector correctness not revalidated.','Same eight DEV parents. Training inputs and evaluation noises both changed; not new model initialization.','No aggregate score or automatic promotion.'])
wj('audit.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k not in ('sources','counts')},indent=2))
print('CORE',json.dumps([dict(ratio=r['ratio'],scope=r['parent_stratum'],metric=r['metric'],delta_pp=100*r['delta'] if num(r['delta']) else None,better=r['parent_better'],worse=r['parent_worse'],loo_pp=[100*r[k] if num(r[k]) else None for k in ('leave_one_parent_out_min','leave_one_parent_out_max')],top2=r['top2_share_of_gross_improvement'],worst_parent=r.get('worst_parent'),worst_parent_delta_pp=100*r['worst_parent_delta'] if num(r.get('worst_parent_delta')) else None) for r in tails['repeat'] if r['metric'] in core],indent=2))
print('DIVERSITY',json.dumps(DD,indent=2))
raise SystemExit(0 if not exceptions else 1)