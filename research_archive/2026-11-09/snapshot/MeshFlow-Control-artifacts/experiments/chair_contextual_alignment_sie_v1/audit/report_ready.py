"""Report-ready views of saved contextual-alignment results; stdlib CPU only.
No scoring/model/geometry import. A final report requires complete saved inputs.
"""
from __future__ import annotations
import argparse,csv,hashlib,html,io,json,math,statistics,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

RUN=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1')
LABELS=('START','A_FM_1000','B_FREE_ALIGN_1000','C_KNOWN_ALIGN_1000')
VIEWS=('front_3q','back_3q','front','side','wire')
SIE=('SeamCov','ConflictCov_C','ExposedRetain')
KEY_METRICS=SIE+(
 'ConflictCov_C_cross','ConflictCov_C_dup','ConflictCov_C_cross_and_dup',
 'ConflictCov_C_uncertain_fraction','reference_visibility_uncertain_fraction',
 'ExposedRetain_uncertain_fraction','D_sym2048','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0',
 'fixed_query_MSE','fixed_query_p95_L','coverage_fraction',
 'CF_proper_affected_free_fraction','CF_proper_pair_count','CF_coplanar_pair_count',
 'CF_unresolved_or_degenerate_pair_count','FF_illegal_affected_free_fraction',
 'FF_illegal_pair_count','FF_proper_pair_count','FF_coplanar_pair_count',
 'FF_unresolved_or_degenerate_pair_count','free_minimum_angle_lt5_fraction',
 'free_near_degenerate_fraction','free_exact_zero_area_fraction','free_exact_duplicate_excess_fraction',
 'free_area_over_GT_free_area','EXACT_COORD_CF_shared_edges','EXACT_COORD_face_components',
 'EXACT_COORD_boundary_edges','EXACT_COORD_nonmanifold_edges','EXACT_COORD_same_direction_two_face_edges',
 'Fscore2048_0.005','Fscore2048_0.01','Fscore2048_0.02','Precision2048_0.01','Recall2048_0.01',
 'Gamma32_edge_RMS_L0','Gamma32_edge_p95_L0','SeamCov_0.005','SeamCov_0.02')
FRACTIONS=set(SIE)|{k for k in KEY_METRICS if k.endswith('_fraction') or k.startswith(('Fscore','Precision','Recall','SeamCov_'))}|{'ConflictCov_C_cross','ConflictCov_C_dup','ConflictCov_C_cross_and_dup'}
INPUTS=('analysis/results.json','analysis/sie_screen.json','analysis/checkpoint_profiles.csv',
 'analysis/comparison_summary.csv','paired_parent_metrics.csv','paired_noise_metrics.csv',
 'analysis/leave_one_parent_out.csv','analysis/control_tails.csv','analysis/diversity_per_condition.csv',
 'analysis/diversity_pairs.csv','per_output_metrics.csv','sie_summary.csv')

def resolve(x):
 s=str(x).replace('\\','/')
 return Path('/mnt/'+s[0].lower()+s[2:]) if len(s)>2 and s[1]==':' else Path(s)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write_json(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read_csv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows,fields=None):
 p.parent.mkdir(parents=True,exist_ok=True)
 names=fields or list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=names,lineterminator='\n');w.writeheader();w.writerows(rows)
def number(x):
 try:
  v=float(x)
  return v if math.isfinite(v) else None
 except (TypeError,ValueError):return None
def mean(xs):return math.fsum(xs)/len(xs) if xs and all(x is not None for x in xs) else None
def shown(x,factor=1):
 v=number(x);return 'NA' if v is None else f'{v*factor:.7g}'
def display_unit(k):
 if k in FRACTIONS:return 'percentage points for delta; percent for absolute'
 if k in ('D_sym2048','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','fixed_query_MSE'):return 'original normalized mean squared distance'
 if k.endswith(('_p95_L','_L0')):return 'original L0-normalized length'
 return 'original metric unit; see metric registry'
def enrich(r):
 k=r['metric'];factor=100 if k in FRACTIONS else 1
 d=dict(r,display_factor=factor,display_unit=display_unit(k))
 for v in ('value','lower','upper','mean_delta','median_delta','delta_lower','delta_upper','min_delta','max_delta'):
  if v in r:d['display_'+v]=number(r[v])*factor if number(r[v]) is not None else None
 return d

def readiness(out):
 p=out/'analysis/results.json'
 r=read_json(p) if p.exists() else {}
 return dict(status=r.get('status','MISSING'),scored=r.get('scored'),expected_outputs=192,
             cached=r.get('cached'),new_scored=r.get('new_scored'),diversity=r.get('diversity',{}),
             final_ready=(r.get('status')=='COMPLETE' and r.get('scored')==192 and
                          r.get('diversity',{}).get('complete_pairs')==288 and r.get('diversity',{}).get('expected_pairs')==288),
             scientific_results_emitted=False,model_calls=0,scoring_calls=0)

def final_tables(out,dest):
 state=readiness(out)
 if not state['final_ready']:raise RuntimeError('Final saved evaluation is not complete; no report numbers emitted: '+json.dumps(state))
 snapshot={p:(out/p).read_bytes() for p in INPUTS}
 hashes={p:hashlib.sha256(b).hexdigest() for p,b in snapshot.items()}
 result=json.loads(snapshot['analysis/results.json'])
 for p,h in result['source_hashes'].items():
  if digest(out/p)!=h:raise RuntimeError('Finalized analysis table hash mismatch: '+p)
 csvs={p:list(csv.DictReader(io.StringIO(b.decode('utf-8-sig')))) for p,b in snapshot.items() if p.endswith('.csv')}
 raw=csvs['per_output_metrics.csv'];counts=Counter((r['label'],r['panel']) for r in raw)
 if len(raw)!=192 or len({(r['label'],r['request_id']) for r in raw})!=192 or any(r['status']!='PASS' for r in raw):raise RuntimeError('Incomplete/duplicate saved output table')
 if counts!=Counter({(l,p):32 if p=='P1' else 16 for l in LABELS for p in ('P1','P2')}):raise RuntimeError('Saved fixed denominators differ')
 if result['cached']!=32 or result['new_scored']!=160:raise RuntimeError('Saved cache/scoring counts differ')
 profiles=csvs['analysis/checkpoint_profiles.csv']; comparisons=csvs['analysis/comparison_summary.csv']
 parents=csvs['paired_parent_metrics.csv']; noises=csvs['paired_noise_metrics.csv']
 # Algebraic consistency of saved summaries only: no geometric query is invoked.
 parent_groups=defaultdict(list)
 for r in parents:parent_groups[tuple(r[k] for k in ('contrast','panel','cohort','metric'))].append(r)
 for c in comparisons:
  pp=parent_groups[tuple(c[k] for k in ('contrast','panel','cohort','metric'))]
  expected=8 if c['panel']=='P1' else 4
  if len(pp)!=expected or len({r['parent_id'] for r in pp})!=expected:raise RuntimeError('Fixed parent denominator mismatch')
  expected_noises=4 if c['cohort'] in ('ALL','OPEN') else 2
  if any(int(r['expected_noises'])!=expected_noises for r in pp):raise RuntimeError('Fixed noise denominator mismatch')
  m=mean([number(r['delta']) for r in pp]);saved=number(c['mean_delta'])
  if (m is None)!=(saved is None) or (m is not None and not math.isclose(m,saved,rel_tol=1e-12,abs_tol=1e-14)):raise RuntimeError('Saved parent average mismatch')
 focus=lambda rows:[r for r in rows if r['metric'] in KEY_METRICS]
 for panel,scopes in (('P1',('OLD','NEW','ALL')),('P2',('OPEN',))):
  for scope in scopes:
   write_csv(dest/f'{panel}_{scope}_absolute.csv',[enrich(r) for r in focus(profiles) if r['panel']==panel and r['cohort']==scope])
   write_csv(dest/f'{panel}_{scope}_comparisons.csv',[enrich(r) for r in focus(comparisons) if r['panel']==panel and r['cohort']==scope])
 write_csv(dest/'parent_patterns.csv',focus(parents))
 write_csv(dest/'noise_patterns.csv',focus(noises))
 write_csv(dest/'leave_one_parent_out.csv',focus(csvs['analysis/leave_one_parent_out.csv']))
 tails=csvs['analysis/control_tails.csv'];write_csv(dest/'control_cost_tails.csv',tails)
 screens=json.loads(snapshot['analysis/sie_screen.json']);screen_rows=[]
 for s in screens['screens']:
  for r in s['criteria']:screen_rows.append(dict(candidate=s['candidate'],reference=s['reference'],screen_status=s['status'],**r))
 write_csv(dest/'screen_bounds.csv',screen_rows)
 write_json(dest/'screens_unchanged.json',screens)
 # Descriptive LOPO extremes preserve the original full eight-parent result.
 grouped=defaultdict(list)
 for r in focus(csvs['analysis/leave_one_parent_out.csv']):grouped[(r['contrast'],r['cohort'],r['metric'])].append(r)
 lopo_ranges=[]
 for (contrast,cohort,metric),rr in grouped.items():
  valid=[r for r in rr if number(r['LOPO_mean']) is not None]
  if len(valid)!=8:continue
  lo=min(valid,key=lambda r:number(r['LOPO_mean']));hi=max(valid,key=lambda r:number(r['LOPO_mean']))
  formal=number(rr[0]['formal_full_mean'])
  lopo_ranges.append(dict(contrast=contrast,cohort=cohort,metric=metric,formal_full_mean=formal,
    LOPO_min=number(lo['LOPO_mean']),minimum_excluded_parent=lo['excluded_parent'],
    LOPO_max=number(hi['LOPO_mean']),maximum_excluded_parent=hi['excluded_parent'],
    sign_flip_exclusions=sum(number(r['LOPO_mean'])*formal<0 for r in valid) if formal is not None else None,
    descriptive_only=True,original_gate_unchanged=True,formal_denominator=8))
 write_csv(dest/'leave_one_ranges.csv',lopo_ranges)
 extreme=[]
 for c in focus(comparisons):
  rr=[r for r in noises if r['contrast']==c['contrast'] and r['metric']==c['metric'] and r['panel']==c['panel'] and
      (c['cohort'] in ('ALL','OPEN') or r['cohort']==c['cohort']) and number(r['delta']) is not None]
  if rr:
   for kind,fn in [('MIN_POINT',min),('MAX_POINT',max)]:
    r=fn(rr,key=lambda x:number(x['delta']))
    extreme.append(dict(r,reported_cohort=c['cohort'],tail_kind=kind,threshold_status='DESCRIPTIVE_NO_ADDED_THRESHOLD'))
 write_csv(dest/'single_output_extrema.csv',extreme)
 div=csvs['analysis/diversity_per_condition.csv'];groups=defaultdict(list)
 if len(csvs['analysis/diversity_pairs.csv'])!=288 or any(r['status']!='PASS' for r in csvs['analysis/diversity_pairs.csv']):raise RuntimeError('Diversity fixed pairs incomplete')
 for r in div:groups[(r['label'],r['panel'],r['cohort'])].append(r)
 diversity=[]
 for (label,panel,cohort),rr in groups.items():
  expected=8 if panel=='P1' else 4
  if len(rr)!=expected:raise RuntimeError('Diversity conditions differ')
  vv=[number(r['value']) for r in rr];complete=all(v is not None for v in vv)
  diversity.append(dict(label=label,panel=panel,cohort=cohort,expected_conditions=expected,
    finite_conditions=sum(v is not None for v in vv),mean=mean(vv),
    condition_min=min(vv) if complete else None,condition_max=max(vv) if complete else None,
    status='COMPLETE' if complete else 'INCOMPLETE',interpretation='Variation; no preferred larger/smaller direction; not original TMD'))
 write_csv(dest/'diversity_condition_rows.csv',div);write_csv(dest/'diversity_summary.csv',diversity)
 facts=[];profile_by={(r['label'],r['panel'],r['cohort'],r['metric']):r for r in profiles}
 for c in focus(comparisons):
  av=number(profile_by[c['candidate_label'],c['panel'],c['cohort'],c['metric']]['value'])
  bv=number(profile_by[c['reference_label'],c['panel'],c['cohort'],c['metric']]['value'])
  facts.append(dict(enrich(c),candidate_group_mean=av,reference_group_mean=bv,
     ratio_of_group_means_change_percent=(100*(av-bv)/bv if av is not None and bv not in (None,0.) else None),
     relative_change_definition='ratio of equally weighted parent means; not mean of per-parent percentages'))
 write_csv(dest/'comparison_facts.csv',facts)
 contracts=('C_output_bitwise','N_preserved','finite','C_all51_bitwise','actual_Gaussian_bitwise','NPZ_C_bitwise','PLY_C_bitwise')
 summary=dict(status='COMPLETE_SAVED_RESULTS_ONLY',scored_outputs=192,cached_outputs=32,new_scored_outputs=160,
   all_inputs_sha256=hashes,upstream_bound_tables_verified=list(result['source_hashes']),
   source_counts=[dict(label=l,panel=p,count=n) for (l,p),n in sorted(counts.items())],
   primary_checkpoints='1000 only; 500 not evaluated',screens=screens,
   contracts={k:dict(Counter(r.get(k,'MISSING') for r in raw)) for k in contracts},
   comparison_facts=facts,diversity=diversity,source_statuses_preserved=True,
   interpretation=['Noise means first, then eight equal P1 parents. OLD=0/1; NEW=6/7; ALL=all four.',
    'NEW uses observed DEV parents; it is new-noise sensitivity, not independent parent confirmation.',
    'Detector bounds are not statistical confidence intervals. E comparisons use global reference-eligibility bounds.',
    'P2 has four control conditions, not four independent source shapes. S/E and source-GT distances stay NA.',
    'Numeric SIE screen is not a quality/usability or promotion gate; no ranking is produced.',
    'Parent I increases/E losses above 10pp are disclosed with possible and confirmed bounds; no invented output threshold.',
    'LOPO excludes one parent only descriptively; no gate/denominator is changed.',
    'FF/CF affected free fractions use free-face denominators and cannot be renamed whole-mesh Ri.',
    'Diversity is the saved symmetric free-surface distance, not original TMD or a larger-is-better score.'],
   human_usability='NOT_RUN',visual_review='READ_SEPARATE_ACTUAL_OBSERVATIONS',model_calls=0,scoring_calls=0)
 # Ensure capture was not interleaved with a writer before publishing the receipt.
 for p,h in hashes.items():
  if digest(out/p)!=h:raise RuntimeError('Upstream result changed during summary: '+p)
 write_json(dest/'summary.json',summary)
 lines=['# Saved-results report tables','','All values are derived from finalized saved CSV/JSON. No geometry was evaluated.','']
 for panel,scopes in (('P1',('OLD','NEW','ALL')),('P2',('OPEN',))):
  for scope in scopes:
   lines += [f'## {panel} {scope}', '', '| Comparison | Metric | Mean change | Bounds | Median | Positive/negative parents | Opposing noises |','|---|---|---:|---|---:|---|---:|']
   for c in facts:
    if c['panel']!=panel or c['cohort']!=scope or c['metric'] not in SIE+('D_sym2048','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','fixed_query_MSE','fixed_query_p95_L','CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction'):continue
    f=100 if c['metric'] in FRACTIONS else 1;suffix=' pp' if f==100 else ''
    lines.append('| '+ ' | '.join([c['contrast'],c['metric'],shown(c['mean_delta'],f)+suffix,
      '['+shown(c['delta_lower'],f)+', '+shown(c['delta_upper'],f)+']',shown(c['median_delta'],f),
      f"{c['positive_parents'] or 'NA'}/{c['negative_parents'] or 'NA'} of {c['expected_parents']}",str(c['noise_opposed_parents'])])+' |')
   lines.append('')
 lines += ['## Limits','']+['- '+x for x in summary['interpretation']]
 (dest/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 return dict(status=summary['status'],scored=192,table_directory=str(dest),model_calls=0,scoring_calls=0)

def gallery(out,dest):
 path=out/'gallery/gallery_manifest.json'
 if not path.exists():raise RuntimeError('Final gallery manifest is not present; no review template issued')
 g=read_json(path)
 if g.get('status')!='FINAL_COMPLETE' or g.get('output_count')!=192 or len(g.get('pages',[]))!=48:raise RuntimeError('Incomplete gallery')
 safe=[];blind=[];reference=[]
 # Do NOT read method_mapping.json or expose the actual labels/identity keys.
 aliases=['Method 1','Method 2','Method 3','Method 4']
 for p in g['pages']:
  if p['anonymous_order']!=aliases or p['view_scope']!=list(VIEWS):raise RuntimeError('Gallery alias/view contract differs')
  for kind in ('blind','reference'):
   if kind+'_page' in p and digest(resolve(p[kind+'_page']))!=p[kind+'_sha256']:raise RuntimeError('Gallery image hash mismatch')
  q={k:p[k] for k in ('request_id','parent_id','condition_id','panel','candidate','cohort','blind_page','blind_sha256','anonymous_order','view_scope')}
  if 'reference_page' in p:q.update(reference_page=p['reference_page'],reference_sha256=p['reference_sha256'])
  safe.append(q)
  for alias in aliases:
   base=dict(request_id=p['request_id'],parent_id=p['parent_id'],condition_id=p['condition_id'],panel=p['panel'],
      cohort=p['cohort'],candidate=p['candidate'],anonymous_method=alias,review_kind='AI_VISUAL_REVIEW',
      review_status='NOT_REVIEWED_TEMPLATE',reviewer='',observed_utc='',
      view_scope='',C_compatibility='',overall_coherence='',missing_body='',duplicate_overlap='',
      long_thin_sheet='',style_vs_fragmentation='',certainty='',render_readability='',notes='',
      human_usability='NOT_RUN')
   blind.append(dict(base,stage='SOURCE_HIDDEN',page=p['blind_page'],page_sha256=p['blind_sha256']))
   if 'reference_page' in p:reference.append(dict(base,stage='SOURCE_REFERENCE',page=p['reference_page'],page_sha256=p['reference_sha256'],source_specific_difference='',alternative_valid_completion='',blind_observation_row_id=p['request_id']+'__'+alias))
 if len(blind)!=192 or len(reference)!=128:raise RuntimeError('Review row scope differs')
 write_json(dest/'gallery_pages_blinded.json',dict(status='TEMPLATE_NOT_A_REVIEW',pages=safe,
   methods_are_anonymous=True,source_manifest_sha256=digest(path),human_usability='NOT_RUN'))
 write_csv(dest/'blind_review_template.csv',blind);write_csv(dest/'reference_review_template.csv',reference)
 for kind in ('blind','reference'):
  lines=['<!doctype html><meta charset="utf-8"><title>Anonymous '+kind+' review</title>',
    '<h1>Anonymous '+kind+' pages</h1><p>These links contain no actual model labels. Record all five views; rendering is not review. Human usability NOT_RUN.</p>']
  for p in safe:
   key=kind+'_page'
   if key not in p:continue
   rel='../../'+resolve(p[key]).relative_to(out).as_posix()
   lines.append('<p><a href="'+html.escape(rel)+'">'+html.escape(p['request_id'])+'</a></p>')
  (dest/(kind.upper()+'_INDEX.html')).write_text('\n'.join(lines),encoding='utf-8')
 return dict(status='TEMPLATES_ONLY_NO_OBSERVATIONS',blind_pages=48,reference_pages=32,
             blind_output_rows=192,reference_output_rows=128,actual_images_inspected=0)
GUIDE='''# Final saved-results reporting and actual visual review

The helper is external to the frozen repository. It imports only Python standard-library modules, reads CSV/JSON/PNG bytes, and writes only `audit/report_ready/`. It never imports scoring, model or geometry implementations and never reads `gallery/method_mapping.json`.

## Commands

```text
python -B audit/report_ready.py --stage schema
python -B audit/report_ready.py --stage tables
python -B audit/report_ready.py --stage gallery
python -B audit/report_ready.py --stage all
```

Use the registered WSL Python and absolute script path. `schema` can run before completion and records readiness only. `tables` requires saved `analysis/results.json` COMPLETE, 192 PASS outputs (32 verified START caches + 160 new scores), diversity 288/288, six upstream table hashes, exact fixed parent/noise denominators, and algebraic consistency of saved means. The scorer is never called. `gallery` requires the final 48-page/192-output manifest and verifies all 48 blind + 32 reference image hashes. It creates templates, **not reviews**. `all` performs tables and gallery sequentially.

## Output navigation

- P1_OLD/NEW/ALL absolute and comparison CSVs: existing point values and conservative bounds, plus explicitly named display columns; S/I/E fractions become percent/percentage points only in the display columns.
- P2_OPEN tables: four hand-designed control conditions. Missing GT/interface/exposure-dependent quantities remain NA, not zero or failures.
- `comparison_facts.csv`: C-A, C-B, B-A and A/B/C-START; parent means/median/sign counts/opposing-noise counts, and clearly named ratio-of-group-means relative changes. It does not substitute per-parent relative changes.
- `parent_patterns.csv`, `noise_patterns.csv`: all fixed parents/noises for the selected metrics; no parent is dropped.
- `screen_bounds.csv`/`screens_unchanged.json`: the original numerical screen and its original PASS/FAIL/INCOMPLETE states. No new threshold or promotion rule is introduced.
- `control_cost_tails.csv`: original severe-parent and output-bound tails; `single_output_extrema.csv` also retains descriptive quality/distance extremes without adding thresholds.
- `leave_one_parent_out.csv` and `leave_one_ranges.csv`: descriptive leave-one sensitivity only. Formal full-eight-parent result and gates remain unchanged.
- `diversity_condition_rows.csv`/`diversity_summary.csv`: original saved variation, not a larger-is-better score or original TMD.
- `summary.json`/`TABLES.md`: report-ready facts, denominator and interpretation caveats. `source_hashes.json` binds input bytes and the helper version; no final training/accounting claim is made.

## Gallery source schema and safe review sequence

The original `gallery/gallery_manifest.json` has top-level `status=FINAL_COMPLETE`, `identity`, `pages`, `output_count=192`, `blind_pages=48`, `reference_pages=32`, `AI_VISUAL_REVIEW`, `human_usability`, `mesh_modified`, timing and worker fields. **Do not read its `labels` or identity output keys before recording anonymous observations**, because they reveal the real method ordering. The helper reads them only to omit them; it does not load the mapping.

Each original page contains request_id, parent_id, condition_id, panel, candidate, cohort, blind_page/blind_sha256, anonymous_order, labels, view_scope and review statuses. P1 additionally has reference_page/reference_sha256. The safe `gallery_pages_blinded.json` copies only request metadata, aliases, image paths/hashes and view names, never actual labels or identity keys.

Actual layout: methods are ROWS. Blind pages are 5 rows (Input C, Method 1, Method 2, Method 3, Method 4) by 5 columns (front_3q, back_3q, front, side, wire). P1 reference pages add a sixth Source reference row. The renderer uses the same condition camera across all four methods and four noises; no mesh repair, recentering or per-method crop is applied. Blue=C, orange=free. The reference is one source completion, not the unique valid answer.

1. Open `BLIND_INDEX.html` and inspect every assigned page at readable scale, including all four views and wire. Record observations in a separate CSV copied from `blind_review_template.csv`; do not mark image inspection merely from a file/receipt existing.
2. Freeze/save blind observations before opening `REFERENCE_INDEX.html`. Record reference-specific observations separately using `reference_review_template.csv`, retaining a link to the original blind row. P2 has no reference page.
3. Only after observations are recorded, reveal the independent method mapping and add resolved labels in a derived joined review. Preserve original anonymous observations.

## Actual review CSV schema

One row = one request/noise and one anonymous model. Blind template has 192 rows; reference template 128 rows, all explicitly NOT_REVIEWED_TEMPLATE with empty observations. They are not completion evidence.

Identification: request_id, parent_id, condition_id, panel, cohort, candidate, anonymous_method, stage, page, page_sha256. Provenance: review_kind=AI_VISUAL_REVIEW, reviewer, observed_utc, view_scope (must list actually inspected front_3q/back_3q/front/side/wire), review_status. Observations: C_compatibility, overall_coherence, missing_body, duplicate_overlap, long_thin_sheet, style_vs_fragmentation, certainty, render_readability, notes. Reference-only fields: source_specific_difference, alternative_valid_completion, blind_observation_row_id.

Use specific observations and uncertainty (not_apparent/possible/evident/unable_to_judge). Different style or leg count is not automatically fragmentation or failure. C coordinate preservation is an audited geometric contract; image impressions of compatibility describe nearby completion relationships and must not override the byte checks. Human usability stays NOT_RUN; do not invent a usable-rate percentage. Seeing all images can coexist with uncertain scientific conclusions.
'''

def self_test():
 assert number('') is None and number('nan') is None and number('0')==0
 assert mean([1.,None]) is None and mean([1.,3.])==2.
 assert enrich(dict(metric='ConflictCov_C',mean_delta='0.01'))['display_mean_delta']==1.
 assert enrich(dict(metric='D_sym2048',mean_delta='0.01'))['display_mean_delta']==.01
 assert mean([None,None]) is None
 return dict(status='PASS',tests=5,scope='pure table/display semantics only',model_calls=0,geometry_calls=0)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--out',type=resolve,default=RUN)
 parser.add_argument('--stage',choices=('schema','tables','gallery','all'),default='schema')
 parser.add_argument('--self-test',action='store_true')
 args=parser.parse_args();out=args.out.resolve()
 if out!=RUN.resolve():raise ValueError('Helper writes only the registered external run')
 dest=out/'audit/report_ready';dest.mkdir(parents=True,exist_ok=True)
 (dest/'README.md').write_text(GUIDE,encoding='utf-8')
 started=time.perf_counter();result={}
 if args.self_test:
  result['self_test']=self_test();write_json(dest/'helper_test.json',result['self_test'])
 if args.stage=='schema':
  result['readiness']=readiness(out);write_json(dest/'readiness.json',result['readiness'])
  result['schema']=dict(key_metrics=list(KEY_METRICS),labels=list(LABELS),panels={'P1':['OLD','NEW','ALL'],'P2':['OPEN']},
    expected_outputs=192,scoring_calls=0,model_calls=0,visual_observations_created=0,
    source_hidden_method_aliases=['Method 1','Method 2','Method 3','Method 4'],views=list(VIEWS))
  write_json(dest/'schema.json',result['schema'])
 if args.stage in ('tables','all'):
  result['tables']=final_tables(out,dest)
  write_json(dest/'source_hashes.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
     script_path=str(Path(__file__).resolve()),script_sha256=digest(Path(__file__)),
     input_sha256={p:digest(out/p) for p in INPUTS},scope='saved result bytes only; no model or new score'))
 if args.stage in ('gallery','all'):result['gallery']=gallery(out,dest)
 receipt=dict(stage=args.stage,created_utc=datetime.now(timezone.utc).isoformat(),
   script_sha256=digest(Path(__file__)),wall_seconds=time.perf_counter()-started,
   outputs=result,model_calls=0,scoring_calls=0,render_calls=0)
 write_json(dest/(args.stage+'_receipt.json'),receipt)
 print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
