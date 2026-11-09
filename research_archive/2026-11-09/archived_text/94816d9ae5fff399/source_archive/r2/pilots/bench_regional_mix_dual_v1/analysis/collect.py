"""Independent arithmetic/accounting from completed frozen CPU results.

No model, detector, GEO, score, rendering, best selection, or visual labels.
The original 177 fields are copied unchanged; every derived denominator is explicit.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse,csv,hashlib,json,math,statistics,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'experiments/r2_bench_regional_mix_dual_v1'
DEST=OUT/'analysis'
LABELS=('D0_ALL','RAW_ALL','ASSIST10','ASSIST01','ASSIST11','AUTO_COVER')
STAGES=('BASE_POST','CLEAN_FINAL_POST')
CORE=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','CF_illegal_pair_count','FF_illegal_pair_count',
      'ConflictCov_C','C_F_valid_edge_count','C_F_valid_edge_total_length','C_F_has_valid_edge',
      'actual_free_faces','deleted_free_faces','free_minimum_angle_lt5_fraction','free_q_gt10_fraction',
      'nonmanifold_edges','hinge_diag_isolated_1face_edge_components','hinge_diag_legal_opposed_edge_C_reached_free_area_fraction')

def utc():return datetime.now(timezone.utc).isoformat()
def resolve(p):
    s=str(p).replace('\\','/')
    for prefix in ('E:/MeshFlow/','/mnt/e/MeshFlow/'):
        if s.startswith(prefix):return ROOT/s[len(prefix):]
    path=Path(s);return path if path.is_absolute() else ROOT/path
def read(p):return json.loads(resolve(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with resolve(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def csvread(p):
    with resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,x):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvwrite(p,rows):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})
def number(v):
    if str(v) in ('True','False'):return float(str(v)=='True')
    try:x=float(v)
    except (TypeError,ValueError):return None
    return x if math.isfinite(x) else None
def avg(vs):
    good=[v for v in vs if v is not None];return statistics.mean(good) if good else None
def divide(a,b):return a/b if b else None
def areas(tri):
    t=tri.astype(np.float64);return .5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
def arrays(p):
    with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def checked_file(p,expected,book):
    h=sha(p)
    if h!=expected:raise ValueError('Hash differs: '+str(p))
    book[str(resolve(p))]=h
def mesh_slots(p,k,n,book):
    z=arrays(p);book[str(resolve(p))]=sha(p)
    ids=z.get('parent_source_raw_slot',z.get('source_face_ids'))
    if ids is None or len(ids)!=len(z['output']):raise ValueError('Missing RAW face mapping')
    originals=ids[ids>=0].astype(int)
    if len(set(originals.tolist()))!=len(originals) or any(v>=n for v in originals):raise ValueError('Invalid original RAW slots')
    aa=areas(z['output']);lookup={int(i):float(a) for i,a in zip(ids,aa) if i>=0}
    if not set(range(k))<=set(lookup):raise ValueError('Known source face lost')
    return lookup,float(aa[ids<0].sum()),int((ids<0).sum()),float(aa[k:].sum())

def ledger_state(l,new):
    """Recover exact sets from an existing D0-vs-candidate ledger, no scan."""
    edge={kind:{} for kind in ('CF','FF')};pairs={kind:set() for kind in ('CF','FF')}
    for r in l['edge_fates']:
        if r['fate'] in (('RETAINED','NEW') if new else ('RETAINED','LOST')):
            key=tuple(tuple(int(v) for v in part) for part in r['source_edge_key'])
            edge[r['kind']][key]=r['new_length' if new else 'old_length']
    for r in l['illegal_pair_fates']:
        if r['fate'] in (('PERSISTENT','NEW') if new else ('PERSISTENT','RESOLVED_RETAINED','FACE_REMOVED')):
            pairs[r['kind']].add((int(r['raw_slot_a']),int(r['raw_slot_b'])))
    faces={int(r['raw_slot']) for r in l['face_fates'] if r['fate'] in (('RETAINED','NEWLY_RETAINED') if new else ('RETAINED','LOST'))}
    return dict(edges=edge,pairs=pairs,faces=faces)

def compare_ledgers(a,b,meta):
    row=dict(meta);details=[]
    for kind in ('CF','FF'):
        ae,be=a['edges'][kind],b['edges'][kind];aa,bb=set(ae),set(be);ap,bp=a['pairs'][kind],b['pairs'][kind]
        gone=ap-bp;gone_removed={p for p in gone if not set(p)<=b['faces']};gone_retained=gone-gone_removed
        row.update({kind+'_old_legal_edges':len(aa),kind+'_retained_legal_edges':len(aa&bb),kind+'_lost_legal_edges':len(aa-bb),
                    kind+'_new_legal_edges':len(bb-aa),kind+'_lost_length':sum(ae[e] for e in aa-bb),kind+'_new_length':sum(be[e] for e in bb-aa),
                    kind+'_original_slot_old_illegal_pairs':len(ap),kind+'_original_slot_new_total_illegal_pairs':len(bp),
                    kind+'_persistent_illegal_pairs':len(ap&bp),kind+'_new_illegal_pairs':len(bp-ap),
                    kind+'_resolved_illegal_pairs_retained':len(gone_retained),kind+'_illegal_no_longer_observable_deleted':len(gone_removed)})
        for edge in sorted(aa|bb):details.append(dict(meta,entity='LEGAL_EDGE',kind=kind,key=edge,
            fate='RETAINED' if edge in aa&bb else 'LOST' if edge in aa else 'NEW',old_length=ae.get(edge),new_length=be.get(edge)))
        for pair in sorted(ap|bp):details.append(dict(meta,entity='ILLEGAL_PAIR',kind=kind,key=pair,
            fate='PERSISTENT' if pair in ap&bp else 'NEW' if pair in bp else 'FACE_REMOVED' if pair in gone_removed else 'RESOLVED_RETAINED'))
    row.update(original_faces_lost=len(a['faces']-b['faces']),original_faces_newly_retained=len(b['faces']-a['faces']))
    return row,details

def run():
    if 'torch' in sys.modules:raise RuntimeError('No model imports permitted')
    if (DEST/'COMPLETE.json').exists():
        done=read(DEST/'COMPLETE.json')
        for p,h in done['files'].items():
            if sha(p)!=h:raise ValueError('Existing analysis changed')
        return done
    reg=read(OUT/'jobs.json');rh=sha(OUT/'jobs.json');collection=read(OUT/'results/complete.json')
    if collection.get('status')!='COMPLETE' or collection['registration_sha256']!=rh:raise RuntimeError('Wait for complete registered results')
    if collection['collected_rows']!=72:raise ValueError('Expected 6 labels x 6 requests x 2 stages')
    fields=reg['metric_fields'];assert len(fields)==len(set(fields))==177
    panel=read(OUT/'panel/manifest.json');qmap={e['request_id']:e['request'] for e in panel['requests']};regions={e['request_id']:e['regions'] for e in panel['requests']}
    book={str(OUT/'jobs.json'):rh,str(OUT/'panel/manifest.json'):sha(OUT/'panel/manifest.json'),str(OUT/'results/complete.json'):sha(OUT/'results/complete.json')}
    checked_file(OUT/'results/per_output.csv',collection['files'][str(OUT/'results/per_output.csv')],book)
    rawrows=csvread(OUT/'results/per_output.csv');rows=[]
    for r in rawrows:
        if any(f not in r for f in fields):raise ValueError('Missing original metric field')
        row=dict(r);row['runtime_label']=r['label'];row['label']='D0_ALL' if r['label']=='M_D0' else r['label'];rows.append(row)
    keys={(r['label'],r['request_id'],r['stage']) for r in rows}
    expected={(label,rid,stage) for label in LABELS for rid in qmap for stage in STAGES}
    if keys!=expected or len(rows)!=72:raise ValueError('Duplicate or missing output identity')
    by={(r['label'],r['request_id'],r['stage']):r for r in rows};parents=sorted({q['parent_id'] for q in qmap.values()})
    # Every original field is preserved verbatim; this is only a display-label alias.
    for original,copy in zip(rawrows,rows):assert all(original[f]==copy[f] for f in fields)
    commonI={stage:{rid for rid in qmap if all(by[label,rid,stage].get('I_status')=='PASS' and number(by[label,rid,stage].get('ConflictCov_C')) is not None for label in LABELS)} for stage in STAGES}
    support=[];perparent=[]
    for stage in STAGES:
        for label in LABELS:
            for parent in parents:
                rr=[r for r in rows if r['label']==label and r['stage']==stage and r['parent_id']==parent]
                assert len(rr)==2
                out=dict(label=label,stage=stage,parent_id=parent,report_group=rr[0]['report_group'],request_count=2,request_ids=[r['request_id'] for r in rr])
                for field in fields:
                    vs=[number(r[field]) for r in rr];valid=[v for v in vs if v is not None]
                    out[field]=avg(vs) if valid else None
                    support.append(dict(label=label,stage=stage,parent_id=parent,field=field,numeric_n=len(valid),missing_or_nonnumeric_n=2-len(valid),original_value_counts=dict(Counter(r[field] for r in rr))))
                ii=[r for r in rr if r['request_id'] in commonI[stage]]
                out['I_common_PASS']=avg([number(r['ConflictCov_C']) for r in ii]);out['I_common_PASS_requests']=len(ii)
                out['I_original_status_counts']=dict(Counter(r.get('I_status','MISSING') for r in rr));perparent.append(out)
    jobs={j['job_id']:j for j in reg['jobs']};receipts={};artifacts={};ledger_states={};boundaries={};selection={}
    for jid,j in jobs.items():
        cp=OUT/'runs'/jid/'complete.json';r=read(cp)
        if r.get('status')!='COMPLETE' or r['registration_sha256']!=rh or r['request_id']!=j['request']['request_id']:raise ValueError('Incomplete/rebound job')
        if r['model_calls'] or r['D0_calls']:raise ValueError('Unexpected model call')
        checked_file(cp,collection['input_job_receipts'][str(cp)],book);receipts[jid]=r
        rid=j['request']['request_id'];label=j['label'];artifacts[label,rid]=r['artifacts']
        for p in (j['selection_summary'],j['mixed_boundary']):book[str(resolve(p))]=sha(p)
        selection[label,rid]=read(j['selection_summary']);boundary=read(j['mixed_boundary']);boundaries[label,rid]=boundary
        for stage in STAGES:
            p=r['artifacts']['ledger_'+stage];checked_file(p,r['files'][str(resolve(p))],book);ledger=read(p)
            ledger_states[label,rid,stage]=ledger_state(ledger,True)
            old=ledger_state(ledger,False);key=('D0_ALL',rid,stage)
            if key in ledger_states and ledger_states[key]!=old:raise ValueError('Different D0 ledger identity')
            ledger_states[key]=old
    # The original D0 baseline is only read; derive its same-slot area table arithmetically.
    refs={t['request_id']:t for t in reg['references']};area_rows=[];region_rows=[];slot_rows=[];boundary_rows=[]
    for rid,q in qmap.items():
        k,n=int(q['K']),int(q['N']);rp=refs[rid]['paths'];raw=arrays(rp['raw'])['raw'];rawarea=areas(raw);rawtotal=float(rawarea[k:].sum());book[rp['raw']]=sha(rp['raw'])
        for label in LABELS:
            if label=='D0_ALL':
                sel=dict(selected_RAW_faces=0,free_faces=n-k,selected_RAW_fraction=0.,selected_RAW_area=0.,raw_free_area=rawtotal,selected_RAW_area_fraction=0.,mixed_boundary_clusters=0,RAW_D0_boundary_clusters=0,noncoincident_RAW_D0_clusters=0,C_bytes_preserved=True)
                b=[]
            else:sel=selection[label,rid];b=boundaries[label,rid]
            gaps=[float(x['maximum_gap']) for x in b if x['D0_flat_corners']]
            boundary_rows.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],label=label,
                **{key:sel[key] for key in ('selected_RAW_faces','free_faces','selected_RAW_fraction','selected_RAW_area','raw_free_area','selected_RAW_area_fraction','mixed_boundary_clusters','RAW_D0_boundary_clusters','noncoincident_RAW_D0_clusters','C_bytes_preserved')},
                mixed_cluster_containing_D0_maximum_gap_including_C=max(gaps) if gaps else None,
                mixed_cluster_containing_D0_mean_maximum_gap_including_C=avg(gaps),
                definition='Original RAW-GEO cluster co-membership. For clusters containing D0, cached maximum_gap compares RAW to D0 AND C members; C may dominate. Cached noncoincident_RAW_D0_clusters tests all members including C. These are not pure RAW-to-D0 gap measurements or final-connectivity claims.'))
            runtime_slots=None
            if label!='D0_ALL':
                j=next(j for j in jobs.values() if j['label']==label and j['request']['request_id']==rid)
                sp=OUT/'runs'/j['job_id']/'per_slot_deletion_area_displacement.csv';checked_file(sp,receipts[j['job_id']]['files'][str(sp)],book)
                runtime_slots={(r['stage'],int(r['raw_slot'])):r for r in csvread(sp)}
            for stage in STAGES:
                mesh=rp['original_base' if stage=='BASE_POST' else 'original_final'] if label=='D0_ALL' else artifacts[label,rid]['base' if stage=='BASE_POST' else 'final']
                lookup,bridgearea,bridgecount,actualfree=mesh_slots(mesh,k,n,book)
                kept=set(lookup)&set(range(k,n));gone=set(range(k,n))-kept
                retainedarea=float(rawarea[list(kept)].sum()) if kept else 0.;deletedarea=rawtotal-retainedarea
                meta=dict(request_id=rid,parent_id=q['parent_id'],report_group=q['report_group'],noise_index=q['noise_index'],label=label,stage=stage)
                area_rows.append(dict(meta,raw_free_faces=n-k,retained_original_free_faces=len(kept),deleted_original_free_faces=len(gone),
                    raw_free_area=rawtotal,retained_RAW_reference_area=retainedarea,deleted_RAW_reference_area=deletedarea,
                    retained_RAW_reference_area_fraction=divide(retainedarea,rawtotal),deleted_RAW_reference_area_fraction=divide(deletedarea,rawtotal),
                    actual_original_free_area=sum(lookup[s] for s in kept),added_bridge_faces=bridgecount,added_bridge_area=bridgearea,
                    actual_free_area_including_bridges=actualfree,actual_free_area_over_RAW=divide(actualfree,rawtotal)))
                for slot in range(k,n):
                    if runtime_slots is not None:
                        oldslot=runtime_slots[stage,slot]
                        if (oldslot['retained']=='True')!=(slot in kept) or not math.isclose(float(oldslot['raw_area']),float(rawarea[slot]),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Slot ledger mismatch')
                    slot_rows.append(dict(meta,raw_slot=slot,retained=slot in kept,raw_reference_area=float(rawarea[slot]),actual_output_area=lookup.get(slot),
                        assisted_region=next((r['region_id'] for r in regions[rid] if slot in r['source_face_ids']),'')))
                for region in regions[rid]:
                    ids=set(region['source_face_ids']);rs=ids&kept;total=float(rawarea[list(ids)].sum());survive=float(rawarea[list(rs)].sum()) if rs else 0.
                    region_rows.append(dict(meta,region_id=region['region_id'],source_RAW_faces=len(ids),retained_original_faces=len(rs),deleted_original_faces=len(ids-rs),
                        RAW_reference_area=total,retained_RAW_reference_area=survive,retained_RAW_reference_area_fraction=divide(survive,total),
                        actual_region_output_area=sum(lookup[s] for s in rs),actual_region_area_over_RAW=divide(sum(lookup[s] for s in rs),total),
                        retained_RAW_slots=sorted(rs),deleted_RAW_slots=sorted(ids-rs),
                        limitation='Slot survival and reference-area mass do not certify support completeness or unchanged shape. Bridges excluded from original region identity.'))
    comparisons=[];ledger_details=[];paired=[]
    for rid,q in qmap.items():
        for stage in STAGES:
            for reference in ('D0_ALL','RAW_ALL'):
                for label in LABELS:
                    if reference==label:continue
                    meta=dict(request_id=rid,parent_id=q['parent_id'],report_group=q['report_group'],noise_index=q['noise_index'],stage=stage,reference=reference,label=label)
                    lr,ld=compare_ledgers(ledger_states[reference,rid,stage],ledger_states[label,rid,stage],meta);comparisons.append(lr);ledger_details.extend(ld)
                    a,b=by[reference,rid,stage],by[label,rid,stage];p=dict(meta,I_common_all_six_PASS=rid in commonI[stage])
                    for field in fields:
                        av,bv=number(a[field]),number(b[field]);p['delta_'+field]=bv-av if av is not None and bv is not None else None
                    p['delta_I_common_PASS']=p['delta_ConflictCov_C'] if rid in commonI[stage] else None
                    p['reference_I_status']=a.get('I_status');p['candidate_I_status']=b.get('I_status');paired.append(p)
    area_by={(r['label'],r['request_id'],r['stage']):r for r in area_rows}
    tradeoff=[]
    for r in rows:
        key=r['label'],r['request_id'],r['stage'];ar=area_by[key];s=next(v for v in boundary_rows if v['label']==r['label'] and v['request_id']==r['request_id'])
        tradeoff.append(dict(**{k:r[k] for k in ('label','request_id','parent_id','report_group','stage')},
            **{k:number(r[k]) for k in CORE if k in r},I_status=r['I_status'],I_common_PASS=number(r['ConflictCov_C']) if r['request_id'] in commonI[r['stage']] else None,
            **{k:ar[k] for k in ('retained_RAW_reference_area_fraction','actual_free_area_over_RAW','deleted_original_free_faces','added_bridge_faces')},
            **{k:s[k] for k in ('selected_RAW_fraction','selected_RAW_area_fraction','RAW_D0_boundary_clusters','noncoincident_RAW_D0_clusters','mixed_cluster_containing_D0_maximum_gap_including_C')}))
    summary=dict(status='COMPLETE',scope='Six repeatedly viewed development requests, three parents. All assisted combinations remain separate; no selected-best method or automatic-success claim.',
                 stages={},definitions=dict(primary_average='First mean of valid noises within each parent, then equal mean across parents. Every valid count is recorded.',
                    I='ConflictCov_C main comparison uses request intersection with I_status PASS and finite value for all six labels, separately per stage. Original unfiltered177 fields remain untouched.',
                    area='RAW-reference mass tracks retained original slots. Actual output area can change under D0/GEO and includes separately reported bridges. These quantities are not interchangeable.',
                    connectivity='Shared original RAW slot/corner ledger only; negative-ID bridges excluded from correspondence, included in full177 scorers.',
                    region='Frozen assisted support regions, not automatic masks or clean-source target labels.',
                    boundary='Cached gaps for mixed clusters containing D0 still include RAW-to-C member comparisons. C may dominate maximum gap and all-member noncoincidence. No pure RAW-to-D0 or final crack interpretation.'),accounting=dict(model_calls=0,scoring_calls=0,detector_calls=0,GEO_calls=0,new_visual_labels=0))
    groups=['ALL',*sorted({q['report_group'] for q in qmap.values()})]
    for stage in STAGES:
        summary['stages'][stage]={}
        for group in groups:
            ids={rid for rid,q in qmap.items() if group=='ALL' or q['report_group']==group};pids=sorted({qmap[rid]['parent_id'] for rid in ids});ii=sorted(ids&commonI[stage]);ipids=sorted({qmap[rid]['parent_id'] for rid in ii})
            gg=dict(parent_count=len(pids),request_count=len(ids),I_common_PASS_request_ids=ii,I_common_PASS_requests=len(ii),I_common_PASS_parent_ids=ipids,I_common_PASS_parents=len(ipids),labels={})
            for label in LABELS:
                rr=[r for r in perparent if r['label']==label and r['stage']==stage and r['parent_id'] in pids];oo=[by[label,rid,stage] for rid in ids]
                metrics={f:dict(parent_equal_mean=avg([r[f] for r in rr]),valid_parent_count=sum(r[f] is not None for r in rr),valid_request_count=sum(number(r[f]) is not None for r in oo)) for f in fields}
                iv=[r['I_common_PASS'] for r in rr if r['I_common_PASS_requests']]
                area_macro={f:avg([avg([a[f] for a in area_rows if a['label']==label and a['stage']==stage and a['parent_id']==parent]) for parent in pids]) for f in ('retained_original_free_faces','deleted_original_free_faces','deleted_RAW_reference_area_fraction','retained_RAW_reference_area_fraction','actual_free_area_over_RAW','added_bridge_faces')}
                gg['labels'][label]=dict(metrics=metrics,I_common_PASS_parent_mean=avg(iv),I_original_status_counts=dict(Counter(r.get('I_status','MISSING') for r in oo)),
                    absolute_CF_pair_sum=sum(number(r['CF_illegal_pair_count']) or 0 for r in oo),absolute_FF_pair_sum=sum(number(r['FF_illegal_pair_count']) or 0 for r in oo),
                    C_exact_status_counts=dict(Counter(r.get('C_preserved_exact','MISSING') for r in oo)),area_parent_equal_means=area_macro)
            summary['stages'][stage][group]=gg
    # Only now create outputs, after all inputs and expected 72 identities validated.
    if DEST.exists():raise RuntimeError('Partial analysis retained; do not overwrite')
    DEST.mkdir()
    for name,table in [('per_output_full177.csv',rows),('per_parent.csv',perparent),('per_parent_field_support.csv',support),('paired_per_request.csv',paired),
                       ('area_costs_per_output.csv',area_rows),('region_survival.csv',region_rows),('per_slot_area_accounting.csv',slot_rows),
                       ('boundary_per_output.csv',boundary_rows),('candidate_tradeoff.csv',tradeoff),('connectivity_pairwise.csv',comparisons),('connectivity_pairwise_details.csv',ledger_details)]:csvwrite(DEST/name,table)
    write(DEST/'summary.json',summary)
    receipt=dict(status='COMPLETE',utc=utc(),version='regional_mix_readonly_accounting_v1',registration_sha256=rh,
        original_metric_fields=177,per_output_rows=len(rows),parent_rows=len(perparent),request_pairs=len(paired),region_rows=len(region_rows),
        source_code_sha256=sha(__file__),source_files=book,model_calls=0,scoring_calls=0,detector_calls=0,GEO_calls=0,new_visual_labels=0,
        files={str(p):sha(p) for p in DEST.iterdir() if p.is_file()})
    write(DEST/'COMPLETE.json',receipt);return receipt

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('run',));ap.parse_args()
    result=run();print(json.dumps({k:v for k,v in result.items() if k not in ('files','source_files')},indent=2))
