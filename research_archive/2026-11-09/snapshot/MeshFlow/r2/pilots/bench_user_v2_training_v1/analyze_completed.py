"""Read completed evidence, independent aggregates, contact sheets. No model/scoring."""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from .common import ROOT, OUT, ARM, LABEL, read, resolve, sha, write_new

DEST=OUT/'analysis'
STAGES=('RAW','BASE_POST','CLEAN_FINAL_POST')
KEYS=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
 'free_minimum_angle_lt5_fraction','CF_illegal_pair_count','FF_illegal_pair_count',
 'actual_free_faces','deleted_free_faces','free_total_surface_area','nonmanifold_edges',
 'hinge_diag_isolated_1face_edge_components','C_F_valid_edge_count','C_F_valid_edge_total_length',
 'legal_seam_bulk_edge_count','free_only_reached_bulk_area_fraction','free_exact_zero_area_count')

def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def csvwrite(p, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    import io
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    text=f.getvalue().encode('utf-8-sig')
    if p.exists():assert p.read_bytes()==text,p
    else:p.write_bytes(text)

def num(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):return None

def jl(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]

def mean(a):return sum(a)/len(a) if a else None

def data():
    files=[OUT/'results/per_output.csv',ROOT/'experiments/r2_bench_coverage_expansion_v1/results/per_output.csv']
    new,cov=map(csvread,files)
    # Identical historical A evidence must not be counted twice.
    def key(r):return r['label'],r['request_id'],r['stage']
    a={key(r):r for r in new if r['label']=='A_S1500'}
    for r in cov:
        if r['label']=='A_S1500':assert a[key(r)]==r
    rows=new+[r for r in cov if r['label']=='COVERAGE_S1500']
    assert len(rows)==432 and len({key(r) for r in rows})==432
    member={r['request_id']:r for r in read(OUT/'data/evaluation_membership.json')}
    for r in rows:
        r['in_current_training_parent_pool']=member[r['request_id']]['in_current_training_parent_pool']
        assert r['C_preserved_exact']=='True'
    tables=[];parents=[];parts=[]
    for grouping,target in [(('label','stage','split'),tables),(('label','stage','split','parent_id'),parents),
                            (('label','stage','split','part_type'),parts)]:
        groups=collections.defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in grouping)].append(r)
        for g,rr in sorted(groups.items()):
            o=dict(zip(grouping,g));o.update(requests=len(rr),parents=len({r['parent_id'] for r in rr}),
                applicable_requests=sum(r['seam_applicability']=='APPLICABLE' for r in rr))
            for k in KEYS:
                keep=[r for r in rr if num(r[k]) is not None and (k!='free_only_reached_bulk_area_fraction' or r['seam_applicability']=='APPLICABLE')]
                per=collections.defaultdict(list)
                for r in keep:per[r['parent_id']].append(float(r[k]))
                o[k]=mean([mean(v) for v in per.values()]);o[k+'__valid_requests']=len(keep)
            applicable=[r for r in rr if r['seam_applicability']=='APPLICABLE']
            o['bulk_ge30_count']=sum((num(r['free_only_reached_bulk_area_fraction']) or 0)>=.3 for r in applicable)
            o['has_CF_valid_count']=sum((num(r['C_F_valid_edge_count']) or 0)>0 for r in rr)
            o['I_status_counts']=json.dumps(dict(collections.Counter(r['ConflictCov_C_status'] for r in rr)),sort_keys=True)
            target.append(o)
    pair=[]
    by={(r['label'],r['stage'],r['split'],r['parent_id']):r for r in parents}
    for ref in ('A_S1500','COVERAGE_S1500'):
        for r in parents:
            if r['label']!=LABEL:continue
            old=by[ref,r['stage'],r['split'],r['parent_id']]
            p=dict(reference=ref,candidate=LABEL,stage=r['stage'],split=r['split'],parent_id=r['parent_id'])
            for k in KEYS:
                p[k+'__reference']=old[k];p[k+'__candidate']=r[k]
                p[k+'__delta']=r[k]-old[k] if r[k] is not None and old[k] is not None else None
            pair.append(p)
    for name,table in [('three_models_per_output',rows),('key_metrics',tables),('per_parent',parents),('per_part',parts),('paired_parents',pair)]:
        csvwrite(DEST/(name+'.csv'),table)
    write_new(DEST/'source_tables.json',{str(p):sha(p) for p in files})
    return tables

def training():
    log=jl(OUT/'training'/ARM/'training_log.jsonl')
    expo=jl(OUT/'training'/ARM/'exposure.jsonl')
    assert len(log)==1500 and [r['local_step'] for r in log]==list(range(1,1501))
    sch=read(OUT/'data/schedule.json')['steps']
    expected=[r for s in sch for r in s['samples']]
    assert len(expo)==len(expected)==12000
    for a,b in zip(expo,expected):
        for k,v in b.items():assert a[k]==v,(k,a.get(k),v)
    groups=collections.Counter(r['parent_id'] for r in expo)
    assert len(groups)==25 and set(groups.values())=={480}
    curve=[]
    for start in range(0,1500,100):
        rr=log[start:start+100];o=dict(first_step=start+1,last_step=start+100,objective=mean([r['objective'] for r in rr]))
        for kind in ('all_free','bulk','seam_tip'):
            den=sum(r[kind+'_scalar_count'] for r in rr)
            o[kind+'_velocity_MSE']=sum(r[kind+'_velocity_sse'] for r in rr)/den if den else None
            o[kind+'_scalars']=den
        curve.append(o)
    csvwrite(DEST/'training_windows.csv',curve)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ax[0].plot([r['local_step'] for r in log],[r['objective'] for r in log],alpha=.2,lw=.5,label='each update')
    ax[0].plot([r['last_step'] for r in curve],[r['objective'] for r in curve],label='100-update mean')
    for k in ('bulk','seam_tip'):
        ax[1].plot([r['last_step'] for r in curve],[r[k+'_velocity_MSE'] for r in curve],label=k)
    for a in ax:a.grid(alpha=.2);a.set_xlabel('Local optimizer updates');a.legend()
    ax[0].set_ylabel('FM objective');ax[1].set_ylabel('sum SSE / sum valid scalars')
    fig.suptitle('USER v2: changing training samples, NOT fixed-probe generalization')
    fig.savefig(DEST/'training_curve.png',dpi=160);plt.close(fig)
    return dict(updates=1500,exposures=12000,parents=25,parent_exposure=480,
        first100=curve[0],last100=curve[-1],max_reserved_bytes=max(r['peak_reserved_bytes'] for r in log))

def contacts():
    from PIL import Image,ImageDraw,ImageFont
    gallery=read(OUT/'gallery/completion.json')
    assert len(gallery['images'])==144
    qs=read(OUT/'eval_manifest.json')['requests']
    parents=sorted({q['parent_id'] for q in qs})
    images={r['page_id']:r for r in gallery['images']}
    target=DEST/'contact_sheets';target.mkdir(exist_ok=True)
    records=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for parent in parents:
        requests=sorted([q for q in qs if q['parent_id']==parent],key=lambda q:q['request_id'])
        sheet=Image.new('RGB',(2320,1750),'white');draw=ImageDraw.Draw(sheet)
        evidence=[]
        for row,stage in enumerate(STAGES):
            for col,q in enumerate(requests):
                im=images[stage+'__'+q['request_id']];p=resolve(im['path'])
                assert sha(p)==im['image_sha256']
                source=Image.open(p).convert('RGB')
                # Same fixed rectangular crops for every page; keep both cameras and both section axes.
                crops=[source.crop((0,160,1166,405)),source.crop((0,470,1166,720)),
                       source.crop((0,790,1166,1090)),source.crop((0,1100,1166,1415))]
                tile=Image.new('RGB',(1166,sum(x.height for x in crops)),'white');y=0
                for c in crops:tile.paste(c,(0,y));y+=c.height
                tile=tile.resize((580,555),Image.Resampling.LANCZOS)
                x,y=col*580,row*582
                draw.text((x+6,y+1),f'{stage} | {q["condition_id"].split("__")[1]} n{q["noise_index"]}',fill='black',font=font)
                draw.text((x+6,y+17),'LEFT A / RIGHT USER_V2 (blue C, orange seam)',fill='black',font=font)
                sheet.paste(tile,(x,y+29))
                evidence.append(dict(page_id=im['page_id'],source=str(p),source_sha256=sha(p)))
        dest=target/(parent+'.png');sheet.save(dest)
        records.append(dict(parent_id=parent,path=str(dest),sha256=sha(dest),pages=evidence,
            transformation='Uniform fixed whitespace crop and contact-sheet resize, originals untouched; uncertain small details require full page'))
    write_new(target/'manifest.json',dict(status='PREPARED_NOT_YET_REVIEWED',records=records))

def audit():
    reg=read(OUT/'training_registration.json');status=read(OUT/'training'/ARM/'status.json')
    assert status['status']=='COMPLETE' and status['local_step']==1500
    calls=jl(OUT/'training'/ARM/'calls.jsonl');started={};done=collections.Counter()
    for r in calls:
        i=r['call_id']
        if r['event']=='CALL_BEGIN':assert i not in started;started[i]=r['kind']
        elif r['event']=='CALL_DONE':assert started.pop(i)==r['kind'];done[r['kind']]+=1
        else:raise AssertionError('Unexpected failed scientific call '+str(r))
    assert not started and dict(done)==dict(train_forward=12000,train_backward=12000,optimizer_update=1500)
    for r in (status['checkpoint'],status['inference_export']):assert sha(r['path'])==r['sha256']
    for p,h in reg['training_code_hashes'].items():assert sha(p)==h,p
    for key in ('manifest','input_schedule','source_checkpoint','eval_manifest'):
        assert sha(reg[key])==reg[key+'_sha256'],key
    for p,h in read(OUT/'cpu_gate.json')['protected'].items():assert sha(p)==h,p
    coll=read(OUT/'results/completion.json')
    for field in ('files','source_rows','source_receipts'):
        for p,h in coll[field].items():assert sha(p)==h,p
    for q in read(OUT/'eval_manifest.json')['requests']:
        for stage in ('raw','post'):
            p=OUT/'evaluation'/LABEL/q['request_id']/stage/'receipt.json';r=read(p)
            assert r['status']=='COMPLETE'
            for path,h in r['files'].items():assert sha(path)==h,path
    for p in (OUT/'training').rglob('failure.json'):raise AssertionError('Failure record '+str(p))
    return dict(status='PASS',training_calls=dict(done),protected_training_code=len(reg['training_code_hashes']),
        protected_input_files=len(read(OUT/'cpu_gate.json')['protected']),new_meshes=48,new_D0=48,
        source_csv_rows=432,new_score_rows=144,new_model_calls_this_analysis=0,new_scoring_calls=0)

def main():
    DEST.mkdir(exist_ok=True)
    if (DEST/'completion.json').exists():raise RuntimeError('Completed analysis immutable; inspect existing outputs')
    tables=data();train=training();contacts();receipt=audit()
    write_new(DEST/'audit.json',receipt)
    write_new(DEST/'training_summary.json',train)
    write_new(DEST/'completion.json',dict(status='NUMERICS_VERIFIED_VISUAL_REVIEW_PENDING',audit=receipt,
        training=train,source_code_sha256=sha(__file__)))
    print(json.dumps(dict(audit=receipt,training=train,final=[r for r in tables if r['stage']=='CLEAN_FINAL_POST']),ensure_ascii=False))

if __name__=='__main__':main()
