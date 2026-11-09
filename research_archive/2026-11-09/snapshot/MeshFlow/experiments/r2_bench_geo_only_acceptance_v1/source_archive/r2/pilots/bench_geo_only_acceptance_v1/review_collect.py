"""Validate actual visual observations and apply the preregistered gate."""
import csv
import datetime as dt
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_geo_only_acceptance_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    if p.exists():
        assert read(p)==x, str(p)+' changed'
        return
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvwrite(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def joint(r):return int(r['overall_shape']=='REASONABLE' and r['C_use']=='PLAUSIBLE' and r['support_retention']=='PRESENT')

def run():
    dest=OUT/'visual_review'
    if (dest/'COMPLETE.json').exists():
        old=read(dest/'COMPLETE.json')
        for p,h in old['files'].items():assert sha(ROOT/p)==h
        return old
    template=read(OUT/'gallery/review_manifest.json')
    expected={(r['request_id'],r['label'],r['stage']):r for r in template['cells_to_review']}
    found={};inputs={}
    for name in ['review_root.json','review_assets.json','review_reference.json']:
        p=dest/name;data=read(p);inputs[str(p.relative_to(ROOT))]=sha(p)
        assert data['reviewer_kind']=='AI_PROXY' and data['blind'] is False
        for page in data['pages']:
            assert page['actually_viewed'] is True and page['viewed_at_utc']
            image=OUT/'gallery'/page['request_id']/'comparison.png'
            assert page['image_sha256']==sha(image)
            assert len(page['cells'])==5
            for cell in page['cells']:
                key=(page['request_id'],cell['label'],cell['stage']);assert key in expected and key not in found
                row=dict(expected[key]);assert row['image_sha256']==page['image_sha256']
                for axis,values in template['axes'].items():assert cell[axis] in values
                assert cell['note'].strip()
                row.update(cell,reviewer=data['reviewer'],reviewer_kind='AI_PROXY',blind=False,actually_viewed=True,
                    actual_view_source='tools.view_image; signed manual observation '+name,viewed_at_utc=page['viewed_at_utc'])
                row['visual_joint']=joint(row);found[key]=row
    assert set(found)==set(expected) and len(found)==150
    rows=sorted(found.values(),key=lambda r:(r['request_id'],r['column']))
    csvwrite(dest/'per_cell.csv',rows)
    grouped=defaultdict(list)
    for r in rows:grouped[(r['eval_group'],r['label'],r['stage'])].append(r)
    summaries=[]
    for (group,label,stage),rr in sorted(grouped.items()):
        parents=defaultdict(list)
        for r in rr:parents[r['parent_id']].append(r)
        row=dict(eval_group=group,label=label,stage=stage,parents=len(parents),outputs=len(rr),visual_joint=sum(r['visual_joint'] for r in rr),
                 visual_joint_parent_rate=sum(sum(r['visual_joint'] for r in pr)/len(pr) for pr in parents.values())/len(parents))
        for axis,values in template['axes'].items():
            for val in values:row[axis+'__'+val]=sum(r[axis]==val for r in rr)
        summaries.append(row)
    csvwrite(dest/'group_summary.csv',summaries)
    paired=[]
    for request_id in sorted({r['request_id'] for r in rows}):
        old=found[request_id,'COVERAGE_FINAL','CLEAN_FINAL_POST'];new=found[request_id,'COVERAGE_FINAL','GEO_ONLY']
        row=dict(request_id=request_id,parent_id=new['parent_id'],eval_group=new['eval_group'],old_visual_joint=old['visual_joint'],new_visual_joint=new['visual_joint'],
                 gained_joint=not old['visual_joint'] and bool(new['visual_joint']),lost_joint=bool(old['visual_joint']) and not new['visual_joint'])
        for axis in template['axes']:row['old_'+axis]=old[axis];row['new_'+axis]=new[axis]
        row['support_gain']=old['support_retention']!='PRESENT' and new['support_retention']=='PRESENT'
        paired.append(row)
    csvwrite(dest/'paired_transitions.csv',paired)
    geometry=read(OUT/'scores/geometry_gate_audit.json')
    # Recompute geometry alert directly from saved per-parent trigger fields below.
    with (OUT/'scores/geometry_gate_per_parent.csv').open(encoding='utf-8-sig',newline='') as f:gg=list(csv.DictReader(f))
    old_geometry=[]
    for r in gg:
        if r['eval_group'] in ['CURRENT_TRAIN','DEV']:
            # Independent audit JSON gives the authoritative recomputed keys;
            # CSV delta columns are inspected by the root before first execution.
            old_geometry.append(r)
    group_checks={}
    for group in sorted({r['eval_group'] for r in rows}):
        rr=[r for r in paired if r['eval_group']==group]
        group_checks[group]=dict(old_joint=sum(r['old_visual_joint'] for r in rr),new_joint=sum(r['new_visual_joint'] for r in rr),outputs=len(rr))
        group_checks[group]['not_lower']=group_checks[group]['new_joint']>=group_checks[group]['old_joint']
    support_extra=sorted({r['parent_id'] for r in paired if r['support_gain'] and r['eval_group']!='NEW_TRAIN'})
    visual_ok=all(r['not_lower'] for r in group_checks.values()) and bool(support_extra)
    alerts=[r for r in old_geometry if float(r['FF_delta_pp'])>=10 or float(r['CF_delta_pp'])>=5 or float(r['I_delta_pp'])>=5]
    assert len(alerts)==geometry['old_alert_parent_count']
    broad_alert=len(alerts)>=2
    assert broad_alert==geometry['old_geometry_gate_alert']
    failed=broad_alert or not visual_ok
    gate=dict(status='FAIL' if failed else 'PASS',run_source_identity=failed,
        decision='Uniform GEO_ONLY is not accepted as next priority full-pipeline candidate; keep bounded support preservation positives.',
        basis='Preregistered broad geometry alert; independent full-panel scores and all 150 actual AI_PROXY cells completed.',
        geometry_audit_path='scores/geometry_gate_audit.json',geometry_audit_sha256=sha(OUT/'scores/geometry_gate_audit.json'),
        preregistered_protocol_sha256=sha(OUT/'PROTOCOL.md'),
        broad_geometry_alert=broad_alert,old_geometry_alert_parents=len(alerts),
        visual_qualification=visual_ok,visual_group_checks=group_checks,nonNEW_support_gain_parents=support_extra,
        lost_old_joint_request_ids=[r['request_id'] for r in paired if r['lost_joint']],
        gained_joint_request_ids=[r['request_id'] for r in paired if r['gained_joint']],
        limits='Development gate, not an independent test or universal geometry quality threshold. Non-blind AI_PROXY, not human certification.',
        authorized_next=dict(D0_source_forwards=2,GEO_calls=2,generator_forwards=0,training=0,new_noise=0))
    write(OUT/'acceptance_gate.json',gate)
    result=dict(status='COMPLETE',pages=30,cells=150,reviewer_kind='AI_PROXY',blind=False,
        actual_image_hashes_verified=True,inputs=inputs,
        files={str(p.relative_to(ROOT)):sha(p) for p in dest.iterdir() if p.is_file()})
    write(dest/'COMPLETE.json',result)
    print(json.dumps(gate,ensure_ascii=False,indent=2));return result

if __name__=='__main__':run()
