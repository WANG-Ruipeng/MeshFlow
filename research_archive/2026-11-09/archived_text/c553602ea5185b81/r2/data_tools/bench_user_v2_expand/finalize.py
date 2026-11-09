"""Seal the actually viewed expansion; keep first-batch acceptance read-only."""
from .run import *
from collections import defaultdict
import copy
import html
import csv
import io
from r2.data_tools.bench_user_v2 import validate as V

# Written after viewing each named parent comparison and all overview pages.
# These are AI_PROXY judgments, not automatic segmentation confidence or human ratings.
PARENT_NOTES = {
 '838c48':'Organic-outline slab on two broad pedestals. Source and target retain the same visible whole surface; irregular outline is a complete design, not random scattered triangles.',
 '91e169':'Complete rounded-end solid bench block; top and side walls remain present in both views.',
 'bf504b':'Ornate bench with arm/end panels and slatted supporting frame; target preserves the complete visible structure.',
 'f35697':'HOLD: seat has a broad central empty region with only narrow strips; this is already visible in the source. Do not call this a repaired complete training answer.',
 '5f0e28':'Complete folded slab bench with two broad long walls and curved shoulders; removing coincident layers preserves the visible surface.',
 '85bfe7':'Simple narrow bench with back and end supports. Subdivision adds triangles without adding new style information.',
 '8967e6':'Complete rectangular seat/back with four legs; refined surface matches the source views.',
 '8d4693':'Complete narrow four-leg bench; interior subdivisions preserve the outer shape.',
 '98fe81':'Complete seat on two wide end supports; no visible source-to-target surface loss.',
 'ac1b9a':'Complete plain four-leg bench; refined triangle counts do not create additional independent geometry.',
 'b77813':'Complete broad seat/back bench on splayed broad feet; visible structure retained.',
 'c89d4c':'Complete bench with back and raised end sides; notches are part of the original joints, not random source cuts.',
 'dbd069':'Short broad backed bench with four legs; complete source surface retained after subdivision.',
 'e01c28':'Curved seat on a broad curved pedestal; target retains the curved sheet geometry.',
 'ef207b':'Backless bench with high end arms and low pedestal base. End arm surfaces must not be labelled as a backrest.',
 '4cd23a':'Upholstered rectangular storage bench. Simplification retains cushion and box silhouette in both views; fidelity remains a sampled bound.',
 '7b58a3':'Slatted top with rounded ends and a lower shelf; simplification retains these visible structures.',
 '943dde':'Long curved seat/back bench; simplified target retains the shape. Flat underside must not be mistaken for the curved seat upper surface.',
 '9e429b':'Whole sculpted three-bay seat/end structure retained, but all automatic C proposals are partial cuts or tiny strips and are held.',
 'b6b10e':'Complete curved-shoulder stool/short bench; simplified target retains wide side panels and arched feet.',
 'df6fd3':'Complete two-sided seating on a box base, with a central back divider; simplified structure retained.',
}

# suffix -> (decision, reviewed part, observed extent). Every new candidate is explicit.
NOTES = {
 '838c48':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Whole organic-outline seat upper region; coherent but intentionally nonrectangular.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','One complete planar side surface of a pedestal; not a closed whole leg.'),
  'COMBO_SEAT_SUPPORT':('ACCEPT','SEAT_SUPPORT','Organic-outline seat plus one pedestal side surface.'),
  'COMBO_PART_ASSEMBLY':('ACCEPT','PART_ASSEMBLY','Organic-outline seat plus two pedestal side surfaces.')},
 '91e169':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete rounded-end top region.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','Complete long rectangular pedestal wall surface.'),
  'COMBO_SEAT_SUPPORT':('ACCEPT','SEAT_SUPPORT','Rounded-end top and one long supporting side surface.'),
  'COMBO_PART_ASSEMBLY':('ACCEPT','PART_ASSEMBLY','Top and both long supporting side surfaces.')},
 'bf504b':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Broad seat upper surface including source end tabs; coherent polygon.'),
  'LEG_SUPPORT_2':('ACCEPT','LEG_SUPPORT','Connected L-shaped leg/frame side surface.'),
  'BACK_SURFACE_1':('ACCEPT','OTHER','End arm panel and top cap; this is not a rear backrest.')},
 'f35697':{
  'SEAT_SURFACE_1':('HOLD','SEAT_SURFACE','Single narrow seat strip; whole source target has a large missing seat region.'),
  'LEG_SUPPORT_1':('HOLD','LEG_SUPPORT','Coherent end frame, but its complete target is not accepted.')},
 '5f0e28':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete rectangular top plane, two triangles.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','Complete curved U-shaped end cross-section surface.'),
  'LEG_SUPPORT_2':('ACCEPT','LEG_SUPPORT','Opposite curved U-shaped end cross-section surface.'),
  'SEAT_SUPPORT_1':('ACCEPT','SEAT_SUPPORT','Top plane and complete U-shaped end surface.')},
 '85bfe7':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete narrow seat upper plane.'),
  'BACK_SURFACE_1':('ACCEPT','BACK_SURFACE','Complete inner back plane.'),
  'LEG_SUPPORT_1':('HOLD_DUPLICATE','SEAT_SUPPORT','Includes seat and supports; identical face set to SEAT_SUPPORT_1, not pure legs.'),
  'SEAT_BACK_1':('ACCEPT','SEAT_BACK','Complete seat and back plane assembly.'),
  'SEAT_SUPPORT_1':('ACCEPT','SEAT_SUPPORT','Whole lower base including seat and both end supports.')},
 '8967e6':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete rectangular seat upper plane.'),
  'BACK_SURFACE_1':('ACCEPT','BACK_SURFACE','Complete inner rectangular back plane.'),
  'LEG_SUPPORT_1':('HOLD_DUPLICATE','SEAT_SUPPORT','Complete lower assembly, identical to SEAT_SUPPORT_1; adapter check also fails.'),
  'SEAT_BACK_1':('ACCEPT','SEAT_BACK','Complete seat and back plane assembly.'),
  'SEAT_SUPPORT_1':('HOLD_ADAPTER','SEAT_SUPPORT','Plausible lower assembly including seat and legs, but production candidate geometry/adapter gate fails.')},
 '8d4693':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete narrow rectangular seat plane.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','Complete side frame surface: two legs and connecting apron.'),
  'LEG_SUPPORT_2':('ACCEPT','LEG_SUPPORT','Opposite complete side frame surface.'),
  'SEAT_SUPPORT_1':('ACCEPT','SEAT_SUPPORT','Seat plane and one side frame.')},
 '98fe81':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete rectangular seat plane.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','Coherent lower supporting frame including broad end walls and connecting crosspiece.'),
  'SEAT_SUPPORT_1':('ACCEPT','SEAT_SUPPORT','Seat plane plus supporting frame.')},
 'ac1b9a':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete rectangular seat plane.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','One complete long-side support frame surface.'),
  'LEG_SUPPORT_2':('ACCEPT','LEG_SUPPORT','Opposite long-side support frame surface.'),
  'SEAT_SUPPORT_1':('ACCEPT','SEAT_SUPPORT','Seat plane plus one support frame.')},
 'b77813':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete seat upper plane.'),
  'BACK_SURFACE_1':('ACCEPT','BACK_SURFACE','Complete inner back plane.'),
  'LEG_SUPPORT_1':('HOLD_DUPLICATE','SEAT_SUPPORT','Includes seat and splayed feet; identical to SEAT_SUPPORT_1.'),
  'SEAT_BACK_1':('ACCEPT','SEAT_BACK','Seat and back planes.'),
  'SEAT_SUPPORT_1':('ACCEPT','SEAT_SUPPORT','Complete lower base including seat and splayed feet.')},
 'c89d4c':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete recessed seat plane between end sides.'),
  'BACK_SURFACE_1':('ACCEPT','BACK_SURFACE','Back inner plane with original stepped end outline.'),
  'LEG_SUPPORT_1':('HOLD_DUPLICATE','SEAT_SUPPORT','Same notched lower assembly as SEAT_SUPPORT_1, includes seat.'),
  'SEAT_BACK_1':('ACCEPT','SEAT_BACK','Seat and back surface assembly.'),
  'SEAT_SUPPORT_1':('ACCEPT','SEAT_SUPPORT','Lower assembly with original notched junctions to the upper back/sides.')},
 'dbd069':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete rectangular seat upper plane.'),
  'BACK_SURFACE_1':('ACCEPT','BACK_SURFACE','Complete inner back plane.'),
  'LEG_SUPPORT_1':('HOLD_DUPLICATE','SEAT_SUPPORT','Includes seat and four legs; same set as SEAT_SUPPORT_1; adapter gate fails.'),
  'SEAT_BACK_1':('ACCEPT','SEAT_BACK','Complete seat and back plane assembly.'),
  'SEAT_SUPPORT_1':('HOLD_ADAPTER','SEAT_SUPPORT','Plausible base assembly, but production candidate geometry/adapter gate fails.')},
 'e01c28':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Whole curved seat upper strip, coherent and complete.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','One narrow complete end face of the curved pedestal.'),
  'LEG_SUPPORT_2':('ACCEPT','LEG_SUPPORT','Opposite complete pedestal end face.'),
  'SEAT_SUPPORT_1':('ACCEPT','SEAT_SUPPORT','Curved seat plane plus one complete pedestal end face.')},
 'ef207b':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete broad upper seat plane; three triangles.'),
  'BACK_SURFACE_1':('HOLD','OTHER','One arm interior plane plus a partial triangular fragment at the opposite arm; not a complete back.'),
  'LEG_SUPPORT_1':('HOLD_DUPLICATE','SEAT_SUPPORT','Includes complete seat/base, identical to SEAT_SUPPORT_1.'),
  'SEAT_BACK_1':('HOLD','PART_ASSEMBLY','Seat plus asymmetric partial arm planes; not a clean seat-back input.'),
  'SEAT_SUPPORT_1':('ACCEPT','SEAT_SUPPORT','Complete lower seat/base assembly, leaving both upper arms to generate.')},
 '4cd23a':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete rectangular upper cushion plane.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','Complete closed lower box/base, not separate stick legs.'),
  'SEAT_SUPPORT_1':('HOLD','SEAT_SUPPORT','Top plane floats above the box without the intervening cushion shoulder; hold incoherent combination.')},
 '7b58a3':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Whole regular set of seat slat upper surfaces; intentionally multiple complete strips, not random isolated faces.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','Complete long side frame surface with rounded upper corners and lower rail.'),
  'LEG_SUPPORT_2':('ACCEPT','LEG_SUPPORT','Opposite complete side frame surface.'),
  'SEAT_SUPPORT_1':('ACCEPT','SEAT_SUPPORT','Whole slat upper-surface set plus one complete side frame.')},
 '943dde':{
  'SEAT_SURFACE_1':('HOLD','OTHER','Planar underside of the curved seat, not its upper seating surface.'),
  'BACK_SURFACE_1':('ACCEPT','BACK_SURFACE','Complete outer rear back surface, explicitly not the curved inner face.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','Connected lower support/underside assembly with two end feet.'),
  'SEAT_BACK_1':('HOLD','PART_ASSEMBLY','Outer back combined with seat underside; not the proposed seat/back upper input.'),
  'SEAT_SUPPORT_1':('HOLD_DUPLICATE','LEG_SUPPORT','Same lower support assembly as LEG_SUPPORT_1, not an additional seat condition.')},
 '9e429b':{
  'SEAT_SURFACE_1':('HOLD','SEAT_SURFACE','Tiny strip at an end transition; not the complete curved three-bay seat.'),
  'LEG_SUPPORT_1':('HOLD','LEG_SUPPORT','Lower portion of one end cut by a height threshold; incomplete end support.'),
  'LEG_SUPPORT_2':('HOLD','LEG_SUPPORT','Lower portions on both ends; incomplete surfaces cut by height threshold.'),
  'SEAT_SUPPORT_1':('HOLD','SEAT_SUPPORT','Incomplete support plus disconnected tiny seat strip.')},
 'b6b10e':{
  'SEAT_SURFACE_1':('ACCEPT','SEAT_SURFACE','Complete flat upper seat plane.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','Complete U-shaped frame side surface; not a closed whole support.'),
  'LEG_SUPPORT_2':('ACCEPT','LEG_SUPPORT','Opposite complete U-shaped frame side surface.'),
  'SEAT_SUPPORT_1':('ACCEPT','SEAT_SUPPORT','Seat plane and one rounded U-shaped side frame.')},
 'df6fd3':{
  'SEAT_SURFACE_1':('HOLD_ADAPTER','SEAT_SURFACE','Two coherent seat-plane strips around central back; candidate geometry/adapter gate fails.'),
  'LEG_SUPPORT_1':('ACCEPT','LEG_SUPPORT','Complete lower box/plinth assembly.'),
  'SEAT_SUPPORT_1':('HOLD_ADAPTER','SEAT_SUPPORT','Plinth plus both upper seating strips; candidate geometry/adapter gate fails.')},
}

def evidence(p):return dict(path=str(p),sha256=b.sha(p))

def csvout(p,rows):
    stream=io.StringIO(newline='');fields=list(dict.fromkeys(k for r in rows for k in r))
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    writer.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
    data=('\ufeff'+stream.getvalue()).encode('utf-8')
    if p.exists():assert p.read_bytes()==data,p
    else:p.write_bytes(data)

def seal():
    configure()
    candidate=OUT/'standard_candidate_manifest.json';m=b.read(candidate);old=b.read(OLD/'manifest.json')
    oldp={p['parent_id']:p for p in old['parents']};oldc={c['condition_id']:c for c in old['conditions']}
    stage={}
    for name,folder in [('render_manifest.json','review_v1'),('render_supply_manifest.json','supply_review'),('render_standard_manifest.json','standard_review')]:
        stage.update({p['parent_id']:folder for p in b.read(OUT/name)['parents']})
    reviews=dict(reviewer_type='AI_PROXY',reviewer='root_actual_multiview_nonblind',
        source='Actual parent before/after and overview C-only/C+Y PNGs viewed in this session; no generator images.',
        candidate_manifest_sha256=b.sha(candidate),parents=[],conditions=[])
    parents={};heldp=[]
    for p0 in m['parents']:
        p=copy.deepcopy(p0);uid=p['parent_id'];prefix=uid[:6]
        if uid in oldp:p=copy.deepcopy(oldp[uid])
        elif uid in stage:
            assert prefix in PARENT_NOTES
            review=dict(parent_id=uid,decision='HOLD' if prefix=='f35697' else 'ACCEPT',
                actually_viewed=True,notes=PARENT_NOTES[prefix],image_evidence=[evidence(OUT/'gallery'/stage[uid]/'parents'/(uid+'.png'))])
            reviews['parents'].append(review);p['review']=review;p['review_status']=review['decision']
        else:p.update(review_status='NOT_REVIEWED_TARGET_BLOCKED',review=None)
        parents[uid]=p
    rows=[];active=[]
    for uid in sorted(parents):
        group=sorted([r for r in m['conditions'] if r['parent_id']==uid],key=lambda x:x['condition_id'])
        for i,r0 in enumerate(group):
            r=copy.deepcopy(r0);cid=r['condition_id'];prefix=uid[:6];suffix=cid.split('__')[1]
            if cid in oldc:r=copy.deepcopy(oldc[cid])
            elif uid in stage:
                assert suffix in NOTES[prefix],(prefix,suffix)
                decision,part,note=NOTES[prefix][suffix]
                review=dict(parent_id=uid,condition_id=cid,decision=decision,actually_viewed=True,
                    C_standalone_plausible=decision in ('ACCEPT','HOLD_ADAPTER'),observed_extent=note,
                    actual_part=part,notes=note,image_evidence=[evidence(OUT/'gallery'/stage[uid]/'overview'/(uid+f'__p{i//4+1:02d}.png'))])
                reviews['conditions'].append(review)
                r.update(review=review,review_status=decision,proposed_part_type=r['part_type'],part_type=part,extent_description=note)
            reasons=[]
            if parents[uid].get('reasons'):reasons.append('TARGET_GEOMETRY_BLOCKED')
            if parents[uid].get('review_status')!='ACCEPT':reasons.append('PARENT_VISUAL_NOT_ACCEPTED')
            if r.get('review_status')!='ACCEPT':reasons.append('CONDITION_NOT_ACCEPTED')
            if not r.get('adapter_compatible'):reasons.append('CONDITION_GEOMETRY_OR_ADAPTER_BLOCKED')
            r['admission_reasons']=reasons
            if not reasons:
                r.update(eligibility='ACCEPTED_PENDING_CPU_PACKING',source_group_id=uid,
                    near_duplicate_group_status='Distinct source UID and exact-source hash checked; shape-family independence UNKNOWN')
                active.append(r)
            else:r['eligibility']='HELD'
            rows.append(r)
    # Same-parent duplicate face sets are not independent training conditions.
    keys=[(r['parent_id'],tuple(sorted(r['target_face_ids']))) for r in active]
    assert len(keys)==len(set(keys)),'Unremoved duplicate C'
    accepted_ids={r['parent_id'] for r in active}
    for r in reviews['parents']+reviews['conditions']:
        for e in r['image_evidence']:assert b.sha(e['path'])==e['sha256']
    b.save(OUT/'visual_review.json',reviews)
    all_decisions=dict(parents=list(parents.values()),conditions=rows)
    b.save(OUT/'all_admission_decisions.json',all_decisions)
    final={k:copy.deepcopy(v) for k,v in m.items() if k not in ('parents','conditions','train')}
    final.update(parents=[parents[u] for u in sorted(accepted_ids)],conditions=active,train=active,
        status='VISUALLY_ACCEPTED_PENDING_CPU_PACKING',candidate_manifest_path=str(candidate),candidate_manifest_sha256=b.sha(candidate),
        visual_review_path=str(OUT/'visual_review.json'),visual_review_sha256=b.sha(OUT/'visual_review.json'),
        all_decisions_path=str(OUT/'all_admission_decisions.json'),training_registered=False,
        scale_goal=[47,67],scale_goal_status='BLOCKED_INSUFFICIENT_QUALIFIED_PARENTS',training_schedule=None)
    b.save(OUT/'manifest.json',final)
    old_ids={r['condition_id'] for r in old['train']}
    new=copy.deepcopy(final);new['conditions']=new['train']=[r for r in active if r['condition_id'] not in old_ids]
    newparents={r['parent_id'] for r in new['train']};new['parents']=[p for p in final['parents'] if p['parent_id'] in newparents]
    b.save(OUT/'new_validation_manifest.json',new)
    csvout(OUT/'accepted_conditions.csv',active)
    csvout(OUT/'held_conditions.csv',[r for r in rows if r['admission_reasons']])
    csvout(OUT/'parents_admission.csv',list(parents.values()))
    summary=dict(candidate_target_parents=len(parents),target_geometry_pass=sum(not p['reasons'] for p in parents.values()),
        accepted_parents=len(accepted_ids),accepted_conditions=len(active),new_CPU_validation_conditions=len(new['train']),
        reused_original_CPU_conditions=len(old_ids),accepted_parts=dict(Counter(r['part_type'] for r in active)),
        part_parent_counts={k:len({r['parent_id'] for r in active if r['part_type']==k}) for k in sorted({r['part_type'] for r in active})},
        N_range=[min(r['N'] for r in active),max(r['N'] for r in active)],K_range=[min(r['K'] for r in active),max(r['K'] for r in active)],
        adapters=dict(Counter(r['adapter_semantics'] for r in active)),
        unchanged_DEV_conditions=len(final['dev']),unchanged_DEV_parents=len({r['parent_id'] for r in final['dev']}),
        scale_goal_status=final['scale_goal_status'],missing_to_47=47-len(accepted_ids),model_calls=0,training_updates=0)
    b.save(OUT/'data_summary.json',summary)
    policy=dict(schema='BENCH_V2_EXPANDED_PARENT_FIRST',training_registered=False,training_budget=None,
        sampling_order=['source parent uniformly','available reviewed part uniformly','condition within part uniformly','joint M/C/Y transform and noise'],
        per_parent_probability={u:1/len(accepted_ids) for u in sorted(accepted_ids)},
        parent_parts={u:{part:[r['condition_id'] for r in active if r['parent_id']==u and r['part_type']==part]
            for part in sorted({r['part_type'] for r in active if r['parent_id']==u})} for u in sorted(accepted_ids)},
        source_family_independence='UNKNOWN beyond exact hashes and separate source IDs',
        augmentation='Existing R2 joint transform of complete M and derived C/Y; no independent C normalization',
        no_global_part_oversampling=True,no_old_schedule_reuse=True)
    b.save(OUT/'sampling_policy.json',policy)
    event('VISUAL_ADMISSION_SEALED',**summary);print(json.dumps(summary))

def validate_new():
    result=V.validate(OUT/'new_validation_manifest.json',OUT/'validation')
    if result['status']!='PASS':raise RuntimeError('CPU validation blocked; inspect preserved receipt')

def close():
    m=b.read(OUT/'manifest.json');s=b.read(OUT/'data_summary.json');old=b.read(OLD/'DATASET_READY.json')
    assert b.sha(old['manifest_path'])==old['manifest_sha256']
    assert b.sha(old['validation_receipt'])==old['validation_receipt_sha256']
    prior=b.read(old['validation_receipt']);assert prior['status']=='PASS'
    newp=b.read(OUT/'validation/latest.json');assert newp['status']=='PASS'
    assert b.sha(newp['receipt'])==newp['receipt_sha256'];new=b.read(newp['receipt'])
    accepted={r['condition_id']:r for r in m['train']}
    for r in b.read(old['manifest_path'])['train']:
        for k in ('case_sha256','input_sha256','target_face_ids'):assert accepted[r['condition_id']][k]==r[k]
    receipts=prior['condition_results']+new['condition_results']
    assert {r['condition_id'] for r in receipts}==set(accepted)
    assert all(r['status']=='PASS' for r in receipts)
    for p,h in m['protected_source_hashes'].items():assert b.sha(p)==h,p
    s.update(status='DATA_READY_SCALE_GOAL_NOT_REACHED',passed_CPU_conditions=len(receipts),
        new_CPU_OT_attempts=new['new_CPU_OT_attempts'],new_CPU_OT_completed=new['new_CPU_OT_completed'],
        reused_CPU_receipts=len(prior['condition_results']),old_assets_unchanged=True,
        source_independence_limit='Source IDs and exact hashes, not verified style-family independence')
    b.save(OUT/'final_summary.json',s)
    ready=dict(schema=VERSION,status=s['status'],accepted_parents=s['accepted_parents'],accepted_conditions=s['accepted_conditions'],
        manifest_path=str(OUT/'manifest.json'),manifest_sha256=b.sha(OUT/'manifest.json'),
        validation_new=newp,validation_reused=old,summary_path=str(OUT/'final_summary.json'),summary_sha256=b.sha(OUT/'final_summary.json'),
        scale_goal_status=s['scale_goal_status'],training_authorized=False,default_pool_changed=False)
    b.save(OUT/'DATASET_READY.json',ready)
    cards=[]
    for p in m['parents']:
        uid=p['parent_id'];pr=p.get('review') or {};pe=(pr.get('image_evidence') or [{}])[0].get('path')
        cs=[r for r in m['train'] if r['parent_id']==uid]
        def link(path):return Path(path).relative_to(ROOT).as_posix() if not str(path).startswith('/mnt') else str(path).split('/MeshFlow/',1)[1]
        def url(path):return '../../'+link(path)
        parts=[]
        for r in cs:
            e=r['review']['image_evidence'][0]['path']
            parts.append(f'<li><a href="{url(e)}">{html.escape(r["condition_id"].split("__")[1])}</a> | {r["part_type"]} | K={r["K"]}<br>{html.escape(r["extent_description"])}</li>')
        cards.append(f'<section><h2>{uid} | N={p["N"]}</h2><a href="{url(pe)}"><img loading="lazy" src="{url(pe)}"></a><ul>{"".join(parts)}</ul></section>')
    doc=f'''<!doctype html><meta charset="utf-8"><title>Bench USER v2 expanded reviewed data</title>
<style>body{{font:16px sans-serif;max-width:1300px;margin:auto;padding:25px}}section{{border-top:1px solid #ccc;margin:25px 0}}img{{width:480px;max-width:100%}}li{{margin:10px 0}}</style>
<h1>Bench USER v2: {s['accepted_parents']} parents / {s['accepted_conditions']} accepted C</h1>
<p>Actual AI_PROXY multiview review plus production CPU packing. No training or model call. 47–67 parent scale target not reached.
Images retain their creation-time NOT_REVIEWED captions; current decisions are in visual_review.json and manifest.json. Original first-batch reviews/receipts are reused unchanged.</p>
<p><a href="REPORT.md">Report</a> | <a href="accepted_conditions.csv">Accepted conditions</a> | <a href="held_conditions.csv">Held conditions</a> | <a href="DATASET_READY.json">Machine receipt</a></p>{''.join(cards)}'''
    (OUT/'index.html').write_text(doc,encoding='utf-8')
    event('DATA_READY',**s);print(json.dumps(s))

def verify():
    m=b.read(OUT/'manifest.json');ready=b.read(OUT/'DATASET_READY.json')
    assert b.sha(ready['manifest_path'])==ready['manifest_sha256']
    assert b.sha(ready['summary_path'])==ready['summary_sha256']
    for p,h in m['protected_source_hashes'].items():assert b.sha(p)==h,p
    for p in m['parents']:
        for key in ('target_mesh','target_case'):assert b.sha(p[key+'_path'])==p[key+'_sha256']
    for r in m['train']:
        for key in ('case','input','paired'):assert b.sha(r[key+'_path'])==r[key+'_sha256']
        for e in r['review']['image_evidence']:assert b.sha(e['path'])==e['sha256']
    for key in ('validation_new','validation_reused'):
        r=ready[key];path=r.get('receipt',r.get('validation_receipt'));h=r.get('receipt_sha256',r.get('validation_receipt_sha256'))
        assert b.sha(path)==h
    print('PASS: sealed source/data/review/validation hashes; no model, OT or geometry calls')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['seal','validate','close','verify']);a=p.parse_args()
    {'seal':seal,'validate':validate_new,'close':close,'verify':verify}[a.stage]()
