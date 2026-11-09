"""CPU-only anonymous comparison gallery for the frozen noise-transfer panel.

The condition-level model permutation is frozen before generation.  Public
images/templates contain anonymous labels only; the separate blinding directory
holds the model key and source paths.  No metric determines a page or ordering.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, html, itertools, json, random, time
from pathlib import Path
import numpy as np
from . import common as C

EV=C.OUT/'evaluation'
GALLERY=EV/'gallery'
PRIVATE=EV/'blinding'
PLAN=GALLERY/'blind_plan.json'
KEY=PRIVATE/'model_key.json'
LABELS=('EXP67_FROZEN','SMALL_C200','FULL_C200')
ANON=('A','B','C')
VERSION='r2_seat_noise_transfer_anonymous_gallery_v1'
META=('request_id','condition_id','parent_id','panel','split','condition_type','region_type','tier','N','K','variant','candidate')
FIELDS=('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer',
        'whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')
DEFINITION=dict(version=VERSION,request_pages=48,POST_noise_sheets=8,review_rows=288,
    blinded='Model names hidden; A/B/C permutation fixed within each condition. Reviewers may recognize familiar meshes; anonymity does not certify successful blinding.',
    views=['whole mesh, orthographic camera (22,-60)','front XY section at C centroid Z','side ZY section at C centroid X'],
    bounds='One whole-mesh scale for all models, both stages and all registered noises of a condition, including all fragments. C-only local bounds.',
    local_margins_L=dict(X=.075,Y=.10,Z=.075),
    sections='Exact triangle/plane intersections. Pale blue projected C edges are context, not intersections. Central cuts can miss off-plane joints.',
    no_smoothing=True,no_geometry_change=True,no_quality_selection=True,
    review='AI_PROXY; YES/NO/UNCERTAIN judgments require actual viewing. Visible contact does not certify indexed topology or semantic correctness.',
    source_target='No source completion is displayed; source C is used only for the POST preservation contract.')


def verify(files):
    for path,digest in files.items():
        if C.sha(path)!=digest:raise ValueError('Frozen gallery source changed: '+str(path))


def definitions():
    paths=[Path(__file__),Path(C.__file__),C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/rendering.py',
           C.ROOT/'r2/tools/student_multistep_denoise_v1/gallery.py']
    return {str(p):C.sha(p) for p in paths}


def request_groups(requests):
    if len(requests)!=48 or len({q['request_id'] for q in requests})!=48:
        raise ValueError('Expected 48 distinct frozen requests')
    groups=collections.defaultdict(list)
    for q in requests:groups[q['condition_id']].append(q)
    if len(groups)!=8 or sorted(map(len,groups.values()))!=[4]*4+[8]*4:
        raise ValueError('Expected four conditions with eight noises and four with four noises')
    result={}
    for cid,qs in sorted(groups.items()):
        if len({(q['parent_id'],int(q['K']),int(q['N']),q['condition_type']) for q in qs})!=1:
            raise ValueError('Condition identity changes across noises: '+cid)
        if len({str(q['candidate']) for q in qs})!=len(qs):raise ValueError('Repeated noise label')
        result[cid]=sorted(qs,key=lambda q:(int(q['candidate']),q['request_id']))
    return result


def freeze_blind():
    """No model/scorer imports. Safe to call again only with identical manifest."""
    manifest_path=C.OUT/'inputs/manifest.json';manifest=C.read(manifest_path)
    groups=request_groups(manifest['requests']);mh=C.sha(manifest_path)
    if PLAN.exists():
        plan,key=load_plan()
        if plan['manifest_sha256']!=mh:raise ValueError('Manifest changed after blinding')
        return dict(status='FROZEN',conditions=8,requests=48,plan_sha256=C.sha(PLAN),key_sha256=C.sha(KEY),model_calls=0)
    if any((EV/stage).exists() and any((EV/stage).rglob('*')) for stage in ('raw','post')):
        raise RuntimeError('Anonymous assignment must be frozen before any RAW/POST generation artifacts')
    # A separate fixed PRNG affects presentation only, never sampling or training.
    rng=random.Random(2026101401)
    permutations=list(itertools.permutations(LABELS));rng.shuffle(permutations)
    chosen=permutations+[rng.choice(permutations) for _ in range(2)]
    rng.shuffle(chosen)
    assignments={cid:dict(zip(ANON,order)) for cid,order in zip(groups,chosen)}
    key=dict(status='FROZEN',version=VERSION,manifest_sha256=mh,
             randomization='Python Random(2026101401), all six permutations plus two drawn repeats; shuffled before assigning sorted condition IDs.',
             assignments=assignments,unblinding_policy='Do not expose this key or private source receipts until row-level visual judgments are saved.')
    C.write(KEY,key)
    conditions=[dict(condition_id=cid,parent_id=qs[0]['parent_id'],condition_type=qs[0]['condition_type'],
                     panel=qs[0].get('panel',''),split=qs[0].get('split',''),anonymous_rows=list(ANON),
                     request_ids=[q['request_id'] for q in qs],noise_labels=[q['candidate'] for q in qs]) for cid,qs in groups.items()]
    C.write(PLAN,dict(status='FROZEN',version=VERSION,manifest_sha256=mh,key_sha256=C.sha(KEY),
                     conditions=conditions,definition=DEFINITION,created_before_generation=True))
    return dict(status='FROZEN',conditions=8,requests=48,plan_sha256=C.sha(PLAN),key_sha256=C.sha(KEY),model_calls=0)


def load_plan():
    plan,key=C.read(PLAN),C.read(KEY)
    if plan['status']!='FROZEN' or key['status']!='FROZEN' or plan['key_sha256']!=C.sha(KEY):
        raise ValueError('Blinding key changed')
    if plan['manifest_sha256']!=key['manifest_sha256'] or C.sha(C.OUT/'inputs/manifest.json')!=plan['manifest_sha256']:
        raise ValueError('Blinding manifest changed')
    for assignment in key['assignments'].values():
        if set(assignment)!=set(ANON) or set(assignment.values())!=set(LABELS):raise ValueError('Invalid anonymous permutation')
    return plan,key


def get_indices(requests):
    expected={(label,q['request_id']) for label in LABELS for q in requests};result={}
    for stage in ('raw','post'):
        obj=C.read(EV/f'{stage}_index.json');rows=obj['rows']
        if obj.get('status')!='COMPLETE' or len(rows)!=144:raise ValueError('Incomplete '+stage+' index')
        mapping={(row['label'],row['request_id']):row for row in rows}
        if set(mapping)!=expected:raise ValueError('Wrong '+stage+' model/request identities')
        result[stage.upper()]=mapping
    return result


def helpers():
    R=C.module('r2.pilots.seat_failure_diagnosis_v1._noise_transfer_sections',C.ROOT/'r2/pilots/seat_failure_diagnosis_v1/rendering.py')
    G=C.module('_noise_transfer_draw',C.ROOT/'r2/tools/student_multistep_denoise_v1/gallery.py')
    return R,G


def load_condition(qs,indices,assignment):
    meshes={};hashes={};centers={};condition=None
    R,_=helpers()
    for q in qs:
        rid=q['request_id'];k=int(q['K']);cc=C.arrays(q['input_path'])['C'].reshape(k,3,3)
        source=C.arrays(q['condition_source'])['C_original'].reshape(k,3,3)
        if condition is None:condition=cc
        elif not C.same(condition,cc):raise ValueError('C changes across noise')
        centers[rid]=R.center_of_c(cc.astype(float))
        hashes[q['input_path']]=q['input_sha256'];hashes[q['condition_source']]=q['condition_source_sha256']
        for anon,label in assignment.items():
            for stage in ('RAW','POST'):
                rec=indices[stage][label,rid];field=stage.lower();path=rec[field+'_path'];arrays=C.arrays(path)
                mesh=arrays['raw' if stage=='RAW' else 'output'].reshape(-1,3,3)
                if mesh.dtype!=np.float32 or not np.isfinite(mesh).all():raise ValueError('Bad render array')
                if not C.same(mesh[:k],cc if stage=='RAW' else source):raise ValueError('C preservation contract failed')
                if stage=='POST' and not C.same(arrays['vertices'][arrays['faces']],mesh):raise ValueError('POST index mismatch')
                meshes[rid,anon,stage]=mesh;hashes[path]=rec[field+'_sha256']
    verify(hashes)
    xyz=np.concatenate([m.reshape(-1,3) for m in meshes.values()]).astype(float);lo=xyz.min(0);hi=xyz.max(0)
    cxyz=condition.reshape(-1,3).astype(float);margins=np.array([.075,.1,.075])*C.L
    bounds=dict(center_model_xyz=((lo+hi)/2).tolist(),span=float((hi-lo).max()*1.10),
                local_low=(cxyz.min(0)-margins).tolist(),local_high=(cxyz.max(0)+margins).tolist())
    return meshes,hashes,centers,bounds


def save_figure(dest,private_identity,draw):
    """Recover rendered pages by hash; never overwrite an inconsistent page."""
    receipt_path=PRIVATE/'receipts'/dest.with_suffix('.json').name
    if receipt_path.exists():
        receipt=C.read(receipt_path)
        if receipt['identity']!=C.clean(private_identity) or C.sha(dest)!=receipt['image_sha256']:
            raise ValueError('Existing anonymous page differs: '+str(dest))
        return receipt
    if dest.exists():raise RuntimeError('Image without receipt; explicit engineering recovery required: '+str(dest))
    import matplotlib.pyplot as plt
    from PIL import Image
    fig=draw();C.checked(dest).parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.tmp')
    with tmp.open('xb') as file:fig.savefig(file,format='jpg',pil_kwargs={'quality':92,'optimize':True})
    plt.close(fig);os.replace(tmp,dest)
    with Image.open(dest) as image:image.verify()
    receipt=dict(status='RENDERED_NOT_REVIEWED',identity=private_identity,image_path=str(dest),image_sha256=C.sha(dest))
    C.write(receipt_path,receipt);return receipt


def request_figure(q,meshes,center,bounds):
    import matplotlib.pyplot as plt
    R,G=helpers();rid=q['request_id'];k=int(q['K'])
    fig=plt.figure(figsize=(18,8.7),dpi=135,facecolor='white')
    for i,anon in enumerate(ANON):
        for j,stage in enumerate(('RAW','POST')):
            mesh=meshes[rid,anon,stage];start=i*6+j*3
            ax=fig.add_subplot(3,6,start+1,projection='3d');G.draw(ax,mesh,k,bounds,R.CAMERA)
            ax.set_title(anon+' | '+stage+f' | F={len(mesh)-k}',fontsize=9)
            for viewnumber,view in enumerate(('front','side'),2):
                ax=fig.add_subplot(3,6,start+viewnumber);R.section_plot(ax,mesh,k,center,bounds,view)
    fig.suptitle(q['parent_id'][:8]+' | '+q['condition_type']+f' | noise {q["candidate"]} | K={k} N={q["N"]}',fontsize=14)
    fig.text(.5,.040,'BLUE C, GRAY free | Anonymous model rows A/B/C, fixed within this C | left RAW, right single D0 + GEO_C + source orientation',ha='center',fontsize=9)
    fig.text(.5,.020,'Whole view and two C-centered cuts. Pale blue outline is a projection, not an intersection. All models/noises of this C share scale.',ha='center',fontsize=9)
    fig.text(.5,.003,'Anonymous AI_PROXY review: visible contact does not certify topology. No smoothing or mesh edits. Do not consult model key before reviewing.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.02,right=.99,top=.91,bottom=.10,wspace=.25,hspace=.43)
    return fig


def contact_figure(qs,meshes,bounds):
    import matplotlib.pyplot as plt
    R,G=helpers();n=len(qs);first=qs[0]
    fig=plt.figure(figsize=(3.15*n,9.3),dpi=135,facecolor='white')
    for i,anon in enumerate(ANON):
        for j,q in enumerate(qs):
            mesh=meshes[q['request_id'],anon,'POST'];ax=fig.add_subplot(3,n,i*n+j+1,projection='3d')
            G.draw(ax,mesh,int(q['K']),bounds,R.CAMERA)
            ax.set_title(anon+f' | noise {q["candidate"]} | F={len(mesh)-int(q["K"])}',fontsize=10)
    fig.suptitle(first['parent_id'][:8]+' | '+first['condition_type']+f' | POST across {n} frozen noises',fontsize=16)
    fig.text(.5,.035,'Anonymous model rows A/B/C match this condition\'s detailed pages; independent condition mappings. BLUE C, GRAY free. Same camera and scale.',ha='center',fontsize=10)
    fig.text(.5,.012,'Compare styles and consistency without treating the source parent as the unique correct shape. All registered noises shown; no selection.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.015,right=.995,top=.91,bottom=.075,wspace=.015,hspace=.12)
    return fig


def render():
    tick=time.perf_counter();plan,key=load_plan();reg=C.read(EV/'registration.json');groups=request_groups(reg['requests'])
    public_groups={p['condition_id']:p for p in plan['conditions']}
    if set(groups)!=set(public_groups):raise ValueError('Registration differs from blind plan')
    for cid,qs in groups.items():
        if [q['request_id'] for q in qs]!=public_groups[cid]['request_ids']:raise ValueError('Request order differs')
    indices=get_indices(reg['requests']);defs=definitions()
    identity=dict(registration_sha256=C.sha(EV/'registration.json'),blind_plan_sha256=C.sha(PLAN),key_sha256=C.sha(KEY),
                  index_hashes={str(EV/f'{s}_index.json'):C.sha(EV/f'{s}_index.json') for s in ('raw','post')},definition_hashes=defs)
    final=GALLERY/'manifest.json'
    if final.exists():
        result=C.read(final)
        if result['identity']!=identity:raise ValueError('Gallery identity changed after completion')
        verify(result['output_hashes']);return result
    C.write(GALLERY/'definition.json',DEFINITION)
    if not (PRIVATE/'resources_before.json').exists():C.resources(PRIVATE/'resources_before.json')
    images=[];reviews=[];contacts=[];contact_reviews=[];cards=[];all_sources={}
    for cid,qs in groups.items():
        meshes,hashes,centers,bounds=load_condition(qs,indices,key['assignments'][cid]);all_sources.update(hashes)
        for q in qs:
            rid=q['request_id'];dest=GALLERY/'images'/(rid+'.jpg')
            private_identity=dict(public_identity=identity,request=q,source_hashes=hashes,bounds=bounds,anonymous_mapping=key['assignments'][cid])
            rec=save_figure(dest,private_identity,lambda q=q:request_figure(q,meshes,centers[q['request_id']],bounds))
            images.append(dict(request_id=rid,condition_id=cid,parent_id=q['parent_id'],panel=q.get('panel',''),candidate=q['candidate'],image_path=str(dest),image_sha256=rec['image_sha256']))
            for anon in ANON:
                for stage in ('RAW','POST'):
                    reviews.append(dict(**{k:q.get(k,'') for k in META},anonymous_label=anon,stage=stage,
                        review_status='NOT_REVIEWED',reviewer='AI_PROXY',model_names_hidden=True,prior_familiarity='UNASSESSED',
                        image_path=str(dest),image_sha256=rec['image_sha256'],reviewed_image_sha256='',
                        **{k:'UNKNOWN' for k in FIELDS},notes=''))
            rel='images/'+dest.name;cards.append('<h2>'+html.escape(rid)+'</h2><a href="'+rel+'"><img loading="lazy" style="width:100%;max-width:2430px" src="'+rel+'"></a>')
            print(json.dumps(dict(stage='RENDER',pages=len(images),expected=48,request_id=rid)),flush=True)
        dest=GALLERY/'contact_sheets'/(cid+'__POST_NOISES.jpg')
        private_identity=dict(public_identity=identity,requests=qs,source_hashes=hashes,bounds=bounds,anonymous_mapping=key['assignments'][cid])
        rec=save_figure(dest,private_identity,lambda:contact_figure(qs,meshes,bounds))
        contacts.append(dict(condition_id=cid,parent_id=qs[0]['parent_id'],panel=qs[0].get('panel',''),noise_count=len(qs),image_path=str(dest),image_sha256=rec['image_sha256']))
        for anon in ANON:
            contact_reviews.append(dict(condition_id=cid,parent_id=qs[0]['parent_id'],anonymous_label=anon,stage='POST',noise_count=len(qs),
                review_status='NOT_REVIEWED',reviewer='AI_PROXY',model_names_hidden=True,image_path=str(dest),image_sha256=rec['image_sha256'],
                reviewed_image_sha256='',coarse_style_variation='UNASSESSED',near_identical_across_noises='UNKNOWN',notes=''))
        verify(hashes)
        C.write(PRIVATE/'render_progress.json',dict(status='RENDERING',request_pages=len(images),contact_sheets=len(contacts)),replace=True)
    if len(images)!=48 or len(reviews)!=288 or len(contacts)!=8:raise ValueError('Incomplete gallery')
    C.csvwrite(GALLERY/'images.csv',images);C.csvwrite(GALLERY/'POST_contact_sheets.csv',contacts)
    C.csvwrite(GALLERY/'AI_PROXY_review_template.csv',reviews);C.csvwrite(GALLERY/'AI_PROXY_noise_style_template.csv',contact_reviews)
    C.write(PRIVATE/'source_hashes.json',dict(input_output_hashes=all_sources,definition_hashes=defs))
    links=''.join('<li><a href="contact_sheets/'+html.escape(Path(x['image_path']).name)+'">'+html.escape(x['condition_id'])+'</a></li>' for x in contacts)
    content='<!doctype html><meta charset="utf-8"><title>Noise-transfer anonymous review</title><h1>Noise transfer: anonymous model comparison</h1><p>48 fixed requests; all RAW/POST, with two local sections. A/B/C are fixed within each condition only. Do not read the separate model key before judgments. Familiar images may reveal identities; record prior familiarity. No quality scores or source completions are displayed.</p><h2>POST noise contact sheets</h2><ul>'+links+'</ul>'+''.join(cards)
    C._write_bytes(GALLERY/'index.html',content.encode('utf8'))
    owned=[PLAN,GALLERY/'definition.json',GALLERY/'images.csv',GALLERY/'POST_contact_sheets.csv',GALLERY/'AI_PROXY_review_template.csv',GALLERY/'AI_PROXY_noise_style_template.csv',GALLERY/'index.html']
    owned += [C.resolve(x['image_path']) for x in images+contacts]
    result=dict(status='RENDERED_NOT_REVIEWED',identity=identity,request_pages=48,POST_noise_sheets=8,review_rows=288,style_review_rows=24,
                seconds=time.perf_counter()-tick,output_hashes={str(p):C.sha(p) for p in owned},actual_visual_review='NOT_RUN_BY_RENDERER',model_calls=0,quality_calls=0)
    C.write(final,result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=('freeze-blind','render','check-plan'))
    args=parser.parse_args()
    if args.stage=='freeze-blind':result=freeze_blind()
    elif args.stage=='check-plan':
        plan,_=load_plan();result=dict(status='PASS',conditions=len(plan['conditions']),requests=48,plan_sha256=C.sha(PLAN),key_sha256=C.sha(KEY),model_calls=0)
    else:result=render()
    print(json.dumps(C.clean({k:v for k,v in result.items() if k not in ('output_hashes','identity')}),allow_nan=False),flush=True)


if __name__=='__main__':main()
