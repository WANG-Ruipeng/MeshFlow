from pathlib import Path
import json,csv,time,hashlib
r=Path('/mnt/e/MeshFlow-Control-artifacts/followups/readout_precision_path_v1')
p=r/'gallery/gallery_manifest.json';g=json.loads(p.read_text());out=[]
labels=[('LEGACY_BF16','ON'),('LEGACY_BF16','OFF'),('FP32_REFERENCE','ON'),('FP32_REFERENCE','OFF')]
for page in g['pages']:
    m,req=page['model_label'],page['request_id']
    for precision,mode in labels:
        x=dict(model_label=m,request_id=req,panel=page['panel'],precision=precision,readout=mode,label=m+'__'+precision+'__'+mode,
            review_type='AI_VISUAL_REVIEW',status='COMPLETE',human_usability='NOT_RUN',
            blind_page=page['blind_page'],blind_sha256=page['blind_sha256'],reference_page=page.get('reference_page','NA'),reference_sha256=page.get('reference_sha256','NA'),
            views_seen='front_3q;back_3q;front;side;wire',review_order='ALL_8_WITHOUT_REFERENCE_THEN_4_P1_REFERENCE',
            geometry_modification='NONE',hard_C_retention='PASS_FROM_ARRAY_AUDIT; visual blue visibility is not the proof',confidence='MEDIUM',
            reference_interpretation='Source is one completion, not a required chair style' if page['panel']=='P1' else 'NA_NO_GT')
        if '33ec' in req:
            x.update(C_use='Blue seat/frame/leg fragments participate in a recognizable chair; welded attachment not visually certified',
                whole_chair='Coherent seat/back/support silhouette in all four settings; similar tall solid curved-top back',
                missing_or_fragmented='No large missing body or exploded geometry; irregular seat-edge and back-top triangular tips remain',
                overlay_or_unrelated='No clearly new large unrelated appendage; overlap/contact at blue/orange frame cannot be resolved from these views',
                thin_wall='Some layering can be finite panel thickness; cannot label all overlap hits semantic errors',
                shape_vs_defect='Curved/solid back differs from rectangular-open source back; treated as style, not automatic failure',
                comparison='NO_CLEAR_VISUAL_DIFFERENCE in condition-use quality across precision or ON/OFF')
            if m=='C1000' and precision=='FP32_REFERENCE':x['comparison']='Visible support position/overlap changes versus BF16 in both ON and OFF; no clear condition-use improvement'
        elif 'f596' in req:
            arm=m=='B1000' and precision=='LEGACY_BF16'
            x.update(C_use='Blue seat patch lies at the generated seat; outline is interpretable, exact interface quality unresolved',
                whole_chair='Recognizable tall-back '+('armchair' if arm else 'armless chair')+' with complete main seat/back/support silhouette',
                missing_or_fragmented='No large missing body; thin dangling lines/small slivers around supports remain possible',
                overlay_or_unrelated='No clear extra seat/back separate from C; small seat-underbody triangular details need caution',
                thin_wall='Back and seat panel thickness is plausible; small thin slivers do not establish a useful structure',
                shape_vs_defect='Armrests and rounded top are style differences; absence of source crossbars is not automatically missing geometry',
                comparison='NO_CLEAR_VISUAL_DIFFERENCE in condition-use quality among similar armless variants')
            if m=='B1000':x['comparison']='Both FP32_ON and FP32_OFF lose BF16 armrests: shared precision-related style change, not readout-specific quality gain'
            if m=='B1000' and precision=='FP32_REFERENCE' and mode=='ON':
                x['missing_or_fragmented']='Visible triangular flap below seat in front view; possible local geometric defect, no large body loss'
                x['comparison']+='; FP32_ON has more conspicuous under-seat flap than FP32_OFF'
            if m=='C1000' and precision=='LEGACY_BF16' and mode=='OFF':x['missing_or_fragmented']='Small pointed horizontal sliver below seat in front view; no large body loss'
        elif req=='P2_seat2__n0':
            x.update(C_use='C remains at seat-region height but much is covered/embedded by orange panels; visual views do not establish faithful use as an already-complete seat',
                whole_chair='Recognizable low broad bench/sofa-like form; this style is not rejected for lacking four separate chair legs',
                missing_or_fragmented='Diagonal slit/gap in back and lower body, sharp panel/arm tips; no explosion or entire body loss',
                overlay_or_unrelated='Orange seating panels lie over/around C in all settings; uncertain whether some are legitimate thickness or undesirable layering',
                thin_wall='Thickness is a plausible partial explanation; not enough evidence to classify every near-duplicate hit semantically',
                shape_vs_defect='Broad sofa/bench style allowed; diagonal openings and jagged protrusions remain quality concerns',
                comparison='NO_CLEAR_VISUAL_DIFFERENCE in resolution of C integration; only small slit/edge/shape changes',confidence='MEDIUM')
        else:
            dense=(m=='B1000' and precision=='FP32_REFERENCE' and mode=='OFF') or (m=='C1000' and precision=='LEGACY_BF16' and mode=='OFF')
            x.update(C_use='Generated orange back stands at opposite seat edge from supplied blue back, persisting in every cell; ordinary-chair interpretation is questionable',
                whole_chair='Chair-like supports and seat plus '+('dense curved solid/slatted orange back with thin legs' if dense else 'ladder-like orange back and thicker legs'),
                missing_or_fragmented='Main body present; '+('many narrow back strips/thin supports' if dense else 'small seat/support spikes and thin stray lines remain'),
                overlay_or_unrelated='Two separated upright structures at opposite seat edges; generated back does not simply thicken the supplied blue back',
                thin_wall='Seat-depth separation between the two upright panels is too large for a simple thin-wall explanation; alternative unusual chair use remains unverified',
                shape_vs_defect='Ladder versus dense back is a style change; conflicting use of the supplied back is the stronger concern',
                comparison='Switch changes orange-back style in selected OFF cells, but no configuration resolves opposing supplied/generated backs',confidence='MEDIUM_HIGH_FOR_SEPARATED_PANELS; MEDIUM_FOR_SEMANTIC_JUDGMENT')
        out.append(x)
    page['AI_VISUAL_REVIEW']='COMPLETE'
assert len(out)==32 and len({(x['label'],x['request_id']) for x in out})==32
with (r/'visual_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
g['AI_VISUAL_REVIEW']='COMPLETE';g['visual_review_path']=str(r/'visual_review.csv');g['human_usability']='NOT_RUN'
p.write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
review=dict(status='COMPLETE',reviewer='AI_VISUAL_REVIEW',human_usability='NOT_RUN',outputs_reviewed=32,blind_pages_seen=8,reference_pages_seen=4,
    order='All eight source-hidden pages first, then all four source-reference pages',views_per_output=5,method='Actual inline original PNG inspection; standard view_image had Windows1385 error, read-only exec base64 display succeeded',
    inspected_pages=[{k:v for k,v in p.items() if k.endswith('page') or k.endswith('sha256')} for p in g['pages']],
    output_csv_sha256=hashlib.sha256((r/'visual_review.csv').read_bytes()).hexdigest(),time_unix=time.time(),semantics='Qualitative AI observation, not human usability or metric-selected pass/fail')
(r/'visual_review_receipt.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
h=r/'OPEN_GALLERY.html';s=h.read_text();s=s.replace('Rendering is not review.','AI_VISUAL_REVIEW COMPLETE: all32 outputs and all12 pages inspected; human usability NOT_RUN. <a href="visual_review.csv">Per-output review</a>');h.write_text(s)
print(json.dumps(dict(outputs=32,pages=12,status='AI_VISUAL_REVIEW_COMPLETE')))