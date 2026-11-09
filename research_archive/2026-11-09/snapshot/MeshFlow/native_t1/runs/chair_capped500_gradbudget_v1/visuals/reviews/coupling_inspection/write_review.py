"""Retain manually authored observations after actual inspection of all48 sheets.

This binds notes to immutable rendered evidence; it does not infer visual labels
from metrics, rerender meshes, or recalculate any geometry.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
DEST = HERE.parent
LABELS = ['START','HIST_TEACHER500','HIST_CAPPED500','T500','T1000','L500','L1000','G500','G1000']
VIEWS = ['front_3q','back_3q','front','side','wire']

def note(text, sheet='not_seen', overlap='not_confirmed', major='not_seen', extra='not_seen', compare=None):
    return dict(observation=text, sheet=sheet, overlap=overlap, major=major,
                new_major_artifact=extra, comparison=compare)

NOTES = {
('OLD','866e',0): {
'START': note('Recognizable complete frame chair. Layered narrow triangles roughen the ladder back, seat rim and legs; the circular outline of C is not completed as a round opening.', 'confirmed', 'possible'),
'HIST_TEACHER500': note('Coherent round-opening back, continuous seat and leg assemblies; small rim and back-to-C slits remain. Cleaner global structure than START.'),
'HIST_CAPPED500': note('Coherent round-opening back and solid seat; small triangular seams near C and the hole rim. Similar body completeness to historical teacher.'),
'T500': note('Coherent round-opening chair, continuous seat and legs. Jagged small triangles at the opening and C seam, without new large missing assembly.'),
'T1000': note('Chair body retained; narrow triangular cracks and small spikes around the back opening and C joint. Not a large body loss relative to START.'),
'L500': note('Round-opening chair remains recognizable, but a split narrow rear strip and irregular back rim are more apparent than T500; localized deterioration.', 'confirmed', 'possible'),
'L1000': note('Complete chair with round opening; narrow rear-side strips and back-edge cracks remain. No major body loss compared with T1000.', 'confirmed'),
'G500': note('Complete chair with rounded opening, continuous seat and leg assemblies. Small rim slits, similar overall completeness to T500.'),
'G1000': note('Complete chair; uneven opening rim and a narrow side/rear sliver remain. No new major body failure relative to T1000.', 'confirmed'),
},
('OLD','866e',1): {
'START': note('Seat, back and legs present, but back opening and seat contain conspicuous angular overlapping-looking flaps and ragged gaps; legs have thin side fragments.', 'confirmed', 'possible'),
'HIST_TEACHER500': note('Round opening and seat are more coherent than START; a narrow rear-side strip and small under-seat fragment persist.', 'confirmed'),
'HIST_CAPPED500': note('Round opening is clearer, but an upright triangular fin projects above the seat and a thin hanging fragment remains below; whole chair retained.', 'confirmed', 'possible', compare='New seat fin versus historical teacher; localized defect, not disappearance of the seat.'),
'T500': note('Whole chair retained; a visible upright seat fin and small under-seat/floor fragments are clear in side and reverse views.', 'confirmed', 'possible'),
'T1000': note('Round-opening chair retained, with an upright seat flap and a larger under-seat triangular fragment. This is a local artifact, not an absent leg assembly.', 'confirmed', 'possible'),
'L500': note('Smoother visible seat than T500, with a small hanging strip and scattered tiny lower fragments; rounded back and all major body portions retained.', 'confirmed'),
'L1000': note('Seat fin seen in T1000 is reduced; a diagonal under-seat sheet and narrow rear-side strip are visible. Body remains a recognizable chair.', 'confirmed', 'possible'),
'G500': note('Seat is cleaner than T500; thin underside/rear strips and an angular round-opening rim remain. No major body loss.', 'confirmed'),
'G1000': note('Main seat surface is cleaner than T1000 and G500, while narrow rear-side strips and ragged small hole-rim triangles remain.', 'confirmed'),
},
('OLD','86ec',0): {
'START': note('Complete high-backed chair with slats. Small top-cap cracks and irregular slat edges are visible; seat and legs are coherent.'),
'HIST_TEACHER500': note('Complete high-backed slatted chair; modest local top and slat seams, similar to START.'),
'HIST_CAPPED500': note('Complete chair retained; several slats contain thin triangular splits. No missing body assembly.', 'confirmed'),
'T500': note('Complete slatted chair with a rougher top cap and narrow cracks; no clear new major failure.'),
'T1000': note('Complete chair with cleaner seat and retained slat structure; a thin top-cap split remains.'),
'L500': note('Complete chair, but top/back cap is flared and displaced relative to blue posts; a small seat-side spike is visible.', 'confirmed', 'possible'),
'L1000': note('Complete chair retained; flared top/back joint and slat seams remain, without wholesale back or seat loss.'),
'G500': note('Complete chair; rough top cap, irregular slat strips and localized back gaps are more apparent than T500.', 'confirmed'),
'G1000': note('Complete chair retained. A triangular flap cuts across the central slat opening, with rough cap fragments; worse local back detail than T1000.', 'confirmed', 'possible'),
},
('OLD','86ec',1): {
'START': note('Recognizable complete slatted chair; jagged cap and narrow strips between slats. Seat and legs remain coherent.'),
'HIST_TEACHER500': note('Complete slatted chair with a few cap and slat cracks; similar overall body to START.'),
'HIST_CAPPED500': note('Complete chair with uneven/sliver-like slats and a rough top joint. These are local defects, not missing major assemblies.', 'confirmed'),
'T500': note('Solid back replaces slats; this style change alone is not a failure. A distinct narrow strip hangs below the seat between the legs.', 'confirmed', 'possible'),
'T1000': note('Slat openings return; the extra hanging strip conspicuous at T500 is no longer visible at the same scale. Seat remains present.'),
'L500': note('Coherent solid-backed chair. Compared with T500, the hanging central strip is reduced; small leg-edge slivers persist. Solid back is not itself a defect.', 'confirmed'),
'L1000': note('Solid-backed chair with intact seat and leg assemblies; small front-leg/seat-edge cracks. Style differs from T1000, without demonstrated body collapse.'),
'G500': note('Coherent solid-backed chair with small leg-edge slivers; less conspicuous hanging strip than T500. No style-based failure assigned.', 'confirmed'),
'G1000': note('Solid-backed chair retains all main portions; small cracks in front legs and seat apron. No major missing assembly relative to T1000.'),
},
('FRESH','866e',2): {
'START': note('Ladder-back chair body retained. Several thin diagonal seat/back strips and rough joints interfere with an otherwise visible seat.', 'confirmed', 'possible'),
'HIST_TEACHER500': note('Coherent round-opening chair with solid seat; noticeably cleaner than START, with only small back-rim seams.'),
'HIST_CAPPED500': note('Round-opening chair remains coherent, but a floating-looking triangular seat flap and small underside fragment are clear in side view.', 'confirmed', 'possible', compare='Local seat artifact relative to historical teacher; body completeness retained.'),
'T500': note('Close to START ladder-back form with layered thin seat/back strips; no new missing assembly.', 'confirmed', 'possible'),
'T1000': note('Seat appears more continuous, but top/back opening is ragged and partly fragmented. Main chair body remains visible.', 'confirmed', 'possible'),
'L500': note('Back becomes a warped jagged panel with a compressed irregular opening; bulky leg/rear strips and uneven seat layers are visible. Worse local coherence than T500.', 'confirmed', 'possible', 'possible', 'possible'),
'L1000': note('Round-opening back and seat recover coherence compared with L500 and T1000. Thin rear/under-seat strips and a lower spike remain.', 'confirmed'),
'G500': note('Ragged mixed back opening with sharp strips above the seat; visible seat/back gaps and layered triangles. Recognizable chair remains.', 'confirmed', 'possible', 'possible', 'possible'),
'G1000': note('New sharp triangular protrusions span above the seat; reverse view shows a pronounced triangular seat opening, with ragged back. Worse local seat structure than T1000; severity as whole-body collapse is uncertain.', 'confirmed', 'possible', 'possible', 'confirmed'),
},
('FRESH','866e',3): {
'START': note('Main seat/back/legs remain, but opening rim is fragmented and seat edges are layered. Chair recognizable with rough details.', 'confirmed', 'possible'),
'HIST_TEACHER500': note('Back opening is covered by a broad surface seen from both front and rear; a small forward spike remains. Body is retained, but C-opening compatibility is worse than START.', 'confirmed', 'possible'),
'HIST_CAPPED500': note('Back opening reappears, though jagged; seat and legs retain chair structure, with rear-side slivers.', 'confirmed'),
'T500': note('Broad diagonal triangles bridge the back opening and a long sharp fin projects toward the seat. Chair body remains present.', 'confirmed', 'possible'),
'T1000': note('Back-opening diagonal sheet persists; seat rim remains rough but the main seat and leg support structure are visible.', 'confirmed', 'possible'),
'L500': note('Crossed raised seat sheets, sharp rear strips and a long triangular under-seat flap are conspicuous; more local clutter than T500. Major whole-body severity remains uncertain.', 'confirmed', 'possible', 'possible', 'confirmed'),
'L1000': note('Seat remains layered with long thin protrusions and a lower wedge; back is broad and irregular. No complete loss of seat/back, but local defects remain severe.', 'confirmed', 'possible', 'possible', 'confirmed'),
'G500': note('Crossed seat layers, ragged back and a conspicuous long blade beneath the seat are visible in side/rear views. Worse local artifacts than T500.', 'confirmed', 'possible', 'possible', 'confirmed'),
'G1000': note('Seat is more continuous and the large underside blade is reduced relative to G500; small seat-level prongs and ragged back persist. Do not carry the full G500 defect forward unchanged.', 'confirmed', 'possible', 'possible', 'possible'),
},
('FRESH','86ec',2): {
'START': note('Complete coherent slatted chair with a small detached-looking lower triangle and narrow slat cracks. No missing body assembly.', 'confirmed'),
'HIST_TEACHER500': note('Complete slatted chair with a tiny lower fragment and small local cap cracks; similar structure to START.', 'confirmed'),
'HIST_CAPPED500': note('Complete chair with relatively coherent seat, legs and slats. Minor top-cap and slat splits remain.'),
'T500': note('Complete coherent slatted chair. The lower floating triangle seen at START is no longer conspicuous; minor seams remain.'),
'T1000': note('Complete chair retained with small cap/slat cracks. No large hanging sheet in the front leg gap.'),
'L500': note('Complete chair, very similar in main body to T500; narrow slat and cap splits remain.'),
'L1000': note('Complete chair retained; slat splits are a little more pronounced, without a large under-seat artifact.'),
'G500': note('Complete coherent chair, similar to T500. Small slat/cap cracks only at this viewing scale.'),
'G1000': note('A large triangular sheet now hangs from the seat toward the floor in the front leg gap, visible in front and oblique views and absent in START, T1000 and G500. Main seat/back/legs still exist; major-body-collapse severity is possible, not certified.', 'confirmed', 'possible', 'possible', 'confirmed'),
},
('FRESH','86ec',3): {
'START': note('Coherent high-backed chair with broad slats and jagged cap; seat and legs intact. Slat count/style is not judged against GT as a failure.'),
'HIST_TEACHER500': note('Complete chair with broad slats, modest cap cracks, and a small lower fragment. No missing body assembly.', 'confirmed'),
'HIST_CAPPED500': note('Chair body retained, but a long narrow triangular strip hangs under the seat; irregular slats remain.', 'confirmed', 'possible', compare='Definite new local underside strip versus historical teacher, without whole-body loss.'),
'T500': note('Coherent solid-backed chair with a rough cap; the replacement of slats by a panel is not itself a visual failure.'),
'T1000': note('Broad central back panel/slat and leg assemblies remain coherent; cap roughness persists, without the large hanging underside sheet.'),
'L500': note('A large triangular sheet projects into the under-seat space in side and reverse views. The broad back also has a long diagonal gap. Body recognizable but local artifact severity is high.', 'confirmed', 'possible', 'possible', 'confirmed'),
'L1000': note('Slat openings return; the large L500 under-seat triangle reduces to a much narrower hanging strip, still visible in side/rear views. Body retained.', 'confirmed', 'possible'),
'G500': note('Broad back panel with diagonal gap and rough cap. Compared with L500 the conspicuous under-seat sheet is not seen; body remains coherent.'),
'G1000': note('A long triangular hanging strip appears under the seat, broader than the narrow L1000 strip; back slats are irregular. Main seat/back/legs remain visible.', 'confirmed', 'possible', 'possible', 'confirmed'),
},
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def group_for(label):
    if label in LABELS[:3]:
        return 'history'
    return 'primary' if label.endswith('500') else 'stability'

def teacher_for(label):
    if label in ['START','HIST_TEACHER500','T500','T1000']:
        return None
    if label == 'HIST_CAPPED500':
        return 'HIST_TEACHER500'
    return 'T500' if label.endswith('500') else 'T1000'

receipts = []
for panel in ['OLD','FRESH']:
    rows, evidence = [], []
    keys = sorted(k for k in NOTES if k[0] == panel)
    for _, short, candidate in keys:
        metas = {}
        for group in ['history','primary','stability']:
            for overlay in [False, True]:
                stem = f'{panel}_{short}_n{candidate}_{group}' + ('_overlay' if overlay else '')
                path = HERE / (stem + '.json')
                meta = json.loads(path.read_text())
                image = HERE / (stem + '.png')
                assert digest(image) == meta['png_sha256']
                metas[group, overlay] = meta
                evidence.append(dict(inspection_image=str(image), inspection_image_sha256=digest(image),
                                     inspection_metadata=str(path), inspection_metadata_sha256=digest(path),
                                     actually_viewed=True, **meta))
        for label in LABELS:
            n = NOTES[panel,short,candidate][label]
            group = group_for(label)
            m, ov = metas[group,False], metas[group,True]
            teacher = teacher_for(label)
            compare_note = n['comparison'] or n['observation']
            row = dict(request_id=m['request_id'],uid=m['uid'],label=label,candidate=candidate,
                missing_body=dict(status='not_seen', notes='No complete seat, back, or leg assembly is visibly absent in these five views. Narrow cracks/openings are separately described; this does not certify watertightness.'),
                duplicate_overlap=dict(status=n['overlap'],notes='Layered or crossing-looking surfaces are visual observations only; exact duplicate faces, penetration and topology are not inferable from these raster views.'),
                long_thin_sheet=dict(status=n['sheet'],notes=n['observation']),
                chair_plausibility=dict(status='recognizable_with_local_defects',notes=n['observation']),
                certainty=dict(visible_shape='high',hidden_surface_or_exact_intersection='uncertain',major_body_severity=n['major']),
                vs_START=dict(reference_label='START',new_major_body_failure='not_applicable' if label=='START' else n['major'],
                    new_major_artifact='not_applicable' if label=='START' else n['new_major_artifact'],notes=compare_note),
                vs_matched_T=dict(reference_label=teacher,new_major_body_failure='not_applicable' if teacher is None else n['major'],
                    new_major_artifact='not_applicable' if teacher is None else n['new_major_artifact'],notes='No distinct matched teacher comparator for this reference label.' if teacher is None else compare_note),
                evidence_pages=[dict(path=x['evidence_page'],sha256=x['evidence_page_sha256'],kind='SIE' if i else 'four_views_and_wire') for i,x in enumerate([m,ov])],
                gallery_manifest=m['gallery_manifest'],gallery_manifest_sha256=m['gallery_manifest_sha256'],
                camera_manifest=m['camera_manifest'],camera_sha256=m['camera_sha256'],
                inspection_metadata=[str(HERE/(f'{panel}_{short}_n{candidate}_{group}'+suffix+'.json')) for suffix in ['', '_overlay']],
                view_scope=VIEWS,feedback_scope=['S','I','E'],actually_viewed=True,
                overlay_notes='All saved S/I/E overlays were viewed. Colored samples localize registered feedback near C; they are not used to certify absence of remote free-surface defects or to recalculate metric values.')
            rows.append(row)
    assert len(rows)==36 and len({(r['request_id'],r['label']) for r in rows})==36
    result = dict(reviewer_type='AI_VISUAL_REVIEW', reviewer='coupling', scope_status='PARTIAL',panel=panel,
        assigned_scope_complete=True, actually_viewed=True, human_review='NOT_RUN',
        parent_ids=sorted({r['uid'] for r in rows}),request_ids=sorted({r['request_id'] for r in rows}),
        label_order=LABELS,outputs=rows,inspection_evidence=evidence,
        scope=dict(parents=2,requests=4,outputs=36,full_view_sheets=12,feedback_sheets=12,
                   views=VIEWS,feedback_views=['S','I','E'],original_panel_pixels=320,pixels_resized=False,
                   generated_mesh_or_render_or_score=False,CONFIRM_read=False),
        interpretation='Partial visual observations, not an all-eight-parent gate. Different back styles, leg counts or departure from GT are not automatically failures. Confirmed visible thin-sheet artifacts are separated from possible whole-body severity. Exact penetration, watertightness and physical usability were not tested.',
        recovery_notes=['OLD86ec n1 T500 hanging strip is not carried forward as persisting at T1000.',
                        'FRESH866e n2 L500 ragged back is substantially more coherent at L1000.',
                        'FRESH866e n3 G500 large underside blade is reduced at G1000; local prongs remain.',
                        'FRESH86ec n3 L500 broad underside sheet reduces to a narrow strip at L1000.'],
        authored_after_actual_viewing_utc=datetime.now(timezone.utc).isoformat(),
        source_path=str(Path(__file__).resolve()),source_sha256=digest(__file__))
    target=DEST/f'partial_coupling_{panel}.json'
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    receipts.append(dict(path=str(target),sha256=digest(target),outputs=len(rows),scope_status='PARTIAL'))
(HERE/'review_receipt.json').write_text(json.dumps(dict(status='COMPLETE_ASSIGNED_SCOPE',reviews=receipts,
    helper_source_sha256=digest(HERE/'materialize.py'),writer_source_sha256=digest(__file__),
    actual_images_viewed=48,model_calls=0,scorer_calls=0,render_calls=0),indent=2)+'\n')
print(json.dumps(receipts,indent=2))
