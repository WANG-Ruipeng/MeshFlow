"""Literal observations after actually viewing all 16 TRAIN full PNGs.

Lookup is only saved finite-section identity lookup at manually chosen positions;
it never assigns visual judgements, rerenders, rescores or calls a model.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_seam_s2000_section_audit_v1'
QUERIES = [(f'73b369b9_n{n}', 'POST', 'side', [0., -1.15]) for n in range(4)] + [
    ('73b369b9_n1', 'POST', 'side', [-1.4, -.82]),
    ('3f741759_n0', 'POST', 'side', [-.95, .8]),
    ('3f741759_n2', 'POST', 'side', [-.8, .81]),
] + [(f'85b16941_n{n}', 'POST', 'side', [-1.13, .72]) for n in range(4)] + [
    ('85b16941_n2', 'RAW', 'side', [-1.15, .59]),
] + [(f'8c9b2023_n{n}', 'POST', 'side', [-.95, .27]) for n in range(4)] + [
    ('8c9b2023_n3', 'POST', 'front', [0., -.52]),
]

# Each tuple was written after view_image displayed that request's complete
# four-stage PNG. Labels are literal visual observations, not metric rules.
# Listed slots identify selected visible examples, including counterexamples;
# neither their selection nor a separated section line is a defect verdict.
OBSERVATIONS = {
 '73b369b9_n0': ('UNCERTAIN', 'YES', [216], 'Side section: long lower contour below C around z=0,y=-1.13; side cutaway under-seat plate.', 'RAW',
    'The lower contour and the under-seat layer are already visible in RAW and remain after D0 and anchor restoration. POST makes gaps at its end junctions clearer. The cutaway shows a broad underside plate and the lower base, not an unequivocally free-floating triangular scrap.',
    'A lower contour can be the outside of a thick shell. Its separation inside the section does not determine connections outside the plane; purple rims are artificial cuts.'),
 '73b369b9_n1': ('UNCERTAIN', 'YES', [145,41], 'Side section: lower seat contour and short oblique line at rear C junction near z=-1.3,y=-.84.', 'RAW',
    'The short oblique rear-junction line is visible in RAW, shifts with D0, and is again distinct after restoring C; POST retains it. The lower seat contour also persists. The cutaways retain the continuous recognizable chair shell, with no clearly independent loose sheet.',
    'The oblique line can be a seam or wall intersection. No off-plane connection or geometric illegality is certified from these two sections.'),
 '73b369b9_n2': ('UNCERTAIN', 'YES', [85], 'Side section: underside around z=0,y=-1.18; front section: lower contour central gap.', 'RAW',
    'A lower shell surface is present in RAW and all later stages. D0 straightens its section; POST leaves clearer interruptions where the lower contour approaches the vertical returns. The cutaway does not establish that this broad plate is floating.',
    'The openings may be local joins rather than an internal redundant sheet; connections can lie outside the central plane.'),
 '73b369b9_n3': ('UNCERTAIN', 'YES', [55], 'Side section: underside around z=0,y=-1.19 and its rear/front returns.', 'RAW',
    'The lower section line and hollow space below the blue surface occur from RAW onward. POST is straighter but its end gaps remain visible. Whole cutaways retain the base and surrounding shell without a clearly detached tiny face.',
    'A section gap is not proof that the complete underside surface is a detached component or an unwanted duplicate.'),
 '3f741759_n0': ('UNCERTAIN', 'YES', [21], 'Side section: short horizontal line above seat near z=-.95,y=.79; front cutaway: oblique back strips.', 'RAW',
    'The upper short section line is present in RAW, moves downward under D0, and remains after restoration and GEO. Oblique triangular strips in the back opening are also visible throughout. The cutaway locates these around the back structure, not inside the closed seat volume.',
    'An isolated cross-section of a back strip can connect to the frame away from the plane. Whether the oblique strips are a desirable design is not judged against a unique source chair.'),
 '3f741759_n1': ('NO', 'NO', [], 'Central front/side seat sections and both clipped chair views.', 'NONE',
    'The central sections mainly trace the blue seat and a lower enclosing contour; no additional short internal line is clearly visible. D0 smooths RAW irregularities and POST retains a recognizable slatted-back chair. The front right return has an opening but is part of a larger contour, not a separate floating section.',
    'NO means not clearly visible in these two planes and cutaways; hidden or off-plane faces are not excluded.'),
 '3f741759_n2': ('UNCERTAIN', 'YES', [21], 'Side section: short horizontal line near z=-.8,y=.81; front cutaway: back-opening oblique strips.', 'RAW',
    'The short line above C is visible in RAW, tilts slightly in D0, and becomes horizontal in RESTORED and POST. The front cutaway retains sloping back pieces around the open slats. This image identifies a repeatable line but does not show a standalone floating face.',
    'The short line can be a section of an off-plane-connected back piece; complete POST adjacency is required to resolve its connectivity.'),
 '3f741759_n3': ('NO', 'NO', [], 'Central seat sections and front/side clipped views.', 'NONE',
    'No additional upper isolated short segment or clear internal loose sheet is visible. The seat box and regular back frame remain through all four stages; POST removes some RAW surface irregularity while the front right return remains open at its top.',
    'Finite central sections cannot certify absence of hidden sheets, and an open return alone does not prove a detached component.'),
 '85b16941_n0': ('NO', 'YES', [55], 'Side section: upper U-shaped contour near z=-1.13,y=.71, above the seat section.', 'RAW',
    'The upper U-shaped section appears separate from the seat contour in every stage. The clipped side view places it at the broad back panel, with support connections outside this central section. The seat block itself shows no clearly extra internal sheet.',
    'The U contour extends beyond the local section window; apparent separation in this plot is not a full-mesh floating-panel verdict.'),
 '85b16941_n1': ('NO', 'YES', [129], 'Side section: lower edge of upper back-panel U contour near z=-1.13,y=.69.', 'RAW',
    'The back-panel section remains separated from the seat section throughout RAW, D0, RESTORED and POST. The cutaways consistently show a broad back and slender supports. Small light marks below the seat do not clearly resolve into an independent sheet.',
    'Back-support connections are away from the central plane. Fine white marks and cut rims do not prove voids or loose triangles.'),
 '85b16941_n2': ('NO', 'YES', [132,52], 'Side section: RAW short line near z=-1.15,y=.583 below the back-panel base; POST near y=.688.', 'RAW',
    'RAW has a distinct short segment just below the upper U contour. That extra separated line is no longer visually distinct after D0; RESTORED and POST retain the back-panel base instead. Saved section identity locates RAW slot132 at the low line and POST slot132 near slot52 at the panel base. This is a stage change, not evidence that POST has a loose internal sheet.',
    'Overprinting can conceal two nearby sections; disappearance from the raster alone does not mean deletion. Identity lookup is reported separately from the visual label.'),
 '85b16941_n3': ('NO', 'YES', [118], 'Side section: upper L-shaped back-panel contour near z=-1.13,y=.70.', 'RAW',
    'The L-shaped upper section is present from RAW through POST. The side cutaway locates it on the back panel. The front cutaway retains a narrow opening at the right underside/apron, but it does not expose a clear free-floating internal patch.',
    'The missing side of an L section may lie outside the local view or plane; exact surface closure and legality are not inferred.'),
 '8c9b2023_n0': ('NO', 'YES', [63,151], 'Side section: small rectangle above C, bottom around z=-.95,y=.27.', 'RAW',
    'The seemingly floating rectangle is already present in RAW. D0 makes it more regular, and it remains in RESTORED/POST. Front and side cutaways place it at a back crossbar. An extra short RAW line above the rectangle is not clearly distinct in POST.',
    'A crossbar can meet side posts away from the central plane. This isolated rectangle does not by itself demonstrate an internal detached sheet.'),
 '8c9b2023_n1': ('NO', 'YES', [146], 'Side section: open small rectangle above C near z=-.95,y=.28.', 'RAW',
    'The upper section rectangle and the lower seat contour remain across all stages. Its bottom-right opening is visible in POST; the cutaways still show the corresponding back crossbar and vertical frame. No clearly independent interior patch is resolved.',
    'The open section rectangle is a finite-plane observation, not proof of a disconnected back crossbar or a complete surface crack.'),
 '8c9b2023_n2': ('UNCERTAIN', 'YES', [78,35], 'Side section: upper rectangle; side cutaway: inclined light gap immediately below the blue seat.', 'RAW',
    'The upper rectangle exists in RAW and becomes regular after D0; the cutaways place it on the back crossbar. The side cutaway also exposes an inclined light wedge under the seat in D0, RESTORED and POST, whereas RAW has denser overlapping fine lines there. This may be an apron-layer gap; it is not confidently identified as an independent internal sheet.',
    'The inclined wedge is a different view from the earlier rear-apron observation. Their face identity has not been equated, and an artificial purple cut rim is not treated as a crack.'),
 '8c9b2023_n3': ('UNCERTAIN', 'YES', [139,59,27], 'Side section: upper rectangle; front section: long isolated lower contour around x=0,y=-.529.', 'RAW',
    'The lower horizontal seat contour is visibly separated from the short side returns from RAW onward. The upper rectangle is also present throughout. Front cutaways expose a diagonal white opening in the right apron after D0 and it persists through POST; the broader apron remains visible.',
    'A separated lower section may be an underside surface attached off-plane. Neither it nor the white opening alone proves a floating face; the upper rectangle corresponds visually to a back crossbar.'),
}


def write_reviews():
    fields = ['request_id','split','reviewer','actually_viewed','image_path','image_sha256',
              'post_suspect_internal_sheet_visible','post_isolated_section_line_visible',
              'suspect_raw_slots','view_location','first_visible_stage','stage_change_note','uncertainty']
    rows=[]
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8-sig'))
    expected={q['request_id'] for q in protocol['requests'] if q['split']=='TRAIN'}
    assert set(OBSERVATIONS)==expected and len(expected)==16
    for rid, observation in OBSERVATIONS.items():
        sheet,line,slots,where,first,note,uncertainty=observation
        assert sheet in ('YES','NO','UNCERTAIN') and line in ('YES','NO','UNCERTAIN')
        page=OUT/'gallery'/(rid+'.png')
        rec=json.loads(page.with_suffix('.json').read_text(encoding='utf-8-sig'))
        h=hashlib.sha256(page.read_bytes()).hexdigest()
        assert rec['image_sha256']==h
        with np.load(OUT/'gallery'/(rid+'_identity.npz'),allow_pickle=False) as a:
            displayed=set(int(x) for view in ('front','side') for x in a['POST_'+view+'_section_raw_slots'])
        assert set(slots).issubset(displayed), (rid,slots)
        rows.append(dict(zip(fields,[rid,'TRAIN','AI_PROXY_NONBLIND',True,str(page),h,sheet,line,
            json.dumps(slots),where,first,note,uncertainty])))
    with (OUT/'visual_train.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='ACTUALLY_VIEWED_RECORDED',pages=16,rows=16,
        only_full_main_PNGs_viewed=True,model_calls=0,scorer_calls=0,
        CSV_sha256=hashlib.sha256((OUT/'visual_train.csv').read_bytes()).hexdigest())))


def lookup():
    for rid, stage, view, point in QUERIES:
        prefix = stage + '_' + view + '_section'
        with np.load(OUT / 'gallery' / (rid + '_identity.npz'), allow_pickle=False) as z:
            project = [0, 1] if view == 'front' else [2, 1]
            seg = z[prefix + '_segments'][:, :, project]
            slots = z[prefix + '_raw_slots']
        p = np.asarray(point)
        d = seg[:, 1] - seg[:, 0]
        t = np.clip(np.einsum('ij,ij->i', p - seg[:, 0], d) / np.maximum(np.sum(d*d, axis=1), 1e-30), 0, 1)
        gap = np.linalg.norm(seg[:, 0] + t[:, None]*d - p, axis=1)
        order = np.argsort(gap)[:2]
        print(json.dumps(dict(request_id=rid, stage=stage, view=view, manually_selected_position=point,
            nearest=[dict(raw_slot=int(slots[i]), distance=float(gap[i]), endpoints=seg[i].tolist()) for i in order])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lookup', action='store_true')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.lookup:
        lookup()
    elif args.write:
        write_reviews()
    else:
        print('Literal visual observation recording pending identity lookup; no computation dispatched')
