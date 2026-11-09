"""Transcribe the named reviewer's actual inspection of all 16 registered pages.

The judgments below are manually specified after opening every page. This
script only attaches frozen identities/image hashes and writes unchanged cells.
It does not infer labels, inspect numeric scores, render or call a model.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path('/mnt/e/MeshFlow')
RUN = ROOT/'experiments/r2_seam_abc_v1'
OUT = RUN/'evaluation'
MODELS = {'L':'LONG','A':'A_COUPLE','B':'B_POOL47','C':'C_SHARED_TIP'}
LABELS = {'Y':'YES','N':'NO','U':'UNCERTAIN'}
FIELDS = ['C_adoption','whole_body_complete','support_missing','fragments_or_folded_layers','visible_cracks','visible_join']

# Six labels use the order in FIELDS. Every RAW and POST tuple is an independent
# transcription of an actually inspected page, including when labels agree.
DATA_85 = [
(0,'RAW','L','YYNNUY','Blue seat, four legs and rounded back remain. A large triangular opening is visible within the back panel; small rough gaps occur around the rim.','REFERENCE','The back opening may be a style or a missing panel; the image cannot resolve that distinction.'),
(0,'RAW','A','YYNNUY','Blue seat and four legs remain. The back is more filled, with a narrow white slot below its rounded top.','The large triangular back opening is replaced by a much narrower top slot; body scale is close to LONG.','The narrow top opening may be intentional or a crack.'),
(0,'RAW','B','YYNNNY','Blue seat sits above a continuous box-like apron, four thin legs and a flat rectangular back. No prominent hanging sheet is visible.','Changes to a rectangular back and thinner legs; no large back opening like LONG.','Different chair style is admissible; absence of a visible defect does not certify hidden surfaces.'),
(0,'RAW','C','YYNUUY','Blue seat, thin legs and rectangular back remain. The front apron has an uneven triangular lower edge and a small open-looking region beneath the seat.','Back and legs resemble B rather than LONG; front apron is less uniformly filled than B.','The apron shape may be an intentional angled panel or a local missing/folded patch.'),
(0,'POST','L','YYNNUY','The rounded-back chair and four legs remain after processing. The triangular back opening is still visible; the outer surfaces look smoother.','REFERENCE','The back opening remains ambiguous rather than automatically counted as a defect.'),
(0,'POST','A','YYNNUY','The back panel is largely filled and the narrow slot under the rounded top remains. Seat and supports look continuous at this scale.','Compared with LONG the large triangular opening is absent, but a narrow upper slot persists.','A narrow top slot alone does not distinguish style from a missing strip.'),
(0,'POST','B','YYNNNY','Flat rectangular back, four thin legs and broad apron remain visibly coherent around the blue seat.','A different, relatively regular rectangular-back style replaces LONG rounded back.','Internal overlap beneath blue C remains hidden.'),
(0,'POST','C','YYNUUY','Rectangular back and four supports remain. The front apron is still uneven and a light triangular/open area is visible under its front edge.','No LONG-like back opening, but front apron looks less uniformly closed than B.','The front apron may be angled by design; its local surface closure is uncertain.'),
(1,'RAW','L','YYNYNY','A long gray sheet hangs from the rear of the seat/back almost to leg-bottom height. The seat, rectangular back and four legs otherwise remain.','REFERENCE','The extra rear sheet is visibly distinct; hidden intersections cannot be judged.'),
(1,'RAW','A','YYNNNY','Blue seat, four legs and rectangular back form a regular chair silhouette. The long hanging rear sheet seen in LONG is not visible.','Removes the conspicuous long rear sheet while preserving the main chair.','No visible hanging sheet does not establish absence of hidden overlap.'),
(1,'RAW','B','YUUYUY','Blue C occupies the seat region, but the body has a large triangular slab dropping below its front/right and a broad L-shaped foot or shelf on the opposite side.','Replaces the thin-leg chair with irregular wide panels and an oversized low foot; it is not simply a cleaner version of LONG.','Whether the altered supports form a complete intended structure is unclear; do not require four separate legs as a style rule.'),
(1,'RAW','C','YYNUUY','Seat, back and four legs remain; no long rear sheet like LONG. A pointed triangular front-apron extension and small light gaps are visible below C.','The long rear sheet is gone, but a pointed front-apron region remains.','The pointed apron can be an angled surface or a folded fragment; the view is inconclusive.'),
(1,'POST','L','YYNYNY','The long rear hanging sheet survives processing alongside the four legs and rectangular back.','REFERENCE','The visibly unwanted-looking extra sheet is retained; topology is not certified by the image.'),
(1,'POST','A','YYNNNY','A regular four-leg silhouette remains with no long rear hanging sheet; the apron and back look continuous.','The clear LONG rear-sheet problem is not visible in this output.','Near-coplanar hidden layers under blue C cannot be excluded.'),
(1,'POST','B','YUUYUY','The large downward triangular panel and broad low foot/shelf persist. The blue seat remains nominally usable but support organization is irregular.','The additional large slab/foot shapes persist after processing instead of converging to LONG thin-leg body.','A pedestal/panel support is not itself invalid; this particular irregular arrangement is not confidently complete.'),
(1,'POST','C','YYNUUY','The four-leg chair has no LONG rear sheet, but the triangular front-apron point remains below the blue seat.','Avoids the rear long sheet, with an unresolved front-apron irregularity.','Cannot determine whether the pointed apron is intended decoration or a folded patch.'),
(2,'RAW','L','YYNNUY','Blue seat and rectangular back remain above four legs. A narrow pale opening appears along the rear apron below the blue seat.','REFERENCE','The small rear opening may be a gap between panels or a legitimate border.'),
(2,'RAW','A','YYNNNY','Four legs, rectangular back and seat apron look coherent; no conspicuous open stripe like the one under LONG rear seat is visible.','Slightly more continuously filled rear apron; the overall chair remains close to LONG.','This visual continuity does not certify shared indices.'),
(2,'RAW','B','YYNNNY','The seat apron is broadly filled and the four-leg body remains regular. The rectangular back is similar to LONG.','A largely similar body with a more uniformly filled apron in the visible views.','Subtle seat-layer differences may be occluded by blue C.'),
(2,'RAW','C','YYNUYY','A large white opening crosses the left-side apron; a narrow lower strip remains below it, and the front apron drops to a triangular point. Back, seat and four legs remain.','Introduces a conspicuous side-apron opening relative to LONG while retaining the main body.','The open side reads as a local torn/missing surface; whether adjacent strips are folded is less certain.'),
(2,'POST','L','YYNNUY','The four-leg chair remains and a small pale stripe persists near the rear apron under C.','REFERENCE','The rear stripe is too small to distinguish a crack from a designed border.'),
(2,'POST','A','YYNNNY','The blue seat sits over a relatively continuous apron; the rectangular back and four legs remain.','Visible rear apron is more continuously filled than LONG.','Fine hidden folds or overlaps cannot be assessed.'),
(2,'POST','B','YYNNNY','Regular rectangular back, apron and four legs remain; no large side opening is apparent.','Overall close to LONG, with a more uniformly filled visible apron.','The image alone does not establish a numerical quality improvement.'),
(2,'POST','C','YYNUYY','The large left-side apron opening and residual lower strip survive processing. The front triangular apron point is still visible.','The local side opening is substantially larger than the small LONG rear stripe.','Main body is retained; the defect is local rather than a missing whole support system.'),
(3,'RAW','L','YYNNNY','Blue seat, box-like apron, rectangular back and four thin legs are all present, with no prominent loose sheet or opening.','REFERENCE','The lower interior remains occluded by the seat.'),
(3,'RAW','A','YYNNUY','The body is very similar to LONG; a tiny light notch appears along the back top edge.','No clear overall change besides a small top-edge irregularity.','The notch may be a render-scale gap or a real tiny missing strip.'),
(3,'RAW','B','YYNNUY','A thinner seat frame and taller open gap below a smaller back panel give a light chair style. A thin diagonal light slit crosses the back.','Changes style to a thinner apron and smaller back, rather than merely repairing LONG.','The wide gap below the back is a plausible chair opening; only the small diagonal slit is uncertain.'),
(3,'RAW','C','YYNUYY','Seat, back and four legs remain, but the left-side apron contains an open band with narrow strips; the front edge has a pointed triangular panel.','Adds a conspicuous side-apron opening relative to LONG.','The side band looks locally missing; the triangular panel could be intentional or folded.'),
(3,'POST','L','YYNNNY','The processed chair retains a coherent apron, rectangular back and four supports around the blue seat.','REFERENCE','No claim about hidden seat overlap follows from the smooth exterior.'),
(3,'POST','A','YYNNUY','Very similar to LONG after processing; a tiny light top-edge irregularity on the back remains visible.','No clear whole-chair improvement or loss relative to LONG.','The small top notch is unresolved at this rendered scale.'),
(3,'POST','B','YYNNNY','The thinner-frame chair remains coherent. The back opening is a regular style gap; the earlier small diagonal slit is no longer conspicuous.','Different thinner-frame style with no obvious new major defect.','The style change itself is not a success or failure criterion.'),
(3,'POST','C','YYNUUY','The side apron looks more filled than in RAW, but retains diagonal structure and a pointed front panel; seat, back and legs remain.','Less uniformly closed apron than LONG, although the large RAW side gap is less apparent.','Residual angled apron geometry may be decoration or folding; closure cannot be certified.')
]

DATA_610 = [
(0,'RAW','L','YYNUUU','Blue C occupies an armchair seat between broad side panels. The boxy body and small feet remain; the front below the seat has several uneven horizontal layers.','REFERENCE','The layers could be cushion/base divisions or folds; the seat-to-body seam is partly hidden.'),
(0,'RAW','A','UUNYYU','The blue plate extends over a narrow stacked base. Large white wedges separate it from folded-looking side/front layers, although a back and small feet remain.','The armchair side enclosure of LONG is lost and the seat/base relationship is more fragmented.','C is at nominal seat height, but its integration into the lower body is unclear; hidden support connections are not visible.'),
(0,'RAW','B','YYNNNY','A broad low armchair body with thick sides, a back and small feet surrounds the blue seat. The visible panels are relatively coherent.','Uses a wider, lower box-armchair style with less conspicuous front layering than LONG.','This is an alternative completion style, not a match to a unique source shape.'),
(0,'RAW','C','UUUYUU','A polygonal back rises above the blue plate; below it are large angular stacked panels and an offset flat base. The load-bearing organization is not clear.','Replaces LONG enclosed armchair body with irregular angular pedestal-like layers.','A pedestal is admissible, but these views cannot confirm that the offset stacked structure is a complete coherent support.'),
(0,'POST','L','YYNUUU','The armchair-like body remains with blue seat between side panels and several front base layers. Processing smooths the surfaces without removing the layered appearance.','REFERENCE','Panel divisions versus unintended folding remain ambiguous below the seat.'),
(0,'POST','A','UUNYYU','The blue plate still overhangs the stacked base with a conspicuous white wedge beside/below it. Large folded-looking panels remain.','The fragmented seat/base arrangement remains worse-looking than LONG enclosed side panels.','The visual role of C and exact connection are uncertain despite its seat-like height.'),
(0,'POST','B','YYNNNY','The thick-sided low armchair remains visibly complete and relatively regular around the blue seat.','A different coherent box-armchair body, with fewer conspicuous irregular front layers than LONG.','Blue C can hide coincident free surfaces; no hidden-quality guarantee is implied.'),
(0,'POST','C','UUUYUU','The angular pedestal/base arrangement and stacked panels persist below the blue plate, with the polygonal back above.','No clear resolution of the irregular support organization relative to LONG.','Cannot distinguish a fully connected unusual pedestal design from damaged/disconnected folded panels.') ,
(1,'RAW','L','UYNUUU','Four legs and a short slatted-looking back remain. The blue board extends well beyond the narrower lower seat/body, including behind the back, with separated contours in central cuts.','REFERENCE','C is at seat height but natural whole-patch adoption and continuous seat-to-body contact are uncertain; thickness and layers can be confused.'),
(1,'RAW','A','UYNUNU','Four legs, short back and an oversized blue plate resemble LONG. A broad rear triangular gray surface is visible, while the lower body remains narrower than C.','Mostly similar to LONG; no clear correction of the oversized plate versus lower body.','The rear triangular surface may be back thickness or a folded extension; under-seat connections are occluded.'),
(1,'RAW','B','UYNNNU','A wider slatted-looking back and a rectangular lower seat box remain above four legs. Blue C still projects outside that lower body.','The back looks broader and the rear triangular extension is less apparent than LONG.','The large overhang and separated central contours prevent a confident natural-use or joining judgment.'),
(1,'RAW','C','UYNUNU','A short solid-looking back and splayed legs remain below the broad blue plate. The lower seat outline is narrower and interrupted in the cuts.','Changes back and leg proportions without visibly resolving the C/body size mismatch.','A different reasonable style is allowed; whether the large board is coherently adopted remains uncertain.'),
(1,'POST','L','UYNUUU','The short-back four-leg chair remains, with blue C extending beyond the narrower seat body. The central sections still show separated layers/contours.','REFERENCE','Smooth exterior panels do not resolve whether the lower contour is legitimate thickness or an alternate seat surface.'),
(1,'POST','A','UYNUNU','The back, legs and wide blue plate persist; the narrower base and rear triangular gray region remain.','Very similar overall to LONG; no clear gain in integrating the full blue patch.','Blue C occludes the lower seat and the triangular rear area is not unambiguously a defect.'),
(1,'POST','B','UYNNNU','The broader back and four legs remain coherent, but the blue seat board still overhangs the lower rectangular seat box.','Some silhouette cleanup relative to LONG; full-patch integration remains uncertain.','Do not equate the nominal seat position with proven C adoption or legal connection.'),
(1,'POST','C','UYNUNU','The solid back and splayed supports remain, under an oversized blue board. Interrupted lower outlines persist in the finite cuts.','Different proportions without a decisive visible improvement in seat/base integration.','Central cuts can miss off-plane connections and blue C can conceal extra layers.'),
(2,'RAW','L','UYNUUU','Four thick legs and an open slat back remain. The blue board is much wider than the lower seat, and short sloping edge pieces stop above the lower contour in the cuts.','REFERENCE','It is unclear whether this is a thick beveled seat or a plate not fully integrated with the narrower chair.'),
(2,'RAW','A','UYNUUU','The slat-back chair and four supports resemble LONG. A rear gray wedge and separate lower seat contours remain under the broad blue plate.','No clear improvement in the full C/body relation; back and supports remain similar.','Slat openings are intentional-looking and are not labelled cracks; the under-seat gaps are ambiguous.'),
(2,'RAW','B','UYNUUU','The slat back and four legs remain. The lower rectangular body is somewhat fuller, but still sits inside the wide blue-board footprint.','A slightly fuller lower seat box, with the same visible overhang problem as LONG.','Possible legitimate seat thickness and hidden layer overlap cannot be separated visually.'),
(2,'RAW','C','UYNUNU','The back becomes a compact solid panel with a large triangular rear surface; four legs remain under a broad projecting blue board.','Changes slat back to solid back, while preserving the same oversized C versus lower-body relation.','The rear triangle may express back thickness; no unique source-style requirement is imposed.'),
(2,'POST','L','UYNUUU','The open-back chair and supports remain. The broad blue board still has separated lower contours and a narrower seat body beneath it.','REFERENCE','Natural adoption and visible continuous joining remain unresolved after processing.'),
(2,'POST','A','UYNUUU','Slat back, four legs and broad blue plate remain; the under-seat contours remain separated.','Overall very close to LONG without a decisive correction of the seat/base relation.','Open back slats are admissible; the uncertain gap concerns the seat/base region.'),
(2,'POST','B','UYNUUU','The lower seat box looks fuller, but the blue board still projects past it and the cuts retain multiple separated contours.','Some local regularization of the lower box; no clear full-patch integration gain.','Layer count and legality cannot be inferred from these cuts alone.'),
(2,'POST','C','UYNUNU','The solid-back chair remains with its rear triangular surface, splayed legs and broad blue board.','A different back style, rather than a visible resolution of LONG C/body mismatch.','Under-seat and rear-surface relations remain ambiguous and partly occluded.'),
(3,'RAW','L','UYNUNU','A rectangular back with a circular opening, four broad legs and a large blue board are present. The board extends beyond the narrower seat/base; the underside has layered contours.','REFERENCE','The round back opening is a plausible style feature, not a crack. C integration and hidden layers remain uncertain.'),
(3,'RAW','A','UYNUNU','The circular-opening back and broad-leg body are almost the same as LONG, including the wide blue board and lower inset contours.','No clear whole-chair difference or improvement relative to LONG.','The unusual wide board can be an overhang, but its full integration is not visually established.'),
(3,'RAW','B','UYNUNU','The circular back opening, four broad legs and wide blue plate remain; small extra internal section segments appear under the plate.','Very similar silhouette to LONG, with no decisive adoption/connection improvement.','Short section segments can represent thickness or hidden folds; they do not by themselves prove intersections.'),
(3,'RAW','C','UYNUNU','The broad-leg chair and round back opening remain. The blue plate is wider than the lower body and the rear/underside shows interrupted section outlines.','Similar overall to LONG with small shape changes; natural seat integration remains unresolved.','The round opening is a legitimate possible style; lower interruptions may be occluded geometry.'),
(3,'POST','L','UYNUNU','The rounded-hole back and four broad supports remain, with the same projecting blue board and narrower lower body.','REFERENCE','The exterior looks smoother, but the board/body relation is still uncertain.'),
(3,'POST','A','UYNUNU','The chair remains close to LONG in back, legs, board overhang and lower contours.','No clear visual improvement or major new damage relative to LONG.','Image appearance does not settle hidden double layers or indexed joins.'),
(3,'POST','B','UYNUNU','The circular-opening back and broad supports remain. The wide blue board and separate lower seat contours persist.','Mostly the same macro structure as LONG, with no decisive visible integration gain.','The under-seat contour variation may be thickness or layering, so it stays uncertain.'),
(3,'POST','C','UYNUUU','The circular-opening back and supports remain; small blue/light slivers appear around the back panel edge, and the underside remains interrupted.','No clear C/body integration gain over LONG; a small back-edge irregularity is newly conspicuous.','The tiny back-edge sliver may be a local crack or visibility artifact. The round central opening itself is not treated as a defect.')
]

NOTES = {
'85': '''# 85: actual AI_PROXY image review

All eight pages (n0–n3, RAW and POST) were opened and inspected across LONG, A_COUPLE, B_POOL47 and C_SHARED_TIP. These are 32 new, nonblind records; no historical visual labels were inherited and no numerical geometry scores were consulted.

- n0: LONG retains a triangular back opening. A fills most of it but keeps a narrow upper slot. B/C use a rectangular-back style; the different style is not itself an improvement criterion. C has an uncertain uneven front-apron region.
- n1: A visibly lacks LONG's conspicuous long rear hanging sheet. C also lacks that rear sheet, but retains a pointed front-apron shape. B introduces a large downward triangular slab and an unusually broad low foot/shelf; its overall support organization is uncertain rather than confidently complete.
- n2: A/B are relatively regular around the blue seat. C has a conspicuous side-apron opening and residual narrow lower strip in both RAW and POST. The four-leg body and C seat role remain; local damage is recorded independently.
- n3: LONG/A are close. B changes to a lighter, thinner-frame chair with a smaller back; plausible back openings are not counted as defects. C has a clear RAW side-apron opening; POST fills more of the side but leaves uncertain angled/pointed apron structure.

Blue C is visibly a seat in this parent. Visible contact is not certification of indexed shared edges, legal topology or absence of hidden FF/I. Small back openings and angled apron shapes retain UNCERTAIN where style cannot be separated from a defect. This panel shows a specific A improvement at n1, not a universal visual win for every new arm.

[Full 32-row record](85.csv). The image hashes in every row bind the actually viewed page.
''',
'610': '''# 610: actual DEV AI_PROXY image review

All eight pages (n0–n3, RAW and POST) were opened and inspected across all four model columns. These are 32 new, nonblind DEV records. No numerical CF/FF/I or connectivity results were used to assign labels.

- n0: LONG is an armchair-like completion with ambiguous layered front/base panels. B forms a relatively coherent broad low armchair around C. A has conspicuous seat/base wedges and folded-looking layers; C has an irregular stacked pedestal-like arrangement. A/C completeness and C adoption remain uncertain. A different pedestal or armchair style is allowed, so absence of four separate legs is not automatically a defect.
- n1: All models preserve a chair body and supports, but the large blue board extends beyond the narrower lower seat/body and partly behind the back. B's broader back looks cleaner in silhouette, but that does not settle whether the complete C patch is naturally adopted. C adoption and joining remain UNCERTAIN.
- n2: LONG/A/B have an open slat back; C changes it to a solid back with a rear triangle. The large board versus narrow lower body persists across methods and processing. Slat openings are plausible style features, not automatically cracks. Under-seat contours remain ambiguous between legitimate thickness and separated layers.
- n3: All have broadly similar chairs with a circular back opening. This opening is not labelled as damage. The wide blue board and lower inset contours remain; C's POST has a small ambiguous back-edge sliver. There is no decisive full-patch adoption/connection improvement in these views.

C being at seat height is insufficient for an unqualified adoption YES when the whole blue patch projects well beyond a narrower body and the finite cuts leave its integration unresolved. These cases are UNCERTAIN, not NO. Central cuts can miss off-plane connections, and blue C can hide coincident free layers. The review does not establish disconnected topology, physical stability or absence/presence of numerical intersections.

[Full 32-row record](610.csv). Every image was actually viewed; labels were not inherited from cached LONG geometry.
'''
}

def main():
    schema=json.loads((OUT/'review_schema.json').read_text(encoding='utf-8'))
    manifest=json.loads((RUN/'eval_manifest.json').read_text(encoding='utf-8'))
    requests={q['request_id']:q for q in manifest['requests']}
    dst=OUT/'reviews';dst.mkdir(exist_ok=True)
    for short,prefix,data in [('85','85b16941',DATA_85),('610','610b1621',DATA_610)]:
        rows=[]
        assert len(data)==32 and len({(n,s,m) for n,s,m,*_ in data})==32
        for noise,stage,model,flags,observation,change,uncertainty in data:
            req_id=f'{prefix}_n{noise}';q=requests[req_id]
            image=OUT/'gallery'/f'{req_id}_{stage}.png'
            image_hash=hashlib.sha256(image.read_bytes()).hexdigest()
            receipt=json.loads(image.with_suffix('.json').read_text(encoding='utf-8'))
            assert receipt['request_id']==req_id and receipt['stage']==stage and receipt['image_sha256']==image_hash
            assert len(flags)==6 and set(flags)<=set(LABELS)
            row=dict(request_id=req_id,parent_id=q['parent_id'],split=q['split'],noise_index=noise,stage=stage,
                model=MODELS[model],reviewer='AI_PROXY_SEAM_EVIDENCE',review_kind='AI_PROXY',
                review_status='ACTUALLY_VIEWED',blinding='NONBLIND_MODEL_LABELS_VISIBLE',
                **{f:LABELS[v] for f,v in zip(FIELDS,flags)},visible_observations=observation,
                change_vs_LONG=change,uncertainty=uncertainty+' Blue C can hide coincident free surfaces; visual contact is not a legal-topology certificate.',
                image_path=image.relative_to(ROOT).as_posix(),image_sha256=image_hash)
            assert set(row)==set(schema['csv_columns'])
            rows.append(row)
        assert {(r['noise_index'],r['stage'],r['model']) for r in rows}=={(n,s,m) for n in range(4) for s in schema['stages'] for m in schema['model_order']}
        out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=schema['csv_columns'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
        for path,content in [(dst/(short+'.csv'),out.getvalue().encode('utf-8')),(dst/(short+'.md'),NOTES[short].encode('utf-8'))]:
            if path.exists():assert path.read_bytes()==content,'Refuse changed existing review'
            else:path.write_bytes(content)
        print(json.dumps(dict(parent=short,rows=32,pages=8,status='ACTUALLY_VIEWED_RECORDED',csv_sha256=hashlib.sha256((dst/(short+'.csv')).read_bytes()).hexdigest())))

if __name__=='__main__':main()
