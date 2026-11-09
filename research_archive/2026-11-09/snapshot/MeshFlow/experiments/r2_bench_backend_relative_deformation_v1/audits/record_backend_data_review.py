"""Transcribe manual image review after actual view_image inspection, without metrics."""
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/mnt/e/MeshFlow/experiments/r2_bench_backend_relative_deformation_v1')
M = json.loads(Path('/mnt/e/MeshFlow/experiments/r2_bench_geo_only_acceptance_v1/manifest.json').read_text())
axes = ['overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']
good = ['REASONABLE','PLAUSIBLE','PRESENT','NO_OBVIOUS','NO_OBVIOUS']
gap = ['REASONABLE','PLAUSIBLE','PRESENT','YES','NO_OBVIOUS']
ambiguous = ['REASONABLE','PLAUSIBLE','PRESENT','UNCERTAIN','UNCERTAIN']
damaged = ['INCOMPLETE','PLAUSIBLE','MISSING_OR_DAMAGED','YES','NO_OBVIOUS']
replacement = ['REASONABLE','REPLACED_OR_IGNORED','PRESENT','NO_OBVIOUS','YES']
wire_replacement = ['INCOMPLETE','REPLACED_OR_IGNORED','MISSING_OR_DAMAGED','YES','YES']
slat_replacement = ['UNCERTAIN','REPLACED_OR_IGNORED','PRESENT','UNCERTAIN','YES']

# All entries below are judgments made from the rendered images, not generated from counts.
manual = {
10: ([gap]*6, [
    'Blue upper panel plausibly forms the seat of a complete box bench. Thin vertical slits/opening remain at the left end; no clear second seat layer.',
    'Blue top still serves as the seat, with both end supports. A distinct white end notch/slit remains; no obvious new folded layer.',
    'Box bench and blue seat remain recognizable. The left end retains narrow gaps; no clear replacement seat.',
    'Blue seat and box body retained. Left end opening remains visible in the second surface view; central sections do not establish full closure.',
    'Candidate retains the complete box arrangement and blue top, but the left end notch is still visible. No evident gross change from preceding adapted backend.',
    'Final candidate has the same visible left end notch and otherwise coherent box bench. Blue top remains plausible; no obvious second seat layer.'
]),
11: ([good]*6, [
    'Coherent box bench with blue seat/top and both end supports; faceting at the ends but no obvious detached patch or open hole at this scale.',
    'Box body and blue seat remain coherent in both views; central sections retain the end walls and no obvious new layer.',
    'Blue top remains in the seat role; end walls are present and no obvious hole or loose fragment appears in the surface views.',
    'Complete box bench retained; faceted left end is not by itself a visible crack. No clear duplicate seat.',
    'Candidate retains blue top and end support surfaces. Small end faceting/protrusion is visible but not clearly detached or a hole.',
    'Final candidate remains a coherent box bench with blue top; no obvious detached fragment or second seat layer at the inherited views.'
]),
12: ([ambiguous]*6, [
    'High-backed bench with blue curved seat and two substantial end supports. Back section has multiple short pieces; thickness versus discontinuity is uncertain.',
    'Back, blue seat and end supports remain present. Section pieces alone do not certify a floating fragment or a folded duplicate surface.',
    'Overall high-backed bench retained and C plausibly forms the seat. Interior section layering and small gaps remain visually ambiguous.',
    'Blue seat, back and supports retained. Fragmentary section lines do not establish a detached surface; no obvious macro replacement seat.',
    'Candidate has the same high back and end supports around the blue seat. No clear macro upgrade; minor gaps/layering cannot be resolved in these views.',
    'Final candidate keeps the plausible seat and full back/end structure. Interior section ambiguity remains; no claim of gap-free geometry.'
]),
13: ([ambiguous]*6, [
    'Blue seat is integrated in the visual arrangement of a high-backed bench. Small end notch and interior section pieces are ambiguous at this scale.',
    'Back and supports remain visible with blue seat. Small surface/section discontinuities cannot be distinguished confidently from faceting.',
    'Complete main bench structure and blue seat retained; minor end opening and internal layer interpretation remain uncertain.',
    'No major missing back or support, and C retains a plausible seat role. Small section gaps do not alone prove floating fragments.',
    'Candidate resembles prior adapted backend: back and ends survive, blue seat stays exposed. Minor crack/layering assessment remains uncertain.',
    'Final candidate still has a complete main back/end structure around the blue seat. No clear resolution of the minor visual ambiguities.'
]),
14: ([good,damaged,good,damaged,damaged,damaged], [
    'Blue three-bay curved seat is plausibly used, with two broad continuous end/support panels and a lower seat shell. No obvious major end holes.',
    'Blue seat remains, but both gray end/support panels have conspicuous missing strips and holes; section fragments agree with visibly broken surfaces.',
    'Both end/support panels are substantially continuous and the blue three-bay seat is retained; no obvious detached patch at the viewed scale.',
    'Blue seat retained but end panels still show many open strips/holes, especially the rear-facing right end. Main support surfaces are damaged.',
    'Candidate still has shredded/open end panels around the intact blue seat. It does not restore the broad continuous end surfaces seen in RAW/GEO-only.',
    'Final candidate retains the same visible end-panel holes and strip fragmentation. C role remains plausible, but support-surface damage remains.'
]),
15: ([good,damaged,good,damaged,damaged,damaged], [
    'Blue three-bay seat has two broad end supports and a coherent lower shell. End panels are visually continuous despite triangulation.',
    'Large missing strips/holes break both end panels, while the blue seat persists. Supports are visibly damaged rather than merely absent from a central section.',
    'End panels and blue seat remain coherent; no obvious major hole or separate small piece in the two surface views.',
    'Both end panels have open strips and cutouts. Blue seat is plausible but the surrounding support surfaces remain incomplete.',
    'Candidate preserves blue seat but end panels still have substantial holes/strip-like fragmentation. No qualitative recovery to the RAW end-panel condition.',
    'Final candidate has the same end holes and damaged support shell. No obvious change in seat role or folding beyond this persistent defect.'
]),
16: ([replacement]*6, [
    'Overall bench, back rail and end supports are recognizable, but a gray seat lies above blue C in both sections; C remains a lower separate layer.',
    'Whole bench structure survives, but gray seat above blue C persists. The back-rail opening looks intentional, not evidence of a crack.',
    'Seat-height mismatch remains visible: gray seat above the blue panel. End supports remain present with no obvious detached patch.',
    'Back rail/end supports retained, but central sections still show gray seat above blue C rather than adoption of C as the main seat.',
    'Candidate retains the same complete bench and the same separated gray-above-blue seat relation. No obvious macro repair of C use.',
    'Final candidate still has a gray seat above C, while supports remain present. This is a visible replacement/layer relation, not merely a small weld gap.'
]),
17: ([replacement]*6, [
    'Complete bench with back rail and end supports, yet the gray seating surface is above blue C; the side section makes the separate layer clear.',
    'Bench shape remains coherent but blue C is still below the gray seat. No obvious missing support or independent floating fragment.',
    'Blue panel remains lower than the gray seating surface in both fixed sections; end supports and back rail are present.',
    'The gray-above-blue seat relation is unchanged despite a coherent overall bench. Intentional back opening is not counted as a defect.',
    'Candidate retains supports and overall bench but does not resolve the seat substitution/vertical separation visible in the sections.',
    'Final candidate remains a complete-looking bench with C below the generated gray seat. No visible qualitative correction of C adoption.'
]),
18: ([['REASONABLE','PLAUSIBLE','PRESENT','NO_OBVIOUS','UNCERTAIN'],damaged,['REASONABLE','PLAUSIBLE','PRESENT','NO_OBVIOUS','UNCERTAIN'],damaged,damaged,damaged], [
    'Blue panel visually serves as the seat; broad back and rectangular end frames are present. Gray lower surface may be an underside; its layering is uncertain.',
    'Blue seat and a few back strips remain but supporting legs/frames are almost entirely lost, with isolated thin segments. Clearly incomplete.',
    'Blue seat, back and end frames retain coherent surfaces. Lower gray layer cannot confidently be called an improper duplicate from these views.',
    'More leg outlines remain than with official D0, but the back/end surfaces have become skeletal thin strips; substantial support-surface loss persists.',
    'Candidate keeps wire-like end frames and sparse back strips, but not the solid end/back surfaces visible in RAW. Blue seat remains in a plausible role.',
    'Final candidate still has skeletal leg frames and sparse back surfaces. This remains incomplete/damaged despite retention of the blue seat.'
]),
19: ([replacement,wire_replacement,replacement,slat_replacement,slat_replacement,slat_replacement], [
    'Four-leg table/bench body is coherent, but its gray top is well above the blue input panel; C is a separate lower layer.',
    'Gray top has become sparse strips and legs extremely thin/broken-looking, while C remains below it. Overall/support completeness is visibly degraded.',
    'Four-leg body remains coherent with gray top above blue C. No obvious detached fragment, but C does not supply the generated top.',
    'Four legs remain visible, with a sparse slatted/strip-like upper surface above C. Whether this constitutes a complete slatted design is uncertain.',
    'Candidate keeps four visible legs and thin upper strips above the blue panel. Completeness and gaps are uncertain; the replacement-layer relation is clear.',
    'Final candidate preserves visible legs and the sparse upper strip surface, still above C. No clear adoption repair; possible missing skin remains uncertain.'
])
}
rows=[]
for i in range(10,20):
    q=M['tasks'][i]
    rid=q['request_id']
    comp=json.loads((ROOT/'gallery'/rid/'complete.json').read_text())
    assert hashlib.sha256(Path(comp['path']).read_bytes()).hexdigest()==comp['image_sha256']
    labels,notes=manual[i]
    assert len(labels)==len(notes)==len(comp['cells'])==6
    for cell,values,note in zip(comp['cells'],labels,notes):
        row={k:comp[k] for k in ['request_id','parent_id','condition_id','noise_index','eval_group']}
        row.update({k:cell[k] for k in ['cell_id','column','label','stage']})
        row.update(image_path=comp['path'],image_sha256=comp['image_sha256'],reviewer='backend_data',reviewer_kind='AI_PROXY',blind=False,actually_viewed=True,actual_view_source='functions.view_image original; two surface views and two fixed central sections',viewed_at_utc=datetime.now(timezone.utc).isoformat(),note=note)
        row.update(dict(zip(axes,values)))
        rows.append(row)
assert len(rows)==60 and len({r['cell_id'] for r in rows})==60
out=ROOT/'gallery'/'review_shards'/'backend_data.json'
out.parent.mkdir(parents=True,exist_ok=True)
assert not out.exists()
out.write_text(json.dumps({'review_scope':'manifest task indices 10..19, six columns each; labels based on actual rendered views, not score tables','rows':rows},indent=2)+'\n')
print(json.dumps({'path':str(out),'rows':len(rows),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
