"""Transcribe actual first-ten-page image review; no metric-derived labels."""
import hashlib,json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'gallery/review_manifest.json').read_text())
axes=['overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']
def rating(shape,use,support,gap,fold,note):
    return dict(zip(axes,[shape,use,support,gap,fold]),note=note)
good=lambda note:rating('REASONABLE','PLAUSIBLE','PRESENT','NO_OBVIOUS','NO_OBVIOUS',note)
pages=[]
pages.append([
rating('UNCERTAIN','UNCERTAIN','PRESENT','YES','NO_OBVIOUS','Tub-like long front/back walls enclose the blue plane below their upper edges; its seating use is ambiguous. Small end openings and ragged end strips remain.'),
rating('UNCERTAIN','UNCERTAIN','PRESENT','YES','NO_OBVIOUS','Same enclosing walls and ambiguous use of C. The end opening is visibly enlarged/simplified; this is not a restored ordinary seat.'),
rating('UNCERTAIN','UNCERTAIN','PRESENT','YES','NO_OBVIOUS','Tub-like organization retained from RAW, including small end openings. C remains enclosed below both long walls.'),
rating('UNCERTAIN','UNCERTAIN','PRESENT','YES','NO_OBVIOUS','Candidate preserves the long walls but does not resolve C-use ambiguity; open end and projecting blue edge remain.'),
rating('UNCERTAIN','UNCERTAIN','PRESENT','YES','NO_OBVIOUS','Final candidate remains visually like its BASE: enclosing walls, open end and uncertain seating use of C.')])
pages.append([
good('Backrest, blue seating surface and both end supports form a coherent bench; no obvious detached fragment in the two views.'),
good('Coherent backrest/seat/end-support arrangement retained; slight contour smoothing does not change the visible task interpretation.'),
good('The same plausible bench arrangement and supports remain; central section agrees with C serving as the seat.'),
good('Candidate retains blue seat, backrest and both end supports. A narrow under-seat rail is visible, without an obvious floating fragment.'),
good('Final candidate retains the same coherent seat/back/support organization as BASE; no obvious new visual damage.')])
pages.append([
good('Box-supported bench with blue upper seating plane; side/end walls provide visible support. Interior horizontal surface could be a box shelf, not a uniquely diagnosed duplicate.'),
good('Box-supported seat survives the original D0 chain. No obvious exterior loss in these two views; internal shelf is not judged against a unique style.'),
good('Box geometry and blue upper seat stay intact. No obvious detached surface in the fixed views.'),
good('Candidate preserves the box-supported blue seat and outer shell; no clear visual failure relative to the paired view.'),
good('Final candidate and BASE have the same visible complete box/seat organization.')])
pages.append([
rating('REASONABLE','REPLACED_OR_IGNORED','PRESENT','NO_OBVIOUS','YES','Outer stool/bench is coherent, but gray upper surface sits far above blue C in both sections; C is an extra lower plane.'),
rating('INCOMPLETE','REPLACED_OR_IGNORED','MISSING_OR_DAMAGED','YES','YES','Gray top still replaces C; large open strips at the top-left corner and fragmented end/leg panel show D0-stage damage.'),
rating('REASONABLE','REPLACED_OR_IGNORED','PRESENT','NO_OBVIOUS','YES','RAW-like outer shell is retained, but the detached lower blue plane remains below a separate gray seating surface.'),
rating('REASONABLE','REPLACED_OR_IGNORED','PRESENT','YES','YES','Candidate keeps more of the curved end shell than official D0 but leaves narrow corner slits. The gray top remains far above blue C.'),
rating('REASONABLE','REPLACED_OR_IGNORED','PRESENT','YES','YES','Same corner slits and wrong-height gray seating plane as candidate BASE; C-use failure is unchanged.')])
pages.append([
rating('REASONABLE','PLAUSIBLE','PRESENT','YES','YES','Seat/back/legs are recognizable and C lies in the seating zone. A triangular back opening and gray panels crossing the blue seating region remain.'),
rating('REASONABLE','PLAUSIBLE','PRESENT','YES','YES','Recognizable chair-like bench with supports; back opening and overlapping/folded-looking seat surfaces remain after official D0.'),
rating('REASONABLE','PLAUSIBLE','PRESENT','YES','YES','Same seat-height adoption and supports, with visible back opening and competing gray seating panels.'),
rating('REASONABLE','PLAUSIBLE','PRESENT','YES','YES','Candidate retains main structure but does not close the triangular back gap; gray seating panels still overlay/obscure much of C.'),
rating('REASONABLE','PLAUSIBLE','PRESENT','YES','YES','Final candidate remains like BASE: main structure present, back gap and gray/blue seating overlap visible.')])
pages.append([
rating('INCOMPLETE','PLAUSIBLE','PRESENT','YES','UNCERTAIN','C is in the seat region, but the remaining seat has conspicuous open holes and sparse strips; legs remain. Layering cannot be resolved confidently from these views.'),
rating('INCOMPLETE','PLAUSIBLE','PRESENT','YES','UNCERTAIN','Seat holes remain conspicuous after original D0; supports persist, and no complete seating surface is restored.'),
rating('INCOMPLETE','PLAUSIBLE','PRESENT','YES','UNCERTAIN','RAW-like perforated/missing seat region remains around blue C, with visible back opening and intact legs.'),
rating('INCOMPLETE','PLAUSIBLE','PRESENT','YES','UNCERTAIN','Candidate keeps legs and back but leaves broad missing seat patches; it does not recover a continuous seating surface.'),
rating('INCOMPLETE','PLAUSIBLE','PRESENT','YES','UNCERTAIN','Same broad seat gaps as candidate BASE. C occupies a plausible seat zone, but completion remains incomplete.')])
for noise in (0,1):
    pages.append([
good('Slatted blue seat and rounded end supports form a complete bench/shelf-like body; the large under-seat opening is intentional-looking, not missing support.'),
rating('INCOMPLETE','PLAUSIBLE','MISSING_OR_DAMAGED','YES','NO_OBVIOUS','Original D0 removes the rounded upper shoulder of the left end support: a conspicuous gap separates the blue seat from that panel.'),
good('GEO-only preserves the rounded end shoulders and full end panels from RAW; blue slats remain the seating surface.'),
rating('REASONABLE','PLAUSIBLE','PRESENT','YES','NO_OBVIOUS','Candidate retains more of the rounded left shoulder and end support than official D0; narrow parallel slits remain at the upper corner.'),
rating('REASONABLE','PLAUSIBLE','PRESENT','YES','NO_OBVIOUS','Final candidate retains the restored-looking end shoulder/support from BASE, with remaining narrow upper-corner slits.')])
for noise in (0,1):
    pages.append([
good('Irregular blue slab sits on substantial gray supports; coherent stone/log-like bench, with no obvious detached piece in the fixed views.'),
good('Blue slab and gray supports remain coherent after original D0; a central cut misses some supports but the surface views show them.'),
good('GEO-only preserves the irregular slab and both support groups without an obvious exterior gap.'),
good('Candidate retains the blue slab and support groups. No obvious new major missing part or floating surface in the views.'),
good('Final candidate is visually similar to BASE; slab/support arrangement remains plausible and complete.')])

assert len(pages)==10
rows=[]
for page_idx,ratings in enumerate(pages):
    entries=manifest['cells_to_review'][page_idx*5:page_idx*5+5]
    assert len(entries)==5 and len({r['request_id'] for r in entries})==1
    image=Path(entries[0]['image_path'])
    assert hashlib.sha256(image.read_bytes()).hexdigest()==entries[0]['image_sha256']
    for col,(base,assessment) in enumerate(zip(entries,ratings)):
        assert base['column']==col
        rows.append(dict(base,**assessment,reviewer='AI_PROXY_backend_scope_audit',reviewer_kind='AI_PROXY',blind=False,
            actually_viewed=True,actual_view_source='functions.view_image',viewed_at_utc='2026-10-23T15:15:32Z',
            review_time_semantics='timestamp of completed first-ten-page review transcription; not a claimed exact per-tool invocation time',
            image_display='3745x1932 source displayed by tool at 2048x1057; two fixed surface views and two fixed sections reviewed',
            metric_values_consulted_for_labels=False,page_index=page_idx))
dest=root/'gallery/review_shards/backend_scope_audit.json'
dest.parent.mkdir(parents=True,exist_ok=True)
with dest.open('x',encoding='utf-8') as f:
    json.dump(dict(status='ACTUALLY_REVIEWED_AI_PROXY',pages=10,cells=50,reviewer='AI_PROXY_backend_scope_audit',
        disclosure='Nonblind AI proxy assessment of saved mesh renders; not independent human certification or topology scoring.',
        rows=rows),f,ensure_ascii=False,indent=2)
print(dest, len(rows))
