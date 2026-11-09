"""Transcribe 12 pages actually opened by AI_PROXY; no metric-based ratings."""
from pathlib import Path
import json,hashlib,datetime
R=Path('/mnt/e/MeshFlow');root=R/'experiments/r2_chair_multipart_seam_v1'
t=json.loads((root/'visual_review_templates/seat_part_01.json').read_text())
def s(flags,notes):return (flags,notes)
data={
'3f741759_n0':dict(stages=[
s('YYUYNY','Blue C serves as the seat of a complete four-support chair. Narrow orange strips occupy parts of the back openings; rough small section offsets remain, without a clearly detached full-view fragment.'),
s('YYNYNY','The gray chair is more regular and C remains the seat. Thin orange back strips persist; section endpoints do not establish an actual crack.'),
s('YYNYNY','The overall chair remains intact; a small green patch appears at the front/side apron. Thin orange back strips remain and exact welds cannot be read from the image.'),
s('YYUYNY','Seat and four supports remain recognizable. Orange triangular sheets are broader and extend farther into several back openings; a small separate-looking cut trace is not proof of a detached triangle.'),
s('YYNYNY','The gray chair stays regular but the tall orange fin-like back patches remain. The central sections retain open-looking endpoint gaps; no exact topology is inferred.'),
s('YYNYNY','No clear macroscopic change from new BASE. Broader orange back-opening infill and a small orange cut trace persist; no demonstrated global repair.')],
changes=['MIXED','MIXED','MIXED'],positive='The blue seat role and complete legged body are retained.',cost='New output has broader orange fin-like sheets in back openings. Their legality and whether the opening should remain empty are not determined by these views.',
processing=[('BETTER','Gray small contour discontinuities are reduced and FINAL adds a small visible front/side patch; the chair and back-strip arrangement remain similar.'),('NO_CLEAR_CHANGE','Postprocessing regularizes gray traces but does not visibly remove the larger orange fins or change the macro body.')]),
'3f741759_n1':dict(stages=[
s('YYUYNY','Seat, back and four supports form a recognizable chair; the RAW top cap is slightly rough. Thin orange strips enter the back openings.'),
s('YYNYNY','Gray top cap and lower cut are more regular; blue C still forms the seat. Orange strips remain narrow.'),
s('YYNYNY','Whole-chair silhouette and back strips are essentially unchanged from BASE; fixed cuts look more regular without certifying shared edges.'),
s('YYUYNY','The chair stays complete with the same blue seat; several broad orange tapered faces rise from the seat into the back openings.'),
s('YYNYNY','Gray structure is regular, but the new broad orange tapered faces still occupy much of the lower back openings.'),
s('YYNYNY','New FINAL retains the broad orange fins; no clear removal of this additional visible clutter is seen.')],
changes=['MIXED','MIXED','MIXED'],positive='Overall chair completeness and blue seat use are retained.',cost='The new model adds large tapered orange back-opening patches relative to the old narrow strips; images alone do not classify them as illegal.',
processing=[('BETTER','The rough RAW gray cap and section segments become more regular; no large change of chair structure.'),('NO_CLEAR_CHANGE','The same complete body and broad back fins persist through the processing stages.')]),
'3f741759_n2':dict(stages=[
s('YYUYNY','Blue seat, four supports and a tall gray back remain. Thin oblique orange pieces are visible within the back openings; the cap has small irregularities.'),
s('YYNYNY','The gray body is regular and the orange back strips remain. No isolated full-view fragment is clear at this scale.'),
s('YYNYNY','A small green apron patch appears; the chair and thin orange back strips are otherwise retained.'),
s('YYUYYY','New RAW keeps the chair but adds broad orange back fins and a conspicuous long taper from the seat down beside a gray support. This is an abnormal thin appendage, not evidence that the gray support is missing.'),
s('YYUYYY','The long orange taper beside the leg and the back fins survive BASE processing. The body remains present, but the surface arrangement is visibly cluttered.'),
s('YYUYYY','FINAL does not clearly remove the long leg-side taper or broader back fins. Exact intersection and attachment of these patches remain UNKNOWN.')],
changes=['WORSE','WORSE','WORSE'],positive='The seat role and major chair structure are still recognizable.',cost='A large narrow orange leg-side appendage and broader back fins appear in the new model; this is a visible cleanliness cost independent of matching a source style.',
processing=[('NO_CLEAR_CHANGE','Small gray regularization and a green apron patch do not change the overall body or back strips.'),('NO_CLEAR_CHANGE','The conspicuous orange taper and back fins persist after processing; no clear visual rescue.')]),
'3f741759_n3':dict(stages=[
s('YYUYNY','Blue C is the seat of a regular four-support chair. The back has only narrow orange strips and the RAW cap is slightly rough.'),
s('YYNYNY','Gray edges look cleaner; the whole chair and sparse back strips are retained.'),
s('YYNYNY','No clear macroscopic change from old BASE; blue seat and gray supports remain complete in the displayed views.'),
s('YYUYNY','The same main chair shape is present, but tall orange tapered sheets fill more of the back openings and the upper cap is rough.'),
s('YYNYNY','The new cap is more regular after BASE, while the large orange back sheets persist.'),
s('YYNYNY','New FINAL keeps the additional orange sheets; no visible reduction in that fin-like clutter.')],
changes=['WORSE','WORSE','WORSE'],positive='Blue seat use and the principal chair body are retained.',cost='Compared with the old sparse back strips, the new model introduces more tall fin-like orange patches; no macro quality gain is apparent.',
processing=[('NO_CLEAR_CHANGE','Small gray regularization leaves an already similar complete chair silhouette.'),('NO_CLEAR_CHANGE','The upper gray cap becomes cleaner but broad orange back patches remain; no clear overall change.')]),
'73b369b9_n0':dict(stages=[
s('YYUUNN','A cantilever-like seat/back/base assembly is recognizable and blue C is the seat. An oblique orange strip lies on the rear face; close blue/gray section layers may represent thickness.'),
s('YYUUNN','The base and back become smoother, but the oblique rear orange strip and separated parallel section traces persist. Fine crack status is unclear.'),
s('YYUUNN','The same recognizable assembly and rear orange strip remain; no large visual change from BASE.'),
s('YYUUNN','The new chair preserves seat/back/base proportions while the conspicuous oblique rear orange strip is absent. The blue/gray thickness offset in cuts persists.'),
s('YYUUNN','New BASE has a cleaner rear surface without the old diagonal orange strip; the major cantilever-like form remains intact.'),
s('YYUUNN','A small green side/corner patch is visible; the cleaner rear surface remains. Parallel cut layers and exact joining remain uncertain.')],
changes=['BETTER','BETTER','BETTER'],positive='The conspicuous old rear orange strip is absent, making that face visibly cleaner while retaining the seat/back/base assembly.',cost='Close parallel blue/gray layers and some cut endpoint gaps persist; no conclusion about legal connection or thickness follows.',
processing=[('NO_CLEAR_CHANGE','Gray edges become a little more regular but the rear orange strip and global shape persist.'),('NO_CLEAR_CHANGE','BASE and FINAL keep the cleaner rear surface; the small green patch is local and not a demonstrated whole-object change.')]),
'73b369b9_n1':dict(stages=[
s('YYUUNN','Blue C sits in a plausible cantilever-like seat/back/base assembly. RAW has slight cap/rim irregularities and close parallel cut traces.'),
s('YYUUNN','Gray planar faces are more regular. A small orange rear seat-corner segment remains in the side cut; its exact connection is not visible.'),
s('YYUUNN','A green segment fills part of the rear corner in the side cut and gray back/base are preserved. This is a local visible patch, not proof of legal attachment.'),
s('YYUUNN','New RAW has a similar complete seat/back/base form and a fairly clean rear panel. Blue/gray parallel layers remain.'),
s('YYUUNN','New BASE preserves the assembly; gray section traces are smoother, with a remaining underside separation.'),
s('YYUUNN','New FINAL looks similar to its BASE; no clear macro restoration or new gross damage is visible.')],
changes=['NO_CLEAR_CHANGE','NO_CLEAR_CHANGE','NO_CLEAR_CHANGE'],positive='Both models preserve a recognizable seat/back/base structure and plausible C role.',cost='Small triangulation/rim differences and cut gaps are insufficient to rank exact connection or hidden surface quality.',
processing=[('BETTER','Gray faces are regularized and a local green rear corner patch appears; the rest of the body is similar.'),('NO_CLEAR_CHANGE','The overall assembly and parallel thickness traces persist without a clear macroscopic change.')]),
'73b369b9_n2':dict(stages=[
s('YUYUNN','The seat/base remain recognizable, but the arched gray back has a visible long slit and a small opening near the top in the surface view.'),
s('YUYUNN','The long upper-back slit remains visible after BASE; the seat/base and blue C are retained.'),
s('YUYUNN','FINAL preserves the upper-back opening rather than visibly closing it. Fine connection and whole-surface completeness remain uncertain.'),
s('YUYUNN','The new upper back has less of the old long horizontal slit, but a conspicuous opening remains along its upper-left boundary. Seat/base are retained.'),
s('YUYUNN','The upper-left opening remains after BASE; the rest of the seat/back/base assembly is recognizable.'),
s('YUYUNN','New FINAL still has the upper-left back opening; there is no convincing complete repair of the back surface.')],
changes=['MIXED','MIXED','MIXED'],positive='Part of the old long horizontal back slit is less conspicuous in the new output.',cost='A different visible opening remains at the upper-left back boundary, so the new result is not a clean repair or a complete-surface certification.',
processing=[('NO_CLEAR_CHANGE','The back slit remains through RAW, BASE and FINAL.'),('NO_CLEAR_CHANGE','Postprocessing does not clearly close the visible upper-left back opening.')]),
'73b369b9_n3':dict(stages=[
s('YYUUNN','Seat/base and broad arched back are recognizable. Slight RAW edge irregularity is visible, without a clear large missing back patch.'),
s('YYNUNN','The old back is a relatively continuous gray surface and blue C remains the seat; the underside cut offset persists.'),
s('YYNUNN','Old FINAL keeps the continuous-looking back and seat/base structure, with no obvious new large fragment.'),
s('YUYUNN','New RAW has a small conspicuous notch/opening at the upper back rim and irregular upper gray edges; seat/base remain recognizable.'),
s('YUYUNN','The new upper-back notch remains after BASE even though planar gray edges are smoother.'),
s('YUYUNN','The small upper-back rim opening persists in new FINAL; no clear correction is visible.')],
changes=['WORSE','WORSE','WORSE'],positive='The seat/base assembly and C role are preserved.',cost='A small visible upper-back notch/opening appears in the new output compared with the cleaner old back; this is a local surface cost, not a source-style mismatch.',
processing=[('NO_CLEAR_CHANGE','Old postprocessing regularizes a few lines without altering the recognizable structure.'),('NO_CLEAR_CHANGE','Gray regularization does not visibly remove the new top-rim opening.')]),
'85b16941_n0':dict(stages=[
s('YYUUNN','The blue seat is part of a slender-legged chair with a broad rectangular back. Small RAW rim/section irregularities remain; the thin legs are coherent supports rather than clearly missing.'),
s('YYNUNN','The old chair retains regular slender supports and a broad back; the lower gray cut is more regular.'),
s('YYNUNN','A green apron patch is visible; the same complete-looking old chair is retained without a large new appendage.'),
s('YUY YNY'.replace(' ',''),'The new model changes to a rounded back and thicker legs, which is not itself a failure. However the back cap is ragged, with visible gaps and orange fin-like patches at the lower back and leg sides.'),
s('YUY YNY'.replace(' ',''),'New BASE regularizes the gray cap but has a conspicuous diagonal opening through the back panel; orange lower-back and leg-side sheets persist.'),
s('YUY YNY'.replace(' ',''),'The diagonal back opening and orange appendage-like sheets survive FINAL. Main chair parts remain present, but whole-surface completeness is uncertain.')],
changes=['WORSE','WORSE','WORSE'],positive='Blue seat use and a legged chair structure are preserved; the changed back/leg style is allowed.',cost='The new output shows a ragged/partly open back and conspicuous orange appendage-like sheets; after processing a diagonal back gap remains. The cost is surface quality, not style.',
processing=[('BETTER','Small RAW gray irregularities are reduced and a local apron patch appears; major structure is maintained.'),('MIXED','Gray edges become smoother, but a diagonal back opening remains conspicuous and the orange appendages are not removed.')]),
'85b16941_n1':dict(stages=[
s('YYUUNN','Blue seat, broad back and four slender supports form a recognizable chair. RAW cut endpoints have small offsets.'),
s('YYNUNN','Old BASE preserves a regular chair with an open space between seat and raised back; no large detached fragment is clear.'),
s('YYNUNN','Old FINAL remains close to BASE; the blue seat and gray support structure are retained.'),
s('YYUYN Y'.replace(' ',''),'The new body is similar, but an orange triangular sheet projects into the gap between the rear seat and back, with an oblique orange side-section trace.'),
s('YYNYNY','The added orange rear-seat triangle persists; the gray body and slender supports stay recognizable.'),
s('YYNYNY','The same orange rear-seat infill remains in FINAL, with no clear macro change from BASE.')],
changes=['MIXED','MIXED','MIXED'],positive='The main chair and blue seat role are retained without obvious loss of a support.',cost='An additional orange triangular infill occupies part of the seat/back gap. Whether that is an intended valid transition cannot be certified visually; no empty-hole requirement is assumed.',
processing=[('NO_CLEAR_CHANGE','Gray cut traces become smoother but the old whole-chair appearance is similar.'),('NO_CLEAR_CHANGE','The gray body is regularized while the added rear-seat triangle remains.')]),
'85b16941_n2':dict(stages=[
s('YYUUNN','The old chair has a blue seat, broad gray back and coherent slender supports; small RAW rim and cut offsets are visible.'),
s('YYNUNN','BASE preserves the complete-looking chair and cleans small gray cut traces; exact interfaces remain UNKNOWN.'),
s('YYNUNN','FINAL has no clear macroscopic difference from old BASE in the two surface views.'),
s('YYUUNN','The new chair is close to the old form with no obvious major missing part. Small rim irregularities and short section gaps remain.'),
s('YYNUNN','New BASE keeps the back, seat and slender supports; no clear large crack or detached patch is apparent at the displayed resolution.'),
s('YYNUNN','New FINAL looks similar to new BASE. No clear broad quality advantage or deterioration over the old chair is visible.')],
changes=['NO_CLEAR_CHANGE','NO_CLEAR_CHANGE','NO_CLEAR_CHANGE'],positive='Both models retain the seat role and a similar coherent chair body.',cost='Subtle edge and role-color differences do not establish a visible quality gain; tiny cracks/intersections remain beyond image certification.',
processing=[('NO_CLEAR_CHANGE','Local section lines become cleaner without a clear macroscopic change.'),('NO_CLEAR_CHANGE','Small line regularization does not change the main chair appearance.')]),
'85b16941_n3':dict(stages=[
s('YYYUNN','The old chair remains recognizable, but a small white notch/opening is visible in the top rim of the back in the opposite surface view.'),
s('YYYUNN','The old top-rim notch remains after BASE; blue seat and slender supports are retained.'),
s('YYYUNN','Old FINAL keeps the small back-rim opening; no visible repair of that local defect is clear.'),
s('YYUUNN','The new chair retains the same seat and support organization, with a smoother-looking top back rim and no equally conspicuous notch in these views.'),
s('YYNUNN','New BASE preserves the cleaner-looking top back rim and recognizable chair body.'),
s('YYNUNN','New FINAL retains the smoother top-rim appearance; no major new fragment is apparent at this scale.')],
changes=['BETTER','BETTER','BETTER'],positive='The conspicuous old small back-rim notch is less apparent or absent in the new displayed views while the main chair remains.',cost='This is a local visible improvement; exact hole closure, manifoldness and intersection-free quality are still UNKNOWN.',
processing=[('NO_CLEAR_CHANGE','The old top-rim notch remains despite cleaner section lines.'),('NO_CLEAR_CHANGE','The new cleaner rim is already present in RAW and largely persists through BASE and FINAL.')])
}
assert set(data)=={r['request_id'] for r in t['requests']}
flags=t['flags'];mapping={'Y':'YES','N':'NO','U':'UNCERTAIN'}
for r in t['requests']:
    d=data[r['request_id']];assert len(d['stages'])==6
    for row,(ratings,notes) in zip(r['stages'],d['stages']):
        assert len(ratings)==6,(r['request_id'],ratings)
        row.update({key:mapping[v] for key,v in zip(flags,ratings)});row['notes']=notes
    for row,change in zip(r['old_new_comparisons'],d['changes']):
        row.update(change=change,positive_observations=d['positive'],costs_or_uncertainties=d['cost'])
    for row,(change,notes) in zip(r['within_model_processing'],d['processing']):
        row.update(RAW_to_FINAL_change=change,observations=notes)
    path=Path(r['image_path']);r.update(actual_view_status='ACTUALLY_VIEWED',
        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        viewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        request_notes='Actually opened both complete-object views and both fixed sections for all six columns. AI_PROXY nonblind; blue seat use is semantic only. Finite cuts and role colors do not certify exact connectivity or illegal intersections. No unique source style or required empty aperture is assumed.')
t['status']='ENTERED_ACTUAL_AI_PROXY_REVIEW'
dest=Path(__file__).parent/'entered_review.json'
if dest.exists():raise FileExistsError(dest)
dest.write_text(json.dumps(t,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(dest)
