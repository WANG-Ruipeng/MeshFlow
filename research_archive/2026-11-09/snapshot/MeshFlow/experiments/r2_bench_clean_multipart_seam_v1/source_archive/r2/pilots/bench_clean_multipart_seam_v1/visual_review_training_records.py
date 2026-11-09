"""Manual AI_PROXY observations after actually opening the assigned image sheets.

Compact columns: overall R/I/U; C-use P/X/U/A; support P/M/U/A;
visible fragment/gap Y/N/U; layering/fold Y/N/U. No metric-derived labels.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

OUT = Path("/mnt/e/MeshFlow/experiments/r2_bench_clean_multipart_seam_v1")
RECORDS = {}

RECORDS[0] = [
("RPPUN", "Blue end frame carries a continuous gray deck; two far gray supports remain visible but thin. Small edge roughness is not enough to certify a detached fragment."),
("IPMYN", "Blue frame and deck remain, but far free legs are reduced to broken hairline strips in both oblique views; side section retains the frame and deck band."),
("IPMYN", "Final view still has the same impoverished far supports; no visible recovery of leg faces relative to BASE_POST."),
("RPPUN", "New RAW has a full deck and four apparent supports, with ragged free leg edges; blue end frame is used at the deck end."),
("IPMYN", "New BASE_POST keeps the deck but far supports become very thin discontinuous strips; the blue end frame is intact."),
("IPMYN", "New final preserves the blue frame but still lacks substantial far-leg surfaces; no obvious new fold appears."),
("IPMYN", "Native post deck is intact but the opposite free supports are hairline and interrupted; C is positioned as the end frame."),
("IPMYN", "Native final has the same skeletal far supports and intact deck/end frame in both views."),
("IPMYN", "Old SEAM21000 keeps a deck against C but free supports on the far end are greatly thinned."),
("IPMYN", "Old SEAM21000 final does not visibly restore the sparse far-leg surfaces."),
("IPMYN", "Old-layout new generator gives a coherent deck and C end frame, but opposite legs remain broken thin strips."),
("IPMYN", "Old-layout final still has thin interrupted far supports; both cuts show the deck band under C without an obvious fold."),
]
RECORDS[1] = [
("RPPUN", "RAW has a gray deck and two substantial far legs, joined visually to the blue near end frame; minor rim irregularity is uncertain."),
("IPMYN", "After post one far support is nearly absent and the other is a narrow strip; blue support remains well expressed."),
("IPMYN", "Final keeps the same missing/thinned far-leg appearance; deck is present and no broad folded sheet is visible."),
("RPPUU", "New RAW retains both far legs and a deck; the underside has several short uneven rim segments, with their connectivity unclear."),
("IPMYN", "New post far legs become thin partly interrupted strips while the blue end frame stays intact."),
("IPMYN", "New final has essentially the same skeletal opposite legs and deck; no visual recovery of support volume."),
("IPMYN", "Native post loses almost all thickness of the generated far legs; the deck remains against the C end."),
("IPMYN", "Native final still shows hairline far supports and a complete rectangular deck."),
("IPMYN", "SEAM21000 post has visible thin far support lines but not substantial leg faces; blue frame is retained."),
("IPMYN", "SEAM21000 final preserves the same thin-legged silhouette and unbroken deck."),
("IPMYN", "Old-layout new output has far supports reduced to narrow strips, with the deck and blue frame clear."),
("IPMYN", "Old-layout final retains the sparse supports; its two cuts do not show a new large folded panel."),
]
RECORDS[2] = [
("IXPYU", "Blue upper strip and orange side panels sit above a separate gray deck; side cut shows a vertical gap between orange ends and lower gray surface. Gray legs are present."),
("IXMYU", "Upper C/orange assembly remains separated above the deck in the side cut, and generated legs lose most of their faces."),
("IXMYU", "Final retains the upper-versus-lower surface separation and skeletal generated supports."),
("IXPYU", "New RAW has substantial gray legs, but upper blue/orange assembly remains above a separate lower deck; both cuts support the separation."),
("IXMYU", "New post still separates the upper condition assembly and deck, with free supports reduced to thin strips."),
("IXMYU", "New final shows the same separated upper component and weakened legs; no clear restoration is visible."),
("IXMYY", "Native post leaves the blue strip visibly floating above a separate gray seat; free legs are hairline and interrupted."),
("IXMYY", "Native final keeps the distinct floating blue layer and sparse free supports."),
("IXMYU", "SEAM21000 adds orange material below C but it stops above the lower deck in the side section; legs remain very thin."),
("IXMYU", "SEAM21000 final retains the upper/lower gap and thin generated legs."),
("IXMYU", "Old-layout new generator has an orange wall below C, with its ends above a separate gray deck in the cut; legs are sparse."),
("IXMYU", "Old-layout final retains the gap below the upper condition assembly and reduced leg surfaces."),
]
RECORDS[3] = [
("IXPYU", "RAW upper blue/orange component is separated from the lower gray deck in the side cut; four gray supports are visible."),
("IXMYU", "Post has an obvious upper/lower separation and almost vanishing generated leg surfaces."),
("IXMYU", "Final preserves the same split-level separation and sparse supports rather than closing it."),
("IXPYU", "New RAW keeps gray supports but the orange ends beneath C stop well above the lower deck in the side cut."),
("IXMYU", "New post leaves the upper condition assembly apart from the deck and thins the legs to isolated strips."),
("IXMYU", "New final retains separated upper/lower assemblies and markedly thinned supports."),
("IXMYY", "Native post has a floating blue strip above its gray seating slab; far support faces are mostly absent."),
("IXMYY", "Native final does not visibly integrate the floating blue strip or restore the thin supports."),
("IXMYU", "SEAM21000 keeps orange side panels below C but leaves a side-section gap to the separate gray deck; legs are sparse."),
("IXMYU", "SEAM21000 final retains the upper/lower gap and skeletal legs in both oblique views."),
("IXMYU", "Old-layout new output has the same separate upper condition and lower slab, with thin interrupted leg faces."),
("IXMYU", "Old-layout final still has separated upper and lower levels and weakened supports."),
]
RECORDS[4] = [
("RUPUY", "A bench-like two-level assembly and end supports are present. Blue upper surface is near a gray upright wall and orange front panel; lower gray bands are visibly uneven, so exact integration remains unclear."),
("UUPYU", "End supports remain, but upper blue/orange structure and lower slab are visibly separated in the side cut; interpretation as a coherent shelf/seat assembly is uncertain."),
("UUPYU", "Final keeps the two separated levels and end supports, with no clear visible change in their relation."),
("RUPUY", "New RAW has intact end supports and a bench silhouette; lower horizontal bands and short upright fragments are irregular under the blue upper assembly."),
("UUPYU", "New post retains end walls but upper C and lower deck remain separated in the side cut; central underside fragments persist."),
("UUPYU", "New final keeps the same two-level geometry, short underside fragments, and uncertain upper assembly integration."),
("UUPYU", "Native post shows a blue upper strip above a mostly open lower gray framework; supports remain, but upper/lower joining is visually uncertain."),
("UUPYU", "Native final preserves the upper strip and open lower frame with no clear restoration of missing wall surfaces."),
("RUPUU", "SEAM21000 has substantial end supports and a lower slab beneath the C/orange upper assembly; open space between levels may be structural rather than a defect."),
("RUPUU", "SEAM21000 final retains the same bench silhouette; a tiny green addition is visible at one end but does not settle the ambiguous two-level relation."),
("UUPYU", "Old-layout new post keeps side supports and orange front panel, but short central underside pieces and a large upper/lower separation remain."),
("UUPYU", "Old-layout final has the same separated levels and uncertain upper-to-lower visual integration."),
]
RECORDS[5] = [
("RPPYY", "Blue seat and broad end walls form a bench, but gray layers sit just above the seat and several short ragged pieces hang below the middle."),
("RPPYN", "Post exposes the blue seat and retains both broad end supports; thin dangling central strips remain below the seat."),
("RPPYN", "Final retains the end walls and blue seat but still has thin dangling middle strips."),
("RPPYY", "New RAW has broad end supports, a gray layer close over C, and a triangular short flap under the seat."),
("RPPYN", "New post reveals the blue seat with substantial end supports; a short hanging triangular face remains near the middle."),
("RPPYN", "New final retains the middle hanging flap and large end supports; cuts show no broad new folded layer."),
("RPPUN", "Native post is a recognizable bench with blue seat, broad side walls and thin gray border rails; tiny rim pieces have uncertain connectivity."),
("RPPUN", "Native final keeps the same seat and end-wall structure; no clear detached large fragment is visible."),
("RPPUN", "SEAM21000 post preserves a broad blue seating surface and end walls; small border strips are visually ambiguous."),
("RPPUN", "SEAM21000 final has a coherent bench outline with the same small rim irregularities."),
("RPPYN", "Old-layout new post keeps the blue seat and end supports but has a conspicuous short triangular flap suspended beneath the middle."),
("RPPYN", "Old-layout final retains that underside triangular flap; broad bench structure is otherwise present."),
]
RECORDS[6] = [
("RUPUN", "RAW contains a long deck over two rounded/log-like side supports; the blue vertical condition protrudes through the deck, making its structural role uncertain."),
("UUMYN", "Post turns the rounded support surfaces into open bands and interrupted strips; blue condition still protrudes above the deck."),
("UUMYN", "Final keeps skeletal rounded supports and the same blue protrusion; no substantial support surface restoration is visible."),
("RUPUN", "New RAW preserves a deck over rounded supports, with the blue vertical member extending slightly above the deck; interface role remains ambiguous."),
("UUMYN", "New post removes much of the rounded support shell, leaving multiple separated bands and small interrupted pieces."),
("UUMYN", "New final retains the open-band supports and uncertain blue/deck relation."),
("UUMYN", "Native post has a coherent deck but rounded free supports appear as fragmented open strips; the blue vertical member projects above it."),
("UUMYN", "Native final still has skeletal rounded supports and no obvious improvement in the blue member's integration."),
("UUMYN", "SEAM21000 post leaves open bands in place of substantial rounded support surfaces; deck is retained."),
("UUMYN", "SEAM21000 final preserves the same open support bands and protruding C member."),
("UUMYN", "Old-layout new output retains the deck, but rounded support shells are visibly reduced to interrupted bands."),
("UUMYN", "Old-layout final has no clear recovery of the rounded support surfaces or blue-condition alignment."),
]
RECORDS[7] = [
("RUPNN", "RAW forms a slatted bench with a long back rail and end supports; blue vertical condition protrudes above the seat, so its exact use remains uncertain."),
("RUPUN", "Post retains slatted seat, back rail and end supports. Tiny edge remnants are ambiguous; section endpoints alone are not counted as cracks."),
("RUPUN", "Final retains the same recognizable slatted bench and support ends; blue projection above the seat remains."),
("RUPUN", "New RAW keeps the slatted bench and back rail with end supports; minor rim roughness is present but no definite detached large piece."),
("RUPUN", "New post retains back rail and slats with slightly thin end supports; the central blue vertical member remains visually ambiguous."),
("RUPUN", "New final preserves the same bench structure and blue member; no clear new folded surface appears."),
("RUPUN", "Native post is a slatted bench with a continuous back rail and end supports; a small blue protrusion remains above the seating level."),
("RUPUN", "Native final keeps the same slatted bench; tiny end-edge pieces are too small to classify confidently."),
("RUPUN", "SEAM21000 post has the same slatted seat/back structure and visible end supports; C-use remains uncertain because the blue member projects upward."),
("RUPUN", "SEAM21000 final retains the bench structure with no obvious broad fold."),
("RUPUN", "Old-layout new post gives a slatted bench with back rail and end supports, with the blue vertical member still above the seat."),
("RUPUN", "Old-layout final retains the same overall bench and uncertain central blue-member role."),
]


RECORDS[8] = [
("RPPUY", "Bench seat and two broad end supports are present around the blue back, but a large orange triangular sheet lies in front of the back; side cut shows close offset back layers."),
("RPPUY", "Post keeps end supports and a slatted-looking seat. The large orange sheet still overlays part of the blue back; whether all thin seat strips are intact is uncertain."),
("RPPUY", "Final retains the same orange-over-blue back layering and broad end supports; no clear large detached piece appears."),
("RPPUY", "New RAW uses C at the back and retains seat/end walls, but orange triangles cover almost the whole front of the blue back."),
("RPPUY", "New post has broad supports and sparse seat strips; the orange back layer remains visibly offset from blue in the side section."),
("RPPUY", "New final keeps the large orange back layer, now with a straighter meeting line, and the same sparse seat strips."),
("RPPUN", "Native post has a blue back, broad end supports and thin seat slats, without an obvious extra orange back layer."),
("RPPUN", "Native final retains the same coherent bench silhouette; fine seat-strip completeness remains uncertain."),
("RPPUN", "SEAM21000 post leaves the blue back exposed except small edge pieces; broad end supports remain."),
("RPPUN", "SEAM21000 final keeps the exposed blue back and slatted seat; no obvious broad folded sheet is visible."),
("RPPUY", "Old-layout new post again covers the blue back with large orange triangles; both end supports remain substantial."),
("RPPUY", "Old-layout final keeps the orange-over-blue back layer and sparse seat strips with no major silhouette change."),
]
RECORDS[9] = [
("RPPUY", "RAW forms a bench with rounded/log-like supports and a blue back, but a large orange triangular sheet overlays the back."),
("UPMYY", "Post turns rounded supports into open bands while preserving the seat; orange material still overlays the blue back."),
("UPMYY", "Final retains band-like fragmented supports and the extra orange back layer."),
("RPPUY", "New RAW gives a conventional four-leg bench around C, with an orange triangle covering the upper back area."),
("IPMYY", "New post retains seat/back but most generated leg surfaces vanish into hairline fragments; orange overlay persists."),
("IPMYY", "New final still lacks substantial leg surfaces and retains the orange back overlay."),
("IPMYN", "Native post keeps seat and blue back but generated supports are nearly absent, leaving a floating-looking seat."),
("IPMYN", "Native final has the same unsupported-looking seat/back silhouette and thin leg remnants."),
("IPMYN", "SEAM21000 post retains C as the back, but its generated legs are almost entirely absent."),
("IPMYN", "SEAM21000 final still has no substantial visible supports beneath the seat."),
("IPMYY", "Old-layout new output has near-vanished supports and a large orange sheet across the blue back."),
("IPMYY", "Old-layout final keeps the same missing-support appearance and extra back layer."),
]
RECORDS[10] = [
("RPPNN", "Blue seat is naturally placed between two substantial side supports and below a gray two-rail back. No obvious large fragment or fold is visible in either view."),
("RPPNN", "Post retains blue seat, two side supports and back rails; the cuts show a shallow underside band rather than an obvious displaced seat."),
("RPPNN", "Final retains the same complete bench silhouette and blue seating surface."),
("RPPNN", "New RAW has a coherent blue seat, gray back rails and substantial side supports; orange rim follows the seat edge."),
("RPPNN", "New post retains seat/back/support structure with a narrow orange seat apron; no obvious detached piece appears."),
("RPPNN", "New final keeps the coherent bench and seat use, with no large visible crack or fold."),
("RPPNN", "Native post shows a complete bench with C as seat, gray back rails and two substantial side supports."),
("RPPNN", "Native final retains the same complete seat/back/support arrangement."),
("RPPNN", "SEAM21000 post uses the blue seat with an orange apron and substantial gray supports/back rails."),
("RPPNN", "SEAM21000 final preserves a complete bench silhouette and regular seat position."),
("RPPNN", "Old-layout new post retains C as the seat, with a regular back and substantial supports."),
("RPPNN", "Old-layout final remains visually coherent with no obvious broad folded layer or detached piece."),
]
RECORDS[11] = [
("RPPUN", "RAW resembles a box bench with broad back and side walls; blue C serves as the upper seat over a lower box panel. Minor rim irregularity is ambiguous."),
("RPPUN", "Post retains blue seat and broad back/side walls, with thin foot extensions and no definite large detached fragment."),
("RPPUN", "Final keeps the box-bench structure and blue upper seating surface; small rim gaps are uncertain."),
("IPPYN", "New RAW has a large triangular missing region in the back panel and ragged seat-side junctions; blue seat and side supports remain."),
("IPPYN", "New post still has the large triangular back opening and thin foot extensions; C remains at the seating level."),
("IPPYN", "New final does not fill the triangular back-panel opening; the blue seat and box-like lower structure remain."),
("RPPUN", "Native post gives a box bench with a broad back panel, blue seat and lower storage-like box; small feet are thin."),
("RPPUN", "Native final retains the broad back and side walls around the blue seat, with small ambiguous rim remnants."),
("IXMYY", "SEAM21000 instead places a gray deck above the blue C with orange flaps below; short sparse supports and the two distinct seat levels remain visible in both cuts."),
("IXMYY", "SEAM21000 final keeps the offset gray deck above C and sparse supports; the condition is not visibly adopted as the deck."),
("RPPUN", "Old-layout new output has a full gray back panel and box-like side supports around the blue upper seat, with a narrow orange apron."),
("RPPUN", "Old-layout final keeps the complete back and upper C seat; thin foot and rim details remain visually uncertain."),
]


RECORDS[12] = [
("RPPUY", "C forms a blue seat and two substantial supports within an armchair-like bench; orange triangular fins lie against the blue legs, while gray back and arms remain complete."),
("RPPUY", "Post retains the blue seat/supports and full gray back/arms, with conspicuous orange fins alongside the legs."),
("RPPUY", "Final preserves the same complete silhouette and leg-adjacent orange fins; tiny rim details remain uncertain."),
("RPPUY", "New RAW retains C as seat/supports and adds a complete gray back and arms; orange pointed pieces fold close to the leg junctions."),
("RPPUY", "New post keeps the bench assembly and blue supports, with pointed orange fins and a narrow irregular band under the back."),
("RPPUY", "New final retains the same blue seat and supports; orange leg-adjacent fins are still visible."),
("RPPUY", "Native post uses blue seat/supports and full gray back/arms, but gray pointed triangular flaps extend down beside the blue legs."),
("RPPUY", "Native final keeps those gray leg-adjacent triangular flaps and the complete bench assembly."),
("RPPUY", "SEAM21000 post retains C as seat/supports with orange fins beside legs and a small orange wedge near the seat/back corner."),
("RPPUY", "SEAM21000 final keeps the corner wedge and leg fins, while the gray back and arms remain substantial."),
("RPPUY", "Old-layout new post preserves blue seat/supports; an orange corner wedge and long leg-adjacent fins remain visible."),
("RPPUY", "Old-layout final retains the complete armchair-like silhouette but keeps the same orange folds around the leg joints."),
]
RECORDS[13] = [
("RPPUY", "RAW uses blue C as the seat and two legs; gray back/arms form a recognizable bench, with orange fins beside the legs."),
("RPPUY", "Post retains the full gray back and blue seating/support structure; leg-adjacent orange triangular fins remain."),
("RPPUY", "Final keeps the same recognizable bench and orange fin-like additions beside the blue legs."),
("RPPUY", "New RAW preserves blue seat/supports and a full gray back, with a narrow orange strip at the seat/back join and larger leg fins."),
("RPPUY", "New post retains the complete seat/back/arm assembly but still has triangular orange sheets beside the supports."),
("RPPUY", "New final keeps the complete silhouette and orange leg-adjacent sheets; no clear large detached fragment is visible."),
("RPPUY", "Native post preserves the blue seat and supports; gray pointed flaps descend beside them while the back and arms remain complete."),
("RPPUY", "Native final retains the same blue/gray bench assembly and downward gray corner flaps."),
("RPPUY", "SEAM21000 post has a complete bench structure, with orange triangular faces beside the blue legs and a thin seat apron."),
("RPPUY", "SEAM21000 final keeps the same full back and arm structure with leg-adjacent orange fins."),
("RPPUY", "Old-layout new post has a complete bench silhouette but orange sheets fold close to the support corners."),
("RPPUY", "Old-layout final retains blue seat/support use and the same orange support-corner fins."),
]
RECORDS[14] = [
("RPPUY", "Old RAW builds a broad seat, back and arms above the blue leg frame; large orange triangular sheets lie against the supports."),
("RPPUY", "Old post keeps the full seat/back assembly and blue support frame; orange leg fins persist."),
("RPPUY", "Old final retains the same coherent bench silhouette with pointed orange support-side additions."),
("UUPUY", "New RAW instead places a narrow gray deck with short gray legs well above the blue crossbar; tall orange spikes bridge toward it, making C's structural use uncertain."),
("IUMYY", "New post retains the elevated narrow deck but most of its free leg surfaces become hairline remnants; tall orange spikes flank the lower blue frame."),
("IUMYY", "New final still has the elevated deck, weak free supports and tall orange spikes; its relation to the lower blue frame remains uncertain."),
("RPPUN", "Native post forms a full seat and tall back above the blue rear frame, with substantial gray front legs; small edge irregularities are ambiguous."),
("RPPUN", "Native final retains the full bench with gray front supports and blue rear supports; no obvious large folded panel appears."),
("RPPUY", "SEAM21000 post retains a full seat/back/arm arrangement and blue legs, with orange pointed sheets close to the supports."),
("RPPUY", "SEAM21000 final keeps the same bench structure and leg-adjacent orange fins."),
("IUMYY", "Old-layout new post leaves the narrow gray deck above the blue frame, with hairline free legs and a large diagonal orange sheet spanning the intervening region."),
("IUMYY", "Old-layout final retains the elevated deck, weak generated supports and large diagonal orange sheet; C-use is still ambiguous."),
]
RECORDS[15] = [
("RPPUY", "RAW forms a broad bench with back and arms above the blue support frame; several gray interior diagonals and an orange leg fin are visible."),
("RPPUY", "Post keeps the complete broad seat/back/arms and blue supports, with an orange fin down one leg and small nearby gray slivers."),
("RPPUY", "Final retains that full bench structure and the same leg-side fin; small sliver connectivity is uncertain."),
("RPPUY", "New RAW has a full gray seat/back/arms over the blue support frame, with uneven internal diagonal sheets and orange support fins."),
("RPPUY", "New post retains the full bench outline and blue supports, with narrow gray layers under the seat and orange fins beside legs."),
("RPPUY", "New final preserves the same seat/back/arm structure and close leg-adjacent orange sheets."),
("RPPUY", "Native post retains a complete bench and blue support frame, but gray triangular flaps descend beside a blue leg."),
("RPPUY", "Native final keeps the complete bench silhouette and the same gray leg-side triangular flaps."),
("RPPUY", "SEAM21000 post preserves broad seat/back/arms and blue supports, with orange pointed sheets near the support corners."),
("RPPUY", "SEAM21000 final retains the support-corner sheets and complete bench structure."),
("RPPUY", "Old-layout new post retains a full bench above the blue frame; orange leg fins and a small orange patch on the seat remain visible."),
("RPPUY", "Old-layout final keeps the complete bench silhouette, small seat patch and support-side orange fins."),
]


def write():
    assignment = json.loads((OUT / "visual_review/assignments.json").read_text())["assignments"]["bench_clean_training"]
    manifest = json.loads((OUT / "gallery_manifests/BENCH_CLEAN_FIXED48_MAIN_AUX.json").read_text())
    pages = {(p["request_id"], p["sheet"]): p for p in manifest["pages"]}
    maps = (
        ("overall_shape", dict(R="REASONABLE", I="INCOMPLETE", U="UNCERTAIN")),
        ("C_use", dict(P="PLAUSIBLE", X="REPLACED_OR_IGNORED", U="UNCERTAIN", A="NA")),
        ("support_retention", dict(P="PRESENT", M="MISSING_OR_DAMAGED", U="UNCERTAIN", A="NA")),
        ("visible_fragment_or_gap", dict(Y="YES", N="NO_OBVIOUS", U="UNCERTAIN")),
        ("layering_or_fold", dict(Y="YES", N="NO_OBVIOUS", U="UNCERTAIN")),
    )
    assert sorted(RECORDS) == list(range(16))
    records = []
    for i, req in enumerate(assignment):
        assert len(RECORDS[i]) == 12
        for sheet_index, sheet in enumerate(("MAIN", "AUX")):
            page = pages[(req, sheet)]
            image = OUT / "gallery/BENCH_CLEAN_FIXED48_MAIN_AUX" / (page["page_id"] + ".png")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            for j, cell in enumerate(page["cells"]):
                code, note = RECORDS[i][sheet_index * 6 + j]
                assert len(code) == 5 and note.strip()
                records.append(dict(
                    request_id=req, model=cell["model"], stage=cell["stage"],
                    image_sha256=digest, actually_viewed=True, views_and_sections_checked=True,
                    **{key: table[value] for value, (key, table) in zip(code, maps)},
                    notes=note,
                ))
    target = OUT / "visual_review/review_training.json"
    assert not target.exists()
    target.write_text(json.dumps(dict(reviewer="bench_clean_training", reviewer_kind="AI_PROXY",
                                     blind=False, rows=records), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="COMPLETE", actually_viewed_sheets=32, cells=len(records), path=str(target))))


if __name__ == "__main__":
    write()
