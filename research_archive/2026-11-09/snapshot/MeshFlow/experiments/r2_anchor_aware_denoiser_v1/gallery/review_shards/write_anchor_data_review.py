"""Persist actual non-blind PNG review performed by /root/anchor_data.

This file encodes observations made after displaying all 20 assigned original
PNG sheets. It neither renders images nor infers observations from scores.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, '/mnt/e/MeshFlow/r2/pilots/anchor_aware_denoiser_v1')
from common import OUT, read, sha, csvwrite, write, textfile, checked

G = OUT / 'gallery'
S = G / 'review_shards'

# Each entry: C use, boundary continuation, whole shape, thin structures,
# shared observation, then separately observed per-denoiser comparison.
OBS = {
('33ec', 'n0'): (
    'C remains visibly below the gray seat rather than replacing its seating surface.',
    'VISIBLE_OFFSET: local wireframe shows a lower independent C outline; continuous boundary adoption is not demonstrated.',
    'Recognizable chair remains, with back and seat preserved; support legs become visibly thinner after POST.',
    'Several formerly thick RAW legs are reduced to extremely thin strips or nearly line-like supports.',
    'Same lower blue patch and separate gray seat persist in all three POST columns. No visible anchor-specific correction.',
    ['D0: strong thinning is visible on multiple support legs in solid and wireframe views.',
     'NO_ROLE: some legs retain more visible strip width than D0, but at least one support remains almost line-like.',
     'ANCHOR: very similar to NO_ROLE; at least one almost line-like support remains and the lower C patch is unchanged.']),
('33ec', 'n1'): (
    'C lies within the seat region but appears as a slanted interior patch mostly occluded by gray seating geometry.',
    'UNCERTAIN: a separate blue interior outline remains in wireframe; visible placement does not establish edge connection.',
    'A complete armchair-like silhouette and four substantial legs remain.',
    'Legs and arm rails remain visibly substantial; no whole support disappearance is apparent.',
    'Denoising smooths edges slightly, while the seat/C relationship and complete chair organization remain nearly unchanged.',
    ['D0: slanted blue seat patch remains; no clearly resolved continuation.',
     'NO_ROLE: no clear C-use or attachment improvement over D0 in these views.',
     'ANCHOR: no clear C-use or attachment improvement over NO_ROLE in these views.']),
('534b', 'n0'): (
    'C is an upright back panel, but its top protrudes above and behind a separate gray back layer.',
    'VISIBLE_LAYERING: blue and gray back outlines remain offset; local views do not show a resolved common border.',
    'Boxy armchair-like body persists with short supports and multiple bulky layers.',
    'Short supports remain; no newly missing whole support is apparent.',
    'The pre-existing layered back/body relation survives all three POST variants with almost identical silhouettes.',
    ['D0: minor surface cleanup; protruding blue back panel remains.',
     'NO_ROLE: layered back and arm blocks remain, visually close to D0.',
     'ANCHOR: layered back and arm blocks remain, visually close to NO_ROLE.']),
('534b', 'n1'): (
    'C occupies the high back region, crossed or occluded by a large slanted gray sheet and a gray top loop.',
    'VISIBLE_MISMATCH: rectangular blue boundary is offset from the narrow gray back structure; continuous adoption is not shown.',
    'Tall chair-like form remains, but the back organization is irregular with a diagonal sheet and isolated-looking top loop.',
    'Top loop and short legs remain; no gross new structural disappearance is apparent.',
    'The gray diagonal across the blue back and narrow lower continuation are present in all three POST results.',
    ['D0: top-loop triangulation becomes slightly simpler; back mismatch persists.',
     'NO_ROLE: top loop and diagonal sheet persist with no clear connection repair.',
     'ANCHOR: visually almost the same back mismatch as NO_ROLE.']),
('866e', 'n0'): (
    'C is at a plausible seat height and inside the armchair perimeter, but nearby gray seat sheets still partly cover it.',
    'UNCERTAIN: local and wireframe views show a separate blue outline within the seat; adjacency alone does not certify joining.',
    'Complete armchair-like body, rounded back and four legs remain.',
    'Thin openings in the rounded back rim and slender rails remain; legs are not visibly lost.',
    'This remains a plausible seat placement case, not visual proof of boundary attachment; small openings remain in the curved back.',
    ['D0: cleanup slightly reshapes rim openings without resolving the C/gray-seat relationship.',
     'NO_ROLE: comparable plausible overall chair; no unmistakable boundary repair relative to D0.',
     'ANCHOR: nearly indistinguishable seat/C relation from NO_ROLE at these views.']),
('866e', 'n1'): (
    'C sits inside the boxy seat interior but is almost entirely occluded by gray surfaces in solid views.',
    'UNCERTAIN: wireframe exposes an inner blue rectangle among several gray seat layers; there is no clear visual evidence of fused borders.',
    'Broad low-backed armchair or box-like chair silhouette remains with short feet.',
    'Short feet and narrow rear rails remain; no obvious new whole-structure loss.',
    'All POST columns keep the layered seat interior and nearly the same boxy outer shape.',
    ['D0: substantial triangulation cleanup with the same hidden C region.',
     'NO_ROLE: some additional triangles remain, without a clear change in C use.',
     'ANCHOR: nearly the same layered interior and shape as NO_ROLE.']),
('86ec', 'n0'): (
    'Blue C slats occupy the back region and roughly track the generated back frame, with strips protruding along its sides.',
    'UNCERTAIN: back slats and gray frame are adjacent, but several blue outlines remain separate; topological continuation cannot be read from the sheet.',
    'A narrow, complete chair with an arched open back and four supports remains.',
    'Back slats and supports remain; no gross new disappearance.',
    'Overall back organization is similar across POST variants, with only small changes to surrounding triangles.',
    ['D0: blue slats still offset along the gray back frame.',
     'NO_ROLE: no clear visual improvement in blue/gray boundary continuation.',
     'ANCHOR: no clear visual improvement over NO_ROLE.']),
('86ec', 'n1'): (
    'Blue C back slats sit near the back frame but protrude above and beside the gray cap.',
    'VISIBLE_OFFSET/UNCERTAIN: side/top offsets remain, and common edge joining is not established.',
    'Complete narrow chair persists, with large triangular free sheets spanning part of the lower support region.',
    'Supports remain; broad triangular sheets between legs persist and their legality cannot be judged from the image alone.',
    'The large lower triangular sheets and the top/side blue offsets survive all three POST stages.',
    ['D0: slight simplification without gross silhouette change.',
     'NO_ROLE: similar lower support sheets and back offset to D0.',
     'ANCHOR: similar lower support sheets and back offset to NO_ROLE.']),
('bdd2', 'n0'): (
    'C lies at the back but extends substantially to one side and above the narrower generated gray back.',
    'VISIBLE_MISMATCH: lateral overhang remains; no continuous adoption of the full supplied region is visible.',
    'Complete narrow chair with arched/slatted back remains, with an asymmetric blue extension.',
    'Four supports and narrow back slats remain; no gross new support loss.',
    'The unadopted lateral extent of C is conspicuous and remains after all three denoisers.',
    ['D0: gray shape slightly smoother but lateral C overhang remains.',
     'NO_ROLE: no clear removal of the C/back mismatch.',
     'ANCHOR: no clear removal of the C/back mismatch relative to NO_ROLE.']),
('bdd2', 'n1'): (
    'C forms a separate upright plane in front of the main gray back and protrudes above it.',
    'VISIBLE_GAP: side view retains a clear blue/gray back separation; connection at its lower edge remains uncertain.',
    'Complete armchair-like form remains, with a duplicated or offset-looking back layer.',
    'Arms and legs remain, without a new obvious collapse.',
    'The side-view gap between supplied upright geometry and the gray back remains in each POST column.',
    ['D0: some top-edge smoothing; distinct blue plane persists.',
     'NO_ROLE: distinct blue plane persists with similar overall body.',
     'ANCHOR: no visible resolution of the distinct blue plane relative to NO_ROLE.']),
('f596', 'n0'): (
    'The blue back frame is near the generated back but protrudes above it and remains offset in side view.',
    'VISIBLE_OFFSET: thin blue frame and the gray back are not visibly joined continuously at the rim.',
    'Decorative chair-like form remains with a circular back opening and irregular, faceted legs.',
    'Thin rim, circular opening and kinked/split-looking lower leg areas persist; no whole leg newly disappears.',
    'All POST variants preserve the opening and overall structure but leave offset C and irregular lower supports.',
    ['D0: nearby triangles are cleaner, but blue frame offset remains.',
     'NO_ROLE: comparable opening and supports; no obvious new boundary repair.',
     'ANCHOR: comparable opening and supports; no obvious new boundary repair relative to NO_ROLE.']),
('f596', 'n1'): (
    'C frame is close to the tall gray back but sticks above its top and remains separated in side view.',
    'VISIBLE_OFFSET/UNCERTAIN: blue tips and side gap persist; apparent bottom proximity is not proof of connection.',
    'Complete high-backed chair remains; simpler gray back and supports retain their large-scale form.',
    'Thin C frame stays intact and all large supports remain; substantial free-face simplification does not visibly erase a whole leg.',
    'All three POST variants simplify the mesh strongly while retaining almost the same silhouette and offset frame.',
    ['D0: strongly simplified triangulation, same offset blue frame.',
     'NO_ROLE: several extra triangles remain, no clear C-adoption gain.',
     'ANCHOR: comparable simplification and no clear C-adoption gain over NO_ROLE.']),
}

manifest = read(G / 'gallery_manifest.json')
with (G / 'review_pending.csv').open(newline='', encoding='utf-8-sig') as f:
    rows = [r for r in csv.DictReader(f) if r['generator'] == 'R2_USER_MIX']
assert len(rows) == 36
stages = ['D0_OFFICIAL', 'D_CONT_NO_ROLE', 'D_ANCHOR']
for row in rows:
    prefix = row['parent_id'][:4]
    noise = row['request_id'].rsplit('__', 1)[-1]
    c_use, boundary, overall, thin, common, specific = OBS[(prefix, noise)]
    row.update(review_kind='AI_PROXY', review_status='COMPLETE', blind=False,
        C_use=c_use, boundary_continuation=boundary, overall_quality=overall,
        thin_structure=thin, observations=common + ' ' + specific[stages.index(row['denoiser'])])
receipts = []
for rec in manifest['receipts']:
    if rec['identity']['generator'] != 'R2_USER_MIX':
        continue
    assert sha(rec['png']) == rec['png_sha256']
    receipts.append(dict(path=rec['png'], sha256=rec['png_sha256'], actually_viewed=True,
        reviewer_agent='/root/anchor_data', view_kind='LOCAL' if rec['identity']['bounds']['local_plot_crop'] else 'WHOLE',
        display_method='Original PNG bytes via approved read-only exec_command and functions.image; no re-render or JPEG substitution'))
assert len(receipts) == 20
assert sum(r['view_kind'] == 'WHOLE' for r in receipts) == 12
assert sum(r['view_kind'] == 'LOCAL' for r in receipts) == 8
csvwrite(S / 'anchor_data.csv', rows)
write(S / 'anchor_data_images.json', dict(status='COMPLETE', review_kind='AI_PROXY', blind=False,
    reviewed_POST_outputs=36, reviewed_images=20, images=receipts,
    gallery_manifest_sha256=sha(G / 'gallery_manifest.json'),
    visual_only=True, numerical_scores_used_for_observations=False,
    failed_displays_not_counted=True,
    transport_note='866e n0 local PNG exceeded the tool single-output byte cap; original bytes were read in three base64-aligned chunks and assembled for a successful actual display. Failed displays were not counted.',
    limitations=['Non-blind AI proxy, not human rating.', 'Visible alignment cannot certify vertex-ID attachment.', 'No source GT resemblance ranking.', 'No quantitative win rate inferred from these free-text observations.']))
textfile(S / 'anchor_data_notes.md', '''# USER_MIX actual visual review

Reviewer: `/root/anchor_data`; non-blind **AI_PROXY**, not human evaluation. All 12 whole-object original PNGs and all 8 predeclared local original PNGs were actually displayed. The 36 POST rows cover D0_OFFICIAL, D_CONT_NO_ROLE and D_ANCHOR for every assigned request; RAW is the reference. Observations were written without looking at numerical scores or source GT.

The main result of this visual review is the similarity of the three POST variants at the level of C use and overall shape. I do not see an unmistakable anchor-specific adoption of the supplied boundary in these 12 requests. This does not exclude smaller numerical differences.

- **33ec seat n0:** the blue patch remains below a separate gray seat in all three POST outputs. All three thin the RAW support legs markedly, with at least one nearly line-like support. The adapted variants retain more visible width on some legs than D0 but do not cure the condition placement.
- **33ec seat n1:** a complete armchair remains; C is a tilted interior seat patch. The layered/occluded C relationship is similar across variants.
- **534b back n0/n1:** n0 retains a protruding blue panel behind a bulky layered gray back. n1 retains a large slanted gray sheet across the blue panel and a top loop. No variant visibly resolves these back relationships.
- **866e seat n0/n1:** n0 has plausible seat placement, yet local/wire views show a separate blue outline and nearby gray sheets; attachment remains uncertain. n1 largely buries the blue plane inside a layered boxy seat. The three variants retain similar arrangements.
- **86ec back n0/n1:** narrow recognizable chairs remain, with blue strips along and above a gray back frame. n1 retains broad triangular free sheets at the lower supports. Neither their legality nor topological joining is inferred from the image.
- **bdd2 back n0/n1:** n0 retains conspicuous lateral/upper C overhang. n1 retains a distinct upright blue plane separated from the main gray back in side view. The mismatch persists through all variants.
- **f596 back n0/n1:** both retain blue frame tips above and offset from the gray back. n0 retains the circular back opening and irregular faceted lower supports; n1 is simplified strongly while maintaining the large-scale chair. No anchor-specific boundary repair is apparent.

The review intentionally separates a plausible chair silhouette, plausible semantic placement of C, and actual boundary connection. Neither hard C preservation nor gray/blue proximity is counted as certified connection. Fine intersection legality and near-zero gaps require the saved numerical/topological results. Small triangle-count differences are not treated as visual quality wins.
''')
print({'status': 'COMPLETE', 'review_rows': len(rows), 'actually_viewed_original_PNGs': len(receipts),
    'csv_sha256': sha(S / 'anchor_data.csv'), 'receipt_sha256': sha(S / 'anchor_data_images.json')})
