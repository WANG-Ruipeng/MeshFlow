"""Transcribe the agent's actual 24 original-PNG inspections; no metric/model calls."""
from pathlib import Path
import csv, sys
sys.path.insert(0, '/mnt/e/MeshFlow')
from r2.pilots.anchor_aware_denoiser_v1.common import OUT, read, sha, csvwrite, write, textfile

G = OUT / 'gallery'
S = G / 'review_shards'

# Each entry was written after seeing all four columns and four views in the
# original PNG. Fields are C use, boundary continuation, whole-object quality,
# thin structure, and a direct observation. They are not inferred from scores.
obs = {}
def record(gen, parent, noise, *values):
    assert len(values) == 5
    obs[(gen, parent, str(noise))] = list(values)

U='R2_CONT_USER'; F='R2_FROZEN'
p33='33ec1e64a02ff718faaadcdc0221eb72'
p53='534b0e6e2c53a8e1a7cdfd3874ed3723'
p86='866e7248434968051cc7750d0afca796'
p8e='86ec9c796a6c1789e792598ed58dd1cc'
pbd='bdd29e651e5f6fb2b079317292bdc5d4'
pf5='f59627a077dcc2566c63b8b48495261a'

record(U,p33,0,
 'PARTIAL/UNCERTAIN: blue C is at seat height but mostly buried beneath the thick gray seat.',
 'UNCERTAIN: small blue front projections remain; no visually clear seam closure.',
 'A recognizable complete four-leg ladder-back chair remains.',
 'Legs and narrow back slats remain visible; no obvious new missing support.',
 'Front and side show only blue seat slices and a projecting tip. All three POST shapes are very similar; no clear anchor-specific C-use improvement.')
record(U,p33,1,
 'UNCERTAIN/OCCLUDED: blue C is hidden inside the box-like seat in solid views and visible in wireframe.',
 'UNCERTAIN: interior occlusion prevents judging an exposed C/free boundary.',
 'A coherent box armchair with low arms and short feet remains.',
 'Short feet and box walls remain; no obvious loss unique to either adaptation.',
 'The exterior is nearly unchanged across POST columns. The image does not establish that the hidden C supplies the visible seat.')
record(U,p53,0,
 'POORLY ALIGNED: the blue back panel extends above and behind the gray generated back.',
 'VISIBLE OFFSET/UNCERTAIN: side view suggests parallel back layers, without clear boundary continuation.',
 'A recognizable cantilever chair with a loop-like base remains.',
 'Base and side-brace triangulation change; no entire base disappearance is visible.',
 'The generated gray back supplies a separate panel. D0 has a more open rectangular lower loop; both adapted columns show a diagonal wedge/brace. C offset persists.')
record(U,p53,1,
 'PARTIAL: the blue panel occupies the back region but extends laterally beyond the gray frame.',
 'UNCERTAIN/OVERHANG: a gray crown and triangular region overlap the blue panel; edge joining is not clear.',
 'A recognizable complete four-leg chair remains.',
 'All four legs and the back outline are retained across POST.',
 'Blue lateral overhang and gray material crossing the back panel remain in all three POST columns; no clear anchor-specific improvement.')
record(U,p86,0,
 'PARTIAL/UNCERTAIN: C is at seat height but the thick gray seat covers most of it.',
 'UNCERTAIN: a blue front band and small blue triangles remain at the lip; no clear join improvement.',
 'A complete four-leg chair with a tall slatted/curved back remains.',
 'Legs and back slats remain; minor gaps differ but no major new loss is apparent.',
 'Blue can also be seen through the back slats. All POST columns preserve the same seat layering and overall silhouette.')
record(U,p86,1,
 'UNCERTAIN/OCCLUDED: C is visible inside the seat in wireframe, but hidden by the box armchair in solid views.',
 'UNCERTAIN: the interior C boundary cannot be judged from the solid views.',
 'A recognizable box armchair with short feet remains.',
 'Short feet and box walls persist across all POST columns.',
 'No clear exterior difference among denoisers; hidden C is not visual evidence of a successful exposed-seat continuation.')
record(U,p8e,0,
 'PARTIAL: blue strips lie along/behind gray back bars, broadly consistent with a back role.',
 'UNCERTAIN: blue edge offsets remain; close projected bars do not establish connectivity.',
 'A complete narrow four-leg chair with an arched/slatted back remains.',
 'Legs and narrow slats are retained.',
 'The blue back is partly hidden by gray structure in all three POST columns; no clear anchor-specific boundary gain.')
record(U,p8e,1,
 'PARTIAL: blue bars sit near the generated back struts, with small side overhang.',
 'UNCERTAIN: duplicated/overlaid gray and blue back structure remains.',
 'A recognizable narrow chair with triangular under-seat bracing remains; the upper cap is angular.',
 'Back struts, legs and under-seat braces remain visible.',
 'All three POST columns are close in appearance, including the angular back cap and side braces.')
record(U,pbd,0,
 'POORLY ALIGNED: the blue back patch is shifted laterally and projects beyond the generated back.',
 'VISIBLE OVERHANG/UNCERTAIN: gray slats form their own back; blue outer edges do not visibly continue into them.',
 'The four-leg chair body remains complete and recognizable.',
 'Legs and back slats persist across POST.',
 'The left/outside blue projection is conspicuous in front and oblique views, and is not visibly corrected by either adaptation.')
record(U,pbd,1,
 'POORLY ALIGNED: side view shows a blue back plane offset forward from the gray back.',
 'VISIBLE GAP/UNCERTAIN: the offset blue plane remains; a shared boundary is not established.',
 'The chair body and four supports remain; the gray back is broad and angular.',
 'See stage-specific observation for the mid-back horizontal projection.',
 'C offset persists across all POST. The adapted models share a local thin-projection change, not an anchor-specific attachment improvement.')
record(U,pf5,0,
 'POORLY ALIGNED: blue C forms an offset frame whose top bar extends above the decorative gray back.',
 'VISIBLE GAP: side view shows separation between the blue vertical plane and gray back; topology remains unverified.',
 'A recognizable four-leg chair with a decorative back opening remains, with irregular foot ends.',
 'All legs remain; small irregular or fragment-like foot regions persist.',
 'The blue frame remains offset in every POST column. Small back-opening/triangulation differences do not resolve the C/free relationship.')
record(U,pf5,1,
 'POORLY ALIGNED: blue top/frame is above and offset from the solid gray back panel.',
 'VISIBLE GAP: side view separates the blue vertical frame from the generated gray back.',
 'A complete chunky four-leg chair remains.',
 'Legs and solid back persist with no clear new support loss.',
 'The three POST shapes are nearly identical and retain the same blue-frame overhang/gap.')

record(F,p33,0,
 'PARTIAL/UNCERTAIN: the gray seat buries most of C; a blue front tip projects in side view.',
 'UNCERTAIN: front blue patch and overhang persist without visible boundary closure.',
 'A complete conventional four-leg ladder-back chair remains.',
 'Legs remain; adapted columns show a slightly more open small gap under an upper back bar.',
 'Upper-back slat gaps differ locally, but the gray-over-blue seat relationship is unchanged and no anchor-specific improvement is apparent.')
record(F,p33,1,
 'PARTIAL/UNCERTAIN: blue C sits under the generated gray seat and projects at the front.',
 'UNCERTAIN/OVERHANG: the blue/gray seat layering remains; no clear shared edge is visible.',
 'See stage-specific support observation; the seat and back themselves remain recognizable.',
 'See stage-specific thin-leg observation.',
 'This is the clearest shared adaptation benefit in this reviewed subset: both adapted outputs retain more of the four-leg silhouette than D0. It does not fix the seat/C layering.')
record(F,p53,0,
 'POORLY ALIGNED: the blue rectangular back stands behind the gray back and extends above it.',
 'VISIBLE OFFSET/UNCERTAIN: side view separates blue and gray back layers.',
 'A box armchair with short feet remains; a small side/front gray tab is visible.',
 'Short feet, box walls and the small protruding tab persist.',
 'Both adapted columns are close to D0. The generated gray back remains a separate panel and the blue offset persists.')
record(F,p53,1,
 'PARTIAL: blue occupies the back region but is oversized relative to the generated gray frame.',
 'UNCERTAIN/OVERLAY: blue extends outside the frame, with a gray crown near its top.',
 'A recognizable four-leg chair with a diagonal under-seat brace remains.',
 'Legs and under-seat brace are preserved in all POST.',
 'Relative to RAW, gray triangular material visible against blue is reduced by all POST columns. D0 and both adaptations remain very similar and do not clearly join the blue edges.')
record(F,p86,0,
 'PARTIAL/UNCERTAIN: the blue horizontal C lies inside/below a gray seat shell.',
 'UNCERTAIN/LAYERED: a blue front/side band remains below the gray cover, without clear seam closure.',
 'A complete four-leg chair with a tall curved/slatted back remains.',
 'Four legs and the narrow back slats are retained.',
 'The three POST bodies and back slats are nearly identical; no clear anchor-specific C-use improvement is visible.')
record(F,p86,1,
 'PARTIAL: blue is exposed at part of the seat perimeter but gray sheets lie above/against it.',
 'UNCERTAIN/LAYERED: exposed blue edge does not demonstrate a continuous joined seam.',
 'A complete four-leg chair with an arched/slatted back remains.',
 'Legs are retained; tiny back gaps differ without an obvious major structural change.',
 'All POST columns preserve the same broadly aligned but layered seat relationship.')
record(F,p8e,0,
 'PARTIAL: blue strips/frame lie behind or along gray back bars and broadly match a back role.',
 'UNCERTAIN: edge offsets remain; projected proximity does not certify connection.',
 'A complete narrow four-leg arched-back chair remains.',
 'Narrow back bars and legs persist with no obvious new large hole or missing support.',
 'The three POST columns are visually close; no clear anchor-specific continuation improvement.')
record(F,p8e,1,
 'PARTIAL: blue frame/strips appear alongside the generated back but are partly covered.',
 'UNCERTAIN: thin outer blue offsets remain around gray back bars.',
 'A recognizable narrow chair with triangular under-seat side bracing remains.',
 'Legs, back bars and side braces remain. Loose-looking bottom protrusions seen in RAW are reduced in all POST.',
 'All POST columns have a similar angular top and side brace. The blue back relationship is not visibly resolved by ANCHOR.')
record(F,pbd,0,
 'POORLY ALIGNED: blue back patch extends substantially beyond the left side of the gray back.',
 'VISIBLE OVERHANG/UNCERTAIN: the outer blue boundary has no evident gray continuation.',
 'A complete four-leg chair with a slatted back remains.',
 'Legs and back slats remain intact in silhouette.',
 'Blue lateral overhang is clear in front/oblique views and remains nearly unchanged across POST.')
record(F,pbd,1,
 'PARTIAL/POORLY ALIGNED: C occupies part of the back but side view shows a separate forward blue plane.',
 'VISIBLE OFFSET/UNCERTAIN: blue and gray back surfaces remain parallel or layered in side view.',
 'A complete four-leg chair remains; POST turns much of the irregular RAW back into a more filled gray panel.',
 'Legs are retained; back openings/slat separation decrease relative to RAW for all POST.',
 'D0 and the two adapted outputs are similar. Smoother/filled gray back appearance does not establish blue-boundary integration.')
record(F,pf5,0,
 'POORLY ALIGNED: blue frame sits offset from the decorative gray back and extends above it.',
 'VISIBLE GAP: side view shows a separate blue vertical plane; no visible seam repair.',
 'The chair is recognizable and has its main body, but foot/base protrusions and broad under-seat strips remain irregular.',
 'Legs survive; spike/skirt-like foot ends and thin diagonal under-seat strips remain in POST.',
 'This is a conspicuous persistent C-gap plus base-irregularity example. All three POST columns retain the same main issues.')
record(F,pf5,1,
 'POORLY ALIGNED: the gray solid back panel hides most of the blue frame, with blue bars above it.',
 'VISIBLE GAP: side view shows separation between blue vertical C and gray back.',
 'A complete chunky four-leg chair with a large solid back remains.',
 'Legs and back panel persist; no clear new large support loss.',
 'All POST columns preserve the same offset frame and solid-panel appearance; no clear anchor-specific C-use gain.')

pending = list(csv.DictReader((G/'review_pending.csv').open(encoding='utf-8-sig',newline='')))
rows=[]
for source in pending:
    if source['generator'] not in (U,F): continue
    row=dict(source)
    noise=row['request_id'].rsplit('__n',1)[1]
    values=list(obs[(row['generator'],row['parent_id'],noise)])
    if row['generator']==F and row['parent_id']==p33 and noise=='1':
        if row['denoiser']=='D0_OFFICIAL':
            values[2]='SUPPORT LOSS: seat/back remain, but substantially fewer long leg silhouettes are visible than in RAW or either adapted output.'
            values[3]='CLEAR VISUAL CONCERN: oblique and opposite-wireframe views show about two long leg paths, versus a four-leg arrangement visible in RAW and both adaptations.'
        else:
            values[2]='The chair has a more complete four-leg silhouette than D0; the blue/gray seat layering remains.'
            values[3]='SHARED ADAPTATION BENEFIT: four long leg paths are visible again relative to D0, although some are thin/open or wire-like. Physical connectivity is unverified.'
    if row['generator']==U and row['parent_id']==pbd and noise=='1':
        values[3]=('A thin horizontal gray projection extends forward from mid-back in D0; four main legs remain.' if row['denoiser']=='D0_OFFICIAL' else 'SHARED ADAPTATION CHANGE: the thin mid-back horizontal projection visible in D0 is absent; the four main legs remain. No anchor-only effect is evident.')
    row.update(review_kind='AI_PROXY',review_status='COMPLETE',blind='False')
    row.update(zip(('C_use','boundary_continuation','overall_quality','thin_structure','observations'), values))
    rows.append(row)
assert len(obs)==24 and len(rows)==72
assert len({(r['denoiser'],r['generator'],r['request_id']) for r in rows})==72

manifest=read(G/'gallery_manifest.json')
figures=sorted({r['figure'] for r in rows})
assert len(figures)==24
receipts={Path(x['png']).name:x for x in manifest['receipts']}
images=[]
for figure in figures:
    receipt=receipts[figure]
    path=Path(receipt['png'])
    actual=sha(path)
    assert actual==receipt['png_sha256']
    images.append(dict(path=str(path),sha256=actual,actually_viewed=True,reviewer_agent='/root/anchor_wrapper',
                       viewing_method='Original PNG bytes displayed with functions.image; four columns and four views inspected.',
                       displayed_resolution='1663x1497 (renderer downsample from original 2000x1800)'))
csvwrite(S/'wrapper.csv',rows)
write(S/'wrapper_images.json',dict(status='COMPLETE',review_kind='AI_PROXY',blind=False,reviewer_agent='/root/anchor_wrapper',
    reviewed_POST_outputs=72,reviewed_images=24,images=images,review_csv_sha256=sha(S/'wrapper.csv'),
    original_png_only=True,numeric_metrics_used_to_assign_visual_labels=False,
    limitations='Non-blinded AI visual proxy, not a human rating. Images cannot establish shared vertex IDs, exact edge connectivity, or intersection validity. UNCERTAIN is retained for occlusion and ambiguous proximity.'))
textfile(S/'wrapper_notes.md', '''# R2_CONT_USER and R2_FROZEN visual inspection

Completed actual inspection of 24 original whole-object PNGs, 72 POST outputs. Each sheet was viewed in front, side, oblique and opposite-wireframe views; all four RAW/D0/NO_ROLE/ANCHOR columns were inspected. Review is non-blinded AI_PROXY, not human scoring. No model/scorer call was made. The receipt hashes the PNGs actually displayed, not JPEG previews.

Across these two generators, most denoiser differences are local. The blue seat is often buried beneath a gray generated seat, while blue back patches/frames often overhang or remain offset from gray back structure. The images do not show a clear ANCHOR-only improvement in C adoption or boundary continuation. Projected proximity never establishes shared-edge connectivity.

R2_CONT_USER: examples 33ec n1 and 866e n1 are box armchairs with C occluded inside the seat; C use therefore remains uncertain. The 534b n0 back panel and f596 frames are visibly offset; bdd n0 has conspicuous lateral overhang. In bdd n1 both adapted denoisers remove a thin forward projection present in D0 at mid-back, while keeping the main legs. The blue/gray back-plane offset remains. This is a shared adaptation change, not an anchor-specific join gain.

R2_FROZEN: the most important positive shared-adaptation case is 33ec seat-medium n1. D0 visibly loses substantial thin-leg structure; the two adapted outputs retain more of the four-leg silhouette. Some supports still look thin/open, so this is visual preservation rather than a connectivity certificate. The blue seat still lies beneath/protrudes from the gray seat. Persistent bad C relationships are visible in 534b n0, bdd n0, and f596 n0/n1. F596 n0 also retains irregular skirt/spike-like foot and under-seat geometry after all three denoisers.

No conclusion that all POST outputs are identical is warranted: the thin-leg and mid-back-projection cases are material contrary evidence to that statement. Conversely, those shared gains do not demonstrate an effect specific to the new anchor-role embedding. Keep the independent quantitative evidence and full per-output observations alongside these images.
''')
print({'status':'COMPLETE','rows':len(rows),'images':len(images),'csv_sha256':sha(S/'wrapper.csv')})
