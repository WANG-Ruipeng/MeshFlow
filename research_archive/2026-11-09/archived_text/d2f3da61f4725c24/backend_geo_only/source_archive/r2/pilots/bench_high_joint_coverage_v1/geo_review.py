"""Literal observations entered after eight actual view_image calls; no scoring."""
from . import geo_only as C
from .geo_report import AXES,STAGES

# Every tuple is an explicit observation, not an inferred score or default label.
OBS=[
('28b9',0,'2026-10-26T15:26:45Z','2026-10-26T15:30:00Z',[
 ('REASONABLE','PLAUSIBLE','PRESENT','YES','NO_OBVIOUS','Two block supports and the five blue-topped seat strips form a coherent bench. Side sections retain strip side/bottom contours. Detail reveals narrow white splits at block corners; no large detached piece or replacement seat is visible.'),
 ('REASONABLE','UNCERTAIN','PRESENT','YES','NO_OBVIOUS','Both blocks remain, but gray strip sidewalls/bottom profiles largely disappear. The side section contains blue top strokes with only tiny gray remnants; enlarged views show a gap between thin blue tops and support blocks. C is not visibly replaced, but integrated adoption is uncertain.'),
 ('REASONABLE','PLAUSIBLE','PRESENT','YES','NO_OBVIOUS','Strip sidewalls and underside profiles survive, and the blocks remain aligned beneath them. Some lower strip contours are open in the side cut and enlarged block corners show narrow splits. The cut alone is not used to certify a hole.'),
 ('REASONABLE','PLAUSIBLE','PRESENT','YES','NO_OBVIOUS','Seat-strip thickness and both support blocks remain. Front and side cuts are close to GEO_BASE; enlarged images still show a narrow block-corner slit. Useful whole-shape preservation, without a crack-free claim.')]),
('28b9',1,'2026-10-26T15:27:10Z','2026-10-26T15:30:25Z',[
 ('REASONABLE','PLAUSIBLE','PRESENT','UNCERTAIN','NO_OBVIOUS','Blue strip tops sit on visible gray strip bodies, carried by two complete block supports. Tiny pale corner lines are visible in detail but their distinction from edge rendering is uncertain; no gross floating patch or alternate seat is visible.'),
 ('REASONABLE','UNCERTAIN','PRESENT','YES','NO_OBVIOUS','Blocks survive but most gray strip thickness disappears; the side section is almost only blue strokes. Enlarged views show the thin top strips above the blocks. Whether this remaining geometry coherently incorporates C is uncertain.'),
 ('REASONABLE','PLAUSIBLE','PRESENT','YES','NO_OBVIOUS','Strip sides and lower profiles remain. A narrow white split is visible on a block corner and the right block has a short diagonal opening in the rear view. Overall bench/support organization survives.'),
 ('REASONABLE','PLAUSIBLE','PRESENT','YES','NO_OBVIOUS','The gray strip bodies and two block supports survive. The rear block looks more continuous than BASE, but the other view still has a small corner split. No large fold or replacement top is apparent.')]),
('abf250',0,'2026-10-26T15:27:25Z','2026-10-26T15:29:30Z',[
 ('REASONABLE','PLAUSIBLE','PRESENT','UNCERTAIN','NO_OBVIOUS','Curved blue-topped slats have gray underside/edge thickness and two intact end supports. A tiny pale seam at a support corner is uncertain; the side cut tracks the curved seat underside instead of an alternate seat.'),
 ('REASONABLE','PLAUSIBLE','MISSING_OR_DAMAGED','YES','NO_OBVIOUS','The bench and blue seat role remain recognizable, but the enlarged views show both end supports split into thin vertical sheets with open lower corners. Gray seat underside thickness also largely vanishes in the side section.'),
 ('REASONABLE','PLAUSIBLE','PRESENT','UNCERTAIN','NO_OBVIOUS','Both end supports retain volumetric appearance and the seat underside follows the blue arch. A fine lower support seam remains in the rear view, so a watertight or crack-free claim is not made.'),
 ('REASONABLE','PLAUSIBLE','PRESENT','NO_OBVIOUS','NO_OBVIOUS','Both end supports appear intact in both views, with a continuous gray seat rim under the blue top. The side section retains the curved underside. No obvious detached piece or large open support slit is visible at this resolution.')]),
('abf250',1,'2026-10-26T15:29:06Z','2026-10-26T15:29:46Z',[
 ('REASONABLE','PLAUSIBLE','PRESENT','UNCERTAIN','NO_OBVIOUS','Two end supports carry the curved/slatted seat. Gray underside contours follow C. Fine white corner seams are visible but their physical openness is uncertain; no gross support loss is apparent.'),
 ('REASONABLE','PLAUSIBLE','MISSING_OR_DAMAGED','YES','NO_OBVIOUS','Both enlarged views show the end-support faces split apart into sheets, especially the left lower corner; the gray seat rim is also largely gone. The blue surface still reads as the seat, but supporting geometry is damaged.'),
 ('REASONABLE','PLAUSIBLE','PRESENT','YES','NO_OBVIOUS','End supports retain their block-like shape and gray seat edges survive. The rear view has a visible elongated white interruption along a seat-side strip and a fine support slit, so local defects remain.'),
 ('REASONABLE','PLAUSIBLE','PRESENT','UNCERTAIN','NO_OBVIOUS','The elongated side-strip opening seen in BASE is largely filled, and both end supports remain. A fine pale seam at the near support remains ambiguous; the curved gray underside still follows the blue input.')]),
]

def main():
    m=C.prepare();rows=[];events=[]
    for prefix,noise,main_time,detail_time,cells in OBS:
        t=next(t for t in m['tasks'] if t['request']['parent_id'].startswith(prefix) and t['request']['noise_index']==noise)
        q=t['request'];rid=q['request_id'];p=C.OUT/'gallery'/(rid+'.png');dp=p.with_name(rid+'__detail.png')
        evid=[]
        for kind,path,utc in [('original_views_and_sections',p,main_time),('higher_resolution_same_camera',dp,detail_time)]:
            event=dict(event_id='actual_view_image_'+prefix+'_n'+str(noise)+'_'+kind,
                tool='view_image',path=str(path),sha256=C.sha(path),viewed_at_utc=utc)
            events.append(event);evid.append(event)
        for stage,values in zip(STAGES,cells):
            axes=dict(zip(AXES,values[:5]))
            assert all(v in AXES[k] for k,v in axes.items())
            rows.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=noise,stage=stage,
                reviewer='backend_runtime',reviewer_kind='AI_PROXY',blind=False,actually_viewed=True,
                views_and_sections_checked=True,image_sha256=C.sha(p),image_receipt_sha256=C.sha(p.with_suffix('.json')),
                detail_image_sha256=C.sha(dp),evidence=evid,**axes,note=values[5]))
    counts={stage:sum(r['stage']==stage and r['overall_shape']=='REASONABLE' and r['C_use']=='PLAUSIBLE' and r['support_retention']=='PRESENT' for r in rows) for stage in STAGES}
    summary=('On these four familiar high-N requests, the strict visual conjunction (reasonable overall shape, plausible C use, support present) is RAW 4/4, original D0 FINAL 0/4, GEO_BASE 4/4 and GEO_FINAL 4/4. '
        'For 28b9 both noises, GEO_ONLY retains the gray seat-strip sidewalls and underside contours that the original D0 chain largely removes. For abf both noises, GEO_ONLY retains the end supports instead of splitting them into narrow sheets. '
        'This is a bounded shape-preservation positive. Small corner splits remain visible on 28b9, and abf n1 has an uncertain fine support seam; BASE abf n1 has a visible side-strip gap reduced by the final CPU cascade. '
        'These observations do not certify legal connectivity, zero intersections, watertightness or unseen-parent generalization; all geometric costs must be reported separately.')
    C.write(C.OUT/'visual_review.json',dict(status='ACTUALLY_REVIEWED',reviewer_kind='AI_PROXY',blind=False,rows=rows,
        view_events=events,actual_tool_views=8,cells=16,strict_visual_joint=counts,summary=summary,
        no_metric_inference=True,no_unique_source_style_ranking=True))
    print(counts)

if __name__=='__main__':main()
