"""Transcribe actual root image inspections; no metric-derived labels."""
from .common import *

def run():
    cells=read(OUT/'inputs/cells.json');rows=[]
    notes={
      ('f495',0,'ORIGINAL'):('REASONABLE','IMPLAUSIBLE','PRESENT','UNCERTAIN','YES',
        'Two three-dimensional views show an upper gray slab and lower shelf between end supports. Fixed front/side sections place the upper gray slab around y0.65-0.8 above blue C around y0.48; C is under a replacement upper slab. Small detached shards cannot be certified from these views.'),
      ('f495',0,'REFINED_NATIVE'):('REASONABLE','PLAUSIBLE','PRESENT','UNCERTAIN','UNCERTAIN',
        'Blue surface is visibly exposed in a low long bench with end supports and lower cross-board; the upper gray slab from ORIGINAL is absent. Front section has blue at y0.48 and lower gray around y0, with an edge/back rim. This is a plausible use, not a certification of a continuous seam or collision-free surface; side section contains short irregular strokes.'),
      ('f495',1,'ORIGINAL'):('REASONABLE','UNCERTAIN','PRESENT','UNCERTAIN','YES',
        'Bench has two end supports and a broad gray seating shell; blue C is largely hidden. Front gray line is just below C, but side section has many strokes spanning both sides of C height, so acceptable thickness versus replacement/overlap is ambiguous.'),
      ('f495',1,'REFINED_NATIVE'):('REASONABLE','PLAUSIBLE','PRESENT','UNCERTAIN','UNCERTAIN',
        'Blue C is visible across the upper strip; free forms end supports and a lower horizontal board, close to the other noise after refinement. Fixed side section remains irregular and cannot certify attachment or absence of gaps.'),
      ('bcb',0,'ORIGINAL'):('REASONABLE','PLAUSIBLE','PRESENT','UNCERTAIN','UNCERTAIN',
        'Blue top spans a coherent rounded-end ring-frame bench. Front section joins the height of blue C to the curved ends; side vertical boundaries extend below it. No alternative source style is required. Exact welding and small fragments are not certified.'),
      ('bcb',0,'REFINED_NATIVE'):('REASONABLE','PLAUSIBLE','PRESENT','UNCERTAIN','UNCERTAIN',
        'Rounded-end frame and blue top relationship remain. A gray inner/lower sheet is more evident in the side section, around y0.38 below C y0.66; this changes thickness/internal organization but does not visibly replace the blue top.'),
      ('bcb',1,'ORIGINAL'):('REASONABLE','PLAUSIBLE','PRESENT','UNCERTAIN','UNCERTAIN',
        'Rounded-end ring-frame bench with exposed blue top, similar broad organization to n0. Fixed sections show curved end supports around the top surface. This is a plausible alternative completion without requiring unique source reconstruction.'),
      ('bcb',1,'REFINED_NATIVE'):('UNCERTAIN','IMPLAUSIBLE','PRESENT','UNCERTAIN','YES',
        'Both views show an L-like gray board and several thin supports, while the original long blue C crosses above/away from that seating organization. Fixed front and side slices leave blue at y0.66 with most gray seating traces around y0.45-0.55 and only partial spatial coverage. Style change alone is not failure; the unsupported/misaligned blue-to-body relationship is the adoption concern.'),
    }
    for c in cells:
        p=next((p for p in ('f495','bcb') if c['q']['parent_id'].startswith(p)),None)
        if p is None:continue
        key=(p,c['q']['noise_index'],c['label']);overall,adoption,support,fragments,layers,note=notes[key]
        image=OUT/'gallery'/c['q']['request_id']/'comparison.png'
        rows.append(dict(cell_id=c['cell_id'],request_id=c['q']['request_id'],parent_id=c['q']['parent_id'],
            noise_index=c['q']['noise_index'],label=c['label'],stage='RAW',reviewer='AI_PROXY_root',blind=False,
            actually_viewed=True,view_tool='functions.exec/tools.view_image detail=original',
            source_image=str(image),source_image_sha256=sha(image),whole_shape=overall,C_adoption=adoption,
            main_support=support,visible_detached_fragments=fragments,layer_gap_or_overlap_concern=layers,observations=note,
            limitation='Two fixed views and two fixed sections; not watertightness, exact topology or human validation. Short section segments alone are not labeled floating fragments.'))
    assert len(rows)==8
    write(OUT/'visual_review/root_f495_bcb.json',dict(schema='actual_nonblind_AI_PROXY_v1',rows=rows,actual_pages_viewed=4,
        prior_labels_reused=False,ratings_inferred_from_metrics=False))
    csvwrite(OUT/'visual_review/root_f495_bcb.csv',rows)
if __name__=='__main__':run()
