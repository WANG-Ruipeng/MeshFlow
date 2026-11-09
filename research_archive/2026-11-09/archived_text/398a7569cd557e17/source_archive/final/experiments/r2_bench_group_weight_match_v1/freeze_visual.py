"""Root-owned visual criteria; existing anchors actually opened before M sampling."""
from datetime import datetime, timezone
from r2.pilots.bench_group_weight_match_v1.common import ROOT, OUT, sha, write_new

old = ROOT/'experiments/r2_bench_high_joint_coverage_v1'
anchors = [
 ('gallery/RAW__9452149917cb2e27f4fb4dee5181bee__COMPLETE_SEAT_UPPER_SURFACE_GROUP__HIGH_JOINT_V1__n1.png',
  'W illustrates plausible use of blue slats as the seat, with visible end frames and supports. B has a separate gray upper seat above blue C. Thin intentionally coherent frames are not missing support merely because they are thin.',
  'root_actual_view_20261027_945_RAW'),
 ('gallery/CLEAN_FINAL_POST__77aa33479c2ec92f610c2a68437007d6__COMPLETE_THICK_SEAT__HIGH_JOINT_V1__n0.png',
  'W retains fragments of two pedestal supports but their middle surfaces are visibly torn/open. This is MISSING_OR_DAMAGED support, not PRESENT merely because legs remain. B box support is an alternative permissible completion, not a source-style ground truth.',
  'root_actual_view_20261027_77_FINAL'),
 ('backend_geo_only/gallery/28b9f2908f0cbb179c21c139c0613648__ALL_SEAT_BOARD_UPPER_SURFACES__EXPOSURE_PREP_V1__n0__detail.png',
  'Original D0 FINAL has two substantial support blocks yet a visible separation below the seat. Distinguish support retention from the fragment/gap axis; existence of a gap alone does not mean the block surfaces vanished. RAW and GEO views demonstrate intact supporting surfaces.',
  'root_actual_view_20261027_28_DETAIL'),
 ('backend_geo_only/gallery/abf250a01f526137a035c800c58595d9__ALL_SEAT_BOARD_UPPER_SURFACES__EXPOSURE_PREP_V1__n0__detail.png',
  'Original D0 FINAL loses much of the gray longitudinal frame and leaves split/open pedestal sheets, unlike coherent RAW/GEO support surfaces. Blue seat visibility does not certify a complete supported bench. This anchor is for definitions only, not a new B panel rating.',
  'root_actual_view_20261027_ABF_DETAIL'),
]

def main():
    rows=[]
    for rel,note,event in anchors:
        p=old/rel
        rows.append(dict(path=str(p),sha256=sha(p),actually_viewed=True,reviewer='root',
                         reviewer_kind='AI_PROXY',blind=False,view_event_id=event,note=note))
    d=dict(schema='BENCH_VISUAL_CRITERIA_V2',status='FROZEN',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),before_new_M_sampling=True,
        main_comparison=['W','M'],B='Quantitative reference only; no new complete B visual count',
        historical_ratings='Read-only; do not pool counts across interpretation versions',
        anchors=rows,
        definitions={
            'overall_shape': 'REASONABLE: recognizable structurally complete bench with main seating body and support organization; INCOMPLETE: a major body/support is absent, torn apart or incompatible; UNCERTAIN when views do not settle it. Does not require source style, watertightness or one connected component.',
            'C_use': 'PLAUSIBLE: C participates in a plausible functional relation to the generated body; REPLACED_OR_IGNORED: a separate gray substitute displaces its apparent role or C remains an unrelated floating insert; UNCERTAIN if ambiguous. Exact shared edges are not required by this visual axis. Mere blue visibility is insufficient.',
            'support_retention': 'PRESENT requires substantial coherent intended support surfaces or coherent intentional rods/frames. MISSING_OR_DAMAGED if only torn shells, open narrow remnants or displaced scraps remain. A small local opening does not automatically mean major support loss; count such openings separately. UNCERTAIN if occlusion or framing prevents judgment.',
            'visible_fragment_or_gap': 'YES for visible loose scraps, abnormal tears or separation; NO_OBVIOUS only if neither view nor section provides clear evidence. Section endpoints alone do not prove holes: cross-check full views. Designed slat spacing and ordinary separate assembly parts are not automatically defects.',
            'layering_or_fold': 'YES for visibly redundant layers, obvious interpenetrating/folded sheets or a competing seat; NO_OBVIOUS when none is visible; UNCERTAIN for occluded ambiguous thin layers. Do not infer this label from numerical conflict scores.'},
        rules=['Read both oblique views and both fixed sections for every W/M page.',
               'Record five axes independently and one concrete visual note for each cell.',
               'Allow uncertainty; do not manufacture agreement with historic labels or metrics.',
               'No unique-source reconstruction grading, no forced watertight or single-piece requirement.',
               'No metric-derived labels; all results remain nonblind AI_PROXY, not independent human assessment.',
               'These anchors clarify interpretation before sampling; they do not select checkpoints or requests.'])
    write_new(OUT/'visual_review/v2_definition.json',d)
    print('FROZEN V2: four existing anchors actually viewed; zero new model calls')

if __name__=='__main__':main()
