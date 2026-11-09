# SEAM A/B/C actual visual-review aggregation

28 fixed requests × 4 models × RAW/POST = 224 unchanged review records from 56 pages. Each page has four ACTUALLY_VIEWED column records. This program verifies identities and hashes; it does not perform a second image review.

[Full CSV](visual_review.csv) · [JSON and page ledger](visual_review_summary.json) · [Validation receipt](visual_review_receipt.json)

Every table cell is **YES / NO / UNCERTAIN**. Defect-field YES and positive-property YES have different meanings. Uncertain labels remain uncertain; columns are not added into a quality score.

## ALL

| Stage | Model | n | C_adoption | whole_body_complete | support_missing |
| --- | --- | ---: | --- | --- | --- |
| RAW | LONG | 28 | 25 / 0 / 3 | 27 / 1 / 0 | 1 / 27 / 0 |
| RAW | A_COUPLE | 28 | 24 / 0 / 4 | 26 / 1 / 1 | 1 / 27 / 0 |
| RAW | B_POOL47 | 28 | 22 / 1 / 5 | 27 / 0 / 1 | 0 / 27 / 1 |
| RAW | C_SHARED_TIP | 28 | 24 / 0 / 4 | 26 / 1 / 1 | 1 / 26 / 1 |
| POST | LONG | 28 | 25 / 0 / 3 | 27 / 1 / 0 | 1 / 27 / 0 |
| POST | A_COUPLE | 28 | 24 / 0 / 4 | 26 / 1 / 1 | 1 / 27 / 0 |
| POST | B_POOL47 | 28 | 22 / 1 / 5 | 27 / 0 / 1 | 0 / 27 / 1 |
| POST | C_SHARED_TIP | 28 | 24 / 0 / 4 | 26 / 1 / 1 | 1 / 26 / 1 |

| Stage | Model | n | fragments_or_folded_layers | visible_cracks | visible_join |
| --- | --- | ---: | --- | --- | --- |
| RAW | LONG | 28 | 6 / 11 / 11 | 8 / 5 / 15 | 20 / 0 / 8 |
| RAW | A_COUPLE | 28 | 4 / 14 / 10 | 8 / 5 / 15 | 20 / 0 / 8 |
| RAW | B_POOL47 | 28 | 5 / 12 / 11 | 6 / 5 / 17 | 17 / 3 / 8 |
| RAW | C_SHARED_TIP | 28 | 12 / 6 / 10 | 20 / 3 / 5 | 16 / 0 / 12 |
| POST | LONG | 28 | 3 / 13 / 12 | 5 / 11 / 12 | 20 / 0 / 8 |
| POST | A_COUPLE | 28 | 3 / 14 / 11 | 7 / 12 / 9 | 20 / 0 / 8 |
| POST | B_POOL47 | 28 | 4 / 14 / 10 | 5 / 7 / 16 | 17 / 3 / 8 |
| POST | C_SHARED_TIP | 28 | 10 / 7 / 11 | 17 / 2 / 9 | 16 / 0 / 12 |

## TRAIN

| Stage | Model | n | C_adoption | whole_body_complete | support_missing |
| --- | --- | ---: | --- | --- | --- |
| RAW | LONG | 16 | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| RAW | A_COUPLE | 16 | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| RAW | B_POOL47 | 16 | 15 / 1 / 0 | 15 / 0 / 1 | 0 / 15 / 1 |
| RAW | C_SHARED_TIP | 16 | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| POST | LONG | 16 | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| POST | A_COUPLE | 16 | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| POST | B_POOL47 | 16 | 15 / 1 / 0 | 15 / 0 / 1 | 0 / 15 / 1 |
| POST | C_SHARED_TIP | 16 | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |

| Stage | Model | n | fragments_or_folded_layers | visible_cracks | visible_join |
| --- | --- | ---: | --- | --- | --- |
| RAW | LONG | 16 | 3 / 9 / 4 | 2 / 4 / 10 | 15 / 0 / 1 |
| RAW | A_COUPLE | 16 | 0 / 12 / 4 | 2 / 3 / 11 | 15 / 0 / 1 |
| RAW | B_POOL47 | 16 | 2 / 8 / 6 | 2 / 2 / 12 | 13 / 1 / 2 |
| RAW | C_SHARED_TIP | 16 | 7 / 4 / 5 | 13 / 0 / 3 | 12 / 0 / 4 |
| POST | LONG | 16 | 1 / 11 / 4 | 2 / 10 / 4 | 15 / 0 / 1 |
| POST | A_COUPLE | 16 | 0 / 12 / 4 | 2 / 10 / 4 | 15 / 0 / 1 |
| POST | B_POOL47 | 16 | 2 / 9 / 5 | 2 / 4 / 10 | 13 / 1 / 2 |
| POST | C_SHARED_TIP | 16 | 6 / 5 / 5 | 12 / 0 / 4 | 12 / 0 / 4 |

## DEV

| Stage | Model | n | C_adoption | whole_body_complete | support_missing |
| --- | --- | ---: | --- | --- | --- |
| RAW | LONG | 12 | 9 / 0 / 3 | 11 / 1 / 0 | 1 / 11 / 0 |
| RAW | A_COUPLE | 12 | 8 / 0 / 4 | 10 / 1 / 1 | 1 / 11 / 0 |
| RAW | B_POOL47 | 12 | 7 / 0 / 5 | 12 / 0 / 0 | 0 / 12 / 0 |
| RAW | C_SHARED_TIP | 12 | 8 / 0 / 4 | 10 / 1 / 1 | 1 / 10 / 1 |
| POST | LONG | 12 | 9 / 0 / 3 | 11 / 1 / 0 | 1 / 11 / 0 |
| POST | A_COUPLE | 12 | 8 / 0 / 4 | 10 / 1 / 1 | 1 / 11 / 0 |
| POST | B_POOL47 | 12 | 7 / 0 / 5 | 12 / 0 / 0 | 0 / 12 / 0 |
| POST | C_SHARED_TIP | 12 | 8 / 0 / 4 | 10 / 1 / 1 | 1 / 10 / 1 |

| Stage | Model | n | fragments_or_folded_layers | visible_cracks | visible_join |
| --- | --- | ---: | --- | --- | --- |
| RAW | LONG | 12 | 3 / 2 / 7 | 6 / 1 / 5 | 5 / 0 / 7 |
| RAW | A_COUPLE | 12 | 4 / 2 / 6 | 6 / 2 / 4 | 5 / 0 / 7 |
| RAW | B_POOL47 | 12 | 3 / 4 / 5 | 4 / 3 / 5 | 4 / 2 / 6 |
| RAW | C_SHARED_TIP | 12 | 5 / 2 / 5 | 7 / 3 / 2 | 4 / 0 / 8 |
| POST | LONG | 12 | 2 / 2 / 8 | 3 / 1 / 8 | 5 / 0 / 7 |
| POST | A_COUPLE | 12 | 3 / 2 / 7 | 5 / 2 / 5 | 5 / 0 / 7 |
| POST | B_POOL47 | 12 | 2 / 5 / 5 | 3 / 3 / 6 | 4 / 2 / 6 |
| POST | C_SHARED_TIP | 12 | 4 / 2 / 6 | 5 / 2 / 5 | 4 / 0 / 8 |

## Index of explicit adverse observations

The following preserves original defect YES or positive-property NO labels. All uncertain records remain in the full CSV and JSON. This is an index, not an additional evaluation.

| Request / split | Stage / model | Original labels | Original visible observations |
| --- | --- | --- | --- |
| 73b369b9_n0 / TRAIN | RAW / LONG | visible_cracks=YES | Blue C is the seat of a rectangular-back cantilever chair. A narrow gap is visible at the lower back and a white strip at the front seat rim. |
| 73b369b9_n0 / TRAIN | RAW / A_COUPLE | visible_cracks=YES | Rectangular back, front support and broad base remain present. The lower-back gap and narrow front rim separation remain visible. |
| 73b369b9_n0 / TRAIN | RAW / B_POOL47 | visible_cracks=YES | Blue seat and rectangular back appear continuous at the rear at this scale. A narrow white strip remains at the front seat/support boundary. |
| 73b369b9_n0 / TRAIN | RAW / C_SHARED_TIP | visible_cracks=YES | Rounded faceted back and complete cantilever body. Blue C forms the seat; rear contact appears continuous, with a thin front rim separation. |
| 73b369b9_n0 / TRAIN | POST / LONG | visible_cracks=YES | Rectangular-back cantilever body is present. The blue seat still has a visible lower-back gap and narrow white front strip. |
| 73b369b9_n0 / TRAIN | POST / A_COUPLE | visible_cracks=YES | Complete rectangular-back body; a small triangular part of the lower back is more filled, but the seat/back and front-rim gaps remain. |
| 73b369b9_n0 / TRAIN | POST / B_POOL47 | visible_cracks=YES | Rear blue-seat/back junction looks continuous. A narrow white front seat strip remains; base and front support are present. |
| 73b369b9_n0 / TRAIN | POST / C_SHARED_TIP | visible_cracks=YES | Rounded back, broad base and front support remain complete. Blue C is the seat with apparent rear contact and a thin front rim gap. |
| 73b369b9_n1 / TRAIN | RAW / LONG | visible_cracks=YES | Blue C forms the seat and meets the rectangular back visually. A thin front seat strip and a small white notch at the lower base corner are visible. |
| 73b369b9_n1 / TRAIN | RAW / A_COUPLE | visible_cracks=YES | Rectangular-back chair retains its support and base. Rear contact looks continuous; a narrow front seat gap remains visible. |
| 73b369b9_n1 / TRAIN | RAW / B_POOL47 | C_adoption=NO; fragments_or_folded_layers=YES; visible_join=NO | A raised gray seat is visibly above blue C, which remains as a lower strip. The frame chair body is complete, but the two seat levels are distinct in the central cuts. |
| 73b369b9_n1 / TRAIN | RAW / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | Blue C is the seat of a rounded-back cantilever chair. A long thin diagonal sliver crosses the base plate; a narrow front seat gap remains. |
| 73b369b9_n1 / TRAIN | POST / LONG | visible_cracks=YES | Rectangular-back body remains complete. Rear seat contact appears continuous; the front white strip and small lower-base notch remain visible. |
| 73b369b9_n1 / TRAIN | POST / A_COUPLE | visible_cracks=YES | Blue seat, back and cantilever base are present. The narrow front seat strip is still visible. |
| 73b369b9_n1 / TRAIN | POST / B_POOL47 | C_adoption=NO; fragments_or_folded_layers=YES; visible_join=NO | The gray seating surface remains above blue C after processing. C is visible below the frame chair seat, with two distinct levels in the cuts. |
| 73b369b9_n1 / TRAIN | POST / C_SHARED_TIP | visible_cracks=YES | Rounded-back chair remains complete. The conspicuous diagonal base sliver from RAW is absent; a thin front seat gap remains. |
| 73b369b9_n2 / TRAIN | RAW / B_POOL47 | visible_cracks=YES | Blue seat and support are present, but a conspicuous white wedge interrupts the upper-left back panel. |
| 73b369b9_n2 / TRAIN | RAW / C_SHARED_TIP | visible_cracks=YES | Blue C is the seat. A narrow white slit is visible in the upper back and a thin separation appears at the front seat rim. |
| 73b369b9_n2 / TRAIN | POST / B_POOL47 | visible_cracks=YES | The upper-left white wedge in the back remains visible. The blue seat and supporting cantilever body remain present. |
| 73b369b9_n2 / TRAIN | POST / C_SHARED_TIP | visible_cracks=YES | The upper-back slit is less apparent than in RAW, but a narrow white front seat gap remains. Body and blue-seat use remain plausible. |
| 3f741759_n0 / TRAIN | RAW / C_SHARED_TIP | visible_cracks=YES | 主体仍完整，第一视角左侧座裙有白色开口，另一视角前缘也有细缝；底部剖线明显缺段。 |
| 3f741759_n0 / TRAIN | POST / C_SHARED_TIP | visible_cracks=YES | 侧裙左上缘的白色开口仍可见；座面、靠背和四腿没有整块缺失。 |
| 3f741759_n1 / TRAIN | RAW / LONG | fragments_or_folded_layers=YES | 座面和主体齐全；一条内侧腿末端有尖薄突出片，背条略不齐。 |
| 3f741759_n1 / TRAIN | RAW / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | 靠背弧板有三角白洞，内部竖条倾斜并呈尖薄形；蓝色座面和腿仍存在。 |
| 3f741759_n1 / TRAIN | POST / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | 弧形靠背的三角白洞仍在，内部有尖斜竖条；主体、座面和四腿仍可辨。 |
| 3f741759_n2 / TRAIN | RAW / LONG | fragments_or_folded_layers=YES | 蓝C作为座面，主体完整；靠背中间数根窄条扭斜，侧剖底部有短碎线。 |
| 3f741759_n2 / TRAIN | RAW / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | 弧形靠背出现多条白色裂口，内部竖条尖斜、错位；蓝座面及支撑保留。 |
| 3f741759_n2 / TRAIN | POST / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | 靠背右上弧板有明显三角缺口，中间竖条仍斜尖；蓝C和腿保持完整轮廓。 |
| 3f741759_n3 / TRAIN | RAW / C_SHARED_TIP | visible_cracks=YES | 蓝座面、背条和四腿仍在，圆弧背板呈多条放射状白缝与开口。 |
| 3f741759_n3 / TRAIN | POST / C_SHARED_TIP | visible_cracks=YES | 弧形靠背上多条窄白裂口仍可辨，C及整体椅身保留。 |
| 85b16941_n1 / TRAIN | RAW / LONG | fragments_or_folded_layers=YES | A long gray sheet hangs from the rear of the seat/back almost to leg-bottom height. The seat, rectangular back and four legs otherwise remain. |
| 85b16941_n1 / TRAIN | RAW / B_POOL47 | fragments_or_folded_layers=YES | Blue C occupies the seat region, but the body has a large triangular slab dropping below its front/right and a broad L-shaped foot or shelf on the opposite side. |
| 85b16941_n1 / TRAIN | POST / LONG | fragments_or_folded_layers=YES | The long rear hanging sheet survives processing alongside the four legs and rectangular back. |
| 85b16941_n1 / TRAIN | POST / B_POOL47 | fragments_or_folded_layers=YES | The large downward triangular panel and broad low foot/shelf persist. The blue seat remains nominally usable but support organization is irregular. |
| 85b16941_n2 / TRAIN | RAW / C_SHARED_TIP | visible_cracks=YES | A large white opening crosses the left-side apron; a narrow lower strip remains below it, and the front apron drops to a triangular point. Back, seat and four legs remain. |
| 85b16941_n2 / TRAIN | POST / C_SHARED_TIP | visible_cracks=YES | The large left-side apron opening and residual lower strip survive processing. The front triangular apron point is still visible. |
| 85b16941_n3 / TRAIN | RAW / C_SHARED_TIP | visible_cracks=YES | Seat, back and four legs remain, but the left-side apron contains an open band with narrow strips; the front edge has a pointed triangular panel. |
| 8c9b2023_n0 / TRAIN | RAW / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | The blue seat and main ladder-back/legs remain recognizable. White triangular openings interrupt the rear apron, and a small missing patch is visible at the front-left seat frame. The lower contour under C is mostly absent in the cuts. |
| 8c9b2023_n0 / TRAIN | POST / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | The chair keeps its back and four legs, but small triangular holes remain in the rear apron and the front-left seat frame is chipped. Central cuts still lack most of the underside contour. |
| 8c9b2023_n1 / TRAIN | RAW / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | The chair body remains recognizable, but jagged white gaps interrupt the rear apron and the front-left frame has a missing patch. Most of the grey seat underside is absent from both central cuts. |
| 8c9b2023_n1 / TRAIN | POST / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | The blue seat is still plausible, but the rear apron contains hanging triangular pieces and open gaps; the front-left frame is chipped. The grey lower seat contour remains mostly missing in the cuts. |
| 8c9b2023_n2 / TRAIN | RAW / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | Blue C remains the seat, but the rear apron breaks into separated-looking strips and white triangular gaps. The front-left apron/leg junction is chipped; most of the lower seat contour is absent in the cuts. |
| 8c9b2023_n2 / TRAIN | POST / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | The main chair remains, but an obvious missing central portion interrupts the rear apron. The front-left frame is chipped and central cuts lack most of the lower seat contour. |
| 8c9b2023_n3 / TRAIN | RAW / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | The blue seat and chair silhouette persist. Several white holes and thin hanging strips break up the rear apron; the front-left seat frame is chipped. The lower grey seat contour is missing from the front cut. |
| 8c9b2023_n3 / TRAIN | POST / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | A recognizable chair remains around the blue seat. Rear-apron openings and triangular remnants persist, and the front-left frame stays chipped. Both cuts have little grey underside contour. |
| 2d08a64e_n0 / DEV | RAW / LONG | visible_cracks=YES | Blue C lies at the seating surface of a complete sled-frame chair. Gray surfaces curve well below C; the left underside has broken white slits and several close lower traces. |
| 2d08a64e_n0 / DEV | RAW / A_COUPLE | visible_cracks=YES | The blue top seat, back and sled frame are present. Several lower gray traces and small side-underseat openings remain visible. |
| 2d08a64e_n0 / DEV | RAW / C_SHARED_TIP | visible_cracks=YES | Blue C forms the top seat of a complete frame chair. The lower gray shell has multiple nearby traces and a small open slit on the left side. |
| 2d08a64e_n1 / DEV | RAW / LONG | visible_cracks=YES | Blue C is the seat above two broad panel supports and an upright back. A white triangular gap is visible at the left seat-side junction. |
| 2d08a64e_n1 / DEV | RAW / C_SHARED_TIP | visible_cracks=YES | Blue seat and rounded-underseat shell sit above a cantilever base. A large white triangular opening is visible at the left seat-side transition. |
| 2d08a64e_n1 / DEV | POST / LONG | visible_cracks=YES | Panel-supported chair is complete with blue C as its seat. The left seat-side triangular opening is still visible. |
| 2d08a64e_n1 / DEV | POST / C_SHARED_TIP | visible_cracks=YES | Cantilever body remains recognizable and complete. A large left-side triangular gap and a narrow seat/front-support separation persist. |
| 2d08a64e_n2 / DEV | RAW / LONG | whole_body_complete=NO; support_missing=YES; fragments_or_folded_layers=YES; visible_cracks=YES | Blue seat and upper back are recognizable, but the lower supports break into many horizontal strips, slivers and open holes. |
| 2d08a64e_n2 / DEV | RAW / A_COUPLE | whole_body_complete=NO; support_missing=YES; fragments_or_folded_layers=YES; visible_cracks=YES | Blue C is still the seat. The front and side supports are heavily split into overlapping-looking strips with large missing intervals. |
| 2d08a64e_n2 / DEV | RAW / C_SHARED_TIP | whole_body_complete=NO; support_missing=YES; fragments_or_folded_layers=YES; visible_cracks=YES | Upper chair and blue seat remain present, but the lower panel supports have broad horizontal breaks and loose-looking strips. |
| 2d08a64e_n2 / DEV | POST / LONG | whole_body_complete=NO; support_missing=YES; fragments_or_folded_layers=YES; visible_cracks=YES | Processing fills some strips, but lower supports still show multiple horizontal tears and irregular plate fragments beneath the blue seat. |
| 2d08a64e_n2 / DEV | POST / A_COUPLE | whole_body_complete=NO; support_missing=YES; fragments_or_folded_layers=YES; visible_cracks=YES | Some lower panels become more filled, but side openings and torn strip-like supports remain obvious. Blue C is still the seat. |
| 2d08a64e_n2 / DEV | POST / C_SHARED_TIP | whole_body_complete=NO; support_missing=YES; fragments_or_folded_layers=YES; visible_cracks=YES | Horizontal openings and torn lower support strips remain after processing. The back and blue seat are recognizable but the supporting body is visibly damaged. |
| 2d08a64e_n3 / DEV | RAW / LONG | visible_cracks=YES | Blue seat and complete cantilever body are present. A diagonal white slit crosses the lower back panel; a small front seat-side opening is also visible. |
| 2d08a64e_n3 / DEV | RAW / A_COUPLE | visible_cracks=YES | Rounded-back cantilever chair uses blue C as its seat. The diagonal back line is much narrower, while a small front/side seat opening remains. |
| 2d08a64e_n3 / DEV | RAW / B_POOL47 | visible_cracks=YES | Blue seat and complete body remain. A thin white diagonal opening is visible higher in the back, rather than in the same lower-back location as LONG. |
| 2d08a64e_n3 / DEV | RAW / C_SHARED_TIP | visible_cracks=YES | Blue C remains the seat of a complete cantilever body. A broad diagonal white slit crosses the lower back and a white triangular opening appears near its lower corner. |
| 2d08a64e_n3 / DEV | POST / LONG | visible_cracks=YES | Lower-back diagonal slit is largely closed, but a narrow front seat/support separation remains. The blue-seat cantilever body is complete. |
| 2d08a64e_n3 / DEV | POST / A_COUPLE | visible_cracks=YES | Back panel appears closed at this scale. Small front/side seat gaps remain in a complete cantilever body. |
| 2d08a64e_n3 / DEV | POST / B_POOL47 | visible_cracks=YES | Back panel is largely filled. A conspicuous white strip remains at the front seat/support transition; the body is otherwise complete. |
| 2d08a64e_n3 / DEV | POST / C_SHARED_TIP | visible_cracks=YES | A broad diagonal white opening remains across the lower back after processing. Blue seat, front support and base are still present. |
| 610b1621_n0 / DEV | RAW / A_COUPLE | fragments_or_folded_layers=YES; visible_cracks=YES | The blue plate extends over a narrow stacked base. Large white wedges separate it from folded-looking side/front layers, although a back and small feet remain. |
| 610b1621_n0 / DEV | RAW / C_SHARED_TIP | fragments_or_folded_layers=YES | A polygonal back rises above the blue plate; below it are large angular stacked panels and an offset flat base. The load-bearing organization is not clear. |
| 610b1621_n0 / DEV | POST / A_COUPLE | fragments_or_folded_layers=YES; visible_cracks=YES | The blue plate still overhangs the stacked base with a conspicuous white wedge beside/below it. Large folded-looking panels remain. |
| 610b1621_n0 / DEV | POST / C_SHARED_TIP | fragments_or_folded_layers=YES | The angular pedestal/base arrangement and stacked panels persist below the blue plate, with the polygonal back above. |
| 608b46b0_n0 / DEV | RAW / LONG | fragments_or_folded_layers=YES; visible_cracks=YES | Blue C occupies the seat of a broad solid-back chair with four visible legs. The rounded back crown is irregular with small white holes and folded-looking facets; narrow grey streaks cross portions of the blue seat. |
| 608b46b0_n0 / DEV | RAW / A_COUPLE | fragments_or_folded_layers=YES; visible_cracks=YES | C forms the seat in a similar solid-back chair. The crown is rough and pitted, and a front leg has a long narrow white slit; a thin grey streak lies across the seat. |
| 608b46b0_n0 / DEV | RAW / B_POOL47 | fragments_or_folded_layers=YES; visible_cracks=YES; visible_join=NO | The chair is recognizable, but a thin grey rectangular rim lies within the blue seat footprint. A gap separates the lower back from C, also visible in the side cut; the crown and one leg contain openings. |
| 608b46b0_n0 / DEV | RAW / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | Blue C remains the seat of a solid-back four-leg chair. The crown is irregular and a front leg has a large vertical slit; a smaller inner support also has an open-looking face. |
| 608b46b0_n0 / DEV | POST / A_COUPLE | visible_cracks=YES | C is the blue seat of a complete solid-back chair. A small hole remains near the back crown; legs are substantially more regular than RAW. |
| 608b46b0_n0 / DEV | POST / B_POOL47 | fragments_or_folded_layers=YES; visible_cracks=YES; visible_join=NO | A solid-back chair remains, but a grey inner rim crosses the blue seat footprint and a clear lower-back/C gap persists. One leg still has a long open slit. |
| 608b46b0_n0 / DEV | POST / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | C is retained as a plausible blue seat with a solid back and four legs. The front-leg surface still contains a long white slit even after POST; the crown has small irregular patches. |
| 608b46b0_n1 / DEV | RAW / B_POOL47 | fragments_or_folded_layers=YES; visible_cracks=YES | C is a plausible seat within a shorter chair with a deeper apron. A large irregular opening with folded-looking edges appears near the top of the back; the central front cut leaves a gap above the apron. |
| 608b46b0_n1 / DEV | RAW / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | Blue C forms the seat in a shorter, deep-apron chair. A large ragged opening and triangular flaps interrupt the upper back; the seat rim reaches C at the rendered scale. |
| 608b46b0_n3 / DEV | RAW / LONG | fragments_or_folded_layers=YES; visible_cracks=YES | The chair has a blue seat, solid back and four legs. A long grey strip crosses the front part of C, and the back crown has several small irregular holes; the lower seat section is fragmented. |
| 608b46b0_n3 / DEV | RAW / A_COUPLE | fragments_or_folded_layers=YES; visible_cracks=YES | The broad-back chair retains the same seat/support arrangement. A long grey strip still crosses C and the crown is rough, with small holes and fine protrusions. |
| 608b46b0_n3 / DEV | RAW / B_POOL47 | fragments_or_folded_layers=YES; visible_cracks=YES; visible_join=NO | The chair is recognizable, but a grey rectangular outline lies over the blue seat. A broad white gap separates the lower back from C, and a front leg has a long open slit. |
| 608b46b0_n3 / DEV | RAW / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | Blue C remains a plausible seat and the long grey front strip is much reduced. However the lower back has a visible gap above C and diagonal cracks/slits appear in the back surface and rear frame. |
| 608b46b0_n3 / DEV | POST / LONG | fragments_or_folded_layers=YES | A complete-looking solid-back chair retains the blue seat. A long grey strip remains across its front portion; the crown is smoother but still has small irregularities. |
| 608b46b0_n3 / DEV | POST / A_COUPLE | fragments_or_folded_layers=YES; visible_cracks=YES | The chair still has a blue seat crossed by the long grey strip. A small open crown patch and projecting top facet remain after POST. |
| 608b46b0_n3 / DEV | POST / B_POOL47 | fragments_or_folded_layers=YES; visible_cracks=YES; visible_join=NO | The grey rim over C and a wide seat/back gap remain. A long opening in a front-leg surface is visible, although the overall four-support chair silhouette remains. |
| 608b46b0_n3 / DEV | POST / C_SHARED_TIP | fragments_or_folded_layers=YES; visible_cracks=YES | The blue seat is less crossed by grey strips than LONG. A gap persists below the back and small triangular openings remain near the back corners/crown, so local continuity is not clean. |

## Original reviewer notes

- [3f7417590f1bcfded5c89ecb06d1099b / TRAIN](reviews/3f.md): AI_PROXY_ROOT; 32 original rows.
- [73b369b9808bc2d0f6ba522682be8978 / TRAIN](reviews/73.md): AI_PROXY_SEAM_PORTABLE_CORE; 32 original rows.
- [85b16941984902f8facfa12c7d71c89f / TRAIN](reviews/85.md): AI_PROXY_SEAM_EVIDENCE; 32 original rows.
- [8c9b2023ecdeb7b049d998127ea6baf9 / TRAIN](reviews/8c.md): AI_PROXY_SEAM_CODE_AUDIT; 32 original rows.
- [2d08a64e4a257e007135fc51795b4038 / DEV](reviews/2d.md): AI_PROXY_SEAM_PORTABLE_CORE; 32 original rows.
- [610b1621d1997ee1daf3c002be658861 / DEV](reviews/610.md): AI_PROXY_SEAM_EVIDENCE; 32 original rows.
- [608b46b0f8f9eba81b1a986f753a489b / DEV](reviews/608.md): AI_PROXY_SEAM_CODE_AUDIT; 32 original rows.

## All 56 viewed pages

| Request / split | Stage | Original reviewer | Model columns | Page |
| --- | --- | --- | ---: | --- |
| 73b369b9_n0 / TRAIN | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n0_RAW.png](gallery/73b369b9_n0_RAW.png) |
| 73b369b9_n0 / TRAIN | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n0_POST.png](gallery/73b369b9_n0_POST.png) |
| 73b369b9_n1 / TRAIN | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n1_RAW.png](gallery/73b369b9_n1_RAW.png) |
| 73b369b9_n1 / TRAIN | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n1_POST.png](gallery/73b369b9_n1_POST.png) |
| 73b369b9_n2 / TRAIN | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n2_RAW.png](gallery/73b369b9_n2_RAW.png) |
| 73b369b9_n2 / TRAIN | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n2_POST.png](gallery/73b369b9_n2_POST.png) |
| 73b369b9_n3 / TRAIN | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n3_RAW.png](gallery/73b369b9_n3_RAW.png) |
| 73b369b9_n3 / TRAIN | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n3_POST.png](gallery/73b369b9_n3_POST.png) |
| 3f741759_n0 / TRAIN | RAW | AI_PROXY_ROOT | 4 | [3f741759_n0_RAW.png](gallery/3f741759_n0_RAW.png) |
| 3f741759_n0 / TRAIN | POST | AI_PROXY_ROOT | 4 | [3f741759_n0_POST.png](gallery/3f741759_n0_POST.png) |
| 3f741759_n1 / TRAIN | RAW | AI_PROXY_ROOT | 4 | [3f741759_n1_RAW.png](gallery/3f741759_n1_RAW.png) |
| 3f741759_n1 / TRAIN | POST | AI_PROXY_ROOT | 4 | [3f741759_n1_POST.png](gallery/3f741759_n1_POST.png) |
| 3f741759_n2 / TRAIN | RAW | AI_PROXY_ROOT | 4 | [3f741759_n2_RAW.png](gallery/3f741759_n2_RAW.png) |
| 3f741759_n2 / TRAIN | POST | AI_PROXY_ROOT | 4 | [3f741759_n2_POST.png](gallery/3f741759_n2_POST.png) |
| 3f741759_n3 / TRAIN | RAW | AI_PROXY_ROOT | 4 | [3f741759_n3_RAW.png](gallery/3f741759_n3_RAW.png) |
| 3f741759_n3 / TRAIN | POST | AI_PROXY_ROOT | 4 | [3f741759_n3_POST.png](gallery/3f741759_n3_POST.png) |
| 85b16941_n0 / TRAIN | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n0_RAW.png](gallery/85b16941_n0_RAW.png) |
| 85b16941_n0 / TRAIN | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n0_POST.png](gallery/85b16941_n0_POST.png) |
| 85b16941_n1 / TRAIN | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n1_RAW.png](gallery/85b16941_n1_RAW.png) |
| 85b16941_n1 / TRAIN | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n1_POST.png](gallery/85b16941_n1_POST.png) |
| 85b16941_n2 / TRAIN | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n2_RAW.png](gallery/85b16941_n2_RAW.png) |
| 85b16941_n2 / TRAIN | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n2_POST.png](gallery/85b16941_n2_POST.png) |
| 85b16941_n3 / TRAIN | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n3_RAW.png](gallery/85b16941_n3_RAW.png) |
| 85b16941_n3 / TRAIN | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n3_POST.png](gallery/85b16941_n3_POST.png) |
| 8c9b2023_n0 / TRAIN | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n0_RAW.png](gallery/8c9b2023_n0_RAW.png) |
| 8c9b2023_n0 / TRAIN | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n0_POST.png](gallery/8c9b2023_n0_POST.png) |
| 8c9b2023_n1 / TRAIN | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n1_RAW.png](gallery/8c9b2023_n1_RAW.png) |
| 8c9b2023_n1 / TRAIN | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n1_POST.png](gallery/8c9b2023_n1_POST.png) |
| 8c9b2023_n2 / TRAIN | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n2_RAW.png](gallery/8c9b2023_n2_RAW.png) |
| 8c9b2023_n2 / TRAIN | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n2_POST.png](gallery/8c9b2023_n2_POST.png) |
| 8c9b2023_n3 / TRAIN | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n3_RAW.png](gallery/8c9b2023_n3_RAW.png) |
| 8c9b2023_n3 / TRAIN | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n3_POST.png](gallery/8c9b2023_n3_POST.png) |
| 2d08a64e_n0 / DEV | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [2d08a64e_n0_RAW.png](gallery/2d08a64e_n0_RAW.png) |
| 2d08a64e_n0 / DEV | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [2d08a64e_n0_POST.png](gallery/2d08a64e_n0_POST.png) |
| 2d08a64e_n1 / DEV | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [2d08a64e_n1_RAW.png](gallery/2d08a64e_n1_RAW.png) |
| 2d08a64e_n1 / DEV | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [2d08a64e_n1_POST.png](gallery/2d08a64e_n1_POST.png) |
| 2d08a64e_n2 / DEV | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [2d08a64e_n2_RAW.png](gallery/2d08a64e_n2_RAW.png) |
| 2d08a64e_n2 / DEV | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [2d08a64e_n2_POST.png](gallery/2d08a64e_n2_POST.png) |
| 2d08a64e_n3 / DEV | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [2d08a64e_n3_RAW.png](gallery/2d08a64e_n3_RAW.png) |
| 2d08a64e_n3 / DEV | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [2d08a64e_n3_POST.png](gallery/2d08a64e_n3_POST.png) |
| 610b1621_n0 / DEV | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [610b1621_n0_RAW.png](gallery/610b1621_n0_RAW.png) |
| 610b1621_n0 / DEV | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [610b1621_n0_POST.png](gallery/610b1621_n0_POST.png) |
| 610b1621_n1 / DEV | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [610b1621_n1_RAW.png](gallery/610b1621_n1_RAW.png) |
| 610b1621_n1 / DEV | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [610b1621_n1_POST.png](gallery/610b1621_n1_POST.png) |
| 610b1621_n2 / DEV | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [610b1621_n2_RAW.png](gallery/610b1621_n2_RAW.png) |
| 610b1621_n2 / DEV | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [610b1621_n2_POST.png](gallery/610b1621_n2_POST.png) |
| 610b1621_n3 / DEV | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [610b1621_n3_RAW.png](gallery/610b1621_n3_RAW.png) |
| 610b1621_n3 / DEV | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [610b1621_n3_POST.png](gallery/610b1621_n3_POST.png) |
| 608b46b0_n0 / DEV | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [608b46b0_n0_RAW.png](gallery/608b46b0_n0_RAW.png) |
| 608b46b0_n0 / DEV | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [608b46b0_n0_POST.png](gallery/608b46b0_n0_POST.png) |
| 608b46b0_n1 / DEV | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [608b46b0_n1_RAW.png](gallery/608b46b0_n1_RAW.png) |
| 608b46b0_n1 / DEV | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [608b46b0_n1_POST.png](gallery/608b46b0_n1_POST.png) |
| 608b46b0_n2 / DEV | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [608b46b0_n2_RAW.png](gallery/608b46b0_n2_RAW.png) |
| 608b46b0_n2 / DEV | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [608b46b0_n2_POST.png](gallery/608b46b0_n2_POST.png) |
| 608b46b0_n3 / DEV | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [608b46b0_n3_RAW.png](gallery/608b46b0_n3_RAW.png) |
| 608b46b0_n3 / DEV | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [608b46b0_n3_POST.png](gallery/608b46b0_n3_POST.png) |

These are nonblind AI_PROXY observations. Source-chair style is not a unique target. Visually continuous surfaces do not establish legal topology or absence of FF/I. Prior CPU/historical reviews are excluded.
