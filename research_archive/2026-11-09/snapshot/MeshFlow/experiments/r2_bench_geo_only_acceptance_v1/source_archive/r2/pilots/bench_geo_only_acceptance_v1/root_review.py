"""Transcribe root's actual ten-page visual review; no image inference in code."""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_geo_only_acceptance_v1'
AXES=['overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']
SH={'R':'REASONABLE','I':'INCOMPLETE','U':'UNCERTAIN'}
CU={'P':'PLAUSIBLE','X':'REPLACED_OR_IGNORED','U':'UNCERTAIN'}
SU={'P':'PRESENT','M':'MISSING_OR_DAMAGED','U':'UNCERTAIN'}
YN={'Y':'YES','N':'NO_OBVIOUS','U':'UNCERTAIN'}
# Each five-character string is a manual observation after view_image,
# in fixed RAW/BASE/FINAL/GEO_ONLY/Native FINAL column order.
REV={
 ('35948e',0):(['UUPNU']*5,[
  'Tall box-like walls surround C at an interior height; support walls remain, but seat use and overall bench interpretation are uncertain.'
 ]*5),
 ('35948e',1):(['RPPNN']*5,[
  'Both surface views show a plausible blue seat, continuous back and two broad end supports; section endpoints alone do not establish a crack.'
 ]*5),
 ('4cd23a',0):(['RPPUU']*5,[
  'Blue top can be a box-bench seat with broad supporting sides. There are internal horizontal surfaces; whether the small rim gaps or folds are defects is uncertain.'
 ]*5),
 ('4cd23a',1):(['UXPNY','UXPYY','UXPYY','UXPNY','RPPYY'],[
  'Gray tabletop/seat sits well above blue C; broad side supports remain but C is not the upper used surface.',
  'Gray surface above C remains and large triangular openings appear in top/side panels.',
  'Final still has the elevated gray surface and large top/side gaps.',
  'Preserves broad panels, but the substantial gray surface above C still replaces its apparent seat role.',
  'Blue seat has plausible box/end supports; visible crossed/creased side panels and openings remain.'
 ]),
 ('2d1e5e',0):(['RXPYY']*5,[
  'Body, back and end supports remain bench-like, but sections show a continuous gray seat above the blue C with a gap; C is obscured in surface views.'
 ]*5),
 ('2d1e5e',1):(['RXPYY']*5,[
  'Both views retain bench body and supports; the blue input lies below a parallel gray seat layer rather than being the visible seating surface.'
 ]*5),
 ('459f90',0):(['RPPYY','IPMYN','IPMYN','RPPYY','UPPYU'],[
  'Blue seat, gray back and loop end legs form a plausible bench, though a lower gray layer and gaps remain.',
  'D0 backend leaves the seat/back rails with legs largely missing; visible detached thin segments.',
  'Final retains the incomplete thin seat/back structure and missing supports.',
  'Both loop end supports and fuller back are retained, with a blue upper seat; lower layering/gaps persist.',
  'End panels could support the blue seat but the remaining thin back/side geometry is incomplete or ambiguous.'
 ]),
 ('459f90',1):(['RXPUY','IXMYY','IXMYY','RXPUY','UXPYY'],[
  'Four legs and a gray table-like upper surface survive; that surface is visibly above C, so input seat is replaced.',
  'Thin floating upper strips remain above C and supports mostly disappear.',
  'Final preserves the floating upper strips and support loss.',
  'Keeps the four-legged body but also keeps its displaced gray surface above the blue input.',
  'Broad side panels survive, but gray upper slab is separated from C and panels appear folded/gapped.'
 ]),
 ('a8e987',0):(['RPPNN','IPMYN','IPMYN','RPPNN','RPPUU'],[
  'Blue slatted seat is supported by four visible slim legs; plausible intact bench.',
  'Seat remains but most leg surfaces disappear, leaving very thin traces.',
  'Final still has severe leg loss rather than a complete supported bench.',
  'Four slim legs and seat perimeter survive; no obvious large missing structure in either surface view.',
  'Blue seat with broad end supports is a plausible alternative style, while lower internal layers and gaps are ambiguous.'
 ]),
 ('a8e987',1):(['RPPNN','IPMYN','IPMYN','RPPNN','RPPNN'],[
  'Four slim legs visibly support the supplied slatted seat.',
  'Legs shrink to near-line remnants under the intact blue seat.',
  'Final does not restore the lost leg surfaces.',
  'Retains four legs, seat perimeter and plausible overall structure.',
  'Box-like alternative with broad end and lower support panels reasonably uses the blue seat.'
 ])
}

def main():
    m=json.loads((OUT/'gallery/review_manifest.json').read_text(encoding='utf-8-sig'))
    pages=[]
    for (prefix,noise),(codes,notes) in REV.items():
        rows=[r for r in m['cells_to_review'] if r['parent_id'].startswith(prefix) and r['noise_index']==noise]
        rows.sort(key=lambda r:r['column']);assert len(rows)==5
        p=OUT/'gallery'/rows[0]['request_id']/'comparison.png'
        digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==rows[0]['image_sha256']
        cells=[]
        for row,code,note in zip(rows,codes,notes):
            vals=[lookup[c] for lookup,c in zip([SH,CU,SU,YN,YN],code)]
            cells.append(dict(label=row['label'],stage=row['stage'],**dict(zip(AXES,vals)),note=note))
        pages.append(dict(request_id=rows[0]['request_id'],image_path=str(p),image_sha256=digest,
            viewed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),actually_viewed=True,
            view_record='Root actually opened page via tools.view_image before transcription; timestamp is transcription after viewing',cells=cells))
    result=dict(reviewer='/root',reviewer_kind='AI_PROXY',blind=False,
        limitations='Non-blind proxy review; finite page resolution, no human certification. Root saw group score summary before these views; axes based on images, not metric thresholds.',pages=pages)
    dest=OUT/'visual_review/review_root.json';dest.parent.mkdir(exist_ok=True)
    with dest.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('Recorded 10 actual pages / 50 cells')

if __name__=='__main__':main()
