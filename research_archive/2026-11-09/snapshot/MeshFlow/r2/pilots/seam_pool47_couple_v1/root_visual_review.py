"""Transcribe root's actual 2d image observations, not metric-derived labels."""
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_pool47_couple_v1'
LABELS=('B_LONG_LAYOUT','PLAIN3000','COUPLE3000')
# Four flags: body, C use, visible fragments/folded sheets, visible cracks.
# Each page was viewed by root with view_image before transcribing these notes.
OBS={
 (0,'RAW'):[
 ('YES','UNCERTAIN','NO','UNCERTAIN','Recognizable framed chair; blue upper seat is visible, but the broad curved grey layer below makes its integration versus an additional seat uncertain. Thin section gaps do not certify a tear.'),
 ('YES','UNCERTAIN','UNCERTAIN','YES','Framed body remains; blue upper seat and lower grey seat shell persist. Irregular openings and strips along the lower/left back are visible.'),
 ('YES','UNCERTAIN','YES','YES','Body remains; blue seat is above a grey shell. Rear-facing view shows jagged folded triangular strips below the back and gaps around that structure.')],
 (0,'POST'):[
 ('YES','UNCERTAIN','NO','UNCERTAIN','D0 has straightened frame and lower-seat traces; blue upper seat and a separate lower curved shell remain visible. Their intended relationship is uncertain.'),
 ('YES','UNCERTAIN','UNCERTAIN','YES','Body and blue upper seat remain. Narrow irregular back openings persist after smoothing; section still has a lower seat shell.'),
 ('YES','UNCERTAIN','YES','YES','Frame and blue seat remain, but the rear view retains a fan of angular sheets under the back and a visible opening above the rear rail.')],
 (1,'RAW'):[
 ('YES','YES','NO','UNCERTAIN','Simple panel chair: blue surface clearly occupies the seat, back and panel supports are present. Section shows spacing below the seat; exact joining is not visible.'),
 ('YES','YES','NO','UNCERTAIN','Panel chair and blue seat retained, with some rough edges around the support transition. Central section cannot distinguish an intentional hollow underside from a join gap.'),
 ('YES','YES','NO','UNCERTAIN','Panel-backed chair with recognizable blue seat and support panels; no large detached sheet is apparent. Underside spacing remains uncertain.')],
 (1,'POST'):[
 ('YES','YES','NO','UNCERTAIN','Panel chair becomes straighter; blue seat remains the visible seating surface. An underside separation is visible in section but does not establish a specific tear.'),
 ('YES','YES','NO','UNCERTAIN','Straight back and panel supports remain; blue seat is clear. No large loose fragment; section retains a gap between seat and lower shell.'),
 ('YES','YES','NO','UNCERTAIN','Straight panel chair and blue seat retained. No obvious loose sheet in the rendered views; lower-shell separation persists in the section.')],
 (2,'RAW'):[
 ('YES','UNCERTAIN','UNCERTAIN','UNCERTAIN','Recognizable frame and curved back; blue surface sits above a broad curved grey seat layer. Multiple close traces below it make intended seat integration uncertain.'),
 ('YES','UNCERTAIN','UNCERTAIN','UNCERTAIN','Frame is complete in the two views; blue upper seat and lower shell coexist. Thin irregular back seams are visible, but slat design versus tearing is uncertain.'),
 ('YES','UNCERTAIN','UNCERTAIN','YES','Blue upper seat and lower shell coexist; chair frame remains. The back has visibly interrupted narrow horizontal strips.')],
 (2,'POST'):[
 ('YES','UNCERTAIN','NO','UNCERTAIN','Chair body remains, though several supports become extremely thin in the rear view. Blue seat remains above the lower curved shell; no definite new detached fragment.'),
 ('YES','UNCERTAIN','NO','UNCERTAIN','Body and frame remain; blue upper seat/lower shell relation is still uncertain. Back seams are narrower after D0; images do not certify their topology.'),
 ('YES','UNCERTAIN','UNCERTAIN','YES','Frame remains and blue surface is visible, but interrupted back strips persist. Lower seat shell remains separate in the central section.')],
 (3,'RAW'):[
 ('YES','YES','NO','YES','Round-topped panel chair uses the blue seat. A substantial slit/open interval under the seat front is visible; back has a narrow diagonal opening.'),
 ('YES','YES','UNCERTAIN','YES','Blue seat is recognizable in a broader box-like chair body. Front edge has an irregular triangular gap and side/back panels are rough and folded-looking.'),
 ('NO','YES','YES','YES','Blue seat/back are recognizable, but a detached horizontal support bar and an incomplete base/support arrangement are conspicuous; C use does not imply a complete body.')],
 (3,'POST'):[
 ('YES','YES','NO','YES','Round-back chair survives smoothing, but a conspicuous under-seat/front slit remains. Blue C still serves as the seat.'),
 ('YES','YES','UNCERTAIN','YES','Box-like chair and blue seat persist. Surface edges are straighter, but the rear/side junction still has angular narrow gaps; possible folds remain uncertain.'),
 ('NO','YES','YES','YES','Detached support bar remains after D0 and the chair base is incomplete. Seat/back are present, so blue C use is separate from the body failure.')],
}

def main():
    # BOUND015 entries are supplied only after root actually views those pages.
    bound=json.loads((OUT/'root_bound_visual_observations.json').read_text(encoding='utf-8'))
    for k, entries in bound.items(): OBS[int(k),'BOUND015']=entries
    rows=[]
    for (n,stage),entries in sorted(OBS.items()):
        rid=f'2d08a64e_n{n}'; kind='endpoint_binding' if stage=='BOUND015' else 'training'
        page=OUT/'gallery'/kind/(rid+'_'+stage+'.png')
        sha=hashlib.sha256(page.read_bytes()).hexdigest()
        for label,entry in zip(LABELS,entries):
            rows.append(dict(request_id=rid,label=label,stage=stage,page_path=str(page),page_sha256=sha,
                reviewer='AI_PROXY/root',actually_viewed=True,body_complete=entry[0],C_adopted_as_seat=entry[1],
                obvious_fragments_or_folded_sheets=entry[2],visible_cracks=entry[3],note=entry[4]))
    assert len(rows)==36 and len({r['page_path'] for r in rows})==12
    dest=OUT/'visual_review_endpoint_root.csv'
    with dest.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with dest.with_suffix('.json').open('x',encoding='utf-8') as f:
        json.dump(dict(status='ACTUALLY_VIEWED',reviewer='AI_PROXY/root',nonblind=True,pages=12,rows=rows,
            limit='Finite rendered views and central sections; no topology or absence-of-intersection certification.'),f,indent=2)

if __name__=='__main__':main()
