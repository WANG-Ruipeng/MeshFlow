"""Transcribe root's actual displayed Wave2 pages, never derive labels from metrics."""
import csv, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
labels=('CLEAN_BULK_FM_S1000','SURGICAL_LATE4_S1000')
rows=[]
data={
'2d08a64e_n0':[
('YES','UNCERTAIN','NO','UNCERTAIN','Complete cantilever chair; blue upper seat and distinct lower curved gray shell leave adoption ambiguous; no obvious loose sheet.'),
('YES','UNCERTAIN','NO','UNCERTAIN','Complete angular cantilever chair; blue upper seat remains above a deeper gray seat shell; fine gaps uncertain.')],
'2d08a64e_n1':[
('YES','YES','NO','UNCERTAIN','Complete panel-back chair and blue seat; fine panel-edge/central-section gaps are insufficient to certify tears.'),
('YES','YES','NO','UNCERTAIN','Complete panel chair with more horizontal back striations; blue seat role clear; fine gaps uncertain.')],
'2d08a64e_n2':[
('YES','UNCERTAIN','NO','UNCERTAIN','Complete cantilever body with gray under-seat shell; reference bright horizontal back split becomes less visible, not proof of healed topology.'),
('YES','UNCERTAIN','NO','UNCERTAIN','Complete cantilever body with lower gray seat; thin back junction and under-seat gaps remain uncertain.')],
'2d08a64e_n3':[
('YES','YES','NO','UNCERTAIN','Complete cantilever panel chair; tall rectangular back replaces reference faceted crest; blue seat maintained, fine junction gaps uncertain.'),
('YES','YES','UNCERTAIN','YES','Complete compact panel chair; reclined inset back has a visible long pale slit at the blue seat junction; corner slivers uncertain.')],
'608b46b0_n0':[
('YES','YES','NO','UNCERTAIN','Four legs and broad arched back remain; blue plane acts as seat. Gray under-seat shell is separated in section; no clear large tear in perspective.'),
('YES','YES','NO','UNCERTAIN','Four-leg chair retained; blue seat clear. Several section gaps and under-seat offset remain, without obvious detached flap.')],
'608b46b0_n1':[
('YES','YES','UNCERTAIN','YES','Recognizable four-leg chair; irregular white openings at back crest and thin slivers remain; blue seat retained.'),
('YES','YES','UNCERTAIN','YES','Recognizable chair and blue seat; back crest has small white holes/irregular strips; no obvious whole-part loss.')],
'608b46b0_n2':[
('YES','YES','NO','UNCERTAIN','Complete box-like armchair enclosing blue seat; broad side/back panels continuous in these views, narrow corner gaps uncertain.'),
('YES','YES','UNCERTAIN','UNCERTAIN','Complete panel armchair and blue seat; lower inner back has a jagged patch/strip, insufficient to call a detached fragment.')],
'608b46b0_n3':[
('YES','YES','NO','YES','Complete four-leg chair and blue seat; back-crest holes and long white slits through rear/side leg panels remain visible.'),
('YES','YES','NO','YES','Complete chair and blue seat; leg panels look more filled than CLEAN, but a small bright back-crest opening remains.')],
'610b1621_n0':[
('YES','YES','NO','UNCERTAIN','Complete broad armchair with short feet and blue seat; front rim/seat narrow gap uncertain.'),
('YES','YES','NO','UNCERTAIN','Complete broad armchair; lower back edge is stepped/notched above blue seat, meaning of small openings uncertain.')],
'610b1621_n1':[
('YES','YES','NO','YES','Complete slender chair with blue seat; a thin white horizontal split is visible across the upper back slat.'),
('YES','YES','NO','UNCERTAIN','Complete slender chair and blue seat; fine back-slat junction lines are visible but no clear large white split.')],
'610b1621_n2':[
('YES','YES','UNCERTAIN','YES','Four-leg chair with circular back opening retained; additional narrow irregular white gaps and lower-back strips remain.'),
('YES','YES','YES','YES','Four-leg chair retained but a large triangular sheet protrudes behind the back; irregular back-top holes and strips are clear.')],
'610b1621_n3':[
('YES','YES','YES','YES','Large sheet still protrudes behind the slatted back; fine horizontal white back split; blue seat and main chair retained.'),
('YES','YES','UNCERTAIN','YES','Changes to arched/slatted back and thicker legs; no matching large rear flap, but irregular back-post gaps and slivers remain.')],
}
for rid,entries in data.items():
    page=ROOT/'gallery/wave2_POST'/(rid+'_POST.png')
    for label,entry in zip(labels,entries):
        row=dict(request_id=rid,label=label,stage='POST',page_path=str(page),page_sha256=hashlib.sha256(page.read_bytes()).hexdigest(),reviewer='AI_PROXY_NONBLIND',reviewer_identity='/root',actually_viewed='True')
        row.update(zip(('body_complete','C_adopted_as_seat','obvious_fragments_or_folded_sheets','visible_cracks','note'),entry)); rows.append(row)
with (ROOT/'visual_review_parts/wave2_DEV_root.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('ACTUAL_REVIEW_TRANSCRIBED',len(rows))

# These eight POST/BOUND pages were separately displayed and inspected by root.
# All keep their visible macro-level labels; exact geometric deltas belong to scorers.
bound=[]
notes={
'2d08a64e_n3':'Both displayed views and sections keep the same chair silhouette after binding; no visible repair of existing junction ambiguity/cracks and no new macro defect.',
'608b46b0_n3':'After binding the back-crest openings and chair silhouette are visually unchanged; CLEAN leg slits persist; no visible new missing major part.',
'610b1621_n1':'Bound view preserves the slender chair and blue seat. CLEAN upper-back thin split persists; no additional visible macro sheet or tear.',
'610b1621_n2':'Bound view leaves the back geometry unchanged: CLEAN irregular openings/strips remain; SURGICAL large rear triangular sheet and crest holes remain.',
}
for row in rows:
    if row['request_id'] not in notes: continue
    page=ROOT/'gallery/wave2_BOUND'/(row['request_id']+'_'+row['label']+'.png')
    r=dict(row,stage='BOUND015',page_path=str(page),page_sha256=hashlib.sha256(page.read_bytes()).hexdigest(),note=notes[row['request_id']])
    bound.append(r)
with (ROOT/'visual_review_parts/wave2_bound_DEV_root.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(bound[0]));w.writeheader();w.writerows(bound)
print('ACTUAL_BOUND_REVIEW_TRANSCRIBED',len(bound))
