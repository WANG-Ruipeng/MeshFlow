"""Transcribe root's actual view_image review of 12 complete comparison pages."""
from r2.pilots.seam_abc_v1 import eval_common as C
OUT=C.ROOT/'experiments/r2_seam_pool47_couple_v1'
NOTES={
'73b369b9_n0':('YES','YES','Original front underside/side section gap narrows after both caps; pedestal chair silhouette and visible blue seat preserved. Rear section opening remains.'),
'73b369b9_n1':('YES','NO','Gray seat remains visibly above and covers most blue C; low blue condition still does not serve as the visible top seat. Both bindings leave this mismatch.'),
'73b369b9_n2':('YES','YES','Blue seat is used; upper back gap and local underside/side-section gaps persist. Both caps only shift local edges, without a visible whole-shape change.'),
'73b369b9_n3':('YES','YES','Blue seat/pedestal chair preserved. Local side contour changes slightly; lower free-sheet separation remains in section.'),
'85b16941_n0':('YES','YES','Blue top seat and four-leg/back structure preserved. Under-seat section gaps remain; no clear whole-shape improvement or new gross fragment visible.'),
'85b16941_n1':('YES','YES','Blue seat, apron, four legs and back preserved. Rear seat corner tightens slightly in section; no obvious new defect.'),
'85b16941_n2':('YES','YES','Blue seat and complete chair remain; tiny local perimeter/section differences, no obvious large visual gain or loss.'),
'85b16941_n3':('YES','YES','Blue seat and chair remain; small front skirt fold/detail remains. Binding differences are visually minor.'),
'8c9b2023_n0':('YES','YES','Blue seat/slatted back/four legs preserved. Front section left rim tilts and shows a small local gap after binding; no visible gross body loss.'),
'8c9b2023_n1':('YES','YES','Chair and blue seat preserved. Small side-section shift at the front rim; underside gap remains.'),
'8c9b2023_n2':('YES','YES','Chair and blue seat preserved. Small left rim section discontinuity remains; local slope changes only.'),
'8c9b2023_n3':('YES','YES','Blue seat and slatted chair preserved. Front underside section gaps persist in all columns; no gross new damage visible.')}

def main():
    rows=[]
    for request,(body,adoption,note) in NOTES.items():
        p=OUT/'gallery/cpu_binding'/(request+'_POST.png')
        for cap in ('015','020'):
            rows.append(dict(request_id=request,stage='POST',baseline='B_LONG_LAYOUT',candidate='B_LONG_LAYOUT_BIND_'+cap,
                reviewer='AI_PROXY_ROOT',nonblind=True,actually_viewed=True,image=str(p),image_sha256=C.sha(p),
                baseline_body_complete=body,candidate_body_complete=body,baseline_C_adoption=adoption,candidate_C_adoption=adoption,
                new_gross_fragment_visible='NO',new_gross_body_loss_visible='NO',topology_certified_by_images=False,
                visual_change='LOCAL_MINOR',uncertainty='Fixed views/central sections do not certify all tiny cracks, intersection or full connectivity.',note=note))
    C.csvwrite(OUT/'visual_review_cpu_root.csv',rows)
    C.write(OUT/'visual_review_cpu_root.json',dict(status='ACTUALLY_VIEWED',images=12,paired_rows=24,rows=rows,
        policy='Manual model-view observations, not generated from numerical metrics; unavailable/occluded topology not certified.'))
if __name__=='__main__':main()
