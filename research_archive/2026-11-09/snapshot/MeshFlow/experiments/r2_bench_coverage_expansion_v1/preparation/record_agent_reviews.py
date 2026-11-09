"""Literal observations recorded after actually inspecting seven multiview pages."""
import json, hashlib
from pathlib import Path
import numpy as np

root=Path('/mnt/e/MeshFlow/experiments/r2_bench_coverage_expansion_v1')
m=json.loads((root/'candidates/manifest.json').read_text())
observations={
'8d074f177479ac42628516f95b4691f':[
('SEAT_SURFACE_1','SEAT_SURFACE','Narrow longitudinal seat strip near the rear seat/back junction; not the complete curved seat.','The original coherent rectangle is clear, but its depth is very small.'),
('LEG_SUPPORT_1','LEG_SUPPORT','One rectangular side surface of a support leg; not a complete solid leg.','The selected surface belongs to the source lower support.'),
('LEG_SUPPORT_2','LEG_SUPPORT','A rectangular side surface of the other support leg; not a complete solid leg.','The selected surface belongs to the source lower support.'),
('BACK_SURFACE_1','BACK_SURFACE','Broad rectangular rear-facing backrest surface; not the entire bent seat/back shell.','The blue region is unambiguously on the backrest.')],
'a8e987a6a6486b962d6c8e51365a5a87':[
('SEAT_SURFACE_1','SEAT_SURFACE','Horizontal lattice/slatted seat-top region with narrow bars.','Semantically intelligible, but the independent geometry gate quarantines nonmanifold vertex contacts; this review does not override that exclusion.'),
('LEG_SUPPORT_1','LEG_SUPPORT','One planar longitudinal support-frame side containing two legs and the connecting rail.','This is a coherent frame side, not one complete solid leg.'),
('LEG_SUPPORT_2','LEG_SUPPORT','The opposite planar support-frame side containing two legs and a connecting rail.','This is a coherent frame side, not one complete solid leg.')],
'a9aa868b77c3769ba873941124e3356f':[
('SEAT_SURFACE_1','SEAT_SURFACE','Broad polygonal seat-top surface between the curved arms.','The selected region corresponds to the visible seating surface.'),
('LEG_SUPPORT_1','LEG_SUPPORT','Complete arched lower support/base shell; not a single leg.','The closed arched support is a meaningful input component; closed C is retained by the current experiment.'),
('BACK_SURFACE_1','BACK_SURFACE','Broad rectangular backrest-board surface.','The region is a back surface, not the entire bench.')],
'd0fa12d918d97314738e43095496b061':[
('SEAT_SURFACE_1','SEAT_SURFACE','One complete narrow longitudinal seat-slat top among several slats.','Do not describe this narrow strip as a complete seat.'),
('LEG_SUPPORT_1','LEG_SUPPORT','One complete solid rectangular support leg/post below the seating surface.','Closed source component; no surface patch was invented.'),
('LEG_SUPPORT_2','LEG_SUPPORT','Another complete solid rectangular support leg/post below the seating surface.','Closed source component at the other end.')],
'd673c7ec4a08aa6db6f5e7b313e4eaae':[
('SEAT_SURFACE_1','SEAT_SURFACE','One narrow longitudinal seat-slat top; not the full seat.','The slat is coherent and belongs to the seat.'),
('LEG_SUPPORT_1','LEG_SUPPORT','One angled end-support assembly with a broad triangular brace/plate and a sloping support.','Its unusually broad triangular brace is present in the original complete source; this is not certified as an ordinary rectangular leg.'),
('LEG_SUPPORT_2','LEG_SUPPORT','The other angled end-support assembly with a broad triangular brace/plate and sloping support.','A coherent original support component with unusual shape; no source repair or simplification.')],
'f1fe8c6c1f9f28321739c3c7bfb786a5':[
('SEAT_SURFACE_1','BACK_SURFACE','Narrow horizontal top cap of the upper backrest rail.','Proposal said SEAT, but both full-object views locate it at the top of the back; relabelled for exposure bookkeeping only.'),
('LEG_SUPPORT_1','PART_ASSEMBLY','One complete stepped end-side block spanning lower support, seat ledge and tall back support.','It is a multi-part side assembly, not just a leg.'),
('LEG_SUPPORT_2','PART_ASSEMBLY','The opposite complete stepped end-side block spanning lower support, seat ledge and back support.','It is a multi-part side assembly, not just a leg.')],
'f35697e0ec1d35dd357ea07594f25728':[
('SEAT_SURFACE_1','SEAT_SURFACE','One narrow, slightly tilted seat-level slat surface near the rear seating region.','Only this strip is provided; the complete seat is not claimed.'),
('LEG_SUPPORT_1','LEG_SUPPORT','One complete end support frame with two sloping legs and a top crosspiece.','The connected U-like end frame is a coherent user component.'),
('LEG_SUPPORT_2','LEG_SUPPORT','The opposite end support frame with two sloping legs and a top crosspiece.','The connected end frame is a coherent user component.')],
}
lookup={r['candidate_id']:r for r in m['rows']}
rows=[]
for parent,items in observations.items():
    image=root/'gallery/data/candidates'/(parent+'.png')
    imagehash=hashlib.sha256(image.read_bytes()).hexdigest()
    for suffix,part,extent,notes in items:
        cid=parent+'__'+suffix
        assert cid in lookup
        rows.append(dict(candidate_id=cid,parent_id=parent,group='TRAIN',decision='ACCEPT',actual_part=part,
            extent_description=extent,human_plausible_input='YES',notes=notes,reviewer='AI_PROXY_bench_coverage_expansion_agent',
            actually_viewed=True,image_path=str(image),image_sha256=imagehash,source='Actual four-column multiview source/C review; not generator output'))
out=root/'preparation/visual_reviews_agent.json'
assert not out.exists()
out.write_text(json.dumps(dict(schema='ACTUAL_AI_PROXY_REVIEW_V1',rows=rows),indent=2)+'\n')
print('recorded_actual_reviews',len(rows))

# Small read-only aid for the root reviewer's hidden seat-region question.
parent='381317c1e4c91e87855931d119219022'
a=np.load(root/'candidates/sources'/(parent+'.npz'))
print('source_keys',a.files)
v=a['source_vertices'];f=a['source_faces'];tri=v[f]
cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)/2
normal=cross/np.maximum(2*area[:,None],1e-30)
ids=np.where((np.abs(normal[:,1])>.99)&(area>.01))[0]
probe=[dict(face_id=int(i),centroid=tri[i].mean(axis=0).tolist(),normal=normal[i].tolist(),area=float(area[i]),
    min_xyz=tri[i].min(axis=0).tolist(),max_xyz=tri[i].max(axis=0).tolist(),is_candidate=int(i) in [104,106]) for i in ids]
(root/'preparation/381317_horizontal_surface_probe.json').write_text(json.dumps(dict(parent_id=parent,faces=probe),indent=2)+'\n')
print(json.dumps(probe,indent=2))
