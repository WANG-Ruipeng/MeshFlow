# Third-point normal guard: CPU falsification

The proposed normal angle <60 degrees guard is **not safe as stated**. It rejects 50/62 legal source interfaces in the four-C200 training conditions, and 341/389 across the existing source interface table. These are existing verified local interface contacts, not certification of whole-source quality.

Among retained seam faces with legal CF in outputs with bulk reach >=30%, it rejects 36/50. Existing legal CF table also evaluates both control and candidate. No faces are changed or deleted by this audit.

## Geometric contract

Let t be the directed C boundary edge, n the incident original-C face normal, and m=n cross t its inward in-plane conormal. Write x3-edge_midpoint = a t + r m + h n. The seam uses reversed base (c1,c0,x3). Then dot(n_seam,n_C) = -r/sqrt(r^2+h^2). Requiring angle<60 means r < -sqrt(3)*abs(h): it forces a narrow outward in-plane wedge. A valid vertical sidewall has r=0, h!=0 and angle=90, so it is rejected. A valid underside slant can have r>0,h<0 and angle>90. The plane normal is not the boundary outward conormal.

An inward coplanar extension (r>0,h=0) is indeed rejected, but detecting this one bad case does not validate the broader rule. Even a passing angle does not certify semantic use, absence of intersections with nonincident C faces, FF quality or bulk connection. Analytic examples are saved separately.

## Allowed next design inference

Preserve reversed owner-edge winding. If a geometric penalty is designed, target finite forbidden overlap/penetration with actual C, or a specifically validated near-coplanar inward wedge, while allowing right-angle sidewalls. C alone does not specify one unique sidewall normal or dihedral target. Using source-owner normals would be supervised training information, not an inference oracle, and requires a separate experiment. Nothing here trains that loss.

The 3f/n2 slot16 RAW/PRE_GEO/POST local coordinates are in summary.json and generated_per_seam_stage.csv. A temporal trajectory claim cannot be inferred from these endpoints.

## Scope

Single CPU worker/thread; zero model imports/calls, no GPU computation, no new noise, no mesh edits, no old scoring changes. The existing source classifications and actual indexed-edge cache are read-only. Normal orientation references original saved C winding; global outwardness is not independently certified. Null is not treated as a required connection.
