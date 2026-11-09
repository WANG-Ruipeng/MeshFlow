"""CPU summaries of frozen Bench inputs; no manifest mutation or model import."""
from pathlib import Path
import os,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_control_transfer_v1/data'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def spread(values):
    a=np.asarray(values,float)
    return dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()),mean=float(a.mean()))
def text_range(s,digits=2):return f"{s['min']:.{digits}f} / {s['median']:.{digits}f} / {s['max']:.{digits}f}"

def main():
    before=sha(OUT/'manifest.json');m=json.loads((OUT/'manifest.json').read_text('utf-8'));rows=[];groups={}
    for r in m['train']+m['dev']:
        with np.load(r['case_path'],allow_pickle=False) as z:
            t=z['full_target'].astype(float);known=z['known_ids'];cv=z['C_vertices'].astype(float);cf=z['C_faces_original']
        a=.5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
        ca=.5*np.linalg.norm(np.cross(cv[cf[:,1]]-cv[cf[:,0]],cv[cf[:,2]]-cv[cf[:,0]]),axis=1)
        assert np.isclose(ca.sum(),a[known].sum(),rtol=1e-6)
        rows.append(dict(parent_id=r['parent_id'],group=r['group'],official_split=r['official_split'],N=r['N'],K=r['K'],
            C_face_fraction_percent=100*r['K']/r['N'],C_area_fraction_counted_source_percent=100*float(ca.sum()/a.sum()),
            C_area_model=float(ca.sum()),M_counted_face_area_model=float(a.sum()),C_boundary_edges=r['C_boundary_edges'],
            source_owners=r['source_owner_count'],source_alias=r['source_alias_count'],source_unsupported=r['source_null_unsupported_count'],
            source_duplicate_ID_faces=r['source_exact_duplicate_ID_faces'],source_degenerate_faces=r['source_degenerate_faces'],region_kind=r['region_kind']))
    for name in ('TRAIN','DEV'):
        a=[r for r in rows if r['group']==name]
        groups[name]=dict(parents=len(a),N=spread([r['N'] for r in a]),K=spread([r['K'] for r in a]),
            K_over_N_percent=spread([r['C_face_fraction_percent'] for r in a]),C_over_M_area_percent=spread([r['C_area_fraction_counted_source_percent'] for r in a]),
            boundary_edges=sum(r['C_boundary_edges'] for r in a),owners=sum(r['source_owners'] for r in a),aliases=sum(r['source_alias'] for r in a),unsupported=sum(r['source_unsupported'] for r in a),
            zero_owner_parents=sum(r['source_owners']==0 for r in a),duplicate_ID_faces=sum(r['source_duplicate_ID_faces'] for r in a),
            parents_with_duplicate_ID_faces=sum(r['source_duplicate_ID_faces']>0 for r in a),degenerate_faces=sum(r['source_degenerate_faces'] for r in a),
            parents_with_degenerate_faces=sum(r['source_degenerate_faces']>0 for r in a),nonhorizontal_fallback=sum(r['region_kind']=='PLANAR_PATCH' for r in a))
    result=dict(schema='bench_dataset_card_statistics_v1',manifest_sha256=before,groups=groups,per_parent=rows,
        area_denominator='Sum of every retained original source face area, including duplicate faces; not unique exterior surface area',
        source_unsupported='No verified unique opposite-winding legal C/Y source neighbor; combines absence, incidence/winding conflicts, and unresolved/illegal contacts. Not all are semantic Null.',
        source_whole_FF='NOT_RUN',model_calls=0)
    dest=OUT/'dataset_card_statistics.json'
    content=json.dumps(result,indent=2)+'\n'
    if dest.exists():
        if dest.read_text('utf-8')!=content:raise FileExistsError(dest)
    else:dest.write_text(content,encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    specs=[('K','Input triangles K',[2,4,8,16,32,64]),('C_face_fraction_percent','Input face fraction K/N (%)',[0,2,4,8,16,32]),
           ('C_area_fraction_counted_source_percent','Input / counted source area (%)',[0,5,10,15,20,25,30,35,40]),('source_owners','Verified source owner edges',[0,1,4,8,16,32,64])]
    for ax,(key,label,bins) in zip(axes.flat,specs):
        for name,color in [('TRAIN','#3c78a8'),('DEV','#d98739')]:
            vals=[r[key] for r in rows if r['group']==name]
            ax.hist(vals,bins=bins,weights=np.full(len(vals),100/len(vals)),histtype='step',linewidth=2,label=f'{name} ({len(vals)} parents)',color=color)
        ax.set_xlabel(label);ax.set_ylabel('Parents (%)');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=9);fig.suptitle('Frozen Bench control inputs: 64 TRAIN + 8 DEV',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'dataset_overview.png',dpi=170);plt.close(fig)
    tr,dv=groups['TRAIN'],groups['DEV']
    lines=['# Bench control transfer — dataset card','',
      '**Frozen data:** 64 TRAIN parents and 8 DEV parents, one existing mesh and one geometric `PLANAR_PATCH` condition per parent. No source mesh was repaired, welded, remeshed, deduplicated, or replaced.','',
      '| Quantity | TRAIN | DEV |','|---|---:|---:|',
      f"| Independent source parents | {tr['parents']} | {dv['parents']} |",
      f"| N: min / median / max | {text_range(tr['N'],0)} | {text_range(dv['N'],0)} |",
      f"| K: min / median / max | {text_range(tr['K'],1)} | {text_range(dv['K'],1)} |",
      f"| K/N (%): min / median / max | {text_range(tr['K_over_N_percent'])} | {text_range(dv['K_over_N_percent'])} |",
      f"| Counted C/M area (%): min / median / max | {text_range(tr['C_over_M_area_percent'])} | {text_range(dv['C_over_M_area_percent'])} |",
      f"| C open boundary edges | {tr['boundary_edges']} | {dv['boundary_edges']} |",
      f"| Verified source owner / alias / unsupported | {tr['owners']} / {tr['aliases']} / {tr['unsupported']} | {dv['owners']} / {dv['aliases']} / {dv['unsupported']} |",
      f"| Parents with zero verified source owners | {tr['zero_owner_parents']} | {dv['zero_owner_parents']} |",'',
      'Area uses the sum of **all retained original source face areas**, including duplicate faces. It is not unique exterior surface area. Distribution summaries give equal weight to each parent; the edge totals pool edges.',
      '', '![Frozen input distributions](dataset_overview.png)','',
      '## Source and split','',
      'The existing official `shapenet-cls.tar.gz` archive contains 560 Bench NPZ meshes. Of these, 88 have N128–256; 84 have a qualifying connected region, with 84 distinct exact normalized unordered-triangle geometry hashes. The 72 pilot parents were selected deterministically before model outputs. Remaining candidates are not used as a result-dependent replacement pool.','',
      'DEV contains all seven qualifying parents from the official Bench test split and one parent from the official training split, held out from this conditional adaptation. Results must distinguish these two exposure groups. TRAIN contains 64 other official training parents. Exact-geometry duplicates were grouped for parent selection; near-duplicate or original-family equivalence has not been exhaustively established.','',
      'This is a category-specific adaptation test using the official Bench prior. It does not show zero-shot transfer of the Chair checkpoint or arbitrary-category generalization.','',
      '## Input construction and preservation','',
      'The constructor selects an existing source-index-connected nearly planar region, preferring a broad horizontal surface above the object bottom. Qualification requires K≥2, nonzero constituent area, consistent internal C winding, no C nonmanifold edge, sufficient planar extent, and area/convex-hull compactness at least 0.60. Local adjacent normals must have dot product >0.999 and adjacent plane offset <1e-4 in normalized object units. No new faces are created.','',
      f"The selected regions include broad boards, narrow bench slats, lower/bottom boards and connected frame-like patches with holes. {tr['nonhorizontal_fallback']} TRAIN regions use a non-horizontal fallback. These are geometric `PLANAR_PATCH` inputs, **not a set of verified seat-top semantic labels**. K≥2 alone does not imply a rectangle or a complete semantic part.",'',
      'Each full object receives the official FP32 bounding-box normalization once. Pre-OT coordinates use division by 0.3762; model coordinates are exactly pre-OT ×2. C is never normalized separately. The case packs retain source vertices, source faces, canonical corner permutations, C source vertex IDs, original/model C topology, source face identities, and the untouched free complement.','',
      'All 72 full/source views and C-only principal-plane views were actually inspected by AI_PROXY reviewers. C-only views avoid confusing full-mesh occlusion with a missing input region. A separate finite-triangle pair audit on all selected C regions found **0 illegal pairs and 0 unresolved pairs**. This result certifies that bounded detector check only; it is not a quality certificate for the unknown completion or full source mesh.','',
      '## Source supervision and remaining defects','',
      'A source owner is an open C edge with exactly one source C face and one source free face, opposite edge winding, and a verified legal finite shared-edge contact. If one free face serves multiple C edges, only the lowest edge ID owns the seam slot and the others are aliases. No alias case occurred in this selected pool. Unsupported edges combine absent neighbors, incidence conflicts, winding mismatches and nonlegal/uncertain contacts; they should not all be interpreted as semantic “do not connect” labels.','',
      '| Retained source defect diagnostic | TRAIN | DEV |','|---|---:|---:|',
      f"| Same unordered source-vertex-ID face copies beyond first | {tr['duplicate_ID_faces']} | {dv['duplicate_ID_faces']} |",
      f"| Parents containing those copies | {tr['parents_with_duplicate_ID_faces']}/{tr['parents']} | {dv['parents_with_duplicate_ID_faces']}/{dv['parents']} |",
      f"| Source degenerate faces (twice area ≤1e-12 model units²) | {tr['degenerate_faces']} | {dv['degenerate_faces']} |",
      f"| Parents containing those faces | {tr['parents_with_degenerate_faces']}/{tr['parents']} | {dv['parents_with_degenerate_faces']}/{dv['parents']} |",'',
      'These original defects remain in the full target/free complement. Whole-source FF was **NOT_RUN** in this preparation; there is no claim that all other source faces are clean. The pilot isolates applying the existing method to another category, rather than combining category transfer with target cleaning.','',
      '## Fixed evaluation inputs','',
      'The panel contains four training parents and all eight DEV parents, one fixed C each, two preregistered Gaussian arrays each: 24 requests per model, shared by PLAIN and SEAM. The model and occupancy head receive C geometry/topology and requested N only. Source owner labels and complete source geometry are training/audit information, not inference inputs. Source reconstruction is not the open-ended generation success criterion.','',
      '## Files','',
      '- [Frozen manifest](manifest.json), [per-parent numerical summary](dataset_card_statistics.json), and [fixed request list](evaluation_requests.json).',
      '- [C finite-pair audit](condition_geometry_audit.json), [source interface audit](source_interface_audit.json), [actual visual review](input_visual_review.json), and [review rows](input_visual_review.csv).',
      '- Full/source sheets: `gallery/`; input-only sheets: `gallery_C_only/`; source-preserving NPZ cases: `cases/`.',
      '- Constructor: `r2/pilots/bench_control_transfer_v1/data_prepare.py`; this summary: `dataset_card.py`.','',
      f'Frozen manifest SHA256: `{before}`. This card and its plot were generated from saved inputs on CPU with zero model calls and without changing the manifest.','']
    (OUT/'DATASET_CARD.md').write_text('\n'.join(lines),encoding='utf-8')
    assert sha(OUT/'manifest.json')==before
    print(json.dumps(groups,indent=2))

if __name__=='__main__':main()
