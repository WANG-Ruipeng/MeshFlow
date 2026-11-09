"""Read-only structural descriptors used to freeze a per-part fallback recipe."""
from .common import *
from r2.data_tools.bench_parts_v1.quality import topology
from r2.data_tools.bench_deployment_tasks_v1.source_tasks import describe

def main():
    result=[]
    for parent in freeze()['parents']:
        row=parent['source'];a=load(row['clean_case_path']);d=describe(row)
        v=a['model_vertices'];fs=a['source_faces'];components=[]
        for c in d['components']:
            ids=c['face_ids'];f=fs[ids];verts=np.unique(f);xyz=v[verts]
            cov=np.cov(xyz.T);eig=np.linalg.eigvalsh(cov)
            components.append(dict(component_id=c['component_id'],face_ids=ids,N=len(ids),V=len(verts),
              bbox_min=xyz.min(0).tolist(),bbox_max=xyz.max(0).tolist(),eigenvalues=eig.tolist(),
              unique_axis_counts=[len(np.unique(xyz[:,j])) for j in range(3)],
              axis_values=[np.unique(xyz[:,j]).tolist() for j in range(3)],
              topology={k:(len(z) if isinstance(z,list) else z) for k,z in topology(f).items()}))
        r=dict(parent_id=row['parent_id'],N=len(fs),components=components)
        save(OUT/'part_inspection'/(row['parent_id']+'.json'),r);result.append(r)
        print(json.dumps(dict(parent=row['parent_id'][:6],components=[{k:c[k] for k in ('component_id','N','V','bbox_min','bbox_max','unique_axis_counts')} for c in components])))
    save(OUT/'part_inspection/manifest.json',dict(rows=result,source_mutations=0,model_calls=0))

if __name__=='__main__':main()
