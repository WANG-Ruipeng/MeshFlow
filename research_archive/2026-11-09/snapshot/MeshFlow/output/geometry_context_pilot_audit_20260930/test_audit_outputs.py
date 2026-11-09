"""One CPU synthetic integration fixture; never loads a model or real output."""
import json
from pathlib import Path
import tempfile
import numpy as np
import audit_outputs as audit


def run_cpu_fixture():
    with tempfile.TemporaryDirectory(prefix="fixture_",dir=Path(__file__).parent) as directory:
        root=Path(directory)
        target=np.zeros((112,3,3),dtype=np.float32)
        for i in range(112):
            target[i]=np.asarray([[3*i,0,0],[3*i+1,0,0],[3*i,1,0]],dtype=np.float32)
        vertices=target.reshape(-1,3).copy()
        faces=np.arange(336,dtype=np.int64).reshape(112,3)
        C=target[:12].copy()
        arrays=root/"inputs.npz"
        np.savez(arrays,C=C,target=target,known=np.arange(12),vertices=vertices,faces=faces)
        full=target.copy()
        full[12]=C[0]  # One C/free positive-area coplanar duplicate, not a repair.
        raw=root/"raw.npz"
        np.savez(raw,output=full.reshape(112,9),path=np.repeat(full[None],51,axis=0))
        manifest=dict(schema=audit.SCHEMA,arrays_path=str(arrays),arrays_sha256=audit.file_sha256(arrays),
            patches=[dict(patch_id=0,parent=0,constraints_key="C",target_key="target",known_face_ids_key="known",source_vertices_key="vertices",source_face_vertex_ids_key="faces",bbox_diagonal=float(np.linalg.norm(np.ptp(target.reshape(-1,3),axis=0))))],
            outputs=[dict(output_id="synthetic_T0_P0_patch0",model="T0",parent=0,patch_id=0,patch_group="SYNTHETIC_FIXTURE_ONLY",seed=9701,raw_path=str(raw),raw_sha256=audit.file_sha256(raw))])
        manifest_path=root/"manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        result=audit.run(manifest_path,root/"audit")
        row=result["rows"][0]
        assert result["status"]=="PASS_EXECUTION" and len(result["patch_audits"])==1
        assert row["scans"]["C_free"]["tested_pairs"]==1200 and row["scans"]["free_free"]["tested_pairs"]==4950
        assert result["patch_audits"]["P0:0"]["context"]["pair_scan"]["tested_pairs"]==66
        assert row["scalars"]["CF_coplanar_pair_count"]==1 and row["scalars"]["FF_illegal_pair_count"]==0
        assert row["scalars"]["exact_C_free_shared_edge_count"]==3 and row["C_all51_bitwise_preserved"]
        assert result["CUDA_initialized"] is False
        assert all(result["ledger"][k]==0 for k in ("model_forwards","backward_calls","optimizer_updates","OT_calls","autograd_calls","geometry_repairs"))
        try:audit.run(manifest_path,root/"audit")
        except FileExistsError:pass
        else:raise AssertionError("Existing audit directory overwritten")
        return dict(status="PASS",checks=8,synthetic_endpoint_audits=1,synthetic_context_audits=1,
                    ledger=result["ledger"],seconds=result["total_seconds"],
                    formal_outputs_audited=0,model_forwards=0,autograd_calls=0,OT_calls=0,optimizer_updates=0)


if __name__=="__main__":
    result=run_cpu_fixture()
    destination=Path(__file__).parent/"wrapper_cpu_test.json"
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
