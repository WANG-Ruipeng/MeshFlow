"""Public CPU integration checks: provenance, stage order, and lossless export."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from meshflow_control import denoise
from meshflow_control.source_winding import capture_source


def fixture(tmp_path):
    v = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], dtype=np.float32)
    ids = np.array([[0,1,2],[1,3,2]], dtype=np.int64)
    original = v[ids]
    source = capture_source(original, source_vertex_ids=ids,
                            corner_permutation=np.array([[0,1,2],[0,2,1]]))
    C = source["C"]
    free = np.array([[[1,0,0],[1,0,-1],[1,1,0]],
                     [[1,1,0],[1,0,-1],[1,1,-1]]], dtype=np.float32)
    raw = np.concatenate([C, free])
    proposal = raw.copy()
    proposal[:2,:,0] += np.float32(.03)
    proposal[2:,:,0] += np.float32(.02)
    np.savez_compressed(tmp_path/"raw.npz", raw=raw)
    np.savez_compressed(tmp_path/"C.npz", **source)
    np.savez_compressed(tmp_path/"proposal.npz", official_unconstrained_proposal=proposal)
    receipt = dict(status="PASS", raw_sha256=denoise.geo_io.file_sha256(tmp_path/"raw.npz"),
                   checkpoint_sha256=denoise.OFFICIAL_SHA,
                   output_sha256=denoise.geo_io.file_sha256(tmp_path/"proposal.npz"))
    (tmp_path/"receipt.json").write_text(json.dumps(receipt))
    return dict(input=tmp_path/"raw.npz",known_faces=2,condition=tmp_path/"C.npz",
                proposal=tmp_path/"proposal.npz",proposal_receipt=tmp_path/"receipt.json"), original, proposal


def test_cached_full_pipeline_preserves_source_C_and_does_not_load_model(tmp_path, monkeypatch):
    args, original, proposal = fixture(tmp_path)
    monkeypatch.setattr(denoise,"actual_proposal",lambda *a:pytest.fail("Cached CPU path loaded model"))
    out=tmp_path/"result"
    result=denoise.run(**args,out=out)
    assert result["status"]=="COMPLETE" and result["model_forwards_this_invocation"]==0
    final=denoise.load_npz(out/"post.npz")
    canonical=denoise.load_npz(out/"post_canonical.npz")
    assert final["C"].tobytes()==original.tobytes()
    assert final["vertices"].tobytes()==canonical["vertices"].tobytes()
    assert np.array_equal(np.sort(final["faces"],axis=1),np.sort(canonical["faces"],axis=1))
    assert denoise.load_npz(out/"proposal.npz")["official_unconstrained_proposal"].tobytes()==proposal.tobytes()
    assert not (out/"resources_before_model.json").exists()
    resumed=denoise.run(**args,out=out,resume=True)
    assert resumed["reused"] and resumed["model_forwards_this_invocation"]==0
    with pytest.raises(FileExistsError):denoise.run(**args,out=out)
    with (out/"post.ply").open("ab") as f:f.write(b"broken")
    with pytest.raises(ValueError,match="changed"):denoise.run(**args,out=out,resume=True)


def test_rbf_reads_unrestored_proposal_and_remains_explicit(tmp_path):
    args, original, proposal=fixture(tmp_path)
    denoise.run(**args,out=tmp_path/"none")
    denoise.run(**args,out=tmp_path/"rbf",coordination="rbf",reference_length=10.)
    none=denoise.load_npz(tmp_path/"none/denoised.npz")["output"]
    rbf=denoise.load_npz(tmp_path/"rbf/denoised.npz")["output"]
    assert none[2:].tobytes()==proposal[2:].tobytes()
    assert rbf[2:].tobytes()!=none[2:].tobytes()
    assert rbf[:2].tobytes()==none[:2].tobytes()
    assert np.max(np.linalg.norm(rbf[2:].astype(float)-proposal[2:],axis=-1))<=.2
    with pytest.raises(ValueError,match="reference-length"):
        denoise.run(**args,out=tmp_path/"bad",coordination="rbf")
    assert not (tmp_path/"bad").exists()


def test_identity_or_missing_mapping_stops_before_outputs(tmp_path):
    args, _, _=fixture(tmp_path)
    receipt=denoise.read_json(args["proposal_receipt"])
    receipt["raw_sha256"]="0"*64
    Path(args["proposal_receipt"]).write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="identity mismatch"):
        denoise.run(**args,out=tmp_path/"bad")
    assert not (tmp_path/"bad").exists()
    np.savez(tmp_path/"nomap.npz",C=denoise.load_npz(args["condition"])["C"])
    args["condition"]=tmp_path/"nomap.npz"
    with pytest.raises(ValueError):denoise.run(**args,out=tmp_path/"bad2")
    assert not (tmp_path/"bad2").exists()


def test_explicit_capture_indexed_user_C_without_GT(tmp_path):
    v=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    f=np.array([[2,0,1]],np.int64)
    np.savez(tmp_path/"user.npz",vertices=v,faces=f)
    denoise.prepare_condition(input=tmp_path/"user.npz",input_key="indexed",out=tmp_path/"source.npz")
    saved=denoise.load_npz(tmp_path/"source.npz")
    assert saved["C_original"].tobytes()==v[f].tobytes()
    assert np.array_equal(saved["C_source_vertex_ids"],f)
    assert np.array_equal(saved["C_source_corner_permutations"],[[0,1,2]])


def test_orient_existing_post_uses_no_GEO_or_model(tmp_path,monkeypatch):
    args,original,_=fixture(tmp_path)
    denoise.run(**args,out=tmp_path/"first",winding="canonical")
    monkeypatch.setattr(denoise.geo_c,"process",lambda *a:pytest.fail("Repeated GEO"))
    result=denoise.orient(input=tmp_path/"first/post.npz",condition=args["condition"],out=tmp_path/"oriented")
    assert result["model_forwards"]==result["GEO_calls"]==0
    assert denoise.load_npz(tmp_path/"oriented/post.npz")["C"].tobytes()==original.tobytes()


def test_no_implicit_retry_of_incomplete_export(tmp_path):
    args,_,_=fixture(tmp_path)
    denoise.run(**args,out=tmp_path/"done")
    (tmp_path/"done/denoise.json").unlink()
    with pytest.raises(RuntimeError,match="Incomplete geometry"):
        denoise.run(**args,out=tmp_path/"done",resume=True)


def test_help_paths_do_not_import_torch():
    code="""import sys
from meshflow_control.r2_cli import main
try: main(['denoise','--help'])
except SystemExit as e: assert e.code==0
assert 'torch' not in sys.modules
assert not any(n.startswith('native_t1') for n in sys.modules)
"""
    r=subprocess.run([sys.executable,"-B","-c",code],text=True,capture_output=True)
    assert r.returncode==0,r.stderr
