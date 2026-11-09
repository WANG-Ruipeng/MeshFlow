import subprocess,sys
import numpy as np
from meshflow_control.config import validate_config
from meshflow_control.mesh_io import write_ply,read_ply,validate_C
def test_configuration_rejects_science_changes():
    import pytest
    with pytest.raises(ValueError):validate_config({"training":{"updates":1001}})
    with pytest.raises(ValueError):validate_config({"arm":"C","readout_mode":"mixed"})
def test_raw_export_preserves_winding_multiplicity_bits(tmp_path):
    t=np.array([[[0.,-0.,1.],[2.,3.,4.],[5.,6.,7.]]]*2,dtype=np.float32)
    write_ply(tmp_path/"raw.ply",t);back,indices=read_ply(tmp_path/"raw.ply")
    assert back.tobytes()==t.tobytes()
    assert np.array_equal(indices,np.arange(6).reshape(2,3))
def test_public_input_does_not_infer_N():
    import pytest
    with pytest.raises(ValueError):validate_C(np.zeros((2,3,3),np.float32),127)
def test_help_no_cuda_or_asset_load():
    script="import sys; from meshflow_control.cli import main;\ntry: main(['--help'])\nexcept SystemExit as e: assert e.code==0\nassert 'torch' not in sys.modules"
    subprocess.run([sys.executable,"-c",script],check=True,capture_output=True)
