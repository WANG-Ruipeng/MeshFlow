from pathlib import Path
p=Path('/mnt/e/MeshFlow-Control/tests/test_evaluation.py')
s=p.read_text().replace("    import meshflow_control.evaluation.evaluate\n    assert 'torch' not in sys.modules","    import subprocess\n    code=\"import sys;import meshflow_control.evaluation.evaluate;assert 'torch' not in sys.modules\"\n    subprocess.run([sys.executable,'-B','-c',code],check=True)")
p.write_text(s)
