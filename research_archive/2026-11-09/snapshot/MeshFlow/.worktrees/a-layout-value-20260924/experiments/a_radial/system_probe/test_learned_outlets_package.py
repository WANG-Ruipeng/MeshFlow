import copy,os,shutil,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import pytest
from . import learned_outlets_package as p


def fixture_rows():
 return [{'seed':s,'sample_id':f'{i:05d}','arm':a,'raw_only':True,'fallback_to_Fixed':False,'metric_status':'ok'} for s in (10,11) for i in range(64) for a in ('longfit','balanced')]


def test_full_denominator_retains_failure():
 rows=fixture_rows();r=rows[24];r.update(metric_status='failed',metric_error='AssertionError()',metric_traceback='legacy.chart.retract',paired_available=False,rms_gain_percent_two_measurements=None)
 assert p.validate_slots(rows)==[[10,'00012','longfit']]
 for malformed in [rows[:-1],rows[:-1]+[rows[0]]]:
  with pytest.raises(ValueError,match='slots'):p.validate_slots(malformed)
 r['rms_gain_percent_two_measurements']=[0,0]
 with pytest.raises(ValueError,match='invents metric'):p.validate_slots(rows)


def test_no_fallback_disguised_as_raw():
 rows=fixture_rows();rows[0]['fallback_to_Fixed']=True
 with pytest.raises(ValueError,match='Raw-only'):p.validate_slots(rows)


def test_typed_array_hash_cannot_be_replaced_by_canonical(tmp_path):
 v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=np.float64);f=np.array([[0,1,2]],dtype=np.int32);m=tmp_path/'m.npz';np.savez(m,vertices=v,faces=f)
 row={'mesh_npz_sha256':p.file_hash(m),'stored_array_hash':p.array_hash(v,f),'canonical_geometry_hash':p.array_hash(v,f.astype(np.int64)),'quality':{'face_count':1}}
 p.verify_mesh(m,row)
 np.savez(m,vertices=v,faces=f.astype(np.int64));row['mesh_npz_sha256']=p.file_hash(m)
 with pytest.raises(ValueError,match='typed'):p.verify_mesh(m,row)


def test_copy_byte_change_detected(tmp_path,monkeypatch):
 f=tmp_path/'src';f.write_bytes(b'original');copy2=p.shutil.copy2
 def changed(a,b):copy2(a,b);Path(b).write_bytes(b'changed')
 monkeypatch.setattr(p.shutil,'copy2',changed)
 with pytest.raises(ValueError,match='digest'):p.copy_files({'a':f},tmp_path/'dest')


def test_zip_security_and_self_manifest_binding(tmp_path):
 for index,name in enumerate(['../escape','/absolute','FILE_MANIFEST.json']):
  z=tmp_path/f'{index}.zip'
  with zipfile.ZipFile(z,'w') as a:a.writestr(name,'{}')
  with pytest.raises(ValueError):p.extract(z,tmp_path/f'out{index}','wrong')


def test_fresh_source_closure_cpu_only(tmp_path):
 repo=Path(p.__file__).resolve().parents[3];source=tmp_path/'source'
 names=set(p.SOURCE_NAMES)|set(p.EXTRA_SOURCES)
 assert not any('fm_tail' in n or 'system_formal' in n or 'margin_' in n or 'warm_start_controls' in n for n in names)
 for name in names:
  src=repo/'experiments/a_radial'/name;dst=source/'experiments/a_radial'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 cmd='import torch; from pathlib import Path; import experiments.a_radial.system_probe.learned_outlets_package as p; import experiments.a_radial.system_probe.learned_outlets as l; import experiments.a_radial.layout_longfit.train; import experiments.a_radial.layout_balanced.train; assert Path(p.__file__).resolve().is_relative_to(Path.cwd()/"source"); assert not torch.cuda.is_initialized(); print("PASS")'
 r=subprocess.run([sys.executable,'-B','-c',cmd],cwd=tmp_path,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source)),capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 assert r.stdout.strip()=='PASS'
