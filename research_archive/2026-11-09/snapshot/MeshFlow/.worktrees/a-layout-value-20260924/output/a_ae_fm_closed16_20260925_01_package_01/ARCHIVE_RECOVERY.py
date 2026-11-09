"""Recover only an archive-verifier relative PYTHONPATH error; preserve original ZIP/FAILED."""
import json,os,subprocess,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd().resolve()))
from experiments.a_radial.system_probe import ours_package as p

repo=Path.cwd().resolve();out=(repo/'output/a_ae_fm_closed16_20260925_01_package_01').resolve()
payload=out/'payload';fresh=out/'fresh_extract';archive=out/'AEFM8K_CLOSED16_FULL.zip'
started=time.perf_counter()
p.require((out/'FAILED.json').is_file() and not (out/'SEALED.json').exists(),'Expected original final-import failure')
failed=p.read(out/'FAILED.json');p.require('Fresh source dependency import failed' in failed['error']
    and "No module named 'experiments'" in failed['traceback'],'Different failure cannot be recovered by this script')
archive_before=p.file_hash(archive)
p.require(p.file_hash(payload/'FILE_MANIFEST.json')==p.file_hash(fresh/'FILE_MANIFEST.json'),'Fresh manifest bytes differ')
verified=p.verify_archive_payload(fresh)
with zipfile.ZipFile(archive) as z:
    p.require(z.testzip() is None,'ZIP CRC failed')
    p.require(len(z.namelist())==len(set(z.namelist()))==verified['files']+1,'ZIP file list differs')
    p.require(z.read('FILE_MANIFEST.json')==(payload/'FILE_MANIFEST.json').read_bytes(),'ZIP manifest bytes differ')
paths=p.read(payload/'PORTABLE_PATHS.json');audit=p.read(payload/'AUDIT.json')
for name,record in p.read(payload/'SOURCE_WHITELIST.json')['files'].items():
    for actual in [Path(paths['original_repository'])/name,payload/'source'/name,fresh/'source'/name]:
        p.require(p.file_hash(actual)==record['sha256'] and actual.stat().st_size==record['bytes'],'Source original/copy/fresh differs: '+name)
for actual in [Path(paths['original_data']),payload/'data',fresh/'data']:
    p.verify_files(actual,audit['data']['files'])
gate_files=p.files_manifest(paths['original_gate'])
for actual in [payload/paths['gate'],fresh/paths['gate']]:p.verify_files(actual,gate_files)
p.verify_files(Path(paths['original_run']),p.read(out/'ORIGINAL_RUN_BEFORE.json'))
child_code='''import json,sys
from pathlib import Path
import experiments.a_radial.system_probe.ours_package
import experiments.a_radial.system_probe.ours_timing
source=Path.cwd()/"source"
local={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if name.startswith("experiments.a_radial") and getattr(module,"__file__",None)}
assert local and all(Path(path).is_relative_to(source.resolve()) for path in local.values()),local
print(json.dumps({"imported_local_modules":local,"all_local_module_origins_in_fresh_source":True}))'''
completed=subprocess.run([sys.executable,'-B','-c',child_code],cwd=fresh,env=dict(os.environ,
    PYTHONPATH=str((fresh/'source').resolve()),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
p.require(completed.returncode==0,'Absolute-path fresh import failed: '+completed.stderr)
origins=json.loads(completed.stdout)
# Verify import did not create extra archived files or mutate source bytes.
p.require(p.files_manifest(fresh/'source')==p.files_manifest(payload/'source'),'Fresh import changed source files')
p.require(p.file_hash(archive)==archive_before,'Original ZIP modified')
record={'status':'SEALED_AFTER_VERIFIER_PATH_RECOVERY','first_failure_preserved':'FAILED.json',
    'recovery_scope':'original final import used relative PYTHONPATH under changed cwd; independent absolute-path import only',
    'recovery_source_sha256':p.file_hash(__file__),'archive':archive.name,'archive_bytes':archive.stat().st_size,
    'archive_sha256':archive_before,'ZIP_unchanged_from_original_failed_packaging':True,
    'archive_CRC_all_members_pass':True,'payload_fresh_ZIP_FILE_MANIFEST_bytes_identical':True,
    'file_manifest_sha256':p.file_hash(payload/'FILE_MANIFEST.json'),'fresh_extraction':verified,
    'source_original_payload_fresh_all_equal':True,'data_original_payload_fresh_all_equal':True,
    'GPU_gate_original_payload_fresh_all_equal':True,'original_run_unchanged':True,
    'all1834_original_files_preserved':True,'new_384_numerical_reproduction':False,'new_GPU_replay':False,
    'prior120_GPU_replay_arrays_rechecked':True,'fresh_source_import':origins,
    'elapsed_recovery_seconds':time.perf_counter()-started,
    'standalone_archive_note':'Historical package command needs an absolute --out path; verify mode remains valid. Original packaged source preserved.'}
p.write_json(out/'POSTSEAL_COPY_AUDIT.json',record)
p.write_json(out/'SEALED_AFTER_RECOVERY.json',{k:v for k,v in record.items() if k!='fresh_source_import'})
print(json.dumps({k:v for k,v in record.items() if k!='fresh_source_import'}),flush=True)

