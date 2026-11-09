import hashlib,zipfile
from pathlib import Path
import pytest
from . import warm_start_package as p


@pytest.mark.parametrize('name',['../escape','/absolute','a/../../escape','C:/windows','a\\b'])
def test_inside_rejects_nonportable_or_escaping_name(tmp_path,name):
    with pytest.raises(ValueError):p.inside(tmp_path,name)


def test_fixed_slots_retain_errors_and_reject_missing_duplicate():
    protocol={'uids':['a','b','c','d'],'models':list(p.NEW_MODELS)}
    attempts=[{'uid':u,'model':m,'rows':[{'uid':u,'model':m,'kind':k,'status':'failed'} for k in p.KINDS]} for u in protocol['uids'] for m in p.NEW_MODELS]
    assert len(p.check_slots(attempts,protocol,p.NEW_MODELS))==48
    with pytest.raises(ValueError):p.check_slots(attempts[:-1],protocol,p.NEW_MODELS)
    attempts[0]['rows'][-1]=attempts[0]['rows'][0]
    with pytest.raises(ValueError):p.check_slots(attempts,protocol,p.NEW_MODELS)


def test_copy_and_manifest_reject_source_or_member_change(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'one');digest=p.file_hash(source)
    target=tmp_path/'payload/file';p.copy_checked(source,target,digest)
    manifest=p.inventory(target.parent);p.verify_files(target.parent,manifest)
    target.write_bytes(b'two')
    with pytest.raises(ValueError):p.verify_files(target.parent,manifest)
    with pytest.raises(ValueError):p.copy_checked(source,tmp_path/'another','wrong')


def test_zip_extraction_binds_member_set_crc_sha_and_no_overwrite(tmp_path):
    archive=tmp_path/'artifact.zip';content=b'evidence';digest=hashlib.sha256(content).hexdigest()
    with zipfile.ZipFile(archive,'x') as z:z.writestr('dir/file',content)
    fresh=p.extract_checked(archive,tmp_path/'fresh',{'dir/file':digest})
    assert (fresh/'dir/file').read_bytes()==content
    with pytest.raises(FileExistsError):p.extract_checked(archive,fresh,{'dir/file':digest})
    with pytest.raises(ValueError):p.extract_checked(archive,tmp_path/'bad',{'dir/file':'wrong'})


def test_explicit_source_whitelist_excludes_parallel_research():
    assert 'system_probe/warm_start.py' in p.SOURCE_NAMES and 'system_probe/warm_start_controls.py' in p.SOURCE_NAMES
    assert 'assets/original_A_10.pt' in p.SOURCE_NAMES
    assert not any(name.startswith(('layout_margin_single/','layout_balanced/','fm_tail/','system_formal/')) for name in p.SOURCE_NAMES)


def test_identical_content_and_basename_from_different_sources_remain_distinct(tmp_path):
    a=tmp_path/'long'/'same.npz';b=tmp_path/'balanced'/'same.npz'
    assert p.bound_name(a,'same-digest')!=p.bound_name(b,'same-digest')
    assert p.bound_name(a,'same-digest')==p.bound_name(a,'same-digest')
