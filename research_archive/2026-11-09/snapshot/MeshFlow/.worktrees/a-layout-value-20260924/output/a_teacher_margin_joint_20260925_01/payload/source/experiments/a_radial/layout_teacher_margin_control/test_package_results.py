"""Small immutable archive/path contract tests, without geometry or training."""
import hashlib
from pathlib import Path
import zipfile
import pytest
from . import package_results as pack


def test_inside_rejects_parent_escape(tmp_path):
    assert pack.inside(tmp_path,'folder/a')==tmp_path/'folder/a'
    with pytest.raises(ValueError):pack.inside(tmp_path,'../outside')
    with pytest.raises(ValueError):pack.inside(tmp_path,str(tmp_path.parent/'outside'))


def test_verify_files_detects_extra_missing_and_changed_bytes(tmp_path):
    (tmp_path/'a').write_bytes(b'a');expected=pack.inventory(tmp_path)
    assert pack.verify_files(tmp_path,expected)==1
    (tmp_path/'b').write_bytes(b'b')
    with pytest.raises(ValueError):pack.verify_files(tmp_path,expected)
    (tmp_path/'b').unlink();(tmp_path/'a').write_bytes(b'c')
    with pytest.raises(ValueError):pack.verify_files(tmp_path,expected)
    (tmp_path/'a').unlink()
    with pytest.raises(ValueError):pack.verify_files(tmp_path,expected)


def test_copy_binding_is_original_copied_hash_not_just_presence(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'a');target=tmp_path/'copy'
    digest=hashlib.sha256(b'a').hexdigest();pack.copy_checked(source,target,digest)
    assert target.read_bytes()==b'a'
    with pytest.raises(ValueError):pack.copy_checked(source,target,digest)
    with pytest.raises(ValueError):pack.copy_checked(source,tmp_path/'bad',hashlib.sha256(b'b').hexdigest())
    assert (tmp_path/'bad').read_bytes()==b'a' # failed output stays visible


def test_inventory_rejects_symlink(tmp_path):
    (tmp_path/'a').write_bytes(b'a');(tmp_path/'linked').symlink_to(tmp_path/'a')
    with pytest.raises(ValueError,match='Symlink'):pack.inventory(tmp_path)


def test_zip_exact_members_hash_and_path_checks(tmp_path):
    valid=tmp_path/'valid.zip';digest=hashlib.sha256(b'a').hexdigest()
    with zipfile.ZipFile(valid,'x') as z:z.writestr('a',b'a')
    out=pack.extract_checked(valid,tmp_path/'valid_out',{'a':digest});assert (out/'a').read_bytes()==b'a'
    with pytest.raises(FileExistsError):pack.extract_checked(valid,out,{'a':digest})
    with pytest.raises(ValueError,match='member hash'):pack.extract_checked(valid,tmp_path/'wrong_hash',{'a':'0'*64})
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'x') as z:z.writestr('../escaped',b'a')
    with pytest.raises(ValueError,match='escapes'):pack.extract_checked(bad,tmp_path/'bad_out',{'../escaped':digest})
    assert not (tmp_path/'escaped').exists()
