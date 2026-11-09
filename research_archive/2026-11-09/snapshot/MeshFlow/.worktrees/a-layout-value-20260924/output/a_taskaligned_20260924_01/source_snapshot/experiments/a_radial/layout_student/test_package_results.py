"""Historical source teacher dtype is preserved, never compared ambiguously."""
import numpy as np
import pytest
from ..io_utils import array_hash
from .package_results import verify_teacher_arrays


def fixture_arrays():
    v=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=np.float64)
    f=np.array([[0,1,2]],dtype=np.int32)
    source={'vertices':v.copy(),'faces':f.astype(np.int64),'u':v.copy()}
    reference={'vertices':v.copy(),'faces':f.copy(),'u':v.copy()}
    return source,reference,f


def check(source,reference,f,expected_source_hash=None):
    return verify_teacher_arrays(source,reference,f,
        array_hash(source['vertices'],source['faces']) if expected_source_hash is None else expected_source_hash,
        array_hash(reference['vertices'],f),array_hash(reference['vertices'],reference['faces']))


def test_teacher_dtype_difference_preserves_both_typed_hashes():
    source,reference,f=fixture_arrays();report=check(source,reference,f)
    assert report['face_dtype_differs']
    assert report['source_mesh_hash']!=report['reference_logical_mesh_hash']
    assert report['vertices_exactly_equal'] and report['directions_exactly_equal']
    source['faces']=f.copy();report=check(source,reference,f)
    assert not report['face_dtype_differs']
    assert report['source_mesh_hash']==report['reference_logical_mesh_hash']


@pytest.mark.parametrize('field',['vertices','u','faces'])
def test_teacher_reference_values_cannot_change_even_with_recomputed_hashes(field):
    source,reference,f=fixture_arrays()
    reference[field][0,0]+=1 if field=='faces' else 1e-8
    with pytest.raises(ValueError,match='changed'):check(source,reference,f)


def test_teacher_original_source_hash_must_match():
    source,reference,f=fixture_arrays()
    with pytest.raises(ValueError,match='source array hash mismatch'):
        check(source,reference,f,'0'*64)
