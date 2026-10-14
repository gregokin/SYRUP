"""Input maps define prescribed vegetation; output switches are irrelevant."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from maple_syrup.sediment_event import SedimentEventError
from maple_syrup.windowed_experiment import prescribed_vegetation


def test_bound_nonzero_map_is_used_when_veg_output_is_false(tmp_path):
    path=tmp_path/'cover.asc'
    path.write_text('ncols 3\nnrows 4\nxllcorner 0\nyllcorner 0\ncellsize 1\nnodata_value -9999\n'+' '.join(map(str,[0,0,0,0,25,0,0,75,0,0,0,0])))
    xml=tmp_path/'model.xml'
    xml.write_text('<mahleran_input><vegetation-cover_map value="cover.asc"/>'
                   '<mahleran_output_files><veg>false</veg></mahleran_output_files></mahleran_input>')
    v=SimpleNamespace(case_dir=tmp_path,binding={},case=SimpleNamespace(config=SimpleNamespace(geometry=SimpleNamespace(ny=2,nx=1))),fields={'active':np.ones((2,1),dtype=bool)},report={'staged_sources':{'cover.asc':{'relpath':'cover.asc','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}})
    pin=hashlib.sha256(xml.read_bytes()).hexdigest()
    got,record=prescribed_vegetation(v,xml,np,xml_sha256=pin)
    assert got.tobytes()==np.array([[.75],[.25]]).tobytes()
    assert record['original_units']=='percent' and record['growth'] is False
    path.write_text(path.read_text()+' ')
    with pytest.raises(SedimentEventError,match='map changed'):
        prescribed_vegetation(v,xml,np,xml_sha256=pin)


def test_changed_xml_is_refused_before_it_is_parsed(tmp_path,monkeypatch):
    """CW-B15: the XML is hashed against its bound digest BEFORE the parse (and after)."""
    from maple_syrup import case_import

    xml=tmp_path/'model.xml'
    xml.write_text('<mahleran_input><vegetation-cover_map value="cover.asc"/></mahleran_input>')
    pin=hashlib.sha256(xml.read_bytes()).hexdigest()
    xml.write_text(xml.read_text()+' ')
    monkeypatch.setattr(case_import,'parse_mahleran_xml',lambda *a,**k:pytest.fail('XML parsed before its hash check'))
    v=SimpleNamespace(case_dir=tmp_path,binding={},case=SimpleNamespace(config=SimpleNamespace(geometry=SimpleNamespace(ny=2,nx=1))),fields={'active':np.ones((2,1),dtype=bool)},report={'staged_sources':{}})
    with pytest.raises(SedimentEventError,match='does not hash to its bound digest'):
        prescribed_vegetation(v,xml,np,xml_sha256=pin)
    with pytest.raises(SedimentEventError,match='digest is missing'):
        prescribed_vegetation(v,xml,np,xml_sha256=None)
