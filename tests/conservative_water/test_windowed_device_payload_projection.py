"""Pure closed-Source payload arithmetic; no Native/provider/model calls.

Only fixed Source metadata is temporarily changed by negative cases. These
tests create no bed, arrays, budget, Native capability or observed resource law.
"""
import copy

import pytest

from maple_syrup import windowed_shared_bed as generic
from maple_syrup import windowed_shared_bed_device as device
from maple_syrup.sediment_bed import BedIntegrationError


def test_projection_keeps_three_request_banks_and_real_device_report_pairs():
    before = generic._positive_interval_units()
    result = device._qualification_visible_device_payload(271839)
    assert type(result) is int
    assert result == 458061696 + 2949120 == 461010816
    assert before == generic._positive_interval_units() == (271839, 297579)


@pytest.mark.parametrize('count', [None, False, True, -1, 0, 271838, 271840,
                                  1 << 127, 271839.0, '271839'])
def test_projection_refuses_any_noncanonical_request_count(count):
    with pytest.raises(BedIntegrationError, match='payload_request_count'):
        device._qualification_visible_device_payload(count)


def test_projection_has_no_configurable_fee_bank_or_report_multiplier():
    with pytest.raises(TypeError):
        device._qualification_visible_device_payload(271839, banks=1)
    with pytest.raises(TypeError):
        device._qualification_visible_device_payload(271839, max_report_rows=1)
    with pytest.raises(TypeError):
        device._qualification_visible_device_payload()


@pytest.mark.parametrize('name', ['_POSITIVE_WORD', '_POSITIVE_HEADER', '_POSITIVE_INTERFACE',
    '_POSITIVE_GRID', '_POSITIVE_FLAG', '_POSITIVE_REPORT_ROWS', '_POSITIVE_PLANNER_EXTENT'])
def test_projection_refuses_changed_fixed_Source_controls(monkeypatch, name):
    monkeypatch.setattr(generic, name, getattr(generic, name)+1)
    with pytest.raises(BedIntegrationError, match='payload_Source_controls'):
        device._qualification_visible_device_payload(271839)


def test_projection_refuses_foreign_metadata_before_equality_or_iteration(monkeypatch):
    class Foreign:
        def __getattribute__(self, name):
            raise AssertionError('foreign attribute hook: '+name)
        def __eq__(self, other):
            raise AssertionError('foreign equality hook')
        def __iter__(self):
            raise AssertionError('foreign iteration hook')

    with pytest.raises(BedIntegrationError, match='payload_request_count'):
        device._qualification_visible_device_payload(Foreign())
    with monkeypatch.context() as patch:
        patch.setattr(generic, '_POSITIVE_GRID', Foreign())
        with pytest.raises(BedIntegrationError, match='payload_Source_controls'):
            device._qualification_visible_device_payload(271839)
    with monkeypatch.context() as patch:
        patch.setattr(generic, '_VISIBLE_REQUEST_CENSUS', Foreign())
        with pytest.raises(BedIntegrationError, match='payload_Source_census'):
            device._qualification_visible_device_payload(271839)
    with monkeypatch.context() as patch:
        patch.setattr(generic, '_POSITIVE_CENSUS_NAMES', (Foreign(),)*11)
        with pytest.raises(BedIntegrationError, match='payload_Source_names'):
            device._qualification_visible_device_payload(271839)


def test_projection_refuses_missing_extra_or_reordered_Source_names(monkeypatch):
    for names in (generic._POSITIVE_CENSUS_NAMES[:-1],
                  tuple(reversed(generic._POSITIVE_CENSUS_NAMES))):
        with monkeypatch.context() as patch:
            patch.setattr(generic, '_POSITIVE_CENSUS_NAMES', names)
            with pytest.raises(BedIntegrationError, match='payload_Source_names'):
                device._qualification_visible_device_payload(271839)
    for mode in ('missing', 'extra'):
        census = copy.deepcopy(generic._VISIBLE_REQUEST_CENSUS)
        if mode == 'missing':
            census.pop('quiet')
        else:
            census['extra'] = census['quiet']
        with monkeypatch.context() as patch:
            patch.setattr(generic, '_VISIBLE_REQUEST_CENSUS', census)
            with pytest.raises(BedIntegrationError, match='payload_Source_census'):
                device._qualification_visible_device_payload(271839)


def test_projection_checks_the_complete_entry_HOST_polynomial(monkeypatch):
    census = copy.deepcopy(generic._VISIBLE_REQUEST_CENSUS)
    side = ((830, (0, 0, 0, 0, 0, 0, 0)), (18, (0, 0, 0, 0, 0, 0, 1)))
    census['complete_entry'] = (side, side)
    monkeypatch.setattr(generic, '_VISIBLE_REQUEST_CENSUS', census)
    with pytest.raises(BedIntegrationError, match='HOST_boundary_polynomial'):
        device._qualification_visible_device_payload(271839)


@pytest.mark.parametrize('row', [(True, (0,)*7), (1 << 127, (0,)*7),
    (1727.0, (0,)*7), (1727, [0]*7), (1727, (False,)*7),
    (1727, (0,)*8), (1727, (3,)*7), [1727, (0,)*7]])
def test_projection_refuses_nonliteral_or_out_of_domain_polynomial_rows(monkeypatch, row):
    census = copy.deepcopy(generic._VISIBLE_REQUEST_CENSUS)
    side = (row, census['initial'][0][1])
    census['initial'] = (side, side)
    monkeypatch.setattr(generic, '_VISIBLE_REQUEST_CENSUS', census)
    with pytest.raises(BedIntegrationError, match='payload_Source_census'):
        device._qualification_visible_device_payload(271839)


def test_projection_refuses_balanced_mutation_even_when_old_totals_match(monkeypatch):
    census = copy.deepcopy(generic._VISIBLE_REQUEST_CENSUS)
    for name, delta in (('initial', 1), ('terminal', -1)):
        census[name] = tuple(((side[0][0]+delta, side[0][1]), *side[1:]) for side in census[name])
    monkeypatch.setattr(generic, '_VISIBLE_REQUEST_CENSUS', census)
    # The old total guard alone cannot distinguish these altered laws.
    assert generic._positive_interval_units() == (271839, 297579)
    with pytest.raises(BedIntegrationError, match='payload_Source_census_hash'):
        device._qualification_visible_device_payload(271839)
