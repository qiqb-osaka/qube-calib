from __future__ import annotations

from types import MappingProxyType

import pytest
from qubecalib.task.quelware.direct.multi import (
    Action,
    BoxAction,
    BoxSetting,
    Quel1System,
)
from quel_clock_master import QuBEMasterClient
from quel_ic_config import Quel1BoxWithRawWss


def test_create_quel1system(quel1system: Quel1System) -> None:
    assert isinstance(quel1system, Quel1System)
    assert isinstance(quel1system._clockmaster, QuBEMasterClient)
    for box in quel1system._boxes.values():
        assert isinstance(box, Quel1BoxWithRawWss)


def test_build_action(quel1system: Quel1System, box_settings: list[BoxSetting]) -> None:
    a = Action.build(
        quel1system=quel1system,
        settings=box_settings,
    )
    assert isinstance(a, Action)
    assert isinstance(a._quel1system, Quel1System)
    assert isinstance(a._box_actions, MappingProxyType)
    for box in a._box_actions.values():
        assert isinstance(box, BoxAction)
    assert a._get_reference_box_name() == "10.1.0.26"


def test_action(quel1system: Quel1System, box_settings: list[BoxSetting]) -> None:
    a = Action.build(
        quel1system=quel1system,
        settings=box_settings,
    )

    with pytest.raises(ValueError) as e:
        a.emit_at()
    # assert str(e.value) == "no sysref time offset is measured"
    assert str(e.value) == "no estimated time difference is measured"

    results = a.action()
    # a.measure_timediff()
    # a26 = a._box_actions["10.1.0.26"]
    # a7 = a._box_actions["10.1.0.7"]
    # assert a7._sysref_time_offset == 0
    # assert a26._sysref_time_offset == 0
    # assert a._cap_sysref_time_offset == 0
    assert results.keys() == {("10.1.0.26", 1, 0)}
