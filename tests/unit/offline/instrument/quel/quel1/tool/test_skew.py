from __future__ import annotations

import yaml
from qubecalib.instrument.quel.quel1.tool.skew import Skew, SkewAdjust, SkewSetting
from qubecalib.qubecalib import SystemConfigDatabase


def test_create_database() -> None:
    """create_database should create a database."""
    db = SkewAdjust(SystemConfigDatabase())
    assert isinstance(db, SkewAdjust)


def test_load_skew_setting() -> None:
    """load_skew_setting should load the skew setting."""
    with open("skew_work.yaml", "r") as f:
        config = yaml.safe_load(f)
    skew_setting = SkewSetting.load(config)
    assert isinstance(skew_setting, SkewSetting)
    assert skew_setting.monitor_port == ("Q73A", 12)


def test_acquire_freq_setting() -> None:
    """acquire_freq_setting should acquire the frequency setting."""
    setting = [
        Skew.acquire_freq_setting(f) for f in [6.805266, 10, 10.124, 10.125, 10.5]
    ]
    assert [s["sideband"] for s in setting] == ["L", "U", "U", "U", "U"]
    assert [s["lo_freq"] for s in setting] == [9.0, 8.0, 8.0, 8.0, 8.5]
    assert [s["cnco_freq"] for s in setting] == [2.125, 2.0, 2.0, 2.125, 2.0]
    assert [s["fnco_freq"] for s in setting] == len(setting) * [0]
