from __future__ import annotations

from qubecalib.instrument.quel.quel1.tool import create_sysdb_items_quel1_riken8
from qubecalib.sysconfdb import SystemConfigDatabase


def test_define_port() -> None:
    """define_port should set the port settings."""
    # Given
    sysdb = SystemConfigDatabase()
    kwargs = dict(box_name="Q132SE8", ipaddr_wss="10.1.0.132")

    # When
    create_sysdb_items_quel1_riken8(sysdb, **kwargs)

    # Then
    def is_equal(a: dict, b: dict) -> bool:
        """Check if two dictionaries are equal."""
        if a.keys() != b.keys():
            return False
        for key in a.keys():
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                if not is_equal(a[key], b[key]):
                    return False
            elif isinstance(a[key], list) and isinstance(b[key], list):
                if len(a[key]) != len(b[key]):
                    return False
                for i in range(len(a[key])):
                    if not is_equal(a[key][i], b[key][i]):
                        return False
            elif a[key] != b[key]:
                return False
        return True

    band_settings: dict[str, list[dict[str, int | str]]] = {}
    for band_name, setting in sysdb._relation_channel_port:
        if band_name not in band_settings:
            band_settings[band_name] = []
        band_settings[band_name].append(setting)

    s = band_settings
    pname = "Q132SE8.READ.IN"
    assert pname in sysdb._port_settings
    assert sysdb._port_settings[pname].port == 0
    assert "Q132SE8.READ.IN4" not in band_settings
    for i in range(3):
        cname = f"{pname}{i}"
        assert len(band_settings[cname]) == 1
        assert is_equal(s[cname][0], {"channel_number": i, "port_name": pname})

    pname = "Q132SE8.READ.OUT"
    assert pname in sysdb._port_settings
    assert sysdb._port_settings[pname].port == 1
    assert "Q132SE8.READ.OUT1" not in band_settings
    cname = "Q132SE8.READ.OUT0"
    assert len(band_settings[cname]) == 1
    assert is_equal(s[cname][0], {"channel_number": 0, "port_name": pname})

    pname = "Q132SE8.READ.FOGI.OUT"
    assert pname in sysdb._port_settings
    assert sysdb._port_settings[pname].port == (1, 1)
    assert "Q132SE8.READ.FOGI.OUT1" not in band_settings
    cname = "Q132SE8.READ.FOGI.OUT0"
    assert len(band_settings[cname]) == 1
    assert is_equal(s[cname][0], {"channel_number": 0, "port_name": pname})

    pname = "Q132SE8.CTRLX"
    assert pname in sysdb._port_settings
    assert sysdb._port_settings[pname].port == 3
    assert "Q132SE8.CTRLX.CH3" not in band_settings
    for i in range(3):
        cname = f"{pname}.CH{i}"
        assert len(band_settings[cname]) == 1
        assert is_equal(s[cname][0], {"channel_number": i, "port_name": pname})

    pname = "Q132SE8.CTRL0"
    assert pname in sysdb._port_settings
    assert sysdb._port_settings[pname].port == 6
    assert "Q132SE8.CTRL0.CH3" not in band_settings
    for i in range(3):
        cname = f"{pname}.CH{i}"
        assert len(band_settings[cname]) == 1
        assert is_equal(s[cname][0], {"channel_number": i, "port_name": pname})

    pname = "Q132SE8.CTRL1"
    assert pname in sysdb._port_settings
    assert sysdb._port_settings[pname].port == 7
    assert "Q132SE8.CTRL1.CH3" not in band_settings
    for i in range(3):
        cname = f"{pname}.CH{i}"
        assert len(band_settings[cname]) == 1
        assert is_equal(s[cname][0], {"channel_number": i, "port_name": pname})

    pname = "Q132SE8.CTRL2"
    assert pname in sysdb._port_settings
    assert sysdb._port_settings[pname].port == 8
    assert "Q132SE8.CTRL2.CH1" not in band_settings
    for i in range(1):
        cname = f"{pname}.CH{i}"
        assert len(band_settings[cname]) == 1
        assert is_equal(s[cname][0], {"channel_number": i, "port_name": pname})

    pname = "Q132SE8.CTRL3"
    assert pname in sysdb._port_settings
    assert sysdb._port_settings[pname].port == 9
    assert "Q132SE8.CTRL3.CH1" not in band_settings
    for i in range(1):
        cname = f"{pname}.CH{i}"
        assert len(band_settings[cname]) == 1
        assert is_equal(s[cname][0], {"channel_number": i, "port_name": pname})

    pname = "Q132SE8.PUMP"
    assert pname in sysdb._port_settings
    assert sysdb._port_settings[pname].port == 2
    assert "Q132SE8.PUMP.CH3" not in band_settings
    for i in range(3):
        cname = f"{pname}.CH{i}"
        assert len(band_settings[cname]) == 1
        assert is_equal(s[cname][0], {"channel_number": i, "port_name": pname})

    pname = "Q132SE8.MNTR0.IN"
    assert pname in sysdb._port_settings
    assert sysdb._port_settings[pname].port == 4
    assert "Q132SE8.MNTR0.IN1" not in band_settings
    cname = "Q132SE8.MNTR0.IN0"
    assert len(band_settings[cname]) == 1
    assert is_equal(s[cname][0], {"channel_number": 0, "port_name": pname})

    pname = "Q132SE8.MNTR1.IN"
    assert pname in sysdb._port_settings
    assert sysdb._port_settings[pname].port == 10
    assert "Q132SE8.MNTR1.IN1" not in band_settings
    cname = "Q132SE8.MNTR1.IN0"
    assert len(band_settings[cname]) == 1
    assert is_equal(s[cname][0], {"channel_number": 0, "port_name": pname})
