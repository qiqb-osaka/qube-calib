import os
from typing import Generator

import pytest
from qubecalib.qubecalib import QubeCalib

CONFIG_KEYS = {
    "clockmaster_setting",
    "box_settings",
    "box_aliases",
    "port_settings",
    "target_settings",
    "relation_channel_target",
    "relation_channel_port",
}


@pytest.fixture(name="qc")
def qubecalib() -> Generator[QubeCalib, None, None]:
    """Return a QubeCalib object with a sample config file."""
    cwd = os.getcwd()  # Save the current working directory
    os.chdir(os.path.dirname(__file__))  # Change to the directory of this file
    yield QubeCalib("sample_config.json")  # Return the QubeCalib object
    os.chdir(cwd)  # Restore the current working directory


def test_test() -> None:
    assert True


def test_empty_init() -> None:
    """QubeCalib should initialize with no arguments."""
    qc = QubeCalib()
    config = qc.system_config_database.asdict()
    assert config.keys() == CONFIG_KEYS


def test_error_handling_invalid_config() -> None:
    """QubeCalib should raise FileNotFoundError for invalid config file."""
    with pytest.raises(FileNotFoundError):
        QubeCalib("nonexistent_config.json")


def test_init_with_config(qc: QubeCalib) -> None:
    """QuabeCalib should initialize with a configuration file."""
    config = qc.system_config_database.asdict()
    assert config.keys() == CONFIG_KEYS
    assert config["clockmaster_setting"] is None
    assert config["box_settings"]["riken_1-08"]["ipaddr_wss"] == "10.1.0.26"
    assert config["box_aliases"]["A1"] == "riken_1-08"
    assert config["port_settings"]["Q00"]["port"] == 5
    assert config["target_settings"]["CQ00_0"]["frequency"] == 10000000000.0
    assert ["MUX00GENCH0", "RQ00"] in config["relation_channel_target"]
    assert [
        "MUX00GENCH0",
        {"port_name": "MUX00GEN", "channel_number": 0},
    ] in config["relation_channel_port"]


def test_modify_target_frequency(qc: QubeCalib) -> None:
    """modify_target_frequency should change the frequency of a target."""
    qc.modify_target_frequency("CQ00_0", 7.5e9)
    assert qc.get_target_info("CQ00_0")["target_frequency"] == 7500000000.0


def test_define_clockmaster() -> None:
    """define_clockmaster should set the clockmaster settings."""
    qc = QubeCalib()
    ipaddr = "10.3.0.255"
    qc.define_clockmaster(ipaddr=ipaddr, reset=True)
    clockmaster_setting = qc.system_config_database.asdict()["clockmaster_setting"]
    assert clockmaster_setting["ipaddr"] == ipaddr


def test_define_box() -> None:
    """define_box should set the box settings."""
    qc = QubeCalib()
    box_name1 = "BOX_NAME_1"
    box_name2 = "BOX_NAME_2"
    ipaddr1 = "10.1.0.101"
    ipaddr2 = "10.1.0.102"
    boxtype1 = "qube-riken-a"
    boxtype2 = "qube-riken-b"
    qc.define_box(box_name=box_name1, ipaddr_wss=ipaddr1, boxtype=boxtype1)
    qc.define_box(box_name=box_name2, ipaddr_wss=ipaddr2, boxtype=boxtype2)
    box_settings = qc.system_config_database.asdict()["box_settings"]
    assert box_settings[box_name1]["ipaddr_wss"] == ipaddr1
    assert box_settings[box_name2]["ipaddr_wss"] == ipaddr2


def test_define_port() -> None:
    """define_port should set the port settings."""
    qc = QubeCalib()
    box_name_1 = "BOX_NAME_1"
    box_name_2 = "BOX_NAME_2"
    port_name1 = "PORT_NAME_1"
    port_name2 = "PORT_NAME_2"
    port_number1 = 5
    port_number2 = 6
    qc.define_port(port_name=port_name1, box_name=box_name_1, port_number=port_number1)
    qc.define_port(port_name=port_name2, box_name=box_name_2, port_number=port_number2)
    port_settings = qc.system_config_database.asdict()["port_settings"]
    assert port_settings[port_name1]["port"] == port_number1
    assert port_settings[port_name2]["port"] == port_number2


def test_define_channel() -> None:
    """define_channel should set the channel settings."""
    qc = QubeCalib()
    qc.define_port(port_name="PORT", box_name="BOX", port_number=5)
    qc.define_channel(channel_name="CHANNEL", port_name="PORT", channel_number=1)
    relation_channel_port = qc.system_config_database.asdict()["relation_channel_port"]
    assert (
        "CHANNEL",
        {"port_name": "PORT", "channel_number": 1},
    ) in relation_channel_port


def test_define_target() -> None:
    """define_target should set the target settings."""
    qc = QubeCalib()
    qc.define_target(
        target_name="TARGET",
        target_frequency=10.0e9,
        channel_name="CHANNEL",
    )
    target_settings = qc.system_config_database.asdict()["target_settings"]
    assert target_settings["TARGET"]["frequency"] == 10000000000.0
