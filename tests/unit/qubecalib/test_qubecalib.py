import os
import pytest
from qubecalib import QubeCalib


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
def qubecalib():
    """Return a QubeCalib object with a sample config file."""
    cwd = os.getcwd()  # Save the current working directory
    os.chdir(os.path.dirname(__file__))  # Change to the directory of this file
    yield QubeCalib("config.json")  # Return the QubeCalib object
    os.chdir(cwd)  # Restore the current working directory


def test_empty_init():
    """QubeCalib should initialize with no arguments."""
    qc = QubeCalib()
    config = qc.system_config_database.asdict()
    assert config.keys() == CONFIG_KEYS


def test_init_with_config(qc):
    """QuabeCalib should initialize with a configuration file."""
    config = qc.system_config_database.asdict()
    assert config.keys() == CONFIG_KEYS
    assert config["clockmaster_setting"] is None
    assert config["box_settings"]["riken_1-08"]["ipaddr_wss"] == "10.1.0.26"
    assert config["box_aliases"]["A1"] == "riken_1-08"
    assert config["port_settings"]["MUX00GEN"]["port_name"] == "MUX00GEN"
    assert config["target_settings"]["RQ00"]["frequency"] == 10000000000.0
    assert config["relation_channel_target"][0][0] == "MUX00GENCH0"
    assert config["relation_channel_port"][0][0] == "MUX00GENCH0"


def test_error_handling_invalid_config():
    """QubeCalib should raise FileNotFoundError for invalid config file."""
    with pytest.raises(FileNotFoundError):
        QubeCalib("nonexistent_config.json")


def test_modify_target_frequency(qc):
    """modify_target_frequency should change the frequency of a target."""
    qc.modify_target_frequency("RQ00", 7.5e9)
    assert qc.get_target_info("RQ00")["target_frequency"] == 7500000000.0
