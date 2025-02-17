from __future__ import annotations

from qubecalib.instrument.quel.quel1.tool.skew import Skew, SkewAdjust
from qubecalib.qubecalib import SystemConfigDatabase


def test_create_database() -> None:
    """create_database should create a database."""
    db = SkewAdjust(SystemConfigDatabase())
    assert isinstance(db, SkewAdjust)


def test_acquire_target(sysdb: SystemConfigDatabase) -> None:
    assert Skew.acquire_target(sysdb, ("Q73A", 12)) == "MON1"
    assert Skew.acquire_target(sysdb, ("Q73A", 1)) in ["RQ52", "RQ53", "RQ54", "RQ55"]


# def test_read_setup(system: Quel1System) -> None:
#     assert (
#         Skew._setup_monitor_port(
#             target_port=("10.1.0.73", 1),
#             monitor_port=("10.1.0.73", 12),
#             system=system,
#         )
#         == {}
#     )
