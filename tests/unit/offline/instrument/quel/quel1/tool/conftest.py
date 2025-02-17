import pytest
from qubecalib.instrument.quel.quel1.driver import Quel1System
from qubecalib.qubecalib import SystemConfigDatabase
from quel_clock_master import QuBEMasterClient
from quel_ic_config import Quel1BoxType


@pytest.fixture
def sysdb() -> SystemConfigDatabase:
    db = SystemConfigDatabase()

    db.define_port(
        port_name="Q73A.MON1.IN",
        box_name="Q73A",
        port_number=12,
    )
    db.define_channel(
        channel_name="Q73A.MON1.IN0",
        port_name="Q73A.MON1.IN",
        channel_number=0,
        ndelay_or_nwait=7,
    )
    db._relation_channel_target.append(("Q73A.MON1.IN0", "MON1"))

    db.define_port(
        port_name="Q73A.READ0.OUT",
        box_name="Q73A",
        port_number=1,
    )
    db.define_channel(
        channel_name="Q73A.READ0.OUT0",
        port_name="Q73A.READ0.OUT",
        channel_number=0,
        ndelay_or_nwait=7,
    )
    db._relation_channel_target.append(("Q73A.READ0.OUT0", "RQ52"))
    db._relation_channel_target.append(("Q73A.READ0.OUT0", "RQ53"))
    db._relation_channel_target.append(("Q73A.READ0.OUT0", "RQ54"))
    db._relation_channel_target.append(("Q73A.READ0.OUT0", "RQ55"))
    return db


@pytest.fixture
def system() -> Quel1System:
    system = Quel1System.create(
        clockmaster=QuBEMasterClient("10.3.0.255"),
        boxes=[
            SystemConfigDatabase._create_box(
                ipaddr_wss="10.1.0.73",
                boxtype=Quel1BoxType.QuEL1_TypeA,
                auto_relinkup=False,
            ),
            SystemConfigDatabase._create_box(
                ipaddr_wss="10.1.0.2",
                boxtype=Quel1BoxType.QuEL1_TypeA,
                auto_relinkup=False,
            ),
        ],
    )
    return system


# dict(
#     port=dict(
#         port_name="Q73A.MON1.IN",
#         box_name="Q73A",
#         port_number=12,
#     ),
#     channels=dict(
#         channel=dict(
#             channel_name="Q73A.MON1.IN0",
#             channel_number=0,
#             ndelay_or_nwait=7,
#         ),
#         targets=[
#             "MON1",
#         ],
#     ),
# )
