import pytest
from quel_ic_config import Quel1BoxType, Quel1BoxWithRawWss


@pytest.fixture
def box() -> Quel1BoxWithRawWss:
    return Quel1BoxWithRawWss.create(
        # QuBE_OU_TypeA = ("qube", "ou-type-a")
        # QuBE_OU_TypeB = ("qube", "ou-type-b")
        # QuBE_RIKEN_TypeA = ("qube", "riken-type-a")
        # QuBE_RIKEN_TypeB = ("qube", "riken-type-b")
        # ipaddr_wss=os.environ["BOX"],
        # boxtype=os.environ["BOXTYPE"],
        # 64QMUX00
        ipaddr_wss="10.1.0.15",
        boxtype=Quel1BoxType.QuBE_OU_TypeA,
        # 16QMUX00
        # ipaddr_wss="10.1.0.20",
        # boxtype=Quel1BoxType.QuBE_RIKEN_TypeA,
        # 16QMUX02
        # ipaddr_wss="10.1.0.28",
        # boxtype=Quel1BoxType.QuBE_RIKEN_TypeA,
    )
