import pytest
from e7awgsw import CaptureParam, WaveSequence
from qubecalib.task.quelware.direct import multi, single
from quel_clock_master import QuBEMasterClient
from quel_ic_config import Quel1BoxType, Quel1BoxWithRawWss

MASTER_IPADDR = "10.3.0.255"
# 64QMUX10 R26A, U7B
BOX_KEYS = ["10.1.0.26", "10.1.0.7"]
BOX_TYPES = [Quel1BoxType.QuBE_RIKEN_TypeA, Quel1BoxType.QuBE_OU_TypeB]


@pytest.fixture
def master() -> QuBEMasterClient:
    return QuBEMasterClient(master_ipaddr=MASTER_IPADDR)


@pytest.fixture
def box() -> Quel1BoxWithRawWss:
    return Quel1BoxWithRawWss.create(
        ipaddr_wss=list(BOX_KEYS)[0],
        boxtype=list(BOX_TYPES)[0],
        # QuBE_OU_TypeA = ("qube", "ou-type-a")
        # QuBE_OU_TypeB = ("qube", "ou-type-b")
        # QuBE_RIKEN_TypeA = ("qube", "riken-type-a")
        # QuBE_RIKEN_TypeB = ("qube", "riken-type-b")
        # ipaddr_wss=os.environ["BOX"],
        # boxtype=os.environ["BOXTYPE"],
        # 64QMUX00 U15A, R21B
        # ipaddr_wss="10.1.0.15",
        # boxtype=Quel1BoxType.QuBE_OU_TypeA,
        # 16QMUX00 R20A
        # ipaddr_wss="10.1.0.20",
        # boxtype=Quel1BoxType.QuBE_RIKEN_TypeA,
        # 16QMUX03 R28A
        # ipaddr_wss="10.1.0.28",
        # boxtype=Quel1BoxType.QuBE_RIKEN_TypeA,
    )


@pytest.fixture
def quel1system() -> multi.Quel1System:
    return multi.Quel1System.create(
        clockmaster=QuBEMasterClient(
            master_ipaddr=MASTER_IPADDR,
        ),
        boxes=[
            Quel1BoxWithRawWss.create(
                ipaddr_wss=list(BOX_KEYS)[0],
                boxtype=list(BOX_TYPES)[0],
            ),
            Quel1BoxWithRawWss.create(
                ipaddr_wss=list(BOX_KEYS)[1],
                boxtype=list(BOX_TYPES)[1],
            ),
        ],
    )


PERIOD = 1280 * 128

w7 = WaveSequence(
    num_wait_words=0,  # words
    num_repeats=1000,  # times
)
w7.add_chunk(
    # iq_samples must be a multiple of 64
    iq_samples := int(256 // 64) * (32 * 4 * [(32767, 0)] + 32 * 4 * [(0, 0)]),
    num_blank_words=PERIOD - len(iq_samples) // 4,
    num_repeats=1,
)

w26 = WaveSequence(
    num_wait_words=0,  # words
    num_repeats=1000,  # times
)
w26.add_chunk(
    # iq_samples must be a multiple of 64
    iq_samples := int(256 // 64)
    * (42 * 4 * [(0, 0)] + 10 * 4 * [(32767, 0)] + 12 * 4 * [(0, 0)]),
    num_blank_words=PERIOD - len(iq_samples) // 4,
    num_repeats=1,
)

c = CaptureParam()
c.capture_delay = 6 * 16  # words
c.num_integ_sections = 1
c.add_sum_section(
    num_words := 2 * 256,
    num_post_blank_words=PERIOD - num_words,
)


@pytest.fixture
def box_settings() -> list[multi.BoxSetting]:
    return [
        multi.BoxSetting(
            name=list(BOX_KEYS)[0],
            settings=[
                single.AwgSetting(awg=single.AwgId(port=0, channel=0), wseq=w26),
                single.RunitSetting(
                    runit=single.RunitId(port=1, runit=0), cprm=CaptureParam()
                ),
                single.TriggerSetting(
                    triggerd_port=1, trigger_awg=single.AwgId(port=0, channel=0)
                ),
            ],
        ),
        multi.BoxSetting(
            name=list(BOX_KEYS)[1],
            settings=[
                single.AwgSetting(awg=single.AwgId(port=0, channel=0), wseq=w7),
            ],
        ),
    ]
