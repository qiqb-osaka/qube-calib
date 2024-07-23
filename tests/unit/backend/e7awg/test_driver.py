from e7awgsw import CaptureModule, CaptureParam, CaptureUnit
from qubecalib.backend.e7awg.driver import CapuSetting


def test_capu_setting() -> None:
    param = CaptureParam()
    unit_to_mod = {
        CaptureUnit.U0: CaptureModule.U0,
        CaptureUnit.U1: CaptureModule.U0,
        CaptureUnit.U2: CaptureModule.U0,
        CaptureUnit.U3: CaptureModule.U0,
        CaptureUnit.U4: CaptureModule.U1,
        CaptureUnit.U5: CaptureModule.U1,
        CaptureUnit.U6: CaptureModule.U1,
        CaptureUnit.U7: CaptureModule.U1,
        CaptureUnit.U8: CaptureModule.U2,
        CaptureUnit.U9: CaptureModule.U3,
    }
    for unit, module in unit_to_mod.items():
        setting = CapuSetting(capu=unit, capprm=param)
        assert setting.capm == module
