from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from typing import Collection, Dict, Final, Sequence, Set, Union

import yaml
from quel_ic_config import (
    # Quel1AnyBoxConfigSubsystem,
    Quel1BoxType,
    Quel1ConfigOption,
    # Quel1E7ResourceMapper,
    Quel1Feature,
)
from quel_ic_config.quel1_config_subsystem import (
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1NecConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
)

# Quel1seRiken8ConfigSubsystem,
from quel_ic_config_utils.e7resource_mapper import Quel1E7ResourceMapper
from quel_ic_config_utils.e7workaround import E7FwLifeStage, E7FwType
from quel_ic_config_utils.linkupper import LinkupFpgaMxfe
from quel_ic_config_utils.quel1_wave_subsystem import (
    Quel1WaveSubsystem,
)

from .rc import __running_config__ as rc
from .units import GHz

logger = logging.getLogger(__name__)

Quel1AnyBoxConfigSubsystem = Union[
    QubeOuTypeAConfigSubsystem,
    QubeOuTypeBConfigSubsystem,
    Quel1TypeAConfigSubsystem,
    Quel1TypeBConfigSubsystem,
    Quel1NecConfigSubsystem,
    # Quel1seRiken8ConfigSubsystem,
]

YamlBoxTypeMapper = {
    "quel-1": {
        "A": Quel1BoxType.QuEL1_TypeA,
        "B": Quel1BoxType.QuEL1_TypeB,
    },
    "qube_ou": {
        "A": Quel1BoxType.QuBE_OU_TypeA,
        "B": Quel1BoxType.QuBE_OU_TypeB,
    },
    "qube_riken": {
        "A": Quel1BoxType.QuBE_RIKEN_TypeA,
        "B": Quel1BoxType.QuBE_RIKEN_TypeB,
    },
}


def listup_configfiles() -> list[str]:
    """ClassicQube の yml ファイル

    Returns:
        list[str]: 使用可能な yml ファイル名
    """
    # TODO: os.getcwd() の中にある yaml もロードチェック後にリストに加えるように機能強化したい
    return [
        f
        for f in os.listdir(rc.path_to_config)
        if os.path.isfile(os.path.join(rc.path_to_config, f))
    ]


@dataclass(frozen=True)
class BoxInfo:
    pass


@dataclass(frozen=True)
class ClassicQubeInfo(BoxInfo):
    """Individual information for ClassicQube"""

    type: str = ""
    bitfile: str = ""
    macfpga: str = ""
    maclsi: str = ""
    adapter_au50: str = ""
    ipfpga: IPv4Address | IPv6Address = ip_address("10.1.0.1")
    iplsi: IPv4Address | IPv6Address = ip_address("10.5.0.1")
    ipmulti: IPv4Address | IPv6Address = ip_address("10.2.0.1")
    confpath: os.PathLike = rc.path_to_config


@dataclass(frozen=True)
class QcBoxInfo(BoxInfo):
    """Individual information for SimpleBox"""

    ipaddr_css: IPv4Address | IPv6Address = ip_address("10.1.0.1")
    ipaddr_wss: IPv4Address | IPv6Address = ip_address("10.5.0.1")
    ipaddr_sss: IPv4Address | IPv6Address = ip_address("10.2.0.1")
    confpath: os.PathLike = rc.path_to_config
    boxtype: Quel1BoxType = Quel1BoxType.QuBE_OU_TypeA


@dataclass(frozen=True)
class SimpleBoxInfo(BoxInfo):
    """Individual information for SimpleBox"""

    boxtype: Quel1BoxType = Quel1BoxType.QuEL1_TypeA


# class ConfigLoader:
#     def __init__(self, config_path: ConfigPath):
#         p = config_path.path
#         with open(p, "rb") as f:
#             self.content = yaml.safe_load(f)

#     @abstractmethod
#     def get_box(self) -> BoxInfo:
#         return BoxInfo()


# class ClassicQubeConfigLoader(ConfigLoader):
#     def get_box(self) -> ClassicQubeInfo:
#         kw = {
#             k: ip_address(v) if k == "ipfpga" or k == "iplsi" else v
#             for k, v in self.content.items()
#         }
#         if "ipmulti" not in kw:
#             kw["ipmulti"] = kw["ipfpga"] + 0x10000
#         return ClassicQubeInfo(**kw)


# class QcBoxConfigLoader:
#     def get_box(self) -> ClassicQubeInfo:
#         kw = {
#             k: ip_address(v) if k == "ipfpga" or k == "iplsi" else v
#             for k, v in self.content.items()
#         }
#         if "ipmulti" not in kw:
#             kw["ipmulti"] = kw["ipfpga"] + 0x10000
#         return ClassicQubeInfo(**kw)


class ConfigPath:
    # def __new__(cls, *args: any):
    #     cls = WindowsPath if os.name == "nt" else PosixPath
    #     return cls._from_parts(args)

    def __init__(self, config_path: os.PathLike):
        self._path = get_absolute_path_to_config(config_path)

    @property
    def path(self) -> Path:
        return self._path

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self) -> str:
        return self.__str__()


def get_absolute_path_to_config(config_path: os.PathLike) -> Path:
    """basename で指定されたファイルのフルパスを返す. ipynbを実行したディレクトリに
    basename のファイルが存在すればそのフルパスを，そうでなければ dir 内に
    存在するかを確認してそのフルパスを，存在しなければ FileNotFoundError を raise する.

    Args:
        basename (str): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        Path: _description_
    """
    path = Path(config_path)
    absolute = (
        Path(path.absolute()) if path.exists() else rc.path_to_config / Path(path.name)
    )
    if not absolute.exists():
        raise FileNotFoundError(f"File {absolute} not found")
    return absolute


class QcBoxFactory:
    """yaml に合わせて適切な QcBox オプジェクトを生成するファクトリクラス"""

    # def __new__(cls, *args: any): TODO: pathlib.Path と同じように一つのファクトリで種別の異なる Box を生成できたらかっこいいなぁ
    #     cls = QcBoxInfo

    def __init__(
        self, config_path: os.PathLike
    ):  # TODO:生成時にできるチェックをここでする
        self.config_path = ConfigPath(config_path)
        self.content = self.load_config(self.config_path)

    def load_config(self, config_path: ConfigPath) -> dict:
        with open(config_path.path, "rb") as f:
            c = yaml.safe_load(f)
        return c

    def parse_config(self) -> dict:
        kwmapper = {
            "ipfpga": "ipaddr_wss",
            "iplsi": "ipaddr_css",
            "ipmulti": "ipaddr_sss",
        }
        # kw = {}
        # for k, v in self.content.items():
        #     if k in kwmapper:
        #         kw[kwmapper[k]] = ip_address(v) if k == "ipfpga" or k == "iplsi" else v
        kw = {
            kwmapper[k]: ip_address(v) if k == "ipfpga" or k == "iplsi" else v
            for k, v in self.content.items()
            if k in kwmapper
        }
        if "ipaddr_sss" not in kw:
            kw["ipaddr_sss"] = kw["ipaddr_wss"] + 0x10000
        kw["confpath"] = self.config_path.path.absolute()
        boxseries = self.parse_series(self.config_path.path.name)
        kw["boxtype"] = YamlBoxTypeMapper[boxseries][self.content["type"]]
        return kw

    def parse_series(self, x: str) -> str:
        s = x.split("_")
        if s[0] == "quel-1":
            return "quel-1"
        elif s[0] == "qube":
            if s[1] == "ou":
                return "qube_ou"
            elif s[1] == "riken":
                return "qube_riken"
            else:
                raise ValueError(f"{x} is not supported.")
        else:
            raise ValueError(f"{x} is not supported.")

    def produce(self) -> QcBox:
        info = QcBoxInfo(**self.parse_config())
        boxtype = info.boxtype
        ipaddr_css = str(info.ipaddr_css)
        ipaddr_wss = str(info.ipaddr_wss)
        ipaddr_sss = str(info.ipaddr_sss)
        config_root = None
        config_options = [
            Quel1ConfigOption.USE_READ_IN_MXFE0,
            Quel1ConfigOption.USE_READ_IN_MXFE1,
        ]

        if boxtype in {
            Quel1BoxType.QuBE_OU_TypeA,
            Quel1BoxType.QuBE_RIKEN_TypeA,
            Quel1BoxType.QuEL1_TypeA,
            Quel1BoxType.QuBE_OU_TypeB,
            Quel1BoxType.QuBE_RIKEN_TypeB,
            Quel1BoxType.QuEL1_TypeB,
            Quel1BoxType.QuEL1_NEC,
            # Quel1BoxType.QuEL1SE_ProtoAdda,
            Quel1BoxType.QuEL1SE_Proto8,
            Quel1BoxType.QuEL1SE_Proto11,
            Quel1BoxType.QuEL1SE_Adda,
            # Quel1BoxType.QuEL1SE_RIKEN8,
        }:
            wss: Quel1WaveSubsystem = Quel1WaveSubsystem(ipaddr_wss)
        else:
            raise ValueError(f"unsupported boxtype: {boxtype}")

        if wss.hw_lifestage == E7FwLifeStage.TO_DEPRECATE:
            logger.warning(
                f"the firmware will deprecate soon, consider to update it as soon as possible: {wss.hw_version}"
            )
        elif wss.hw_lifestage == E7FwLifeStage.EXPERIMENTAL:
            logger.warning(
                f"be aware that the firmware is still in an experimental stage: {wss.hw_version}"
            )

        wss.validate_installed_e7awgsw()

        features: Set[Quel1Feature] = set()

        if wss.hw_type in {E7FwType.SIMPLEMULTI_CLASSIC}:
            features.add(Quel1Feature.SINGLE_ADC)
        elif wss.hw_type in {E7FwType.FEEDBACK_VERYEARLY}:
            features.add(Quel1Feature.BOTH_ADC_EARLY)
        elif wss.hw_type in {E7FwType.FEEDBACK_EARLY}:
            features.add(Quel1Feature.BOTH_ADC)
        else:
            raise ValueError(f"unsupported firmware is detected: {wss.hw_type}")

        if boxtype in {
            Quel1BoxType.QuBE_RIKEN_TypeA,
            Quel1BoxType.QuEL1_TypeA,
        }:
            css: Quel1AnyBoxConfigSubsystem = Quel1TypeAConfigSubsystem(
                ipaddr_css, boxtype, features, config_root, config_options
            )
        elif boxtype in {
            Quel1BoxType.QuBE_RIKEN_TypeB,
            Quel1BoxType.QuEL1_TypeB,
        }:
            css = Quel1TypeBConfigSubsystem(
                ipaddr_css, boxtype, features, config_root, config_options
            )
        elif boxtype == Quel1BoxType.QuBE_OU_TypeA:
            css = QubeOuTypeAConfigSubsystem(
                ipaddr_css, boxtype, features, config_root, config_options
            )
        elif boxtype == Quel1BoxType.QuBE_OU_TypeB:
            css = QubeOuTypeBConfigSubsystem(
                ipaddr_css, boxtype, features, config_root, config_options
            )
        elif boxtype == Quel1BoxType.QuEL1_NEC:
            css = Quel1NecConfigSubsystem(
                ipaddr_css, boxtype, features, config_root, config_options
            )
        # elif boxtype == Quel1BoxType.QuEL1SE_ProtoAdda:
        #     css = Quel1seProtoAddaConfigSubsystem(
        #         ipaddr_css, boxtype, features, config_root, config_options
        #     )
        # elif boxtype == Quel1BoxType.QuEL1SE_Proto8:
        #     css = Quel1seProto8ConfigSubsystem(
        #         ipaddr_css, boxtype, features, config_root, config_options
        #     )
        # elif boxtype == Quel1BoxType.QuEL1SE_Proto11:
        #     css = Quel1seProto11ConfigSubsystem(
        #         ipaddr_css, boxtype, features, config_root, config_options
        #     )
        # elif boxtype == Quel1BoxType.QuEL1SE_Adda:
        #     css = Quel1seAddaConfigSubsystem(
        #         ipaddr_css, boxtype, features, config_root, config_options
        #     )
        # elif boxtype == Quel1BoxType.QuEL1SE_RIKEN8:
        #     css = Quel1seRiken8ConfigSubsystem(
        #         ipaddr_css, boxtype, features, config_root, config_options
        #     )
        else:
            raise ValueError(f"unsupported boxtype: {boxtype}")

        # if not isinstance(css, Quel1ConfigSubsystemAd9082Mixin):
        #     raise AssertionError(
        #         "the given ConfigSubsystem Object doesn't provide AD9082 interface"
        #     )

        rmap = Quel1E7ResourceMapper(css, wss)

        # TODO: reconsider the design
        # if (
        #     isinstance(css, QubeConfigSubsystem)
        #     or isinstance(css, Quel1NecConfigSubsystem)
        #     or isinstance(css, Quel1seRiken8ConfigSubsystem)
        # ):
        #     if refer_by_port:
        #         box: Union[SimpleBox, SimpleBoxIntrinsic, None] = SimpleBox(
        #             css, wss, rmap
        #         )
        #     else:
        #         box = SimpleBoxIntrinsic(css, wss, rmap)
        # else:
        #     box = None
        Boxtype2QcBox: Final[Dict[Quel1BoxType, type[QcBox]]] = {
            Quel1BoxType.QuBE_RIKEN_TypeA: QubeRikenTypeAQcBox,
            Quel1BoxType.QuBE_OU_TypeA: QubeOuTypeAQcBox,
        }
        #     _PORT2LINE: Final[Dict[Quel1BoxType, Dict[int, Tuple[int, Union[int, str]]]]] = {
        # Quel1BoxType.QuBE_OU_TypeA: _PORT2LINE_QuBE_OU_TypeA,
        # Quel1BoxType.QuBE_OU_TypeB: _PORT2LINE_QuBE_OU_TypeB,
        # Quel1BoxType.QuBE_RIKEN_TypeA: _PORT2LINE_QuBE_RIKEN_TypeA,
        # Quel1BoxType.QuBE_RIKEN_TypeB: _PORT2LINE_QuBE_RIKEN_TypeB,
        # Quel1BoxType.QuEL1_TypeA: _PORT2LINE_QuEL1_TypeA,
        # Quel1BoxType.QuEL1_TypeB: _PORT2LINE_QuEL1_TypeB,
        # Quel1BoxType.QuEL1_NEC: _PORT2LINE_QuEL1_NEC,
        # Quel1BoxType.QuEL1SE_RIKEN8: _PORT2LINE_QuEL1SE_RIKEN8,
        # }

        box = Boxtype2QcBox[boxtype](css, wss, rmap)

        # # TODO: write scheduler object creation here.
        _ = ipaddr_sss

        return box


class QcBoxIntrinsic:
    def __init__(
        self,
        css: Quel1AnyBoxConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: Quel1E7ResourceMapper | None = None,
    ):
        self._css = css
        self._wss = wss
        if rmap is None:
            rmap = Quel1E7ResourceMapper(css, wss)
        self._rmap = rmap

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self._wss._wss_addr}>"

    @property
    def css(self) -> Quel1AnyBoxConfigSubsystem:
        return self._css

    @property
    def wss(self) -> Quel1WaveSubsystem:
        return self._wss

    @property
    def rmap(self) -> Quel1E7ResourceMapper:
        return self._rmap


class QcBox:
    """quel_ic_config の SimpleBox に相当する QcBox クラス"""

    # _PORT2LINE_QuBE_OU_TypeA: Dict[int, Tuple[int, Union[int, str]]] = {
    #     0: (0, 0),
    #     1: (0, "r"),
    #     2: (0, 1),
    #     5: (0, 2),
    #     6: (0, 3),
    #     7: (1, 3),
    #     8: (1, 2),
    #     11: (1, 1),
    #     12: (1, "r"),
    #     13: (1, 0),
    # }

    # _PORT2LINE_QuBE_OU_TypeB: Dict[int, Tuple[int, Union[int, str]]] = {
    #     0: (0, 0),
    #     2: (0, 1),
    #     5: (0, 2),
    #     6: (0, 3),
    #     7: (1, 3),
    #     8: (1, 2),
    #     11: (1, 1),
    #     13: (1, 0),
    # }

    # _PORT2LINE_QuBE_RIKEN_TypeA: Dict[int, Tuple[int, Union[int, str]]] = {
    #     0: (0, 0),
    #     1: (0, "r"),
    #     2: (0, 1),
    #     # 3: group-0 monitor-out
    #     4: (0, "m"),
    #     5: (0, 2),
    #     6: (0, 3),
    #     7: (1, 3),
    #     8: (1, 2),
    #     9: (1, "m"),
    #     # 10: group-1 monitor-out
    #     11: (1, 1),
    #     12: (1, "r"),
    #     13: (1, 0),
    # }

    # _PORT2LINE_QuBE_RIKEN_TypeB: Dict[int, Tuple[int, Union[int, str]]] = {
    #     0: (0, 0),
    #     2: (0, 1),
    #     # 3: group-0 monitor-out
    #     4: (0, "m"),
    #     5: (0, 2),
    #     6: (0, 3),
    #     7: (1, 3),
    #     8: (1, 2),
    #     9: (1, "m"),
    #     # 10: group-1 monitor-out
    #     11: (1, 1),
    #     13: (1, 0),
    # }

    # _PORT2LINE_QuEL1_TypeA: Dict[int, Tuple[int, Union[int, str]]] = {
    #     0: (0, "r"),
    #     1: (0, 0),
    #     2: (0, 2),
    #     3: (0, 1),
    #     4: (0, 3),
    #     5: (0, "m"),
    #     # 6: group-0 monitor-out
    #     7: (1, "r"),
    #     8: (1, 0),
    #     9: (1, 3),
    #     10: (1, 1),
    #     11: (1, 2),
    #     12: (1, "m"),
    #     # 13: group-1 monitor-out
    # }

    # _PORT2LINE_QuEL1_TypeB: Dict[int, Tuple[int, Union[int, str]]] = {
    #     1: (0, 0),
    #     2: (0, 1),
    #     3: (0, 2),
    #     4: (0, 3),
    #     5: (0, "m"),
    #     # 6: group-0 monitor-out
    #     8: (1, 0),
    #     9: (1, 1),
    #     10: (1, 3),
    #     11: (1, 2),
    #     12: (1, "m"),
    #     # 13: group-1 monitor-out
    # }

    # _PORT2LINE_QuEL1_NEC: Dict[int, Tuple[int, Union[int, str]]] = {
    #     0: (0, 0),
    #     1: (0, 1),
    #     2: (0, "r"),
    #     3: (1, 0),
    #     4: (1, 1),
    #     5: (1, "r"),
    #     6: (2, 0),
    #     7: (2, 1),
    #     8: (2, "r"),
    #     9: (3, 0),
    #     10: (3, 1),
    #     11: (3, "r"),
    # }

    # _PORT2LINE_QuEL1SE_RIKEN8: Dict[int, Tuple[int, Union[int, str]]] = {
    #     0: (0, "r"),
    #     1: (0, 0),
    #     2: (0, 2),
    #     3: (0, 3),
    #     4: (0, "m"),
    #     # 5: group-0 monitor-out
    #     6: (1, 0),
    #     7: (1, 1),
    #     8: (1, 2),
    #     9: (1, 3),
    #     10: (1, "m"),
    #     # 11: group-1 monitor-out
    # }

    # _PORT2LINE: Final[Dict[Quel1BoxType, Dict[int, Tuple[int, Union[int, str]]]]] = {
    #     Quel1BoxType.QuBE_OU_TypeA: _PORT2LINE_QuBE_OU_TypeA,
    #     Quel1BoxType.QuBE_OU_TypeB: _PORT2LINE_QuBE_OU_TypeB,
    #     Quel1BoxType.QuBE_RIKEN_TypeA: _PORT2LINE_QuBE_RIKEN_TypeA,
    #     Quel1BoxType.QuBE_RIKEN_TypeB: _PORT2LINE_QuBE_RIKEN_TypeB,
    #     Quel1BoxType.QuEL1_TypeA: _PORT2LINE_QuEL1_TypeA,
    #     Quel1BoxType.QuEL1_TypeB: _PORT2LINE_QuEL1_TypeB,
    #     Quel1BoxType.QuEL1_NEC: _PORT2LINE_QuEL1_NEC,
    #     # Quel1BoxType.QuEL1SE_RIKEN8: _PORT2LINE_QuEL1SE_RIKEN8,
    # }

    def __init__(
        self,
        css: Quel1AnyBoxConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: Quel1E7ResourceMapper | None = None,
    ):
        self._dev = QcBoxIntrinsic(css, wss, rmap)
        self._boxtype = css._boxtype
        # if self._boxtype not in self._PORT2LINE:
        #     raise ValueError(f"Unsupported boxtype; {self._boxtype}")
        # port_number = sorted([k for k in self._PORT2LINE[self._boxtype]])
        # p2l = self._PORT2LINE[self._boxtype]
        # self._ports = [
        #     TxPort(i, *p2l[i])
        #     if isinstance(p2l[i][1], int)
        #     else RxPort(i, *p2l[i])
        #     if p2l[i][1] == "r"
        #     else MonPort(i, *p2l[i])
        #     for i in port_number
        # ]
        self._linkupper = LinkupFpgaMxfe(css, wss, rmap)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self._dev._wss._wss_addr}>"

    @property
    def css(self) -> Quel1AnyBoxConfigSubsystem:
        return self._dev.css

    @property
    def wss(self) -> Quel1WaveSubsystem:
        return self._dev.wss

    @property
    def rmap(self) -> Quel1E7ResourceMapper:
        return self._dev.rmap

    # @property
    # def ports(self) -> Sequence[Port]:
    #     return self._ports

    # port = ports

    @property
    def linkupper(self) -> LinkupFpgaMxfe:
        return self._linkupper

    def linkup(
        self,
        skip_init: bool = False,
        save_dirpath: Union[Path, None] = None,
        background_noise_threshold: Union[float, None] = 512,
    ) -> Dict[int, bool]:
        linkupper = self._linkupper

        mxfe_list: Sequence[int] = (0, 1)
        hard_reset: bool = False
        use_204b: bool = False
        ignore_crc_error_of_mxfe: Union[Collection[int], None] = None
        ignore_access_failure_of_adrf6780: Union[Collection[int], None] = None
        ignore_lock_failure_of_lmx2594: Union[Collection[int], None] = None
        ignore_extraordinal_converter_select_of_mxfe: Union[
            Collection[int], None
        ] = None

        if ignore_crc_error_of_mxfe is None:
            ignore_crc_error_of_mxfe = {}

        if ignore_access_failure_of_adrf6780 is None:
            ignore_access_failure_of_adrf6780 = {}

        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = {}

        if ignore_extraordinal_converter_select_of_mxfe is None:
            ignore_extraordinal_converter_select_of_mxfe = {}

        if not skip_init:
            linkupper._css.configure_peripherals(
                ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
                ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
            )
            linkupper._css.configure_all_mxfe_clocks(
                ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
            )
        linkup_ok: Dict[int, bool] = {}
        for mxfe in mxfe_list:
            linkup_ok[mxfe] = linkupper.linkup_and_check(
                mxfe,
                hard_reset=hard_reset,
                use_204b=use_204b,
                background_noise_threshold=background_noise_threshold,
                ignore_crc_error=mxfe in ignore_crc_error_of_mxfe,
                ignore_extraordinal_converter_select=mxfe
                in ignore_extraordinal_converter_select_of_mxfe,
                save_dirpath=save_dirpath,
            )

        return linkup_ok


class SideBandMode(Enum):
    UpperSideBand = "U"
    LowerSideBand = "L"


class Port:
    def __init__(self, id: int):
        self._id = id

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self._id}>"


class RfPort(Port):
    def __init__(
        self,
        id: int,
        css: Quel1AnyBoxConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: Quel1E7ResourceMapper,
        group: int,
    ):
        super().__init__(id)
        self._css = css
        self._wss = wss
        self._rmap = rmap
        self._group = group
        self._add_channels_()

    @property
    def css(self) -> Quel1AnyBoxConfigSubsystem:
        return self._css

    @property
    def wss(self) -> Quel1WaveSubsystem:
        return self._wss

    @property
    def rmap(self) -> Quel1WaveSubsystem:
        return self._rmap

    @property
    def group(self) -> int:
        return self._group

    def _add_channels_(self) -> None:
        pass

    def set(
        self,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
    ) -> None:
        return None


class TxPortBase(RfPort):
    def __init__(
        self,
        id: int,
        css: Quel1AnyBoxConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: Quel1E7ResourceMapper,
        group: int,
        line: int,
    ):
        self._line = line
        super().__init__(id, css, wss, rmap, group)

    @property
    def line(self) -> int:
        return self._line

    def set(
        self,
        lo_freq: Union[float, None] = None,
        cnco_freq: Union[float, None] = None,
        vatt: Union[int, None] = None,
        sideband: SideBandMode = SideBandMode.UpperSideBand,
    ) -> None:
        if vatt is not None:
            self.css.set_vatt(self.group, self.line, vatt)
        sidebandmode = "U" if sideband == SideBandMode.UpperSideBand else "L"
        if sideband is not None:
            self.css.set_sideband(self.group, self.line, sidebandmode)
        if lo_freq is not None:
            self.css.set_lo_multiplier(
                self.group, self.line, int(lo_freq) // 100_000_000
            )
        if cnco_freq is not None:
            self.css.set_dac_cnco(self.group, self.line, round(cnco_freq))


class RxPortBase(RfPort):
    # def __init__(self, id: int, group: int, line: Union[int, str]):
    #     super().__init__(id, group, line)
    pass


class MoPortBase(RxPortBase):
    # def __init__(self, id: int, group: int, line: Union[int, str]):
    #     super().__init__(id, group, line)
    pass


class Channel:
    def __init__(
        self,
        id: int,
        css: Quel1AnyBoxConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: Quel1E7ResourceMapper,
        group: int,
    ):
        self._id = id
        self._css = css
        self._wss = wss
        self._rmap = rmap
        self._group = group
        self._add_channels_()

        self.freq = 10 * GHz

    def _add_channels_(self) -> None:
        pass

    @property
    def id(self) -> int:
        return self._id

    @property
    def css(self) -> Quel1AnyBoxConfigSubsystem:
        return self._css

    @property
    def wss(self) -> Quel1WaveSubsystem:
        return self._wss

    @property
    def rmap(self) -> Quel1WaveSubsystem:
        return self._rmap

    @property
    def group(self) -> int:
        return self._group

    @property
    def channel(self) -> int:
        return self._id

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self._id}>"


class TxChannel(Channel):
    def __init__(
        self,
        id: int,
        css: Quel1AnyBoxConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: Quel1E7ResourceMapper,
        group: int,
        line: int,
    ):
        super().__init__(id, css, wss, rmap, group)
        self._line = line

    @property
    def line(self) -> int:
        return self._line


class RxChannel(Channel):
    @property
    def capunit(self) -> int:
        return self._id

    @property
    def capmod(self) -> str:
        return self.rmap.get_capture_module_of_rline(self.group, self.line)

    @property
    def line(self) -> str:
        return "r"


class MoChannel(Channel):
    @property
    def line(self) -> str:
        return "m"


class QubeAnyTypeAQcBox(QcBox):
    class TxPortC1(TxPortBase):
        def _add_channels_(self) -> None:
            self.channel = TxChannel(
                0, self.css, self.wss, self.rmap, self.group, self.line
            )

    class TxPortC3(TxPortBase):
        def _add_channels_(self) -> None:
            self.channel0 = TxChannel(
                0, self.css, self.wss, self.rmap, self.group, self.line
            )
            self.channel1 = TxChannel(
                1, self.css, self.wss, self.rmap, self.group, self.line
            )
            self.channel2 = TxChannel(
                2, self.css, self.wss, self.rmap, self.group, self.line
            )

    class RxPortC4(RxPortBase):
        def _add_channels_(self) -> None:
            self.channel0: RxChannel | None = None
            self.channel1: RxChannel | None = None
            self.channel2: RxChannel | None = None
            self.channel3: RxChannel | None = None

    class MoPortC1(MoPortBase):
        def _add_channels_(self) -> None:
            self.channel = MoChannel(0, self.css, self.wss, self.rmap, self.group)


class QubeRikenTypeAQcBox(QubeAnyTypeAQcBox):
    def __init__(
        self,
        css: Quel1AnyBoxConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: Quel1E7ResourceMapper | None = None,
    ):
        super().__init__(css, wss, rmap)
        self.port0 = self.TxPortC1(0, css, wss, rmap, 0, 0)
        # 他の部分もこの実装に寄せる（新しい構成への対応のため）
        self.port1 = self.RxPortC4(1, css, wss, rmap, 0)
        if True:
            self.port1.channel0 = RxChannel(0, css, wss, rmap, 0)
            self.port1.channel1 = RxChannel(1, css, wss, rmap, 0)
            self.port1.channel2 = RxChannel(2, css, wss, rmap, 0)
            self.port1.channel3 = RxChannel(3, css, wss, rmap, 0)
        self.port2 = self.TxPortC1(2, css, wss, rmap, 0, 1)
        # self.port3 = self.TxPort(0, 0, 0)
        self.port4 = self.MoPortC1(4, css, wss, rmap, 0)
        self.port5 = self.TxPortC3(5, css, wss, rmap, 0, 2)
        self.port6 = self.TxPortC3(6, css, wss, rmap, 0, 3)
        self.port7 = self.TxPortC3(7, css, wss, rmap, 1, 3)
        self.port8 = self.TxPortC3(8, css, wss, rmap, 1, 2)
        self.port9 = self.MoPortC1(9, css, wss, rmap, 1)
        # self.port10 = self.TxPort(0, 0, 0)
        self.port11 = self.TxPortC1(11, css, wss, rmap, 1, 1)
        self.port12 = self.RxPortC4(12, css, wss, rmap, 1)
        if True:
            self.port12.channel0 = RxChannel(0, css, wss, rmap, 1)
            self.port12.channel1 = RxChannel(1, css, wss, rmap, 1)
            self.port12.channel2 = RxChannel(2, css, wss, rmap, 1)
            self.port12.channel3 = RxChannel(3, css, wss, rmap, 1)
        self.port13 = self.TxPortC1(13, css, wss, rmap, 1, 0)
        self.ports = [
            getattr(self, f"port{i}") for i in [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13]
        ]


class QubeOuTypeAQcBox(QubeAnyTypeAQcBox):
    def __init__(
        self,
        css: Quel1AnyBoxConfigSubsystem,
        wss: Quel1WaveSubsystem,
        rmap: Quel1E7ResourceMapper | None = None,
    ):
        super().__init__(css, wss, rmap)
        self.port0 = self.TxPortC1(0, css, wss, rmap, 0, 0)
        self.port1 = self.RxPortC4(1, css, wss, rmap, 0)
        if True:
            self.port1.channel0 = RxChannel(0, css, wss, rmap, 0)
            self.port1.channel1 = RxChannel(1, css, wss, rmap, 0)
            self.port1.channel2 = RxChannel(2, css, wss, rmap, 0)
            self.port1.channel3 = RxChannel(3, css, wss, rmap, 0)
        self.port2 = self.TxPortC1(2, css, wss, rmap, 0, 1)
        # self.port3 = self.TxPort(0, 0, 0)
        # self.port4 = self.MoPort(css, wss, rmap, 4, 0)
        self.port5 = self.TxPortC3(5, css, wss, rmap, 0, 2)
        self.port6 = self.TxPortC3(6, css, wss, rmap, 0, 3)
        self.port7 = self.TxPortC3(7, css, wss, rmap, 1, 3)
        self.port8 = self.TxPortC3(8, css, wss, rmap, 1, 2)
        # self.port9 = self.MoPort(css, wss, rmap, 9, 1)
        # self.port10 = self.TxPort(0, 0, 0)
        self.port11 = self.TxPortC1(11, css, wss, rmap, 1, 1)
        self.port12 = self.RxPortC4(12, css, wss, rmap, 1)
        if True:
            self.port12.channel0 = RxChannel(0, css, wss, rmap, 0)
            self.port12.channel1 = RxChannel(1, css, wss, rmap, 0)
            self.port12.channel2 = RxChannel(2, css, wss, rmap, 0)
            self.port12.channel3 = RxChannel(3, css, wss, rmap, 0)
        self.port13 = self.TxPortC1(13, css, wss, rmap, 1, 0)
        self.ports = [
            getattr(self, f"port{i}") for i in [0, 1, 2, 5, 6, 7, 8, 11, 12, 13]
        ]
