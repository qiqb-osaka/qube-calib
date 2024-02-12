"""quel_ic_config_util.SimpleBox をスムーズに導入するための互換性レイヤ
"""

from __future__ import annotations

import logging
import os
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from typing import Any, Collection, Dict

import yaml
from quel_ic_config import Quel1AnyConfigSubsystem, Quel1BoxType, Quel1ConfigOption
from quel_ic_config_utils import (
    LinkupFpgaMxfe,
    Quel1E7ResourceMapper,
    Quel1WaveSubsystem,
    SimpleBox,
    create_box_objects,
)

from .rc import __running_config__ as rc

logger = logging.getLogger(__name__)


class QcBox:
    """SimpleBox の機能追加までの互換性オブジェクト
    LSI 設定は SimpleBox を踏襲し，SimpleBox への自然な移行を促す
    """

    def __init__(
        self,
        ipaddr_wss: str | IPv4Address | IPv6Address,
        ipaddr_sss: str | IPv4Address | IPv6Address,
        ipaddr_css: str | IPv4Address | IPv6Address,
        boxtype: Quel1BoxType,
        config_root: Path | None,
        config_options: Collection[Quel1ConfigOption],
    ):
        _, _, _, linkupper, box = create_box_objects(
            ipaddr_wss,
            ipaddr_sss,
            ipaddr_css,
            boxtype,
            config_root,
            config_options,
            refer_by_port=True,
        )
        box.init()
        self._box = box
        self._linkupper = linkupper
        self._ipaddr_sss = ip_address(ipaddr_sss)
        self._config_options = config_options

    @property
    def box(self) -> SimpleBox:
        return self._box

    @property
    def css(self) -> Quel1AnyConfigSubsystem:
        return self.box.css

    @property
    def wss(self) -> Quel1WaveSubsystem:
        return self.box.wss

    @property
    def rmap(self) -> Quel1E7ResourceMapper:
        return self.box._dev.rmap

    @property
    def linkupper(self) -> LinkupFpgaMxfe:
        return self._linkupper

    @property
    def boxtype(self) -> Quel1BoxType:
        return self.box._boxtype

    @property
    def ipaddr_sss(self) -> str | IPv4Address | IPv6Address:
        return self._ipaddr_sss

    @property
    def config_options(self) -> Collection[Quel1ConfigOption]:
        return self._config_options

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.box.wss._wss_addr}>"

    def dump_config(self) -> dict[str, dict[str, Any]]:
        return self.box.dump_config()

    def dump_port_config(self, port: int) -> dict[str, Any]:
        c = self.box.dump_config()
        return c[f"port-#{port:02}"]

    def get_awg_of_channel(self, port: int, channel: int) -> int:
        group, line = self._convert_all_port(port)
        return self.rmap.get_awg_of_channel(group, line, channel)

    # def _convert_tx_port(self, port: int) -> Tuple[int, int]:
    #     return self.box._convert_tx_port(port)

    # def _convert_rx_port(self, port: int) -> Tuple[int, int]:
    #     return self.box._convert_tx_port(port)

    # def _convert_all_port(self, port: int) -> Tuple[int, int | str]:
    #     return self.box._convert_all_port(port)


class QcBoxFactory:
    _QUBECALIB_SERIES_MAPPER: Dict[str, str | Dict[str, str]] = {
        "quel-1": "quel-1",
        "qube": {
            "ou": "qube_ou",
            "riken": "qube_riken",
        },
    }
    _YAML_BOX_TYPE_MAPPER: Dict[str, Dict[str, Quel1BoxType]] = {
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

    @classmethod
    def get_absolute_path_to_config(cls, config_path: Path) -> Path:
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
        absolute = (
            Path(config_path.absolute())
            if config_path.exists()
            else rc.path_to_config / Path(config_path.name)
        )
        if not absolute.exists():
            raise FileNotFoundError(f"File {absolute} not found")
        return absolute

    @classmethod
    def produce(cls, config_path: str | os.PathLike) -> QcBox:
        path = Path(config_path)
        c = cls.load(cls.get_absolute_path_to_config(path))
        d = cls.parse_config(path.name, c)
        if d["boxtype"] in [
            Quel1BoxType.QuBE_OU_TypeB,
            Quel1BoxType.QuBE_RIKEN_TypeB,
            Quel1BoxType.QuEL1_TypeB,
        ]:
            config_options = [
                Quel1ConfigOption.USE_MONITOR_IN_MXFE0,
                Quel1ConfigOption.USE_MONITOR_IN_MXFE1,
            ]
        else:
            config_options = [
                Quel1ConfigOption.USE_READ_IN_MXFE0,
                Quel1ConfigOption.USE_READ_IN_MXFE1,
            ]
        return QcBox(
            ipaddr_wss=str(d["ipaddr_wss"]),
            ipaddr_sss=str(d["ipaddr_sss"]),
            ipaddr_css=str(d["ipaddr_css"]),
            boxtype=d["boxtype"],
            config_root=None,
            config_options=config_options,
        )

    @classmethod
    def parse_qubecalib_series(cls, config_file_name: str) -> str:
        m = cls._QUBECALIB_SERIES_MAPPER
        s = config_file_name.split("_")
        n = m[s[0]]
        if isinstance(n, dict):
            if isinstance(n[s[1]], str):
                return n[s[1]]
        elif isinstance(n, str):
            return n
        raise ValueError(f"{config_file_name} is not supported.")

    @classmethod
    def load(cls, config_path: str | os.PathLike) -> Dict:
        with open(config_path, "rb") as f:
            c = yaml.safe_load(f)
        return c

    @classmethod
    def parse_config(cls, config_file_name: str | os.PathLike, content: dict) -> dict:
        kwmapper = {
            "ipfpga": "ipaddr_wss",
            "iplsi": "ipaddr_css",
            "ipmulti": "ipaddr_sss",
        }
        kw = {
            kwmapper[k]: ip_address(v) if k == "ipfpga" or k == "iplsi" else v
            for k, v in content.items()
            if k in kwmapper
        }
        if "ipaddr_sss" not in kw:
            kw["ipaddr_sss"] = kw["ipaddr_wss"] + 0x10000
        s = cls.parse_qubecalib_series(str(config_file_name))
        kw["boxtype"] = cls._YAML_BOX_TYPE_MAPPER[s][content["type"]]
        return kw
