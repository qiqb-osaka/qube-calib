"""quel_ic_config_util.Quel1Box をスムーズに導入するための互換性レイヤ"""

from __future__ import annotations

import logging
import os
from enum import Enum
from ipaddress import IPv4Address, IPv6Address, ip_address
from os import PathLike
from pathlib import Path
from typing import Any, Collection, Dict, Optional, Sequence, Tuple

import yaml
from quel_ic_config import (
    LinkupFpgaMxfe,
    Quel1AnyConfigSubsystem,
    Quel1Box,
    Quel1BoxIntrinsic,
    Quel1BoxType,
    Quel1ConfigOption,
    Quel1E7ResourceMapper,
    Quel1WaveSubsystem,
)

# create_box_objects,
from .rc import __running_config__ as rc

logger = logging.getLogger(__name__)


class Sideband(Enum):
    UpperSideBand: str = "U"
    LowerSideBand: str = "L"


class QcBox:
    """Quel1Box の機能追加までの互換性オブジェクト
    LSI 設定は Quel1Box を踏襲し，Quel1Box への自然な移行を促す
    """

    def __init__(
        self,
        *,
        ipaddr_wss: str | IPv4Address | IPv6Address,
        ipaddr_sss: str | IPv4Address | IPv6Address,
        ipaddr_css: str | IPv4Address | IPv6Address,
        boxtype: Quel1BoxType,
        config_root: Path | None,
        config_options: Collection[Quel1ConfigOption],
        auto_init: bool = True,
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
        if auto_init:
            box.init()
        self._box = box
        self._linkupper = linkupper
        self._ipaddr_sss = ip_address(ipaddr_sss)
        self._config_options = config_options
        # TODO: workaround
        for port in [0, 5, 6, 7, 8, 13]:
            try:
                self.box.config_port(port, vatt=0x800)
            except ValueError:
                continue

    @property
    def box(self) -> Quel1Box:
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

    @property
    def _dev(self) -> Quel1BoxIntrinsic:
        return self.box._dev

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.box.wss._wss_addr}>"

    def dump_config(self) -> dict[str, dict[str, Any]]:
        return self.box.dump_config()

    def dump_port_config(self, port: int) -> dict[str, Any]:
        c = self.box.dump_config()
        return c[f"port-#{port:02}"]

    def config_port(
        self,
        port: int,
        *,
        lo_freq: Optional[float] = None,
        cnco_freq: Optional[float] = None,
        vatt: Optional[float] = None,
        sideband: Optional[Sideband] = None,
    ) -> None:
        kwargs: Dict[str, float | str] = {}
        if lo_freq is not None:
            kwargs["lo_freq"] = lo_freq
        if cnco_freq is not None:
            kwargs["cnco_freq"] = cnco_freq
        if vatt is not None:
            kwargs["vatt"] = vatt
        if sideband is not None:
            kwargs["sideband"] = sideband.value
        if not kwargs:
            raise ValueError("no config parameters")
        return self.box.config_port(port, **kwargs)

    def config_channel(
        self,
        port: int,
        channel: int,
        *,
        fnco_freq: Optional[float] = None,
    ) -> None:
        kwargs = {}
        if fnco_freq is not None:
            kwargs["fnco_freq"] = fnco_freq
        if not kwargs:
            raise ValueError("no config parameters")
        return self.box.config_channel(port, channel, **kwargs)

    def get_awg_of_channel(self, port: int, channel: int) -> int:
        group, line = self.box._convert_all_port(port)
        return self.rmap.get_awg_of_channel(group, line, channel)

    def get_port(self, port: int) -> Tuple[int, int | str]:
        group, line = self.box._convert_all_port(port)
        return group, line

    def linkup(
        self,
        skip_init: bool = False,
        save_dirpath: Optional[Path] = None,
        background_noise_threshold: Optional[float] = 512,
    ) -> Dict[int, bool]:
        linkupper = self._linkupper

        mxfe_list: Sequence[int] = (0, 1)
        hard_reset: bool = False
        use_204b: bool = False
        ignore_crc_error_of_mxfe: Optional[Collection[int]] = None
        ignore_access_failure_of_adrf6780: Optional[Collection[int]] = None
        ignore_lock_failure_of_lmx2594: Optional[Collection[int]] = None
        ignore_extraordinal_converter_select_of_mxfe: Optional[Collection[int]] = None

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


class QubeYamlFiles(dict):
    def __init__(self, *config_paths: str | PathLike):
        d = {
            Path(k).stem.replace("qube_", ""): QubeYamlFile(Path(k))
            for k in config_paths
        }
        super().__init__(d)


class QubeYamlFile(dict):
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

    def __init__(self, config_path: str | PathLike):
        path = Path(config_path)
        config_path = self.get_absolute_path_to_config(path)
        yaml = self.load(config_path)

        d = self.parse_config(config_path.name, yaml)
        d["ipaddr_wss"] = str(d["ipaddr_wss"])
        d["ipaddr_sss"] = str(d["ipaddr_sss"])
        d["ipaddr_css"] = str(d["ipaddr_css"])
        d["config_root"] = None
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
        d["config_options"] = config_options

        super().__init__(d)

    def load(self, config_path: str | os.PathLike) -> Dict:
        with open(config_path, "rb") as f:
            c = yaml.safe_load(f)
        return c

    def get_absolute_path_to_config(self, config_path: Path) -> Path:
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

    def parse_config(self, config_file_name: str | os.PathLike, content: dict) -> dict:
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
        s = self.parse_qubecalib_series(str(config_file_name))
        kw["boxtype"] = self._YAML_BOX_TYPE_MAPPER[s][content["type"]]
        return kw

    def parse_qubecalib_series(self, config_file_name: str) -> str:
        m = self._QUBECALIB_SERIES_MAPPER
        s = config_file_name.split("_")
        n = m[s[0]]
        if isinstance(n, dict):
            if isinstance(n[s[1]], str):
                return n[s[1]]
        elif isinstance(n, str):
            return n
        raise ValueError(f"{config_file_name} is not supported.")

    @classmethod
    def list_configfiles(cls) -> list[str]:
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
