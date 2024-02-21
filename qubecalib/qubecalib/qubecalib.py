from __future__ import annotations

import os
from collections import deque
from enum import EnumMeta
from pathlib import Path
from typing import Any, Collection, Mapping, MutableSequence, Optional, Sequence

import json5
from quel_clock_master import SequencerClient
from quel_ic_config import QUEL1_BOXTYPE_ALIAS, Quel1ConfigOption
from quel_ic_config_utils import SimpleBoxIntrinsic
from quel_ic_config_utils.linkupper import LinkupFpgaMxfe

from .temp.general_looptest_common_mod import BoxPool


class SettingsForBox:
    def __init__(self, dct: dict[str, Any]):
        self._aliases: Mapping[str, str] = dict()
        self._contents = dct
        for v in self._contents.values():
            v["boxtype"] = QUEL1_BOXTYPE_ALIAS[v["boxtype"]]
            v["config_options"] = [
                self._value_of(Quel1ConfigOption, _) for _ in v["config_options"]
            ]

    def _value_of(self, enum: EnumMeta, target: str) -> EnumMeta:
        if target not in [_.value for _ in enum]:
            raise ValueError(f"{target} is invalid")
        return [_ for _ in enum if _.value == target][0]

    @property
    def aliases(self) -> Mapping[str, str]:
        return self._aliases

    @aliases.setter
    def aliases(self, aliases: Mapping[str, str]) -> None:
        self._aliases = aliases

    def get_boxname(self, boxname_or_alias: str) -> str:
        if boxname_or_alias in self._aliases:
            return self._aliases[boxname_or_alias]
        elif boxname_or_alias in self._contents:
            return boxname_or_alias
        else:
            raise ValueError(f"boxname_or_alias: {boxname_or_alias} is not found")

    def __getitem__(self, __key: Any) -> Any:
        if __key in self._contents:
            return self._contents[__key]
        if __key in self._aliases:
            return self._contents[self._aliases[__key]]
        return self._contents[__key]


class QubeCalib:
    @classmethod
    def _exec(self, comand_queue: MutableSequence[str]) -> None:
        pass

    def __init__(
        self,
        path_for_setting_file_of_boxes: Optional[str | os.PathLike] = None,
    ):
        self._command_queue: MutableSequence = deque()
        if path_for_setting_file_of_boxes is None:
            self._settings_for_box = SettingsForBox({})
            self._definition_of_phylines = {}
            self._setting_for_clockmaster = {"ipaddr": "10.3.0.255", "reset": True}
            return
        with open(Path(os.getcwd()) / path_for_setting_file_of_boxes, "r") as f:
            jsn = json5.load(f)
            if "settings_for_box" in jsn:
                self._settings_for_box = SettingsForBox(jsn["settings_for_box"])
            if "aliases_of_boxname" in jsn:
                self._settings_for_box.aliases = jsn["aliases_of_boxname"]
            if "definition_of_phylines" in jsn:
                self._definition_of_phylines = jsn["definition_of_phylines"]
            if "setting_for_clockmaster" in jsn:
                self._setting_for_clockmaster = jsn["setting_for_clockmaster"]
        # self._boxpool: Optional[BoxPool] = None
        # self._channels: dict = dict()

    def _create_boxpool(self) -> BoxPool:
        setting = {"CLOCK_MASTER": self._setting_for_clockmaster}
        return BoxPool(settings=setting)

    def _create_box(
        self, boxpool: BoxPool, boxname_or_alias: str
    ) -> tuple[str, SimpleBoxIntrinsic, SequencerClient]:
        boxname = self._settings_for_box.get_boxname(boxname_or_alias)
        if "BOX" + boxname in boxpool._boxes:
            raise ValueError(f"box: {boxname} is alreaddy created")
        # self._settings_for_box[boxname]
        # boxpool.create_and_add_box_object(
        #     boxpool=boxpool,
        #     boxname=boxname,
        #     setting=setting,
        # )
        # settings = {"CLOCK_MASTER": self._setting_for_clockmaster}
        # boxname = self._settings_for_box.get_boxname(boxname_or_alias)
        settings = {
            "BOX" + boxname: self._settings_for_box[boxname],
        }
        boxpool._parse_settings(settings=settings)
        return (
            boxname,
            boxpool._boxes["BOX" + boxname][0],
            boxpool._boxes["BOX" + boxname][1],
        )

    def _create_box_all(self, exclude_boxnames_or_aliases: tuple = tuple()) -> BoxPool:
        excludes = [
            self._settings_for_box.get_boxname(_) for _ in exclude_boxnames_or_aliases
        ]
        boxnames = [_ for _ in self._settings_for_box._contents if _ not in excludes]
        boxpool = self._create_boxpool()
        # for _ in boxnames:
        #     self._create_box(boxpool, _)
        settings = {
            "BOX" + boxname: self._settings_for_box[boxname] for boxname in boxnames
        }
        print(settings)
        # self._settings_for_box()[self._settings_for_box[_] for _ in boxnames]
        boxpool._parse_settings(settings=settings)

        return boxpool

    # def create_channel(
    #     self,
    #     name: str,
    #     portname: str,
    #     lane: int,
    # ):
    #     self._channels[name] = None

    def _show_clock(
        self,
        boxpool: BoxPool,
        boxname_or_alias: str,
    ) -> tuple[bool, int, int]:
        """同期用FPGA内部クロックの現在値を表示する

        Returns
        -------
        tuple[bool, int, int]
            _description_
        """
        return boxpool._boxes["BOX" + boxname_or_alias][1].read_clock()

    def show_clock_all(
        self,
        exclude_boxnames_or_aliases: tuple = tuple(),
    ) -> dict[str, tuple[bool, int, int] | tuple[bool, int]]:
        """登録されているすべての筐体の同期用FPGA内部クロックの現在値を表示する

        Returns
        -------
        tuple[tuple[bool, int], dict[str, tuple[bool, int, int]]]
            _description_
        """
        boxpool = self._create_box_all(exclude_boxnames_or_aliases)

        boxnames = [_.replace("BOX", "", 1) for _ in boxpool._boxes]
        client_results = {_: self._show_clock(boxpool, _) for _ in boxnames}
        results: dict[str, tuple[bool, int, int] | tuple[bool, int]] = {
            k: v for k, v in client_results.items()
        }
        results["master"] = boxpool._clock_master.read_clock()
        return results

    def resync_box(
        self,
        boxname_or_alias: str,
    ) -> None:
        """resync したい機体だけを個別に resync する"""
        boxpool = self._create_boxpool()
        name, box, sc = self._create_box(
            boxpool=boxpool, boxname_or_alias=boxname_or_alias
        )
        boxpool._clock_master.kick_clock_synch((sc.ipaddress,))

    def exec(self) -> None:
        pass

    def linkup(
        self,
        boxname_or_alias: str,
        skip_init: bool = False,
        save_dirpath: Optional[Path] = None,
        background_noise_threshold: Optional[float] = 512,
    ) -> dict[int, bool]:
        boxpool = self._create_boxpool()
        _, box, _ = self._create_box(boxpool=boxpool, boxname_or_alias=boxname_or_alias)
        linkupper = LinkupFpgaMxfe(box.css, box.wss, box.rmap)

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
        linkup_ok: dict[int, bool] = {}
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
