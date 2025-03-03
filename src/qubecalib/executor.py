from __future__ import annotations

import datetime
import getpass
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Final

from .instrument.quel.quel1.command import Command
from .instrument.quel.quel1.sequencer import Sequencer
from .instrument.quel.quel1.system import BoxPool
from .sysconfdb import SystemConfigDatabase

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, sysdb: SystemConfigDatabase) -> None:
        self._work_queue: Final[deque] = deque()
        self._boxpool: BoxPool = BoxPool()
        self._config_buffer: Final[deque] = deque()
        self.sysdb = sysdb

    def reset(self) -> None:
        self._work_queue.clear()
        self._boxpool = BoxPool()

    def collect_boxes(self) -> set[Any]:
        return set(
            sum(
                [
                    [
                        __["box"].box_name
                        for _ in command.resource_map.values()
                        for __ in _
                    ]
                    for command in self._work_queue
                    if isinstance(command, Sequencer)
                ],
                [],
            )
        )

    def collect_sequencers(self) -> set[Sequencer]:
        return {_ for _ in self._work_queue if isinstance(_, Sequencer)}

    def __iter__(self) -> Executor:
        # if not self._work_queue:
        #     return self
        # last_command = self._work_queue[-1]
        # if not isinstance(last_command, Sequencer):
        #     raise ValueError("_work_queue should end with a Sequencer command")
        self.clear_log()  # clear config for last execution
        return self

    def __next__(self) -> tuple[Any, dict, dict]:
        # ワークキューが空になったら実行を止める
        if not self._work_queue:
            self.check_config()
            self._boxpool._box_config_cache.clear()
            self._boxpool = BoxPool()
            self.clear_log()
            raise StopIteration()
        # Sequencer が見つかるまでコマンドを逐次実行
        while True:
            # もしワークキューが空になったらエラーを出す
            if not self._work_queue:
                raise ValueError(
                    "command que should include at least one Sequencer command."
                )
            next = self._work_queue.pop()
            # 次に実行するコマンドが Sequencer ならばループを抜ける
            if isinstance(next, Sequencer):
                # for box, _ in self._boxpool._boxes.values():
                #     box.initialize_all_awgs()
                break
            # Sequencer 以外のコマンドを逐次実行
            next.execute(self._boxpool)
        for command in self._work_queue:
            # もしコマンドキューに Sequencer が残っていれば次の Sequencer を実行する
            if isinstance(command, Sequencer):
                # if self._quel1system is None:
                #     raise ValueError("Quel1System is not defined")
                # status, iqs, config = next.execute(self._boxpool, self._quel1system)
                results = next.execute(self._boxpool)
                user_name = getpass.getuser()
                current_pyfile = os.path.abspath(__file__)
                date_time = datetime.datetime.now()
                clock_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
                self._config_buffer.append(
                    (
                        # config,
                        user_name,
                        current_pyfile,
                        # __version__,
                        date_time,
                        clock_ns,
                    )
                )
                if not self._work_queue:
                    self.check_config()
                    self._boxpool._box_config_cache.clear()
                    self._boxpool = BoxPool()
                    self.clear_log()
                # return status, iqs, config
                return results
        # これ以上 Sequencer がなければ残りのコマンドを実行する
        # if self._quel1system is None:
        #     raise ValueError("Quel1System is not defined")
        # status, iqs, config = next.execute(self._boxpool, self._quel1system)
        results = next.execute(self._boxpool)
        # status, iqs, config = next.execute(self._boxpool)
        user_name = getpass.getuser()
        current_pyfile = os.path.abspath(__file__)
        date_time = datetime.datetime.now()
        clock_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
        self._config_buffer.append(
            (
                # config,
                user_name,
                current_pyfile,
                # __version__,
                date_time,
                clock_ns,
            )
        )
        for command in self._work_queue:
            command.execute(self._boxpool)
        if not self._work_queue:
            self.check_config()
            self._boxpool._box_config_cache.clear()
            self._boxpool = BoxPool()
            self.clear_log()
        # return status, iqs, config
        return results

    def check_config(self) -> None:
        box_configs = {
            box_name: self._boxpool.get_box(box_name)[0].dump_box()
            for box_name in self._boxpool._box_config_cache
        }
        for box_name, initial in self._boxpool._box_config_cache.items():
            if box_name not in box_configs:
                raise ValueError(f"The BoxPool is inconsistent with {box_name}")
            final = box_configs[box_name]
            if initial != final:
                logger.warning(
                    f"The box {box_name} configuration has changed since the start of the process: {initial} -> {final}"
                )

    def add_command(self, command: Command) -> None:
        self._work_queue.appendleft(command)

    def get_log(self) -> list:
        return list(self._config_buffer)

    def clear_log(self) -> None:
        self._config_buffer.clear()

    def execute(self) -> tuple:
        """queue に登録されている command を実行する（未実装）"""
        return "", "", ""

    def step_execute(
        self,
        repeats: int = 1,
        interval: float = 10240,
        integral_mode: str = "integral",  # "single"
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
    ) -> Executor:
        """queue に登録されている command を実行する iterator を返す"""
        # work queue を舐めて必要な box を生成する
        boxes = self.collect_boxes()
        # もし box が複数で clockmaster_setting が設定されていれば QuBEMasterClient を生成する
        if len(boxes) > 1 and self.sysdb._clockmaster_setting is not None:
            self._boxpool.create_clock_master(
                ipaddr=str(self.sysdb._clockmaster_setting.ipaddr)
            )
        # boxpool を生成する
        for box_name in boxes:
            setting = self.sysdb._box_settings[box_name]
            box = self._boxpool.create(
                box_name,
                ipaddr_wss=str(setting.ipaddr_wss),
                ipaddr_sss=str(setting.ipaddr_sss),
                ipaddr_css=str(setting.ipaddr_css),
                boxtype=setting.boxtype,
                config_root=Path(setting.config_root)
                if setting.config_root is not None
                else None,
                config_options=setting.config_options,
            )
            status = box.reconnect()
            for mxfe_idx, s in status.items():
                if not s:
                    logger.error(
                        f"be aware that mxfe-#{mxfe_idx} is not linked-up properly"
                    )

        # sequencer に measurement_option を設定する
        for sequencer in self.collect_sequencers():
            if sequencer.interval is None:
                new_interval = interval
            else:
                new_interval = sequencer.interval
            sequencer.set_measurement_option(
                repeats=repeats,
                interval=new_interval,
                integral_mode=integral_mode,
                dsp_demodulation=dsp_demodulation,
                software_demodulation=software_demodulation,
            )

        return self
