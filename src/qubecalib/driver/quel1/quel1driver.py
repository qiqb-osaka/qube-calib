from __future__ import annotations

from logging import Formatter, Handler, Logger, StreamHandler, getLogger
from typing import Any

from quel_ic_config import Quel1Box, QuelClockMasterV1

from .boxregistry import BoxRegistry
from .killtimer import KillTimer


def show_log(
    name: str = "qubecalib",
    *,
    level: str = "DEBUG",
    handler: Handler = StreamHandler(),
    formatter: Formatter = Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ),
) -> Logger:
    handler.setFormatter(formatter)
    logger = getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class Quel1Driver:
    def __init__(self, clock_master_ip: str | None, deadline_min: int = 60) -> None:
        self.registry = BoxRegistry()
        self.kill_timer = KillTimer(self.registry)
        self.kill_timer.start()
        self.deadline = 60 * deadline_min

        self._clock_master: QuelClockMasterV1 | None = (
            QuelClockMasterV1(ipaddr=clock_master_ip, boxes=[])
            if clock_master_ip
            else None
        )

    def setup_boxes(self, mapping: dict[str, dict[str, Any]]) -> None:
        boxes = {
            name: Quel1Box.create(
                **((config | {"name": name}) if "name" not in config else config)
            )
            for name, config in mapping.items()
        }
        for name, box in boxes.items():
            box.reconnect()
        keys = list(boxes.keys())
        for name in keys:
            self.registry.register(name, boxes[name], self.deadline)
            boxes.pop(name)

    def box(self, name: str) -> Quel1Box:
        return self.registry.get(name)

    def release(self, name: str | None = None) -> None:
        if name is not None:
            self._release(name)
        else:
            for name in list(self.registry._handles.keys()):
                self._release(name)

    def _release(self, name: str) -> None:
        if self._clock_master is not None:
            self._clock_master._boxes.remove(self.registry.get(name))
        self.registry.release(name)
