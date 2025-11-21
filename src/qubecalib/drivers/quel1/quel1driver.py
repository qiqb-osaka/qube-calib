from __future__ import annotations

from dataclasses import dataclass
from logging import Formatter, Handler, Logger, StreamHandler, getLogger
from typing import Any, Iterable

from quel_ic_config import (
    Quel1Box,
    Quel1PortType,
)
from quel_ic_config_utils import configuration

from ...core.context import Driver
from .boxregistry import BoxRegistry
from .killtimer import KillTimer

logger = getLogger(__name__)


def show_log(
    level: str = "DEBUG",
    *,
    name: str = "qubecalib",
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


class Quel1Driver(Driver):
    def __init__(
        self,
        # clock_master_ip: str | None = None,
        # deadline: float | None = 3600,
        # interval_sec: int = 10,
    ) -> None:
        self.registry = BoxRegistry(
            default_deadline=None,
            on_state_change=self._on_registry_state_change,
        )
        self.kill_timer: KillTimer | None = None
        # self.interval_sec = interval_sec
        self.interval_sec = 10

        # self._clock_master: QuelClockMasterV1 | None = (
        #     QuelClockMasterV1(ipaddr=clock_master_ip, boxes=[])
        #     if clock_master_ip
        #     else None
        # )

    @classmethod
    def from_configuration(
        cls,
        boxes: list[str],
        # *,
        # deadline: float | None = 18000,  # 5 hours
        # interval_sec: int = 10,
    ) -> Quel1Driver:
        conf = configuration.load_default_configuration()
        driver = Quel1Driver(
            # deadline=deadline,
            # interval_sec=interval_sec,
        )
        driver.setup_boxes([b for b in conf.boxes if b.name in boxes])
        return driver

    def _on_registry_state_change(self, count: int) -> None:
        """Called whenever the number of active Boxces changes."""
        if count > 0 and self.kill_timer is None:
            # start a new KillTimer
            # logger.debug("Starting KillTimer thread.")
            # self.kill_timer = KillTimer(self.registry, interval_sec=self.interval_sec)
            # self.kill_timer.start()
            pass
        elif count == 0 and self.kill_timer is not None:
            # stop and deleter KillTimer
            # logger.debug("Stopping KillTimer thread (no active boxes).")
            # self.kill_timer.stop()
            # self.kill_timer.join(timeout=self.interval_sec + 5.0)
            # self.kill_timer = None
            pass

    def setup_boxes(self, boxes: Iterable[configuration.Box]) -> None:
        name_to_box = {
            box.name: box for box in configuration.get_boxes_in_parallel(boxes)
        }

        keys = list(name_to_box.keys())
        for name in keys:
            self.registry.register(name, name_to_box[name])

        logger.debug(f"Boxes setting up: {list(name_to_box.keys())}")
        configuration.reconnect_and_get_link_status_in_parallel(name_to_box.values())
        logger.debug("Boxes initialized and reconnected.")

        name_to_box.clear()

    def release(self, name: str | None = None) -> None:
        if name is not None:
            self.registry.release(name)
        else:
            for name in list(self.registry._handles.keys()):
                self.registry.release(name)
            if self.kill_timer:
                # self.kill_timer.stop()
                # # self.kill_timer.join(timeout=self.interval_sec + 5.0)
                # self.kill_timer = None
                pass

    def list_active(self) -> list[str]:
        return list(self.registry.list_active())

    def box(self, name: str) -> Quel1Box:
        return self.registry.get(name)

    def dump_box(self, name: str) -> dict[str, Any]:
        return self.box(name).dump_box()

    def dump_port(self, name: str, port: Quel1PortType) -> dict[str, Any]:
        return self.box(name).dump_port(port)


@dataclass
class AwgSampledSequence:
    channel_key: str


@dataclass
class CapSampledSequence:
    channel_key: str
