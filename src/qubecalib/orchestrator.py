from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

from typing_extensions import deprecated

from .driverbase import Driver

logger = getLogger(__name__)


@dataclass
class DriverRegistry:
    drivers: dict[str, Driver] = field(default_factory=dict)

    def register(self, kind: str, driver: Driver) -> None:
        self.drivers[kind] = driver

    def get(self, kind: str) -> Driver:
        if kind not in self.drivers:
            raise KeyError(f"Driver '{kind}' not registered.")
        return self.drivers[kind]


@dataclass
class ExecutionContext:
    session_key: str = ""  # Unique identifier for the execution session TODO: required?
    driver_registry: DriverRegistry = field(default_factory=DriverRegistry)
    cache: dict[str, Any] = field(default_factory=dict)
    logdir: str | None = None

    def get_driver(self, kind: str) -> Driver:
        return self.driver_registry.get(kind)

    @deprecated("Use 'get_driver' instead.")
    def driver(self, kind: str) -> Driver:
        return self.get_driver(kind)
