from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger

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
    driver_registry: DriverRegistry = field(default_factory=DriverRegistry)

    def driver(self, kind: str) -> Driver:
        return self.driver_registry.get(kind)
