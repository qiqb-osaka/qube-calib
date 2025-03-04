from __future__ import annotations

from typing import Final

from .executor import Executor
from .sysconfdb import SystemConfigDatabase


class QubeCalibBase:
    def __init__(self) -> None:
        self._system_config_database: Final[SystemConfigDatabase] = (
            SystemConfigDatabase()
        )
        self._executor: Final[Executor] = Executor(self._system_config_database)
