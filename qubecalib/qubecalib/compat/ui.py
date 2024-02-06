from __future__ import annotations

import os

from ..qcbox import QcBox, QcBoxFactory


class QubeControl:
    def __init__(self, config_path: str | os.PathLike):
        self._qube = QcBoxFactory.produce(config_path)

    @property
    def qube(self) -> QcBox:
        return self._qube
