from __future__ import annotations

import os

from ..compat.qube import ClassicQube, Qube


class QubeControl:
    def __init__(self, config_path: str | os.PathLike):
        self._qube = Qube.create(config_path)

    @property
    def qube(self) -> ClassicQube:
        return self._qube
