from __future__ import annotations

import os

from . import qube as lib_qube


class QubeControl:
    def __init__(self, config_path: str | os.PathLike):
        self._qube = lib_qube.Qube.create(config_path)

    @property
    def qube(self) -> lib_qube.QubeBase:
        return self._qube
