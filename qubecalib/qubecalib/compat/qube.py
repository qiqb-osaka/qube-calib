"""後方互換性確保のためのモジュール群
"""

from __future__ import annotations

import os
from pathlib import Path

from ..rc import __running_config__ as rc


def get_absolute_path_to_config(config_path: os.PathLike) -> Path:
    """basename で指定されたファイルのフルパスを返す. ipynbを実行したディレクトリに
    basename のファイルが存在すればそのフルパスを，そうでなければ dir 内に
    存在するかを確認してそのフルパスを，存在しなければ FileNotFoundError を raise する.

    Args:
        basename (str): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        Path: _description_
    """
    path = Path(config_path)
    absolute = (
        Path(path.absolute()) if path.exists() else rc.path_to_config / Path(path.name)
    )
    if not absolute.exists():
        raise FileNotFoundError(f"File {absolute} not found")
    return absolute


class QcBox:
    def __init__(self, config_path: str | os.PathLike):
        pass


class ClassicQube:
    def __init__(self, qube: QcBox):
        pass


class Qube:
    @classmethod
    def create(cls, config_path: str | os.PathLike) -> ClassicQube:
        return ClassicQube(QcBox(config_path))
