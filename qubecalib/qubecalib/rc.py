from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunningConfig:
    path_to_config: Path


def check_running_config_exists() -> None:
    p = __running_config__.path_to_config
    if not p.exists():
        raise FileNotFoundError(f"No such difrectory: {p = }")


__running_config__ = RunningConfig(
    path_to_config=Path(os.environ["QUBECALIB_PATH_TO_CONFIG"])
)

check_running_config_exists()
