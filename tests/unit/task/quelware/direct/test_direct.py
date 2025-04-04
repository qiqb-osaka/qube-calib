from __future__ import annotations

import quel_ic_config


def test_quel_ic_config_version() -> None:
    assert quel_ic_config.__version__ == "0.8.13"
