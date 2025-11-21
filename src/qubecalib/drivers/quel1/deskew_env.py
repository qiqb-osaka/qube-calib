from __future__ import annotations

from dataclasses import dataclass

from quel_ic_config_utils import deskew_tools

from ...core.context import BackendEnv


@dataclass
class Quel1DeskewEnv(BackendEnv):
    """Quel1 向けの deskew 関連ユーティリティをまとめた環境。"""

    wait_amount_resolver: deskew_tools.WaitAmountResolver
    delay_compensator: deskew_tools.E7awgDelayCompensator
    count_proposer: deskew_tools.StableCountProposer
