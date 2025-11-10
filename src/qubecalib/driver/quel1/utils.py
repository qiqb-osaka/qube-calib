from __future__ import annotations

from typing import cast

from quel_ic_config import Quel1Box

from ...orchestrator import ExecutionContext
from ..quel1.quel1driver import Quel1Driver


def quel1(ctx: ExecutionContext) -> Quel1Driver:
    return cast(Quel1Driver, ctx.driver("quel1"))


def box(ctx: ExecutionContext, box_key: str) -> Quel1Box:
    return cast(Quel1Box, cast(Quel1Driver, ctx.driver("quel1")).box(box_key))
