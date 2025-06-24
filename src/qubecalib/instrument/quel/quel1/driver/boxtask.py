from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from quel_ic_config import CaptureReturnCode, Quel1BoxWithRawWss


class BoxTask(ABC):
    @abstractmethod
    def load(self, box: Quel1BoxWithRawWss) -> None: ...

    @abstractmethod
    def execute(
        self,
    ) -> tuple[
        dict[int, CaptureReturnCode],
        dict[tuple[int, int], list[npt.NDArray[np.complex64]]],
    ]: ...
