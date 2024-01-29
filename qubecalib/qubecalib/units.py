from typing import Final

import numpy as np

Sec: Final[float] = 1e9
mS: Final[float] = 1e6
uS: Final[float] = 1e3
nS: Final[float] = 1.0

Hz: Final[float] = 1e-9
kHz: Final[float] = 1e-6
MHz: Final[float] = 1e-3
GHz: Final[float] = 1.0

BLOCK: Final[float] = 128.0  # ns
BLOCKs: Final[float] = BLOCK

WORD: Final[float] = 8.0  # ns
WORDs: Final[float] = WORD

RAD: Final[float] = 1.0
DEG: Final[float] = np.pi / 180.0
