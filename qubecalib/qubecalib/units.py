from typing import Final
import numpy as np

Sec: Final[float] = 1e+9
mS: Final[float] = 1e+6
uS: Final[float] = 1e+3
nS: Final[float] = 1.

Hz: Final[float] = 1e-9
kHz: Final[float] = 1e-6
MHz: Final[float] = 1e-3
GHz: Final[float] = 1.

BLOCK: Final[float] = 128. # ns
BLOCKs: Final[float] = BLOCK

WORD: Final[float] = 8. # ns
WORDs: Final[float] = WORD

RAD: Final[float] = 1.
DEG: Final[float] = np.pi / 180.

