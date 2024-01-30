from enum import Enum
from typing import Final

import numpy as np


class Units(Enum):
    Sec = 1e9
    mS = 1e6
    uS = 1e3
    nS = 1.0
    Hz = 1e-9
    kHz = 1e-6
    MHz = 1e-3
    GHz = 1.0
    BLOCK = 128.0  # ns
    BLOCKs = BLOCK
    WORD = 8.0  # ns
    WORDs = WORD
    RAD = 1.0
    DEG = np.pi / 180.0


Sec: Final[Units] = Units.Sec
mS: Final[Units] = Units.mS
uS: Final[Units] = Units.uS
nS: Final[Units] = Units.nS

Hz: Final[Units] = Units.Hz
kHz: Final[Units] = Units.kHz
MHz: Final[Units] = Units.MHz
GHz: Final[Units] = Units.GHz

BLOCK: Final[Units] = Units.BLOCK
BLOCKs: Final[Units] = Units.BLOCKs

WORD: Final[Units] = Units.WORD
WORDs: Final[Units] = Units.WORDs

RAD: Final[Units] = Units.RAD
DEG: Final[Units] = Units.DEG
