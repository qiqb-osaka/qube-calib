from typing import Tuple

import numpy as np
import numpy.typing as npt


def pca(
    x: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r = np.dot(x, x.transpose()) / x.shape[0]
    d, e = np.linalg.eig(r)
    return d, e


def to_float(x: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    return np.stack([np.real(x), np.imag(x)], axis=0)


def to_complex(x: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
    return x[0, :] + 1j * x[1, :]


def principal_axis_rotation(
    x: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], float]:
    """Action to rotate the principal axis of the signal to the imaginary axis"""
    xx = to_float(x)
    m = xx.mean(axis=1).reshape(xx.shape[0], 1)
    d, e = pca(xx - m)
    idx = np.argsort(d)
    d, e = d[idx], e[:, idx]
    e = (
        e if np.dot(m[:, 0], e[:, -1]) > 0 else -e
    )  # 固有ベクトルの向きを信号の向きに合わせる
    return (
        to_complex(
            np.dot(xx.transpose(), e).transpose()
        ),  # signal rotated to align the principal component with the imaginary axis
        np.arctan2(e[0, -1], e[1, -1]),  # phase angle in the IQ plane of principal axis
    )
