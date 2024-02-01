import numpy as np

from .neopulse import Sequence, ceil
from .units import MHz


def oversample(
    t: np.ndarray,  # TODO 第一引数は不要になったので削除する
    begin: float,
    end: float,
    ratio: int = 500,
    sampling_rate: float = 500 * MHz,
) -> np.ndarray:
    """サンプル時系列 t を ratio 倍にオーバーサンプルする

    Args:
        t (np.ndarray): サンプル時系列
        begin (float): サンプル開始点
        end (float): サンプル終了点（この点は含まれない）
        ratio (int, optional): オーバーサンプル率. Defaults to 500.
        sampling_rate (float, optional): サンプリングレート. Defaults to 500 MHz.

    Returns:
        np.ndarray: オーバーサンプルした時系列
    """
    dt = 1 / sampling_rate / ratio
    if ratio == 1:
        v = np.arange(ceil(begin, dt), ceil(end, dt), dt)
    else:
        v = np.arange(ceil(begin, dt), ceil(end, dt) + dt, dt)

    return v


def get_sampling_data(
    flattened_sequence: Sequence, oversampling_ratio: int = 1, endpoint: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """sequence のサンプルデータを取り出す

    Args:
        flattened_sequence (Sequence): sequence.flatten().slots[<channel>] の様に取り出した slot のリスト
        oversampling_ratio (int, optional): オーバーサンプル率. Defaults to 1.
        endpoint (bool, optional): True ならサンプル終了点を含む. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: オーバーサンプルした時系列と値系列の組
    """
    s, r = flattened_sequence, oversampling_ratio
    for si in s:
        oversample(si.sampling_points, si.begin, si.end, r)
    x0: list = [oversample(si.sampling_points, si.begin, si.end, r) for si in s]
    y0: list = [si.ufunc(t - si.begin) for t, si in zip(x0, s)]
    dx = (x0[0][1] - x0[0][0]) / r
    if endpoint:
        x = np.concatenate(
            x0
            + [
                np.array(
                    [
                        x0[-1][-1] + dx,
                    ]
                )
            ],
            0,
        )
        y = np.concatenate(
            y0
            + [
                np.array(
                    [
                        0 + 0j,
                    ]
                )
            ],
            0,
        )
    else:
        x = np.concatenate(x, 0)
        y = np.concatenate(y, 0)

    return x, y
