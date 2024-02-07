from __future__ import annotations

import weakref
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final, Optional

import matplotlib.patches as patches
import numpy as np
from traitlets import Float, HasTraits, link, observe

from .units import RAD, nS

# The internal time and frequency units are [ns] and [GHz], respectively.


@dataclass
class RunningConfig:
    contexts: deque = deque()


__rc__: Final[RunningConfig] = RunningConfig()


def ceil(value: float, unit: float = 1) -> float:
    """valueの値を指定したunitの単位でその要素以上の最も近い数値に丸める

    Args:
        value (float): 対象の値
        unit (float, optional): 丸める単位. Defaults to 1.

    Returns:
        float: 丸めた値
    """
    MAGNIFIER = 1_000_000

    # unit が循環小数の場合に丸めなければならない場合がある
    if value % unit < 1e-9:
        return value

    value, unit = int(value * MAGNIFIER), int(unit * MAGNIFIER)
    if value % unit:
        return int((value // unit + 1) * unit) / MAGNIFIER
    else:
        return int((value // unit) * unit) / MAGNIFIER


class Channel:
    def __init__(self, frequency: float, *args: Any, **kw: Any):
        self.frequency = frequency


class Control(Channel):
    pass


class Readout(Channel):
    pass


class ContextNode:
    def __init__(self, **kw: Any):
        c = __rc__.contexts
        if len(c):
            c[-1].append(self)


class Slot(HasTraits, ContextNode):
    begin = Float(None, allow_none=True)
    duration = Float(None, allow_none=True)
    end = Float(None, allow_none=True)

    def __init__(
        self,
        begin: Optional[int | float] = None,
        duration: Optional[int | float] = None,
        end: Optional[int | float] = None,
        **kw: Any,
    ):
        self.__mute__ = False

        if (begin is not None) and (duration is not None) and (end is not None):
            raise ValueError(
                'Simultaneously setting "begin", "end" and "duration" is not possible.'
            )
        if duration is not None:
            self.duration = duration
        if begin is not None:
            self.begin = begin
        if end is not None:
            self.end = end

        super().__init__(**kw)

    def replace(self) -> None:
        self.begin = None
        self.end = None

    @observe("begin")
    def notify_begin_change(self, e: Any) -> None:
        if self.duration is None:
            raise ValueError("'duration' member valiable is not initialized.")
        if self.begin is None:
            raise ValueError("'begin' member valiable is not initialized.")
        if e["new"] is not None:
            self.end = self.begin + self.duration

    @observe("end")
    def notify_end_change(self, e: Any) -> None:
        if self.duration is None:
            raise ValueError("'duration' member valiable is not initialized.")
        if self.end is None:
            raise ValueError("'end' member valiable is not initialized.")
        if e["new"] is not None:
            self.begin = self.end - self.duration


class ChannelMixin(object):
    def set_channel(self, ch):
        self.ch = ch
        return self


class SlotWithIQ(Slot):
    SAMPLING_PERIOD: Final[float] = 2 * nS

    def __init__(self, **kw):
        self.__iq__ = None
        self.__virtual_z_theta__ = 0

        super().__init__(**kw)

    def func(self, t):
        raise NotImplementedError()

    def ufunc(self, t):
        return np.frompyfunc(self.func, 1, 1)(t).astype(complex)

    def virtual_z(self, theta):
        self.__virtual_z_theta__ = theta

    @property
    def iq(self):
        if self.begin == None or self.end == None:
            raise ValueError("Either or both 'begin' and 'end' are not initialized.")
        self.__iq__ = self.ufunc(self.sampling_points_zero) * np.exp(
            1j * self.__virtual_z_theta__
        )
        return self.__iq__

    @property
    def sampling_points(self):
        return np.arange(
            ceil(self.begin, 2 * nS), ceil(self.end, 2 * nS), self.SAMPLING_PERIOD
        )  # sampling points [ns]

    @property
    def sampling_points_zero(self):
        return self.sampling_points - self.begin  # sampling points [ns]


class Arbit(SlotWithIQ, ChannelMixin):
    """ "サンプリング点を直接与えるためのオブジェクト"""

    def __init__(self, **kw):
        if "init" in kw:
            self.init = kw["init"]
            del kw["init"]
        else:
            self.init = 0 + 0j
        super().__init__(**kw)

    @observe("duration")
    def notify_duration_change(self, e):
        self.__iq__ = np.zeros(int(self.duration // self.SAMPLING_PERIOD)).astype(
            complex
        )  # iq data

    def ufunc(self, t=None):
        """
        iq データを格納している numpy array への参照を返す

        Parameters
        ----------
        t : numpy.ndarray(float)
            与えると対象の時間列に則した点数にサンプルした iq データを返す
        """
        if t is None:
            return self.__iq__
        else:
            rslt = np.zeros(t.shape).astype(complex)
            b, e = self.begin, self.end
            iq = self.__iq__
            idx = (ceil(b, 2) <= t + b) & (t + b < ceil(e, 2))
            # 開始点が 31.999968 の様に誤差を含む場合に開始点を含む
            # idx[0] = True if ceil(b, 2) - (t + b)[idx][0] < 1e-4 else False
            # 終点が 41.999968 の様に誤差を含む場合に終点を除外する
            # idx[-1] = False if ceil(e,2) - (t + b)[idx][-1] < 1e-4 else True
            l, m = t[idx].shape[0], iq.shape[0]
            n = int(l // m)
            v = (
                np.stack(
                    n
                    * [
                        iq,
                    ]
                )
                .transpose()
                .reshape(n * m)
            )
            o = v.shape[0]

            if l == o:
                rslt[idx] = v
            elif l < o:
                rslt[idx] = v[: (l - o)]
            else:
                idx[(o - l) :] = False
                rslt[idx] = v

            return rslt

    @property
    def iq_array(self):
        return self.__iq__


class Shadow(HasTraits):
    """ "繰り返し機能を使うための影オブジェクト"""

    SAMPLING_PERIOD: Final[float] = 2 * nS

    begin = Float(None, allow_none=True)
    end = Float(None, allow_none=True)

    def __init__(self, body, **kw):
        self.weakref_body = weakref.ref(body)
        super().__init__(**kw)

    def replace(self):
        self.begin = None
        self.end = None

    @observe("begin")
    def notify_begin_change(self, e):
        if self.duration == None:
            raise ValueError("'duration' member valiable is not initialized.")
        if e["new"] != None:
            self.end = self.begin + self.duration

    @observe("end")
    def notify_end_change(self, e):
        if self.duration == None:
            raise ValueError("'duration' member valiable is not initialized.")
        if e["new"] != None:
            self.begin = self.end - self.duration

    @property
    def ch(self):
        return self.weakref_body().ch

    @property
    def duration(self):
        return self.weakref_body().duration

    @property
    def iq(self):
        return self.weakref_body().iq

    @property
    def __iq__(self):
        return self.weakref_body().__iq__

    @property
    def sampling_points(self):
        return np.arange(
            ceil(self.begin, 2), ceil(self.end, 2), self.SAMPLING_PERIOD
        )  # sampling points [ns]

    @property
    def sampling_points_zero(self):
        return self.sampling_points - self.begin  # sampling points [ns]


class Range(Slot, ChannelMixin):
    pass


Capture = Range


class Blank(Slot):
    SAMPLING_PERIOD: Final[float] = 2 * nS

    @property
    def sampling_points(self) -> np.ndarray:
        return np.arange(
            ceil(self.begin, 2), ceil(self.end, 2), self.SAMPLING_PERIOD
        )  # sampling points [ns]

    def func(self, t: float) -> float:
        return 0.0

    def ufunc(self, t: float) -> np.ufunc:
        return np.frompyfunc(self.func, 1, 1)(t).astype(complex)


class VirtualZ(Slot, ChannelMixin):
    def __init__(self, theta=0, begin=None, end=None, **kw):
        self.theta = theta

        super().__init__(begin, 0, end, **kw)
        ChannelMixin.__init__(self)


class RaisedCosFlatTop(SlotWithIQ, ChannelMixin):
    """
    Raised Cosine FlatTopパルス

    Attributes
    ----------
    ampl : float
        全体にかかる振幅
    phase : float
        全体にかかる位相[rad]
    rise_time: float
        立ち上がり・立ち下がり時間[ns]
    """

    def __init__(
        self, ampl: float = 1, phase: float = 0 * RAD, rise_time: float = 0 * nS, **kw
    ):
        self.ampl = ampl
        self.phase = phase
        self.rise_time = rise_time

        super().__init__(**kw)

    def func(self, t: float) -> float:
        flattop_duration = self.duration - self.rise_time * 2

        t1 = 0
        t2 = t1 + self.rise_time  # 立ち上がり完了時刻
        t3 = t2 + flattop_duration  # 立ち下がり開始時刻
        t4 = t3 + self.rise_time  # 立ち下がり完了時刻

        if (t1 <= t) & (t < t2):  # 立ち上がり時間領域の条件ブール値
            v = (1.0 - np.cos(np.pi * (t - t1) / self.rise_time)) / (
                2.0 + 0.0j
            )  # 立ち上がり時間領域
        elif (t2 <= t) & (t < t3):  # 一定値領域の条件ブール値
            v = 1.0 + 0.0j  # 一定値領域
        elif (t3 <= t) & (t < t4):  # 立ち下がり時間領域の条件ブール値
            v = (1.0 - np.cos(np.pi * (t4 - t) / self.rise_time)) / (
                2.0 + 0.0j
            )  # 立ち下がり時間領域
        else:
            v = 0.0 + 0.0j

        return v * self.ampl * np.exp(1j * self.phase)


class Rectangle(SlotWithIQ, ChannelMixin):
    def __init__(self, ampl=1, phase=0 * RAD, **kw):
        self.ampl = ampl
        self.phase = phase
        super().__init__(**kw)

    def func(self, t):
        return (1 + 0j) * self.ampl * np.exp(1j * self.phase)


class DequeWithContext(deque):
    def __enter__(self):
        __rc__.contexts.append(self)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        __rc__.contexts.pop()
        __rc__.contexts[-1].append(self)


class HasFlatten(object):
    def flatten(self):
        rslt = Sequence()

        for o in self:
            if isinstance(o, HasFlatten):
                for p in o.flatten():
                    rslt.append(p)
            else:
                rslt.append(o)

        return rslt


class Sequence(DequeWithContext, HasFlatten):
    def replace(self):
        for s in self:
            s.replace()

    @property
    def slots(self):
        r = {}
        for v in self:
            if hasattr(v, "ch"):
                if v.ch not in r:
                    r[v.ch] = deque([v])
                else:
                    r[v.ch].append(v)
            else:
                if None not in r:
                    r[None] = deque([v])
                else:
                    r[None].append(v)
        return r

    def apply_virtual_z(self):
        # Apply Virtual Z
        theta = {}
        for o in self.flatten():
            if isinstance(o, VirtualZ):
                if hasattr(o, "ch"):
                    if o.ch not in theta:
                        theta[o.ch] = o.theta
                    else:
                        theta[o.ch] += o.theta
            if isinstance(o, SlotWithIQ):
                if hasattr(o, "ch"):
                    if o.ch not in theta:
                        theta[o.ch] = 0
                    o.virtual_z(theta[o.ch])

    def __exit__(self, exception_type, exception_value, traceback):
        __rc__.contexts.pop()
        self.apply_virtual_z()


class LayoutBase(HasTraits, DequeWithContext, HasFlatten):
    begin = Float(None, allow_none=True)
    end = Float(None, allow_none=True)

    def replace(self):
        for s in self:
            s.replace()


class Series(LayoutBase):
    def __init__(self, repeats=1, **kw):
        super().__init__(**kw)
        self.repeats = repeats
        self.bodies = Sequence()

    def __exit__(self, exception_type, exception_value, traceback):
        for o in self:
            self.bodies.append(o)
        for i in range(1, self.repeats):
            for o in self.bodies:
                if isinstance(o, Flushright):
                    self.append(FlushrightShadow(o))
                elif isinstance(o, Flushleft):
                    self.append(FlushleftShadow(o))
                elif isinstance(o, Series):
                    self.append(SeriesShadow(o))
                elif isinstance(o, Slot):
                    self.append(Shadow(o))

        for i in range(len(list(self)[:-1])):
            link((self[i], "end"), (self[i + 1], "begin"))
        link((self[0], "begin"), (self, "begin"))
        link((self[-1], "end"), (self, "end"))

        super().__exit__(exception_type, exception_value, traceback)


class SeriesShadow(HasTraits, HasFlatten, deque):
    begin = Float(None, allow_none=True)
    end = Float(None, allow_none=True)

    def __init__(self, body, **kw):
        super().__init__(**kw)

        self.weakref_body = weakref.ref(body)

        for o in body:
            if isinstance(o, (Flushright, FlushrightShadow)):
                self.append(FlushrightShadow(o))
            elif isinstance(o, (Flushleft, FlushleftShadow)):
                self.append(FlushleftShadow(o))
            elif isinstance(o, (Series, SeriesShadow)):
                self.append(SeriesShadow(o))
            elif isinstance(o, (Slot, Shadow)):
                self.append(Shadow(o))

        for i in range(len(list(self)[:-1])):
            link((self[i], "end"), (self[i + 1], "begin"))
        link((self[0], "begin"), (self, "begin"))
        link((self[-1], "end"), (self, "end"))


class Flushright(LayoutBase):
    leftmost = None

    def __exit__(self, exception_type, exception_value, traceback):
        for i in range(len(list(self)[:-1])):
            link((self[i], "end"), (self[i + 1], "end"))
        link(
            (self[0] if self.leftmost is None else self.leftmost, "begin"),
            (self, "begin"),
        )
        link((self[-1], "end"), (self, "end"))

        super().__exit__(exception_type, exception_value, traceback)


class FlushrightShadow(HasTraits, HasFlatten, deque):
    begin = Float(None, allow_none=True)
    end = Float(None, allow_none=True)
    leftmost = None

    def __init__(self, body, **kw):
        super().__init__(**kw)

        self.weakref_body = weakref.ref(body)

        for o in body:
            if isinstance(o, Flushright):
                self.append(FlushrightShadow(o))
            elif isinstance(o, Flushleft):
                self.append(FlushleftShadow(o))
            elif isinstance(o, Series):
                self.append(SeriesShadow(o))
            elif isinstance(o, Slot):
                self.append(Shadow(o))
                if body.leftmost == o:
                    self.leftmost = self[-1]

        for i, _ in enumerate(list(self)[:-1]):
            link((self[i], "end"), (self[i + 1], "end"))
        link(
            (self[0] if self.leftmost is None else self.leftmost, "begin"),
            (self, "begin"),
        )
        link((self[-1], "end"), (self, "end"))


class Flushleft(LayoutBase):
    rightmost = None

    def __exit__(self, exception_type, exception_value, traceback):
        for i in range(len(list(self)[:-1])):
            link((self[i], "begin"), (self[i + 1], "begin"))
        link((self[0], "begin"), (self, "begin"))
        link(
            (self[-1] if self.rightmost is None else self.rightmost, "end"),
            (self, "end"),
        )

        super().__exit__(exception_type, exception_value, traceback)


class FlushleftShadow(HasTraits, HasFlatten, deque):
    begin = Float(None, allow_none=True)
    end = Float(None, allow_none=True)
    rightmost = None

    def __init__(self, body, **kw):
        super().__init__(**kw)

        self.weakref_body = weakref.ref(body)

        for o in body:
            if isinstance(o, Flushright):
                self.append(FlushrightShadow(o))
            elif isinstance(o, Flushleft):
                self.append(FlushleftShadow(o))
            elif isinstance(o, Series):
                self.append(SeriesShadow(o))
            elif isinstance(o, Slot):
                self.append(Shadow(o))
                if body.rightmost == o:
                    self.rightmost = self[-1]

        for i in range(len(list(self)[:-1])):
            link((self[i], "begin"), (self[i + 1], "begin"))
        link((self[0], "begin"), (self, "begin"))
        link(
            (self[-1] if self.rightmost is None else self.rightmost, "end"),
            (self, "end"),
        )


def leftmost(slot):
    __rc__.contexts[-1].leftmost = slot


def rightmost(slot):
    __rc__.contexts[-1].rightmost = slot


@contextmanager
def new_sequence():
    __rc__.contexts.append(Sequence())
    try:
        yield __rc__.contexts[-1]
    finally:
        __rc__.contexts.pop()


class UserSequenceBase(object):
    def set_duration(self, **kw):
        for k, v in kw.items():
            setattr(getattr(self, k), "duration", v)
        return self

    def set_channel(self, **kw):
        for k, v in kw.items():
            setattr(getattr(self, k), "channel", v)
        return self


SequenceBase = UserSequenceBase


def organize_slots(sequence):
    r = {}
    for v in sequence:
        if hasattr(v, "ch"):
            if v.ch not in r:
                r[v.ch] = deque([v])
            else:
                r[v.ch].append(v)
    return r


def body(x):
    if not isinstance(x, Shadow):
        return x
    else:
        b = x.weakref_body()
        if not isinstance(b, Shadow):
            return b
        else:
            return body(b)


def find_sequence(sequence, target):
    def recursive_dfs(x, target, collection):
        for o in x:
            if isinstance(o, target):
                if o.repeats > 1:
                    collection.append(o)
            if isinstance(o, HasFlatten):
                yield from recursive_dfs(o, target, collection)
            else:
                yield (o, collection)

    c = list(recursive_dfs(sequence, target, []))
    c = [o for _, v in c for o in v]
    buf = []
    for o in c:
        if o not in buf:
            buf.append(o)
    return buf


def plot_sequence(fig, sequence):
    slots = sequence.flatten().slots
    for i, logch in enumerate(slots):
        if i == 0:
            ax = ax1 = fig.add_subplot(len(slots), 1, i + 1)
        else:
            ax = fig.add_subplot(len(slots), 1, i + 1, sharex=ax1)
        for s in slots[logch]:
            if isinstance(body(s), Range) or isinstance(body(s), Blank):
                begin = int(s.begin)
                duration = int(s.duration)
                ax.add_patch(
                    patches.Rectangle(xy=(begin, -1), width=duration, height=2)
                )
                ax.set_ylim(-1.2, 1.2)
            elif isinstance(body(s), VirtualZ):
                pass
            else:
                t = s.sampling_points
                ax.plot(t, np.real(s.iq))
                ax.plot(t, np.imag(s.iq))
                ax.set_ylim(-1.2, 1.2)
