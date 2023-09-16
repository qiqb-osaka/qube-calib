from .qube import AWG, UNIT
import e7awgsw
from e7awgsw import WaveSequence, CaptureParam, AwgCtrl, IqWave

from typing import Final
from contextlib import contextmanager
from collections import namedtuple, deque
from traitlets import HasTraits, Int, Float, observe, link, directional_link
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
import weakref

# The internal time and frequency units are [ns] and [GHz], respectively.

RunningConfig = namedtuple(
    'RunningConfig',
    [
        'contexts',
    ]
)
RunningConfig.__new__.__defaults__ = (deque(),)
__rc__: Final[RunningConfig] = RunningConfig()


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


class Channel(object):

    def __init__(self,frequency,*args,**kw):

        super().__init__(*args,**kw)
        self.frequency = frequency
        
class Control(Channel): pass
class Readout(Channel): pass


class ContextNode ( object ):

    def __init__(self,**kw):
        super().__init__(**kw)
        c = __rc__.contexts
        if len(c):
            c[-1].append(self)


class Slot (  HasTraits, ContextNode ):

    begin = Float(None,allow_none=True)
    duration = Float(None,allow_none=True)
    end = Float(None,allow_none=True)

    def __init__(self,begin=None,duration=None,end=None,**kw):

        self.__mute__ = False
        
        if begin != None and duration != None and end != None:
            raise ValueError('Simultaneously setting "begin", "end" and "duration" is not possible.')
        if duration != None:
            self.duration = duration
        if begin != None:
            self.begin = begin
        if end != None:
            self.end = end
        super().__init__(**kw)
    
    def replace(self):
        
        self.begin = None
        self.end = None
        
    @observe("begin")
    def notify_begin_change(self,e):
        if self.duration == None:
            raise ValueError("'duration' member valiable is not initialized.")
        if e['new'] != None:
            self.end = self.begin + self.duration
        
    @observe("end")
    def notify_end_change(self,e):
        if self.duration == None:
            raise ValueError("'duration' member valiable is not initialized.")
        if e['new'] != None:
            self.begin = self.end - self.duration
        
        
class ChannelMixin ( object ):
    
    def set_channel(self,ch):
        self.ch = ch
        return self


class SlotWithIQ ( Slot ):
    
    SAMPLING_PERIOD: Final[float] = 2*nS
    
    def __init__(self,**kw):
        
        self.__iq__ = None
        super().__init__(**kw)

    def update_iq(self):
        raise NotImplementedError()
    
    @property
    def iq(self):
        self.update_iq()
        return self.__iq__
    
    @observe("duration")
    def notify_duration_change(self,e):
        self.__iq__ = np.zeros(int(self.duration // self.SAMPLING_PERIOD)).astype(complex) # iq data
        
    @property
    def sampling_points(self):
        return self.sampling_points_zero + self.begin # sampling points [ns]
        
    @property
    def sampling_points_zero(self):
        return np.arange(0,len(self.__iq__)) * self.SAMPLING_PERIOD # sampling points [ns]


class Arbit ( SlotWithIQ, ChannelMixin ):

    def __init__(self,**kw):
        
        if 'init' in kw:
            self.init = kw['init']
            del kw['init']
        else:
            self.init = 0+0j
        super().__init__(**kw)
    
    def update_iq(self): 
        self.iq[:] = self.init

class Shadow( HasTraits ):

    SAMPLING_PERIOD: Final[float] = 2*nS

    begin = Float(None,allow_none=True)
    end = Float(None,allow_none=True)

    def __init__(self, body, **kw):

        self.weakref_body = weakref.ref(body)
        super().__init__(**kw)
    
    def replace(self):
        
        self.begin = None
        self.end = None
        
    @observe("begin")
    def notify_begin_change(self,e):
        if self.duration == None:
            raise ValueError("'duration' member valiable is not initialized.")
        if e['new'] != None:
            self.end = self.begin + self.duration
        
    @observe("end")
    def notify_end_change(self,e):
        if self.duration == None:
            raise ValueError("'duration' member valiable is not initialized.")
        if e['new'] != None:
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
        return self.sampling_points_zero + self.begin # sampling points [ns]
        
    @property
    def sampling_points_zero(self):
        return np.arange(0,len(self.__iq__)) * self.SAMPLING_PERIOD # sampling points [ns]



class Range( Slot, ChannelMixin ):
    pass


class Blank ( Slot ):
    pass


class RaisedCosFlatTop ( SlotWithIQ, ChannelMixin ):
    """
    Raised Cosine FlatTopパルスを追加する. 
    ampl : 全体にかかる振幅
    phase : 全体にかかる位相[rad]
    rise_time: 立ち上がり・立ち下がり時間[ns]
    """
    def __init__(self,ampl=1,phase=0*RAD,rise_time=0*nS,**kw):

        self.ampl = ampl
        self.phase = phase
        self.rise_time = rise_time
        super().__init__(**kw)

    def update_iq(self):

        t = self.sampling_points_zero
        flattop_duration = self.duration - self.rise_time * 2

        t1 = 0
        t2 = t1 + self.rise_time # 立ち上がり完了時刻
        t3 = t2 + flattop_duration # 立ち下がり開始時刻 
        t4 = t3 + self.rise_time # 立ち下がり完了時刻 

        cond12 = (t1 <= t) & (t < t2) # 立ち上がり時間領域の条件ブール値
        cond23 = (t2 <= t) & (t < t3) # 一定値領域の条件ブール値
        cond34 = (t3 <= t) & (t < t4) # 立ち下がり時間領域の条件ブール値

        t12 = t[cond12] # 立ち上がり時間領域の時間リスト
        t23 = t[cond23] # 一定値領域の時間リスト
        t34 = t[cond34] # 立ち下がり時間領域の時間リスト
        
        self.__iq__[cond12] = (1.0 - np.cos(np.pi*(t12-t1)/self.rise_time)) / (2.0 + 0.0j) # 立ち上がり時間領域
        self.__iq__[cond23] = 1.0 + 0.0j                                                    # 一定値領域
        self.__iq__[cond34] = (1.0 - np.cos(np.pi*(t4-t34)/self.rise_time)) / (2.0 + 0.0j) # 立ち下がり時間領域
        
        self.__iq__[:] *= self.ampl * np.exp(1j * self.phase)


class Rectangle ( SlotWithIQ, ChannelMixin ):

    def __init__(self,ampl=1,phase=0*RAD,**kw):

        self.ampl = ampl
        self.phase = phase
        super().__init__(**kw)

    def update_iq(self):

        self.__iq__[:] = self.ampl * np.exp(1j * self.phase)


class DequeWithContext( deque ):

    def __enter__(self):

        __rc__.contexts.append(self)

        return self
    
    def __exit__(self, exception_type, exception_value, traceback):

        __rc__.contexts.pop()
        __rc__.contexts[-1].append(self)


class HasFlatten( object ):

    def flatten(self):

        rslt = Sequence()

        for o in self:
            if isinstance(o, HasFlatten):
                for p in o.flatten():
                    rslt.append(p)
            else:
                rslt.append(o)

        return rslt


class Sequence( DequeWithContext, HasFlatten ):

    def replace(self):

        for s in self:
            s.replace()

    @property
    def slots(self):
        r = {}
        for v in self:
            if hasattr(v,'ch'):
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
    
    def __exit__(self, exception_type, exception_value, traceback):

        __rc__.contexts.pop()


class LayoutBase( HasTraits, DequeWithContext, HasFlatten ):

    begin = Float(None,allow_none=True)
    end = Float(None,allow_none=True)


class Series( LayoutBase ):


    def __init__(self, repeats=1, **kw):

        super().__init__(**kw)
        self.repeats = repeats
        self.bodies = Sequence()

    def __exit__(self, exception_type, exception_value, traceback):

        for o in self:
            self.bodies.append(o)
        for i in range(1,self.repeats):
            for o in self.bodies:
                if isinstance(o,Flushright):
                    self.append(FlushrightShadow(o))
                elif isinstance(o,Flushleft):
                    self.append(FlushleftShadow(o))
                elif isinstance(o,Series):
                    self.append(SeriesShadow(o))
                elif isinstance(o,Slot):
                    self.append(Shadow(o))

        for i in range(len(list(self)[:-1])):
            link((self[i],'end'), (self[i+1],'begin'))
        link((self[0],'begin'),(self,'begin'))
        link((self[-1],'end'),(self,'end'))
        super().__exit__(exception_type, exception_value, traceback)
        

class SeriesShadow( HasTraits, HasFlatten, deque):

    begin = Float(None,allow_none=True)
    end = Float(None,allow_none=True)

    def __init__(self, body, **kw):

        super().__init__(**kw)

        self.weakref_body = weakref.ref(body)

        for o in body:
            if isinstance(o,(Flushright,FlushrightShadow)):
                self.append(FlushrightShadow(o))
            elif isinstance(o,(Flushleft,FlushleftShadow)):
                self.append(FlushleftShadow(o))
            elif isinstance(o,(Series,SeriesShadow)):
                self.append(SeriesShadow(o))
            elif isinstance(o,(Slot,Shadow)):
                self.append(Shadow(o))

        for i in range(len(list(self)[:-1])):
            link((self[i],'end'), (self[i+1],'begin'))
        link((self[0],'begin'),(self,'begin'))
        link((self[-1],'end'),(self,'end'))


class Flushright( LayoutBase ):

    leftmost = None

    def __exit__(self, exception_type, exception_value, traceback):

        for i in range(len(list(self)[:-1])):
            link((self[i],'end'), (self[i+1],'end'))
        link((self[0] if self.leftmost is None else self.leftmost,'begin'),(self,'begin'))
        link((self[-1],'end'),(self,'end'))

        super().__exit__(exception_type, exception_value, traceback)


class FlushrightShadow( HasTraits, HasFlatten, deque ):

    begin = Float(None,allow_none=True)
    end = Float(None,allow_none=True)
    leftmost = None

    def __init__(self, body, **kw):

        super().__init__(**kw)

        self.weakref_body = weakref.ref(body)

        for o in body:
            if isinstance(o,Flushright):
                self.append(FlushrightShadow(o))
            elif isinstance(o,Flushleft):
                self.append(FlushleftShadow(o))
            elif isinstance(o,Series):
                self.append(SeriesShadow(o))
            elif isinstance(o,Slot):
                self.append(Shadow(o))
                if body.leftmost == o:
                    self.leftmost = self[-1]

        for i, _ in enumerate(list(self)[:-1]):
            link((self[i],'end'),(self[i+1],'end'))
        link((self[0] if self.leftmost is None else self.leftmost,'begin'),(self,'begin'))
        link((self[-1],'end'),(self,'end'))

    

class Flushleft( LayoutBase ):

    rightmost = None

    def __exit__(self, exception_type, exception_value, traceback):

        for i in range(len(list(self)[:-1])):
            link((self[i],'begin'), (self[i+1],'begin'))
        link((self[0],'begin'),(self,'begin'))
        link((self[-1] if self.rightmost is None else self.rightmost,'end'),(self,'end'))
        super().__exit__(exception_type, exception_value, traceback)

class FlushleftShadow( HasTraits, HasFlatten, deque ):

    begin = Float(None,allow_none=True)
    end = Float(None,allow_none=True)
    rightmost = None

    def __init__(self, body, **kw):

        super().__init__(**kw)

        self.weakref_body = weakref.ref(body)

        for o in body:
            if isinstance(o,Flushright):
                self.append(FlushrightShadow(o))
            elif isinstance(o,Flushleft):
                self.append(FlushleftShadow(o))
            elif isinstance(o,Series):
                self.append(SeriesShadow(o))
            elif isinstance(o,Slot):
                self.append(Shadow(o))
                if body.rightmost == o:
                    self.rightmost = self[-1]

        for i in range(len(list(self)[:-1])):
            link((self[i],'begin'), (self[i+1],'begin'))
        link((self[0],'begin'),(self,'begin'))
        link((self[-1] if self.rightmost is None else self.rightmost,'end'),(self,'end'))

    


def leftmost(slot):

    __rc__.contexts[-1].leftmost = slot


def rightmost(slot):

    __rc__.contexts[-1].rightmost = slot


class DictWithContext(dict):

    def __enter__(self):

        __rc__.contexts.append(deque())

        return self

    def __exit__(self, exception_type, exception_value, traceback):

        __rc__.contexts.pop()


class Sections(DictWithContext):
    
    def __init__(self,*args,**kw):
        
        super().__init__(**kw)
        for k in args:
            self[k] = None

    def __exit__(self, exception_type, exception_value, traceback):

        q = __rc__.contexts.pop()
        for k in self:
            if isinstance(q,Sections):
                raise ValueError('Deep nesting is not allowed.')
            self[k] = q
        if len(self):
            __rc__.contexts[-1].append(self)
        else:
            for o in q:
                for k,v in o.items():
                    if k in self:
                        raise KeyError('Key dupulication is detected.')
                    self[k] = v
        self.place()

    def place(self):
        
        for k, v in self.items():
            if v:
                v0 = v[0]
                v0.begin = v0.prior
                v0.end = v0.begin + v0.duration
                for i in range(1,len(v)):
                    total = v[i-1].prior + v[i-1].duration + v[i-1].post
                    v[i].begin = v[i-1].end + v[i-1].post + (v[i-1].repeats - 1) * total + v[i].prior
                    v[i].end = v[i].begin + v[i].duration


class Allocation(DictWithContext):
    pass


class AllocTable(dict): pass


@contextmanager
def new_sequence():

    __rc__.contexts.append(Sequence())
    try:
        yield __rc__.contexts[-1]
    finally:
        __rc__.contexts.pop()


@contextmanager
def new_section():

    __rc__.contexts.append(Sections())
    try:
        yield __rc__.contexts[-1]
    finally:
        __rc__.contexts.pop()


class SectionBase( ContextNode ):

    def __init__(self, prior=0, duration=BLOCK, post=0, repeats=1):
        
        super().__init__()
        self.prior = prior
        self.duration = duration
        self.post = post
        self.repeats = repeats

    @property
    def total(self):

        return self.prior + self.duration + self.post


class TxSection( SectionBase ):

    def __init__(self, wait=0, duration=BLOCK, blank=0, repeats=1):

        super().__init__(wait, duration, blank, repeats)
        self._iq = None

    @property
    def iq(self):

        if self._iq is None:
            n = int(self.duration // 2)
            self._iq = np.zeros(n).astype(complex)
        return self._iq
    
    @property
    def sampling_points(self):

        return np.arange(len(self.iq)) * 2 + self.begin

class RxSection( SectionBase ):

    def __init__(self, delay=0, duration=BLOCK, blank=0, repeats=1):

        super().__init__(delay, duration, blank, repeats)


class ContextNodeAlloc(object):

    def __init__(
        self,
        begin,
        duration=None,
        end=None,
    ):
        
        self.begin = begin
        self._end_duration(end,duration)
        
    def _end_duration(self,end,duration):
        if duration is None:
            self.end = end
            self.duration = end - self.begin
        elif end is None:
            self.end = self.begin + duration
            self.duration = duration
        else:
            raise()
        
    def alloc(self,*args):

        c = __rc__.contexts
        if len(c):
            d = c[-1]
            for a in args:
                if a not in d:
                    d[a] = deque([self.__class__(self.begin,self.duration)])
                else:
                    d[a].append(self.__class__(self.begin,self.duration))


class Chunk(ContextNodeAlloc):

    def __init__(
        self,
        begin,
        duration=None,
        end=None,
    ):

        super().__init__(begin,duration,end)
        duration_in_samples = n = int(self.duration // 2)
        self.iq = np.zeros(n).astype(complex)
    
    @property
    def sampling_points(self):
        
        return np.arange(len(self.iq)) * 2 + self.begin # sampling points [ns]

class SumSect(ContextNodeAlloc): pass


class UserSequenceBase ( object ):

    def set_duration(self, **kw):
        for k,v in kw.items():
            setattr(getattr(self,k),'duration',v)
        return self
    
    def set_channel(self, **kw):
        for k,v in kw.items():
            setattr(getattr(self,k),'channel',v)
        return self

SequenceBase = UserSequenceBase

class Setup(list):

    def __init__(self,*args):

        super().__init__(args)

    def get(self,arg):

        keys = [k for k,v in self]
        return self[keys.index(arg)][1]

def organize_slots(sequence):
    r = {}
    for v in sequence:
        if hasattr(v,'ch'):
            if v.ch not in r:
                r[v.ch] = deque([v])
            else:
                r[v.ch].append(v)
    return r

def split_awg_unit(alloc_table):
    a,c = {},{}
    for k,v in alloc_table.items():
        for vv in v:
            if isinstance(k,UNIT):
                c[vv] = k
            else:
                a[vv] = k
    return a,c

def __acquire_chunk__(chunk, repeats=1, blank=0):

    n = WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK
    i, q = np.real(32767*chunk.iq).astype(int), np.imag(32767*chunk.iq).astype(int)
    s = IqWave.convert_to_iq_format(i, q, n)

    b = max(int(blank), 0)

    return {
        'iq_samples': s,
        'num_blank_words': b // int(WORD),
        'num_repeats': repeats
    }

def chunks2wseq(chunks, period, repeats):
    # c = chunks
    # if int(c[0].begin) % WORD:
    #     raise('The wait duration must be an integer multiple of ()ns.'.format(int(WORD)))
    # w = e7awgsw.WaveSequence(num_wait_words=int(c[0].begin) // int(WORD), num_repeats=repeats)
    # boundary = b = [int(cc.begin) - int(c[0].begin) for cc in list(c)[1:]] + [period,]
    # for s,p in zip(c,b):
    #     if (p - (s.end - c[0].begin)) % WORD:
    #         raise('The blank duration must be an integer multiple of ()ns.'.format(int(WORD)))
    #     kw = {
    #         'iq_samples': list(zip(np.real(32767*s.iq).astype(int), np.imag(32767*s.iq).astype(int))),
    #         'num_blank_words': int(p - (s.end - c[0].begin)) // int(WORD),
    #         'num_repeats': 1,
    #     }
    #     w.add_chunk(**kw)

    c = chunks
    w = e7awgsw.WaveSequence(num_wait_words=int(c[0].prior) // int(WORD), num_repeats=repeats)

    for j in range(len(c)-1):
        if c[j].repeats > 1:
            w.add_chunk(**__acquire_chunk__(c[j],c[j].repeats-1,c[j].post + c[j].prior))
        w.add_chunk(**__acquire_chunk__(c[j],1,c[j].post + c[j+1].prior))
    if c[-1].repeats > 1:
        w.add_chunk(**__acquire_chunk__(c[-1],c[-1].repeats-1,c[-1].post + c[-1].prior))
    w.add_chunk(**__acquire_chunk__(c[-1],1,(period - c[-1].end)))

    return w


def sect2capt(section,period,repeats):
    s = section
    p = e7awgsw.CaptureParam()
    p.num_integ_sections = repeats

    if int(s[0].prior) % int(2*WORD):
        raise('The capture_delay must be an integer multiple of ()ns.'.format(int(2*WORD)))
    p.capture_delay = int(s[0].prior) // int(WORD)

    for j in range(len(s)-1):
        for i in range(s[j].repeats-1):
            duration = int(s[j].duration) // int(WORD)
            blank = int(s[j].post + s[j].prior) // int(WORD)
            p.add_sum_section(num_words=duration, num_post_blank_words=blank)
        duration = int(s[j].duration) // int(WORD)
        blank = int(s[j].post + s[j+1].prior) // int(WORD)
        p.add_sum_section(num_words=duration, num_post_blank_words=blank)
    for i in range(s[-1].repeats-1):
        duration = int(s[-1].duration) // int(WORD)
        blank = int(s[-1].post + s[-1].prior) // int(WORD)
        p.add_sum_section(num_words=duration, num_post_blank_words=blank)
    duration = int(s[-1].duration) // int(WORD)
    blank = int(period - s[-1].end) // int(WORD)
    p.add_sum_section(num_words=duration, num_post_blank_words=blank)
    return p


def body(x):
    if not isinstance(x,Shadow):
        return x
    else:
        b = x.weakref_body()
        if not isinstance(b,Shadow):
            return b
        else:
            return body(b)


def convert(sequence, alloc_table, period, repeats=1, section=None, warn=False):

    # channel2slot = r = organize_slots(sequence) # channel -> slot
    channel2slot = r = sequence.slots
    a,c = split_awg_unit(alloc_table) # channel -> txport, rxport
    
    # Tx チャネルはデータ変調
    for k in a:
        if k not in r:
            with warnings.catch_warnings():
                if not warn:
                    warnings.simplefilter('ignore')
                warnings.warn('Channel {} is Iqnored.'.format(k))
            continue
        for v in r[k]:
            if isinstance(body(v),SlotWithIQ):
                m = a[k].modulation_frequency(mhz=k.frequency*1e+3)*MHz
                t = v.sampling_points
                v.miq = v.iq * np.exp(1j*2*np.pi*(m*t))
            # 位相合わせについては後日

    # 各AWG/UNITの各セクション毎に属する Slot のリストの対応表を作る
    section2slots = {}
    for k,v in section.items(): # k:AWG,v:List[TxSection] or k:UNIT,v:List[RxSection]
        for vv in v: # vv:Union[TxSection,RxSection]
            if vv not in section2slots:
                section2slots[vv] = deque([])
            if k not in alloc_table: # k:Union[AWG,UNIT] 
                with warnings.catch_warnings():
                    if not warn:
                        warnings.simplefilter('ignore')
                    warnings.warn('AWG|UNIT {} is Iqnored.'.format(k))
                continue
            for c in alloc_table[k]: # c:Channel
                if c not in r: # sequence の定義内で使っていない論理チャネルがあり得る
                    with warnings.catch_warnings():
                        if not warn:
                            warnings.simplefilter('ignore')
                        warnings.warn('Channel {} is Iqnored.'.format(c))
                    continue
                for s in r[c]:
                    for i in range(vv.repeats):
                        if vv.begin + i * vv.total <= s.begin and s.begin + s.duration <= vv.end + i * vv.total:
                            bawg = isinstance(k,AWG) and isinstance(body(s),SlotWithIQ)
                            bunit = isinstance(k,UNIT) and isinstance(body(s),Range)
                            if bawg and i == 0:
                                section2slots[vv].append(s)
                            if bunit:
                                section2slots[vv].append(s)

    # 各セクション毎に Chunk を合成する
    awgs = [k for k in alloc_table if isinstance(k,AWG)]
    for i,k in enumerate(awgs):
        if k in section:
            for s in section[k]:
                t = s.sampling_points
                s.iq[:] = 0
                ss = section2slots[s]
                for v in ss:
                    rng = (v.begin <= t) * (t < v.end)
                    s.iq[rng] += v.miq / len(ss)
    
    # # 束ねるチャネルを WaveSequence に変換
    awgs = [k for k in alloc_table if isinstance(k,AWG)] # alloc_table から AWG 関連だけ抜き出す
    wseqs = [(k, chunks2wseq(section[k], period, repeats)) for k in awgs if k in section] # chunk obj を wseq へ変換する

    units = [k for k in alloc_table if isinstance(k,UNIT)] 
    capts = [(k,sect2capt(section[k],period,repeats)) for k in units if k in section]
    
    return Setup(*(wseqs + capts))
    

def find_sequence(sequence, target):

    def recursive_dfs(x,target,collection):
        for o in x:
            if isinstance(o,target):
                if o.repeats > 1:
                    collection.append(o)
            if isinstance(o,HasFlatten):
                yield from recursive_dfs(o,target,collection)
            else:
                yield (o,collection)

    c = list(recursive_dfs(sequence,target,[]))
    c = [o for _, v in c for o in v]
    buf = []
    for o in c:
        if o not in buf:
            buf.append(o)
    return buf

def acquire_section(sequence, alloc_table, section=None):

    if section is None:
        section = Sections()

    tx = acquire_tx_section(sequence, alloc_table)
    rx = acquire_rx_section(sequence.flatten(), alloc_table)

    for k, v in tx.items():
        section[k] = v

    for k, v in rx.items():
        section[k] = v

    section.place()

    return section

def acquire_tx_section(sequence, alloc_table, section=None):

    log2phy = {o: k for k,v in alloc_table.items() if isinstance(k,AWG) for o in v}
    logch = [k for k in sequence.flatten().slots if k is not None]
    phych = [k for k in alloc_table for l in logch if l in alloc_table[k] and isinstance(k,AWG)]

    # AWG と Series の対応辞書を作る
    series = find_sequence(sequence, Series)
    awg2series = d = dict()
    for p in phych:
        d[p] = deque()
        for s in series:
            if p in [log2phy[k] for k in s.flatten().slots.keys() if k is not None]:
                d[p].append(s)

    # 重複した Series を削除する
    for k, v in awg2series.items():
        item = lambda x: (x.begin, (x.end - x.begin) / x.repeats, x)
        rng = r = sorted([item(o) for o in v], key=lambda x:x[1], reverse=True)
        buffer = b = Sequence()
        isout = lambda x, y: (y[0] + y[1] < x[0]) + (x[0] + x[1] < y[0])
        isin = lambda x, y: (x[0] < y[0]) * (y[0] + y[1] < x[0] + x[1])
        for i, o in enumerate(rng[:-1]):
            skip = False
            for m in rng[i+1:]:
                if skip:
                    continue
                if isin(o, m):
                    skip = True
                    continue
                b.append(o[2])
        awg2series[k] = buffer
        
    # AWG と Slots の対応辞書を作る
    awg2slots = dict()
    slots = sequence.flatten().slots
    for phychi in phych:
        for k, v in slots.items():
            if phychi is log2phy[k] if k is not None else None:
                awg2slots[phychi] = deque()
                for o in v:
                    if isinstance(body(o),SlotWithIQ):
                        if o not in awg2series[phychi].flatten():
                            awg2slots[phychi].append(o)

    if section is None:
        section = Sections()

    for phychi in phych:

        slots = np.array([(s.begin,(s.end-s.begin)/s.repeats,s.end,s.repeats) for s in awg2series[phychi] if isinstance(s,Series)])
        if not slots:
            if not awg2slots[phychi]:
                continue
            begin = min([o.begin for o in awg2slots[phychi]])
            end = max([o.end for o in awg2slots[phychi]])
            if begin % int(WORD):
                raise ValueError('begin is not aligned')
            if (end - begin) % int(BLOCK):
                raise ValueError('durations is not aligned')
            section[phychi] = deque()
            section[phychi].append(TxSection(wait=begin,duration=end-begin,blank=0,repeats=1))
            continue
        slots = slots[np.argsort(slots[:,0])]

        section[phychi] = deque()

        begin, duration, end, repeats = slots[0]
        section[phychi].append(TxSection(wait=begin,duration=duration,blank=0,repeats=int(repeats)))
        for i in range(1,slots.shape[0]):
            begin, duration, end, repeats = slots[i,0], slots[i,1], slots[i-1,2], slots[i,3]
            section[phychi].append(TxSection(wait=begin-end,duration=duration,blank=0,repeats=int(repeats)))

    return section


def acquire_rx_section(sequence, alloc_table, section=None):

    if section is None:
        section = Sections()

    logch = [k for k in sequence.slots if isinstance(k,Readout)]
    phych = [k for k in alloc_table for l in logch if l in alloc_table[k] and isinstance(k,UNIT)]

    for phychi in phych:

        slots = np.array([(s.begin,s.duration,s.end) for s in sequence if isinstance(body(s),Range) and s.ch in alloc_table[phychi]])
        slots = slots[np.argsort(slots[:,0])]
        
        if (slots % int(WORD)).sum():
            raise ValueError('!!!')
        section[phychi] = deque()

        begin, duration, end = slots[0]
        section[phychi].append(RxSection(delay=begin,duration=duration,blank=0))
        for i in range(1,slots.shape[0]):
            begin, duration, end = slots[i,0], slots[i,1], slots[i-1,2]
            section[phychi].append(RxSection(delay=begin-end,duration=duration,blank=0))
            
    return section


def plot_send_recv(fig, data, mag=False):

    n = len([vv for k,v in data.items() for vv in v])
    i = 1
    for k,v in data.items():
        for vv in v:
            if fig.axes:
                ax = fig.add_subplot(n,1,i, sharex=ax1)
            else:
                ax = ax1 = fig.add_subplot(n,1,i)
            if mag:
                ax.plot(np.abs(vv))
            else:
                ax.plot(np.real(vv))
                ax.plot(np.imag(vv))
            i += 1

def plot_setup(fig,setup,capture_delay=0):
    for i,tpl in enumerate(setup):
        k,v = tpl
        if i == 0:
            ax = ax1 = fig.add_subplot(len(setup),1,i+1)
        else:
            ax = fig.add_subplot(len(setup),1,i+1, sharex=ax1)
        if isinstance(k,AWG):
            blank = [o.num_blank_words * int(WORDs) for o in v.chunk_list]
            begin = v.num_wait_words * int(WORDs)
            for cc,bb in zip(v.chunk_list,blank):
                iq = np.array(cc.wave_data.samples)
                t = begin + np.arange(len(iq)) * 2
                for _ in range(cc.num_repeats-1):
                    ax.plot(t,iq[:,0],'b')
                    ax.plot(t,iq[:,1],'r')
                    ax.set_ylim(-32767*1.2,32767*1.2)
                    t += len(iq) * 2 + bb
                ax.plot(t,iq[:,0],'b')
                ax.plot(t,iq[:,1],'r')
                ax.set_ylim(-32767*1.2,32767*1.2)
                begin = t[-1] + bb
        else:
            blank = [o[1] * int(WORDs) for o in v.sum_section_list]
            begin = (v.capture_delay - capture_delay) * int(WORDs)
            for s,b in zip(v.sum_section_list,blank):
                duration = int(s[0]) * int(WORDs)
                ax.add_patch(patches.Rectangle(xy=(begin,-32767),width=duration,height=2*32767))
                begin += duration + b
                ax.set_ylim(-32767*1.2,32767*1.2)

def plot_sequence(fig,sequence):
    slots = sequence.flatten().slots
    for i,logch in enumerate(slots):
        if i == 0:
            ax = ax1 = fig.add_subplot(len(slots),1,i+1)
        else:
            ax = fig.add_subplot(len(slots),1,i+1, sharex=ax1)
        for s in slots[logch]:
            if isinstance(body(s),Range) or isinstance(body(s),Blank):
                begin = int(s.begin)
                duration = int(s.duration)
                ax.add_patch(patches.Rectangle(xy=(begin,-1),width=duration,height=2))
                ax.set_ylim(-1.2,1.2)
            else:
                t = s.sampling_points
                ax.plot(t,np.real(s.iq))
                ax.plot(t,np.imag(s.iq))
                ax.set_ylim(-1.2,1.2)

def plot_section(fig,section):
    for i,phych in enumerate(section):
        if i == 0:
            ax = ax1 = fig.add_subplot(len(section),1,i+1)
        else:
            ax = fig.add_subplot(len(section),1,i+1,sharex=ax1)
        for s in section[phych]:
            begin = int(s.begin)
            duration = int(s.duration)
            total = int(s.prior + s.duration + s.post)
            for i in range(s.repeats):
                ax.add_patch(patches.Rectangle(xy=(begin + i * total,-1),width=duration,height=2,fill=False))
        ax.set_xlim(0,s.end)
        ax.set_ylim(-1.2,1.2)
