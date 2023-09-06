from .qube import AWG, UNIT
import e7awgsw
from e7awgsw import WaveSequence, CaptureParam

from typing import Final
from contextlib import contextmanager
from collections import namedtuple, deque
from traitlets import HasTraits, Int, Float, observe, link
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

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


class Slot ( ContextNode, HasTraits ):

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


class Sequence(deque):

    def replace(self):

        for s in self:
            s.replace()


class Section(dict): pass
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

    __rc__.contexts.append(Section())
    try:
        yield __rc__.contexts[-1]
    finally:
        __rc__.contexts.pop()


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


class SequenceBase ( object ):

    def set_duration(self, **kw):
        for k,v in kw.items():
            setattr(getattr(self,k),'duration',v)
        return self
    
    def set_channel(self, **kw):
        for k,v in kw.items():
            setattr(getattr(self,k),'channel',v)
        return self


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

def chunks2wseq(chunks, period, repeats):
    c = chunks
    if int(c[0].begin) % WORD:
        raise('The wait duration must be an integer multiple of ()ns.'.format(int(WORD)))
    w = e7awgsw.WaveSequence(num_wait_words=int(c[0].begin) // int(WORD), num_repeats=repeats)
    boundary = b = [int(cc.begin) - int(c[0].begin) for cc in list(c)[1:]] + [period,]
    for s,p in zip(c,b):
        if (p - (s.end - c[0].begin)) % WORD:
            raise('The blank duration must be an integer multiple of ()ns.'.format(int(WORD)))
        kw = {
            'iq_samples': list(zip(np.real(32767*s.iq).astype(int), np.imag(32767*s.iq).astype(int))),
            'num_blank_words': int(p - (s.end - c[0].begin)) // int(WORD),
            'num_repeats': 1,
        }
        w.add_chunk(**kw)
    return w


def sect2capt(section,period,repeats):
    s = section
    p = e7awgsw.CaptureParam()
    p.num_integ_sections = repeats
    if int(s[0].begin) % int(2*WORD):
        raise('The capture_delay must be an integer multiple of ()ns.'.format(int(2*WORD)))
    p.capture_delay = int(s[0].begin) // int(WORD)
    boundary = b = [int(ss.begin) - int(s[0].begin) for ss in s][1:] + [period,]
    for ss,bb in zip(s,b):
        if int(ss.duration) % int(WORD):
            raise('error!!')
        duration = int(ss.duration) // int(WORD)
        if int(bb - (ss.end - s[0].begin)) % int(WORD):
            raise('error!!')
        blank =  int(bb - (ss.end - s[0].begin)) // int(WORD)
        p.add_sum_section(num_words=duration, num_post_blank_words=blank)
    return p



def convert(sequence, section, alloc_table, period, repeats=1, warn=False):
    channel2slot = r = organize_slots(sequence) # channel -> slot
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
            if isinstance(v,SlotWithIQ):
                m = a[k].modulation_frequency(mhz=k.frequency*1e+3)*MHz
                t = v.sampling_points
                v.miq = v.iq * np.exp(1j*2*np.pi*(m*t))
            # 位相合わせについては後日

    # 各AWG/UNITの各セクション毎に属する Slot のリストの対応表を作る
    section2slots = {}
    for k,v in section.items(): # k:AWG,v:List[Chunk] or k:UNIT,v:List[SumSect]
        for vv in v: # vv:Union[Chunk,SumSect]
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
                    if vv.begin <= s.begin and s.begin + s.duration <= vv.end:
                        bawg = isinstance(k,AWG) and isinstance(s,SlotWithIQ)
                        bunit = isinstance(k,UNIT) and isinstance(s,Range)
                        if bawg or bunit:
                            section2slots[vv].append(s)
                            
    # 各セクション毎に Chunk を合成する
    awgs = [k for k in alloc_table if isinstance(k,AWG)]
    for i,k in enumerate(awgs):
        for s in section[k]:
            t = s.sampling_points
            s.iq[:] = 0
            ss = section2slots[s]
            for v in ss:
                rng = (v.begin <= t) * (t < v.end)
                s.iq[rng] += v.miq / len(ss)
    
    # 束ねるチャネルを WaveSequence に変換
    awgs = [k for k in alloc_table if isinstance(k,AWG)] # alloc_table から AWG 関連だけ抜き出す
    wseqs = [(k, chunks2wseq(section[k], period, repeats)) for k in awgs] # chunk obj を wseq へ変換する

    units = [k for k in alloc_table if isinstance(k,UNIT)] 
    capts = [(k,sect2capt(section[k],period,repeats)) for k in units]
    
    return Setup(*(wseqs + capts))
    
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
                t = np.arange(len(iq)) * 2 + begin
                begin = t[-1] + bb
                ax.plot(t,iq[:,0],'b')
                ax.plot(t,iq[:,1],'r')
                ax.set_ylim(-32767*1.2,32767*1.2)
        else:
            blank = [o[1] * int(WORDs) for o in v.sum_section_list]
            begin = v.capture_delay - capture_delay
            for s,b in zip(v.sum_section_list,blank):
                duration = int(s[0]) * int(WORDs)
                ax.add_patch(patches.Rectangle(xy=(begin,-32767),width=duration,height=2*32767))
                begin += duration + b
                ax.set_ylim(-32767*1.2,32767*1.2)

def plot_sequence(fig,sequence):
    for i,s in enumerate(sequence):
        if i == 0:
            ax = ax1 = fig.add_subplot(len(sequence),1,i+1)
        else:
            ax = fig.add_subplot(len(sequence),1,i+1, sharex=ax1)
        if isinstance(s,Range) or isinstance(s,Blank):
            begin = int(s.begin)
            duration = int(s.duration)
            ax.add_patch(patches.Rectangle(xy=(begin,-1),width=duration,height=2))
            ax.set_ylim(-1.2,1.2)
        else:
            t = s.sampling_points
            ax.plot(t,np.real(s.iq))
            ax.plot(t,np.imag(s.iq))
            ax.set_ylim(-1.2,1.2)
