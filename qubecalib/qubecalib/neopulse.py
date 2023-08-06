from .qube import AWG, UNIT
import e7awgsw
from e7awgsw import WaveSequence, CaptureParam

from typing import Final
from contextlib import contextmanager
from collections import namedtuple, deque
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# The internal time and frequency units are [ns] and [GHz], respectively.

RunningConfig = namedtuple(
    'RunningConfig',
    [
        'contexts',
    ]
)
RunningConfig.__new__.__defaults__ = (deque(),)
__rc__: Final[RunningConfig] = RunningConfig()

mS: Final[float] = 1e+6
uS: Final[float] = 1e+3
nS: Final[float] = 1.

MHz: Final[float] = 1e-3
GHz: Final[float] = 1.

BLOCK: Final[float] = 128. # ns
BLOCKs: Final[float] = BLOCK

WORD: Final[float] = 8. # ns
WORDs: Final[float] = WORD


class Channel(object):

    def __init__(self,frequency,*args,**kw):

        super().__init__(*args,**kw)
        self.frequency = frequency
        
class Control(Channel): pass
class Readout(Channel): pass

class ContextNode(object):

    def __init__(self):
        c = __rc__.contexts
        if len(c):
            c[-1].append(self)

class Slot(ContextNode):

    SAMPLING_PERIOD: Final[float] = 2.

    def __init__(self,ch,begin=0*nS,width=64*nS):

        super().__init__()
        self.ch = ch
        self.begin = begin
        self.width = width
        self.reset()

    def reset(self):
        for x in [self.begin,self.width]:
            q ,r = divmod(x, self.SAMPLING_PERIOD)
            if r != 0:
                raise Exception('The specified time must be a multiple of {} ns.')
        self.width_in_samples = int(self.width / self.SAMPLING_PERIOD) + 1
        
    def set(self,**kw):

        for k,v in kw.items():
            setattr(self,k,v)
        self.reset()

class Arbit(Slot):

    def __init__(self,ch,begin=0*nS,width=64*nS,init=0+0j):

        self.init = init
        self.miq = None
        super().__init__(ch,begin,width)

    def reset(self):
        super().reset()
        self.iq = np.zeros(self.width_in_samples).astype(complex) # iq data
        self.iq[:] = self.init

    @property
    def sampling_points(self):

        return np.arange(0,len(self.iq)) * 2 + self.begin # sampling points [ns]
    
class Range(Slot):

    def __init__(self,ch,begin=0*nS,width=64*nS):

        super().__init__(ch,begin,width)

class Sequence(deque): pass
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
        end=None,
        width=None
    ):
        
        self.begin = begin
        self._end_width(end,width)
        
    def _end_width(self,end,width):
        if width is None:
            self.end = end
            self.width = end - self.begin
        elif end is None:
            self.end = self.begin + width
            self.width = width
        else:
            raise()
        
    def alloc(self,*args):

        c = __rc__.contexts
        if len(c):
            d = c[-1]
            for a in args:
                if a not in d:
                    d[a] = deque([self.__class__(self.begin,self.end)])
                else:
                    d[a].append(self.__class__(self.begin,self.end))


class Chunk(ContextNodeAlloc):

    def __init__(
        self,
        begin,
        end=None,
        width=None
    ):

        super().__init__(begin,end,width)
        width_in_samples = n = int(self.width // 2)
        self.iq = np.zeros(n).astype(complex)
    
    @property
    def sampling_points(self):
        
        return np.arange(len(self.iq)) * 2 + self.begin # sampling points [ns]

class SumSect(ContextNodeAlloc): pass

class Setup(list):

    def __init__(self,*args):

        super().__init__(args)

    def get(self,arg):

        keys = [k for k,v in self]
        return self[keys.index(arg)][1]

def organize_slots(sequence):
    r = {}
    for v in sequence:
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

def chunks2wseq(chunks, period):
    c = chunks
    if int(c[0].begin) % WORD:
        raise('The wait width must be an integer multiple of ()ns.'.format(int(WORD)))
    w = e7awgsw.WaveSequence(num_wait_words=int(c[0].begin) // int(WORD), num_repeats=1)
    boundary = b = [int(cc.begin) for cc in list(c)[1:]] + [period,]
    for s,p in zip(c,b):
        if (p - s.end) % WORD:
            raise('The blank width must be an integer multiple of ()ns.'.format(int(WORD)))
        kw = {
            'iq_samples': list(zip(np.real(0.5*32767*s.iq).astype(int), np.imag(0.5*32767*s.iq).astype(int))),
            'num_blank_words': int(p - s.end) // int(WORD),
            'num_repeats': 1,
        }
        w.add_chunk(**kw)
    return w


def sect2capt(section,period):
    s = section
    p = e7awgsw.CaptureParam()
    p.num_integ_sections = 1
    p.capture_delay = int(s[0].begin)
    boundary = b = [int(ss.begin) for ss in s][1:] + [period,]
    for ss,bb in zip(s,b):
        if int(ss.width) % int(WORD):
            raise('error!!')
        width = int(ss.width) // int(WORD)
        if int(bb - ss.end) % int(WORD):
            raise('error!!')
        blank =  int(bb - ss.end) // int(WORD)
        p.add_sum_section(num_words=width, num_post_blank_words=blank)
    return p



def convert(sequence, section, alloc_table, period):
    channel2slot = r = organize_slots(sequence) # channel -> slot
    a,c = split_awg_unit(alloc_table) # channel -> txport, rxport
    
    # Tx チャネルはデータ変調
    for k in a:
        for v in r[k]:
            if not isinstance(v,Range):
                m = a[k].modulation_frequency(mhz=k.frequency*1e+3)*MHz
                t = v.sampling_points
                v.miq = v.iq * np.exp(1j*2*np.pi*(m*t))
            # 位相合わせについては後日

    # 各AWG/UNITの各セクション毎に属する Slot のリストの対応表を作る
    section2slots = {}
    for k,v in section.items():
        for vv in v:
            if vv not in section2slots:
                section2slots[vv] = deque([])
            for c in alloc_table[k]:
                for s in r[c]:
                    if vv.begin <= s.begin and s.begin + s.width <= vv.end:
                        bawg = isinstance(k,AWG) and not isinstance(s,Range)
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
                rng = (v.begin <= t) * (t <= v.begin + v.width)
                s.iq[rng] += v.miq / len(ss)
    
    # 束ねるチャネルを WaveSequence に変換
    awgs = [k for k in alloc_table if isinstance(k,AWG)] # alloc_table から AWG 関連だけ抜き出す
    wseqs = [(k, chunks2wseq(section[k], period)) for k in awgs] # chunk obj を wseq へ変換する

    units = [k for k in alloc_table if isinstance(k,UNIT)] 
    capts = [(k,sect2capt(section[k],period)) for k in units]
    
    return Setup(*(wseqs + capts))
    
def plot_send_recv(fig, data):

    n = len([vv for k,v in data.items() for vv in v])
    i = 1
    for k,v in data.items():
        for vv in v:
            if fig.axes:
                ax = fig.add_subplot(n*100+10+i, sharex=ax1)
            else:
                ax = ax1 = fig.add_subplot(n*100+10+i)
            ax.plot(np.real(vv))
            ax.plot(np.imag(vv))
            i += 1

def plot_setup(fig,setup,capture_delay=0):
    for i,tpl in enumerate(setup):
        k,v = tpl
        if i == 0:
            ax = ax1 = fig.add_subplot(len(setup)*100+10+i+1)
        else:
            ax = fig.add_subplot(len(setup)*100+10+i+1, sharex=ax1)
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
                width = int(s[0]) * int(WORDs)
                ax.add_patch(patches.Rectangle(xy=(begin,-32767),width=width,height=2*32767))
                begin += width + b
                ax.set_ylim(-32767*1.2,32767*1.2)

def plot_sequence(fig,sequence):
    for i,q in enumerate(sequence):
        if i == 0:
            ax = ax1 = fig.add_subplot(len(sequence),1,i+1)
        else:
            ax = fig.add_subplot(len(sequence),1,i+1, sharex=ax1)
        for s in q:
            if isinstance(s,Range):
                begin = int(s.begin)
                width = int(s.width)
                ax.add_patch(patches.Rectangle(xy=(begin,-1),width=width,height=2))
                ax.set_ylim(-1.2,1.2)
            else:
                t = s.sampling_points
                ax.plot(t,np.real(s.iq))
                ax.plot(t,np.imag(s.iq))
                ax.set_ylim(-1.2,1.2)