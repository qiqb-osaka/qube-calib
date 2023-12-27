from .qube import CPT, AWG, UNIT
from .meas import WaveSequenceFactory, CaptureModule, CaptureParam
from .setupqube import _conv_to_e7awgsw, _conv_channel_for_e7awgsw
from .pulse import Read, Arbit
from .neopulse import Series, Blank, Range, Capture, ContextNode, Readout, body, SlotWithIQ
from .units import *

import e7awgsw
from e7awgsw import WaveSequence, CaptureParam, AwgCtrl, IqWave
from typing import Union
import os
import time
import socket
import struct
from collections import namedtuple, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout, contextmanager
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
import datetime
import numpy as np
from e7awgsw import AwgCtrl, CaptureCtrl, DspUnit, DspUnit
# from e7awgsw import AwgCtrl, CaptureCtrl, CaptureParam, CaptureModule, AWG
# from qubecalib.qube import CPT
# from qubecalib.meas import WaveSequenceFactory
# from qubecalib.setupqube import _conv_to_e7awgsw, _conv_channel_for_e7awgsw
from quel_clock_master import QuBEMasterClient, SequencerClient


PORT = 16384
IPADDR = '10.3.0.255'
REPEAT_WAIT_SEC = 0.1
REPEAT_WAIT_SEC = int(REPEAT_WAIT_SEC * 125000000) # 125Mcycles = 1sec
CANCEL_STOP_PACKET = struct.pack(8*'B', 0x2c, *(7*[0]))


def check_clock(*qubes, ipmaster='10.3.0.255'):
    ipmulti = [q.ipmulti for q in qubes]
    m = QuBEMasterClient(ipmaster,16384)
    s = [SequencerClient(ip, seqr_port=16384, synch_port=16385) for ip in ipmulti]
    c = [o.read_time() for o in s]
    c.append(m.read_clock())
    return c

def kick(*qubes, delay=1): # from QubeServer.py
    destinations = [q.ipmulti for q in qubes]
    DAQ_INITSDLY = delay
    cDAQ_SDLY_TAG = DAQ_INITSDLY
    SYNC_CLOCK = 125_000_000 # 125Mcycles = 1sec

    delay = int(cDAQ_SDLY_TAG * SYNC_CLOCK + 0.5)

    seq_cli = {a:SequencerClient(a, seqr_port=16384, synch_port=16385) for a in destinations}
    clock = seq_cli[destinations[0]].read_time() + delay

    for a in destinations:
        seq_cli[a].add_sequencer(16*(clock//16+1))

def search_qube(o):
    return o.port.qube if isinstance(o,AWG) else o.capt.port.qube if isinstance(o,UNIT) else None

def extract_qubes(*setup):
    return tuple(set([search_qube(o1) for o1, o2 in setup]))

def split_qube(*setup):
    rslt = {q:[] for q in extract_qubes(*setup)}
    for o1, o2 in setup:
        rslt[search_qube(o1)].append((o1, o2))
    return rslt
                                    
def split_send_recv(*setup):
    send = tuple((o1, o2) for o1, o2 in setup if isinstance(o1,AWG))
    recv = tuple((o1, o2) for o1, o2 in setup if isinstance(o1,UNIT))
    return send, recv

class ChannelMap(dict):

    def __enter__(self):

        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        
        pass

    def map(self, physical, *logical):

        if isinstance(physical, UNIT):
            if len(logical) != 1:
                raise ValueError('Only one logical channel can be mapped to a capture unit.')

        self[physical] = logical

    @property
    def physical(self):
        
        rslt = {}
        for k, v in self.items():
            for o in v:
                if o not in rslt:
                    rslt[o] = [k]
                else:
                    rslt[o].append(k)

        return rslt

    @property
    def logical(self):

        return {k: v for k, v in self.items()}


# WaveSequence と CaptureParam に直接アクセスできること
# 複数台の Qube を扱う場合にも煩雑にならないこと

class PulseSet(object):
    # 複数台の Qube へのパルス設定 (wave_sequence と capture_parameter) を管理する
    # 一括設定変更のサポート関数なども用意
    pass

class PulseSubSet(object):
    # Qube 単位でのパルス管理
    # どの awg へ，どの capture_unit へ．
    pass

class PulseConverter(object):
    
    Container = namedtuple('Container', ['awg_to_wavesequence', 'capt_to_captparam', 'capt_to_mergedchannel', 'adda_to_channels'])
    
    # repeats は後から設定できるようにするのが e7awgsw 的に良さそう
    # その後に各種 DSP を設定する
    @classmethod
    def conv(cls, channels, offset=0, interval=0):
        
        # qube.port に delay を後付け設定しているので忘れていないか確認する
        for k, v in channels.items():
            if isinstance(k, CPT):
                try:
                    k.port.delay
                except AttributeError as e:
                    raise AttributeError('delay attribute is required for receiver port')
        r = _conv_to_e7awgsw(adda_to_channels=channels, offset=offset, repeats=1, interval=interval, trigger_awg=None)
        # captparam を複製して DSP の設定をクリアする
        func = lambda v: {k2:(cls.duplicate_captparam(v2),) for k2, v2 in v['capt_to_captparam'].items()}
        qube_channels = lambda qube: {k:v for k, v in channels.items() if k.port.qube == qube}
        return dict([(k, cls.Container(v['awg_to_wavesequence'],func(v),v['capt_to_mergedchannel'],qube_channels(k))) for k, v in r.items()])
    
    @classmethod
    def duplicate_captparam(cls, cp, repeats=1, interval=0, delay=0):
        p = CaptureParam()
        p.num_integ_sections = cp.num_integ_sections
        p.capture_delay = cp.capture_delay
        for i in range(repeats):
            for s in cp.sum_section_list:
                p.add_sum_section(*s)
        return p
    
    @classmethod
    def temp(cls, adda_to_channels):
        
        channels = adda_to_channels
        
        print(list(set([k.port.qube for k, v in channels.items()])))
        
        # チャネルを周波数変換し，時間領域でチャネルを結合し，e7awgsw の制約に合わせてスロットを再配置する
        w2c = dict([(k, _conv_channel_for_e7awgsw(v, k, 0)) for k, v in channels.items()])
        # quantized_channel? merged_channel?
        
        print([w.port.delay for w, c in w2c.items() if isinstance(w, CPT)])
        
        return w2c
        # return _conv_to_e7awgsw(adda_to_channels)
        
        
class Recv(CaptureCtrl):
    
    def __init__(self, *unit_cprm_pair):
        
        # argparse = lambda module, *params: (module, params)
        # cond = lambda o: isinstance(o,tuple) or isinstance(o,list)
        # arg = tuple([argparse(o[0],*(o[1] if cond(o[1]) else (o[1],))) for o in module_params_pair])
        
        self.parms = arg = unit_cprm_pair

        # typing で書くのがいまどき？
        if not [isinstance(m,UNIT) for m, l in arg] == len(arg)*[True]:
            raise TypeError('1st element of each tuple should be qubecalib.qube.UNIT instance.')
        
        # if not [qube == m.port.qube for m, l in arg] == len(arg)*[True]:
        #    raise Exception('The qube that owns the CaptureModule candidates in the arguments must all be identical.')
            
        # if not [len(l) < 5 for m, l in arg] == len(arg)*[True]:
        #    raise Exception('Each CaptureParameter list in the argument must have no longer than 4 elements.')
                
        qubes = list(set([u.capt.port.qube for u, p in arg]))
        if len(qubes) > 1:
            raise Exception('All awg objects in the argument must belong to a common qube object.')
       
        super().__init__(qubes[0].ipfpga)
        
        #self._trigger = None # obsoleted
        #self.modules = [m for m, l in arg]
        #self.units = sum([self.assign_param_to_unit(m, l) for m, l in arg],[])
        
        self.capts = capts = list(set([u.capt.id for u, p in arg]))
        self.units = units = [u.id if isinstance(u,UNIT) else u for u, p in arg]

        self.initialize(*units)
        for u, p in arg:
            uu = u.id if isinstance(u,UNIT) else u
            self.set_capture_params(uu,p)

        
    def assign_param_to_unit(self, module, params): # obsoleted
        
        m = module
        units = [u for u in CaptureModule.get_units(m if isinstance(m, CaptureModule) else m.id)[:len(params)]]
        self.initialize(*units)
        for u, p in zip(units, params):
            self.set_capture_params(u, p)
            
        return units
    
    def start(self, timeout=30):
        
        u = self.units
        self.start_capture_units(*u)
        self.wait_for_capture_units_to_stop(timeout, *u)
        self.check_err(*u)
    
    def prepare_for_trigger(self, awg: Union[AWG,e7awgsw.AWG]):
        
        for c in self.capts:
            self.select_trigger_awg(c, awg.id if isinstance(awg,AWG) else awg)
        self.enable_start_trigger(*self.units)
        #trig = awg if isinstance(awg, e7awgsw.AWG) else awg.id
        #for m in self.modules:
        #    self.select_trigger_awg(m.id, trig)
        #self.enable_start_trigger(*self.units)
    
    def wait_for_capture(self, timeout=30):
        
        self.wait_for_capture_units_to_stop(timeout, *self.units)
        self.check_err(*self.units)
    
    def get(self):
        
        return {u: self._get(u, p) for u, p in self.parms}


    def _get(self, u, p):

        u = u.id if isinstance(u,UNIT) else u
        l = p.num_integ_sections
        m = len(p.sum_section_list)
        n = self.num_captured_samples(u)
        if DspUnit.CLASSIFICATION in p.dsp_units_enabled:
            d = np.array(list(self.get_classification_results(u, n)))
        else:
            c = np.array(self.get_capture_data(u, n))
            d = c[:,0] + 1j * c[:,1]
        if DspUnit.INTEGRATION in p.dsp_units_enabled:
            d = d.reshape(1,-1)
        else:
            d = d.reshape(l,-1)
        if DspUnit.SUM in p.dsp_units_enabled:
            d = np.hsplit(d, list(range(m)[1:]))
        else:
            d = np.hsplit(d, np.cumsum(np.array([w for w, b in p.sum_section_list[:-1]])) * p.NUM_SAMPLES_IN_ADC_WORD)
            d = [di.transpose() for di in d]
        
        return d

    
#     def check_err(self, *units):
#         
#         e = super().check_err(*units)
#         print(e)
#         if any(e):
#             raise IOError('CaptureCtrl error.')
    
    def wait(self, timeout=30): # obsoleted
        
        u = self.units
        self.enable_start_trigger(*u)
        self.wait_for_capture_units_to_stop(timeout, *u)
        self.check_err(*u)
    
    @property
    def trigger(self): # obsoleted
        
        return self._trigger
    
    @trigger.setter
    def trigger(self, awg): # obsoleted
        
        self._trigger = awg if isinstance(awg, e7awgsw.AWG) else awg.id
        for m in self.modules:
            self.select_trigger_awg(m.id, self._trigger)
            
            
# class RecvSingleMod(Recv): # for easy access interface
    
#     def __init__(self, module, params):
        
#         super().__init__(((module, params),))

class CaptMemory(e7awgsw.CaptureCtrl): # for data access
    
    def get_data(self, unit):
        
        n = self.num_captured_samples(unit)
        c = np.array(self.get_capture_data(unit,n))
        return c[:,0] + 1j * c[:,1]
        
class Send(AwgCtrl):
    
    def __init__(self, *awg_wseq_pair):
        
        arg = awg_wseq_pair
        
        # typing で書くのがいまどき？
        if not [(isinstance(a, AWG) or isinstance(a, e7awgsw.AWG)) and (isinstance(s, WaveSequenceFactory) or isinstance(s, WaveSequence)) for a, s in arg] == len(arg)*[True]:
            raise TypeError('Element type of each tuple should be (qubecalib.qube.AWG, qubecalib.meas.WaveSequenceFactory).')
        
        # if not [qube == a.port.qube for a, s in arg] == len(arg)*[True]:
        #    raise Exception('The qube that owns the AWG candidates in the arguments must all be identical.')
            
        qubes = list(set([a.port.qube for a, w in arg]))
        if len(qubes) > 1:
            raise Exception('All awg objects in the argument must belong to a common qube object.')
        
        super().__init__(qubes[0].ipfpga)
        
        lawgs = lambda x: x.id if isinstance(x, AWG) else x
        lseqs = lambda x: x.sequence if isinstance(x, WaveSequenceFactory) else x
        self.awgs = awgs = [lawgs(a) for a, w in arg]
        
        self.initialize(*awgs)
        for a, s in arg:
            self.set_wave_sequence(lawgs(a), lseqs(s))
            
    def start(self):
        
        # a = [a.id for a in self.awgs]
        # self.terminate_awgs(*a)
        # self.clear_awg_stop_flags(*a)
        self.start_awgs(*self.awgs)
    send=start
        
    def prepare_for_sequencer(self, timeout=30):
        
        # a = [a.id for a in self.awgs]
        # self.terminate_awgs(*a)
        # self.clear_awg_stop_flags(*a)
        print('wait:', datetime.datetime.now())
        # print('wait for started by sequencer for {}'.format(self.awgs[0].port.qube.ipfpga))
        self.wait_for_awgs_to_stop(timeout, *self.awgs)
        print('awg done:', datetime.datetime.now())
        print('end')
        
# class SendSingleAwg(Send):
    
#     def __init__(self, awg, sequence):
        
#         super().__init__(((awg, sequence),))

class Terminate(AwgCtrl):
    
    def __init__(self, awgs):
        
        super().__init__(awgs.port.qube.ipfpga)
        
        self.awgs = awgs
        
    def terminate(self):
        
        self.terminate_awgs(*self.awgs)

class TerminateSingleAwg(Terminate):
    
    def __init__(self, awg):
        
        super().__init__((awg,))

def multishot(adda_to_channels, triggers, repeats=1, timeout=30, interval=50000):
    
    qube_to_pulse = c = PulseConverter.conv(adda_to_channels, interval)

    if len(c.keys()) == 1:
        qube = tuple(c.keys())[0]
        trigger = [o for o in triggers if o.port.qube() == qube][0]
        units = multishot_single(qube, qube_to_pulse[qube], trigger, repeats, timeout, interval)
    else:
        units = multishot_multi(qube_to_pulse, triggers, repeats, timeout, interval)

    return units, qube_to_pulse

def duplicate_captparam(cp):
    p = CaptureParam()
    p.capture_delay = cp.capture_delay
    for s in cp.sum_section_list:
        p.add_sum_section(*s)
    return p

def multishot_multi(pulse, triggers, repeats=1, timeout=30, interval=50000):
    
    def set_repeats(w, v):
        w.num_repeats = v
        return w

    def enable_integration(p, v):
        n = duplicate_captparam(p)
        n.num_integ_sections = v
        n.sel_dsp_units_to_enable(e7awgsw.DspUnit.INTEGRATION)
        return n

    arg_send = lambda pulse: tuple([(a, set_repeats(w,repeats)) for a, w in pulse.awg_to_wavesequence.items()])
    arg_recv = lambda pulse: tuple([(c, tuple([enable_integration(i,repeats) for i in p])) for c, p in pulse.capt_to_captparam.items()])
    t = {awg.port.qube: awg for awg in triggers}
    
    with redirect_stdout(open(os.devnull, 'w')):
    
        with ThreadPoolExecutor() as e:
            
            capts = [e.submit(lambda: wait_for_awg(qube=k, capt_cparam_pair=arg_recv(v), trigger=t[k], timeout=timeout)) for k, v in pulse.items() if v.capt_to_captparam]
            awgs = [e.submit(lambda: wait_for_sequencer(qube=k, awg_wseq_pair=arg_send(v), timeout=timeout)) for k, v in pulse.items() if v.awg_to_wavesequence]
            time.sleep(0.1)
            
            client = QuBEMasterClient(IPADDR, PORT)
            r, a = client.clear_clock(value=0)
            r, a = client.kick_clock_synch([k.ipmulti for k in pulse.keys()])
            mark = client.read_clock(value=0) + REPEAT_WAIT_SEC
            for qube in pulse.keys():
                a = qube.ipmulti
                s = SequencerClient(a, PORT)
                try:
                    r, a = s.add_sequencer(16*(mark//16+1)+qube.skew)
                except NameError as e:
                    raise NameError('qube.skew is required')
                
            for a in awgs:
                a.result()
            units = [c.result() for c in capts]
        
        
        for qube in pulse.keys():
            SequencerClient(qube.ipmulti, PORT).send_recv(CANCEL_STOP_PACKET)
    
    for qube, v in pulse.items():
        for captm, w in v.capt_to_captparam.items():
            channels = v.adda_to_channels[captm]
            multishot_get_data(captm, channels)
    
    return units

def multishot_single(qube, pulse, trigger, repeats=1, timeout=30, interval=50000):
    
    def set_repeats(w, v):
        w.num_repeats = v
        return w
    
    def enable_integration(p, v):
        n = duplicate_captparam(p)
        n.num_integ_sections = v
        n.sel_dsp_units_to_enable(e7awgsw.DspUnit.INTEGRATION)
        return n
    
    arg_send = tuple([(a, set_repeats(w,repeats)) for a, w in pulse.awg_to_wavesequence.items()])
    arg_recv = tuple([(c, tuple([enable_integration(i,repeats) for i in p])) for c, p in pulse.capt_to_captparam.items()])
    
    with ThreadPoolExecutor() as e:
         thread = e.submit(lambda: wait_for_awg(qube, arg_recv, trigger, timeout))
         time.sleep(0.1)
         with Send(qube, *arg_send) as s:
             s.start()
         units = thread.result()
   
    capt_to_mergedchannel = pulse.capt_to_mergedchannel
    retrieve_data_into_mergedchannel(capt_to_mergedchannel, units)

    capt_to_channels = {cpt: ch for cpt, ch in pulse.adda_to_channels.items() if isinstance(cpt, CPT)}

    # 各 Readin チャネルの読み出しスロットに復調したデータを格納する
    for cpt, chs in capt_to_channels.items():
        for ch in chs:
            is_Read_in_Channel = np.sum([True if isinstance(o, Read) else False for o in ch]) == True
            if not is_Read_in_Channel:
                continue
            mch = capt_to_mergedchannel[cpt]

            for slot in ch:
                if not isinstance(slot, Read):
                    continue
                # スロットが含まれる多重化スロットを見つける
                for mslot in mch:
                    mt = mch.get_timestamp(mslot)
                    st = ch.get_timestamp(slot)
                    if mt[0] <= st[0] and st[-1] <= mt[-1]:
                        break
                slot.iq = np.zeros(len(st)).astype(complex)
                slot.iq[:] = mslot.iq[(st[0] <= mt) & (mt <= st[-1])]
                slot.iq *= np.exp(-1j * 2 * np.pi * cpt.modulation_frequency(ch.center_frequency*1e-6)*1e-3 * st)


#    for captm, p in arg_recv:
#        channels = pulse.adda_to_channels[captm]
#        multishot_get_data(captm, channels)
    
    return units

def retrieve_data_into_mergedchannel(capt_to_mergedchannel, units, offset=0):

    cpt2ch = capt_to_mergedchannel
    qubes = list(set([k.port.qube for k, v in cpt2ch.items()]))
    if not len(qubes) == 1:
        raise ValueError('There must be single qube in capt_to_mergedchannel.')
    qube = qubes[0]

    # 各 Channel には Read スロットが単一であると仮定
    # 各 CaptureModule に割り当てられた合成チャネルの時間軸とデータを生成する
    for cpt, ch in cpt2ch.items():
        # 合成チャネルの読み出しスロットにデータを埋め込む
        for slt in ch:
            if not isinstance(slt, Arbit):
                continue
            t_ns = ch.get_timestamp(slt)
            slt.iq = np.zeros(len(t_ns)).astype(complex)
            with CaptMemory(qube.ipfpga) as m:
                for unit in units:
                    slt.iq[:] = m.get_data(unit)

def multishot_get_data(captm, channels):

    units = CaptureModule.get_units(captm.id)
    unit_to_channel = {units[i]: v for i, v in enumerate(channels)}

    with CaptMemory(captm.port.qube.ipfpga) as m:
        for unit, channel in unit_to_channel.items():
            slot = channel.findall(Read)[-1]
            v = m.get_data(unit)
            t = np.arange(0, len(v)) / CaptureCtrl.SAMPLING_RATE
            v *= np.exp(-1j * 2 * np.pi * captm.modulation_frequency(channel.center_frequency*1e-6)*1e+6 * t)
            d = slot.duration
            slot.iq = v


def _singleshot(adda_to_channels, triggers, repeats, timeout=30, interval=50000):
    
    c = PulseConverter.conv(adda_to_channels, interval)
    
    if len(c.keys()) == 1:
        qube = tuple(c.keys())[0]
        trigger = [o for o in triggers if o.port.qube == qube][-1]
        units = singleshot_singleqube(qube, c[qube], trigger, repeats, timeout)
    else:
        units = singleshot_multiqube(c, triggers, repeats, timeout)
        
    return units

singleshot = _singleshot

def singleshot_singleqube(qube, pulse, trigger, repeats, timeout=30, interval=50000):
    
    def set_repeats(w, v):
        w.num_repeats = v
        return w
    enable_repeats = lambda p, v: PulseConverter.duplicate_captparam(p, repeats=v)
    arg_send = tuple([(a, set_repeats(w,repeats)) for a, w in pulse.awg_to_wavesequence.items()])
    arg_recv = tuple([(c, tuple([enable_repeats(i,repeats) for i in p])) for c, p in pulse.capt_to_captparam.items()])
    
    #for a, w in arg_send:
    #    w.num_repeats = repeats
    #arg_recv = tuple([(c, tuple([PulseConverter.duplicate_captparam(pi, repeats=repeats) for pi in p])) for c, p in arg_recv])
    
    #with Send(qube, *arg_send) as s, Recv(qube, *arg_recv) as r:
    #    r.wait_for_trigger(trigger)
    #    s.start()
    #    r.wait_for_capture(timeout=timeout)
    #    units = r.units
    
    with ThreadPoolExecutor() as e:
         thread = e.submit(lambda: wait_for_awg(qube, arg_recv, trigger, timeout))
         time.sleep(0.1)
         with Send(qube, *arg_send) as s:
             s.start()
         units = thread.result()
    
    captm = arg_recv[0][0]
    for channel in pulse.adda_to_channels[captm]:
        singleshot_get_data(captm, channel, repeats)
    
    return units

def wait_for_sequencer(qube, awg_wseq_pair, timeout=30):

    with Send(qube, *awg_wseq_pair) as s:
        s.wait_for_sequencer(timeout=timeout)

def wait_for_awg(qube, capt_cparam_pair, trigger, timeout=30):
    
    with Recv(qube, *capt_cparam_pair) as r:
        r.wait_for_trigger(trigger)
        r.wait_for_capture(timeout=timeout)
        u = r.units
        
    return u
    

def singleshot_multiqube(pulse, triggers, repeats, timeout=30, interval=50000):
    
    def set_repeats(w, v):
        w.num_repeats = v
        return w

    enable_repeats = lambda p, v: PulseConverter.duplicate_captparam(p, repeats=v)
    arg_send = lambda pulse: tuple([(a, set_repeats(w,repeats)) for a, w in pulse.awg_to_wavesequence.items()])
    arg_recv = lambda pulse: tuple([(c, tuple([enable_repeats(i,repeats) for i in p])) for c, p in pulse.capt_to_captparam.items()])
    t = {awg.port.qube: awg for awg in triggers}
    
    with redirect_stdout(open(os.devnull, 'w')):
    
        with ThreadPoolExecutor() as e:
            
            capts = [e.submit(lambda: wait_for_awg(qube=k, capt_cparam_pair=arg_recv(v), trigger=t[k], timeout=timeout)) for k, v in pulse.items() if v.capt_to_captparam]
            awgs = [e.submit(lambda: wait_for_sequencer(qube=k, awg_wseq_pair=arg_send(v), timeout=timeout)) for k, v in pulse.items() if v.awg_to_wavesequence]
            time.sleep(0.1)
            
            client = QuBEMasterClient(IPADDR, PORT)
            r, a = client.clear_clock(value=0)
            r, a = client.kick_clock_synch([k.ipmulti for k in pulse.keys()])
            mark = client.read_clock(value=0) + REPEAT_WAIT_SEC
            for qube in pulse.keys():
                a = qube.ipmulti
                s = SequencerClient(a, PORT)
                try:
                    r, a = s.add_sequencer(16*(mark//16+1)+qube.skew)
                except NameError as e:
                    raise NameError('qube.skew is required')
                
            for a in awgs:
                a.result()
            units = [c.result() for c in capts]
        
        
        for qube in pulse.keys():
            SequencerClient(qube.ipmulti, PORT).send_recv(CANCEL_STOP_PACKET)
    
    for qube, v in pulse.items():
        for captm, w in v.capt_to_captparam.items():
            for channel in v.adda_to_channels[captm]:
                singleshot_get_data(captm, channel, repeats)
    
    return units
    
def singleshot_get_data(captm, channel, repeats):
    
    unit = CaptureModule.get_units(captm.id)[0] # <- とりあえず OK
    slot = channel.findall(Read)[0]
    with CaptMemory(captm.port.qube.ipfpga) as m:
        v = m.get_data(unit)
        v = v.reshape(repeats,int(len(v)/repeats))
        t = np.arange(0,len(v[0])) / CaptureCtrl.SAMPLING_RATE
        v *= np.exp(-1j * 2 * np.pi * captm.modulation_frequency(channel.center_frequency*1e-6)*1e+6 * t)
        d = slot.duration
        m = max([s.sampling_rate for s in channel])
        slot.iq = v
    

def standalone_recv(*setup,timeout=30):
    with Recv(*setup) as c:
        c.start(timeout)
        d = c.get()
    return d

def standalone_send(*setup):
    with Send(*setup) as c:
        c.send()

def standalone_send_recv(*setup,trig=None,timeout=30):
    if len(extract_qubes(*setup)) > 1:
        raise Exception('The standalone_send_recve() only accepts awg/capt belonging to a single qube in setup.')
    send, recv = split_send_recv(*setup)
    if not recv:
        raise Exception('Invalid setup.')
    if trig is None:
        trig, _ = send[0]
    with Send(*send) as ac, Recv(*recv) as cc:
        cc.prepare_for_trigger(trig)
        ac.send()
        cc.wait_for_capture(timeout)
        d = cc.get()
    return d

def sync_send(*setup):
    with Send(*setup) as c:
        c.prepare_for_sequencer(10)
    return None

def sync_send_recv(*setup,trig=None,timeout=30):
    if len(extract_qubes(*setup)) > 1:
        raise Exception('The send_recv_single() only accepts awg/capt belonging to a single qube in setup.')
    send, recv = split_send_recv(*setup)
    with Send(*send) as ac, Recv(*recv) as cc:
        cc.prepare_for_trigger(trig)
        ac.prepare_for_sequencer(timeout)
        cc.wait_for_capture(timeout)
        d = cc.get()
    return d

def send_recv(*setup,trigs={},delay=1,timeout=30):
    setup_qube = split_qube(*setup)
    if len(setup_qube.keys()) == 1: # standalone mode
        send, recv = split_send_recv(*setup)
        if send and recv:
            q = list(setup_qube.keys())[0]
            if q in trigs:
                trig = trigs[q]
            else:
                trig, _ = send[0]
            rslt = standalone_send_recv(*setup,trig=trig,timeout=timeout)
        elif send and not recv:
            rslt = standalone_send(*setup)
        elif not send and recv:
            rslt = standalone_recv(*setup,timeout=timeout)
        else:
            raise Exception('Invalid setup.')
        return rslt
    elif len(setup_qube.keys()) == 0:
        raise Exception('Invalid setup.')

    with ThreadPoolExecutor() as e:
        threads = []
        for q, s in setup_qube.items():
            send, recv = split_send_recv(*s)
            if not recv:
                threads.append(e.submit(lambda: sync_send(*send)))
            else:
                if q in trigs:
                    trig = trigs[q]
                else:
                    trig, _ = send[0]
                threads.append(e.submit(lambda: sync_send_recv(*s,trig=trig,timeout=timeout)))

        kick(*tuple(setup_qube.keys()),delay=delay)
        
        dct = {}
        for d in [o for o in [t.result() for t in threads] if o is not None]:
            dct = dct | d

    return dct
    

def quantize_sequence_duration(sequence_duration, constrain=10_240*nS):

    return sequence_duration // constrain * constrain


class WaveChunkFactory(object):
    
    def get_timestamp(self):
        duration = self.num_wave_words * WORDs
        samples = int(duration * 1e-9 * e7awgsw.AwgCtrl.SAMPLING_RATE)
        return np.linspace(0, duration, samples)
    
    def __init__(self, num_wave_words=16, num_blank_words=0, num_repeats=1, init=0, amp=32767):
        # duration の初期値は CW 出力を想定して設定した
        # int(duration * AwgCtrl.SAMPLING_RATE) が 64 の倍数だと切れ目のない波形が出力される．
        # 波形チャンクの最小サイズが 128ns (500Msps の繰り返し周期は 2ns)
        
        self._num_wave_words = num_wave_words
        self.num_blank_words = num_blank_words # [s]
        self.num_repeats = num_repeats # times
        self.init = init # iq value
        self.amp = amp
        self.iq = np.zeros(*self.timestamp.shape).astype(complex)
        self.iq[:] = init
        
    @property
    def timestamp(self):
        return self.get_timestamp()
        
    @property
    def chunk(self):
        
        iq = self.amp * self.iq
        i, q = np.real(iq).astype(int), np.imag(iq).astype(int)
        s = e7awgsw.IqWave.convert_to_iq_format(i, q, e7awgsw.WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK)
        
        r = e7awgsw.AwgCtrl.SAMPLING_RATE
        blank = self.num_blank_words * WORDs
        b = int(r * blank * 1e-9)
        n = e7awgsw.WaveSequence.NUM_SAMPLES_IN_AWG_WORD
        
        return {'iq_samples': s, 'num_blank_words': b // n, 'num_repeats': self.num_repeats}
    
    @property
    def num_wave_words(self):
        
        return self._num_wave_words
    
    @num_wave_words.setter
    def num_wave_words(self, v):
        
        self._num_wave_words = v
        self.iq = np.zeros(*self.timestamp.shape).astype(complex)
        self.iq[:] = self.init


class WaveSequenceFactory(object):
    
    def __init__(self, num_wait_words=0, num_repeats=1):
        
        self.num_wait_words = num_wait_words
        self.num_repeats = num_repeats
        self.chunk = []
        
    def new_chunk(self, wave_chunk_factory):
        
        self.chunk.append(wave_chunk_factory)
        
    def acquire(self):
        
        w = e7awgsw.WaveSequence(self.num_wait_words, self.num_repeats)
        for c in self.chunk:
            w.add_chunk(**c.chunk)
        return w

# from QubeServer.py by Tabuchi
# DSPのバンドパスフィルターを構成するFIRの係数を生成.
def acquisition_fir_coefficient(bb_frequency):
    ADCBB_SAMPLE_R = 500
    ACQ_MAX_FCOEF = 16 # The maximum number of the FIR filter taps prior to decimation process.
    ACQ_FCBIT_POW_HALF = 2**15 # equivalent to 2^(ACQ_FCOEF_BITS-1).
    
    sigma = 100.0 # nanoseconds
    freq_in_mhz = bb_frequency # MHz
    n_of_band = 16 # The maximum number of the FIR filter taps prior to decimation process.
    band_step = 500 / n_of_band
    band_idx = ( int( freq_in_mhz/band_step+0.5+n_of_band)-n_of_band )
    band_center = band_step * band_idx
    x = np.arange(ACQ_MAX_FCOEF) - (ACQ_MAX_FCOEF-1)/2
    gaussian = np.exp(-0.5*x**2/(sigma**2))
    phase_factor = 2*np.pi*(band_center/ADCBB_SAMPLE_R)*np.arange(ACQ_MAX_FCOEF)
    coeffs = gaussian*np.exp(1j*phase_factor)*(1-1e-3)
    return list((np.real(coeffs) * ACQ_FCBIT_POW_HALF).astype(int) + 1j * (np.imag(coeffs) * ACQ_FCBIT_POW_HALF).astype(int))


# CaptureParam.capture_delay に num_capture_delay_word を追加する
# パルスシーケンスに必要な delay に追加する際に用いる
def captparam_add_capture_delay(captparam, num_capture_delay_word):

    captparam.capture_delay += num_capture_delay_word


def captparam_enable_dspunit(captparam, dspunit):

    dspunits = captparam.dsp_units_enabled
    dspunits.append(dspunit)
    captparam.sel_dsp_units_to_enable(*dspunits)


def captparam_enable_integration(captparam):

    captparam_enable_dspunit(captparam, e7awgsw.DspUnit.INTEGRATION) # DSPの積算測定モジュールを有効化. 積算回数はrepeatsで設定.


def captparam_enable_sum(captparam):

    captparam_enable_dspunit(captparam, e7awgsw.DspUnit.SUM)


def captparam_enable_classification(captparam):

    captparam_enable_dspunit(captparam, e7awgsw.DspUnit.CLASSIFICATION)


def captparam_enable_demodulation(captparam, physical_channel, logical_channel):

    p, u, o = captparam, physical_channel, logical_channel
    # DSP で周波数変換する複素窓関数を設定
    t = 4*np.arange(p.NUM_COMPLEXW_WINDOW_COEFS)*2*nS
    m = u.capt.modulation_frequency(mhz=o.frequency/MHz)*MHz
    p.complex_window_coefs = list(np.round((2**31-1) * np.exp(-1j*2*np.pi*(m*t))))
    p.complex_fir_coefs = acquisition_fir_coefficient(-m/MHz) # BPFの係数を設定

    dspunits = p.dsp_units_enabled
    # DSPのどのモジュールを有効化するかを指定
    dspunits.append(e7awgsw.DspUnit.COMPLEX_FIR) # DSPのBPFを有効化
    dspunits.append(e7awgsw.DspUnit.DECIMATION) # DSPの間引1/4を有効化
    # dspunits.append(e7awgsw.DspUnit.REAL_FIR)
    dspunits.append(e7awgsw.DspUnit.COMPLEX_WINDOW) # 複素窓関数を有効化
    p.sel_dsp_units_to_enable(*dspunits)

class Setup(list):

    class CaptParamsIter( object ):

        def __init__(self, setup, channel_map, *logical_channels):

            self.setup = setup
            self.channel_map = channel_map
            if logical_channels:
                self.logical_channels = logical_channels
            else:
                self.logical_channels = tuple([channel_map.logical[u][0] for u, c in setup if isinstance(u, UNIT) and u in channel_map.logical])
            self.current_idx = 0
        
        def __iter__(self):

            return self

        def __next__(self):

            # while True:

            #     try:
            #         logical_channel = l = self.logical_channels[self.current_idx]
            #     except IndexError:
            #         raise StopIteration()

            #     self.current_idx += 1

            #     try:
            #         self.channel_map.physical[l]
            #     except KeyError:
            #         pass
            #     else:
            #         break
                    
            try:
                logical_channel = l = self.logical_channels[self.current_idx]
            except IndexError:
                raise StopIteration()

            u = [o for o in self.channel_map.physical[l] if isinstance(o, UNIT)][0]

            self.current_idx += 1

            return self.setup.get(u), u, l

    def __init__(self,*args):

        super().__init__(args)

    def get(self,arg):

        keys = [k for k,v in self]
        return self[keys.index(arg)][1]

    def captparams(self, channel_map, *logical_channels):

        return Setup.CaptParamsIter(self, channel_map, *logical_channels)

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


def acquire_section(sequence, channel_map, section=None):

    if section is None:
        section = Sections()

    tx = acquire_tx_section(sequence, channel_map)
    rx = acquire_rx_section(sequence.flatten(), channel_map)

    for k, v in tx.items():
        section[k] = v

    for k, v in rx.items():
        section[k] = v

    section.place()

    return section

def acquire_tx_section(sequence, channel_map, section=None):

    if section is None:
        section = Sections()

    logch = [k for k in sequence.flatten().slots if k is not None]
    phych = [k for k in channel_map for l in logch if l in channel_map[k] and isinstance(k,AWG)]

    is_multi_chunk = np.array([isinstance(o, Series) for o in sequence]).prod()
    is_multi_chunk *= np.array([isinstance(v, (Series, Blank)) for o in sequence if isinstance(o, Series) for v in o]).prod()

    if is_multi_chunk:

        for phychi in phych:

            section[phychi] = deque()

            wait = 0
            series = [v for o in sequence if isinstance(o, Series) for v in o]
            for o in [v for o in sequence if isinstance(o, (Series, Blank)) for v in o]:

                if isinstance(o, Blank):

                    wait += o.duration
                    
                elif isinstance(o, Series):
                    
                    d = (o.end - o.begin) / o.repeats

                    if o is not series[-1]:

                        blank = d % BLOCK
                        if blank:
                            u = [v for v in o.bodies.flatten() if not isinstance(v, (Range, Blank))]
                            b = min([v.begin for v in u])
                            e = max([v.end for v in u])
                            if o.end / o.repeats - e < blank:
                                raise ValueError('duration is wrong')
                        d = d // BLOCK * BLOCK
                        section[phychi].append(TxSection(wait=wait, duration=d, blank=blank, repeats=o.repeats))

                    else:
                        
                        d = d // BLOCK * BLOCK + BLOCK
                        section[phychi].append(TxSection(wait=wait, duration=d, blank=0, repeats=o.repeats))
                    
                    wait = 0

    else:
        
        # print('Single Chunk Mode')
        for phychi in phych:

            section[phychi] = deque()

            u = [v for v in sequence.flatten() if not isinstance(v, (Range, Blank))]
            b = min([v.begin for v in u]) // WORD * WORD
            e = max([v.end for v in u])
            d = ((e - b) // BLOCK + 1) * BLOCK
            section[phychi].append(TxSection(wait=b, duration=d, blank=0, repeats=1))
    
    return section


    # # log2phy = {o: k for k,v in channel_map.items() if isinstance(k,AWG) for o in v}
    # log2phy = {k: [o for o in v if isinstance(o,AWG)][0] for k, v in channel_map.logical.items()}
    # logch = [k for k in sequence.flatten().slots if k is not None]
    # phych = [k for k in channel_map for l in logch if l in channel_map[k] and isinstance(k,AWG)]

    # # AWG と Series の対応辞書を作る
    # series = find_sequence(sequence, Series)
    # awg2series = d = dict()
    # for p in phych:
    #     d[p] = deque()
    #     for s in series:
    #         if p in [log2phy[k] for k in s.flatten().slots.keys() if k is not None]:
    #             d[p].append(s)

    # # 重複した Series を削除する
    # for k, v in awg2series.items():
    #     item = lambda x: (x.begin, (x.end - x.begin) / x.repeats, x)
    #     rng = r = sorted([item(o) for o in v], key=lambda x:x[1], reverse=True)
    #     buffer = b = Sequence()
    #     isout = lambda x, y: (y[0] + y[1] < x[0]) + (x[0] + x[1] < y[0])
    #     isin = lambda x, y: (x[0] < y[0]) * (y[0] + y[1] < x[0] + x[1])
    #     for i, o in enumerate(rng[:-1]):
    #         skip = False
    #         for m in rng[i+1:]:
    #             if skip:
    #                 continue
    #             if isin(o, m):
    #                 skip = True
    #                 continue
    #             b.append(o[2])
    #     awg2series[k] = buffer
        
    # # AWG と Slots の対応辞書を作る
    # awg2slots = dict()
    # slots = sequence.flatten().slots
    # for phychi in phych:
    #     for k, v in slots.items():
    #         if phychi is log2phy[k] if k is not None else None:
    #             awg2slots[phychi] = deque()
    #             for o in v:
    #                 if isinstance(body(o),SlotWithIQ):
    #                     if o not in awg2series[phychi].flatten():
    #                         awg2slots[phychi].append(o)

    # 最も浅い repeats 付きの Series を参考にして Chunk をつくる
    

    # if section is None:
    #     section = Sections()

    # for phychi in phych:

    #     print(phychi, awg2series[phychi])
    #     slots = np.array([(s.begin,(s.end-s.begin)/s.repeats,s.end,s.repeats) for s in awg2series[phychi] if isinstance(s,Series)])
    #     print(phychi, slots)
    #     if not slots:
    #         if not awg2slots[phychi]:
    #             continue
    #         begin = min([o.begin for o in awg2slots[phychi]])
    #         end = max([o.end for o in awg2slots[phychi]])
    #         if begin % int(WORD):
    #             print(phychi, begin)
    #             raise ValueError('begin is not aligned')
    #         if (end - begin) % int(BLOCK):
    #             raise ValueError('durations is not aligned')
    #         section[phychi] = deque()
    #         section[phychi].append(TxSection(wait=begin,duration=end-begin,blank=0,repeats=1))
    #         continue
    #     slots = slots[np.argsort(slots[:,0])]

    #     section[phychi] = deque()

    #     begin, duration, end, repeats = slots[0]
    #     section[phychi].append(TxSection(wait=begin,duration=duration,blank=0,repeats=int(repeats)))
    #     for i in range(1,slots.shape[0]):
    #         begin, duration, end, repeats = slots[i,0], slots[i,1], slots[i-1,2], slots[i,3]
    #         section[phychi].append(TxSection(wait=begin-end,duration=duration,blank=0,repeats=int(repeats)))

    # return section


def acquire_rx_section(sequence, channel_map, section=None):

    if section is None:
        section = Sections()

    logch = [k for k in sequence.slots if isinstance(k,Readout)]
    phych = [k for k in channel_map for l in logch if l in channel_map[k] and isinstance(k,UNIT)]

    for phychi in phych:

        slots = np.array([(s.begin,s.duration,s.end) for s in sequence if isinstance(body(s),Range) and s.ch in channel_map[phychi]])
        slots = slots[np.argsort(slots[:,0])]
        
        # duration % WORD == 0, blank % WORD == 0
        # begin が条件を満たすように Range を前に offset する
        # duration が条件を満たすように Range を stretch する
        begin, duration, end = slots[0]
        offset = begin - (begin // (2*WORD) * (2*WORD))
        begin = begin - offset
        duration = offset + duration
        stretch = duration - (duration // WORD * WORD)
        duration = duration + stretch
        end = end + stretch
        slots[0,:] = [begin, duration, end]
        for i, slot in enumerate(slots[1:]):
            begin, duration, end = slot
            offset = begin - (begin // WORD * WORD)
            begin = begin - offset
            duration = offset + duration
            stretch = duration - (duration // WORD * WORD)
            duration = duration + stretch
            end = end + stretch
            slots[i+1,:] = [begin, duration, end]
            
        for i, _ in enumerate(slots[:-1]):
            b, d, e = slots[i+1][0], slots[i+1][1], slots[i][2]
            if b - e < 0:
                raise ValueError('Sumsections are overlapped.')
        section[phychi] = deque()

        begin, duration, end = slots[0]
        section[phychi].append(RxSection(delay=begin,duration=duration,blank=0))
        for i in range(1,slots.shape[0]):
            begin, duration, end = slots[i,0], slots[i,1], slots[i-1,2]
            section[phychi].append(RxSection(delay=begin-end,duration=duration,blank=0))
            
    return section

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


def plot_send_recv(fig, data, mag=False):

    n = len([vv for k,v in data.items() for vv in v])
    i = 1
    rslt = []
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
            rslt.append(k)
    return rslt

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


def split_awg_unit(channel_map):
    a,c = {},{}
    for k,v in channel_map.items():
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

    # print(c)

    if len(c) > 1:
        # ここは nS 単位で． __acquire_chunk__ 内で // int(WORD) してる
        for j, d in enumerate(list(c)[:-1]):
            if d.repeats > 1:
                w.add_chunk(**__acquire_chunk__(d, d.repeats-1, d.post + d.prior))
                # print((d.post + d.prior), (d.post + d.prior) // int(WORD))
            w.add_chunk(**__acquire_chunk__(d, 1, d.post + c[j+1].prior))
            # print((d.post + c[j+1].prior), (d.post + c[j+1].prior) // int(WORD))
        d = c[-1]
        if d.repeats > 1:
            w.add_chunk(**__acquire_chunk__(d, d.repeats-1, d.post + d.prior))
            # print((d.post + d.prior), (d.post + d.prior) // int(WORD))
        w.add_chunk(**__acquire_chunk__(d, 1, period - d.end + c[0].prior))
        # print((period - d.end), (period - d.end) // int(WORD))
    else:
        d = c[0]
        # print('LEN=1', d.prior, d.duration, d.end)
        w.add_chunk(**__acquire_chunk__(d, 1, period - d.end + c[0].prior))
    
    return w


def sect2capt(section,period,repeats):
    s = section
    p = e7awgsw.CaptureParam()
    p.num_integ_sections = repeats

    if int(s[0].prior) % int(2*WORD):
        raise('The capture_delay must be an integer multiple of ()ns.'.format(int(2*WORD)))
    p.capture_delay = int(s[0].prior) // int(WORD)

    for j in range(len(s)-1):
        # for i in range(s[j].repeats-1):
        #     duration = int(s[j].duration) // int(WORD)
        #     blank = int(s[j].post + s[j].prior) // int(WORD)
        #     p.add_sum_section(num_words=duration, num_post_blank_words=blank)
        duration = int(s[j].duration) // int(WORD)
        blank = int(s[j].post + s[j+1].prior) // int(WORD)
        p.add_sum_section(num_words=duration, num_post_blank_words=blank)
    # for i in range(s[-1].repeats-1):
    #     duration = int(s[-1].duration) // int(WORD)
    #     blank = int(s[-1].post + s[-1].prior) // int(WORD)
    #     p.add_sum_section(num_words=duration, num_post_blank_words=blank)
    duration = int(s[-1].duration) // int(WORD)
    blank = int(period - s[-1].end + s[0].prior) // int(WORD)
    p.add_sum_section(num_words=duration, num_post_blank_words=blank)

    return p


def convert(sequence, section, channel_map, period, repeats=1, warn=False):

    # channel2slot = r = organize_slots(sequence) # channel -> slot
    channel2slot = r = sequence.slots
    a, c = split_awg_unit(channel_map) # channel -> txport, rxport
    
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
    for k, v in section.items(): # k:AWG,v:List[TxSection] or k:UNIT,v:List[RxSection]
        for vv in v: # vv:Union[TxSection,RxSection]
            if vv not in section2slots:
                section2slots[vv] = deque([])
            if k not in channel_map: # k:Union[AWG,UNIT] 
                with warnings.catch_warnings():
                    if not warn:
                        warnings.simplefilter('ignore')
                    warnings.warn('AWG|UNIT {} is Iqnored.'.format(k))
                continue
            for c in channel_map[k]: # c:Channel
                if c not in r: # sequence の定義内で使っていない論理チャネルがあり得る
                    with warnings.catch_warnings():
                        if not warn:
                            warnings.simplefilter('ignore')
                        warnings.warn('Channel {} is Iqnored.'.format(c))
                    continue
                # for s in r[c]:
                #     for i in range(vv.repeats):
                #         print(vv, s, vv.begin + i * vv.total, s.begin, s.begin + s.duration, vv.end + i * vv.total)
                #         if vv.begin + i * vv.total <= s.begin and s.begin + s.duration <= vv.end + i * vv.total:
                #             bawg = isinstance(k,AWG) and isinstance(s,SlotWithIQ)
                #             bunit = isinstance(k,UNIT) and isinstance(body(s),Range)
                #             if bawg:
                #                 section2slots[vv].append(s)
                #             if bunit:
                #                 section2slots[vv].append(s)
                for s in r[c]:
                    if vv.repeats == 1:
                        bawg = isinstance(k,AWG) and isinstance(body(s),SlotWithIQ)
                    else:
                        bawg = isinstance(k,AWG) and isinstance(s,SlotWithIQ)
                    bunit = isinstance(k,UNIT) and isinstance(body(s),Range)
                    if bawg:
                        # print(vv, s, vv.begin, s.begin, s.begin + s.duration, vv.end)
                        if vv.begin <= s.begin and s.begin <= vv.end:
                            section2slots[vv].append(s)
                    elif bunit:
                        for i in range(vv.repeats):
                            # print(vv, s, vv.begin + i * vv.total, s.begin, s.begin + s.duration, vv.end + i * vv.total)
                            if vv.begin + i * vv.total <= s.begin and s.begin + s.duration <= vv.end + i * vv.total:
                                section2slots[vv].append(s)



    # for k,v in section2slots.items():
    #     print(k,v)

    # 各セクション毎に Chunk を合成する
    awgs = [k for k in channel_map if isinstance(k,AWG)]
    for i, k in enumerate(awgs):
        if k in section:
            for s in section[k]:
                t = s.sampling_points
                s.iq[:] = 0
                ss = section2slots[s]
                for v in ss:
                    rng = (v.begin <= t) * (t < v.end)
                    s.iq[rng] += v.miq # / len(ss)
                if max(abs(np.real(s.iq))) > 32767 or max(abs(np.imag(s.iq))) > 32767:
                    raise ValueError('Exceeds the maximum allowable output.')
    
    # # 束ねるチャネルを WaveSequence に変換
    awgs = [k for k in channel_map if isinstance(k,AWG)] # channel_map から AWG 関連だけ抜き出す
    wseqs = [(k, chunks2wseq(section[k], period, repeats)) for k in awgs if k in section] # chunk obj を wseq へ変換する

    units = [k for k in channel_map if isinstance(k,UNIT)] 
    capts = [(k,sect2capt(section[k],period,repeats)) for k in units if k in section]
    
    return Setup(*(wseqs + capts))
