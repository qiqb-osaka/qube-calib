from .qube import CPT, AWG
from .meas import WaveSequenceFactory, CaptureModule, CaptureCtrl, CaptureParam, AwgCtrl
from .setupqube import _conv_to_e7awgsw, _conv_channel_for_e7awgsw
from .pulse import Read
import e7awgsw

import os
import time
import struct
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import warnings
import datetime
import numpy as np
# from e7awgsw import AwgCtrl, CaptureCtrl, CaptureParam, CaptureModule, AWG
# from qubecalib.qube import CPT
# from qubecalib.meas import WaveSequenceFactory
# from qubecalib.setupqube import _conv_to_e7awgsw, _conv_channel_for_e7awgsw
from qube_master.software.qubemasterclient import QuBEMasterClient
from qube_master.software.sequencerclient import SequencerClient

PORT = 16384
IPADDR = '10.3.0.255'
REPEAT_WAIT_SEC = 0.1
REPEAT_WAIT_SEC = int(REPEAT_WAIT_SEC * 125000000) # 125Mcycles = 1sec
CANCEL_STOP_PACKET = struct.pack(8*'B', 0x2c, *(7*[0]))

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
                    k.port().delay
                except AttributeError as e:
                    raise AttributeError('delay attribute is required for receiver port')
        r = _conv_to_e7awgsw(adda_to_channels=channels, offset=offset, repeats=1, interval=interval, trigger_awg=None)
        # captparam を複製して DSP の設定をクリアする
        func = lambda v: {k2:(cls.duplicate_captparam(v2),) for k2, v2 in v['capt_to_captparam'].items()}
        qube_channels = lambda qube: {k:v for k, v in channels.items() if k.port().qube() == qube}
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
        
        print(list(set([k.port().qube() for k, v in channels.items()])))
        
        # チャネルを周波数変換し，時間領域でチャネルを結合し，e7awgsw の制約に合わせてスロットを再配置する
        w2c = dict([(k, _conv_channel_for_e7awgsw(v, k, 0)) for k, v in channels.items()])
        # quantized_channel? merged_channel?
        
        print([w.port().delay for w, c in w2c.items() if isinstance(w, CPT)])
        
        return w2c
        # return _conv_to_e7awgsw(adda_to_channels)


class Recv(CaptureCtrl):
    
    def __init__(self, qube, *module_params_pair):
        
        argparse = lambda module, *params: (module, params)
        cond = lambda o: isinstance(o,tuple) or isinstance(o,list)
        arg = tuple([argparse(o[0],*(o[1] if cond(o[1]) else (o[1],))) for o in module_params_pair])
        
        # typing で書くのがいまどき？
        if not [isinstance(m,CPT) for m, l in arg] == len(arg)*[True]:
            raise TypeError('1st element of each tuple should be qubecalib.meas.CPT instance.')
        
        if not [qube == m.port().qube() for m, l in arg] == len(arg)*[True]:
            raise Exception('The qube that owns the CaptureModule candidates in the arguments must all be identical.')
            
        if not [len(l) < 5 for m, l in arg] == len(arg)*[True]:
            raise Exception('Each CaptureParameter list in the argument must have no longer than 4 elements.')
                
        super().__init__(qube.ipfpga)
        
        self._trigger = None # obsoleted
        self.modules = [m for m, l in arg]
        self.units = sum([self.assign_param_to_unit(m, l) for m, l in arg],[])
        
    def assign_param_to_unit(self, module, params):
        
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
    
    def wait_for_trigger(self, awg):
        
        trig = awg if isinstance(awg, e7awgsw.AWG) else awg.id
        for m in self.modules:
            self.select_trigger_awg(m.id, trig)
        self.enable_start_trigger(*self.units)
    
    def wait_for_capture(self, timeout=30):
        
        u = self.units
        self.wait_for_capture_units_to_stop(timeout, *u)
        self.check_err(*u)
    
    def get_data(self, unit):
        
        n = self.num_captured_samples(unit)
        c = np.array(self.get_capture_data(unit,n))
        return c[:,0] + 1j * c[:,1]
    
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
    
    def __init__(self, qube, *awg_seqfactory_pair):
        
        arg = awg_seqfactory_pair
        
        # typing で書くのがいまどき？
        if not [isinstance(a, AWG) and isinstance(s, WaveSequenceFactory) for a, s in arg] == len(arg)*[True]:
            raise TypeError('Element type of each tuple should be (qubecalib.qube.AWG, qubecalib.meas.WaveSequenceFactory).')
        
        if not [qube == a.port().qube() for a, s in arg] == len(arg)*[True]:
            raise Exception('The qube that owns the AWG candidates in the arguments must all be identical.')
            
        super().__init__(qube.ipfpga)
        
        self.awgs = awgs = [a for a, s in arg]
        
        self.initialize(*[a.id for a in awgs])
        for a, s in arg:
            self.set_wave_sequence(a.id, s.sequence)
            
    def start(self):
        
        a = [a.id for a in self.awgs]
        self.terminate_awgs(*a)
        self.clear_awg_stop_flags(*a)
        self.start_awgs(*a)
        
    def wait_for_sequencer(self, timeout=30):
        
        a = [a.id for a in self.awgs]
        # self.terminate_awgs(*a)
        self.clear_awg_stop_flags(*a)
        print('wait:', datetime.datetime.now())
        print('wait for started by sequencer for {}'.format(self.awgs[0].port().qube().ipfpga))
        self.wait_for_awgs_to_stop(timeout, *a)
        print('awg done:', datetime.datetime.now())
        print('end')
        
# class SendSingleAwg(Send):
    
#     def __init__(self, awg, sequence):
        
#         super().__init__(((awg, sequence),))

class Terminate(AwgCtrl):
    
    def __init__(self, awgs):
        
        super().__init__(awgs.port().qube().ipfpga)
        
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
        trigger = [o for o in triggers if o.port().qube() == qube][0]
        units = multishot_single(qube, qube_to_pulse[qube], trigger, repeats, timeout, interval)
    else:
        units = multishot_multi(qube_to_pulse, triggers, repeats, timeout, interval)

    return units

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
    t = {awg.port().qube(): awg for awg in triggers}
    
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
                r, a = s.add_sequencer(mark)
                
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
         units = [thread.result()]
    
    for captm, p in arg_recv:
        channels = pulse.adda_to_channels[captm]
        multishot_get_data(captm, channels)
    
    return units

def multishot_get_data(captm, channels):

    units = CaptureModule.get_units(captm.id)
    unit_to_channel = {units[i]: v for i, v in enumerate(channels)}

    with CaptMemory(captm.port().qube().ipfpga) as m:
        for unit, channel in unit_to_channel.items():
            slot = channel.findall(Read)[0]
            v = m.get_data(unit)
            t = np.arange(0, len(v)) / CaptureCtrl.SAMPLING_RATE
            v *= np.exp(-1j * 2 * np.pi * captm.modulation_frequency(channel.center_frequency*1e-6)*1e+6 * t)
            d = slot.duration
            slot.iq = v


def _singleshot(adda_to_channels, triggers, repeats, timeout=30, interval=50000):
    
    c = PulseConverter.conv(adda_to_channels, interval)
    
    if len(c.keys()) == 1:
        qube = tuple(c.keys())[0]
        trigger = [o for o in triggers if o.port().qube() == qube][-1]
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
         units = [thread.result()]
    
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
    t = {awg.port().qube(): awg for awg in triggers}
    
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
                r, a = s.add_sequencer(mark)
                
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
    with CaptMemory(captm.port().qube().ipfpga) as m:
        v = m.get_data(unit)
        v = v.reshape(repeats,int(len(v)/repeats))
        t = np.arange(0,len(v[0])) / CaptureCtrl.SAMPLING_RATE
        v *= np.exp(-1j * 2 * np.pi * captm.modulation_frequency(channel.center_frequency*1e-6)*1e+6 * t)
        d = slot.duration
        m = max([s.sampling_rate for s in channel])
        slot.iq = v
    