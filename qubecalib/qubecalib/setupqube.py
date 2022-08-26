import e7awgsw
import time
import numpy as np
from typing import Final
from concurrent.futures import ThreadPoolExecutor
from .qube import Readout, Readin, Ctrl, SSB
from .pulse import Read, Schedule, Blank, Rect, Arbit
from .meas import Send, Recv, WaveSequenceFactory, CaptureParam, CaptureModule, CaptureCtrl, AwgCtrl

def words(t): # in ns
    return int(t * 1e-9 * AwgCtrl.SAMPLING_RATE // e7awgsw.hwparam.NUM_SAMPLES_IN_AWG_WORD)

def find_allports(schedule, klass):
    r = []
    for k, v in schedule.items():
        if isinstance(v.wire.port, klass):
            r.append(v)
    return r

def select_band(channel):
    fc = channel.center_frequency
    for b in channel.wire.band:
        if b.range[0] < fc and fc <= b.range[1]:
            return b
    raise ValueError('Invalid center_frequency {}.'.format(fc))

def calc_modulation_frequency(channel):
    lo_mhz = channel.wire.port.lo.mhz 
    cnco_mhz = channel.wire.cnco_mhz
    fnco_mhz = select_band(channel).fnco_mhz
    rf_usb = (lo_mhz + (cnco_mhz + fnco_mhz)) * 1e+6
    rf_lsb = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e+6
    if isinstance(channel.wire.port, Readin):
        fc = channel.center_frequency - rf_usb
    else:
        if channel.wire.port.mix.ssb == SSB.USB:
            fc = channel.center_frequency - rf_usb
        elif channel.wire.port.mix.ssb == SSB.LSB:
            fc = rf_lsb - channel.center_frequency
        else:
            raise ValueError('A port.mix.ssb shuld be instance of SSB(Enum).')
    return fc
    
def conv_wseq(channel, repeats, interval, fm, schedule):
    s = schedule
    c = channel.copy()
    if isinstance(c[0], Blank):
        wait = c[0].duration
        c.pop(0)
    else:
        wait = 0
        
    w = WaveSequenceFactory(num_wait_words=words(wait), num_repeats=1)
    
    for ci in c:
        if isinstance(ci, Arbit):
            w.new_chunk(duration=ci.duration*1e-9, amp=int(ci.amplitude * 32767), blank=0)
            t = channel.get_timestamp(ci) * 1e-9 - s.offset * 1e-9
            fc = calc_modulation_frequency(fm[channel.wire][0])
            w.chunk[-1].iq[:] = ci.iq[:] * np.exp(1j * 2 * np.pi * fc * t)
            if len(fm[channel.wire]) > 1:
                for cx in fm[channel.wire][1:]:
                    fc = calc_modulation_frequency(cx)
                    w.chunk[-1].iq[:] += ci.iq[:] * np.exp(1j * 2 * np.pi * fc * t)
            continue
        if isinstance(ci, Rect):
            w.new_chunk(duration=ci.duration*1e-9, amp=int(ci.amplitude * 32767), blank=0, init=1)
            t = channel.get_timestamp(ci) * 1e-9 - s.offset * 1e-9
            fc = calc_modulation_frequency(fm[channel.wire][0])
            w.chunk[-1].iq[:] = w.chunk[-1].iq * np.exp(1j * 2 * np.pi * fc * t)
            if len(fm[channel.wire]) > 1:
                for cx in fm[channel.wire][1:]:
                    fc = calc_modulation_frequency(cx)
                    w.chunk[-1].iq[:] += w.chunk[-1].iq * np.exp(1j * 2 * np.pi * fc * t)
            continue
        if isinstance(ci, Blank):
            w.chunk[-1].blank += ci.duration * 1e-9
            continue
        raise ValueError('Invalid LWC.')
        
    w.chunk[-1].repeats = repeats
    w.chunk[-1].blank += wait * 1e-9
    w.chunk[-1].blank += interval * 1e-9
    
    return w
    
def conv_capt_param(channel, repeats, interval):
    delay_words = words(channel.wire.delay)
    blank_words = 0
    c = channel.copy()
    
    if isinstance(c[0], Blank):
        w = words(c[0].duration)
        delay_words += w
        blank_words += w
        c.pop(0)
        
    for ci in c:
        if isinstance(ci, Read):
            capture_words = words(ci.duration)
            continue
        if isinstance(ci, Blank):
            blank_words += words(ci.duration)
            continue
        raise ValueError('Invalid LWC.')
    
    p = CaptureParam()
    p.num_integ_sections = repeats
    if not (repeats == 1):
        p.add_sum_section(capture_words, blank_words + words(interval))
        p.sum_start_word_no = 0
        p.num_words_to_sum = e7awgsw.CaptureParam.MAX_SUM_SECTION_LEN
        p.sel_dsp_units_to_enable(e7awgsw.DspUnit.INTEGRATION)
    p.capture_delay = delay_words
    
    return p
    
def run(schedule, repeats=1, interval=100000):
    s: Final[Schedule] = schedule
    
    # Search Ctrl
    fm = {}
    for c in find_allports(s, Ctrl):
        if c.wire in fm:
            fm[c.wire].append(c)
        else:
            fm[c.wire] = [c]
    ctrl = [(c, conv_wseq(c, repeats, interval, fm, s)) for c in find_allports(s, Ctrl)]
    
    # Search Readout
    fm = {}
    for c in find_allports(s, Readout):
        if c.wire in fm:
            fm[c.wire].append(c)
        else:
            fm[c.wire] = [c]
    readout = [(c, conv_wseq(c, repeats, interval, fm, s)) for c in find_allports(s, Readout)]

    # Search Readin
    readin = [(c, conv_capt_param(c, repeats, interval)) for c in find_allports(s, Readin)]
    
    # [Todo]: Read を複数使えるように拡張（複数の capt, 対応するトリガの設定）
    o = Send(
        readout[0][0].wire.port.awg.ipfpga,
        [o.wire.port.awg for o, w in readout + ctrl],
        [w.sequence for o, w in readout + ctrl]
    )
    w = dict([(c.wire, p) for c, p in readin])
    r = Recv(
        readin[0][0].wire.port.capt.ipfpga,
        [v.port.capt for v, p in w.items()]
    )
    r.trigger = readout[0][0].wire.port.awg
    
    with ThreadPoolExecutor() as e:
        rslt = e.submit(lambda: r.wait(readin[0][1], timeout=5))
        time.sleep(0.1)
        o.terminate()
        o.start()
        rslt.result()
        
    o.terminate()
    
    for c, p in readin:
        k = CaptureModule.get_units(c.wire.port.capt.id)[0]
        c.timestamp = t = (c.get_timestamp(c.findall(Read)[0]) - s.offset) * 1e-9
        c.iq = np.zeros(r.data[k].shape).astype(complex)
        c.iq[:] = r.data[k] * np.exp(-1j * 2 * np.pi * calc_modulation_frequency(c) * t)
    
    
def maintenance_run(schedule, repeats=1, interval=100000, duration=5242880, capture_delay=0):
    s: Final[Schedule] = schedule
    
    # Search Ctrl
    fm = {}
    for c in find_allports(s, Ctrl):
        if c.wire in fm:
            fm[c.wire].append(c)
        else:
            fm[c.wire] = [c]
    ctrl = [(c, conv_wseq(c, repeats, interval, fm, s)) for c in find_allports(s, Ctrl)]
    
    # Search Readout
    fm = {}
    for c in find_allports(s, Readout):
        if c.wire in fm:
            fm[c.wire].append(c)
        else:
            fm[c.wire] = [c]
    readout = [(c, conv_wseq(c, repeats, interval, fm, s)) for c in find_allports(s, Readout)]

    # Search Readin
    readin = [(c, conv_capt_param(c, repeats, interval)) for c in find_allports(s, Readin)]
    for c, p in readin:
        p.add_sum_section(words(duration), 1)
        p.capture_delay += words(capture_delay)
    
    # [Todo]: Read を複数使えるように拡張（複数の capt, 対応するトリガの設定）
    o = Send(
        readout[0][0].wire.port.awg.ipfpga,
        [o.wire.port.awg for o, w in readout + ctrl],
        [w.sequence for o, w in readout + ctrl]
    )
    w = dict([(c.wire, p) for c, p in readin])
    r = Recv(
        readin[0][0].wire.port.capt.ipfpga,
        [v.port.capt for v, p in w.items()]
    )
    r.trigger = readout[0][0].wire.port.awg
    
    with ThreadPoolExecutor() as e:
        rslt = e.submit(lambda: r.wait(readin[0][1], timeout=5))
        time.sleep(0.1)
        o.terminate()
        o.start()
        rslt.result()
        
    o.terminate()
    
    for c, p in readin:
        k = CaptureModule.get_units(c.wire.port.capt.id)[0]
        c.iq = np.zeros(r.data[k].shape).astype(complex)
        c.timestamp = t = (np.arange(0, r.data[k].shape[0] / AwgCtrl.SAMPLING_RATE, 1 / AwgCtrl.SAMPLING_RATE) - s.offset * 1e-9)
        c.iq[:] = r.data[k] * np.exp(-1j * 2 * np.pi * calc_modulation_frequency(c) * t)
