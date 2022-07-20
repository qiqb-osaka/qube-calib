from e7awgsw import CaptureModule, CaptureCtrl, CaptureParam, DspUnit, AWG
from e7awgsw import AwgCtrl, IqWave, WaveSequence
import e7awgsw
import numpy as np

class CaptureCtrl(e7awgsw.CaptureCtrl):
    
    def check_err(self, *capu):
        e = super().check_err(*capu)
        if any(e):
            raise IOError('CaptureCtrl error.')
    
    def get_capture_data(self, *units):
        return CaptureData(super(), *units)
    
class CaptureData(object):
    def __init__(self, cap_ctrl, *units):
        self.data = v = {}
        for u in units:
            n = cap_ctrl.num_captured_samples(u)
            c = np.array(cap_ctrl.get_capture_data(u, n))
            v[u] = c[:,0] + 1j * c[:,1]
            
class Recv(object):
    
    def __init__(self, ipaddr, modules):
        self._trigger = None
        self.ipaddr = ipaddr
        self.captms = [o if isinstance(o, CaptureModule) else o.id for o in modules]
        self.data = None
        
    def start(self, param, timeout=5):
        units = CaptureModule.get_units(*self.captms)
        with CaptureCtrl(self.ipaddr) as cap_ctrl:
            cap_ctrl.initialize(*units)
            for i in units:
                cap_ctrl.set_capture_params(i, param)
            cap_ctrl.start_capture_units(*units)
            cap_ctrl.wait_for_capture_units_to_stop(timeout, *units)
            cap_ctrl.check_err(*units)
            self.data = cap_ctrl.get_capture_data(*units)
            
    def wait(self, param, timeout=5):
        units = CaptureModule.get_units(*self.captms)
        with CaptureCtrl(self.ipaddr) as cap_ctrl:
            cap_ctrl.initialize(*units)
            if self._trigger is not None:
                for i in self.captms:
                    cap_ctrl.select_trigger_awg(i, self._trigger)
                    cap_ctrl.enable_start_trigger(*CaptureModule.get_units(i))
            for i in units:
                cap_ctrl.set_capture_params(i, param)
            cap_ctrl.wait_for_capture_units_to_stop(timeout, *units)
            cap_ctrl.check_err(*units)
            self.data = cap_ctrl.get_capture_data(*units)
        
    @property
    def trigger(self):
        return self._trigger
        
    @trigger.setter
    def trigger(self, awg):
        self._trigger = awg if isinstance(awg, AWG) else awg.id
        
        
class Send(object):
    
    def __init__(self, ipaddr, awgs, wave_seqs):
        self.ipaddr = ipaddr
        self.awgs = awgs
        self.wave_seqs = wave_seqs
    
    def start(self):
        awgs, wavs = self.awgs, self.wave_seqs
        with AwgCtrl(self.ipaddr) as awg_ctrl:
            awg_ctrl.initialize(*awgs)
            for a, w in zip(awgs, wavs):
                awg_ctrl.set_wave_sequence(a, w)
            awg_ctrl.start_awgs(*self.awgs)

    def terminate(self):
        with AwgCtrl(self.ipaddr) as awg_ctrl:
            awg_ctrl.terminate_awgs(*self.awgs)
            
    def wait(self, timeout=5):
        awgs, wavs = self.awgs, self.wave_seqs
        with AwgCtrl(self.ipaddr) as awg_ctrl:
            awg_ctrl.initialize(*awgs)
            for a, w in zip(awgs, wavs):
                awg_ctrl.set_wave_sequence(a, w)
            print('wait for started by sequencer.')
            awg_ctrl.wait_for_awgs_to_stop(timeout, *self.awgs)

class WaveChunkFactory(object):
    
    def get_timestamp(self):
        period = 1 / AwgCtrl.SAMPLING_RATE
        return np.arange(0, self.dulation, period)
    
    def __init__(self, dulation=128e-9, amp=32767, blank=0, repeats=1):
        # dulation の初期値は CW 出力を想定して設定した
        # int(dulation * AwgCtrl.SAMPLING_RATE) が 64 の倍数だと切れ目のない波形が出力される．
        # 波形チャンクの最小サイズが 128ns (500Msps の繰り返し周期は 2ns)
        
        self.dulation = dulation
        self.amp = amp
        self.iq = np.zeros(*self.get_timestamp().shape).astype(complex)
        self.blank = blank # [s]
        self.repeats = repeats # times
        
    @property
    def chunk(self):
        
        iq = self.amp * self.iq
        i, q = np.real(iq).astype(int), np.imag(iq).astype(int)
        s = IqWave.convert_to_iq_format(i, q, WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK)
        
        r = AwgCtrl.SAMPLING_RATE
        b = max(int(r * self.blank), 0) # 非負の値を取るように
        n = WaveSequence.NUM_SAMPLES_IN_AWG_WORD
        
        return {'iq_samples': s, 'num_blank_words': b // n, 'num_repeats': self.repeats}
      

# class WaveSequence(e7awgsw.WaveSequence):
    
#     def set_iq(self, iq, amp=32767):
        
#         iq = amp * iq
#         i, q = np.real(iq).astype(int), np.imag(iq).astype(int)
#         s = IqWave.convert_to_iq_format(i, q, self.NUM_SAMPLES_IN_WAVE_BLOCK)
        
#         r = AwgCtrl.SAMPLING_RATE
#         blank = 0.03e-3 # [s] ???
#         b = max(int(r * blank), 0)
#         n = self.NUM_SAMPLES_IN_AWG_WORD
#         self.add_chunk(
#             iq_samples = s,
#             num_blank_words = b // n,
#             num_repeats = 1, # num_cunk_repeats
#         )
        

class SendRecvAwgCtrl(AwgCtrl):
    
    def check_err(self, *awgs):
        
        e = super().check_err(*awgs)
        if any(e):
            raise IOError('AwgCtrl error.')

    def start_wait(self, timeout, *awgs):
        
        self.start_awgs(*awgs)
        self.wait_for_awgs_to_stop(timeout, *awgs)
        

class SendRecvCaptureCtrl(CaptureCtrl):
    
    def check_err(self, *capu):
        
        e = super().check_err(*capu)
        if any(e):
            raise IOError('CaptureCtrl error.')
            
    def wait(self, timeout, *capu):
        
        self.wait_for_capture_units_to_stop(timeout, *capu)
        
    def get_capture_data(self, *capu):
        
        return CaptureData(super(), *capu)
        
    def param(self, wave_seq):
        
        cl = additional_capture_length = 1e-6 # [s]
        d = capture_delay = 0
        ro_chunk = c = wave_seq.chunk(0)
        
        capture_param = p = CaptureParam()
        p.num_integ_sections = c.num_repeats # 積算区間数
        
        # readout 波形の長さから, 追加で 1us キャプチャするためのキャプチャワード数を計算
        sampling_rate = r = CaptureCtrl.SAMPLING_RATE
        nw = CaptureParam.NUM_SAMPLES_IN_ADC_WORD
        cw = int(cl * r) // nw
        cw = min(cw, c.num_blank_words - 1)
        
        sum_section_len = s = c.num_words - c.num_blank_words + cw
        num_blank_words = b = c.num_words - s
        
        p.add_sum_section(s, b)
        p.sum_start_word_no = 0
        p.num_words_to_sum = CaptureParam.MAX_SUM_SECTION_LEN
        p.sel_dsp_units_to_enable(DspUnit.INTEGRATION)
        p.capture_delay = int(d * r) // nw
        
        # readout 波形のサンプル数とキャプチャするサンプル数が一致することを確認
        # assert w.num_all_samples == p.num_samples_to_process
        return p
        
    def set_trigger_awg(self, capm, awg):
        
        self.select_trigger_awg(capm, awg)
        self.enable_start_trigger(*CaptureModule.get_units(capm))
        
        
class SendRecvCaptureParam(CaptureParam):
    
    def __init__(self, wave_seq):
        
        cl = additional_capture_length = 1e-6 # [s]
        d = capture_delay = 0
        ro_chunk = c = wave_seq.chunk(0)
        
        capture_param = p = CaptureParam()
        p.num_integ_sections = c.num_repeats # 積算区間数
        
        # readout 波形の長さから, 追加で 1us キャプチャするためのキャプチャワード数を計算
        sampling_rate = r = CaptureCtrl.SAMPLING_RATE
        nw = CaptureParam.NUM_SAMPLES_IN_ADC_WORD
        cw = int(cl * r) // nw
        cw = min(cw, c.num_blank_words - 1)
        
        sum_section_len = s = c.num_words - c.num_blank_words + cw
        num_blank_words = b = c.num_words - s
        
        p.add_sum_section(s, b)
        p.sum_start_word_no = 0
        p.num_words_to_sum = CaptureParam.MAX_SUM_SECTION_LEN
        p.sel_dsp_units_to_enable(DspUnit.INTEGRATION)
        p.capture_delay = int(d * r) // nw
        
        # readout 波形のサンプル数とキャプチャするサンプル数が一致することを確認
        # assert w.num_all_samples == p.num_samples_to_process
        
        
class CaptureData(object):
    
    def __init__(self, cap_ctrl, *capu):
        
        self.data = v = {}
        
        for u in capu:
            n = cap_ctrl.num_captured_samples(u)
            c = np.array(cap_ctrl.get_capture_data(u, n))
            v[u] = c[:,0] + 1j * c[:,1]
            
            
class SendRecv(object):
    
    def get_timestamp(self):
        period = 1 / CaptureCtrl.SAMPLING_RATE
        return np.arange(0, 10e-6, period)
    
    @classmethod
    def gen_timestamp(cls, iq):
        period = 1 / CaptureCtrl.SAMPLING_RATE
        return np.linspace(0, len(iq) * period, len(iq))
    
    def __init__(self, ipfpga, awgs, capm, dulation=10e-6):
        
        self.ipaddr = ipfpga
        self.awgs = awgs
        self.iqs = [np.zeros(*self.get_timestamp().shape).astype(complex) for o in awgs]
        self.capm = capm
        self.capu = CaptureModule.get_units(capm)
        self.data = None
        self.timeout = 5
        
    def prepare(self, awg_ctrl, cap_ctrl):
        
        awg_ctrl.initialize(*self.awgs) # 2 ms / awg ?
        cap_ctrl.initialize(*self.capu) # 20ms

        cap_ctrl.set_trigger_awg(self.capm, self.awgs[0])

        nww = num_wait_words = 0
        trigw = None
        for awg, iq in zip(self.awgs, self.iqs):
            w = WaveSequence(nww, num_repeats = 1, enable_lib_log = True)
            w.set_iq(iq)
            awg_ctrl.set_wave_sequence(awg, w) # 6 ms
            trigw = w

        p = cap_ctrl.param(trigw)
        for u in self.capu:
            cap_ctrl.set_capture_params(u, p) # 20 ms
        
    def ready(self, awg_ctrl, cap_ctrl):
        
        t = self.timeout
        self._ready(awg_ctrl)

        awg_ctrl.wait_for_awgs_to_stop(t, *self.awgs)
        cap_ctrl.wait_for_capture_units_to_stop(t + 5, *self.capu)

        awg_ctrl.check_err(*self.awgs)
        cap_ctrl.check_err(*self.capu)

        self.data = cap_ctrl.get_capture_data(*self.capu)

    def _ready(self, awg_ctrl):
        
        awg_ctrl.start_awgs(*self.awgs)
        
    @property
    def awg_ctrl(self):
        
        return SendRecvAwgCtrl(self.ipaddr)
        
    @property
    def cap_ctrl(self):
        
        return SendRecvCaptureCtrl(self.ipaddr)
        
        
        
class SendRecvMulti(SendRecv):
    
    def _ready(self, awg_ctrl):
        
        print('wait for started by sequencer.')
        


# from enum import IntEnum
# from e7awgsw import CaptureModule, CaptureCtrl, CaptureParam, AwgCtrl, WaveSequence, DspUnit, IqWave, AWG
# import e7awgsw
# from collections import namedtuple
# import numpy as np
# import math
# import warnings
# from . import qube
# from .qube import PortFunc, Port, Lane, PortNo


# class LongSend(object):
#     @classmethod
#     def gen_wave_seq(cls, freq, amp=32767):
#         wave_seq = e7awgsw.WaveSequence(
#             num_wait_words = 16,
#             num_repeats = 0xFFFFFFFF)

#         num_chunks = 1
#         for _ in range(num_chunks):
#             # int(num_cycles * AwgCtrl.SAMPLING_RATE / freq) を 64 の倍数にすると, 切れ目のない波形が出力される.
#             i_wave = e7awgsw.SinWave(num_cycles = 8, frequency = freq, amplitude = amp, phase = math.pi / 2)
#             q_wave = e7awgsw.SinWave(num_cycles = 8, frequency = freq, amplitude = amp)
#             iq_samples = e7awgsw.IqWave(i_wave, q_wave).gen_samples(
#                 sampling_rate = e7awgsw.AwgCtrl.SAMPLING_RATE, 
#                 padding_size = e7awgsw.WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK)

#             wave_seq.add_chunk(
#                 iq_samples = iq_samples,
#                 num_blank_words = 0, 
#                 num_repeats = 0xFFFFFFFF)
#         return wave_seq
    
#     @classmethod
#     def set_wave_sequence(cls, awg_ctrl, awg, amp, freq):
#         awg_to_wave_sequence = {}
        
#         # 単一の awg のみを受け付けるように変更
#         # for awg_id, a, f in zip(awgs, amps, freqs):
#         #     # print("{}: freq={}, amp={}".format(awg_id, f, a))
#         #     wave_seq = cls.gen_wave_seq(f, a) # 5 MHz  5MHz x 8 周期では切れ目のない波形はできない
#         #     awg_to_wave_sequence[awg_id] = wave_seq
#         #     awg_ctrl.set_wave_sequence(awg_id, wave_seq)
        
#         wave_seq = cls.gen_wave_seq(freq, amp) # 5 MHz  5MHz x 8 周期では切れ目のない波形はできない
#         awg_to_wave_sequence[awg] = wave_seq
#         awg_ctrl.set_wave_sequence(awg, wave_seq)
        
#         return awg_to_wave_sequence
        
#     # 単一の awg のみを受け付けるように変更した
#     # 複数必要な場合は [start(a) for a in awgs] のようにする
#     @classmethod
#     def start(cls, port, lane=0, att=0, freq=2.5e6, amax=32767):
#         if port.active:
#             cls.stop(port)
        
#         awg = port.dac.awgs[lane]
#         ipaddr = port.dac.ipfpga
        
#         amp = amax*10**(-att/20)
#         with e7awgsw.AwgCtrl(ipaddr) as awg_ctrl:
#             # 初期化
#             awg_ctrl.initialize(awg)
#             # 波形シーケンスの設定
#             awg_to_wave_sequence = cls.set_wave_sequence(awg_ctrl, awg, amp, freq)
#             # 波形送信スタート
#             awg_ctrl.start_awgs(awg)
            
#         port.active = True
    
#     @classmethod
#     def stop(cls, port, lane=0):
        
#         awg = port.dac.awgs[lane]
#         ipaddr = port.dac.ipfpga
        
#         with e7awgsw.AwgCtrl(ipaddr) as awg_ctrl:
#             awg_ctrl.terminate_awgs(awg)
            
#         port.active = False
#         # AWG が稼働中を示すフラグをとりあえずつけた．でもできればlsiから読み出したい．

# class Recv(object):
#     capture_delay = 100
    
#     @classmethod
#     def set_capture_params(cls, cap_ctrl, num_capture_words, capture_units):
#         capture_param = cls.gen_capture_param(num_capture_words)
#         for captu_unit_id in capture_units:
#             cap_ctrl.set_capture_params(captu_unit_id, capture_param)

#     @classmethod
#     def gen_capture_param(cls, num_capture_words):
#         capture_param = e7awgsw.CaptureParam()
#         capture_param.num_integ_sections = 1
#         capture_param.add_sum_section(num_capture_words, 1) # 総和区間を 1 つだけ定義する
#         capture_param.capture_delay = cls.capture_delay
#         return capture_param

#     @classmethod
#     def get_capture_data(cls, cap_ctrl, capture_units):
#         capture_unit_to_capture_data = {}
#         for capture_unit_id in capture_units:
#             num_captured_samples = cap_ctrl.num_captured_samples(capture_unit_id)
#             capture_unit_to_capture_data[capture_unit_id] = (
#                 cap_ctrl.get_capture_data(capture_unit_id, num_captured_samples))
#         return capture_unit_to_capture_data
    
#     @classmethod
#     def check_err(cls, cap_ctrl, capture_units):
#         cap_unit_to_err = cap_ctrl.check_err(*capture_units)
#         for cap_unit_id, err_list in cap_unit_to_err.items():
#             print('{} err'.format(cap_unit_id))
#             for err in err_list:
#                 print('    {}'.format(err))

#     @classmethod
#     def start(cls, port, num_capture_words=1024):
#         ipaddr = port.adc.ipfpga
#         capture_modules = port.adc.caps
#         capture_units = e7awgsw.CaptureModule.get_units(*capture_modules)
#         with e7awgsw.CaptureCtrl(ipaddr) as cap_ctrl:
#             # 初期化
#             cap_ctrl.initialize(*capture_units)
#             # キャプチャパラメータの設定
#             cls.set_capture_params(cap_ctrl, num_capture_words, capture_units)
#             # キャプチャスタート
#             cap_ctrl.start_capture_units(*capture_units)
#             # キャプチャ完了待ち
#             cap_ctrl.wait_for_capture_units_to_stop(5, *capture_units)
#             # エラーチェック
#             cls.check_err(cap_ctrl, capture_units)
#             # キャプチャデータ取得
#             capture_unit_to_capture_data = cls.get_capture_data(cap_ctrl, capture_units)
#         return capture_unit_to_capture_data
#             # 波形保存
#             #save_sample_data('capture', CaptureCtrl.SAMPLING_RATE, capture_unit_to_capture_data)
#             #print('end')

# class SimpleSendRecvProto(object):
    
#     class WaveProperty(object):
        
#         def __init__(self, t0 = 0, phi = 0, mag = 32767, mhz = 0):
            
#             self.t0 = t0 # 位相基準時間 [s]
#             self.phi = phi # 位相基準時間での位相オフセット [rad]
#             assert mag <= 32767, 'Mag must be less than 32767'
#             self.mag = mag # 信号の振幅 15 bits
#             self.mhz = mhz # SSB変調周波数 [MHz]
    
#     class WaveSequenceFactory(object):

#         def __init__(self, num_wait_words, num_repeats, duration, *, enable_lib_log = True):
            
#             self._num_wait_words = num_wait_words
#             self._num_repeats = 1
#             self._enable_lib_log = enable_lib_log
            
#             self._duration = duration
#             self._timeline = self._gen_timeline()
#             self.iq = self.iq_template
            
#             self.blank = 0.03e-3 # [s]
#             self.num_chunk_repeats = 1
            
#         def _gen_timeline(self):
            
#             padding_size = p = WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK
#             period = 1 / CaptureCtrl.SAMPLING_RATE
#             t = np.arange(0, self._duration, period)
#             if len(t) % p != 0:
#                 num_samples_to_add = a = p - (len(t) % p)
#                 t = np.concatenate([t, np.arange(self._duration, self._duration + period * a, period)])
#             return t
        
#         @property
#         def iq_template(self):
#             return np.zeros(len(self.timeline)).astype(complex)
        
#         @property
#         def timeline(self):
#             return self._timeline

#         @property
#         def duration(self):
#             return self._duration
        
#         @property
#         def awg(self):
#             return self._awg
        
#         def produce(self, wprop):
            
#             iq = wprop.mag * self.iq * np.exp(1j * 2 * np.pi * wprop.mhz * 1e+6 * (self.timeline - wprop.t0) + wprop.phi)
#             i, q = np.real(iq).astype(int), np.imag(iq).astype(int)
#             s = IqWave.convert_to_iq_format(i, q, WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK)
            
#             assert len(iq) == len(s)
            
#             num_blank_samples = max(int(AwgCtrl.SAMPLING_RATE * self.blank), 0) # 非負値
            
#             wseq = WaveSequence(self._num_wait_words, self._num_repeats, enable_lib_log = self._enable_lib_log)
            
#             wseq.add_chunk(
#                 iq_samples = s,
#                 num_blank_words = num_blank_samples // WaveSequence.NUM_SAMPLES_IN_AWG_WORD,
#                 num_repeats = self.num_chunk_repeats,
#             )
            
#             return wseq
        
#     def __init__(self, duration, wait_words, readin_port, readout_awg):
        
#         self._duration = duration
#         self._wait_words = wait_words
       
#         self.readin_port = readin_port
#         self._readout_awg = readout_awg
        
#         self.additional_capture_length = 1e-6 # [s]
#         self.capture_delay = 0
#         self._capture_data = {}
#         self.sequence = {}
#         self.wave_property = {}
        
#         self.repeats = 1
#         self.trigger = None
        
#     @property
#     def readin_port(self):
#         return self._readin_port

#     @readin_port.setter
#     def readin_port(self, v):
#         assert isinstance(v, Port) or isinstance(v, list), 'Must be instance of qubecalib.qube.Port or list'
#         if isinstance(v, list):
#             self._readin_port = l = v
#             self._ipfpga = v[0].adc.ipfpga
#         else:
#             self._readin_port = l = [v,]
#             self._ipfpga = v.adc.ipfpga
#         self._capture_modules = c = sum([p.adc.caps for p in l], [])
#         self._capture_units = CaptureModule.get_units(*c)

#     def assign(self, awg, seq):
        
#         self.sequence[awg] = seq
        
#         if awg not in self.wave_property:
#             self.wave_property[awg] = self.WaveProperty()
        
#         if self.trigger is None:
#             self.trigger = awg
    
#     @property
#     def duration(self):
#         return self._duration
        
#     @property
#     def wait_words(self):
#         return self._wait_words
        
#     @wait_words.setter
#     def wait_words(self, v):
#         self._num_wait_words = v
        
#     def new_sequence(self, awg = None):
#         # 長さの揃ったシーケンスを返す
#         opt = {
#             'num_wait_words': self.wait_words,
#             'num_repeats': self.repeats,
#             'duration': self.duration,
#         }
#         w = self.WaveSequenceFactory(**opt)
#         if awg is not None:
#             self.assign(awg, w)
#         return w
    
#     @property
#     def awg_list(self):
        
#         return list(self.sequence)
        
#     @property
#     def timeline(self):
        
#         duration = 10e-6 # sec.
#         period = 1 / CaptureCtrl.SAMPLING_RATE
#         return np.arange(0, duration, period)
        
#     @property
#     def capture_data(self):
#         return self._capture_data
        
#     def downconv(self, x, awg):
#         w = self.wave_property[awg]
#         t = np.linspace(0, x.shape[0] / CaptureCtrl.SAMPLING_RATE, x.shape[0]) - w.t0
#         e = np.exp(-1j * 2 * np.pi * w.mhz * 1e+6 * t + w.phi)
#         return x * e, t
        
#     def start(self):
        
#         def gen_capture_param(ro_wave_seq):
            
#             ro_chunk = c = ro_wave_seq.chunk(0)
            
#             capture_param = p = CaptureParam()
#             p.num_integ_sections = c.num_repeats # 積算区間数

#             # readout 波形の長さから, 追加で 1us キャプチャするためのキャプチャワード数を計算
#             sampling_rate = CaptureCtrl.SAMPLING_RATE
#             num_samples_in_adc_word = CaptureParam.NUM_SAMPLES_IN_ADC_WORD
#             additional_capture_words = w = int(self.additional_capture_length * sampling_rate) // num_samples_in_adc_word
#             w = min(w, c.num_blank_words - 1)

#             sum_section_len = s = c.num_words - c.num_blank_words + w
#             num_blank_words = b = c.num_words - s

#             p.add_sum_section(s, b)
#             p.sum_start_word_no = 0
#             p.num_words_to_sum = CaptureParam.MAX_SUM_SECTION_LEN
#             p.sel_dsp_units_to_enable(DspUnit.INTEGRATION)
#             p.capture_delay = int(self.capture_delay * sampling_rate) // num_samples_in_adc_word

#             # readout 波形のサンプル数とキャプチャするサンプル数が一致することを確認
#             assert ro_wave_seq.num_all_samples == capture_param.num_samples_to_process
#             return capture_param

#         def set_trigger_awg(cap_module, awg):

#             c = cap_ctrl
#             c.select_trigger_awg(cap_module, awg)
#             c.enable_start_trigger(*CaptureModule.get_units(cap_module))
        
#         def check_err(awgs):

#             awg_to_err = awg_ctrl.check_err(*awgs)
#             for awg_id, err_list in awg_to_err.items():
#                 print(awg_id)
#                 for err in err_list:
#                     print('    {}'.format(err))

#             cap_unit_to_err = cap_ctrl.check_err(*capture_units)
#             for cap_unit_id, err_list in cap_unit_to_err.items():
#                 print('{} err'.format(cap_unit_id))
#                 for err in err_list:
#                     print('    {}'.format(err))

#         def get_capture_data():
#             data = {}
#             for c in capture_units:
#                 num_captured_samples = n = cap_ctrl.num_captured_samples(c)
#                 d = np.array(cap_ctrl.get_capture_data(c, n))
#                 data[c] = d[:,0] + 1j * d[:,1]
#             return data

#         capture_modules = self._capture_modules
#         capture_units = self._capture_units
        
#         # readout 用の waves_sequence が登録されているかチェック
#         if self._readout_awg not in self.sequence:
#             raise('WaveSequence assignment for Readout is required.')

#         for v in self.sequence.values():
#             v.num_chunk_repeats = self.repeats
        
#         with (AwgCtrl(self._ipfpga) as awg_ctrl, CaptureCtrl(self._ipfpga) as cap_ctrl):
        
#             # 初期化
#             awg_ctrl.initialize(*self.awg_list)
#             cap_ctrl.initialize(*capture_units)

#             # トリガ AWG の設定
#             for m in capture_modules:
#                 set_trigger_awg(m, self.trigger)

#             # 波形シーケンスの設定
#             trig_wave_seq = None
#             for k, v in self.sequence.items():
#                 w = v.produce(self.wave_property[k])
#                 awg_ctrl.set_wave_sequence(k, w)
#                 if k == self.trigger:
#                     trig_wave_seq = w
            
#             # キャプチャパラメータの設定
#             for c in capture_units:
#                 p = gen_capture_param(trig_wave_seq)
#                 cap_ctrl.set_capture_params(c, p)

#             # 波形送信スタート
#             awg_ctrl.start_awgs(*self.awg_list)

#             # 波形送信完了待ち
#             awg_ctrl.wait_for_awgs_to_stop(5, *self.awg_list)

#             # キャプチャ完了待ち
#             cap_ctrl.wait_for_capture_units_to_stop(5, *capture_units)

#             # エラーチェック
#             check_err(self.awg_list)

#             # キャプチャデータ取得
#             self._capture_data = get_capture_data()

#         return self._capture_data

