'''Calibration package for QuBE'''

__all__ =[
    'Qube',
    'LongSend',
    'Recv',
]

from .qube import Qube

import os
import math
import subprocess
import yaml
import e7awgsw
import qubelsi.qube
from collections import namedtuple


PATH_TO_BITFILE = '/home/qube/bin/'
PATH_TO_API = './adi_api_mod'
PATH_TO_CONFIG = './.config/'

class Qube64(object):
    def __init__(self, config_file_name=None):
        pass

class Qube16(object):
    def __init__(self, config_file_name=None):
        pass


class AD9082(object):
    def __init__(self, lsi, ch, ipfpga):
        self.lsi = lsi
        self.ch = ch
        self.fnco = 1000
        self.ipfpga = ipfpga
    def get_fnco(self):
        return self.fnco # MHz

class AD9082DAC(AD9082):
    def __init__(self, lsi, ch, ipfpga, awgs):
        super().__init__(lsi, ch, ipfpga)
        self.awgs = awgs
    def set_fnco(self, mhz):
        self.fnco = mhz # ここはlsi読み出しに変更したい
        self.lsi.set_nco(freq=mhz*1e+6, ch=self.ch)

class AD9082ADC(AD9082):
    def __init__(self, lsi, ch, ipfpga, caps):
        super().__init__(lsi, ch, ipfpga)
        self.caps = caps
    def set_fnco(self, mhz):
        self.fnco = mhz # ここはlsi読み出しに変更したい
        self.lsi.set_nco(freq=mhz*1e+6, ch=self.ch, adc_mode=True)

# DAC
# 1CB ~ 1D0 まで 48 bit
# アドレス選択
# 0x001B [3:0] DAC MASK

# ADC
# 0x0A05 ~ 0x0A0A まで 48 bit
# 0x0018 [3:0] ADC MASK

class OutputPort(object):
    def __init__(self, local, dac, upconv):
        self.local = local
        self.dac = dac
        self.upconv = upconv
        self.active = False
    def set_lo(self, mhz): # MHz
        self.local.set_freq(mhz)
    def set_if(self, mhz): # MHz
        self.dac.set_fnco(mhz)
    def set_usb(self):
        self.upconv.lsi.set_usb()
    def set_lsb(self):
        self.upconv.lsi.set_lsb()
    def get_status(self):
        fl, fi = self.local.get_freq(), self.dac.fnco
        is_usb = self.upconv.is_setUSB()
        r = ''
        r += 'RF = {:>5.3f} MHz '.format(fl + fi if is_usb else fl - fi)
        r += 'LO = {:>5.0f}    MHz '.format(int(fl))
        r += 'IF = {:>5.3f} MHz '.format(fi)
        r += 'LSB/USB: {} MODE '.format('USB' if is_usb else 'LSB')
        r += 'AWG: {}'.format('Active' if self.active else 'Inactive')
        return r
    
class CtrlPort(OutputPort):
    def __init__(self, local, dac, upconv):
        super().__init__(local, dac, upconv)
        self.set_lsb()

class ReadoutPort(OutputPort):
    def __init__(self, local_osc, dac, up_conv):
        super().__init__(local_osc, dac, up_conv)
        self.set_usb()

class ReadinPort(object):
    def __init__(self, local, adc):
        self.local = local
        self.adc = adc
        self.active = False
    def set_lo(self, mhz): # MHz
        self.local.set_freq(mhz)
    def set_if(self, mhz): # MHz
        self.adc.set_fnco(mhz)
    def get_status(self):
        fl, fi = self.local.get_freq(), self.adc.fnco
        r = ''
        r += 'RF = {:>5.3f} MHz '.format(fl + fi)
        r += 'LO = {:>5.0f}    MHz '.format(int(fl))
        r += 'IF = {:>5.3f} MHz '.format(fi)
        return r
        
        
class UpConverter(object):
    Vatt = namedtuple('Vatt', ('lsi', 'ch'))
    def __init__(self, lsi, vatt):
        self.lsi = lsi
        self.vatt = vatt
    def is_setUSB(self):
        return True if self.lsi.read_mode() == 0 else False

class LocalOscillator(object):
    def __init__(self, lsi):
        self.lsi = lsi
    def set_freq(self, mhz): # 100 MHz or less truncated
        v = math.floor(mhz/100)
        self.lsi.write_freq_100M(v)
    def get_freq(self):
        return self.lsi.read_freq_100M() * 100

class LongSend(object):
    @classmethod
    def gen_wave_seq(cls, freq, amp=32767):
        wave_seq = e7awgsw.WaveSequence(
            num_wait_words = 16,
            num_repeats = 0xFFFFFFFF)

        num_chunks = 1
        for _ in range(num_chunks):
            # int(num_cycles * AwgCtrl.SAMPLING_RATE / freq) を 64 の倍数にすると, 切れ目のない波形が出力される.
            i_wave = e7awgsw.SinWave(num_cycles = 8, frequency = freq, amplitude = amp, phase = math.pi / 2)
            q_wave = e7awgsw.SinWave(num_cycles = 8, frequency = freq, amplitude = amp)
            iq_samples = e7awgsw.IqWave(i_wave, q_wave).gen_samples(
                sampling_rate = e7awgsw.AwgCtrl.SAMPLING_RATE, 
                padding_size = e7awgsw.WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK)

            wave_seq.add_chunk(
                iq_samples = iq_samples,
                num_blank_words = 0, 
                num_repeats = 0xFFFFFFFF)
        return wave_seq
    
    @classmethod
    def set_wave_sequence(cls, awg_ctrl, awgs, amps, freqs):
        awg_to_wave_sequence = {}
        for awg_id, a, f in zip(awgs, amps, freqs):
            # print("{}: freq={}, amp={}".format(awg_id, f, a))
            wave_seq = cls.gen_wave_seq(f, a) # 5 MHz  5MHz x 8 周期では切れ目のない波形はできない
            awg_to_wave_sequence[awg_id] = wave_seq
            awg_ctrl.set_wave_sequence(awg_id, wave_seq)
        return awg_to_wave_sequence
    
    @classmethod
    def start(cls, port, atts=[0, 0, 0], freqs=[2.5e6, 2.5e6, 2.5e6]):
        if port.active:
            cls.stop(port)
        ipaddr = port.dac.ipfpga
        amps = [10922*10**(-v/20) for v in atts]
        awgs = port.dac.awgs
        with e7awgsw.AwgCtrl(ipaddr) as awg_ctrl:
            awg_ctrl = e7awgsw.AwgCtrl(ipaddr)
            # 初期化
            awg_ctrl.initialize(*awgs)
            # 波形シーケンスの設定
            awg_to_wave_sequence = cls.set_wave_sequence(awg_ctrl, awgs, amps, freqs)
            # 波形送信スタート
            awg_ctrl.start_awgs(*awgs)
        port.active = True
    
    @classmethod
    def stop(cls, port):
        ipaddr = port.dac.ipfpga
        awgs = port.dac.awgs
        awg_ctrl = e7awgsw.AwgCtrl(ipaddr)
        awg_ctrl.terminate_awgs(*awgs)
        port.active = False
        # AWG が稼働中を示すフラグをとりあえずつけた．でもできればlsiから読み出したい．

class Recv(object):
    capture_delay = 100
    
    @classmethod
    def set_capture_params(cls, cap_ctrl, num_capture_words, capture_units):
        capture_param = cls.gen_capture_param(num_capture_words)
        for captu_unit_id in capture_units:
            cap_ctrl.set_capture_params(captu_unit_id, capture_param)

    @classmethod
    def gen_capture_param(cls, num_capture_words):
        capture_param = e7awgsw.CaptureParam()
        capture_param.num_integ_sections = 1
        capture_param.add_sum_section(num_capture_words, 1) # 総和区間を 1 つだけ定義する
        capture_param.capture_delay = cls.capture_delay
        return capture_param

    @classmethod
    def get_capture_data(cls, cap_ctrl, capture_units):
        capture_unit_to_capture_data = {}
        for capture_unit_id in capture_units:
            num_captured_samples = cap_ctrl.num_captured_samples(capture_unit_id)
            capture_unit_to_capture_data[capture_unit_id] = (
                cap_ctrl.get_capture_data(capture_unit_id, num_captured_samples))
        return capture_unit_to_capture_data
    
    @classmethod
    def check_err(cls, cap_ctrl, capture_units):
        cap_unit_to_err = cap_ctrl.check_err(*capture_units)
        for cap_unit_id, err_list in cap_unit_to_err.items():
            print('{} err'.format(cap_unit_id))
            for err in err_list:
                print('    {}'.format(err))

    @classmethod
    def start(cls, port, num_capture_words=1024):
        ipaddr = port.adc.ipfpga
        capture_modules = port.adc.caps
        capture_units = e7awgsw.CaptureModule.get_units(*capture_modules)
        with e7awgsw.CaptureCtrl(ipaddr) as cap_ctrl:
            # 初期化
            cap_ctrl.initialize(*capture_units)
            # キャプチャパラメータの設定
            cls.set_capture_params(cap_ctrl, num_capture_words, capture_units)
            # キャプチャスタート
            cap_ctrl.start_capture_units(*capture_units)
            # キャプチャ完了待ち
            cap_ctrl.wait_for_capture_units_to_stop(5, *capture_units)
            # エラーチェック
            cls.check_err(cap_ctrl, capture_units)
            # キャプチャデータ取得
            capture_unit_to_capture_data = cls.get_capture_data(cap_ctrl, capture_units)
        return capture_unit_to_capture_data
            # 波形保存
            #save_sample_data('capture', CaptureCtrl.SAMPLING_RATE, capture_unit_to_capture_data)
            #print('end')

        

        
# -------------------- qubelsi


def set_lmx2594_freq_100M(lmx2594, n): # equivalent to qubelsi.lmx2594.write_freq_100M
    lmx2594.write_value(0x24, n)
    return n

def apply_lmx2594(o): # fixed ?
    o.write_value(0x00, 0x6418)
    return True

# needs pull request to qubelsi
    
def apply_vatt(ad5328):
    ad5328.write_value(0xA, 0x002)
    return True

def set_vatt(vatt, v, apply=True): # max 4095
    vatt.lsi.write_value(vatt.ch, v)
    if apply:
        apply_vatt(vatt.lsi)
    return v/0xfff*3.3

def read_dac_nco(ad9082, ch):
    return None

def read_adc_nco(ad9082, ch):
    return None

# not so important

def set_lmx2594_OUTA_PD(o, b):
    if b:
        v = o.read_value(44) & 0b1111111110111111
    else:
        v = o.read_value(44) | 0b0000000001000000
    o.write_value(44, v)
    return v

def set_lmx2594_OUTB_PD(o, b):
    if b:
        v = o.read_value(44) & 0b1111111101111111
    else:
        v = o.read_value(44) | 0b0000000010000000
    o.write_value(44, v)
    return v

def set_lmx2594_OUTA_PWR(o, n): # 0 - 63
    v = o.read_value(44) & 0b1100000011111111 | n * 0x100
    o.write_value(44, v)
    return v

def set_lmx2594_OUTB_PWR(o, n): # 0 - 63
    v = o.read_value(45) & 0b1111111111000000 | n
    o.write_value(45, v)
    return v
