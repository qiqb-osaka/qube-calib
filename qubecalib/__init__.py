'''Calibration package for QuBE'''
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

<<<<<<< HEAD
# 極力 qubelsi.qube.Qube の API と揃える
class Qube(object):
    def __init__(self, addr=None, path=None, *args, **kwargs):
        self.config_file_name = None
        self.config = {}
        self.qube = None if addr == None or path == None else qubelsi.qube.Qube(addr, path)
        
    def load(self, config_file_name=None):
        if config_file_name == None:
            config_file_name = self.config_file_name
=======
class Qube3(object):
    def __init__(self):
        pass
    
class Qube(object):
    Vatt = namedtuple('Vatt', ('dac', 'ch'))
    Upconv = namedtuple('Upconv', ('vatt',))
    Ifdac = namedtuple('Ifdac', ('ad9082', 'ch'))
    CtrlPort = namedtuple('CtrlPort', ('losc', 'ifdac', 'upconv'))
    PumpPort = namedtuple('CtrlPort', ('losc', 'ifdac', 'upconv'))
    ReadoutPort = namedtuple('ReadoutPort', ('losc', 'ifdac', 'upconv'))
    ReadinPort = namedtuple('ReadinPort', ('losc'))
    def __init__(self, config_file_name=None):
        self.qube = None
        self.path_bitfile = '/home/qube/bin/'
        if config_file_name is not None:
            self.load(config_file_name)
    def load(self, config_file_name):
        self.config_file_name = config_file_name
        with open('./.config/{}'.format(config_file_name)) as f:
            self.config = o = yaml.safe_load(f)
        self.qube = qubelsi.qube.Qube(
            o['iplsi'], # IP address of eXtickGE
            './adi_api_mod', # Path to API
        )
        self.ports = self._new_port_handler()
    def _new_port_handler(self):
        qube = self.qube
        lmx2594, ad9082, ad5328 = qube.lmx2594, qube.ad9082, qube.ad5328
        ReadoutPort = QubeUnit.ReadoutPort
        ReadinPort = QubeUnit.ReadinPort
        CtrlPort = QubeUnit.CtrlPort
        PumpPort = QubeUnit.PumpPort
        Ifdac = QubeUnit.Ifdac
        Upconv = QubeUnit.Upconv
        Vatt = QubeUnit.Vatt
        return {
            0 : ReadoutPort(lmx2594[0], Ifdac(ad9082[0], 0), Upconv(Vatt(ad5328, 0))),
            1 : ReadinPort(lmx2594[0],),
            2 : PumpPort(lmx2594[1], Ifdac(ad9082[0], 1), Upconv(Vatt(ad5328, 1))),
            5 : CtrlPort(lmx2594[2], Ifdac(ad9082[0], 2), Upconv(Vatt(ad5328, 2))), # CTRL1
            6 : CtrlPort(lmx2594[3], Ifdac(ad9082[0], 3), Upconv(Vatt(ad5328, 3))), # CTRL2
            7 : CtrlPort(lmx2594[4], Ifdac(ad9082[1], 0), Upconv(Vatt(ad5328, 4))), # CTRL3
            8 : CtrlPort(lmx2594[5], Ifdac(ad9082[1], 1), Upconv(Vatt(ad5328, 5))), # CTRL4
            11 : PumpPort(lmx2594[6], Ifdac(ad9082[1], 2), Upconv(Vatt(ad5328, 6))),
            12 : ReadinPort(lmx2594[7],),
            13 : ReadoutPort(lmx2594[7], Ifdac(ad9082[1], 3), Upconv(Vatt(ad5328, 7))),
        }
    def do_init(self, config_fpga=False, message_out=True):
        o = self.qube
        if config_fpga:
            o.do_init(bitfile=self.path_bitfile + self.config['bitfile'], message_out=message_out)
>>>>>>> 5b5a326 (small change)
        else:
            self.config_file_name = config_file_name
        
        fname = PATH_TO_CONFIG + config_file_name
        with open(fname, 'rb') as f:
            self.config = o = yaml.safe_load(f)
        print(o)
            
        self.qube = qubelsi.qube.Qube(o['iplsi'], PATH_TO_API)
        self._ports = self._new_ports()
        
    def do_init(self, rf_type='A', bitfile=None, message_out=False):
        self.qube.do_init(rf_type, bitfile, message_out)
        
    def config_fpga(self, bitfile, message_out=False):
        os.environ['BITFILE'] = bitfile
        print("init FPGA")
        commands = ["vivado", "-mode", "batch", "-source", "{}/utils/config.tcl".format(PATH_TO_BITFILE)]
        ret = subprocess.check_output(commands , encoding='utf-8')
        print(ret)
        
    def restart_ad9082(self, message_out=True):
        for o in self.ad9082:
            c.do_init(message_out=True)
            
    def _new_ports(self):
        CP, LO, DAC, UC = CtrlPort, LocalOscillator, AD9082DAC, UpConverter
        ADC = AD9082ADC
        RO = ReadoutPort
        RI = ReadinPort
        o = self.qube
        AWG = e7awgsw.AWG
        CaptM = e7awgsw.CaptureModule
        print(o, AWG, CaptM)
        return {
            0 : RO(LO(o.lmx2594[0]), DAC(o.ad9082[0], 0, self.config['ipfpga'], [AWG.U15,]), UC(o.adrf6780[0], UC.Vatt(o.ad5328, 0))), # Readout1
            1 : RI(LO(o.lmx2594[0]), ADC(o.ad9082[0], 3, self.config['ipfpga'], [CaptM.U1,])), # Readin1
            5 : CP(LO(o.lmx2594[2]), DAC(o.ad9082[0], 2, self.config['ipfpga'], [AWG.U11, AWG.U12, AWG.U13,]), UC(o.adrf6780[2], UC.Vatt(o.ad5328, 2))), # CTRL1
            6 : CP(LO(o.lmx2594[3]), DAC(o.ad9082[0], 3, self.config['ipfpga'], [AWG.U8, AWG.U9, AWG.U10,]), UC(o.adrf6780[3], UC.Vatt(o.ad5328, 3))), # CTRL2
            7 : CP(LO(o.lmx2594[4]), DAC(o.ad9082[1], 0, self.config['ipfpga'], [AWG.U5, AWG.U6, AWG.U7,]), UC(o.adrf6780[4], UC.Vatt(o.ad5328, 4))), # CTRL3
            8 : CP(LO(o.lmx2594[5]), DAC(o.ad9082[1], 1, self.config['ipfpga'], [AWG.U0, AWG.U3, AWG.U4,]), UC(o.adrf6780[5], UC.Vatt(o.ad5328, 5))), # CTRL4
            12 : RI(LO(o.lmx2594[7]), ADC(o.ad9082[1], 3, self.config['ipfpga'], [CaptM.U0,])), # Readin2
            13 : RO(LO(o.lmx2594[7]), DAC(o.ad9082[1], 3, self.config['ipfpga'], [AWG.U2,]), UC(o.adrf6780[7], UC.Vatt(o.ad5328, 7))), # Readout2
        }
    @property
    def ports(self):
        return self._ports
    @property
    def path(self):
        return self.qube.path
    @path.setter
    def path(self, v):
        self.qube.path = v
    @property
    def ad9082(self):
        return self.qube.ad9082
    @ad9082.setter
    def ad9082(self, v):
        self.qube.ad9082 = v
    @property
    def lmx2594(self):
        return self.qube.lmx2594
    @lmx2594.setter
    def lmx2594(self, v):
        self.qube.lmx2594 = v
    @property
    def lmx2594_ad9082(self):
        return self.qube.lmx2594_ad9082
    @lmx2594_ad9082.setter
    def lmx2594_ad9082(self, v):
        self.qube.lmx2594_ad9082 = v
    @property
    def adrf6780(self):
        return self.qube.adrf6780
    @adrf6780.setter
    def adrf6780(self, v):
        self.qube.adrf6780 = v
    @property
    def ad5328(self):
        return self.qube.ad5328
    @ad5328.setter
    def ad5328(self, v):
        self.qube.ad5328 = v
    @property
    def gpio(self):
        return self.qube.gpio
    @gpio.setter
    def gpio(self, v):
        self.qube.gpio = v
    @property
    def bitfile(self):
        return self.qube.bitfile
    @bitfile.setter
    def bitfile(self, v):
        self.qube.bitfile = v
    @property
    def rf_type(self):
        return self.qube.rf_type
    @rf_type.setter
    def rf_type(self, v):
        self.qube.rf_type = v

    
# class QubeUnit(object):
#     Vatt = namedtuple('Vatt', ('dac', 'ch'))
#     Upconv = namedtuple('Upconv', ('vatt',))
#     Ifdac = namedtuple('Ifdac', ('ad9082', 'ch'))
#     CtrlPort = namedtuple('CtrlPort', ('losc', 'ifdac', 'upconv'))
#     PumpPort = namedtuple('CtrlPort', ('losc', 'ifdac', 'upconv'))
#     ReadoutPort = namedtuple('ReadoutPort', ('losc', 'ifdac', 'upconv'))
#     ReadinPort = namedtuple('ReadinPort', ('losc'))
#     def __init__(self, config_file_name=None):
#         self.qube = None
#         self.path_bitfile = '/home/qube/bin/'
#         if config_file_name is not None:
#             self.load(config_file_name)
#     def load(self, config_file_name):
#         self.config_file_name = config_file_name
#         with open('./.config/{}'.format(config_file_name)) as f:
#             self.config = o = yaml.safe_load(f)
#         self.qube = qubelsi.qube.Qube(
#             o['iplsi'], # IP address of eXtickGE
#             './adi_api_mod', # Path to API
#         )
#         self.ports = self._new_port_handler()
#     def _new_port_handler(self):
#         qube = self.qube
#         lmx2594, ad9082, ad5328 = qube.lmx2594, qube.ad9082, qube.ad5328
#         ReadoutPort = QubeUnit.ReadoutPort
#         ReadinPort = QubeUnit.ReadinPort
#         CtrlPort = QubeUnit.CtrlPort
#         PumpPort = QubeUnit.PumpPort
#         Ifdac = QubeUnit.Ifdac
#         Upconv = QubeUnit.Upconv
#         Vatt = QubeUnit.Vatt
#         return {
#             0 : ReadoutPort(lmx2594[0], Ifdac(ad9082[0], 0), Upconv(Vatt(ad5328, 0))),
#             1 : ReadinPort(lmx2594[0],),
#             2 : PumpPort(lmx2594[1], Ifdac(ad9082[0], 1), Upconv(Vatt(ad5328, 1))),
#             5 : CtrlPort(lmx2594[2], Ifdac(ad9082[0], 2), Upconv(Vatt(ad5328, 2))), # CTRL1
#             6 : CtrlPort(lmx2594[3], Ifdac(ad9082[0], 3), Upconv(Vatt(ad5328, 3))), # CTRL2
#             7 : CtrlPort(lmx2594[4], Ifdac(ad9082[1], 0), Upconv(Vatt(ad5328, 4))), # CTRL3
#             8 : CtrlPort(lmx2594[5], Ifdac(ad9082[1], 1), Upconv(Vatt(ad5328, 5))), # CTRL4
#             11 : PumpPort(lmx2594[6], Ifdac(ad9082[1], 2), Upconv(Vatt(ad5328, 6))),
#             12 : ReadinPort(lmx2594[7],),
#             13 : ReadoutPort(lmx2594[7], Ifdac(ad9082[1], 3), Upconv(Vatt(ad5328, 7))),
#         }
#     def do_init(self, config_fpga=False, message_out=True):
#         o = self.qube
#         if config_fpga:
#             o.do_init(bitfile=self.path_bitfile + self.config['bitfile'], message_out=message_out)
#         else:
#             o.do_init(message_out=message_out)
#     def config_fpga(self, bitfile):
#         os.environ['BITFILE'] = bitfile
#         print("init FPGA")
#         commands = ["vivado", "-mode", "batch", "-source", "{}/utils/config.tcl".format(PATH_TO_BITFILE)]
#         ret = subprocess.check_output(commands , encoding='utf-8')
#         print(ret)
#     def restart_ad9082(self, message_out=True):
#         o = self.qube.ad9082
#         for c in o:
#             c.do_init(message_out=True)
#     def get_status(self):
#         r = ''
#         return r
#     def get_port_status(self):
#         r = ''
#         for p in self.ports:
#             r + str(p)
#         return r

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
<<<<<<< HEAD
        self.upconv = upconv
        self.active = False
    def set_lo(self, mhz): # MHz
        self.local.set_freq(mhz)
    def set_if(self, mhz): # MHz
        self.dac.set_fnco(mhz)
=======
        self.awg_module = awg_modules
        self.upconv = up_conv
    def set_lo(self, frequency): # MHz
        pass
    def set_if(self, frequency): # MHz
        self.fim = frequency
>>>>>>> 5b5a326 (small change)
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
<<<<<<< HEAD
        fl, fi = self.local.get_freq(), self.adc.fnco
=======
        fl, fi = self.local_osc.read_freq_100M()*100, self.fim
        isusb = self.upconv.isUSB()
>>>>>>> 5b5a326 (small change)
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

# class DAC(object):
#     def __init__(self, lsi, ch):
#         self.lsi = lsi
#         self.ch = ch
#     def set_freq(self, mhz):
#         self.freq = mhz
#         self.lsi.set_nco(freq=mhz*1e+6, ch=self.ch)
#     def get_freq(self):
#         return self.freq

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

        

        
# class Recv(object): pass
# class SendRecv(object): pass

# def set_flo_according_to_frf(lmx2594, frf, fif, apply=True): # MHz
#     flo = frf + fif
#     set_lmx2594_freq_100M(lmx2594, int(flo*1e-2))
#     if apply:
#         apply_lmx2594(lmx2594)
#     return flo

# def set_flo_fnco_according_to_frf(port, frf, fif, apply=True): #MHz
#     fnco = fif - 2.5
#     ifdac = port.ifdac
#     ifdac.ad9082.set_nco(ch=ifdac.ch, freq=fnco*1e+6)
#     flo = set_flo_according_to_frf(port.losc, frf, fif, apply=apply)
#     return flo, fnco


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
