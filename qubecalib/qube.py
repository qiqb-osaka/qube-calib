from . import alias

import os
import math
import subprocess
import yaml
from collections import namedtuple
from enum import Enum, IntEnum, auto
from e7awgsw import AWG, CaptureModule
import e7awgsw


class Lane(IntEnum):
    L0 = 0
    L1 = 1
    L2 = 2
    
    
class PortFunc(IntEnum):
    Readout0 = 0
    Readin0 = 1
    Pump0 = 2
    Ctrl0 = 5
    Ctrl1 = 6
    Ctrl2 = 7
    Ctrl3 = 8
    Pump1 = 11
    Readin1 = 12
    Readout1 = 13
    
    
class PortNo(IntEnum):
    P0 = 0
    P1 = 1
    P2 = 2
    P5 = 5
    P6 = 6
    P7 = 7
    P8 = 8
    P11 = 11
    P12 = 12
    P13 = 13
    
    
class Qube(alias.Qube):
    PATH_TO_BITFILE = '/home/qube/bin'
    PATH_TO_API = './adi_api_mod'
    PATH_TO_CONFIG = './.config'
    
    def __init__(self, addr=None, path=None, config_file_name=None):
        super().__init__(addr, path)
        self._port = None
        self.config = dict()
        if config_file_name is not None:
            self.load_config(config_file_name)
            
    def load_config(self,config_file_name):
        
        name = '{}/{}'.format(self.PATH_TO_CONFIG, config_file_name)
        with open(name, 'rb') as f:
            self.config = o = yaml.safe_load(f)
            
        self.prepare(o['iplsi'], self.PATH_TO_API)
        
    def save(self, state_file_name):
        """qubelsi達の設定を保存する"""
        pass
    
    def load(self, state_file_name):
        """qubelsi達の設定を復旧する"""
        pass
    
    def config_fpga(self, bitfile=None, message_out=False):
        
        if bitfile is None and not 'bitfile' in self.config:
            raise ValueError('Specify bitfile.')
        
        if bitfile is None:
            bitfile = self.config['bitfile']
        
        os.environ['BITFILE'] = '{}/{}'.format(self.PATH_TO_BITFILE, bitfile)
        commands = ["vivado", "-mode", "batch", "-source", "{}/utils/config.tcl".format(self.PATH_TO_API)]
        ret = subprocess.check_output(commands , encoding='utf-8')
        return ret
    
    def __getitem__(self, v):
        return self.port[v]
    
    @property
    def port(self):
        
        if self._port is None:
            self._prepare_port()
        return self._port
    
    def _prepare_port(self):
        
        try:
            self._qube
        except AttributeError:
            raise ValueError('Exec prepare method first.')
        
        ADC, DAC = AD9082ADC, AD9082DAC
        Rin, Rout = Readin, Readout
        Vatt = UpConv.Vatt
        ipfpga = self.config['ipfpga']
        ad9082 = self.ad9082
        lmx = self.lmx2594
        adrf = self.adrf6780
        ad5328 = self.ad5328
        CapM = CaptureModule
        
        self._port = {
            PortNo.P0: Readout(
                LO(lmx[0]),
                DAC(ipfpga, ad9082[0], 0, [AWG.U15,]),
                UpConv(adrf[0], Vatt(ad5328, 0))
            ),
            PortNo.P1: Readin(
                LO(lmx[0]),
                ADC(ipfpga, ad9082[0], 3, [CapM.U1,])
            ),
            PortNo.P5: Ctrl(
                LO(lmx[2]),
                DAC(ipfpga, ad9082[0], 2, [AWG.U11, AWG.U12, AWG.U13,]),
                UpConv(adrf[2], Vatt(ad5328, 2))
            ),
            PortNo.P6: Ctrl(
                LO(lmx[3]),
                DAC(ipfpga, ad9082[0], 3, [AWG.U8, AWG.U9, AWG.U10,]),
                UpConv(adrf[3], Vatt(ad5328, 3))
            ),
            PortNo.P7: Ctrl(
                LO(lmx[4]),
                DAC(ipfpga, ad9082[1], 0, [AWG.U5, AWG.U6, AWG.U7,]),
                UpConv(adrf[4], Vatt(ad5328, 4))
            ),
            PortNo.P8: Ctrl(
                LO(lmx[5]),
                DAC(ipfpga, ad9082[1], 1, [AWG.U0, AWG.U3, AWG.U4,]),
                UpConv(adrf[5], Vatt(ad5328, 5))
            ),
            PortNo.P12: Readin(
                LO(lmx[7]),
                ADC(ipfpga, ad9082[1], 3, [CapM.U0,])
            ),
            PortNo.P13: Readout(
                LO(lmx[7]),
                DAC(ipfpga, ad9082[1], 3, [AWG.U2,]),
                UpConv(adrf[7], Vatt(ad5328, 7))
            ),
        }
    
    @property
    def awg(self):
        l2d = lambda l: {k: v for k, v in zip([Lane.L0, Lane.L1, Lane.L2],l)}
        r = {}
        for k, v in self.port.items():
            if isinstance(v, Readout):
                r[k] = v.dac.awgs[0]
            elif isinstance(v, Ctrl):
                r[k] = l2d(v.dac.awgs)
        return r

class FunctionBlock(object):
    pass
    
class LO(FunctionBlock):
    def __init__(self, lsi):
        self.lsi = lsi
    @property
    def freq(self):
        return self.lsi.read_freq_100M() * 100
    @freq.setter
    def freq(self, mhz):
        v = math.floor(mhz / 100)
        self.lsi.write_freq_100M(v)

class UpConv(FunctionBlock):
    Vatt = namedtuple('Vatt', ('lsi', 'ch'))
    def __init__(self, lsi, vatt):
        self.lsi = lsi
        self._vatt = vatt
        self._vatt_value = 0x800
        
    @property
    def mode(self):
        return ConvMode.USB if self.lsi.read_mode() == 0 else ConvMode.LSB
    @mode.setter
    def mode(self, v):
        if v == ConvMode.USB:
            self.lsi.set_usb()
        elif v == ConvMode.LSB:
            self.lsi.set_lsb()
    
    @property
    def vatt(self):
        # raise ValueError('This feature has not yet been implemented.')
        # return None
        
        return self._vatt_value
    
    @vatt.setter
    def vatt(self, v):
        lsi, ch = self._vatt.lsi, self._vatt.ch
        lsi.write_value(ch, v)
        lsi.write_value(0xA, 0x002) # apply value
        return v / 0xfff * 3.3 # volt

class NCO(object):
    def __init__(self, lsi, ch):
        self.lsi = lsi
        self.ch = ch
        self._freq = 1000 # 下位の API で決め打ち

class NCODAC(NCO):
    def __init__(self, lsi, ch):
        super().__init__(lsi, ch)
    @property
    def freq(self):
        return self._freq # MHz
    @freq.setter
    def freq(self, mhz):
        self._freq = mhz
        self.lsi.set_nco(freq=mhz*1e+6, ch=self.ch, adc_mode=False)
        
class NCOADC(NCO):
    def __init__(self, lsi, ch):
        super().__init__(lsi, ch)
    @property
    def freq(self):
        return self._freq # MHz
    @freq.setter
    def freq(self, mhz):
        self._freq = mhz
        self.lsi.set_nco(freq=mhz*1e+6, ch=self.ch, adc_mode=True)
        
class AD9082(object):
    def __init__(self, ipfpga, lsi):
        self.ipfpga = ipfpga
        self.lsi = lsi
        
class AD9082DAC(AD9082):
    def __init__(self, ipfpga, lsi, ch, awgs):
        super().__init__(ipfpga, lsi)
        self.awgs = awgs
        self.nco = NCODAC(lsi, ch)
        
class AD9082ADC(AD9082):
    def __init__(self, ipfpga, lsi, ch, caps):
        super().__init__(ipfpga, lsi)
        self.caps = caps
        self.nco = NCOADC(lsi, ch)
        

class Port(object):
    pass


class Output(Port):
    
    def __init__(self, local, dac, upconv):
        self.local = local
        self.dac = dac
        self.upconv = upconv
        self.active = False # TODO: 実際の動作状況を確認するようにしたい
    
    def set_freq(self, rf_mhz, lo_mhz):
        
        MAGIC_FREQ = 15.625
        trunc = lambda mhz: math.floor(mhz // MAGIC_FREQ) * MAGIC_FREQ
        
        if lo_mhz % 500 != 0:
            raise ValueError('lo_mhz must be a multiple of 500.')
        
        mode = self.upconv.mode
        if mode == ConvMode.LSB:
            nco_mhz = trunc(lo_mhz - rf_mhz)
            if nco_mhz == lo_mhz - rf_mhz:
                nco_mhz += MAGIC_FREQ
            awg_mhz = lo_mhz - nco_mhz - rf_mhz # rf_mhz = lo_mhz - (nco_mhz + awg_mhz)
            
        elif mode == ConvMode.USB:
            nco_mhz = trunc(rf_mhz - lo_mhz)
            if nco_mhz == rf_mhz - lo_mhz:
                nco_mhz -= MAGIC_FREQ
            awg_mhz = rf_mhz - lo_mhz - nco_mhz # rf_mhz = lo_mhz + (nco_mhz + awg_mhz)
        
        self.local.freq = lo_mhz
        self.dac.nco.freq = nco_mhz
        
        return nco_mhz, awg_mhz
    
    @property
    def status(self):
        fl, fi = self.local.freq, self.dac.nco.freq
        m = self.upconv.mode
        rf_mhz = fl + fi if m == ConvMode.USB else fl - fi
        r = ''
        r += 'RF = {:>5.3f} MHz '.format(rf_mhz)
        r += 'LO = {:>5.0f}    MHz '.format(int(fl))
        r += 'IF = {:>5.3f} MHz '.format(fi)
        r += 'LSB/USB: {} MODE '.format('USB' if m == ConvMode.USB else 'LSB')
        r += 'AWG: {}'.format('Active' if self.active else 'Inactive')
        return r
    
    @property
    def vatt(self):
        
        return self.upconv.vatt
    
    @vatt.setter
    def vatt(self, v):
        
        self.upconv.vatt = v
    
class ConvMode(IntEnum):
    LSB = 0
    USB = 1
    
class Input(Port):
    
    def __init__(self, local, adc):
        self.local = local
        self.adc = adc
    
    @property
    def status(self):
        fl, fi = self.local.freq, self.adc.nco.freq
        r = ''
        r += 'RF = {:>5.3f} MHz '.format(fl + fi) # assume USB mode
        r += 'LO = {:>5.0f}    MHz '.format(int(fl))
        r += 'IF = {:>5.3f} MHz '.format(fi)
        return r

    def set_freq(self, rf_mhz, lo_mhz):
        
        MAGIC_FREQ = 15.625
        trunc = lambda mhz: math.floor(mhz // MAGIC_FREQ) * MAGIC_FREQ
        
        if lo_mhz % 500 != 0:
            raise ValueError('lo_mhz must be a multiple of 500.')
        
        mode = self.upconv.mode
        if mode == ConvMode.LSB:
            nco_mhz = trunc(lo_mhz - rf_mhz)
            if nco_mhz == lo_mhz - rf_mhz:
                nco_mhz += MAGIC_FREQ
            awg_mhz = lo_mhz - nco_mhz - rf_mhz # rf_mhz = lo_mhz - (nco_mhz + awg_mhz)
            
        elif mode == ConvMode.USB:
            nco_mhz = trunc(rf_mhz - lo_mhz)
            if nco_mhz == rf_mhz - lo_mhz:
                nco_mhz -= MAGIC_FREQ
            awg_mhz = rf_mhz - lo_mhz - nco_mhz # rf_mhz = lo_mhz + (nco_mhz + awg_mhz)
        
        self.local.freq = lo_mhz
        self.dac.nco.freq = nco_mhz
        
        return nco_mhz, awg_mhz


class Ctrl(Output):
    
    def __init__(self, local, dac, upconv):
        super().__init__(local, dac, upconv)
        self.upconv.mode = ConvMode.LSB


class Readout(Output):
    
    def __init__(self, local, dac, upconv):
        super().__init__(local, dac, upconv)
        self.upconv.mode = ConvMode.USB


class Readin(Input):
    
    def __init__(self, local, adc):
        super().__init__(local, adc)
        
        
        
# import os
# import math
# import subprocess
# import yaml
# import e7awgsw
# import qubelsi.qube
# from collections import namedtuple


# PATH_TO_BITFILE = '/home/qube/bin/'
# PATH_TO_API = './adi_api_mod'
# PATH_TO_CONFIG = './.config/'

# class Qube64(object):
#     def __init__(self, config_file_name=None):
#         pass

# class Qube16(object):
#     def __init__(self, config_file_name=None):
#         pass


# class AD9082(object):
#     def __init__(self, lsi, ch, ipfpga):
#         self.lsi = lsi
#         self.ch = ch
#         self.fnco = 1000
#         self.ipfpga = ipfpga
#     def get_fnco(self):
#         return self.fnco # MHz

# class AD9082DAC(AD9082):
#     def __init__(self, lsi, ch, ipfpga, awgs):
#         super().__init__(lsi, ch, ipfpga)
#         self.awgs = awgs
#     def set_fnco(self, mhz):
#         self.fnco = mhz # ここはlsi読み出しに変更したい
#         self.lsi.set_nco(freq=mhz*1e+6, ch=self.ch)

# class AD9082ADC(AD9082):
#     def __init__(self, lsi, ch, ipfpga, caps):
#         super().__init__(lsi, ch, ipfpga)
#         self.caps = caps
#     def set_fnco(self, mhz):
#         self.fnco = mhz # ここはlsi読み出しに変更したい
#         self.lsi.set_nco(freq=mhz*1e+6, ch=self.ch, adc_mode=True)

# # DAC
# # 1CB ~ 1D0 まで 48 bit
# # アドレス選択
# # 0x001B [3:0] DAC MASK

# # ADC
# # 0x0A05 ~ 0x0A0A まで 48 bit
# # 0x0018 [3:0] ADC MASK

# class OutputPort(object):
#     def __init__(self, local, dac, upconv):
#         self.local = local
#         self.dac = dac
#         self.upconv = upconv
#         self.active = False
#     def set_lo(self, mhz): # MHz
#         self.local.set_freq(mhz)
#     def set_if(self, mhz): # MHz
#         self.dac.set_fnco(mhz)
#     def set_usb(self):
#         self.upconv.lsi.set_usb()
#     def set_lsb(self):
#         self.upconv.lsi.set_lsb()
#     def get_status(self):
#         fl, fi = self.local.get_freq(), self.dac.fnco
#         is_usb = self.upconv.is_setUSB()
#         r = ''
#         r += 'RF = {:>5.3f} MHz '.format(fl + fi if is_usb else fl - fi)
#         r += 'LO = {:>5.0f}    MHz '.format(int(fl))
#         r += 'IF = {:>5.3f} MHz '.format(fi)
#         r += 'LSB/USB: {} MODE '.format('USB' if is_usb else 'LSB')
#         r += 'AWG: {}'.format('Active' if self.active else 'Inactive')
#         return r
    
# class CtrlPort(OutputPort):
#     def __init__(self, local, dac, upconv):
#         super().__init__(local, dac, upconv)
#         self.set_lsb()

# class ReadoutPort(OutputPort):
#     def __init__(self, local_osc, dac, up_conv):
#         super().__init__(local_osc, dac, up_conv)
#         self.set_usb()

# class ReadinPort(object):
#     def __init__(self, local, adc):
#         self.local = local
#         self.adc = adc
#         self.active = False
#     def set_lo(self, mhz): # MHz
#         self.local.set_freq(mhz)
#     def set_if(self, mhz): # MHz
#         self.adc.set_fnco(mhz)
#     def get_status(self):
#         fl, fi = self.local.get_freq(), self.adc.fnco
#         r = ''
#         r += 'RF = {:>5.3f} MHz '.format(fl + fi)
#         r += 'LO = {:>5.0f}    MHz '.format(int(fl))
#         r += 'IF = {:>5.3f} MHz '.format(fi)
#         return r
        
        
# class UpConverter(object):
#     Vatt = namedtuple('Vatt', ('lsi', 'ch'))
#     def __init__(self, lsi, vatt):
#         self.lsi = lsi
#         self.vatt = vatt
#     def is_setUSB(self):
#         return True if self.lsi.read_mode() == 0 else False

# class LocalOscillator(object):
#     def __init__(self, lsi):
#         self.lsi = lsi
#     def set_freq(self, mhz): # 100 MHz or less truncated
#         v = math.floor(mhz/100)
#         self.lsi.write_freq_100M(v)
#     def get_freq(self):
#         return self.lsi.read_freq_100M() * 100




        

        
# # -------------------- qubelsi


# def set_lmx2594_freq_100M(lmx2594, n): # equivalent to qubelsi.lmx2594.write_freq_100M
#     lmx2594.write_value(0x24, n)
#     return n

# def apply_lmx2594(o): # fixed ?
#     o.write_value(0x00, 0x6418)
#     return True

# # needs pull request to qubelsi
    
# def apply_vatt(ad5328):
#     ad5328.write_value(0xA, 0x002)
#     return True

# def set_vatt(vatt, v, apply=True): # max 4095
#     vatt.lsi.write_value(vatt.ch, v)
#     if apply:
#         apply_vatt(vatt.lsi)
#     return v/0xfff*3.3

# def read_dac_nco(ad9082, ch):
#     return None

# def read_adc_nco(ad9082, ch):
#     return None

# # not so important

# def set_lmx2594_OUTA_PD(o, b):
#     if b:
#         v = o.read_value(44) & 0b1111111110111111
#     else:
#         v = o.read_value(44) | 0b0000000001000000
#     o.write_value(44, v)
#     return v

# def set_lmx2594_OUTB_PD(o, b):
#     if b:
#         v = o.read_value(44) & 0b1111111101111111
#     else:
#         v = o.read_value(44) | 0b0000000010000000
#     o.write_value(44, v)
#     return v

# def set_lmx2594_OUTA_PWR(o, n): # 0 - 63
#     v = o.read_value(44) & 0b1100000011111111 | n * 0x100
#     o.write_value(44, v)
#     return v

# def set_lmx2594_OUTB_PWR(o, n): # 0 - 63
#     v = o.read_value(45) & 0b1111111111000000 | n
#     o.write_value(45, v)
#     return v
