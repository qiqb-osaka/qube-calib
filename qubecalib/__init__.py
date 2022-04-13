'''Calibration package for QuBE'''
import os
import subprocess
import yaml
import e7awgsw
import qubelsi.qube
from collections import namedtuple

PATH_TO_BITFILE = '/home/qube/bin/'
PATH_TO_API = './adi_api_mod'

class Qube64(object):
    def __init__(self, config_file_name=None):
        pass

class Qube16(object):
    def __init__(self, config_file_name=None):
        pass

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
        else:
            o.do_init(message_out=message_out)
    def config_fpga(self, bitfile):
        os.environ['BITFILE'] = bitfile
        print("init FPGA")
        commands = ["vivado", "-mode", "batch", "-source", "{}/utils/config.tcl".format(PATH_TO_BITFILE)]
        ret = subprocess.check_output(commands , encoding='utf-8')
        print(ret)
    def restart_ad9082(self, message_out=True):
        o = self.qube.ad9082
        for c in o:
            c.do_init(message_out=True)
    def get_status(self):
        r = ''
        return r
    def get_port_status(self):
        r = ''
        for p in self.ports:
            r + str(p)
        return r

class AD9082(object):
    def __init__(self, ad9082, ch):
        self.ad9082 = ad9082
        self.ch = ch
        self.fnco = 0
    def get_fnco(self):
        return self.fnco # MHz

class AD9082DAC(AD9082):
    def __init__(self, ad9082, ch, awgs):
        super().__init__(self, ad9082, ch)
        self.awgs = awgs
    def set_fnco(self, mhz):
        self.fnco = mhz
        self.ad9082.set_nco(freq=mhz*1e+6, ch=self.ch)

class AD9082ADC(AD9082):
    def __init__(self, ad9082, ch, caps):
        super().__init__(self, ad9082, ch)
        self.caps = caps
    def set_fnco(self, mhz):
        self.fnco = mhz
        self.ad9082.set_nco(freq=mhz*1e+6, ch=self.ch, adc_mode=True)

# DAC
# 1CB ~ 1D0 まで 48 bit
# アドレス選択
# 0x001B [3:0] DAC MASK

# ADC
# 0x0A05 ~ 0x0A0A まで 48 bit
# 0x0018 [3:0] ADC MASK
        
class OutputPort(object):
    def __init__(self, local_osc, dac, up_conv):
        self.local_osc = local_osc
        self.dac = dac
        self.awg_module = awg_modules
        self.upconv = up_conv
    def set_lo(self, frequency): # MHz
        pass
    def set_if(self, frequency): # MHz
        self.fim = frequency
    def set_usb(self):
        pass
    def set_lsb(self):
        pass

class CtrlPort(OutputPort):
    def __init__(self, local_osc, dac, up_conv):
        super().__init__(self, local_osc, dac, up_conv)
        self.set_lsb()
    def get_status(self):
        fl, fi = self.local_osc.read_freq_100M()*100, self.fim
        isusb = self.upconv.isUSB()
        r = ''
        r += 'RF = {:>5.3f} MHz '.format(fl + fi if isusb else fl - fi)
        r += 'LO = {:>5.0f}    MHz '.format(int(fl)*100)
        r += 'IF = {:>5.3f} MHz '.format(fi)
        r += 'LSB/USB: {} MODE'.format('USB' if isusb else 'LSB')
        return r
        
class ReadoutPort(OutputPort):
    def __init__(self, local_osc, if_nco, awg_modules, up_conv):
        super().__init__(self, local_osc, if_nco, awg_modules, up_conv)
        self.set_usb()
        
class Upconv(object):
    Vatt = namedtuple('Vatt', ('dac', 'ch'))
    def __init__(self, upconv, vatt):
        self.upconv = upconv
        self.vatt = vatt # Upconv.Vatt(dac, ch)
    def isUSB(self):
        return True if self.upconv.read_mode() == 0 else False
    
class LongSend(object):
    def __init__(self, port, freq, att):
        pass
    
    
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
    vatt.dac.write_value(vatt.ch, v)
    if apply:
        apply_vatt(vatt.dac)
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
