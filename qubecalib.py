import yaml
import e7awgsw
import qubelsi.qube
from collections import namedtuple

# QUBE.load('qube_riken_1-01.yaml')
class Qube(object):
    def __init__(self, config_file_name=None):
        self.qube = None
        self.path_bitfile = '/home/qube/bin/'
        if config_file_name is not None:
            self.load(config_file_name)
    def load(self, config_file_name):
        with open('./.config/{}'.format(config_file_name)) as f:
            self.config_file_name = config_file_name
            self.config = o = yaml.safe_load(f)
        self.qube = qubelsi.qube.Qube(
            o['iplsi'], # IP address of eXtickGE
            './adi_api_mod', # Path to API
        )
    def do_init(self, config_fpga=False, message_out=True):
        o = self.qube
        if config_fpga:
            o.do_init(bitfile=self.path_bitfile + self.config['bitfile'], message_out=message_out)
        else:
            o.do_init(message_out=message_out)
    def restart_ad9082(self, message_out=True):
        o = self.qube.ad9082
        for c in o:
            c.do_init(message_out=True)


class LongSend(object): pass

class SendRecv(object): pass

Vatt = namedtuple('Vatt', ('dac', 'ch'))
Upconv = namedtuple('Upconv', ('vatt',))
Ifdac = namedtuple('Ifdac', ('ad9082', 'ch'))
CtrlPort = namedtuple('CtrlPort', ('losc', 'ifdac', 'upconv'))

# def new_port_handler(qube):
#     return {
#         5 : CtrlPort(qube.lmx2594[2], Ifdac(qube.ad9082[0], 2), Upconv(Vatt(qube.ad5328, 2))), # CTRL1
#         6 : CtrlPort(qube.lmx2594[3], Ifdac(qube.ad9082[0], 3), Upconv(Vatt(qube.ad5328, 3))), # CTRL2
#         7 : CtrlPort(qube.lmx2594[4], Ifdac(qube.ad9082[1], 0), Upconv(Vatt(qube.ad5328, 4))), # CTRL3
#         8 : CtrlPort(qube.lmx2594[5], Ifdac(qube.ad9082[1], 1), Upconv(Vatt(qube.ad5328, 5))), # CTRL4
#     }

def new_port_handler(qube):
    return {
        0 : CtrlPort(qube.lmx2594[0], Ifdac(qube.ad9082[0], 0), Upconv(Vatt(qube.ad5328, 0))), # Readout
        2 : CtrlPort(qube.lmx2594[1], Ifdac(qube.ad9082[0], 1), Upconv(Vatt(qube.ad5328, 1))), # Pump
        5 : CtrlPort(qube.lmx2594[2], Ifdac(qube.ad9082[0], 2), Upconv(Vatt(qube.ad5328, 2))), # CTRL1
        6 : CtrlPort(qube.lmx2594[3], Ifdac(qube.ad9082[0], 3), Upconv(Vatt(qube.ad5328, 3))), # CTRL2
        7 : CtrlPort(qube.lmx2594[4], Ifdac(qube.ad9082[1], 0), Upconv(Vatt(qube.ad5328, 4))), # CTRL3
        8 : CtrlPort(qube.lmx2594[5], Ifdac(qube.ad9082[1], 1), Upconv(Vatt(qube.ad5328, 5))), # CTRL4
        11 : CtrlPort(qube.lmx2594[6], Ifdac(qube.ad9082[1], 2), Upconv(Vatt(qube.ad5328, 6))), # Pump
        13 : CtrlPort(qube.lmx2594[7], Ifdac(qube.ad9082[1], 3), Upconv(Vatt(qube.ad5328, 7))), # Readout
    }

def set_flo_according_to_frf(lmx2594, frf, fif, apply=True): # MHz
    flo = frf + fif
    set_lmx2594_freq_100M(lmx2594, int(flo*1e-2))
    if apply:
        apply_lmx2594(lmx2594)
    return flo

def set_flo_fnco_according_to_frf(port, frf, fif, apply=True): #MHz
    fnco = fif - 2.5
    ifdac = port.ifdac
    ifdac.ad9082.set_nco(ch=ifdac.ch, freq=fnco*1e+6)
    flo = set_flo_according_to_frf(port.losc, frf, fif, apply=apply)
    return flo, fnco
    

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
