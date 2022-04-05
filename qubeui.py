import ipywidgets as ipw
import long_send
from collections import namedtuple
from IPython.display import display

Vatt = namedtuple('Vatt', ('dac', 'ch'))
Upconv = namedtuple('Upconv', ('vatt',))
Ifdac = namedtuple('Ifdac', ('ad9082', 'ch'))
CtrlPort = namedtuple('CtrlPort', ('losc', 'ifdac', 'upconv'))

def new_port_handler(qube):
    return {
        5 : CtrlPort(qube.lmx2594[2], Ifdac(qube.ad9082[0], 2), Upconv(Vatt(qube.ad5328, 2))), # CTRL1
        6 : CtrlPort(qube.lmx2594[3], Ifdac(qube.ad9082[0], 3), Upconv(Vatt(qube.ad5328, 3))), # CTRL2
        7 : CtrlPort(qube.lmx2594[4], Ifdac(qube.ad9082[1], 0), Upconv(Vatt(qube.ad5328, 4))), # CTRL3
        8 : CtrlPort(qube.lmx2594[5], Ifdac(qube.ad9082[1], 1), Upconv(Vatt(qube.ad5328, 5))), # CTRL4
    }

def apply_lmx2594(o):
    o.write_value(0x00, 0x6418)
    return True
    
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

def set_lmx2594_freq_100M(lmx2594, n):
    lmx2594.write_value(0x24, n)
    return n

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
    
def apply_vatt(ad5328):
    ad5328.write_value(0xA, 0x002)
    return True

def set_vatt(vatt, v, apply=True): # max 4095
    vatt.dac.write_value(vatt.ch, v)
    if apply:
        apply_vatt(vatt.dac)
    return v/0xfff*3.3

class StartupUI(object):
    def __init__(self, qube):
        self.qube = qube
        self.cbx_config_fpga = cb = ipw.Checkbox(value=False, description='Config FPGA', disabled=False, indent=False)
        self.txt_path_bitfile = bitfile = ipw.Text(
            description='Path to bitfile',
            value='/home/qube/bin/069414.bit',
            style={'description_width': 'initial'},
            #layout=ipw.Layout(width='50%'),
            disabled=False)
        self.btn_do_init = btn = ipw.Button(description='Do init', layout={'width': '50%', 'height': '80px'})
        btn.on_click(self.do_init)
        self.btn_ad9082 = ad9082 = ipw.Button(description='Restart AD9082', layout={'width': '50%', 'height': '80px'})
        ad9082.on_click(self.do_init_ad9082)
        display(
            ipw.Text(description='IP Address for eXtickGE', value=qube.gpio.handle.addr, style={'description_width': 'initial'}, disabled=True),
            ipw.Text(description='Path to API', value=qube.path, style={'description_width': 'initial'}, layout=ipw.Layout(width='50%'), disabled=True),
            ipw.HBox([cb, bitfile], layout={'width': '50%'}),
            btn,
            ad9082,
        )
    def do_init(self, e):
        if self.cbx_config_fpga.value:
            self.qube.do_init(bitfile=self.txt_path_bitfile.value, message_out=True)
        else:
            self.qube.do_init(message_out=True)
    def do_init_ad9082(self, e):
        for c in self.qube.ad9082:
            c.do_init(message_out=True)
            
class AwgPanel(object):
    class MultipleText(list):
        def __init__(self, *args):
            self.textbox = [ipw.Text(value=v) for v in args]
    def __init__(self, qube):
        self.qube = qube
        self.txt_freqs = tf = {
            'port5': AwgPanel.MultipleText('2.5', '2.5', '2.5'),
            'port6': AwgPanel.MultipleText('2.5', '2.5', '2.5'),
            'port7': AwgPanel.MultipleText('2.5', '2.5', '2.5'),
            'port8': AwgPanel.MultipleText('2.5', '2.5', '2.5'),
        }
        self.txt_atts = ta = {
            'port5': AwgPanel.MultipleText('0', '0', '0'),
            'port6': AwgPanel.MultipleText('0', '0', '0'),
            'port7': AwgPanel.MultipleText('0', '0', '0'),
            'port8': AwgPanel.MultipleText('0', '0', '0'),
        }
        for i in range(5,9):
            tf['port{}'.format(i)].textbox[0].description = 'Freq{} [MHz]:'.format(i)
        for i in range(5,9):
            ta['port{}'.format(i)].textbox[0].description = 'Att{} [dB]:'.format(i)
        btn = ipw.Button(description='Long Send')#, layout={'width': '50%'})
        btn.on_click(self.do_longsend)
        self.panel = ipw.VBox([
            ipw.HBox(tf['port5'].textbox),
            ipw.HBox(tf['port6'].textbox),
            ipw.HBox(tf['port7'].textbox),
            ipw.HBox(tf['port8'].textbox),
            ipw.HBox(ta['port5'].textbox),
            ipw.HBox(ta['port6'].textbox),
            ipw.HBox(ta['port7'].textbox),
            ipw.HBox(ta['port8'].textbox),
            btn,
        ])
    def display(self):
        display(self.panel)
    def get_attenuation_list(self):
        r = [float(self.txt_atts['port8'].textbox[0].value), 0, 0] +\
        [float(o.value) for o in self.txt_atts['port8'].textbox[1:]] +\
        [float(o.value) for o in self.txt_atts['port7'].textbox] +\
        [float(o.value) for o in self.txt_atts['port6'].textbox] +\
        [float(o.value) for o in self.txt_atts['port5'].textbox] +\
        [0, 0]
        return r
    def get_frequency_list(self):
        r = [float(self.txt_freqs['port8'].textbox[0].value), 0, 0] +\
        [float(o.value) for o in self.txt_freqs['port8'].textbox[1:]] +\
        [float(o.value) for o in self.txt_freqs['port7'].textbox] +\
        [float(o.value) for o in self.txt_freqs['port6'].textbox] +\
        [float(o.value) for o in self.txt_freqs['port5'].textbox] +\
        [0, 0]
        return r
    def get_amplitude_list(self):
        a = [
            10922, 32766, 32766,
            10922, 10922,
            10922, 10922,
            10922, 10922,
            10922, 10922,
            32766,
            32766,
            ]
        r = self.get_attenuation_list()
        return [b * 10**(-s/20)  for b, s in zip(a,r)]
    def do_longsend(self, e):
        ip = self.qube.gpio.handle.addr
        a = self.get_amplitude_list()
        f = self.get_frequency_list()
        long_send.stop(ipaddr='10.1.0.5')
        long_send.start(a, f, ipaddr='10.1.0.5')
