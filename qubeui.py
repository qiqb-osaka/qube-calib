import ipywidgets as ipw
import long_send
from collections import namedtuple
import IPython.display
import qubecalib

def display(*args, **kwargs):
    IPython.display.display(*args, **kwargs)

class Event(object):
    handler = {}
    def bind(self, e, func):
        b = self.handler
        if e in b:
            b[e].append(func) # append event listner
        else:
            b[e] = [func,] # add new event
    def invoke(self, e):
        b = self.handler
        if e in b:
            for func in b[e]:
                func(self)

class Qube(qubecalib.Qube):
    def __init__(self, *args, **kwargs):
        self.event = Event()
        qubecalib.Qube.__init__(self, *args, **kwargs)

class QubeLoadConfigPanel(ipw.VBox):
    def __init__(self, qube, *args, **kwargs):
        self.qube = qube
        self.tb_fname = t = ipw.Text(description='Config', value='qube_riken_1-01.yml')
        self.btn_load = b = ipw.Button(description='Load'); b.on_click(self.load)
        ipw.VBox.__init__(self, [ipw.HBox([t, b]),], *args, **kwargs)
    def load(self, e):
        self.qube.load(self.tb_fname.value)
        self.qube.event.invoke('loaded')

class QubeSetupPanel(ipw.VBox):
    def __init__(self, qube, *args, **kwargs):
        self.qube = qube
        self.tb_ip4lsi = tb_ip4lsi = ipw.Text(description='IP Address for eXtickGE', style={'description_width': 'initial'}, disabled=True)
        self.tb_path2api = tb_path2api = ipw.Text(description='Path to API', style={'description_width': 'initial'}, layout=ipw.Layout(width='50%'), disabled=True)
        self.cb_config_fpga = cb = ipw.Checkbox(value=False, description='Config FPGA', disabled=False, indent=False)
        self.tb_path_bitfile = bitfile = ipw.Text(
            description='Path to bitfile',
            value='',
            style={'description_width': 'initial'},
            disabled=False)
        self.btn_do_init = b = ipw.Button(description='Do init', layout={'width': '50%', 'height': '80px'}, disabled=True); b.on_click(self.do_init)
        self.btn_ad9082 = b = ipw.Button(description='Restart AD9082', layout={'width': '50%', 'height': '80px'}, disabled=True); b.on_click(self.do_init_ad9082)
        self.qube.event.bind('loaded', self.loaded)
        ipw.VBox.__init__(
            self,
            [
                tb_ip4lsi, tb_path2api,
                ipw.HBox([cb, bitfile], layout={'width': '50%'}),
                self.btn_do_init, self.btn_ad9082,
            ],
            *args,
            **kwargs
        )
    def loaded(self, e):
        o = self.qube
        self.tb_ip4lsi.value = o.qube.gpio.handle.addr
        self.tb_path2api.value = o.qube.path
        self.tb_path_bitfile.value = o.config['bitfile']
        self.btn_do_init.disabled = False
        self.btn_ad9082.disabled = False
    def do_init(self, e):
        o = self.qube
        if self.cb_config_fpga.value:
            o.do_init(config_fpga=True, message_out=True)
        else:
            o.do_init(message_out=True)
    def do_init_ad9082(self, e):
        self.qube.restart_ad9082(message_out=True)

class QubeLoControlPanel(ipw.VBox): pass

class QubeNcoControlPanel(ipw.VBox): pass

class QubeLongSendControlPanel(ipw.VBox): pass

# class AwgPanel(object):
#     class MultipleText(list):
#         def __init__(self, *args):
#             self.textbox = [ipw.Text(value=v) for v in args]
#     def __init__(self, qube):
#         self.qube = qube
#         self.txt_freqs = tf = {
#             'port5': AwgPanel.MultipleText('2.5', '2.5', '2.5'),
#             'port6': AwgPanel.MultipleText('2.5', '2.5', '2.5'),
#             'port7': AwgPanel.MultipleText('2.5', '2.5', '2.5'),
#             'port8': AwgPanel.MultipleText('2.5', '2.5', '2.5'),
#         }
#         self.txt_atts = ta = {
#             'port5': AwgPanel.MultipleText('0', '0', '0'),
#             'port6': AwgPanel.MultipleText('0', '0', '0'),
#             'port7': AwgPanel.MultipleText('0', '0', '0'),
#             'port8': AwgPanel.MultipleText('0', '0', '0'),
#         }
#         for i in range(5,9):
#             tf['port{}'.format(i)].textbox[0].description = 'Freq{} [MHz]:'.format(i)
#         for i in range(5,9):
#             ta['port{}'.format(i)].textbox[0].description = 'Att{} [dB]:'.format(i)
#         btn = ipw.Button(description='Long Send')#, layout={'width': '50%'})
#         btn.on_click(self.do_longsend)
#         self.panel = ipw.VBox([
#             ipw.HBox(tf['port5'].textbox),
#             ipw.HBox(tf['port6'].textbox),
#             ipw.HBox(tf['port7'].textbox),
#             ipw.HBox(tf['port8'].textbox),
#             ipw.HBox(ta['port5'].textbox),
#             ipw.HBox(ta['port6'].textbox),
#             ipw.HBox(ta['port7'].textbox),
#             ipw.HBox(ta['port8'].textbox),
#             btn,
#         ])
#     def display(self):
#         display(self.panel)
#     def get_attenuation_list(self):
#         r = [float(self.txt_atts['port8'].textbox[0].value), 0, 0] +\
#         [float(o.value) for o in self.txt_atts['port8'].textbox[1:]] +\
#         [float(o.value) for o in self.txt_atts['port7'].textbox] +\
#         [float(o.value) for o in self.txt_atts['port6'].textbox] +\
#         [float(o.value) for o in self.txt_atts['port5'].textbox] +\
#         [0, 0]
#         return r
#     def get_frequency_list(self):
#         r = [float(self.txt_freqs['port8'].textbox[0].value), 0, 0] +\
#         [float(o.value) for o in self.txt_freqs['port8'].textbox[1:]] +\
#         [float(o.value) for o in self.txt_freqs['port7'].textbox] +\
#         [float(o.value) for o in self.txt_freqs['port6'].textbox] +\
#         [float(o.value) for o in self.txt_freqs['port5'].textbox] +\
#         [0, 0]
#         return r
#     def get_amplitude_list(self):
#         a = [
#             10922, 32766, 32766,
#             10922, 10922,
#             10922, 10922,
#             10922, 10922,
#             10922, 10922,
#             32766,
#             32766,
#             ]
#         r = self.get_attenuation_list()
#         return [b * 10**(-s/20)  for b, s in zip(a,r)]
#     def do_longsend(self, e):
#         ip = self.qube.gpio.handle.addr
#         a = self.get_amplitude_list()
#         f = self.get_frequency_list()
#         long_send.stop(ipaddr='10.1.0.5')
#         long_send.start(a, f, ipaddr='10.1.0.5')
