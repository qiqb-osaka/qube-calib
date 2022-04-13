import ipywidgets as ipw
from collections import namedtuple
import IPython.display
import qubecalib
from traitlets import HasTraits, Unicode, observe, link

def display(*args, **kwargs):
    IPython.display.display(*args, **kwargs)

class Qube(HasTraits): # todo: 多重継承の正しいやり方を知りたい qubecalib.Qube も継承させたい
    config_file_name = Unicode()
    iplsi = Unicode()
    rftype = Unicode()
    def __init__(self, addr=None, path=None, *args, **kwargs):
        super().__init__()
        self.qube = q = qubecalib.Qube(addr, path, *args, **kwargs)
        if q.config_file_name != None:
            self.config_file_name = q.config_file_name
        self._ports = None
    def _config_file_name_changed(self, change):
        self.qube.config_file_name = getattr(self, change)
    def load(self):
        self.qube.load()
        self.iplsi = self.qube.config['iplsi']
        self.rftype = self.qube.config['type']
    def do_init(self, rf_type='A', bitfile=None, message_out=False):
        self.qube.do_init(rf_type, bitfile, message_out)
    def config_fpga(self, bitfile, message_out=False):
        self.qube.config_fpga(bitfile, message_out=False)
    def restart_ad9082(self, message_out=True):
        for o in self.qube.ad9082:
            o.do_init(message_out=message_out)
    @property
    def config(self):
        return self.qube.config
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

class LoadConfigPanel(ipw.HBox):
    def __init__(self, qube, *args, **kwargs):
        self.qube = qube
        self.links = []
        self.tb_fname = t = ipw.Text(description='Config')
        if qube.config_file_name == '':
            qube.config_file_name = 'qube_riken_1-01.yml'
        self.links.append(link((qube, 'config_file_name'), (t, 'value')))
        self.tb_iplsi = t = ipw.Text(description='IP (LSI)', disabled=True)
        self.links.append(link((qube, 'iplsi'), (t, 'value')))
        self.tb_rftype = t = ipw.Text(description='RF Type', disabled=True)
        self.links.append(link((qube, 'rftype'), (t, 'value')))
        self.btn_load = b = ipw.Button(description='Load'); b.on_click(lambda e: self.qube.load())
        ipw.HBox.__init__(self, [self.tb_fname, b, self.tb_iplsi, self.tb_rftype], *args, **kwargs)
    def unlink(self):
        for o in self.links:
            o.unlink()

            
class QubeSetupPanel(ipw.VBox):
    def __init__(self, qube, *args, **kwargs):
        self.qube = qube
        self.tb_ip4lsi = tb_ip4lsi = ipw.Text(description='IP Address (LSI)', style={'description_width': 'initial'}, disabled=True)
        self.tb_path2api = tb_path2api = ipw.Text(description='Path to API', style={'description_width': 'initial'}, layout=ipw.Layout(width='50%'), disabled=True)
        self.cb_config_fpga = cb = ipw.Checkbox(value=False, description='Config FPGA', disabled=False, indent=False)
        self.tb_path_bitfile = bitfile = ipw.Text(
            description='Path to bitfile',
            value='',
            style={'description_width': 'initial'},
            disabled=False)
        kw = {'layout': {'width': '50%', 'height': '80px'}, 'disabled': True}
        self.btn_do_init = b = ipw.Button(description='Do init', **kw); b.on_click(self.do_init)
        self.btn_ad9082 = b = ipw.Button(description='Restart AD9082', **kw); b.on_click(self.do_init_ad9082)
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
