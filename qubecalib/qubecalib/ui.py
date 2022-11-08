from . import qube as lib_qube
from . import meas

import ipywidgets as ipw
import numpy as np
import os
import subprocess
import time
from typing import Final

MATPLOTLIB_PYPLOT = None

def get_port_id(p, q):
    return int(list(filter(lambda x: x is not None, [k if v == p else None for k,v in q.ports.items()]))[0].replace('port',''))        

import sys
import argparse
import socket
import struct
import time

from telnetlib import Telnet
import time

class PDU(object):

    def _connect(self, tn):
        # tn.set_debuglevel(1)
        tn.read_until(b'Login: ')
        tn.write(b'teladmin\n')
        tn.write(b'qubeqube\n')
        tn.read_until(b'Password: *')
        tn.write(b'\n')
        tn.read_until(b'> ')
        try:
            tn.read_until(b'> ')
        except EOFError:
            tn.read_until(b'')
            return False
        return True
    
    def __enter__(self):
        while True:
            tn = Telnet('10.250.0.100', 23)
            if self._connect(tn):
                self.telnet = tn
                break
            print('retrying...')
            time.sleep(0.1)
        return self
    
    def __exit__(self, type, value, traceback):
        self.telnet.write(b'quit\n')
        
    def status(self, id):
        self.telnet.write(bytes('read status o0{}\n'.format(id), 'ascii'))
        return str(self.telnet.read_until(b'> '))
        
    def on(self, id):
        self.telnet.write(bytes('sw o0{} on\n'.format(id), 'ascii'))
        self.telnet.read_until(b'> ')
        return str(self.telnet.read_until(b'> '))

    def off(self, id):
        self.telnet.write(bytes('sw o0{} off\n'.format(id), 'ascii'))
        self.telnet.read_until(b'> ')
        return str(self.telnet.read_until(b'> '))



def lmx2594_ad9082_do_init(self, ad9082_mode=True, readout_mode=False):
    
    if ad9082_mode:
        self.write_value(0x00, 0x6612) # R6 [14]VCO_PHASE_SYNC=0
        self.write_value(0x00, 0x6610)
    else:
        self.write_value(0x00, 0x2612) # R6 [14]VCO_PHASE_SYNC=0, [9]OUT_MUTE=1
        self.write_value(0x00, 0x2610)
    self.read_value(0x00)

    self.write_value(0x4E, 0x0001)
    self.write_value(0x4D, 0x0000)
    self.write_value(0x4C, 0x000C)
    self.write_value(0x4B, 0x0840)
    self.write_value(0x4A, 0x0000) # R74 [15:2] SYSREF_PULSE_COUNT=0
    self.write_value(0x49, 0x003F)

    if ad9082_mode:
        self.write_value(0x48, 0x001F) # R72 [10:0] SYSREF_DIV
        self.write_value(0x47, 0x008D) # R71 [7:5]SYSREF_DIV_PRE=4='Divided by 4', [3]SYSREF_EN=1, [2]SYSREF_REPEAT=1
    else:
        self.write_value(0x48, 0x0000) # R72 [10:0] SYSREF_DIV
        self.write_value(0x47, 0x0081) # R71 [7:5]SYSREF_DIV_PRE=4='Divided by 4', [3]SYSREF_EN=0, [2]SYSREF_REPEAT=0

    self.write_value(0x46, 0xC350)
    self.write_value(0x45, 0x0000)
    self.write_value(0x44, 0x03E8)
    self.write_value(0x43, 0x0000)
    self.write_value(0x42, 0x01F4)
    self.write_value(0x41, 0x0000)
    self.write_value(0x40, 0x1388)
    self.write_value(0x3F, 0x0000)
    self.write_value(0x3E, 0x0322)
    self.write_value(0x3D, 0x00A8)
    self.write_value(0x3C, 0x03E8)
    self.write_value(0x3B, 0x0001)

    if ad9082_mode:
        self.write_value(0x3A, 0x0401) # R58 [15]IGNORE=0, [14]HYST=0, [13:12]INPIN_LVL=0=V
    else:
        self.write_value(0x3A, 0x8001) # R58 [15]IGNORE=0, [14]HYST=0, [13:12]INPIN_LVL=0=V

    self.write_value(0x39, 0x0020)
    self.write_value(0x38, 0x0000)
    self.write_value(0x37, 0x0000)
    self.write_value(0x36, 0x0000)
    self.write_value(0x35, 0x0000)
    self.write_value(0x34, 0x0820)
    self.write_value(0x33, 0x0080)
    self.write_value(0x32, 0x0000)
    self.write_value(0x31, 0x4180)
    self.write_value(0x30, 0x0300)
    self.write_value(0x2F, 0x0300)

    if ad9082_mode:
        self.write_value(0x2E, 0x07FE) # R46 [1:0] OUTB_MUX=2
    else:
        self.write_value(0x2E, 0x07FD) # R46 [1:0] OUTB_MUX=1(=VCO)

    if readout_mode:
        self.write_value(0x2D, 0xC8FF) # R45 [5:0] OUTB_PWR=1F
    else: # ctrl_mode
        self.write_value(0x2D, 0xC8DF) # R45

    if ad9082_mode:
        self.write_value(0x2C, 0x3220) # R44  3220
    elif readout_mode:
        self.write_value(0x2C, 0x3220) # R44  3220
    else: # ctrl_mode
        self.write_value(0x2C, 0x32A0) # R44

    self.write_value(0x2B, 0x0000)
    self.write_value(0x2A, 0x0000)
    self.write_value(0x29, 0x0000)
    self.write_value(0x28, 0x0000)
    self.write_value(0x27, 0x0001)
    self.write_value(0x26, 0x0000)
    self.write_value(0x25, 0x0204)

    if ad9082_mode:
        self.write_value(0x24, 0x001e)
    else:
        self.write_value(0x24, 0x0078)

    self.write_value(0x23, 0x0004)
    self.write_value(0x22, 0x0000)
    self.write_value(0x21, 0x1E21)
    self.write_value(0x20, 0x0393)
    self.write_value(0x1F, 0x43EC)
    self.write_value(0x1E, 0x318C)
    self.write_value(0x1D, 0x318C)
    self.write_value(0x1C, 0x0488)
    self.write_value(0x1B, 0x0002)
    self.write_value(0x1A, 0x0DB0)
    self.write_value(0x19, 0x0C2B)
    self.write_value(0x18, 0x071A)
    self.write_value(0x17, 0x007C)
    self.write_value(0x16, 0x0001)
    self.write_value(0x15, 0x0401)
    self.write_value(0x14, 0xC848)
    self.write_value(0x13, 0x27B7)
    self.write_value(0x12, 0x0064)
    self.write_value(0x11, 0x00FA)
    self.write_value(0x10, 0x0080)
    self.write_value(0x0F, 0x064F)
    self.write_value(0x0E, 0x1E10)
    self.write_value(0x0D, 0x4000)
    self.write_value(0x0C, 0x5001)
    self.write_value(0x0B, 0x0018)
    self.write_value(0x0A, 0x10D8)
    self.write_value(0x09, 0x0604)
    self.write_value(0x08, 0x2000)
    self.write_value(0x07, 0x00B2)
    self.write_value(0x06, 0xC802)
    self.write_value(0x05, 0x00C8)
    self.write_value(0x04, 0x1B43)
    self.write_value(0x03, 0x0642)
    self.write_value(0x02, 0x0500)
    self.write_value(0x01, 0x080B)

    if ad9082_mode:
        self.write_value(0x00, 0x6618)
        self.write_value(0x22, 0x0000)
        self.write_value(0x24, 0x001e)
        self.write_value(0x26, 0x0000)
        self.write_value(0x27, 0x0064)
        self.write_value(0x2A, 0x0000)
        self.write_value(0x2B, 0x0000)
        self.write_value(0x00, 0x6618)
    else:
        self.write_value(0x00, 0x2618)

    return self.read_value(0x00)




class QuBEMonitor(object):
    BUFSIZE = 16384
    TIMEOUT = 25
    
    def __init__(self, ip_addr, port):
        self.__dest_addr = (ip_addr, port)
        self.__sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__sock.settimeout(self.TIMEOUT)
        self.__sock.bind(('', 0))
        print('open: {}:{}'.format(ip_addr, port))
        
    def send_recv(self, data):
        try:
            self.__sock.sendto(data, self.__dest_addr)
            return self.__sock.recvfrom(self.BUFSIZE)
        except socket.timeout as e:
            print('{}, Dest {}'.format(e, self.__dest_addr))
            raise
        except Exeption as e:
            print(e)
            raise
        
    def kick_softreset(self):
        data = struct.pack('BBBB', 0xE0, 0x00, 0x00, 0x00)
        ret, addr = self.send_recv(data)
        print(ret)



    
        
        
class QubeControl(object):
    
    def __init__(self, config_file_name, qube=None):
        
        self._container = {}
        c: Final[dict] = self._container
        if qube is None:
            c['qube'] = lib_qube.Qube.create(config_file_name)
        else:
            c['qube'] = qube
        # c['wout'] = ipw.Output(layout={'border': '1px solid black'})
        c['wout'] = ipw.Output()
        c['fname'] = ipw.Text(description='', value=config_file_name, disabled=True)
        c['mon'] = ipw.Checkbox(value=False, description='Monitor', disabled=False)
        
        class ShowStatusButton(ipw.Button):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.description = 'Show status'
                self.on_click(self._on_click)
            def _on_click(self, e):
                qube, wout = c['qube'], c['wout']
                wout.clear_output()
                with wout:
                    print("GPIO: {:04x}".format(qube.gpio.read_value()))
                    print("LinkStatus:")
                    for o in qube.ad9082:
                        print(o.get_jesd_status())
        
        class ShowConfigButton(ipw.Button):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.description = 'Show config'
                self.on_click(self._on_click)
            def _on_click(self, e):
                c['wout'].clear_output()
                with c['wout']:
                    print("Config:")
                    print(c['qube'].config)

        class RestartAD9082Button(ipw.Button):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.description = 'Restart AD9082'
                self.on_click(self._on_click)
            def _on_click(self, e):
                
                def ad9082_do_init():
                    for o in c['qube'].ad9082:
                        if c['mon'].value:
                            os.environ['TARGET_ADDR'] = o.addr
                            os.environ['AD9082_CHIP'] = o.chip
                            ret = subprocess.check_output('{}/v1.0.6/src/hello_monitor'.format(o.path), encoding='utf-8')
                        else:
                            o.do_init(message_out=False)
                        
                qube, wout, mon = c['qube'], c['wout'], c['mon']
                c['wout'].clear_output()
                with c['wout']:
                    for i in range(100):
                        print(i+1, end=' ', flush=True)
                        for p in c['qube'].lmx2594_ad9082:
                            # p.do_init(ad9082_mode=True, message_out=False)
                            lmx2594_ad9082_do_init(p)
                        time.sleep(0.1)
                        ad9082_do_init()
                        ad9082_do_init()
                        for o in c['qube'].ad9082:
                            print(dict(o.get_jesd_status())['0x55E'], end=' ', flush=True)
                        s = [dict(c['qube'].ad9082[i].get_jesd_status())['0x55E'] == '0xE0' for i in range(2)]
                        if s == [True, True]:
                            break
                    if mon.value:
                        qube.gpio.write_value(0xFFFF)
                    else:
                        qube.gpio.write_value(0x0000)
                    print("\nGPIO: {:04x}".format(qube.gpio.read_value()))
                    print("LinkStatus:")
                    for o in qube.ad9082:
                        print(o.get_jesd_status())
                        
        class DoInitButton(ipw.Button):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.description = 'do_init'
                self.on_click(self._on_click)
            def _on_click(self, e):
                qube, wout = c['qube'], c['wout']
                wout.clear_output()
                with wout:
                    qube.do_init(message_out=True)
                    print("\nGPIO: {:04x}".format(qube.gpio.read_value()))
                    print("LinkStatus:")
                    for o in qube.ad9082:
                        print(o.get_jesd_status())
        
        class ShowRecvButton(ipw.Button):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.description = 'Recv'
                self.on_click(self._on_click)
            def _on_click(self, e):
                assert [dict(c['qube'].ad9082[i].get_jesd_status())['0x55E'] == '0xE0' for i in range(2)] == [True, True], 'Link status is unusual.'
                
                qube = c['qube']
                p = meas.CaptureParam()
                p.num_integ_sections = 1
                p.add_sum_section(num_words=1024, num_post_blank_words=1)
                p.capture_delay = 100
                
                if c['mon'].value:
                    p1 = qube.port4
                    p12 = qube.port9
                else:
                    p1 = qube.port1
                    p12 = qube.port12
                r1 = meas.Recv(qube.ipfpga, p1.capt, p)
                r12 = meas.Recv(qube.ipfpga, p12.capt, p)
                r1.start(timeout=0.5)
                r12.start(timeout=0.5)
                
                plt = MATPLOTLIB_PYPLOT
                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.plot(np.real(r1.data.data[meas.CaptureModule.get_units(p1.capt.id)[0]]))
                ax.plot(np.imag(r1.data.data[meas.CaptureModule.get_units(p1.capt.id)[0]]))
                ax.text(0.05, 0.1, 'port{}'.format(get_port_id(p1, qube)), transform=ax.transAxes)
                ax = fig.add_subplot(212)
                ax.plot(np.real(r12.data.data[meas.CaptureModule.get_units(p12.capt.id)[0]]))
                ax.plot(np.imag(r12.data.data[meas.CaptureModule.get_units(p12.capt.id)[0]]))
                ax.text(0.05, 0.1, 'port{}'.format(get_port_id(p12, qube)), transform=ax.transAxes)
                
                c['wout'].clear_output()
                import matplotlib
                with c['wout']:
                    plt.show(fig)
                plt.close()
                
        class ConfigFPGAButton(ipw.Button):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.description = 'ConfigFPGA'
                self.on_click(self._on_click)
                # self.bitfile = '{}/{}'.format(qubecalib.qube.PATH_TO_BITFILE, c['qube'].bitfile)
            def _on_click(self, e):
                # print(c['qube'], c['qube']['bitfile'])
                c['wout'].clear_output()
                with c['wout']:
                    print('configure FPGA ...')
                # c['qube'].config_fpga()
                os.environ['BITFILE'] = '{}/{}'.format(lib_qube.PATH_TO_BITFILE, c['qube'].bitfile)
                with c['wout']:
                    print('BITFILE: {}'.format(os.environ['BITFILE']))
                commands = ["vivado", "-mode", "batch", "-source", "{}/utils/config.tcl".format(lib_qube.PATH_TO_API)]
                ret = subprocess.run(commands , stdout=subprocess.PIPE, check=True).stdout
                with c['wout']:
                    print(ret)
                
        class ShowPortsButton(ipw.Button):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.description = 'Show port config'
                self.on_click(self._on_click)
            def _on_click(self, e):
                c['wout'].clear_output()
                with c['wout']:
                    print("Port:")
                    for k, v in c['qube'].ports.items():
                        print(k, v)
                    
        class ShowPortStatusButton(ipw.Button):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.description = 'Show port status'
                self.on_click(self._on_click)
            def _on_click(self, e):
                c['wout'].clear_output()
                with c['wout']:
                    print("Port Status:")
                    for k, v in c['qube'].ports.items():
                        print(k,v.status)
                        
        class KickSoftReset(ipw.Button):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.description = 'Kick soft reset'
                self.on_click(self._on_click)
            def _on_click(self, e):
                qube, wout = c['qube'], c['wout']
                wout.clear_output()
                with wout:
                    q = QuBEMonitor(qube.ipmulti, 16384)
                    q.kick_softreset()
                    print("GPIO: {:04x}".format(qube.gpio.read_value()))
                    print("LinkStatus:")
                    for o in qube.ad9082:
                        print(o.get_jesd_status())
                        
        self.widgets = ipw.VBox([
            ipw.HBox([
                c['fname'],
                # CreateQubeInstanceButton(description='Create instance'),
                DoInitButton(),
                ipw.Text(description='', value=c['qube'].bitfile, disabled=True),
                ConfigFPGAButton(),
            ]),
            ipw.HBox([
                ShowStatusButton(),
                ShowConfigButton(),
                ShowRecvButton(),
                c['mon'],
                RestartAD9082Button(),
            ]),
            ipw.HBox([
                ShowPortsButton(),
                ShowPortStatusButton(),
                KickSoftReset(),
            ]),
            ipw.HBox([
                c['wout'],
            ]),
        ])
        display(self.widgets)
        
    @property
    def qube(self):
        if self._container['qube'] == None:
            raise ValueError('An instance of qube must be created before it can be referenced.')
        return self._container['qube']
    
    @property
    def recv(self):
        return self._container['recv']


# class CWControl(object):
    
#     def __init__(self, label, port):
#         self._container = {}
#         c: Final[dict] = self._container
#         c['port'] = port
#         c['wout'] = ipw.Output()
        
#         class LOFrequency(ipw.Text):
#             def __init__(self, *args, **kw):
#                 super().__init__(*args, **kw)
#                 self.description=''
#                 self.value = c['port'].lo.mhz
#             def refresh(self):
#                 self.value = c['port'].lo.mhz
#             def update(self):
#                 c['port'].lo.mhz = int(self.value)
        