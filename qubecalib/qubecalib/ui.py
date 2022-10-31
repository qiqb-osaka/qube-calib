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
                            p.do_init(ad9082_mode=True, message_out=False)
                        time.sleep(1)
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
                
                if qube.gpio.read_value() == 0x3fff:
                    p1 = qube.port3
                    p12 = qube.port10
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
        