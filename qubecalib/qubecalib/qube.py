import sys
from abc import ABC, abstractmethod
from enum import Enum, auto
import yaml
import math
import warnings
import qubelsi
import qubelsi.qube
from e7awgsw import AwgCtrl
import e7awgsw
from typing import Final
import os
import subprocess
import weakref

from . import meas


PATH_TO_CONFIG: Final[str] = os.environ['QUBECALIB_PATH_TO_CONFIG']
PATH_TO_ADIAPIMOD: Final[str] = qubelsi.__path__[0]+'/bin'
PATH_TO_BITFILE: Final[str] = os.environ['QUBECALIB_PATH_TO_BITFILE']

class LSI(ABC):
    
    def __init__(self, lsi):
        self.lsi = lsi
        
    @abstractmethod
    def _status(self):
        pass
    
    @property
    def status(self):
        return self._status()
    
class LMX2594(LSI):
    
    @property
    def mhz(self):
        return self.lsi.read_freq_100M() * 100
    @mhz.setter
    def mhz(self, v):
        
        if v < 8000 or 15000 < v:
            raise ValueError('The frequency must be between 8000MHz and 15000MHz.')
            
        mhz = math.floor(v / 100)

        if v % 100:
            warnings.warn('The frequency can only be set in multiples of 100MHz. {}MHz is set.'.format(mhz*100), stacklevel=2)
            
        if mhz*100 % 500:
            warnings.warn('To synchronize the output phase, the frequency must be a multiple of 500MHz.', stacklevel=2)
            
        self.lsi.write_freq_100M(mhz)
        
    def _status(self):
        
        return {
            'mhz': self.mhz
        }


class AD5328(LSI):
    
    def __init__(self, lsi, ch):
        super().__init__(lsi)
        self.ch = ch
        self._value = 0x800
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, v):
        lsi, ch = self.lsi, self.ch
        lsi.write_value(ch, v)
        lsi.write_value(0xA, 0x002) # apply value
        self._value = v
        return v / 0xfff * 3.3 # volt
    
    def _status(self):
        
        return {
            'value': self.value
        }

    
class SSB(Enum):
    USB = auto()
    LSB = auto()

    
class ADRF6780(LSI):
    
    def __init__(self, lsi, ad5328):
        
        super().__init__(lsi)
        self.ad5328 = ad5328
        
    @property
    def ssb(self):
        
        return SSB.USB if self.lsi.read_mode() == 0 else SSB.LSB
    
    @ssb.setter
    def ssb(self, v):
        
        if v == SSB.USB:
            self.lsi.set_usb()
        elif v == SSB.LSB:
            self.lsi.set_lsb()
    
    @property
    def vatt(self):
        
        return self.ad5328.value
    
    @vatt.setter
    def vatt(self, v):
        
        self.ad5328.value = v

    def _status(self):
        
        return {
            'ssb': self.ssb,
            'vatt': self.vatt,
        }
        
        
class AD9082(LSI):
    def __init__(self, lsi):
        super().__init__(lsi)
        
        
class NCO(AD9082):
    
    def __init__(self, lsi, ch):
        
        super().__init__(lsi)
        self.ch = ch
        self._mhz = 0
        
    @abstractmethod
    def _set_frequency(self, v):
        
        pass
    
    @property
    def mhz(self):
        
        return self._mhz
    
    @mhz.setter
    def mhz(self, v):
        
        self._mhz = v
        self._set_frequency(v)
        
    def _status(self):
        
        return {'mhz': self.mhz}
        
        
class CNCODAC(NCO):
    
    def __init__(self, lsi, ch):
        super().__init__(lsi, ch)
        self._mhz = 970 # 下位の API で決め打ち
        
    def _set_frequency(self, v):
        self.lsi.set_nco(freq=v*1e+6, ch=self.ch, adc_mode=False, fine_mode=False)
        
        
class FNCODAC(NCO):
    
    def __init__(self, lsi, ch):
        super().__init__(lsi, ch)
        self._mhz = 0 # 下位の API で決め打ち
        
    def _set_frequency(self, v):
        self.lsi.set_nco(freq=v*1e+6, ch=self.ch, adc_mode=False, fine_mode=True)
        
        
class CNCOADC(NCO):
    
    def __init__(self, lsi, ch):
        super().__init__(lsi, ch)
        self._mhz = 970 # 下位の API で決め打ち
        
    def _set_frequency(self, v):
        self.lsi.set_nco(freq=v*1e+6, ch=self.ch, adc_mode=True, fine_mode=False)
        
        
class FNCOADC(NCO):
    
    def __init__(self, lsi, ch):
        super().__init__(lsi, ch)
        self._mhz = 0 # 下位の API で決め打ち
        
    def _set_frequency(self, v):
        self.lsi.set_nco(freq=v*1e+6, ch=self.ch, adc_mode=True, fine_mode=True)
        

class AWG(AD9082):
    
    def __init__(self, lsi, awg, ipfpga, nco):
        super().__init__(lsi)
        self.id = awg
        self.ipfpga = ipfpga
        self.nco = nco
        self._port = None
       
    @property
    def port(self):
        return self._port()

    def _status(self):
        
        return {
            'nco': self.nco.status,
        }
    
    def send(self, wave_seq):
        
        o = meas.Send(self.ipfpga, [self], [wave_seq])
        o.start()
        
    def terminate(self):
        
        with AwgCtrl(self.ipfpga) as a:
            a.terminate_awgs(self.id)
        
    def modulation_frequency(self, mhz):
        
        port = self.port
        lo_mhz = port.lo.mhz
        cnco_mhz = port.nco.mhz
        fnco_mhz = self.nco.mhz
        usb_mhz = lo_mhz + (cnco_mhz + fnco_mhz)
        lsb_mhz = lo_mhz - (cnco_mhz + fnco_mhz)
        if port.mix.ssb == SSB.USB:
            df_mhz = mhz - usb_mhz # mhz = usb_mhz + df
        elif port.mix.ssb == SSB.LSB:
            df_mhz = lsb_mhz - mhz # mhz = lsb_mhz - df
        else:
            raise ValueError('A port.mix.ssb shuld be instance of SSB(Enum).')
        return df_mhz

class UNIT(object):
    def __init__(self, capt, unit):
        self.id = unit
        self.__capt = weakref.ref(capt)

    @property
    def capt(self):
        return self.__capt()

class CPT(AD9082):
    
    def __init__(self, lsi, cptm, ipfpga):
        super().__init__(lsi)
        self.id = cptm
        self.ipfpga = ipfpga
        self._port = None
        self.ssb = SSB.USB
        for i,u in enumerate(e7awgsw.CaptureModule.get_units(cptm)):
            setattr(self,'unit{}'.format(i),UNIT(self,u))

    @property
    def port(self):
        return self._port()
        
    def _status(self):
        
        return None
        
    def recv(self, capt_param, timeout=5):
        
        o = meas.Recv(self.ipfpga, self, capt_param)
        o.start(timeout=timeout)
        
        return None
        
    def modulation_frequency(self, mhz):
        
        port = self.port
        lo_mhz = port.lo.mhz
        cnco_mhz = port.nco.mhz
        usb_mhz = lo_mhz + cnco_mhz
        lsb_mhz = lo_mhz - cnco_mhz
        if self.ssb == SSB.USB:
            df_mhz = mhz - usb_mhz
        elif self.ssb == SSB.LSB:
            df_mhz = lsb_mhz - mhz
        else:
            raise ValueError('A port.captX.ssb shuld be instance of SSB(Enum).')
        return df_mhz
        
class DAC(AD9082):
    
    def __init__(self, lsi, ch, ipfpga, awgs):
        super().__init__(lsi)
        self.nco = CNCODAC(lsi, ch)
        self.nawgs = len(awgs)
        f = lambda a: AWG(lsi, a[0], ipfpga, FNCODAC(lsi, a[1]))
        k = lambda i: 'awg{}'.format(i)
        self._awgs = a =  {k(i): f(awg) for i, awg in enumerate(awgs)}
        for k, v in a.items():
            setattr(self, k, v)
    
    def _status(self):
        
        return {
            'nco': self.nco.status,
            'awgs': {k: v.status for k, v in self._awgs.items()},
        }
    
    
class ADC(AD9082):
    
    def __init__(self, lsi, ch, ipfpga, cpts):
        super().__init__(lsi)
        self.nco = CNCOADC(lsi, ch)
        for i, cpt in enumerate(cpts):
            k = lambda i: 'capt{}'.format(i)
            setattr(self, k(i), CPT(lsi, cpt, ipfpga))

    def _status(self):
        
        return {
            'nco': self.nco.status,
        }

class RF(object):
    
    def __init__(self, func):
        
        self.func = func
        
    @property
    def mhz(self):
        
        return self.func()
    
class Port(object):
    
    def __init__(self, qube):
        self._qube = weakref.ref(qube)

    @property
    def qube(self):
        return self._qube()
        
    @abstractmethod
    def _status(self):
        
        pass
    
    @property
    def status(self):
        
        return self._status()
        
        
class Output(Port):
    
    def __init__(self, qube, dac, lo, mix):
        super().__init__(qube)
        self._rf = RF(self._calc_rf)
        self.dac = dac
        self.lo = lo
        self.mix = mix
        
        setattr(self, 'awg', dac.awg0)
        for i in range(dac.nawgs):
            awg = getattr(dac, 'awg{}'.format(i))
            setattr(self, 'awg{}'.format(i), awg)
        
    @property
    def nco(self):
        
        return self.dac.nco
        
    @property
    def awgs(self):
        
        n = self.dac.nawgs
        k = lambda i: 'awg{}'.format(i)
        
        return {k(i): getattr(self.dac, k(i)) for i in range(n)}
        
    @property
    def rf(self):
        
        return self._rf
        
    def _calc_rf(self):
        
        rst = {}
        
        lo = self.lo.mhz
        cnco = self.nco.mhz
        for k, v in self.awgs.items():
            if self.mix.ssb == SSB.LSB:
                rst[k] = lo - cnco - v.nco.mhz
            if self.mix.ssb == SSB.USB:
                rst[k] = lo + cnco + v.nco.mhz
        
        return rst
        
    def _status(self):
        
        return {
            'dac': self.dac.status,
            'lo': self.lo.status,
            'mix': self.mix.status,
        }
        
        
class Input(Port):
    
    def __init__(self, qube, adc, lo):
        super().__init__(qube)
        self._rf = RF(self._calc_rf)
        self.adc = adc
        self.lo = lo
        
    @property
    def capt(self):
        
        return self.adc.capt0

    @property
    def nco(self):
        
        return self.adc.nco
    
    @property
    def rf(self):
        
        return self._rf
        
    def _calc_rf(self):
        
        lo = self.lo.mhz
        cnco = self.nco.mhz
        
        return (lo - cnco, lo + cnco)
        
        
    def _status(self):
        
        return {
            'adc': self.adc.status,
            'lo': self.lo.status,
        }
        
        
class Monitorout(Port):
    
    def __init__(self, qube):
        super().__init__(qube)
        self._rf = RF(self._calc_rf)
    
    def _calc_rf(self):
        
        return None
        
class NotAvailable(Port):
    
    def __init__(self, qube):
        super().__init__(qube)
        self._rf = RF(self._calc_rf)
        
    def _calc_rf(self):
        
        return None
    
        
class Ctrl(Output):
    
    def __init__(self, qube, dac, lo, mix):
        super().__init__(qube, dac, lo, mix)
        self.mix.ssb = SSB.LSB


class Readout(Output):
    
    def __init__(self, qube, dac, lo, mix):
        super().__init__(qube, dac, lo, mix)
        self.mix.ssb = SSB.USB
    
    
class Pump(Output):
    
    def __init__(self, qube, dac, lo, mix):
        super().__init__(qube, dac, lo, mix)
        self.mix.ssb = SSB.USB
    
    def _calc_rf(self):
        
        rst = super()._calc_rf()
        
        return {k: v * 2 for k, v in rst.items()}
    
class Readin(Input):
    
    def __init__(self, qube, adc, lo):
        super().__init__(qube, adc, lo)
    
    
class Monitorin(Input):
    
    def __init__(self, qube, adc, lo):
        super().__init__(qube, adc, lo)

class ConfigFPGA(object):
    
    @classmethod
    def config(cls, bitfile: str) -> None:
        os.environ['BITFILE'] = '{}/{}'.format(PATH_TO_BITFILE, bitfile)
        commands = ["vivado", "-mode", "batch", "-source", "{}/utils/config.tcl".format(PATH_TO_ADIAPIMOD)]
        ret = subprocess.check_output(commands , encoding='utf-8')
        return ret
        

class Qube(object): # QubeInstanceFactory
    '''
    config ファイルを読んで適切な QubeBase インスタンスを生成する
    qube = Qube.create(<config_file_name>)
    '''

    @classmethod
    def load(cls, config_file_name):
        
        name = '{}/{}'.format(PATH_TO_CONFIG, config_file_name)
        with open(name, 'rb') as f:
            o = yaml.safe_load(f)
        
        s = o['ipfpga'].split('.')
        s[1] = "2"
        o['ipmulti'] = '.'.join(map(str, s))
        
        return o

    @classmethod
    def create(cls, config_file_name):
        
        o = cls.load(config_file_name)
        
        if o['type'] == 'A':
            return QubeTypeA(o['iplsi'], PATH_TO_ADIAPIMOD, o)
        
        if o['type'] == 'B':
            return QubeTypeB(o['iplsi'], PATH_TO_ADIAPIMOD, o)
        
        
class QubeBase(qubelsi.qube.Qube):
    '''
    任意の IPADDR で設定する
    qube = QubeA(ipaddr, path_to_api) | QubeB(ipaddr, path_to_api)
    '''
    
    def __init__(self, addr, path, config=None):
        
        super().__init__(addr, path)
        if config is None:
            self.iplsi = addr
            buf = addr.split('.')
            buf[1] = '1'
            self.ipfpga = '.'.join(buf)
            buf = addr.split('.')
            buf[1] = '2'
            self.ipmulti = '.'.join(buf)
            self._config = {
                'iplsi': self.iplsi,
                'ipfpga': self.ipfpga,
                'ipmulti': self.ipmulti,
            }
            adapter_au50 = ''
            return
        self.bitfile: Final[str] = config['bitfile']
        self.ipfpga: Final[str] = config['ipfpga']
        self.iplsi: Final[str] = config['iplsi']
        self.ipmulti: Final[str] = config['ipmulti']
        self.macfpga: Final[str] = config['macfpga']
        self.maclsi: Final[str] = config['maclsi']
        self.type: Final[str] = config['type']
        self.adapter_au50: Final[str] = config['adapter_au50']
        self._config = config
        
    def __getitem__(self, v):
        
        return self._config[v]
        
    def config_fpga(self, bitfile=None):
        
        if bitfile is None and not 'bitfile' in self._config:
            raise ValueError('Specify bitfile.')
        
        if bitfile is None:
            bitfile = self['bitfile']
            
        self._config_fpga(bitfile)
            
    def _config_fpga(self, bitfile):
        
        os.environ['BITFILE'] = '{}/{}'.format(PATH_TO_BITFILE, bitfile)
        commands = ["vivado", "-mode", "batch", "-source", "{}/utils/config.tcl".format(PATH_TO_ADIAPIMOD)]
        ret = subprocess.check_output(commands , encoding='utf-8')
        return ret
    
    @property
    def are_ad9082s_connected_normally(self):
        ad9082s = self.ad9082
        s = [dict(o.get_jesd_status())['0x55E'] == '0xE0' for o in ad9082s]
        return s == [True, True]

    def restart_ad9082s(self):
        ad9082s = self.ad9082
        for o in self.lmx2594_ad9082:
            o.do_init(ad9082_mode=True, message_out=False)
        for i in range(100):
            print(i+1, end=' ', flush=True)
            for o in ad9082s:
                o.do_init(message_out=False)
                print(dict(o.get_jesd_status())['0x55E'], end=' ', flush=True)
            if self.are_ad9082s_connected_normally:
                break

    @property
    def config(self):
        
        return self._config
        
    @property
    def ports(self):
        
        k = lambda i: 'port{}'.format(i)
        p = {k(i): getattr(self, k(i)) for i in range(self['nports'])}
        return p
        

class QubeTypeA(QubeBase):
    def __init__(self, addr, path, config):
        super().__init__(addr, path, config)
        if config is not None:
            self.nports: Final[int] = 14
            self._config['nports'] = self.nports
        ip = self['ipfpga']
        dac = self.ad9082
        adc = self.ad9082
        lo = self.lmx2594
        mix = self.adrf6780
        vatt = self.ad5328
        e7 = e7awgsw
        
        self.port0: Final[Port] = Readout(
            qube = self,
            dac = DAC(lsi = dac[0], ch = 0, ipfpga = ip, awgs = [(e7.AWG.U15, 0),]),
            lo = LMX2594(lsi = lo[0]),
            mix = ADRF6780(lsi = mix[0], ad5328 = AD5328(lsi = vatt, ch = 0)),
        )
        for p in self.port0.awgs.values():
            p._port = weakref.ref(self.port0)
        
        self.port1: Final[Port] = Readin(
            qube = self,
            adc = ADC(lsi = adc[0], ch = 3, ipfpga = ip, cpts = [e7.CaptureModule.U1,]),
            lo = LMX2594(lsi = lo[0]),
        )
        self.port1.adc.capt0._port = weakref.ref(self.port1)

        self.port2: Final[Port] = Pump(
            qube = self,
            dac = DAC(lsi = dac[0], ch = 1, ipfpga = ip, awgs = [(e7.AWG.U14, 1),]),
            lo = LMX2594(lsi = lo[1]),
            mix = ADRF6780(lsi = mix[1], ad5328 = AD5328(lsi = vatt, ch = 1)),
        )
        for p in self.port2.awgs.values():
            p._port = weakref.ref(self.port2)
        
        self.port3: Final[Port] = Monitorout(qube = self)
        
        self.port4: Final[Port] = Monitorin(
            qube = self,
            adc = ADC(lsi = adc[0], ch = 2, ipfpga = ip, cpts = [e7.CaptureModule.U1,]),
            lo = LMX2594(lsi = lo[1]),
        )
        self.port4.adc.capt0._port = weakref.ref(self.port4)
        
        self.port5: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[0], ch = 2, ipfpga = ip, awgs = [(e7.AWG.U11, 4),(e7.AWG.U12, 3),(e7.AWG.U13, 2),]),
            lo = LMX2594(lsi = lo[2]),
            mix = ADRF6780(lsi = mix[2], ad5328 = AD5328(lsi = vatt, ch = 2)),
        )
        for p in self.port5.awgs.values():
            p._port = weakref.ref(self.port5)
        
        self.port6: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[0], ch = 3, ipfpga = ip, awgs = [(e7.AWG.U8, 5),(e7.AWG.U9, 6),(e7.AWG.U10, 7),]),
            lo = LMX2594(lsi = lo[3]),
            mix = ADRF6780(lsi = mix[3], ad5328 = AD5328(lsi = vatt, ch = 3)),
        )
        for p in self.port6.awgs.values():
            p._port = weakref.ref(self.port6)
        
        self.port7: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[1], ch = 0, ipfpga = ip, awgs = [(e7.AWG.U5, 2),(e7.AWG.U6, 1),(e7.AWG.U7, 0),]),
            lo = LMX2594(lsi = lo[4]),
            mix = ADRF6780(lsi = mix[4], ad5328 = AD5328(lsi = vatt, ch = 4)),
        )
        for p in self.port7.awgs.values():
            p._port = weakref.ref(self.port7)
        
        self.port8: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[1], ch = 1, ipfpga = ip, awgs = [(e7.AWG.U0, 5),(e7.AWG.U3, 4),(e7.AWG.U4, 3),]),
            lo = LMX2594(lsi = lo[5]),
            mix = ADRF6780(lsi = mix[5], ad5328 = AD5328(lsi = vatt, ch = 5)),
        )
        for p in self.port8.awgs.values():
            p._port = weakref.ref(self.port8)
        
        self.port9: Final[Port] = Monitorin(
            qube = self,
            adc = ADC(lsi = adc[1], ch = 2, ipfpga = ip, cpts = [e7.CaptureModule.U0,]),
            lo = LMX2594(lsi = lo[6]),
        )
        self.port9.adc.capt0._port = weakref.ref(self.port9)
        
        self.port10: Final[Port] = Monitorout(qube = self)
        
        self.port11: Final[Port] = Pump(
            qube = self,
            dac = DAC(lsi = dac[1], ch = 2, ipfpga = ip, awgs = [(e7.AWG.U1, 6),]),
            lo = LMX2594(lsi = lo[6]),
            mix = ADRF6780(lsi = mix[6], ad5328 = AD5328(lsi = vatt, ch = 6)),
        )
        for p in self.port11.awgs.values():
            p._port = weakref.ref(self.port11)
        
        self.port12: Final[Port] = Readin(
            qube = self,
            adc = ADC(lsi = adc[1], ch = 3, ipfpga = ip, cpts = [e7.CaptureModule.U0,]),
            lo = LMX2594(lsi = lo[7]),
        )
        self.port12.adc.capt0._port = weakref.ref(self.port12)
        
        self.port13: Final[Port] = Readout(
            qube = self,
            dac = DAC(lsi = dac[1], ch = 3, ipfpga = ip, awgs = [(e7.AWG.U2, 7),]),
            lo = LMX2594(lsi = lo[7]),
            mix = ADRF6780(lsi = mix[7], ad5328 = AD5328(lsi = vatt, ch = 7)),
        )
        for p in self.port13.awgs.values():
            p._port = weakref.ref(self.port13)
        
        self.readout0: Final[Port] = self.port0
        self.readin0: Final[Port] = self.port1
        self.pump0: Final[Port] = self.port2
        self.auxout0: Final[Port] = self.port3
        self.auxin0: Final[Port] = self.port4
        self.ctrl0: Final[Port] = self.port5
        self.ctrl1: Final[Port] = self.port6
        self.ctrl2: Final[Port]= self.port7
        self.ctrl3: Final[Port] = self.port8
        self.auxin1: Final[Port] = self.port9
        self.auxout1: Final[Port] = self.port10
        self.pump1: Final[Port] = self.port11
        self.readin1: Final[Port] = self.port12
        self.readout1: Final[Port] = self.port13
        
        
class QubeTypeB(QubeBase):
    def __init__(self, addr, path, config):
        super().__init__(addr, path, config)
        if config is not None:
            self.nports: Final[int] = 14
            self._config['nports'] = self.nports
        ip = self['ipfpga']
        dac = self.ad9082
        adc = self.ad9082
        lo = self.lmx2594
        mix = self.adrf6780
        vatt = self.ad5328
        e7 = e7awgsw
        
        self.port0: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[0], ch = 0, ipfpga = ip, awgs = [(e7.AWG.U15, 0),]),
            lo = LMX2594(lsi = lo[0]),
            mix = ADRF6780(lsi = mix[0], ad5328 = AD5328(lsi = vatt, ch = 0)),
        )
        for p in self.port0.awgs.values():
            p._port = weakref.ref(self.port0)
        
        self.port1: Final[Port] = NotAvailable(qube = self)
        
        self.port2: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[0], ch = 1, ipfpga = ip, awgs = [(e7.AWG.U14, 1),]),
            lo = LMX2594(lsi = lo[1]),
            mix = ADRF6780(lsi = mix[1], ad5328 = AD5328(lsi = vatt, ch = 1)),
        )
        for p in self.port2.awgs.values():
            p._port = weakref.ref(self.port2)
        
        self.port3: Final[Port] = Monitorout(qube = self)

        self.port4: Final[Port] = Monitorin(
            qube = self,
            adc = ADC(lsi = adc[0], ch = 2, ipfpga = ip, cpts = [e7.CaptureModule.U1,]),
            lo = LMX2594(lsi = lo[1]),
        )
        self.port4.adc.capt0._port = weakref.ref(self.port4)

        self.port5: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[0], ch = 2, ipfpga = ip, awgs = [(e7.AWG.U11, 4), (e7.AWG.U12, 3), (e7.AWG.U13, 2),]),
            lo = LMX2594(lsi = lo[2]),
            mix = ADRF6780(lsi = mix[2], ad5328 = AD5328(lsi = vatt, ch = 2)),
        )
        for p in self.port5.awgs.values():
            p._port = weakref.ref(self.port5)
        
        self.port6: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[0], ch = 3, ipfpga = ip, awgs = [(e7.AWG.U8, 5), (e7.AWG.U9, 6), (e7.AWG.U10, 7),]),
            lo = LMX2594(lsi = lo[3]),
            mix = ADRF6780(lsi = mix[3], ad5328 = AD5328(lsi = vatt, ch = 3)),
        )
        for p in self.port6.awgs.values():
            p._port = weakref.ref(self.port6)
        
        self.port7: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[1], ch = 0, ipfpga = ip, awgs = [(e7.AWG.U5, 2), (e7.AWG.U6, 1), (e7.AWG.U7, 0),]),
            lo = LMX2594(lsi = lo[4]),
            mix = ADRF6780(lsi = mix[4], ad5328 = AD5328(lsi = vatt, ch = 4)),
        )
        for p in self.port7.awgs.values():
            p._port = weakref.ref(self.port7)
        
        self.port8: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[1], ch = 1, ipfpga = ip, awgs = [(e7.AWG.U0, 5), (e7.AWG.U3, 4), (e7.AWG.U4, 3),]),
            lo = LMX2594(lsi = lo[5]),
            mix = ADRF6780(lsi = mix[5], ad5328 = AD5328(lsi = vatt, ch = 5)),
        )
        for p in self.port8.awgs.values():
            p._port = weakref.ref(self.port8)
        
        self.port9: Final[Port] = Monitorin(
            qube = self,
            adc = ADC(lsi = adc[1], ch = 2, ipfpga = ip, cpts = [e7.CaptureModule.U0,]),
            lo = LMX2594(lsi = lo[6]),
        )
        self.port9.adc.capt0._port = weakref.ref(self.port9)
        
        self.port10: Final[Port] = Monitorout(qube = self)
        
        self.port11: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[1], ch = 2, ipfpga = ip, awgs = [(e7.AWG.U1, 6),]),
            lo = LMX2594(lsi = lo[6]),
            mix = ADRF6780(lsi = mix[6], ad5328 = AD5328(lsi = vatt, ch = 6)),
        )
        for p in self.port11.awgs.values():
            p._port = weakref.ref(self.port11)
        
        self.port12: Final[Port] =  NotAvailable(qube = self)
        
        self.port13: Final[Port] = Ctrl(
            qube = self,
            dac = DAC(lsi = dac[1], ch = 3, ipfpga = ip, awgs = [(e7.AWG.U2, 7),]),
            lo = LMX2594(lsi = lo[7]),
            mix = ADRF6780(lsi = mix[7], ad5328 = AD5328(lsi = vatt, ch = 7)),
        )
        for p in self.port13.awgs.values():
            p._port = weakref.ref(self.port13)
        
        self.ctrl4: Final[Port] = self.port0
        self.ctrl5: Final[Port] = self.port2
        self.auxout0: Final[Port] = self.port3
        self.auxin0: Final[Port] = self.port4
        self.ctrl0: Final[Port] = self.port5
        self.ctrl1: Final[Port] = self.port6
        self.ctrl2: Final[Port] = self.port7
        self.ctrl3: Final[Port] = self.port8
        self.auxin1: Final[Port] = self.port9
        self.auxout1: Final[Port] = self.port10
        self.ctrl6: Final[Port] = self.port11
        self.ctrl7: Final[Port] = self.port13
