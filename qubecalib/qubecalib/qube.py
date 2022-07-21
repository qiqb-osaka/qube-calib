from . import alias

from abc import ABC, abstractmethod
from enum import Enum, auto
import yaml
import math
import warnings
import traceback
import qubelsi.qube
from e7awgsw import AWG, CaptureModule
import e7awgsw

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
        
    def _status(self):
        
        return {
            'nco': self.nco.status,
        }
        
        
class CPT(AD9082):
    
    def __init__(self, lsi, cptm, ipfpga):
        super().__init__(lsi)
        self.id = cptm
        self.ipfpga = ipfpga
        
    def _status(self):
        
        return None
        
        
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
    
    @abstractmethod
    def __init__(self):
        
        pass
        
    @abstractmethod
    def _status(self):
        
        pass
    
    @property
    def status(self):
        
        return self._status()
        
        
class Output(Port):
    
    def __init__(self, dac, lo, mix):
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
    
    def __init__(self, adc, lo):
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
    
    def __init__(self):
        self._rf = RF(self._calc_rf)
    
    def _calc_rf(self):
        
        return None
        
class NotAvailable(Port):
    
    def __init__(self):
        self._rf = RF(self._calc_rf)
        
    def _calc_rf(self):
        
        return None
    
        
class Ctrl(Output):
    
    def __init__(self, dac, lo, mix):
        super().__init__(dac, lo, mix)
        self.mix.ssb = SSB.LSB


class Readout(Output):
    
    def __init__(self, dac, lo, mix):
        super().__init__(dac, lo, mix)
        self.mix.ssb = SSB.USB
    
    
class Pump(Output):
    
    def __init__(self, dac, lo, mix):
        super().__init__(dac, lo, mix)
        self.mix.ssb = SSB.USB
    
    def _calc_rf(self):
        
        rst = super()._calc_rf()
        
        return {k: v * 2 for k, v in rst.items()}
    
class Readin(Input):
    
    def __init__(self, adc, lo):
        super().__init__(adc, lo)
    
    
class Monitorin(Input):
    
    def __init__(self, adc, lo):
        super().__init__(adc, lo)
    

class ConfigFPGA(object):
    PATH_TO_BITFILE: str = "/home/qube/bin"
    PATH_TO_API: str = "./adi_api_mod"
    
    @classmethod
    def config(cls, bitfile: str) -> None:
        os.environ['BITFILE'] = '{}/{}'.format(cls.PATH_TO_BITFILE, bitfile)
        commands = ["vivado", "-mode", "batch", "-source", "{}/utils/config.tcl".format(cls.PATH_TO_API)]
        ret = subprocess.check_output(commands , encoding='utf-8')
        return ret
        

class Qube(object): # QubeInstanceFactory
    '''
    config ファイルを読んで適切な QubeBase インスタンスを生成する
    qube = Qube.create(<config_file_name>)
    '''

    PATH_TO_CONFIG = './.config'
    PATH_TO_BITFILE = '/home/qube/bin'
    PATH_TO_API = './adi_api_mod'

    
    @classmethod
    def load(cls, config_file_name):
        
        name = '{}/{}'.format(cls.PATH_TO_CONFIG, config_file_name)
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
            QubeA.PATH_TO_API = Qube.PATH_TO_API
            return QubeA(o['iplsi'], Qube.PATH_TO_API, o)
        
        if o['type'] == 'B':
            QubeB.PATH_TO_API = Qube.PATH_TO_API
            return QubeB(o['iplsi'], Qube.PATH_TO_API, o)
        
        
class QubeBase(qubelsi.qube.Qube):
    '''
    任意の IPADDR で設定する
    qube = QubeA(ipaddr, path_to_api) | QubeB(ipaddr, path_to_api)
    '''
    
    PATH_TO_BITFILE = '/home/qube/bin'
    PATH_TO_API = './adi_api_mod'
    
    def __init__(self, addr, path, config=None):
        
        super().__init__(addr, path)
        self._config = config
        
    def __getitem__(self, v):
        
        return self._config[v]
        
    def config_fpga(self, bitfile=None):
        
        if bitfile is None and not 'bitfile' in self:
            raise ValueError('Specify bitfile.')
        
        if bitfile is None:
            bitfile = self['bitfile']
            
        self._config_fpga(bitfile)
            
    def _config_fpga(self, bitfile):
        
        os.environ['BITFILE'] = '{}/{}'.format(self.PATH_TO_BITFILE, bitfile)
        commands = ["vivado", "-mode", "batch", "-source", "{}/utils/config.tcl".format(self.PATH_TO_API)]
        ret = subprocess.check_output(commands , encoding='utf-8')
        return ret
    
    @property
    def are_ad9082s_connected_normally(self):
        ad9082s = self.ad9082
        return [dict(o.get_jesd_status())['0x55E'] == '0xE0' for o in ad9082s] == [True, True]

    def restart_ad9082s(self):
        ad9082s = self.ad9082
        for i in range(100):
            print(i+1, end=' ', flush=True)
            for o in ad9082s:
                o.do_init(message_out=False)
                print(dict(o.get_jesd_status())['0x55E'], end=' ', flush=True)
            if self.are_ad9082s_connected_normally():
                break

    @property
    def config(self):
        
        return self._config
        
    @property
    def ports(self):
        
        k = lambda i: 'port{}'.format(i)
        p = {k(i): getattr(self, k(i)) for i in range(self['nports'])}
        return p
        
    @property
    def port0(self):
        return self._port0

    @property
    def port1(self):
        return self._port1

    @property
    def port2(self):
        return self._port2

    @property
    def port3(self):
        return self._port3

    @property
    def port4(self):
        return self._port4

    @property
    def port5(self):
        return self._port5

    @property
    def port6(self):
        return self._port6

    @property
    def port7(self):
        return self._port7

    @property
    def port8(self):
        return self._port8

    @property
    def port9(self):
        return self._port9

    @property
    def port10(self):
        return self._port10

    @property
    def port11(self):
        return self._port11

    @property
    def port12(self):
        return self._port12

    @property
    def port13(self):
        return self._port13

class QubeA(QubeBase):
    def __init__(self, addr, path, config):
        super().__init__(addr, path, config)
        self._config['nports'] = 14
        ip = self['ipfpga']
        dac = self.ad9082
        adc = self.ad9082
        lo = self.lmx2594
        mix = self.adrf6780
        vatt = self.ad5328
        e7 = e7awgsw
        
        self._port0 = Readout(
            dac = DAC(lsi = dac[0], ch = 0, ipfpga = ip, awgs = [(e7.AWG.U15, 0),]),
            lo = LMX2594(lsi = lo[0]),
            mix = ADRF6780(lsi = mix[0], ad5328 = AD5328(lsi = vatt, ch = 0)),
        )
        
        self._port1 = Readin(
            adc = ADC(lsi = adc[0], ch = 3, ipfpga = ip, cpts = [CaptureModule.U1,]),
            lo = LMX2594(lsi = lo[0]),
        )

        self._port2 = Pump(
            dac = DAC(lsi = dac[0], ch = 1, ipfpga = ip, awgs = [(e7.AWG.U4, 1),]),
            lo = LMX2594(lsi = lo[1]),
            mix = ADRF6780(lsi = mix[1], ad5328 = AD5328(lsi = vatt, ch = 1)),
        )
        
        self._port3 = Monitorin(
            adc = ADC(lsi = adc[0], ch = 2, ipfpga = ip, cpts = [e7.CaptureModule.U1,]),
            lo = LMX2594(lsi = lo[1]),
        )
        
        self._port4 = Monitorout()

        self._port5 = Ctrl(
            dac = DAC(lsi = dac[0], ch = 2, ipfpga = ip, awgs = [(e7.AWG.U11, 2),(e7.AWG.U12, 3),(e7.AWG.U13, 4),]),
            lo = LMX2594(lsi = lo[2]),
            mix = ADRF6780(lsi = mix[2], ad5328 = AD5328(lsi = vatt, ch = 2)),
        )
        
        self._port6 = Ctrl(
            dac = DAC(lsi = dac[0], ch = 3, ipfpga = ip, awgs = [(e7.AWG.U8, 5),(e7.AWG.U9, 6),(e7.AWG.U10, 7),]),
            lo = LMX2594(lsi = lo[3]),
            mix = ADRF6780(lsi = mix[3], ad5328 = AD5328(lsi = vatt, ch = 3)),
        )
        
        self._port7 = Ctrl(
            dac = DAC(lsi = dac[1], ch = 0, ipfpga = ip, awgs = [(e7.AWG.U5, 0),(e7.AWG.U6, 1),(e7.AWG.U7, 2),]),
            lo = LMX2594(lsi = lo[4]),
            mix = ADRF6780(lsi = mix[4], ad5328 = AD5328(lsi = vatt, ch = 4)),
        )
        
        self._port8 = Ctrl(
            dac = DAC(lsi = dac[1], ch = 1, ipfpga = ip, awgs = [(e7.AWG.U0, 3),(e7.AWG.U3, 4),(e7.AWG.U4, 5),]),
            lo = LMX2594(lsi = lo[5]),
            mix = ADRF6780(lsi = mix[5], ad5328 = AD5328(lsi = vatt, ch = 5)),
        )
        
        self._port9 = Monitorout()

        self._port10 = Monitorin(
            adc = ADC(lsi = adc[1], ch = 2, ipfpga = ip, cpts = [CaptureModule.U0,]),
            lo = LMX2594(lsi = lo[6]),
        )
        
        self._port11 = Pump(
            dac = DAC(lsi = dac[1], ch = 2, ipfpga = ip, awgs = [(e7.AWG.U1, 6),]),
            lo = LMX2594(lsi = lo[6]),
            mix = ADRF6780(lsi = mix[6], ad5328 = AD5328(lsi = vatt, ch = 6)),
        )
        
        self._port12 = Readin(
            adc = ADC(lsi = adc[1], ch = 3, ipfpga = ip, cpts = [CaptureModule.U0,]),
            lo = LMX2594(lsi = lo[7]),
        )
        
        self._port13 = Readout(
            dac = DAC(lsi = dac[1], ch = 3, ipfpga = ip, awgs = [(e7.AWG.U2, 7),]),
            lo = LMX2594(lsi = lo[7]),
            mix = ADRF6780(lsi = mix[7], ad5328 = AD5328(lsi = vatt, ch = 7)),
        )
        
    @property
    def readout0(self):
        return self._port0

    @property
    def readin0(self):
        return self._port1

    @property
    def pump0(self):
        return self._port2

    @property
    def auxin0(self):
        return self._port3

    @property
    def auxout0(self):
        return self._port4

    @property
    def ctrl0(self):
        return self._port5

    @property
    def ctrl1(self):
        return self._port6

    @property
    def ctrl2(self):
        return self._port7

    @property
    def ctrl3(self):
        return self._port8

    @property
    def auxout1(self):
        return self._port9

    @property
    def auxin1(self):
        return self._port10

    @property
    def pump1(self):
        return self._port11

    @property
    def readin1(self):
        return self._port12

    @property
    def readout1(self):
        return self._port13

        
class QubeB(QubeBase):
    def __init__(self, addr, path, config):
        super().__init__(addr, path, config)
        self._config['nports'] = 14
        ip = self['ipfpga']
        dac = self.ad9082
        adc = self.ad9082
        lo = self.lmx2594
        mix = self.adrf6780
        vatt = self.ad5328
        e7 = e7awgsw
        
        self._port0 = Ctrl(
            dac = DAC(lsi = dac[0], ch = 0, ipfpga = ip, awgs = [(e7.AWG.U15, 0),]),
            lo = LMX2594(lsi = lo[0]),
            mix = ADRF6780(lsi = mix[0], ad5328 = AD5328(lsi = vatt, ch = 0)),
        )
        
        self._port1 = NotAvailable()

        self._port2 = Ctrl(
            dac = DAC(lsi = dac[0], ch = 1, ipfpga = ip, awgs = [(e7.AWG.U14, 1),]),
            lo = LMX2594(lsi = lo[1]),
            mix = ADRF6780(lsi = mix[1], ad5328 = AD5328(lsi = vatt, ch = 1)),
        )
        
        self._port3 = Monitorin(
            adc = ADC(lsi = adc[0], ch = 2, ipfpga = ip, cpts = [CaptureModule.U1,]),
            lo = LMX2594(lsi = lo[1]),
        )
        
        self._port4 = Monitorout()

        self._port5 = Ctrl(
            dac = DAC(lsi = dac[0], ch = 2, ipfpga = ip, awgs = [(e7.AWG.U11, 2), (e7.AWG.U12, 3), (e7.AWG.U13, 4),]),
            lo = LMX2594(lsi = lo[2]),
            mix = ADRF6780(lsi = mix[2], ad5328 = AD5328(lsi = vatt, ch = 2)),
        )
        
        self._port6 = Ctrl(
            dac = DAC(lsi = dac[0], ch = 3, ipfpga = ip, awgs = [(e7.AWG.U8, 5), (e7.AWG.U9, 6), (e7.AWG.U10, 7),]),
            lo = LMX2594(lsi = lo[3]),
            mix = ADRF6780(lsi = mix[3], ad5328 = AD5328(lsi = vatt, ch = 3)),
        )
        
        self._port7 = Ctrl(
            dac = DAC(lsi = dac[1], ch = 0, ipfpga = ip, awgs = [(e7.AWG.U5, 0), (e7.AWG.U6, 1), (e7.AWG.U7, 2),]),
            lo = LMX2594(lsi = lo[4]),
            mix = ADRF6780(lsi = mix[4], ad5328 = AD5328(lsi = vatt, ch = 4)),
        )
        
        self._port8 = Ctrl(
            dac = DAC(lsi = dac[1], ch = 1, ipfpga = ip, awgs = [(e7.AWG.U0, 3), (e7.AWG.U3, 4), (e7.AWG.U4, 5),]),
            lo = LMX2594(lsi = lo[5]),
            mix = ADRF6780(lsi = mix[5], ad5328 = AD5328(lsi = vatt, ch = 5)),
        )
        
        self._port9 = Monitorout()

        self._port10 = Monitorin(
            adc = ADC(lsi = adc[1], ch = 2, ipfpga = ip, cpts = [CaptureModule.U0,]),
            lo = LMX2594(lsi = lo[6]),
        )
        
        self._port11 = Ctrl(
            dac = DAC(lsi = dac[1], ch = 2, ipfpga = ip, awgs = [(e7.AWG.U1, 6),]),
            lo = LMX2594(lsi = lo[6]),
            mix = ADRF6780(lsi = mix[6], ad5328 = AD5328(lsi = vatt, ch = 6)),
        )
        
        self._port12 =  NotAvailable()
        
        self._port13 = Ctrl(
            dac = DAC(lsi = dac[1], ch = 3, ipfpga = ip, awgs = [(e7.AWG.U2, 7),]),
            lo = LMX2594(lsi = lo[7]),
            mix = ADRF6780(lsi = mix[7], ad5328 = AD5328(lsi = vatt, ch = 7)),
        )
        
    @property
    def ctrl4(self):
        return self._port0

    @property
    def ctrl5(self):
        return self._port2

    @property
    def auxin0(self):
        return self._port3

    @property
    def auxout0(self):
        return self._port4

    @property
    def ctrl0(self):
        return self._port5

    @property
    def ctrl1(self):
        return self._port6

    @property
    def ctrl2(self):
        return self._port7

    @property
    def ctrl3(self):
        return self._port8

    @property
    def auxout1(self):
        return self._port9

    @property
    def auxin1(self):
        return self._port10

    @property
    def ctrl6(self):
        return self._port11

    @property
    def ctrl7(self):
        return self._port13

