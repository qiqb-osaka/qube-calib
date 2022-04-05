import pyvisa
import time
import numpy as np

m = pyvisa.ResourceManager('@py')
inst = m.open_resource("TCPIP0::10.250.0.10::inst0::INSTR")
inst.query('*IDN?')

def new_resource_manager(interface_type='@py'):
    return pyvisa.ResourceManager(interface_type)

class MS2038_SPA(object):
    def __init__(self, manager, resource='TCPIP0::10.250.0.10::inst0::INSTR'):
        self.instrument = manager.open_resource(resource)
    def query(self, v):
        return self.instrument.query(v)
    def write(self, v):
        return self.instrument.write(v)
    def initiate(self):
        o = self.instrument
        o.write(':INIT')
        while True:
            time.sleep(.1)
            if o.query(':STAT:OPER?') == '256':
                break
        return True
    def get_preamble(self):
        o = self.instrument
        b = o.timeout
        o.timeout = 10000
        r = o.query(':TRAC:PRE?')
        o.timeout = b
        return r
    def get_trace(self):
        return self.instrument.query(':TRAC?')
    def convert_trace_to_array(self, trace):
        return np.loadtxt(trace[6:].split(','))
    def extract_frequency_array_from_preamble(self, preamble):
        myfind = lambda k, l: next(filter(lambda x: x.startswith("{}=".format(k)), l))
        myconv = lambda s: float(s.split('=')[1].replace(' MHZ', ''))
        p = preamble[10:].split(',')
        fstart = myconv(myfind('START_FREQ', p))
        fstop = myconv(myfind('STOP_FREQ', p))
        npts = int(float(myfind('UI_DATA_POINTS', p).split('=')[1]))
        return np.linspace(fstart, fstop, npts)
    def measure(self):
        self.initiate()
        t = self.get_trace()
        p = self.get_preamble()
        return {
            'trace': t,
            'preamble': p,
            'numpy_array': np.array([
                self.extract_frequency_array_from_preamble(p),
                self.convert_trace_to_array(t),
            ]),
        }

# # MS2038Cから測定データを読み出す
# def get_freq_from_ms2038c(inst):
#     myfind = lambda k, l: next(filter(lambda x: x.startswith("{}=".format(k)), l))
#     myconv = lambda s: float(s.split('=')[1].replace(' MHZ', ''))
#     inst.timeout = 10000
#     preamble = inst.query(':TRAC:PRE?')[10:].split(',')
#     fstart = myconv(myfind('START_FREQ', preamble))
#     fstop = myconv(myfind('STOP_FREQ', preamble))
#     npts = int(float(myfind('UI_DATA_POINTS', preamble).split('=')[1]))
#     return np.linspace(fstart, fstop, npts)

# def get_trace_from_ms2038c(inst):
#     return np.loadtxt(inst.query(':TRAC?')[6:].split(','))

# def sweep_ms2038c(inst):
#     inst.write(':INIT')
#     while True:
#         time.sleep(.1)
#         if inst.query(':STAT:OPER?') == '256':
#             break
#     return True
