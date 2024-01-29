import time
from collections import namedtuple

import numpy as np
import pyvisa


def new_resource_manager(interface_type="@py"):
    return pyvisa.ResourceManager(interface_type)


class MS2038_SPA(object):
    Contents = namedtuple("Contents", ("preamble", "trace"))

    def __init__(self, manager, resource="TCPIP0::10.250.0.10::inst0::INSTR"):
        self.instrument = manager.open_resource(resource)

    def query(self, v):
        return self.instrument.query(v)

    def write(self, v):
        return self.instrument.write(v)

    def wait(self):
        while True:
            try:
                self.query(":STAT:OPER?")
            except pyvisa.VisaIOError:
                pass
            else:
                break

    def initiate(self):
        o = self.instrument
        o.write(":INIT")
        while True:
            time.sleep(0.1)
            if o.query(":STAT:OPER?") == "256":
                break
        return True

    def get_preamble(self):
        o = self.instrument
        b = o.timeout
        o.timeout = 10000
        r = o.query(":TRAC:PRE?")
        o.timeout = b
        return r

    def get_trace(self):
        return self.instrument.query(":TRAC?")

    def measure(self):
        self.initiate()
        return MS2038_SPA.Contents(
            trace=self.get_trace(),
            preamble=self.get_preamble(),
        )

    @classmethod
    def convert_dict_to_contents(cls, dct):
        return MS2038_SPA.Contents(trace=dct["trace"], preamble=dct[""])

    @classmethod
    def convert_trace_to_array(cls, trace):
        return np.loadtxt(trace[6:].split(","))

    @classmethod
    def extract_frequency_array_from_preamble(cls, preamble):
        myfind = lambda k, l: next(filter(lambda x: x.startswith("{}=".format(k)), l))
        myconv = lambda s: float(s.split("=")[1].replace(" MHZ", ""))
        p = preamble[10:].split(",")
        fstart = myconv(myfind("START_FREQ", p))
        fstop = myconv(myfind("STOP_FREQ", p))
        npts = int(float(myfind("UI_DATA_POINTS", p).split("=")[1]))
        return np.linspace(fstart, fstop, npts)

    @classmethod
    def new_array_freq_trace_pair(cls, contents, old_format=False):
        if old_format:
            return np.array(
                [
                    MS2038_SPA.extract_frequency_array_from_preamble(
                        contents["preamble"]
                    ),
                    MS2038_SPA.convert_trace_to_array(contents["trace"]),
                ]
            )
        else:
            return np.array(
                [
                    MS2038_SPA.extract_frequency_array_from_preamble(contents.preamble),
                    MS2038_SPA.convert_trace_to_array(contents.trace),
                ]
            )
