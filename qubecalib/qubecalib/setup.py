class WireBase(object):
    pass

class Connected(object):
    @classmethod
    def to(self, port): # Read を検出したら _RxWire にするようロジックを変更 setupqube へ移動
        try:
            port.awgs
        except AttributeError:
            o = _RxWire(port, [_RxBand(port.capt)])
        else:
            o = _Wire(port, [_TxBand(v) for k, v in port.awgs.items()])
        return o
        
class _Wire(object):
    def __init__(self, port, band):
        self.port = port
        self.band = band
        self.cnco_mhz = 2000
        
class _RxWire(_Wire):
    def __init__(self, port, band, delay=0):
        super().__init__(port, band)
        self.delay = delay # [ns]
        
class _Band(object):
    pass

class _TxBand(_Band):
    def __init__(self, awg):
        self.awg = awg
        self.range = (7000e+6, 9000e+6)
        self.fnco_mhz = 0
        
class _RxBand(_Band):
    def __init__(self, capt):
        self.capt = capt
        self.range = (10000e+6, 11000e+6)
        self.fnco_mhz = 0
        