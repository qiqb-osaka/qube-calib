import qubelsi.qube

class Qube(object):
    
    def __init__(self, addr=None, path=None):
        if addr is not None and path is not None:
            self.prepare(addr, path)
            
    def prepare(self, addr, path):
        self._qube = qubelsi.qube.Qube(addr, path)
        
    def do_init(self, rf_type='A', bitfile=None, message_out=False):
        self._qube.do_init(rf_type, bitfile, message_out)
        
    @property
    def path(self):
        return self._qube.path
    
    @property
    def ad9082(self):
        return self._qube.ad9082
    
    @property
    def lmx2594(self):
        return self._qube.lmx2594
    
    @property
    def lmx2594_ad9082(self):
        return self._qube.lmx2594_ad9082
    
    @property
    def adrf6780(self):
        return self._qube.adrf6780
    
    @property
    def ad5328(self):
        return self._qube.ad5328
    
    @property
    def gpio(self):
        return self._qube.gpio
    
    @property
    def bitfile(self):
        return self._qube.bitfile
    
    @property
    def rf_type(self):
        return self._qube.rf_type

