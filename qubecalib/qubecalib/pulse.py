import numpy as np
from .meas import AwgCtrl, CaptureCtrl

class LogicalWaveChunk(object):
    def __init__(self, duration, amplitude=32767, sampling_rate=AwgCtrl.SAMPLING_RATE):
        self.duration = duration
        self.amplitude = amplitude
        self.sampling_rate = sampling_rate
        
    @property
    def timestamp(self): # Get local timestamp in wave chunk.
        d = self.duration
        s = self.sampling_rate
        return np.linspace(0, d, int(d * s * 1e-9), endpoint=False).astype(int)
    
    def merge(self, v):
        raise NotImplementedError('{}.merge method shuld be implemented.'.format(type(self)))
        return None
    
    
class LogicalWaveChunkWithIQ(LogicalWaveChunk):
    def merge(self, v):
        if isinstance(v, LogicalWaveChunkWithIQ):
            o = Arbitrary(duration=self.duration, amplitude=self.amplitude, sampling_rate=self.sampling_rate)
            o.duration += v.duration
            sr, vr = o.sampling_rate, v.sampling_rate
            o.sampling_rate = sr if sr < vr else vr
            o.iq = np.append(self.iq, v.iq)
            return o
        else:
            raise ValueError('{}: Invalid merge object.'.format(self))

    
class Blank(LogicalWaveChunk):
    def __init__(self, duration, sampling_rate=AwgCtrl.SAMPLING_RATE):
        super().__init__(duration=duration, amplitude=0, sampling_rate=sampling_rate)
        
    def merge(self, v):
        if isinstance(v, Blank):
            self.duration += v.duration
            sr, vr = self.sampling_rate, v.sampling_rate
            self.sampling_rate = sr if sr < vr else vr
            return self
        else:
            raise ValueError('{}: Invalid merge object.'.format(self))
        
        
class RxBlank(LogicalWaveChunk):
    def __init__(self, duration, sampling_rate=AwgCtrl.SAMPLING_RATE):
        super().__init__(duration=duration, amplitude=0, sampling_rate=sampling_rate)

        
class Read(LogicalWaveChunk):
    def __init__(self, duration, sampling_rate=AwgCtrl.SAMPLING_RATE):
        super().__init__(duration=duration, amplitude=0, sampling_rate=sampling_rate)
        
    def merge(self, v):
        if isinstance(v, Read):
            self.duration += v.duration
            sr, vr = self.sampling_rate, v.sampling_rate
            self.sampling_rate = sr if sr < vr else vr
            return self
        else:
            raise ValueError('{}: Invalid merge object.'.format(self))
            
        
class Arbitrary(LogicalWaveChunkWithIQ):
    def __init__(self, duration, amplitude=32767, sampling_rate=AwgCtrl.SAMPLING_RATE):
        super().__init__(duration=duration, amplitude=amplitude, sampling_rate=sampling_rate)
        self.iq = np.zeros(self.timestamp.shape).astype(complex)

Arbit = Arbitrary


class Rectangular(LogicalWaveChunkWithIQ):
    def __init__(self, duration, amplitude=32767, sampling_rate=AwgCtrl.SAMPLING_RATE):
        super().__init__(duration=duration, amplitude=amplitude, sampling_rate=sampling_rate)
        self.iq = np.ones(self.timestamp.shape).astype(complex)

Rect = Rectangular


class Channel(list):
    
    def __init__(self, wire, center_frequency, band_width=500e+6, *args, **kwargs):
        self.wire = wire
        self.center_frequency = center_frequency
        self.band_width = 500e+6
        
    def simplify(self):
        return self
    
    def __lshift__(self, chunk):
        self.append(chunk)
        return self
    
    def findall(self, klass):
        return list(filter(lambda x: isinstance(x, klass), self))
    
    @property
    def duration(self):
        t = 0
        for o in self:
            t += o.duration
        return t
    
    def get_offset(self, x):
        i = self.index(x)
        t = 0
        for o in self[:i]:
            t += o.duration
        return t
    
    def get_timestamp(self, x): # Get local timestamp in channel.
        return x.timestamp + self.get_offset(x)
    
    # Obsolete
    @property
    def band(self):
        fc = self.center_frequency
        for b in self.wire.band:
            if b.range[0] < fc and fc <= b.range[1]:
                return b
        raise ValueError('Invalid center_frequency {}.'.format(fc))

class RxChannel(Channel):
    
    def __init__(self, wire, center_frequency, *args, **kwargs):
        super().__init__(wire, center_frequency, *args, **kwargs)
        self.timestamp = None
        self.iq = None
        
class Schedule(dict):
    def __init__(self, offset=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset # [ns]
        
    def add_channel(self, key, wire, center_frequency):
        self[key] = Channel(wire, center_frequency)
        
    @property
    def timetable(self):
        return self
