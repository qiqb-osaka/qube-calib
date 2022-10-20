import numpy as np


SAMPLING_RATE = 500000000


class Slot(object):
    def __init__(self, duration, amplitude=1, sampling_rate=SAMPLING_RATE):
        self.duration = duration
        self.amplitude = amplitude
        self.sampling_rate = sampling_rate
        
    @property
    def timestamp(self): # Get local timestamp in slot.
        d = self.duration
        s = self.sampling_rate
        return np.linspace(0, d, int(d * s * 1e-9), endpoint=False).astype(int)
    
    def combine(self, v):
        raise NotImplementedError('{}.combine method shuld be implemented.'.format(type(self)))
        return None
    
    
class SlotWithIQ(Slot):
    def combine(self, v):
        if isinstance(v, SlotWithIQ):
            o = Arbitrary(duration=self.duration, amplitude=self.amplitude, sampling_rate=self.sampling_rate)
            o.duration += v.duration
            sr, vr = o.sampling_rate, v.sampling_rate
            o.sampling_rate = sr if sr < vr else vr
            o.iq = np.append(self.iq, v.iq)
            return o
        else:
            raise ValueError('{}: Invalid combine object.'.format(self))

    
class Blank(Slot):
    def __init__(self, duration, sampling_rate=SAMPLING_RATE):
        super().__init__(duration=duration, amplitude=0, sampling_rate=sampling_rate)
        
    def combine(self, v):
        if isinstance(v, Blank):
            self.duration += v.duration
            sr, vr = self.sampling_rate, v.sampling_rate
            self.sampling_rate = sr if sr < vr else vr
            return self
        else:
            raise ValueError('{}: Invalid combine object.'.format(self))
        
        
class RxBlank(Slot):
    def __init__(self, duration, sampling_rate=SAMPLING_RATE):
        super().__init__(duration=duration, amplitude=0, sampling_rate=sampling_rate)

        
class Read(Slot):
    def __init__(self, duration, sampling_rate=SAMPLING_RATE):
        super().__init__(duration=duration, amplitude=0, sampling_rate=sampling_rate)
        
    def combine(self, v):
        if isinstance(v, Read):
            self.duration += v.duration
            sr, vr = self.sampling_rate, v.sampling_rate
            self.sampling_rate = sr if sr < vr else vr
            return self
        else:
            raise ValueError('{}: Invalid combine object.'.format(self))
            
        
class Arbitrary(SlotWithIQ):
    def __init__(self, duration, amplitude=1, sampling_rate=SAMPLING_RATE):
        super().__init__(duration=duration, amplitude=amplitude, sampling_rate=sampling_rate)
        self.iq = np.zeros(self.timestamp.shape).astype(complex)

Arbit = Arbitrary


class Rectangular(SlotWithIQ):
    def __init__(self, duration, amplitude=1, sampling_rate=SAMPLING_RATE):
        super().__init__(duration=duration, amplitude=amplitude, sampling_rate=sampling_rate)
        self.iq = np.ones(self.timestamp.shape).astype(complex)

Rect = Rectangular


class Channel(list):
    
    def __init__(self, center_frequency, wire=None, band_width=500e+6, *args, **kwargs):
        self.wire = wire
        self.center_frequency = center_frequency
        self.band_width = 500e+6
        
    def simplify(self):
        return self
    
    def __lshift__(self, slot):
        self.append(slot)
        return self
    
    def findall(self, klass):
        return list(filter(lambda x: isinstance(x, klass), self))
    
    @property
    def duration(self):
        # t = 0
        # for o in self:
        #     t += o.duration
        # return t
        return sum([o.duration for o in self])
    
    def get_offset(self, x):
        # i = self.index(x)
        # t = 0
        # for o in self[:i]:
        #     t += o.duration
        # return t
        return sum([o.duration for o in self[:self.index(x)]])
    
    def get_timestamp(self, x): # Get local timestamp in channel.
        return x.timestamp + self.get_offset(x)
    
    def renew(self):
        return Channel(self.center_frequency, self.wire, self.band_width)
    
    # Obsolete
    @property
    def band(self):
        fc = self.center_frequency
        for b in self.wire.band:
            if b.range[0] < fc and fc <= b.range[1]:
                return b
        raise ValueError('Invalid center_frequency {}.'.format(fc))

class RxChannel(Channel):
    
    def __init__(self, center_frequency, wire=None, *args, **kwargs):
        super().__init__(center_frequency, wire, *args, **kwargs)
        self.timestamp = None
        self.iq = None
        
class Schedule(dict):
    def __init__(self, offset=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset # [ns]
        
    def add_channel(self, key, center_frequency, wire=None):
        self[key] = Channel(center_frequency, wire)
        
    @property
    def timetable(self):
        return self

