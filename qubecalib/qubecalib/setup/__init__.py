from typing import Final

class Port(object): # for Qube
    
    @property
    def logical_lines(self):
        return [LogicalLine() for i in range(3)]

class Setup(object): # qube module へ移動
    
    def __init__(self, line_connection):
        self.line = {l.destination: l.create() for l in line_connection}

class LineConnection(object):
    
    def __init__(self, destination: str, port: Port):
        self.destination: Final[str] = destination
        self.port: Final[Port] = port
        
    def create(self):
        return Line(self.destination, self.port)

class LineBase(object):
    pass

class Line(LineBase):
    
    def __init__(self, destination: str, port: Port):
        self.destination: Final[str] = destination
        self.port: Final[Port] = port
    
    @property
    def logical_lines(self):
        return self.port.logical_lines
    
class LogicalLineBase(object):
    pass

class LogicalLine(LogicalLineBase):
    
    def __init__(self):
        pass
    
