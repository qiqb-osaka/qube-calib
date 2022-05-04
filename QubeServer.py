# Copyright (C) 2007  Matthew Neeley
# Edited by Yutaka Tabuchi

"""
### BEGIN NODE INFO
[info]
name = QuBE DAQs
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 20
### END NODE INFO
"""

############################################################
#
# INDEX
#

"""
grep --color -nH --null -e "^\(class\|def\|  def\)" QubeServer.py
116:class QSConstants:
142:  def __init__(self):
145:class QSMessage:
156:  def __init__(self):
166:def pingger(host):
177:class QuBE_ControlLine(DeviceWrapper):
179:  def connect(self, *args, **kw ):
210:  def get_connected(self):
219:  def number_of_shots(self):
222:  def number_of_shots(self,value):
226:  def repetition_time(self):
229:  def repetition_time(self,value_in_ns):
234:  def sequence_length(self):
237:  def sequence_length(self,value):
240:  def get_lo_frequency(self):
243:  def set_lo_frequency(self,freq_in_mhz):
246:  def get_dac_coarse_frequency(self):
249:  def set_dac_coarse_frequency(self,freq_in_mhz):
254:  def static_get_dac_coarse_frequency(self,nco_ctrl,ch):
258:  def static_get_dac_coarse_ftw(self,nco_ctrl,ch):
267:  def static_check_lo_frequency(self,freq_in_mhz):
271:  def static_check_dac_coarse_frequency(self,freq_in_mhz):
275:  def static_check_dac_fine_frequency(self,freq_in_mhz):
279:  def static_check_value(self,value,resolution,multiplier=50):
285:  def static_check_repetition_time(self,reptime_in_nanosec):
289:  def static_check_sequence_length(self,seqlen_in_nanosec):
295:class QuBE_ReadoutLine(QuBE_ControlLine):
297:  def connect(self, *args, **kw ):
333:  def get_connected(self):
356:  def set_adc_coarse_frequency(self,freq_in_mhz):
361:  def get_adc_coarse_frequency(self):
364:  def static_get_adc_coarse_frequency(self,nco_ctrl,ch):
368:  def static_get_adc_coarse_ftw(self,nco_ctrl,ch):
377:  def static_check_adc_coarse_frequency(self,freq_in_mhz):
386:class QuBE_Server(DeviceServer):
394:  def initServer(self):
411:  def initContext(self, c):
414:  def chooseDeviceWrapper(self, *args, **kw):
418:  def instantiateChannel(self,name,channels,awg_ctrl,cap_ctrl,lsi_ctrl):
457:  def instantiateQube(self,name,info):
482:  def findDevices(self):
507:  def number_of_shots(self,c,num_shots = None):
516:  def repeat_count(self,c,repeat = None):
521:  def repetition_time(self,c,reptime = None):
532:  def sequence_length(self,c,length = None):
575:  def local_frequency(self,c,frequency = None):
587:  def coarse_tx_nco_frequency(self,c,frequency = None):
599:  def coarse_rx_nco_frequency(self,c,frequency = None):
619:def basic_config():
657:def load_config(cxn,config):
"""

############################################################
#
# LABRAD SERVER FOR QIQB QUBE UNITS
#   20220522 (draft) Yutaka Tabuchi
#

from labrad                 import types as T, util
from labrad.devices         import DeviceWrapper,   \
                                   DeviceServer
from labrad.server          import setting
from labrad.units           import Value
from twisted.internet.defer import inlineCallbacks, \
                                   returnValue
import sys
                                                            # import socket
import time
import numpy as np
import struct
import json
                                                            # from ftplib import FTP
from e7awgsw                import DspUnit, \
                                   AwgCtrl, \
                                   CaptureCtrl, \
                                   CaptureModule
import qubelsi.qube
import subprocess
import os
#import concurrent                                           # to be used by Suzuki-san
#from labrad.concurrent      import future_to_deferred       # to be used by Suzuki-san


############################################################
#
# CONSTANTS
#

class QSConstants:
  REGDIR             = ['', 'Servers', 'QuBE']
  REGSRV             = 'registry'
  REGLNK             = 'possible_links'
  REGAPIPATH         = 'adi_api_path'
  SRVNAME            = 'QuBE Server'
  THREAD_MAX_WORKERS = 32
  SEQ_MAXLEN         = 199936                               # nano-seconds -> 24,992 AWG Word
  SEQ_INITLEN        = 8192                                 # nano-seconds -> 1,024 AWG Word
  SEQ_INITREPTIME    = 30720                                # nano-seconds -> 3,840 AWG Word
  SEQ_INITSHOTS      = 1                                    # one shot
  ACQ_INITMODE       = '3'
  DAC_SAMPLE_R       = 12000                                # MHz
  ADC_SAMPLE_R       = 6000                                 # MHz
  DAQ_CNCO_BITS      = 48
  DAQ_LO_RESOL       = 100                                  # MHz
  DAC_CNCO_RESOL     = 12000/2**13                          # MHz; DAC_SAMPLE_R/2**13
  DAC_FNCO_RESOL     = 2000/2**12                           # MHz; DAC_SAMPLE_R/M=6/2**12
  ADC_CNCO_RESOL     = 1000/2**11                           # MHz; ADC_SAMPLE_R/M=6/2**11
  DAQ_REPT_RESOL     = 10240                                # nanoseconds
  DAQ_SEQL_RESOL     = 128                                  # nanoseconds
  #ACQ_INITMODE       = DspUnit.INTEGRATION
  ACQ_MAXWINDOW      = 2000                                 # nano-seconds
  ACQ_MODENUMBER     = {'1':0, '2':1, '3':2,'A':3,'B':4 }
  ACQ_INITWINDOW     = [(0,1024)]

  def __init__(self):
    pass

class QSMessage:
  CONNECTING_CHANNEL = 'connecting to {}'
  CHECKING_QUBEUNIT  = 'Checking {} ...'
  CNCTABLE_QUBEUNIT  = 'Link possible: {}'
  CONNECTED_CHANNEL  = 'Link : {}'

  ERR_HOST_NOTFOUND  = 'QuBE {} not found (ping unreachable)'
  ERR_FREQ_SETTING   = '{} accepts a frequency multiple of {} MHz'
  ERR_REP_SETTING    = '{} accepts a multiple of {} ns'
  ERR_INVALID_DEV    = 'Invalid device. You may have called {} specific API in {}'

  def __init__(self):
    pass


    
############################################################
#
# TOOLS
#

def pingger(host):
  cmd = "ping -c 1 -W 1 %s" % host
  with open(os.devnull,'w') as f:
    resp = subprocess.call(cmd.split(' '), stdout=f,stderr=subprocess.STDOUT )
  return resp

############################################################
#
# DEVICE WRAPPERS
#

class QuBE_ControlLine(DeviceWrapper):
  @inlineCallbacks
  def connect(self, *args, **kw ):

    name, awg_ctrl, awg_ch_ids, nco_device, cnco_id, fnco_ids, lo_device = args
    print(QSMessage.CONNECTING_CHANNEL.format(name))
    self.name = name
    self.awg_ctrl   = awg_ctrl
    self.nco_ctrl   = nco_device
    self.lo_ctrl    = lo_device
    self.awg_ch_ids = awg_ch_ids
    self.awg_chs    = len(awg_ch_ids)
    self.cnco_id    = cnco_id
    self.fnco_ids   = fnco_ids
    
    self._initialized = False
    try:
      self.lo_frequency     = self.get_lo_frequency()
      print(self.name,'local',self.lo_frequency)            # DEBUG
      self.coarse_frequency = self.get_dac_coarse_frequency()
      print(self.name,'nco',self.coarse_frequency)          # DEBUG
      self.fine_frequency   = [0,0,0]                       # Mz
      self._initialized     = True
    except Exception as e:
      print(e)

    if self._initialized:
      yield self.get_connected()
    
    print(QSMessage.CONNECTED_CHANNEL.format(self.name))
    yield

  @inlineCallbacks
  def get_connected(self):

    self.shots        = QSConstants.SEQ_INITSHOTS
    self.rep_time     = QSConstants.SEQ_INITREPTIME
    self.seqlen       = QSConstants.SEQ_INITLEN

    yield

  @property
  def number_of_shots(self):
    return self.shots
  @number_of_shots.setter
  def number_of_shots(self,value):
    self.shots = value

  @property
  def repetition_time(self):
    return self.rep_time
  @repetition_time.setter
  def repetition_time(self,value_in_ns):
    self.rep_time = ((value_in_ns+QSConstants.DAQ_REPT_RESOL/2)//QSConstants.DAQ_REPT_RESOL) \
                         *QSConstants.DAQ_REPT_RESOL

  @property
  def sequence_length(self):
    return self.seqlen
  @sequence_length.setter
  def sequence_length(self,value):
    self.seqlen = value

  def get_lo_frequency(self):
    return self.lo_ctrl.read_freq_100M()*100

  def set_lo_frequency(self,freq_in_mhz):
    return self.lo_ctrl.write_freq_100M(int(freq_in_mhz//100))

  def get_dac_coarse_frequency(self):
    return self.static_get_dac_coarse_frequency(self.nco_ctrl,self.cnco_id)

  def set_dac_coarse_frequency(self,freq_in_mhz):
    self.nco_ctrl.set_nco(1e6*freq_in_mhz, self.cnco_id, \
                                           adc_mode = False, fine_mode=False)
    self.coarse_frequency = freq_in_mhz
  
  def static_get_dac_coarse_frequency(self,nco_ctrl,ch):
    ftw = self.static_get_dac_coarse_ftw(nco_ctrl,ch)
    return QSConstants.DAC_SAMPLE_R / (2**QSConstants.DAQ_CNCO_BITS) * ftw

  def static_get_dac_coarse_ftw(self,nco_ctrl,ch):
    page = nco_ctrl.read_value(0x1b) & 0xff                 # dac mainpath page select
    nco_ctrl.write_value(0x1b, page & 0xf0 | (1 << ch) & 0x0f)
    ftw = 0                                                 # ftw stands for freuqyency tuning word
    for i in range(6):                                      #   freq is f_DAC/(2**48)*(ftw)
      res = nco_ctrl.read_value(0x1d0-i)
      ftw = (ftw << 8 | res)
    return ftw

  def static_check_lo_frequency(self,freq_in_mhz):
    resolution = QSConstants.DAQ_LO_RESOL
    return self.static_check_value(freq_in_mhz,resolution)

  def static_check_dac_coarse_frequency(self,freq_in_mhz):
    resolution = QSConstants.DAC_CNCO_RESOL
    return self.static_check_value(freq_in_mhz,resolution)

  def static_check_dac_fine_frequency(self,freq_in_mhz):
    resolution = QSConstants.DAC_FNCO_RESOL
    return self.static_check_value(freq_in_mhz,resolution)
  
  def static_check_value(self,value,resolution,multiplier=50):
    resp = False
    if resolution > multiplier * abs(((2*value + resolution) % (2*resolution)) - resolution):
      resp = True
    return resp

  def static_check_repetition_time(self,reptime_in_nanosec):
    resolution = QSConstants.DAQ_REPT_RESOL
    return self.static_check_value(reptime_in_nanosec,resolution)

  def static_check_sequence_length(self,seqlen_in_nanosec):
    resolution = QSConstants.DAQ_SEQL_RESOL
    return self.static_check_value(seqlen_in_nanosec,resolution)

  

class QuBE_ReadoutLine(QuBE_ControlLine):
  @inlineCallbacks
  def connect(self, *args, **kw ):
      
    name, awg_ctrl, awg_ch_ids, cap_ctrl, capture_units, \
      nco_device, cunco_id, funco_id, cdnco_id,lo_device = args
    print(QSMessage.CONNECTING_CHANNEL.format(name))
    self.name = name
    self.awg_ctrl = awg_ctrl
    self.cap_ctrl = cap_ctrl
    self.cap_unit = capture_units
    self.nco_ctrl = nco_device
    self.lo_ctrl  = lo_device
    self.awg_ch_id= awg_ch_ids
    self.cnco_id  = cunco_id
    self.fnco_ids = [funco_id]
    self.rxcnco_id= cdnco_id

    self._initialized = False
    try:
      self.lo_frequency     = self.get_lo_frequency()
      print(self.name,'local',self.lo_frequency)            # DEBUG
      self.coarse_frequency = self.get_dac_coarse_frequency()
      print(self.name,'nco',self.coarse_frequency)          # DEBUG
      self.rx_coarse_frequency = self.get_adc_coarse_frequency()
      print(self.name,'rxnco',self.rx_coarse_frequency)     # DEBUG
      self.fine_frequency   = [0]                           # Mz
      self._initialized = True
    except Exception  as e:
      print(e)
    
    if self._initialized:
      yield self.get_connected()

    print(QSMessage.CONNECTED_CHANNEL.format(self.name))
    yield

  @inlineCallbacks
  def get_connected(self):
    QuBE_ControlLine.get_connected(self)
                                                            # Capture default parameter settings
    self.window        = QSConstants.ACQ_INITWINDOW             
    self.acq_mode      = QSConstants.ACQ_INITMODE
    self.acq_n_windows = 1

                                                            # to be implemented later (from)
    # self.init_lpfcoef   = False
    # self.init_defwindow = False
    # try:
    #   self.set_filter_coef_pre(                            \
    #     np.array([[0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0],       \
    #               [1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0]]))
    #   self.set_filter_coef_post(np.array([[0,0,0,0,0,0,0,1]]))
    #   self.init_lpfcoef = True
    #   self.set_window_coef(np.vstack((np.ones(2048), np.ones(2048))))
    #   self.init_defwindow = True
    # except:
    #   print('Failed to set filter coefficients')
                                                            # to be implemented later (end)
    yield
    
  def set_adc_coarse_frequency(self,freq_in_mhz):
    self.nco_ctrl.set_nco(1e6*freq_in_mhz, self.rxcnco_id, \
                                           adc_mode = True, fine_mode=False)
    self.rxcoarse_frequency = freq_in_mhz

  def get_adc_coarse_frequency(self):
    return self.static_get_adc_coarse_frequency(self.nco_ctrl,self.rxcnco_id)

  def static_get_adc_coarse_frequency(self,nco_ctrl,ch):
    piw = self.static_get_adc_coarse_ftw(nco_ctrl,ch)
    return QSConstants.ADC_SAMPLE_R / (2**QSConstants.DAQ_CNCO_BITS) * piw

  def static_get_adc_coarse_ftw(self,nco_ctrl,ch):
    page = nco_ctrl.read_value(0x18) & 0xff                 # dac mainpath page select
    nco_ctrl.write_value(0x18, page & 0x0f | (16 << ch) & 0xf0)
    piw = 0                                                 # piw stands for phase incremental word
    for i in range(6):                                      #   freq is f_ADC/(2**48)*(piw)
      res = nco_ctrl.read_value(0xa0a-i)
      piw = (piw << 8 | res)
    return piw
    
  def static_check_adc_coarse_frequency(self,freq_in_mhz):
    resolution = QSConstants.ADC_CNCO_RESOL
    return self.static_check_value(freq_in_mhz,resolution)

############################################################
#
# QUBE SERVER
#

class QuBE_Server(DeviceServer):
  name          = QSConstants.SRVNAME
  deviceWrappers={'readout': QuBE_ReadoutLine,
                  'control': QuBE_ControlLine }
  possibleLinks = { }
  adi_api_path  = None

  @inlineCallbacks
  def initServer(self):
    yield DeviceServer.initServer(self)

    cxn = self.client
    reg = cxn[QSConstants.REGSRV]
    try:
      yield reg.cd(QSConstants.REGDIR)
      config = yield reg.get(QSConstants.REGLNK)      
      self.possibleLinks = json.loads(config)
      self.adi_api_path  = yield reg.get(QSConstants.REGAPIPATH)
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)

                                                            # reserved for Suzuki-san
                                                            #max_workers      = QSConstants.THREAD_MAX_WORKERS
                                                            #self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
  def initContext(self, c):
    yield DeviceServer.initContent(self)

  def chooseDeviceWrapper(self, *args, **kw):
    tag = 'readout' if 'readout' in args[0] else 'control'
    return self.deviceWrappers[tag]

  def instantiateChannel(self,name,channels,awg_ctrl,cap_ctrl,lsi_ctrl):
    def gen_awg(name,channel,awg_ctrl,lsi_ctrl):
      awg_ch_ids = channel[  'ch_dac']
      cnco_id    = channel['cnco_dac']
      fnco_id    = channel['fnco_dac']
      lo_id      = channel[  'lo_dac']
      nco_device = lsi_ctrl.ad9082[cnco_id[0]]
      lo_device  = lsi_ctrl.lmx2594[lo_id]

      args = name,awg_ctrl,awg_ch_ids,nco_device,cnco_id[1],[_id for _chip,_id in fnco_id],lo_device
      return (name,args)

    def gen_mux(name,channel,awg_ctrl,cap_ctrl,lsi_ctrl):
      awg_ch_ids = channel['ch_dac']
      cap_mod_id = channel['ch_adc']
      
      capture_units = CaptureModule.get_units(cap_mod_id)
      
      cunco_id   = channel['cnco_dac']
      cdnco_id   = channel['cnco_adc']
      funco_id   = channel['fnco_dac']
      lo_id      = channel[  'lo_daq']
      nco_device = lsi_ctrl.ad9082[cunco_id[0]]
      lo_device  = lsi_ctrl.lmx2594[lo_id]

      args = name,awg_ctrl,awg_ch_ids,cap_ctrl,capture_units,nco_device,cunco_id[1],funco_id[1],cdnco_id[1],lo_device
      return (name,args)

    devices = []
    for channel in channels:
      channel_type = channel['type']
      channel_name = name + '-' + channel['name']
      to_be_added = gen_awg(channel_name,channel,awg_ctrl,lsi_ctrl)          if channel_type == 'control' else \
                    gen_mux(channel_name,channel,awg_ctrl,cap_ctrl,lsi_ctrl) if channel_type == 'mux'     else \
                    None
      if to_be_added is not None:
        devices.append(to_be_added)
    return devices

  def instantiateQube(self,name,info):
    try:
      ipfpga = info['fpga']
      iplsi  = info[ 'lsi']
      channels = info['channels']
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)
      return list()

    try:
      awg_ctrl = AwgCtrl    (ipfpga)                        # AWG CONTROL (e7awgsw)
      cap_ctrl = CaptureCtrl(ipfpga)                        # CAP CONTROL (e7awgsw)
      lsi_ctrl = qubelsi.qube.Qube(iplsi, self.adi_api_path)# LSI CONTROL (qubelsi.qube)
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)
      return list()

    try:
      devices = self.instantiateChannel(name,channels,awg_ctrl,cap_ctrl,lsi_ctrl)
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)
      devices = list()
    return devices
    
  @inlineCallbacks
  def findDevices(self):
    cxn = self.client
    found = []

    for name in self.possibleLinks.keys():
      print(QSMessage.CHECKING_QUBEUNIT.format(name))
      try:
        res = pingger(self.possibleLinks[name]['fpga'])
        if 0 == res:
          res = pingger(self.possibleLinks[name]['lsi'])
        if 0 != res:
          raise Exception(QSMessage.ERR_HOST_NOTFOUND.format(name))
      except Exception as e:
        print(e)
        continue
    
      print(QSMessage.CNCTABLE_QUBEUNIT.format(name))
      devices = self.instantiateQube(name,self.possibleLinks[name])
      found.extend(devices)
      yield

    print(sys._getframe().f_code.co_name,found)             # DEBUG
    returnValue(found)

  @setting(900, 'Shots', num_shots = ['w'], returns=['w'])
  def number_of_shots(self,c,num_shots = None):
    dev = self.selectedDevice(c)
    if num_shots is not None:
      dev.number_of_shots = num_shots
      return num_shots
    else:
      return dev.number_of_shots

  @setting(401, 'Repeat Count', repeat = ['w'], returns=['w'])
  def repeat_count(self,c,repeat = None):
    raise Exception('obsoleted. use "shots" instead')
    return self.number_of_shots(c,repeat)
    
  @setting(41, 'Repetition Time', reptime = ['v[s]'], returns=['v[s]'])
  def repetition_time(self,c,reptime = None):
    dev = self.selectedDevice(c)
    if reptime is None:
      return T.Value(dev.repetition_time,'ns')
    elif dev.static_check_repetition_time(reptime['ns']):
      dev.repetition_time = int(round(reptime['ns']))
      return reptime
    else:
      raise ValueError(QSMessage.ERR_REP_SETTING.format('Sequencer',QSConstants.DAQ_REPT_RESOL))

  @setting(45, 'DAQ Length', length = ['v[s]'], returns = ['v[s]'])
  def sequence_length(self,c,length = None):
    dev = self.selectedDevice(c)
    if length is None:
      return Value(dev.sequence_length,'ns')
    elif dev.static_check_sequence_length(length['ns']):
      dev.sequence_length = int(length['ns'])
      return length
    else:
      raise ValueError(QSMessage.ERR_REP_SETTING.format('Sequencer',QSConstants.DAQ_SEQL_RESOL))

#  @setting(52, 'Acquisition Count', acqcount = ['w'], returns = ['w'])
#  def acquisition_count(self,c,acqcount = None):
#    dev = self.selectedDevice(c)
#    if acqcount is not None:
#      dev.set_acq_count(acqcount)
#      return acqcount
#    else:
#      return dev.get_acq_count()
#
#  @setting(42, 'Acquisition Window', window = ['*(v[s]v[s])'], returns=['*(v[s]v[s])'])
#  def acquisition_window(self,c,window = None):
#    dev = self.selectedDevice(c)
#    if window is not None:
#      wl = list()
#      for w in window:
#        wl.append((w[0]['ns'],w[1]['ns']))
#      dev.set_acq_window(wl)
#    wl = dev.get_acq_window()
#    window = list()
#    for w in wl:
#      window.append((Value(w[0],'ns'),Value(w[1],'ns')))
#    return window
#
#  @setting(43, 'Acquisition Mode', mode = ['s'], returns=['s'])
#  def acquisition_mode(self,c,mode = None):
#    dev = self.selectedDevice(c)
#    if mode is not None:
#      dev.set_acq_mode(mode)
#    return dev.get_acq_mode()

  

  @setting(100, 'Frequency Local', frequency = ['v[Hz]'], returns = ['v[Hz]'])
  def local_frequency(self,c,frequency = None):
    dev = self.selectedDevice(c)
    if frequency is None:
      resp = dev.get_lo_frequency()
      frequency = T.Value(resp,'MHz')
    elif dev.static_check_lo_frequency(freq_in_mhz):
      dev.set_lo_frequency(frequency['MHz'])
    else:
      raise ValueError(QSMessage.ERR_FREQ_SETTING.format('LO',QSConstants.DAC_CNCO_RESOL))
    return frequency

  @setting(201, 'Frequency TX NCO', frequency = ['v[Hz]'], returns = ['v[Hz]'])
  def coarse_tx_nco_frequency(self,c,frequency = None):
    dev = self.selectedDevice(c)
    if frequency is None:
      resp = dev.get_dac_coarse_frequency()
      frequency = T.Value(resp,'MHz')
    elif dev.static_check_dac_coarse_frequency(frequency['MHz']):
      dev.set_dac_coarse_frequency(frequency['MHz'])
    else:
      raise ValueError(QSMessage.ERR_FREQ_SETTING.format('TX Corse NCO',QSConstants.DAC_CNCO_RESOL))
    return frequency
  
  @setting(202, 'Frequency RX NCO', frequency = ['v[Hz]'], returns = ['v[Hz]'])
  def coarse_rx_nco_frequency(self,c,frequency = None):
    dev = self.selectedDevice(c)
    if 'readout' not in dev.name:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.name))
    elif frequency is None:
      resp = dev.get_adc_coarse_frequency()
      frequency = T.Value(resp,'MHz')
    elif dev.static_check_adc_coarse_frequency(frequency['MHz']):
      dev.set_adc_coarse_frequency(frequency['MHz'])
    else:
      raise ValueError(QSMessage.ERR_FREQ_SETTING.format('RX Corse NCO',QSConstants.ADC_CNCO_RESOL))
    return frequency

############################################################
#
# AUX SUBROUTINES FOR EASY SETUP
#

import json

def basic_config():
  
  readout_control_qube = \
  [
    {'name'    : 'readout_01',
     'type'    : 'mux',
     'ch_dac'  : 15,                                          # awg id
     'ch_adc'  : 1,                                           # module id
     'cnco_dac' : (0,0),                                      # chip, main path
     'cnco_adc' : (0,3),                                      # chip, main path
     'fnco_dac' : (0,0),                                      # chip, link id
     'lo_daq'   : 0,                                          # local oscillator id
    },
    {'name'    : 'readout_cd',
     'type'    : 'mux',
     'ch_dac'  : 2,                                           # awg id
     'ch_adc'  : 0,                                           # module id
     'cnco_dac' : (1,3),                                      # chip, main path
     'cnco_adc' : (1,3),                                      # chip, main path
     'fnco_dac' : (0,7),                                      # chip, link no
     'lo_daq'   : 7,                                          # local oscillator id
    },
    {'name'    : 'control_5',
     'type'    : 'control',
     'ch_dac'  : [11,12,13],                                  # awg id
     'cnco_dac' : (0,2),                                      # chip, main path id
     'fnco_dac' : [(0,2),(0,3),(0,4)],                        # chip, link no
     'lo_dac'   : 2,                                          # local oscillator id
    },
   ]    
  
  servers = \
   {
     'qube004': { 'fpga': '10.1.0.22', 'lsi': '10.5.0.22', 'channels' : readout_control_qube },
     'qube005': { 'fpga': '10.1.0.23', 'lsi': '10.5.0.23', 'channels' : readout_control_qube },
   }
  return json.dumps(servers)   

def load_config(cxn,config):
  reg = cxn[QSConstants.REGSRV]
  try:
    reg.cd(QSConstants.REGDIR)
    if isinstance(config,str):
      reg.set(QSConstants.REGLNK,config)
    else:
      raise TypeError(config)
  except Exception as e:
    print(sys._getframe().f_code.co_name,e)


############################################################
#
# SERVER WORKER
#

__server__ = QuBE_Server()

if __name__ == '__main__':
                                                            #  # Import Psyco if available
                                                            #  try:
                                                            #    import psyco
                                                            #    psyco.full()
                                                            #  except ImportError:
                                                            #    pass
                                                            #  print sys.argv
                                                            #  if sys.argv:
                                                            #    del sys.argv[1:]
  util.runServer(__server__)

    
