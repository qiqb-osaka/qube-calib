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
  SEQ_MAXLEN         = 200*1000                             # nano-seconds
  SEQ_INITLEN        = 8192                                 # nano-seconds
  SEQ_INITREPTIME    = 40960                                # nano-seconds -> 320 JESD blocks
  SEQ_INITSHOTS      = 1                                    # one shot
  SEQ_INIT_ACQMODE   = DspUnit.INTEGRATION
  ACQ_MAXWINDOW      = 2000                                 # nano-seconds

  def __init__(self):
    pass

class QSMessage:
  CONNECTING_CHANNEL = 'connecting to {}'
  CHECKING_QUBEUNIT  = 'Checking {} ...'
  CNCTABLE_QUBEUNIT  = 'Link possible: {}'

  ERR_HOST_NOTFOUND  = 'QuBE {} not found (ping unreachable)'

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

class QuBE_ReadoutLine(DeviceWrapper):
  @inlineCallbacks
  #def connect(self, name, server):
  def connect(self, *args, **kw ):
      
    #print(QSMessage.CONNECTING_CHANNEL.format(name))
    yield

    
class QuBE_ControlLine(DeviceWrapper):
  @inlineCallbacks
  #def connect(self, name, server):
  def connect(self, *args, **kw ):

    name, awg_ctrl, awg_ch_ids, nco_ctrl, cnco_id, fnco_ids = args
    self.name = name
    self.awg_ctrl = awg_ctrl
    self.nco_ctrl = nco_ctrl
    self.awg_ch_ids = awg_ch_ids
    self.awg_chs    = len(awg_ch_ids)
    self.cnco_id    = cnch_id
    self.fnco_ids   = fnco_ids
    print(QSMessage.CONNECTING_CHANNEL.format(name))

    

    yield
          
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
      awg_ch_ids = channel['ch_dac']
      cnco_id    = channel['cnco_dac']
      fnco_id    = channel['fnco_dac']
      nco_device = lsi_ctrl.ad9082[cnco_id[0]]
      args = name,awg_ctrl,awg_ch_ids,nco_device,cnco_id[0],[_id for _chip,_id in fnco_id]
      return (name,args)
    def gen_mux(name,channel,awg_ctrl,cap_ctrl,lsi_ctrl):
      awg_ch_ids = channel['ch_dac']
      cap_mod_id = channel['ch_adc']
      
      capture_units = CaptureModule.get_units(cap_mod_id)
      
      cunco_id    = channel['cnco_dac']
      cdnco_id    = channel['cnco_adc']
      funco_id    = channel['fnco_dac']
      nco_device  = lsi_ctrl.ad9082[cunco_id[0]]
      
      args = name,awg_ctrl,awg_ch_ids,capture_units,nco_device,cunco_id[0],funco_id[0],cdnco_id[0]
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

    print(sys._getframe().f_code.co_name,found)
    returnValue(found)

############################################################
#
# AUX SUBROUTINES FOR EASY SETUP
#

def basic_config():
  import json
  
  readout_control_qube = \
  [
    {'name'    : 'readout_01',
     'type'    : 'mux',
     'ch_dac'  : 15,                                          # awg id
     'ch_adc'  : 1,                                           # module id
     'cnco_dac' : (0,0),                                      # chip, main path
     'cnco_adc' : (0,3),                                      # chip, main path
     'fnco_dac' : (0,0),                                      # chip, link no
    },
    {'name'    : 'readout_cd',
     'type'    : 'mux',
     'ch_dac'  : 2,                                           # awg id
     'ch_adc'  : 0,                                           # module id
     'cnco_dac' : (1,3),                                      # chip, main path
     'cnco_adc' : (1,3),                                      # chip, main path
     'fnco_dac' : (0,7),                                      # chip, link no
    },
    {'name'    : 'control_5',
     'type'    : 'control',
     'ch_dac'  : [11,12,13],                                  # awg id
     'cnco_dac' : (0,2),                                      # chip, main path id
     'fnco_dac' : [(0,2),(0,3),(0,4)],                        # chip, link no
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

    
