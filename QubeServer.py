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
184:class QSConstants:
226:  def __init__(self):
229:class QSMessage:
253:  def __init__(self):
263:def pingger(host):
274:class QuBE_ControlLine(DeviceWrapper):
276:  def connect(self, *args, **kw ):
307:  def get_connected(self):
317:  def number_of_shots(self):
320:  def number_of_shots(self,value):
324:  def repetition_time(self):
327:  def repetition_time(self,value_in_ns):
332:  def sequence_length(self):
335:  def sequence_length(self,value):
339:  def number_of_awgs(self):
343:  def list_of_awg_ids(self):
347:  def channel_enable(self):
351:  def enabled_channels(self):
355:  def enabled_awgs(self):
362:  def check_awg_channels(self,channels):
368:  def check_waveform(self,waveforms,channels):
391:  def upload_parameters(self,channels):
398:  def upload_waveform(self,waveforms,channels):
412:  def start_daq(self,awg_ids):
415:  def stop_daq(self,awg_ids,timeout):
418:  def get_lo_frequency(self):
421:  def set_lo_frequency(self,freq_in_mhz):
424:  def get_dac_coarse_frequency(self):
427:  def set_dac_coarse_frequency(self,freq_in_mhz):
432:  def static_DACify(self, waveform):
436:  def static_get_dac_coarse_frequency(self,nco_ctrl,ch):
440:  def static_get_dac_coarse_ftw(self,nco_ctrl,ch):
449:  def static_check_lo_frequency(self,freq_in_mhz):
453:  def static_check_dac_coarse_frequency(self,freq_in_mhz):
457:  def static_check_dac_fine_frequency(self,freq_in_mhz):
461:  def static_check_value(self,value,resolution,multiplier=50):
467:  def static_check_repetition_time(self,reptime_in_nanosec):
471:  def static_check_sequence_length(self,seqlen_in_nanosec):
476:class QuBE_ReadoutLine(QuBE_ControlLine):
478:  def connect(self, *args, **kw ):
516:  def get_connected(self):
541:  def acquisition_window(self):
544:  def acquisition_window(self, windows_in_ns):
547:  def acquisition_mode(self):
550:  def acquisition_mode(self,mode):
553:  def acquisition_number_of_windows(self):
556:  def acquisition_number_of_windows(self,n):
560:  def acquisition_mux_enable(self):
564:  def acquisition_enabled_channels(self):
572:  def upload_readout_parameters(self,muxchs):
634:  def set_trigger_board(self, awg_board ):
640:  def set_adc_coarse_frequency(self,freq_in_mhz):
645:  def get_adc_coarse_frequency(self):
648:  def static_get_adc_coarse_frequency(self,nco_ctrl,ch):
652:  def static_get_adc_coarse_ftw(self,nco_ctrl,ch):
661:  def static_check_adc_coarse_frequency(self,freq_in_mhz):
665:  def static_check_mux_channel_range(self,mux):
669:  def static_check_acquisition_windows(self,list_of_windows):
670:    def check_value(w):
687:class QuBE_Server(DeviceServer):
695:  def initServer(self):
712:  def initContext(self, c):
719:  def chooseDeviceWrapper(self, *args, **kw):
723:  def instantiateChannel(self,name,channels,awg_ctrl,cap_ctrl,lsi_ctrl):
724:    def gen_awg(name,channel,awg_ctrl,lsi_ctrl):
735:    def gen_mux(name,channel,awg_ctrl,cap_ctrl,lsi_ctrl):
763:  def instantiateQube(self,name,info):
790:  def findDevices(self):
815:  def number_of_shots(self,c,num_shots = None):
835:  def repeat_count(self,c,repeat = None):
851:  def repetition_time(self,c,reptime = None):
875:  def sequence_length(self,c,length = None):
899:  def daq_start(self,c):
921:  def daq_trigger(self,c):
932:  def daq_stop(self,c):
941:  def daq_timeout(self,c,t = None):
950:  def trigger_board(self,c,channel = None):
963:  def upload_parameters(self,c,channels):
983:  def upload_readout_parameters(self,c,muxchs):
1004:  def upload_waveform(self,c, wavedata,channels):
1034:  def acquisition_count(self,c,acqcount = None):
1047:  def acquisition_number(self,c,muxch,acqnumb = None):
1074:  def acquisition_window(self,c,muxch,window = None):
1110:  def acquisition_mode(self,c,muxch,mode = None):
1129:  def acquisition_mux_enable(self,c,muxch = None):
1153:  def local_frequency(self,c,frequency = None):
1165:  def coarse_tx_nco_frequency(self,c,frequency = None):
1177:  def coarse_rx_nco_frequency(self,c,frequency = None):
1191:  def debug_awg_ctrl_reg(self,c, addr, offset, pos, bits, data = None):
1221:  def debug_cap_ctrl_reg(self,c, addr, offset, pos, bits, data = None):
1259:def basic_config():
1297:def load_config(cxn,config):

814:  @setting(100, 'Shots', num_shots = ['w'], returns=['w'])
834:  @setting(101, 'Repeat Count', repeat = ['w'], returns=['w'])
850:  @setting(102, 'Repetition Time', reptime = ['v[s]'], returns=['v[s]'])
874:  @setting(103, 'DAQ Length', length = ['v[s]'], returns = ['v[s]'])
898:  @setting(105, 'DAQ Start', returns = ['b'])
920:  @setting(106, 'DAQ Trigger', returns = ['b'])
931:  @setting(107, 'DAQ Stop', returns = ['b'])
940:  @setting(108, 'DAQ Timeout', t = ['v[s]'], returns = ['b'])
949:  @setting(109, 'Trigger Board', channel = ['w'], returns = ['b'])
962:  @setting(200, 'Upload Parameters', channels=['w','*w'],returns=['b'])
982:  @setting(201, 'Upload Readout Parameters', muxchs=['*w','w'],returns=['b'])
1003:  @setting(202, 'Upload Waveform', wavedata =['*2c','*c'], channels=['*w','w'],returns=['b'])
1033:  @setting(300, 'Acquisition Count', acqcount = ['w'], returns = ['w'])
1046:  @setting(301, 'Acquisition Number', muxch = ['w'], acqnumb = ['w'], returns = ['w'])
1073:  @setting(302, 'Acquisition Window', muxch = ['w'], window = ['*(v[s]v[s])'], returns=['*(v[s]v[s])'])
1109:  @setting(303, 'Acquisition Mode', muxch = ['w'], mode = ['s'], returns=['s'])
1128:  @setting(304, 'Acquisition Mux Enable', muxch = ['w'], returns = ['b','*b'])
1152:  @setting(400, 'Frequency Local', frequency = ['v[Hz]'], returns = ['v[Hz]'])
1164:  @setting(401, 'Frequency TX NCO', frequency = ['v[Hz]'], returns = ['v[Hz]'])
1176:  @setting(402, 'Frequency RX NCO', frequency = ['v[Hz]'], returns = ['v[Hz]'])
1190:  @setting(502, 'DEBUG AWG REG', addr = ['w'], offset = ['w'], pos = ['w'], bits = ['w'], data = ['w'], returns = ['w'])
1220:  @setting(501, 'DEBUG CAP REG', addr = ['w'], offset = ['w'], pos = ['w'], bits = ['w'], data = ['w'], returns = ['w'])
"""

############################################################
#
# LABRAD SERVER FOR QIQB QUBE UNITS
#   20220502-07 (draft) Yutaka Tabuchi
#

from labrad                 import types as T, util
from labrad.devices         import DeviceWrapper,   \
                                   DeviceServer
from labrad.server          import setting
from labrad.units           import Value
from twisted.internet.defer import inlineCallbacks, \
                                   returnValue
import sys
import copy
                                                            # import socket
import time
import numpy as np
import struct
import json
                                                            # from ftplib import FTP
from e7awgsw                import DspUnit,       \
                                   AwgCtrl,       \
                                   AWG,           \
                                   WaveSequence,  \
                                   CaptureModule, \
                                   CaptureCtrl,   \
                                   CaptureParam
import qubelsi.qube
import subprocess
import os
#import concurrent                                          # to be used by Suzuki-san
#from labrad.concurrent      import future_to_deferred      # to be used by Suzuki-san


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
  DAQ_INITTOUT       = 5                                    # seconds
  ACQ_INITMODE       = '3'
  DAC_SAMPLE_R       = 12000                                # MHz
  ADC_SAMPLE_R       = 6000                                 # MHz
  DAC_BITS           = 16                                   # bits
  DAQ_BITS_POW_HALF  = 2**15                                # 2^(DAC_BITS-1)
  DAC_WVSAMP_IVL     = 2                                    # ns; Sampling intervals of waveforms
  DAC_WORD_IVL       = 8                                    # ns; DAC Word in nanoseconds
  DAC_WORD_SAMPLE    = 4                                    # Sample/(DAC word)
  DAQ_CNCO_BITS      = 48
  DAQ_LO_RESOL       = 100                                  # MHz
  DAC_CNCO_RESOL     = 12000/2**13                          # MHz; DAC_SAMPLE_R/2**13
  DAC_FNCO_RESOL     = 2000/2**12                           # MHz; DAC_SAMPLE_R/M=6/2**12
  ADC_CNCO_RESOL     = 1000/2**11                           # MHz; ADC_SAMPLE_R/M=6/2**11
  DAQ_REPT_RESOL     = 10240                                # nanoseconds
  DAQ_SEQL_RESOL     = 128                                  # nanoseconds
  #ACQ_INITMODE       = DspUnit.INTEGRATION
  ACQ_MULP           = 4                                    # 4 channel per mux
  ACQ_MAXWINDOW      = 2000                                 # nano-seconds
  ACQ_MAXNUMCAPT     = 8                                    # No reason for set this value. We'd be-
                                                            # tter to change it later.
  ACQ_CAPW_RESOL     = 8                                    # nano-seconds
  ACQ_CAST_RESOL     = 128                                  # nano-seconds. The first capture window
                                                            # must start from the multiple of 128 ns
  ACQ_MODENUMBER     = {'1':0, '2':1, '3':2,'A':3,'B':4 }
  ACQ_INITWINDOW     = [(0,1024)]
  DAC_CNXT_TAG       = 'awgs'
  ACQ_CNXT_TAG       = 'muxs'
  DAQ_TRIG_TAG       = 'trigger'
  DAQ_TOUT_TAG       = 'timeout'

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
  ERR_INVALID_RANG   = 'Invalid range. {} must be between {} and {}.'
  ERR_INVALID_ITEM   = 'Invalid data. {} must be one of {}'
  ERR_INVALID_WIND   = 'Invalid window range.'
  ERR_INVALID_WAVD   = 'Invalid waveform data. '                                         \
                     + '(1) Inconsistent number of waveforms and channels. '             \
                     + '(2) The number of channels are less than that of # of awgs. '    \
                     + '(3) The sequence length in nano-second must be identical to '    \
                     + 'the value set by daq_length(). '                                 \
                     + '(4) The data length must be multiple of {}. '                    \
                       .format(QSConstants.DAQ_SEQL_RESOL // QSConstants.DAC_WVSAMP_IVL) \
                     + '(5) The absolute value of complex data is less than 1. '         \
                     + 'The problem is {}.'
  ERR_NOARMED_DAC    = 'No ready dac channels.'

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
    self.awg_enabled  = [False for i in range(self.awg_chs)]

    yield

  @property
  def number_of_shots(self):
    return int(self.shots)
  @number_of_shots.setter
  def number_of_shots(self,value):
    self.shots = int(value)

  @property
  def repetition_time(self):
    return int(self.rep_time)
  @repetition_time.setter
  def repetition_time(self,value_in_ns):
    self.rep_time = int(((value_in_ns+QSConstants.DAQ_REPT_RESOL/2)//QSConstants.DAQ_REPT_RESOL) \
                         *QSConstants.DAQ_REPT_RESOL)

  @property
  def sequence_length(self):
    return int(self.seqlen)
  @sequence_length.setter
  def sequence_length(self,value):
    self.seqlen = value

  @property
  def number_of_awgs(self):
    return self.awg_chs

  @property
  def list_of_awg_ids(self):
    return copy.copy(self.awg_ch_ids)

  @property
  def channel_enable(self):
    return copy.copy(self.awg_enabled)

  @property
  def enabled_channels(self):
    return [i for i in range(self.awg_chs) if self.awg_enabled[i]]

  @property
  def enabled_awgs(self):
    '''
      For daq_start() in class QuBE_Server:
        this returns a list of the enabled awg modules ids.
    '''
    return [self.awg_ch_ids[i] for i in range(self.awg_chs) if self.awg_enabled[i]]

  def check_awg_channels(self,channels):
    for _c in channels:
      if _c < 0 or self.number_of_awgs <= _c:
        return False
    return True

  def check_waveform(self,waveforms,channels):
    chans,length = waveforms.shape

    help = 1
    resp = chans == len(channels)
    if resp:
      resp = chans <= self.number_of_awgs
      help += 1
    if resp:
      resp = QSConstants.DAC_WVSAMP_IVL*length == self.sequence_length
      help += 1
    if resp:
      block_restriction = QSConstants.DAQ_SEQL_RESOL // QSConstants.DAC_WVSAMP_IVL
      resp = 0 == length % block_restriction
      help += 1
    if resp:
      resp = np.max(np.abs(waveforms)) < 1.0
      help += 1
    if resp:
      return (True,chans,length)
    else:
      return (False,help,None)

  def upload_parameters(self,channels):
    self.awg_enabled  = [False for i in range(self.awg_chs)]

    for channel in channels:
      self.awg_enabled[channel] = True
    return True

  def upload_waveform(self,waveforms,channels):

    wait_words = int( ((self.repetition_time - self.sequence_length)
                      +QSConstants.DAC_WORD_IVL/2)  // QSConstants.DAC_WORD_IVL)

    for _waveform,_channel in zip(waveforms, channels):
      wave_seq  = WaveSequence( num_wait_words = 0, num_repeats = self.number_of_shots )
      iq_samples = list(zip(*self.static_DACify(_waveform)))
      wave_seq.add_chunk( iq_samples      = iq_samples,
                          num_blank_words = wait_words,
                          num_repeats     = 1 )
      self.awg_ctrl.set_wave_sequence(self.awg_ch_ids[_channel], wave_seq )
    return True

  def start_daq(self,awg_ids):
    self.awg_ctrl.start_awgs(*awg_ids)

  def stop_daq(self,awg_ids,timeout):
    self.awg_ctrl.wait_for_awgs_to_stop(timeout, *awg_ids)

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

  def static_DACify(self, waveform):
    return ((np.real(waveform) * QSConstants.DAQ_BITS_POW_HALF).astype(int),
            (np.imag(waveform) * QSConstants.DAQ_BITS_POW_HALF).astype(int))

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

    name, awg_ctrl, awg_ch_id, cap_ctrl, capture_units, capture_mod_id, \
      nco_device, cunco_id, funco_id, cdnco_id,lo_device = args
    print(QSMessage.CONNECTING_CHANNEL.format(name))
    self.name = name
    self.awg_ctrl   = awg_ctrl
    self.cap_ctrl   = cap_ctrl
    self.cap_mod_id = capture_mod_id
    self.cap_unit   = capture_units
    self.nco_ctrl   = nco_device
    self.lo_ctrl    = lo_device
    self.awg_ch_ids = [awg_ch_id]
    self.awg_chs    = len(self.awg_ch_ids)                  # should be 1
    self.cnco_id    = cunco_id
    self.fnco_ids   = [funco_id]
    self.rxcnco_id  = cdnco_id

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
    self.window        = [QSConstants.ACQ_INITWINDOW for i in range(QSConstants.ACQ_MULP)]
    self.acq_mode      = [QSConstants.ACQ_INITMODE   for i in range(QSConstants.ACQ_MULP)]
    self.acq_n_windows = [1                          for i in range(QSConstants.ACQ_MULP)]
    self.mux_enabled   = [False                      for i in range(QSConstants.ACQ_MULP)]

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

  @property
  def acquisition_window(self):
    return self.window
  @acquisition_window.setter
  def acquisition_window(self, windows_in_ns):
    self.window = windows_in_ns
  @property
  def acquisition_mode(self):
    return self.acq_mode
  @acquisition_mode.setter
  def acquisition_mode(self,mode):
    self.acq_mode = mode
  @property
  def acquisition_number_of_windows(self):
    return self.acq_n_windows
  @acquisition_number_of_windows.setter
  def acquisition_number_of_windows(self,n):
    self.acq_n_windows = n

  @property
  def acquisition_mux_enable(self):
    return self.mux_enabled

  @property
  def acquisition_enabled_channels(self):
    '''
      For daq_start() in class QuBE_Server:
        this returns a tuple of the capture module id and the enabled mux units
    '''
    return ( self.cap_mod_id, \
             [self.cap_unit[i] for i in range(QSConstants.ACQ_MULP) if self.mux_enabled[i]])

  def upload_readout_parameters(self,muxchs):
    """
    Upload readout parameters

    *Note for other guys

    Example for param.num_sum_sections = 1 (a single readout in an experiment like Rabi)
      +----------------------+------------+----------------------+------------+----------------------+
      |   blank   | readout  | post-blank |   blank   | readout  | post-blank |   blank   | readout  |
      | (control  |          | (relax ba- | (control  |          | (relax ba- | (control  |          |
      | operation)|          | ck to |g>) | operation)|          | ck to |g>) | operation)|          |
      +----------------------+------------+----------------------+------------+----------------------+
                  |<------- REPETITION TIME --------->|<------- REPETITION TIME --------->|<---
    ->|-----------|<- CAPTURE DELAY

    - Given that the sum_section is defined as a pair of capture duration and
      post blank, the initial non-readout duration has to be implemented usi-
      ng capture_delay.
    - The repetition duration starts at the beginning of readout operation
      and ends at the end of 2nd control operation (just before 2nd readout)
    """
    self.mux_enabled   = [False for i in range(QSConstants.ACQ_MULP)]

    repetition_word = int((self.repetition_time + QSConstants.ACQ_CAPW_RESOL//2)
                          // QSConstants.ACQ_CAPW_RESOL)
    for mux in muxchs:
      param    = CaptureParam()
      win_word = list()
      for _s,_e in self.acquisition_window[mux]:            # flatten window (start,end) to a series
                                                            # of timestamps

        win_word.append(int((_s+QSConstants.ACQ_CAPW_RESOL/2)//QSConstants.ACQ_CAPW_RESOL))
        win_word.append(int((_e+QSConstants.ACQ_CAPW_RESOL/2)//QSConstants.ACQ_CAPW_RESOL))
      win_word.append(repetition_word)

      param.num_integ_sections = int(self.number_of_shots)
      _s0                      = win_word.pop(0)
      param.capture_delay      = _s0
      win_word[-1]            += _s0                        # win_word[-1] is the end time of a sin-
                                                            # gle sequence. As the repeat duration
                                                            # is offset by capture_delay, we have to
                                                            # add the capture_delay time.
      while len(win_word)>1:
        _e = win_word.pop(0)
        _s = win_word.pop(0)
        blank_length   = _s - _e
        section_length = _e - _s0
        _s0 = _s
        param.add_sum_section(section_length, blank_length)

      param.sel_dsp_units_to_enable(DspUnit.INTEGRATION)
      param.sum_start_word_no = 0                           # UNUSED FOR NOW
      param.num_words_to_sum  = \
         CaptureParam.MAX_SUM_SECTION_LEN                   # UNUSED FOR NOW

      if param.num_samples_to_process * 2 != self.repetition_time:
        raise Exception(None)
      else:
        self.cap_ctrl.set_capture_params(mux, param)
        self.mux_enabled[mux] = True
    return True

  def set_trigger_board(self, awg_board ):
    self.cap_ctrl.select_trigger_awg(self.cap_mod_id, awg_board)
    enabled_cap_units = [self.cap_unit[i] for i in range(QSConstants.ACQ_MULP)
                                          if self.mux_enabled[i]              ]
    self.cap_ctrl.enable_start_trigger(*enabled_cap_units)

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

  def static_check_mux_channel_range(self,mux):
    return True if 0 <= mux and mux < QSConstants.ACQ_MULP else \
           False

  def static_check_acquisition_windows(self,list_of_windows):
    def check_value(w):
      return False if 0 != w % QSConstants.ACQ_CAPW_RESOL else True

    if 0 != list_of_windows[0][0] % QSConstants.ACQ_CAST_RESOL:
      return False

    for _s,_e in list_of_windows:
      if not check_value(_s) or not check_value(_e):
        return False

    return True

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
    DeviceServer.initContext(self,c)
    c[QSConstants.DAC_CNXT_TAG] = list()
    c[QSConstants.ACQ_CNXT_TAG] = dict()
    c[QSConstants.DAQ_TRIG_TAG] = 0
    c[QSConstants.DAQ_TOUT_TAG] = QSConstants.DAQ_INITTOUT

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

      args = name, awg_ctrl, awg_ch_ids, nco_device, cnco_id[1], [_id for _chip,_id in fnco_id], lo_device
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

      args = name, awg_ctrl, awg_ch_ids, cap_ctrl, capture_units, cap_mod_id, \
             nco_device, cunco_id[1], funco_id[1], cdnco_id[1], lo_device
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
      awg_ctrl.initialize(*AWG.all())
      cap_ctrl.initialize(*CaptureModule.all())
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
        print(sys._getframe().f_code.co_name,e)
        continue

      print(QSMessage.CNCTABLE_QUBEUNIT.format(name))
      devices = self.instantiateQube(name,self.possibleLinks[name])
      found.extend(devices)
      yield

    print(sys._getframe().f_code.co_name,found)             # DEBUG
    returnValue(found)

  @setting(100, 'Shots', num_shots = ['w'], returns=['w'])
  def number_of_shots(self,c,num_shots = None):
    """
    Read and write the number of repeated experiments.

    The number of <shots> of an experiment with fixed waveform.

    Args:
        num_shots: w
            The number of repeat in an extire experiments. Used to say "shots"
    Returns:
        num_shots: w
    """
    dev = self.selectedDevice(c)
    if num_shots is not None:
      dev.number_of_shots = num_shots
      return num_shots
    else:
      return dev.number_of_shots

  @setting(101, 'Repeat Count', repeat = ['w'], returns=['w'])
  def repeat_count(self,c,repeat = None):
    """
    OBSOLETED. Use repetition time instead.

    This is no longer used.

    Args:
        repeat: w
            The number of repeat in an extire experiments. Used to say "shots"
    Returns:
        repeat: w
    """
    raise Exception('obsoleted. use "shots" instead')
    return self.number_of_shots(c,repeat)

  @setting(102, 'Repetition Time', reptime = ['v[s]'], returns=['v[s]'])
  def repetition_time(self,c,reptime = None):
    """
    Read and write reperition time.

    The repetition time of a single experiments include control/readout waveform
    plus wait (blank, not output) duration.

    Args:
        reptime: v[s]
            10.24us - 1s can be set. The duration must be a multiple of 10.24 us
            to satisty phase coherence.
    Returns:
        reptime: v[s]
    """
    dev = self.selectedDevice(c)
    if reptime is None:
      return T.Value(dev.repetition_time,'ns')
    elif dev.static_check_repetition_time(reptime['ns']):
      dev.repetition_time = int(round(reptime['ns']))
      return reptime
    else:
      raise ValueError(QSMessage.ERR_REP_SETTING.format('Sequencer',QSConstants.DAQ_REPT_RESOL))

  @setting(103, 'DAQ Length', length = ['v[s]'], returns = ['v[s]'])
  def sequence_length(self,c,length = None):
    """
    Read and write waveform length.

    The waveform length supposed to be identical among all channels. It can be
    different, but we have not done yet.

    Args:
        length: v[s]
            The length of sequence waveforms. The length must be a
            multiple of 128 ns. 0.128ns - 200us can be set.
    Returns:
        length: v[s]
    """
    dev = self.selectedDevice(c)
    if length is None:
      return Value(dev.sequence_length,'ns')
    elif dev.static_check_sequence_length(length['ns']):
      dev.sequence_length = int(length['ns']+0.5)
      return length
    else:
      raise ValueError(QSMessage.ERR_REP_SETTING.format('Sequencer',QSConstants.DAQ_SEQL_RESOL))

  @setting(105, 'DAQ Start', returns = ['b'])
  def daq_start(self,c):
    """
    import pickle
    import base64
    """
    dev = self.selectedDevice(c)

    _to_be_added = [_awg for _awg in dev.enabled_awgs if _awg not in c[QSConstants.DAC_CNXT_TAG]]
    c[QSConstants.DAC_CNXT_TAG].extend(_to_be_added)        # This implementation suppose not to use
    if 'readout' in dev.name:                               # the multiple QuBE units. In synchro-
      module, units = dev.acquisition_enabled_channels      # nous multi-QuBE operation, we may have
      if len(units) > 0:                                    # to add QuBE-name in the tag.
        c[QSConstants.ACQ_CNXT_TAG][module]=units
      print(c[QSConstants.ACQ_CNXT_TAG])                    # DEBUG
    print(c[QSConstants.DAC_CNXT_TAG])                      # DEBUG

    if 'readout' in dev.name:                               # Set trigger board to capture units
      dev.set_trigger_board( c[QSConstants.DAQ_TRIG_TAG] )

    return True

  @setting(106, 'DAQ Trigger', returns = ['b'])
  def daq_trigger(self,c):

    dev = self.selectedDevice(c)
    if len(c[QSConstants.DAC_CNXT_TAG]) > 0:
      dev.start_daq(c[QSConstants.DAC_CNXT_TAG])
    else:
      raise Exception(QSMessage.ERR_NOARMED_DAC)

    return True

  @setting(107, 'DAQ Stop', returns = ['b'])
  def daq_stop(self,c):
    dev = self.selectedDevice(c)
    if len(c[QSConstants.DAC_CNXT_TAG]) > 0:
      dev.stop_daq(c[QSConstants.DAC_CNXT_TAG], c[QSConstants.DAQ_TOUT_TAG])
    else:
      raise Exception(QSMessage.ERR_NOARMED_DAC)
    return True

  @setting(108, 'DAQ Timeout', t = ['v[s]'], returns = ['b'])
  def daq_timeout(self,c,t = None):
    if t is None:
      val = c[QSConstants.DAQ_TOUT_TAG]
      return T.Value(val,'s')
    else:
      c[QSConstants.DAQ_TOUT_TAG] = t['s']
      return t

  @setting(109, 'Trigger Board', channel = ['w'], returns = ['b'])
  def trigger_board(self,c,channel = None):
    dev = self.selectedDevice(c)
    if channel is not None:
      if not dev.check_awg_channels([channel]):
        raise ValueError(QSMessage.ERR_INVALID_RANG.format('awg index',0,dev.number_of_awgs - 1))
    else:
      channel = 0

    c[QSConstants.DAQ_TRIG_TAG] = dev.list_of_awg_ids[channel]

    return True

  @setting(200, 'Upload Parameters', channels=['w','*w'],returns=['b'])
  def upload_parameters(self,c,channels):
    """
    Upload channel parameters.

    Sequence setting.

    Args:
        channels : w, *w
            multiplex channel   0 to 3 [QSConstants.ACQ_MULP-1]
    Returns:
        success  : b
            True if successful.
    """
    dev  = self.selectedDevice(c)
    channels = np.atleast_1d(channels).astype(int)
    if not dev.check_awg_channels(channels):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('awg index',0,dev.number_of_awgs - 1))
    return dev.upload_parameters(channels)

  @setting(201, 'Upload Readout Parameters', muxchs=['*w','w'],returns=['b'])
  def upload_readout_parameters(self,c,muxchs):
    """
    Upload readout demodulator parameters.

    It sends the necessary parameters for readout operation.

    Args:
        muxchs: w, *w
            multiplex channel   0 to 3 [QSConstants.ACQ_MULP-1]
    """
    dev = self.selectedDevice(c)
    if 'readout' not in dev.name:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.name))

    muxchs = np.atleast_1d(muxchs).astype(int)
    for _mux in muxchs:
      if not dev.static_check_mux_channel_range(_mux):
        raise Exception(QSMessage.ERR_INVALID_RANG.format('muxch',0,QSConstants.ACQ_MULP - 1))
    return dev.upload_readout_parameters(muxchs)

  @setting(202, 'Upload Waveform', wavedata =['*2c','*c'], channels=['*w','w'],returns=['b'])
  def upload_waveform(self,c, wavedata,channels):
    """
    Upload waveform to FPGAs.

    Transfer 500MSa/s complex waveforms to the QuBE FPGAs.

    Args:
        wavedata : *2c,*c
            Complex waveform data with a sampling interval of 2 ns [QSConstants.
            DAC_WVSAMP_IVL]. When more than two channels, speficy the waveform
            data using list, i.e.  [data0,data1,...], or tuple (data0,data1,...)

        channels: *w, w
            List of the channels, e.g., [0,1] for the case where the number of
            rows of wavedata is more than 1. You can simply give the channel
            number to set a single-channel waveform.
    """
    dev = self.selectedDevice(c)
    channels  = np.atleast_1d(channels).astype(int)
    waveforms = np.atleast_2d(wavedata).astype(complex)

    if not dev.check_awg_channels(channels):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('awg index',0,dev.number_of_awgs - 1))
    resp,number_of_chans,data_length = dev.check_waveform(waveforms,channels)
    if not resp:
      raise ValueError(QSMessage.ERR_INVALID_WAVD.format(number_of_chans))

    return dev.upload_waveform(waveforms,channels)

  @setting(300, 'Acquisition Count', acqcount = ['w'], returns = ['w'])
  def acquisition_count(self,c,acqcount = None):
    """
    Read and write acquisition count.

    OBSOLETED

    Args:
       acqcount : w
            The number of acquisition in a single experiment. 1 to 8 can be set.
    """
    raise Exception('obsoleted. use "acquisition_number" instead')

  @setting(301, 'Acquisition Number', muxch = ['w'], acqnumb = ['w'], returns = ['w'])
  def acquisition_number(self,c,muxch,acqnumb = None):
    """
    Read and write the number of acquisition windows

    Setting for acquistion windows. You can have several accquisition windows in
    a single experiments.

    Args:
       muxch   : w
            Multiplex channel id. 0 to 3 [QSConstants.ACQ_MULP-1] can be set
       acqnumb : w
            The number of acquisition in a single experiment. 1 to 8 can be set.
    """
    dev = self.selectedDevice(c)
    if 'readout' not in dev.name:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.name))
    elif not dev.static_check_mux_channel_range(muxch):
      raise Exception(QSMessage.ERR_INVALID_RANG.format('muxch',0,QSConstants.ACQ_MULP - 1))
    elif acqnumb is None:
      return dev.acquisition_number_of_windows[muxch]
    elif 0 < acqnumb and acqnumb <= QSConstants.ACQ_MAXNUMCAPT:
      dev.acquisition_number_of_windows[muxch] = acqnumb
      return acqnumb
    else:
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('Acquisition number of windows',1,QSConstants.ACQ_MAXNUMCAPT))

  @setting(302, 'Acquisition Window', muxch = ['w'], window = ['*(v[s]v[s])'], returns=['*(v[s]v[s])'])
  def acquisition_window(self,c,muxch,window = None):
    """
    Read and write acquition windows.

    Setting for acquistion windows. You can have several accquisition windows
    in a single experiments. A windows is defined as a tuple of two timestamps
    e.g., (start, end). Multiples windows can be set like [(start1, end1),
    (start2, end2), ... ]

    Args:
        muxch: w
            multiplex channel   0 to 3 [QSConstants.ACQ_MULP-1]

        window: *(v[s]v[s])
            List of windows. The windows are given by tuples of (window start,
            window end).
    Returns:
        window: *(v[s]v[s])
            Current window setting
    """
    dev = self.selectedDevice(c)
    if 'readout' not in dev.name:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.name))
    elif not dev.static_check_mux_channel_range(muxch):
      raise Exception(QSMessage.ERR_INVALID_RANG.format('muxch',0,QSConstants.ACQ_MULP - 1))
    elif window is None:
      return [(T.Value(_s,'ns'),T.Value(_e,'ns')) for _s,_e in dev.acquisition_window[muxch]]

    wl = [(int(_w[0]['ns']+0.5),int(_w[1]['ns']+0.5)) for _w in window]
    if dev.static_check_acquisition_windows(wl):
      dev.acquisition_window[muxch] = wl
      return window
    else:
      raise ValueError(QSMessage.ERR_INVALID_WIND)

  @setting(303, 'Acquisition Mode', muxch = ['w'], mode = ['s'], returns=['s'])
  def acquisition_mode(self,c,muxch,mode = None):
    '''
      w          : multiplex channel   0 to 3 [QSConstants.ACQ_MULP-1]
      s          : capture mode. one of '2', '3', 'A', 'B' can be set
    '''
    dev = self.selectedDevice(c)
    if 'readout' not in dev.name:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.name))
    elif not dev.static_check_mux_channel_range(muxch):
      raise Exception(QSMessage.ERR_INVALID_RANG.format('muxch',0,QSConstants.ACQ_MULP - 1))
    elif mode is None:
      return dev.acquisition_mode[muxch]
    elif mode in QSConstants.ACQ_MODENUMBER.keys():
      dev.acquisition_mode[muxch] = mode
      return mode
    else:
      raise ValueError(QSMessage.ERR_INVALID_ITEM.format( 'Acquisition mode',','.join(QSConstants.ACQ_MODENUMBER.keys())))

  @setting(304, 'Acquisition Mux Enable', muxch = ['w'], returns = ['b','*b'])
  def acquisition_mux_enable(self,c,muxch = None):
    """
    Obtain enabled demodulation mux channels

    Mux demodulation channels are enabled in upload_readout_parameters(). 

    Args:
        muxch : w
            multiplex channel   0 to 3 [QSConstants.ACQ_MULP-1]. 
            Read all channel if None.
    Returns:
        Enabled(True)/Disabled(False) status of the channel. 
    """
    dev = self.selectedDevice(c)
    if 'readout' not in dev.name:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.name))
    elif muxch is None:
      return dev.acquisition_mux_enable
    elif not dev.static_check_mux_channel_range(muxch):
      raise Exception(QSMessage.ERR_INVALID_RANG.format('muxch',0,QSConstants.ACQ_MULP - 1))
    else:
      return dev.acquisition_mux_enable[muxch]

  @setting(400, 'Frequency Local', frequency = ['v[Hz]'], returns = ['v[Hz]'])
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

  @setting(401, 'Frequency TX NCO', frequency = ['v[Hz]'], returns = ['v[Hz]'])
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

  @setting(402, 'Frequency RX NCO', frequency = ['v[Hz]'], returns = ['v[Hz]'])
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

  @setting(502, 'DEBUG AWG REG', addr = ['w'], offset = ['w'], pos = ['w'], bits = ['w'], data = ['w'], returns = ['w'])
  def debug_awg_ctrl_reg(self,c, addr, offset, pos, bits, data = None):
    """
    Read and write to the AWG registers

    It is useful for debug but should not be used for daily operation.

    Args:

        addr  : w   0x00 for Master control registers
        offset: w   0x00 [32bits] version
                    0x04 [16bits] control select
                    0x08 [4bits]  awg control. bit0 reset, bit 1 prepare,
                                               bit2 start, bit 3 terminate
                    0x10 [16bits] busy
                    0x14 [16bits] ready
                    0x18 [16bits] done
        pos   : w   Bit location
        bits  : w   Bumber of bits to read/write
        data  : w   Data. Read operation is performed if None
    """
    dev = self.selectedDevice(c)
    reg = dev.awg_ctrl._AwgCtrl__reg_access
    if data is None:
      data = reg.read_bits(addr,offset,pos,bits)
      return data
    else:
      reg.write_bits(addr,offset,pos,bits,data)
    return 0

  @setting(501, 'DEBUG CAP REG', addr = ['w'], offset = ['w'], pos = ['w'], bits = ['w'], data = ['w'], returns = ['w'])
  def debug_cap_ctrl_reg(self,c, addr, offset, pos, bits, data = None):
    """
    Read and write to the AWG registers.

    It is useful for debug but should not be used for daily operation.

    Args:
        addr  : w    0x00 for Master control registers
        offset: w    Register index. See below
                     0x00 [32bits] version
                     0x04 [5bits ] select (mod0) bit0 no trig, bit1-4 daq id
                     0x08 [5bits ] select (mod1) bit0 no trig, bit1-4 daq id
                     0x0c [8bits]  trigger mask
                     0x10 [8bits] capmod select
                     0x1c [8bits] busy
                     0x20 [8bits] done
        pos   : w    Bit location
        bits  : w    Number of bits to read/write
        data  : w    Data. Read operation is performed if None
    """
    dev = self.selectedDevice(c)
    if 'readout' not in dev.name:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.name))
    reg = dev.cap_ctrl._CaptureCtrl__reg_access
    if data is None:
      data = reg.read_bits(addr,offset,pos,bits)
      return data
    else:
      reg.write_bits(addr,offset,pos,bits,data)
    return 0

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
     'ch_dac'  : 15,                                        # awg id
     'ch_adc'  : 1,                                         # module id
     'cnco_dac' : (0,0),                                    # chip, main path
     'cnco_adc' : (0,3),                                    # chip, main path
     'fnco_dac' : (0,0),                                    # chip, link id
     'lo_daq'   : 0,                                        # local oscillator id
    },
    {'name'    : 'readout_cd',
     'type'    : 'mux',
     'ch_dac'  : 2,                                         # awg id
     'ch_adc'  : 0,                                         # module id
     'cnco_dac' : (1,3),                                    # chip, main path
     'cnco_adc' : (1,3),                                    # chip, main path
     'fnco_dac' : (0,7),                                    # chip, link no
     'lo_daq'   : 7,                                        # local oscillator id
    },
    {'name'    : 'control_5',
     'type'    : 'control',
     'ch_dac'  : [11,12,13],                                # awg id
     'cnco_dac' : (0,2),                                    # chip, main path id
     'fnco_dac' : [(0,2),(0,3),(0,4)],                      # chip, link no
     'lo_dac'   : 2,                                        # local oscillator id
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


"""
# Typical usage
qs.select_device(0)
data=np.zeros(8192).astype(complex)
qs.daq_length(16384*ns)
qs.upload_waveform([data],0)
qs.upload_parameters([0])
qs.upload_readout_parameters([3])
#
qs.select_device(2)
qs.daq_length(16384*ns)
qs.upload_parameters([1])
qs.upload_waveform([data],1)
qs.trigger_board(1)
[(qs.select_device(i),qs.daq_start()) for i in [0,2]]
qs.daq_trigger()
qs.debug_awg_reg(0,0x04,0,16)
qs.debug_awg_reg(0,0x10,0,16)
qs.debug_awg_reg(0,0x14,0,16)
qs.debug_awg_reg(0,0x18,0,16)
"""
