# Copyright (C) 2022 Yutaka Tabuchi

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
300:class QSConstants:
413:  def __init__(self):
416:class QSMessage:
441:  def __init__(self):
451:def pingger(host):
476:class QuBE_DeviceBase(DeviceWrapper):
478:  def connect(self,*args,**kw):                             # @inlineCallbacks
489:  def get_connected(self,*args,**kwargs):                   # @inlineCallbacks
494:  def device_name(self):                                    # @property
498:  def device_role(self):                                    # @property
502:  def chassis_name(self):
505:  def static_check_value(self,value,resolution,multiplier=50,include_zero=False):
511:class QuBE_Control_FPGA(QuBE_DeviceBase):
514:  def get_connected(self,*args,**kw ):                      # @inlineCallbacks
537:  def number_of_shots(self):                                # @property
540:  def number_of_shots(self,value):                          # @number_of_shots.setter
544:  def repetition_time(self):                                # @property
547:  def repetition_time(self,value_in_ns):                    # @repetition_time.setter
552:  def sequence_length(self):                                # @property
555:  def sequence_length(self,value):                          # @sequence_length.setter
559:  def number_of_awgs(self):                                 # @property
562:  def get_awg_id(self, channel):
565:  def check_awg_channels(self,channels):
571:  def check_waveform(self,waveforms,channels):
594:  def upload_waveform(self,waveforms,channels):
608:  def start_daq(self,awg_ids):
611:  def stop_daq(self,awg_ids,timeout):
614:  def static_DACify(self, waveform):
618:  def static_check_repetition_time(self,reptime_in_nanosec):
622:  def static_check_sequence_length(self,seqlen_in_nanosec):
630:class QuBE_Control_LSI(QuBE_DeviceBase):
633:  def get_connected(self,*args,**kw):                       # @inlineCallbacks
659:  def get_lo_frequency(self):
662:  def set_lo_frequency(self,freq_in_mhz):
665:  def get_mix_sideband(self):
672:  def set_mix_sideband(self,sideband : str):
681:  def get_dac_coarse_frequency(self):
684:  def set_dac_coarse_frequency(self,freq_in_mhz):
689:  def get_dac_fine_frequency(self,channel):
692:  def set_dac_fine_frequency(self,channel,freq_in_mhz):
699:  def static_get_dac_coarse_frequency(self,nco_ctrl,ch):
703:  def static_get_dac_coarse_ftw(self,nco_ctrl,ch):
712:  def static_check_lo_frequency(self,freq_in_mhz):
716:  def static_check_dac_coarse_frequency(self,freq_in_mhz):
720:  def static_check_dac_fine_frequency(self,freq_in_mhz):
728:class QuBE_ControlLine(QuBE_Control_FPGA, QuBE_Control_LSI):
731:  def get_connected(self,*args,**kw ):                      # @inlineCallbacks
735:class QuBE_ReadoutLine(QuBE_ControlLine):
738:  def get_connected(self,*args,**kw):                       # @inlineCallbacks
761:  def get_capture_module_id(self):
764:  def get_capture_unit_id(self, mux_channel):
768:  def acquisition_window(self):                             # @property
771:  def set_acquisition_window(self,mux,window):
775:  def acquisition_mode(self):                               # @property, only referenced in QuBE_Server
778:  def set_acquisition_mode(self,mux,mode):
781:  def set_acquisition_fir_coefficient(self,muxch,coeffs):
787:  def set_acquisition_window_coefficient(self,muxch,coeffs):
793:  def upload_readout_parameters(self,muxchs):
852:  def configure_readout_mode(self,mux,param,mode):
867:  def configure_readout_dsp(self,mux,param,mode):
876:  def configure_readout_decimation(self,mux,param,decimation):
896:  def configure_readout_averaging(self,mux,param,averaging):
915:  def configure_readout_summation(self,mux,param,summation):
942:  def download_waveform(self, muxchs):
964:  def download_single_waveform(self, muxch):
972:  def set_trigger_board(self, trigger_board, enabled_capture_units):
976:  def set_adc_coarse_frequency(self,freq_in_mhz):
981:  def get_adc_coarse_frequency(self):
984:  def static_get_adc_coarse_frequency(self,nco_ctrl,ch):
988:  def static_get_adc_coarse_ftw(self,nco_ctrl,ch):
997:  def static_check_adc_coarse_frequency(self,freq_in_mhz):
1001:  def static_check_mux_channel_range(self,mux):
1005:  def static_check_acquisition_windows(self,list_of_windows):
1020:  def static_check_acquisition_fir_coefs(self,coeffs):
1028:  def static_check_acquisition_window_coefs(self,coeffs):
1041:class QuBE_Server(DeviceServer):
1049:  def initServer(self):                                     # @inlineCallbacks
1070:  def initContext(self, c):
1076:  def chooseDeviceWrapper(self, *args, **kw):
1081:  def instantiateChannel(self,name,channels,awg_ctrl,cap_ctrl,lsi_ctrl):
1131:  def instantiateQube(self,name,info):
1158:  def findDevices(self):                                    # @inlineCallbacks
1187:  def number_of_shots(self,c,num_shots = None):
1207:  def repeat_count(self,c,repeat = None):
1223:  def repetition_time(self,c,reptime = None):
1247:  def sequence_length(self,c,length = None):
1272:  def daq_start(self,c):
1299:  def _readout_mux_start(self,c):
1320:  def daq_trigger(self,c):
1340:  def daq_stop(self,c):
1356:  def daq_timeout(self,c,t = None):
1365:  def daq_channels(self,c):
1379:  def upload_parameters(self,c,channels):
1398:  def _register_awg_channels(self,c,dev,channels):
1425:  def upload_readout_parameters(self,c,muxchs):
1448:  def _register_mux_channels(self,c,dev,selected_mux_channels):
1482:  def upload_waveform(self,c, wavedata,channels):
1513:  def download_waveform(self,c,muxchs):
1540:  def acquisition_count(self,c,acqcount = None):
1553:  def acquisition_number(self,c,muxch,acqnumb = None):
1580:  def acquisition_window(self,c,muxch,window = None):
1616:  def acquisition_mode(self,c,muxch,mode = None):
1675:  def acquisition_mux_enable(self,c,muxch = None):
1716:  def filter_pre_coefficients(self,c,muxch,coeffs):
1724:  def set_window_coefficients(self,c,muxch,coeffs):
1732:  def acquisition_fir_coefficients(self,c,muxch,coeffs):
1762:  def acquisition_window_coefficients(self,c,muxch,coeffs):
1793:  def local_frequency(self,c,frequency = None):
1820:  def coarse_tx_nco_frequency(self,c,frequency = None):
1848:  def fine_tx_nco_frequency(self,c,channel,frequency = None):
1884:  def coarse_rx_nco_frequency(self,c,frequency = None):
1898:  def sideband_selection(self,c,sideband = None):
1922:class QuBE_Device_debug_otasuke(QuBE_Control_FPGA, QuBE_Control_LSI):
1925:  def get_connected(self,*args,**kw):
1938:  def get_microwave_switch(self):
1949:  def set_microwave_switch(self,output):
1960:class QuBE_ControlLine_debug_otasuke(QuBE_ControlLine, QuBE_Device_debug_otasuke):
1963:  def get_connected(self,*args,**kw ):                      # @inlineCallbacks
1967:class QuBE_ReadoutLine_debug_otasuke(QuBE_ReadoutLine, QuBE_Device_debug_otasuke):
1970:  def get_connected(self,*args,**kw ):                      # @inlineCallbacks
1975:class QuBE_Server_debug_otasuke(QuBE_Server):
1980:  def __init__(self,*args,**kw):
1983:  def instantiateChannel(self,name,channels,awg_ctrl,cap_ctrl,lsi_ctrl):
1996:  def debug_awg_ctrl_reg(self,c, addr, offset, pos, bits, data = None):
2026:  def debug_cap_ctrl_reg(self,c, addr, offset, pos, bits, data = None):
2058:  def debug_auto_acquisition_fir_coefficients(self,c,muxch,bb_frequency,sigma = None):
2099:  def debug_auto_acquisition_window_coefficients(self,c,muxch,bb_frequency):
2148:  def debug_microwave_switch(self,c,output = None):
2172:class Qube_Manager_Device(DeviceWrapper):
2175:  def connect(self, *args, **kw):                           # @inlineCallbacks
2190:  def initialize(self):                                     # @inlineCallbacks
2202:  def set_microwave_switch(self,value):                     # @inlineCallbacks
2207:  def read_microwave_switch(self):
2213:  def verbose(self):                                        # @property
2216:  def verbose(self,x):                                      # @verbose.setter
2221:  def synchronize_with_master(self):                        # @inlineCallbacks
2230:class Qube_Manager_Server(DeviceServer):
2237:  def initServer(self):                                     # @inlineCallbacks
2254:  def extract_links(self,link):
2260:  def initContext(self, c):
2264:  def findDevices(self):                                    # @inlineCallbacks
2288:  def instantiateQube(self, name, qube_type, iplsi, ipclk, channel_info):    # @inlineCallbacks
2301:  def device_reinitialize(self,c):
2316:  def microwave_switch(self,c,value = None):
2349:  def debug_verbose_message(self,c,flag = None):
2366:  def reconnect_master(self,c):
2387:  def clear_master_clock(self,c):
2412:  def read_master_clock(self,c):
2429:  def synchronize_with_master(self,c):
2445:  def _synchronize_with_master_clock(self,target_addr):     # @inlineCallbacks
2460:  def _read_master_clock(self):                             # @inlineCallbacks
2479:class QuBESequencerMaster(QuBEMasterClient):
2483:  def __init__(self, ip_addr):
2487:  def read_clock(self, value=0):                            # inherited from QuBEMasterClient
2495:class QuBESequencerClient(SequencerClient):
2499:  def __init__(self, ip_addr):
2502:  def add_sequencer(self, value, awgs = range(16)):
2526:def basic_config():
2722:def load_config(cxn,config):
2742:def usage():
2899:def test_control_ch(device_name):
2928:def test_control_ch_bandwidth(device_name):
2971:def test_readout_ch_bandwidth_and_spurious(device_name):
2978:  def spectrum_analyzer_get():
3018:  def experiment_nco_sweep( vault, fnco, file_idx ):
3088:if server_select is None:
3091:if __name__ == '__main__':

300:class QSConstants:
416:class QSMessage:
476:class QuBE_DeviceBase(DeviceWrapper):
511:class QuBE_Control_FPGA(QuBE_DeviceBase):
630:class QuBE_Control_LSI(QuBE_DeviceBase):
728:class QuBE_ControlLine(QuBE_Control_FPGA, QuBE_Control_LSI):
735:class QuBE_ReadoutLine(QuBE_ControlLine):
1041:class QuBE_Server(DeviceServer):
1922:class QuBE_Device_debug_otasuke(QuBE_Control_FPGA, QuBE_Control_LSI):
1960:class QuBE_ControlLine_debug_otasuke(QuBE_ControlLine, QuBE_Device_debug_otasuke):
1967:class QuBE_ReadoutLine_debug_otasuke(QuBE_ReadoutLine, QuBE_Device_debug_otasuke):
1975:class QuBE_Server_debug_otasuke(QuBE_Server):
2172:class Qube_Manager_Device(DeviceWrapper):
2230:class Qube_Manager_Server(DeviceServer):
2479:class QuBESequencerMaster(QuBEMasterClient):
2495:class QuBESequencerClient(SequencerClient):

1041:class QuBE_Server(DeviceServer):
1186:  @setting(100, 'Shots', num_shots = ['w'], returns=['w'])
1206:  @setting(101, 'Repeat Count', repeat = ['w'], returns=['w'])
1222:  @setting(102, 'Repetition Time', reptime = ['v[s]'], returns=['v[s]'])
1246:  @setting(103, 'DAQ Length', length = ['v[s]'], returns = ['v[s]'])
1271:  @setting(105, 'DAQ Start', returns = ['b'])
1319:  @setting(106, 'DAQ Trigger', returns = ['b'])
1339:  @setting(107, 'DAQ Stop', returns = ['b'])
1355:  @setting(108, 'DAQ Timeout', t = ['v[s]'], returns = ['v[s]'])
1364:  @setting(110, 'DAC Channels', returns = ['w'])
1378:  @setting(200, 'Upload Parameters', channels=['w','*w'],returns=['b'])
1424:  @setting(201, 'Upload Readout Parameters', muxchs=['*w','w'],returns=['b'])
1481:  @setting(202, 'Upload Waveform', wavedata =['*2c','*c'], channels=['*w','w'],returns=['b'])
1512:  @setting(203, 'Download Waveform', muxchs = ['*w','w'], returns = ['*c','*2c'])
1539:  @setting(300, 'Acquisition Count', acqcount = ['w'], returns = ['w'])
1552:  @setting(301, 'Acquisition Number', muxch = ['w'], acqnumb = ['w'], returns = ['w'])
1579:  @setting(302, 'Acquisition Window', muxch = ['w'], window = ['*(v[s]v[s])'], returns=['*(v[s]v[s])'])
1615:  @setting(303, 'Acquisition Mode', muxch = ['w'], mode = ['s'], returns=['s'])
1674:  @setting(304, 'Acquisition Mux Enable', muxch = ['w'], returns = ['b','*b'])
1715:  @setting(305, 'Filter Pre Coefficients', muxch = ['w'], coeffs = ['*c'], returns = ['b'])
1723:  @setting(306, 'Average Window Coefficients', muxch = ['w'], coeffs = ['*c'], returns = ['b'])
1731:  @setting(307, 'Acquisition FIR Coefficients', muxch = ['w'], coeffs = ['*c'], returns = ['b'])
1761:  @setting(308, 'Acquisition Window Coefficients', muxch = ['w'], coeffs = ['*c'], returns = ['b'])
1792:  @setting(400, 'Frequency Local', frequency = ['v[Hz]'], returns = ['v[Hz]'])
1819:  @setting(401, 'Frequency TX NCO', frequency = ['v[Hz]'], returns = ['v[Hz]'])
1847:  @setting(402, 'Frequency TX Fine NCO', channel = ['w'], frequency = ['v[Hz]'], returns = ['v[Hz]'])
1883:  @setting(403, 'Frequency RX NCO', frequency = ['v[Hz]'], returns = ['v[Hz]'])
1897:  @setting(404, 'Frequency Sideband', sideband = ['s'], returns = ['s'])
1975:class QuBE_Server_debug_otasuke(QuBE_Server):
1995:  @setting(502, 'DEBUG AWG REG', addr = ['w'], offset = ['w'], pos = ['w'], bits = ['w'], data = ['w'], returns = ['w'])
2025:  @setting(501, 'DEBUG CAP REG', addr = ['w'], offset = ['w'], pos = ['w'], bits = ['w'], data = ['w'], returns = ['w'])
2057:  @setting(503, 'DEBUG Auto Acquisition FIR Coefficients', muxch = ['w'], bb_frequency = ['v[Hz]'], sigma = ['v[s]'], returns = ['b'])
2098:  @setting(504, 'DEBUG Auto Acquisition Window Coefficients', muxch = ['w'], bb_frequency = ['v[Hz]'], returns = ['b'])
2147:  @setting(505, 'DEBUG Microwave Switch', output = ['b'], returns = ['b'])
2230:class Qube_Manager_Server(DeviceServer):
2300:  @setting(100, 'Reset', returns=['b'])
2315:  @setting(101, 'Microwave Switch', value = ['w'], returns = ['w'])
2348:  @setting(200, 'Debug Verbose', flag = ['b'], returns=['b'])
2365:  @setting(301, 'Reconnect Master Clock', returns = ['b'])
2386:  @setting(302, 'Clear Master Clock', returns = ['b'])
2411:  @setting(303, 'Read Master Clock', returns = ['ww'])
2428:  @setting(304, 'Synchronize Clock', returns = ['b'])
"""

############################################################
#
# LABRAD SERVER FOR QIQB QUBE UNIT
#   20220502-12 (almost done)) Yutaka Tabuchi
#

from labrad                 import types as T, util
from labrad.devices         import DeviceWrapper,   \
                                   DeviceServer
from labrad.server          import setting
from labrad.units           import Value
from twisted.internet.defer import inlineCallbacks, \
                                   returnValue
import sys
import os
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
from software.qubemasterclient \
                            import QuBEMasterClient         # for multi-sync operation
from software.sequencerclient  \
                            import SequencerClient          # for multi-sync operation
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
  REGMASTERLNK       = 'master_link'
  REGAPIPATH         = 'adi_api_path'
  SRVNAME            = 'QuBE Server'
  MNRNAME            = 'QuBE Manager'
  ENV_SRVSEL         = 'QUBE_SERVER'
  THREAD_MAX_WORKERS = 32
  DAQ_MAXLEN         = 199936                               # nano-seconds -> 24,992 AWG Word
  DAC_SAMPLE_R       = 12000                                # MHz
  NCO_SAMPLE_F       = 2000                                 # MHz, NCO frequency at main data path
  ADC_SAMPLE_R       = 6000                                 # MHz
  DACBB_SAMPLE_R     = 500                                  # MHz, baseband sampling frequency
  ADCBB_SAMPLE_R     = 500                                  # MHz, baseband sampling frequency
  ADCDCM_SAMPLE_R    = 125                                  # MHz, decimated sampling frequency
                                                            #   Note: This has been changed from
                                                            #         62.5 MHz in May 2022.
  DAC_BITS           = 16                                   # bits
  DAC_BITS_POW_HALF  = 2**15                                # 2^(DAC_BITS-1)
  DAC_WVSAMP_IVL     = 2                                    # ns; Sampling intervals of waveforms
                                                            #    = 1/DACBB_SAMPLE_R
  ADC_BBSAMP_IVL     = 2                                    # ns; Sampling intervals of readout waveform
                                                            #    = 1/ADCBB_SAMPLE_R
  DAC_WORD_IVL       = 8                                    # ns; DAC WORD in nanoseconds
  DAC_WORD_SAMPLE    = 4                                    # Sample/(DAC word); DEBUG not used
  DAQ_CNCO_BITS      = 48
  DAQ_LO_RESOL       = 100                                  # - The minimum frequency resolution of
                                                            #   the analog local oscillators in MHz.
  DAC_CNCO_RESOL     = 12000/2**13                          # - The frequency resolution of the
                                                            #   coarse NCOs in upconversion paths.
                                                            #   unit in MHz; DAC_SAMPLE_R/2**13
  DAC_FNCO_RESOL     = 2000/2**12                           # - The frequency resolution of the fine
                                                            #   NCOs in digital upconversion paths.
                                                            #   unit in MHz; DAC_SAMPLE_R/M=6/2**12
  ADC_CNCO_RESOL     = 6000/2**13                           # - The frequency resolution of coarse
                                                            #   NCOs in demodulation path
                                                            #   unit in MHz; ADC_SAMPLE_R/2**13
  ADC_FNCO_RESOL     = 1000/2**11                           # - The frequency resolution of fine
                                                            #   NCOs in demodulation path.
                                                            #   unit in MHz; ADC_SAMPLE_R/M=6/2**11
  DAQ_REPT_RESOL     = 10240                                # - The mininum time resolution of a
                                                            #   repetition time in nanoseconds.
  DAQ_SEQL_RESOL     = 128                                  # - The mininum time resolution of a
                                                            #   sequence length in nanoseconds.
  ACQ_MULP           = 4                                    # - 4 channel per mux
  ACQ_MAXWINDOW      = 2048                                 # - The maximum duration of a measure-
                                                            #   ment window in nano-seconds.
  ACQ_MAX_FCOEF      = 16                                   # - The maximum number of the FIR filter
                                                            #   taps prior to decimation process.
  ACQ_FCOEF_BITS     = 16                                   # - The number of vertical bits of the
                                                            #   FIR filter coefficient.
  ACQ_FCBIT_POW_HALF = 2**15                                # - equivalent to 2^(ACQ_FCOEF_BITS-1).
  ACQ_MAX_WCOEF      = 256                                  # - The maximally applicable complex
                                                            #   window coefficients. It is equiva-
                                                            #   lent to ACQ_MAXWINDOW * ADCDCM_
                                                            #   SAMPLE_R.
  ACQ_WCOEF_BITS     = 31                                   # - The number of vertical bits of the
                                                            #   complex window coefficients.
  ACQ_WCBIT_POW_HALF = 2**30                                # - equivalent to 2^(ACQ_WCOEF_BITS-1)
  ACQ_MAXNUMCAPT     = 8                                    # - Maximum iteration number of acquisi-
                                                            #   tion window in a single sequence.
                                                            #   DEBUG: There is no obvious reason to
                                                            #   set the number. We'd better to
                                                            #   change the number later.
  ACQ_CAPW_RESOL     = 8                                    # - The capture word in nano-seconds
                                                            #   prior to the decimation. It is equi-
                                                            #   valent to 4 * ADC_BBSAMP_IVL.
  ACQ_CAST_RESOL     = 128                                  # - The minimum time resolution of start
                                                            #   delay in nano-seconds. The first
                                                            #   capture window must start from the
                                                            #   multiple of 128 ns to maintain the
                                                            #   the phase coherence.
  ACQ_MODENUMBER     = ['1', '2', '3', 'A','B' ]
  ACQ_MODEFUNC       = {'1': (False,False,False),           # ACQ_MODEFUNC
                        '2': ( True,False,False),           # - The values in the dictionary are
                        '3': ( True, True,False),           #   tuples of enable/disable booleans of
                        'A': ( True,False, True),           #   functions: decimation, averaging,
                        'B': ( True, True, True) }          #   and summation.

  DAQ_INITLEN        = 8192                                 # nano-seconds -> 1,024 AWG Word
  DAQ_INITREPTIME    = 30720                                # nano-seconds -> 3,840 AWG Word
  DAQ_INITSHOTS      = 1                                    # one shot
  DAQ_INITTOUT       = 5                                    # seconds
  ACQ_INITMODE       = '3'
  ACQ_INITWINDOW     = [(0,2048)]                           # initial demodulation windows
  ACQ_INITFIRCOEF    = np.array([1]*8).astype(complex)      # initial complex FIR filter coeffs
  ACQ_INITWINDCOEF   = np.array([]).astype(complex)         # initial complex window coeffs
  DAC_CNXT_TAG       = 'awgs'                               # used in the device context
  ACQ_CNXT_TAG       = 'muxs'                               # used in the device context
  # DAQ_TRIG_TAG       = 'trigger'                          # used in the device context DEBUG OBSOLETED
  DAQ_TOUT_TAG       = 'timeout'                            # used in the device context
  SRV_IPLSI_TAG      = 'ip_lsi'                             # refered in the json config
  SRV_IPFPGA_TAG     = 'ip_fpga'                            # refered in the json config
  SRV_IPCLK_TAG      = 'ip_sync'                            # refered in the json config
  SRV_QUBETY_TAG     = 'type'                               # refered in the json config; either
                                                            # 'A' or 'B' is allowed for the value
  SRV_CHANNEL_TAG    = 'channels'                           # refered in the json config
  CNL_NAME_TAG       = 'name'                               # used in the json config. channel(CNL) name.
  CNL_TYPE_TAG       = 'type'                               # used in the json config. channel(CNL) type.
                                                            # either value is to be specified:
  CNL_CTRL_VAL       = 'control'                            #  + the channel is for control
  CNL_READ_VAL       = 'mux'                                #  + the channel is for readout
  CNL_MIXCH_TAG      = 'mixer_ch'                           # used in the json config. mixer channel(CNL).
  CNL_MIXSB_TAG      = 'mixer_sb'                           # used in the json config. mixer channel(CNL)
                                                            # side-band selection. Either value can be set
  CNL_MXUSB_VAL      = 'usb'                                #  + upper sideband
  CNL_MXLSB_VAL      = 'lsb'                                #  + lower sideband
  CNL_GPIOSW_TAG     = 'gpio_mask'                          # used in the json config for gpio-controlled
                                                            # microwave switches. '1'=0xb1 deactivates
                                                            # channel or makes it loopback.

  def __init__(self):
    pass

class QSMessage:
  CONNECTING_CHANNEL = 'connecting to {}'
  CHECKING_QUBEUNIT  = 'Checking {} ...'
  CNCTABLE_QUBEUNIT  = 'Link possible: {}'
  CONNECTED_CHANNEL  = 'Link : {}'

  ERR_HOST_NOTFOUND  = 'QuBE {} not found (ping unreachable). '
  ERR_DEV_NOT_OPEN   = 'Device is not open'
  ERR_FREQ_SETTING   = '{} accepts a frequency multiple of {} MHz. '
  ERR_REP_SETTING    = '{} accepts a multiple of {} ns. '
  ERR_INVALID_DEV    = 'Invalid device. You may have called {} specific API in {}. '
  ERR_INVALID_RANG   = 'Invalid range. {} must be between {} and {}. '
  ERR_INVALID_ITEM   = 'Invalid data. {} must be one of {}. '
  ERR_INVALID_WIND   = 'Invalid window range. '
  ERR_INVALID_WAVD   = 'Invalid waveform data. '                                         \
                     + '(1) Inconsistent number of waveforms and channels. '             \
                     + '(2) The number of channels are less than that of # of awgs. '    \
                     + '(3) The sequence length in nano-second must be identical to '    \
                     + 'the value set by daq_length(). '                                 \
                     + '(4) The data length must be multiple of {}. '                    \
                       .format(QSConstants.DAQ_SEQL_RESOL // QSConstants.DAC_WVSAMP_IVL) \
                     + '(5) The absolute value of complex data is less than 1. '         \
                     + 'The problem is {}. '
  ERR_NOARMED_DAC    = 'No ready dac channels. '

  def __init__(self):
    pass



############################################################
#
# TOOLS
#

def pingger(host):
  cmd = "ping -c 1 -W 2 %s" % host
  with open(os.devnull,'w') as f:
    resp = subprocess.call(cmd.split(' '), stdout=f,stderr=subprocess.STDOUT )
  return resp

############################################################
#
# DEVICE WRAPPERS
#
# X class tree
#
# labrad.devices.DeviceWrapper
#  |
#  + QuBE_DeviceBase
#    |
#    + QuBE_ControlFPGA -.
#    |                    .
#    |                     +-+-- QuBE_ControlLine ----------.
#    |                    /  |    |                          .
#    + QuBE_ControlLSI --/   |    + QuBE_ReadoutLine ----+--- .-- QuBE_ReadoutLine_debug_otasuke
#                            |                          /      .
#                            +-- QuBE_Device_debug_otasuke ----+- QuBE_ControlLine_debug_otasuke
#

class QuBE_DeviceBase(DeviceWrapper):
  @inlineCallbacks
  def connect(self,*args,**kw):                             # @inlineCallbacks
    name, role = args
    self._name      = name
    self._role      = role
    self._chassis   = kw[ 'chassis' ]

    print(QSMessage.CONNECTING_CHANNEL.format(name))
    yield self.get_connected(*args,**kw)
    yield print(QSMessage.CONNECTED_CHANNEL.format(self._name))

  @inlineCallbacks
  def get_connected(self,*args,**kwargs):                   # @inlineCallbacks

    yield

  @property
  def device_name(self):                                    # @property
    return self._name

  @property
  def device_role(self):                                    # @property
    return self._role

  @property
  def chassis_name(self):
    return self._chassis

  def static_check_value(self,value,resolution,multiplier=50,include_zero=False):
    resp = resolution > multiplier * abs(((2*value + resolution) % (2*resolution)) - resolution)
    if resp:
      resp = ((2*value + resolution) // (2*resolution)) > 0 if not include_zero else True
    return resp

class QuBE_Control_FPGA(QuBE_DeviceBase):

  @inlineCallbacks
  def get_connected(self,*args,**kw ):                      # @inlineCallbacks

    yield super(QuBE_Control_FPGA,self).get_connected(*args, **kw)

    self.__initialized = False
    try:
      self._shots           = QSConstants.DAQ_INITSHOTS
      self._reptime         = QSConstants.DAQ_INITREPTIME
      self._seqlen          = QSConstants.DAQ_INITLEN

      self._awg_ctrl   = kw[   'awg_ctrl' ]
      self._awg_ch_ids = kw[ 'awg_ch_ids' ]
      self._awg_chs    = len(self._awg_ch_ids)

      self.__initialized     = True
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)

    if self.__initialized:
      pass
    yield

  @property
  def number_of_shots(self):                                # @property
    return int(self._shots)
  @number_of_shots.setter
  def number_of_shots(self,value):                          # @number_of_shots.setter
    self._shots = int(value)

  @property
  def repetition_time(self):                                # @property
    return int(self._reptime)
  @repetition_time.setter
  def repetition_time(self,value_in_ns):                    # @repetition_time.setter
    self._reptime = int(((value_in_ns+QSConstants.DAQ_REPT_RESOL/2)//QSConstants.DAQ_REPT_RESOL) \
                         *QSConstants.DAQ_REPT_RESOL)

  @property
  def sequence_length(self):                                # @property
    return int(self._seqlen)
  @sequence_length.setter
  def sequence_length(self,value):                          # @sequence_length.setter
    self._seqlen = value

  @property
  def number_of_awgs(self):                                 # @property
    return self._awg_chs

  def get_awg_id(self, channel):
    return self._awg_ch_ids[channel]

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

  def upload_waveform(self,waveforms,channels):

    wait_words = int( ((self.repetition_time - self.sequence_length)
                      +QSConstants.DAC_WORD_IVL/2)  // QSConstants.DAC_WORD_IVL)

    for _waveform,_channel in zip(waveforms, channels):
      wave_seq  = WaveSequence( num_wait_words = 0, num_repeats = self.number_of_shots )
      iq_samples = list(zip(*self.static_DACify(_waveform)))
      wave_seq.add_chunk( iq_samples      = iq_samples,
                          num_blank_words = wait_words,
                          num_repeats     = 1 )
      self._awg_ctrl.set_wave_sequence(self._awg_ch_ids[_channel], wave_seq )
    return True

  def start_daq(self,awg_ids):
    self._awg_ctrl.start_awgs(*awg_ids)

  def stop_daq(self,awg_ids,timeout):
    self._awg_ctrl.wait_for_awgs_to_stop(timeout, *awg_ids)

  def static_DACify(self, waveform):
    return ((np.real(waveform) * QSConstants.DAC_BITS_POW_HALF).astype(int),
            (np.imag(waveform) * QSConstants.DAC_BITS_POW_HALF).astype(int))

  def static_check_repetition_time(self,reptime_in_nanosec):
    resolution = QSConstants.DAQ_REPT_RESOL
    return self.static_check_value(reptime_in_nanosec,resolution)

  def static_check_sequence_length(self,seqlen_in_nanosec):
    resolution = QSConstants.DAQ_SEQL_RESOL
    resp = self.static_check_value(seqlen_in_nanosec,resolution)
    if resp:
      resp = seqlen_in_nanosec < QSConstants.DAQ_MAXLEN
    return resp


class QuBE_Control_LSI(QuBE_DeviceBase):

  @inlineCallbacks
  def get_connected(self,*args,**kw):                       # @inlineCallbacks

    yield super(QuBE_Control_LSI,self).get_connected(*args, **kw)

    self.__initialized = False
    try:
      self.nco_ctrl   = kw[ 'nco_device' ]
      self.lo_ctrl    = kw[  'lo_device' ]
      self.mix_ctrl   = kw[ 'mix_device' ]

      self.cnco_id    = kw[    'cnco_id' ]
      self.fnco_ids   = kw[    'fnco_id' ]
      self.fnco_chs   = len(self.fnco_ids)
      self.mix_usb_lsb= kw[     'mix_sb' ]

      self.lo_frequency     = self.get_lo_frequency()       # print(self._name,'local',self.lo_frequency)
      self.coarse_frequency = self.get_dac_coarse_frequency()
                                                            # print(self._name,'nco',self.coarse_frequency)         # DEBUG
      self.__initialized     = True
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)

    if self.__initialized:
      self.fine_frequencies = [0     for i in range(self.fnco_chs)]
    yield

  def get_lo_frequency(self):
    return self.lo_ctrl.read_freq_100M()*100

  def set_lo_frequency(self,freq_in_mhz):
    return self.lo_ctrl.write_freq_100M(int(freq_in_mhz//100))

  def get_mix_sideband(self):
    resp = self.mix_ctrl.read_mode() & 0x0400
    if 0x400 == resp:
      return QSConstants.CNL_MXLSB_VAL
    else:
      return QSConstants.CNL_MXUSB_VAL

  def set_mix_sideband(self,sideband : str):
    if   QSConstants.CNL_MXUSB_VAL == sideband:
      self.mix_ctrl.set_usb()
    elif QSConstants.CNL_MXLSB_VAL == sideband:
      self.mix_ctrl.set_lsb()
    else:
      return
    self.mix_usb_lsb = sideband

  def get_dac_coarse_frequency(self):
    return self.static_get_dac_coarse_frequency(self.nco_ctrl,self.cnco_id)

  def set_dac_coarse_frequency(self,freq_in_mhz):
    self.nco_ctrl.set_nco(1e6*freq_in_mhz, self.cnco_id, \
                                           adc_mode = False, fine_mode=False)
    self.coarse_frequency = freq_in_mhz

  def get_dac_fine_frequency(self,channel):
    return self.fine_frequencies[channel]                   # - DEBUG better to obtain frequency
                                                            #   information from the deivices
  def set_dac_fine_frequency(self,channel,freq_in_mhz):
    if freq_in_mhz < 0:
      freq_in_mhz = QSConstants.NCO_SAMPLE_F + freq_in_mhz
    self.nco_ctrl.set_nco(1e6*freq_in_mhz, self.fnco_ids[channel], \
                                           adc_mode = False, fine_mode=True)
    self.fine_frequencies[channel] = freq_in_mhz

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
    resp = self.static_check_value(freq_in_mhz,resolution,include_zero=True)
    if resp:
      resp = -QSConstants.NCO_SAMPLE_F < freq_in_mhz and \
                                         freq_in_mhz < QSConstants.NCO_SAMPLE_F
    return resp

class QuBE_ControlLine(QuBE_Control_FPGA, QuBE_Control_LSI):

  @inlineCallbacks
  def get_connected(self,*args,**kw ):                      # @inlineCallbacks
    super(QuBE_ControlLine,self).get_connected(*args,**kw)
    yield

class QuBE_ReadoutLine(QuBE_ControlLine):

  @inlineCallbacks
  def get_connected(self,*args,**kw):                       # @inlineCallbacks

    yield super(QuBE_ReadoutLine,self).get_connected(*args,**kw)

    self.__initialized = False
    try:
      self._cap_ctrl   = kw[      'cap_ctrl' ]
      self._cap_mod_id = kw[    'cap_mod_id' ]
      self._cap_unit   = kw[ 'capture_units' ]
      self._rxcnco_id  = kw[      'cdnco_id' ]

      self._rx_coarse_frequency = self.get_adc_coarse_frequency()
                                                            # print(self._name,'rxnco',self._rx_coarse_frequency)
      self.__initialized = True
    except Exception  as e:
      print(sys._getframe().f_code.co_name,e)

    if self.__initialized:
      self._window        = [QSConstants.ACQ_INITWINDOW   for i in range(QSConstants.ACQ_MULP)]
      self._window_coefs  = [QSConstants.ACQ_INITWINDCOEF for i in range(QSConstants.ACQ_MULP)]
      self._fir_coefs     = [QSConstants.ACQ_INITFIRCOEF  for i in range(QSConstants.ACQ_MULP)]
      self._acq_mode      = [QSConstants.ACQ_INITMODE     for i in range(QSConstants.ACQ_MULP)]

  def get_capture_module_id(self):
    return self._cap_mod_id

  def get_capture_unit_id(self, mux_channel):
    return self._cap_unit [ mux_channel ]

  @property
  def acquisition_window(self):                             # @property
    return copy.copy(self._window)

  def set_acquisition_window(self,mux,window):
    self._window[mux] = window

  @property
  def acquisition_mode(self):                               # @property, only referenced in QuBE_Server
    return copy.copy(self._acq_mode)                        # .acquisition_mode() for @setting 303

  def set_acquisition_mode(self,mux,mode):
    self._acq_mode[mux] = mode

  def set_acquisition_fir_coefficient(self,muxch,coeffs):
    def fircoef_DACify(coeffs):
      return  (np.real(coeffs) * QSConstants.ACQ_FCBIT_POW_HALF).astype(int) \
          +1j*(np.imag(coeffs) * QSConstants.ACQ_FCBIT_POW_HALF).astype(int)
    self._fir_coefs[muxch]    = fircoef_DACify(coeffs)

  def set_acquisition_window_coefficient(self,muxch,coeffs):
    def window_DACify(coeffs):
      return  (np.real(coeffs) * QSConstants.ACQ_WCBIT_POW_HALF).astype(int) \
          +1j*(np.imag(coeffs) * QSConstants.ACQ_WCBIT_POW_HALF).astype(int)
    self._window_coefs[muxch] = window_DACify(coeffs)

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

      |<-------- SINGLE EXPERIMENT ------>|<-------- SINGLE EXPERIMENT ------>|<-------- SINGLE EXP..

    - Given that the sum_section is defined as a pair of capture duration and
      post blank, the initial non-readout duration has to be implemented usi-
      ng capture_delay.
    - The repetition duration starts at the beginning of readout operation
      and ends at the end of 2nd control operation (just before 2nd readout)
    - The capture word is defined as the four multiple of sampling points. It
      corresponds to 4 * ADC_BBSAMP_IVL = ACQ_CAPW_RESOL (nanoseconds).
    """
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
      while len(win_word) > 1:
        _e = win_word.pop(0)
        _s = win_word.pop(0)
        blank_length   = _s - _e
        section_length = _e - _s0
        _s0 = _s
        param.add_sum_section(section_length, blank_length)

      self.configure_readout_mode(mux,param,self._acq_mode[mux])
                                                            # import pickle
                                                            # import base64
                                                            # print('mux setup')
                                                            # print(base64.b64encode(pickle.dumps(param)))
      self._cap_ctrl.set_capture_params(self._cap_unit[mux], param)
    return True

  def configure_readout_mode(self,mux,param,mode):
    """
    Configure readout parametes to acquisition modes.

    It enables and disables decimation, averaging, and summation operations with
    filter coefficients and the number of averaging.

    Args:
        param     : e7awgsw.captureparam.CaptureParam
        mode      : character
            Acceptable parameters are '1', '2', '3', 'A', 'B'
    """
    dsp = self.configure_readout_dsp(mux,param,mode)
    param.sel_dsp_units_to_enable(*dsp)

  def configure_readout_dsp(self,mux,param,mode):
    dsp = []
    decim,averg,summn = QSConstants.ACQ_MODEFUNC[mode]

    resp = self.configure_readout_decimation(mux,param,decim); dsp.extend(resp)
    resp = self.configure_readout_averaging (mux,param,averg); dsp.extend(resp)
    resp = self.configure_readout_summation (mux,param,summn); dsp.extend(resp)
    return dsp

  def configure_readout_decimation(self,mux,param,decimation):
    """
    Configure readout mux channel parameters.

    [Decimation] 500MSa/s datapoints are reduced to 125 MSa/s (8ns interval)

    Args:
        param     : e7awgsw.captureparam.CaptureParam
        decimation: bool
    Returns:
        dsp       : list.
            The list of enabled e7awgsw.hwdefs.DspUnit objects
    """
    dsp = list()
    if decimation:
      param.complex_fir_coefs = list(self._fir_coefs[mux])
      dsp.append(DspUnit.COMPLEX_FIR)
      dsp.append(DspUnit.DECIMATION )
    return dsp

  def configure_readout_averaging(self,mux,param,averaging):
    """
    Configure readout mux channel parameters.

    [Averaging] Averaging datapoints for all experiments.

    Args:
        param    : e7awgsw.captureparam.CaptureParam
        average  : bool
    Returns:
        dsp      : list.
            The list of enabled e7awgsw.hwdefs.DspUnit objects
    """
    dsp = list()
    if averaging:
      dsp.append(DspUnit.INTEGRATION)
    param.num_integ_sections = int(self.number_of_shots)
    return dsp

  def configure_readout_summation(self,mux,param,summation):
    """
    Configure readout mux channel parameters.

    [Summation] For a given readout window, the DSP apply complex window filter.
    (This is equivalent to the convolution in frequency domain of a filter
    function with frequency offset). Then, DSP sums all the datapoints
    in the readout window.

    Args:
        param    : e7awgsw.captureparam.CaptureParam
        summation: bool
    Returns:
        dsp      : list
            The list of enabled e7awgsw.hwdefs.DspUnit objects
    """
    dsp = list()
    if summation:
      param.sum_start_word_no    = 0
      param.num_words_to_sum     = CaptureParam.MAX_SUM_SECTION_LEN
      param.complex_window_coefs = list(self._window_coefs[mux])
      dsp.append(DspUnit.COMPLEX_WINDOW)
      dsp.append(DspUnit.SUM)
    else:
      pass
    return dsp

  def download_waveform(self, muxchs):
    """
    Download captured waveforms (datapoints)

    Transfer datapoints from FPGA to a host computer.

    Args:
        muxchs : List[int]
            A list of the readout mux channels for transfer.
    Returns:
        datapoints: *2c
            Two-dimensional complex data matrix. The row corrsponds to the
            readout mux channel and the column of the matrix is time dimention
            of datapoints.
    """

    vault = []
    for mux in muxchs:
      data = self.download_single_waveform(mux)
      vault.append(data)
    return np.vstack(vault)

  def download_single_waveform(self, muxch):
    capture_unit   = self._cap_unit[muxch]

    n_of_samples   = self._cap_ctrl.num_captured_samples(capture_unit)
    iq_tuple_data  = self._cap_ctrl.get_capture_data(capture_unit, n_of_samples)

    return np.array([(_i+1j*_q) for _i,_q in iq_tuple_data]).astype(complex)

  def set_trigger_board(self, trigger_board, enabled_capture_units):
    self._cap_ctrl.select_trigger_awg(self._cap_mod_id, trigger_board)
    self._cap_ctrl.enable_start_trigger(*enabled_capture_units)

  def set_adc_coarse_frequency(self,freq_in_mhz):
    self.nco_ctrl.set_nco(1e6*freq_in_mhz, self._rxcnco_id, \
                                           adc_mode = True, fine_mode=False)
    self._rx_coarse_frequency = freq_in_mhz # DEBUG seems not used right now

  def get_adc_coarse_frequency(self):
    return self.static_get_adc_coarse_frequency(self.nco_ctrl,self._rxcnco_id)

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
    def check_duration(start,end):
      return False if start > end or end - start > QSConstants.ACQ_MAXWINDOW else True

    if 0 != list_of_windows[0][0] % QSConstants.ACQ_CAST_RESOL:
      return False

    for _s,_e in list_of_windows:
      if not check_value(_s) or not check_value(_e) or not check_duration(_s,_e):
        return False

    return True

  def static_check_acquisition_fir_coefs(self,coeffs):
    length = len(coeffs)

    resp = QSConstants.ACQ_MAX_FCOEF >= length
    if resp:
      resp = 1.0 > np.max(np.abs(coeffs))
    return resp

  def static_check_acquisition_window_coefs(self,coeffs):
    length = len(coeffs)

    resp = QSConstants.ACQ_MAX_WCOEF >= length
    if resp:
      resp = 1.0 > np.max(np.abs(coeffs))
    return resp

############################################################
#
# QUBE SERVER
#

class QuBE_Server(DeviceServer):
  name          =   QSConstants.SRVNAME
  deviceWrappers= { QSConstants.CNL_READ_VAL: QuBE_ReadoutLine,
                    QSConstants.CNL_CTRL_VAL: QuBE_ControlLine }
  possibleLinks = { }
  adi_api_path  = None

  @inlineCallbacks
  def initServer(self):                                     # @inlineCallbacks
    yield DeviceServer.initServer(self)

    cxn = self.client
    reg = cxn[QSConstants.REGSRV]
    try:
      yield reg.cd(QSConstants.REGDIR)
      config = yield reg.get(QSConstants.REGLNK)
      self.possibleLinks = json.loads(config)
      self.master_link   = yield reg.get(QSConstants.REGMASTERLNK)
      self.adi_api_path  = yield reg.get(QSConstants.REGAPIPATH)
      self._master_ctrl  = yield QuBESequencerMaster(self.master_link)
      self._sync_ctrl    = dict()
      self.__is_clock_opened = True
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)

                                                            # reserved for Suzuki-san
                                                            #max_workers      = QSConstants.THREAD_MAX_WORKERS
                                                            #self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

  def initContext(self, c):
    DeviceServer.initContext(self,c)
    c[QSConstants.DAC_CNXT_TAG] = dict()
    c[QSConstants.ACQ_CNXT_TAG] = dict()
    c[QSConstants.DAQ_TOUT_TAG] = QSConstants.DAQ_INITTOUT

  def chooseDeviceWrapper(self, *args, **kw):
    tag = QSConstants.CNL_READ_VAL if QSConstants.CNL_READ_VAL in args[2] else \
                                      QSConstants.CNL_CTRL_VAL
    return self.deviceWrappers[tag]

  def instantiateChannel(self,name,channels,awg_ctrl,cap_ctrl,lsi_ctrl):
    def gen_awg(name,role,chassis,channel,awg_ctrl,cap_ctrl,lsi_ctrl):
      awg_ch_ids = channel[  'ch_dac']
      cnco_id    = channel['cnco_dac']
      fnco_id    = channel['fnco_dac']
      lo_id      = channel[  'lo_dac']
      mix_id     = channel[QSConstants.CNL_MIXCH_TAG]
      mix_sb     = channel[QSConstants.CNL_MIXSB_TAG]
      nco_device = lsi_ctrl.ad9082  [cnco_id[0]]
      lo_device  = lsi_ctrl.lmx2594 [lo_id]
      mix_device = lsi_ctrl.adrf6780[mix_id]

      args = name, role
      kw   = dict ( awg_ctrl   = awg_ctrl,
                    awg_ch_ids = awg_ch_ids,
                    nco_device = nco_device,
                    cnco_id    = cnco_id [1],
                    fnco_id    = [_id for _chip,_id in fnco_id ],
                    lo_device  = lo_device,
                    mix_device = mix_device,
                    mix_sb     = mix_sb,
                    chassis    = chassis )
      return (name,args,kw)

    def gen_mux(name,role,chassis,channel,awg_ctrl,cap_ctrl,lsi_ctrl):
      _name,_args,_kw = gen_awg(name,role,chassis,channel,awg_ctrl,cap_ctrl,lsi_ctrl)

      cap_mod_id    = channel[  'ch_adc']
      cdnco_id      = channel['cnco_adc']
      capture_units = CaptureModule.get_units(cap_mod_id)

      kw   = dict ( cap_ctrl      = cap_ctrl,
                    capture_units = capture_units,
                    cap_mod_id    = cap_mod_id,
                    cdnco_id      = cdnco_id[1] )
      _kw.update(kw)
      return (_name,_args,_kw)

    devices = []
    for channel in channels:
      channel_type= channel[QSConstants.CNL_TYPE_TAG]
      channel_name= name + '-' + channel[QSConstants.CNL_NAME_TAG]
      args        = channel_name,channel_type,name,channel,awg_ctrl,cap_ctrl,lsi_ctrl
      to_be_added = gen_awg(*args) if channel_type == QSConstants.CNL_CTRL_VAL else \
                    gen_mux(*args) if channel_type == QSConstants.CNL_READ_VAL else \
                    None
      if to_be_added is not None:
        devices.append(to_be_added)
    return devices

  def instantiateQube(self,name,info):
    try:
      ipfpga      = info[QSConstants.SRV_IPFPGA_TAG]
      iplsi       = info[QSConstants.SRV_IPLSI_TAG ]
      channels    = info['channels']
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
  def findDevices(self):                                    # @inlineCallbacks
    cxn = self.client
    found = []

    for name in self.possibleLinks.keys():
      print(QSMessage.CHECKING_QUBEUNIT.format(name))
      try:
        res =   pingger(self.possibleLinks[name][QSConstants.SRV_IPFPGA_TAG])
        if 0 == res:
          res = pingger(self.possibleLinks[name][QSConstants.SRV_IPLSI_TAG])
        if 0 != res:
          res = pingger(self.possibleLinks[name][QSConstants.SRV_IPCLK_TAG])
        if 0 != res:
          raise Exception(QSMessage.ERR_HOST_NOTFOUND.format(name))
      except Exception as e:
        print(sys._getframe().f_code.co_name,e)
        continue

      print(QSMessage.CNCTABLE_QUBEUNIT.format(name))
      devices = self.instantiateQube(name,self.possibleLinks[name])
      found.extend(devices)

      sync_ctrl = QuBESequencerClient(self.possibleLinks[name][QSConstants.SRV_IPCLK_TAG])
      self._sync_ctrl.update({name:sync_ctrl})
      yield
                                                            # print(sys._getframe().f_code.co_name,found)
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
      raise ValueError(QSMessage.ERR_REP_SETTING.format('Sequencer',QSConstants.DAQ_SEQL_RESOL)
                     + QSMessage.ERR_INVALID_RANG.format('daq_length','128 ns','{} ns'.format(QSConstants.DAQ_MAXLEN)))

  @setting(105, 'DAQ Start', returns = ['b'])
  def daq_start(self,c):
    """
    Start data acquisition

    The method name [daq_start()] is for backward compatibility with a former
    version of quantum logic analyzer, and I like it. This method finds trigger
    boards to readout FPGA circuits and give them to the boards. All the
    enabled AWGs and MUXs are supposed to be in the current context [c] through
    [._register_awg_channels()] and [_register_mux_channels()].

    Compared to the previous implementation, this method does not require
    [select_device()] before the call.
    """
    dev = self.selectedDevice(c)

    if QSConstants.CNL_READ_VAL == dev.device_role:         # Set trigger board to capture units
      self._readout_mux_start(c)

    for chassis_name in c[QSConstants.ACQ_CNXT_TAG].keys():
      for _dev, _m, _units in c[QSConstants.ACQ_CNXT_TAG][chassis_name]:
        print(chassis_name,_units)                          # DEBUG

    for chassis_name in c[QSConstants.DAC_CNXT_TAG].keys():
      _dev, _awgs = c[QSConstants.DAC_CNXT_TAG][chassis_name]
      print(chassis_name,_awgs)                             # DEBUG
    return True

  def _readout_mux_start(self,c):
    """
    Find trigger AWG bords in multiple chassis

    For each QuBE chassis, we have to select trigger AWG from the AWGs involved
    in the operation. For each QuBE readout module, [_readout_mux_start()]
    sets the trigger AWG and enables the capture units.

    """
    for chassis_name in c[QSConstants.ACQ_CNXT_TAG].keys():
      if chassis_name not in c[QSConstants.DAC_CNXT_TAG].keys():
        raise Exception(QSMessage.ERR_NOARMED_DAC)
      else:
        dev, awgs = c[QSConstants.DAC_CNXT_TAG][chassis_name]
        trigger_board = list(awgs)[0]

      for _dev,_module,_units in c[QSConstants.ACQ_CNXT_TAG][chassis_name]:
        _dev.set_trigger_board( trigger_board, _units )
    return

  @setting(106, 'DAQ Trigger', returns = ['b'])
  def daq_trigger(self,c):
    """
    Start synchronous measurement.

    Read the clock value from the master FPGA board and set a planned timing
    to the QuBE units. Measurement is to start at the given timing.

    """
    if 1 > len(c[QSConstants.DAC_CNXT_TAG].keys()):
      return False                                          # Nothing to start.
    delay = 1*125*1000*1000

    clock = self._master_ctrl.read_clock() + delay
    for chassis_name in c[QSConstants.DAC_CNXT_TAG].keys():
      dev, enabled_awgs = c[QSConstants.DAC_CNXT_TAG][chassis_name]
      self._sync_ctrl[chassis_name].add_sequencer(clock,list(enabled_awgs))

    return True

  @setting(107, 'DAQ Stop', returns = ['b'])
  def daq_stop(self,c):
    """
    Wait until synchronous measurement is done.

    """
    if 1 > len(c[QSConstants.DAC_CNXT_TAG].keys()):
      return False                                          # Nothing to stop

    for chassis_name in c[QSConstants.DAC_CNXT_TAG].keys():

      dev, enabled_awgs = c[QSConstants.DAC_CNXT_TAG][chassis_name]
      dev.stop_daq(list(enabled_awgs),c[QSConstants.DAQ_TOUT_TAG])

    return True

  @setting(108, 'DAQ Timeout', t = ['v[s]'], returns = ['v[s]'])
  def daq_timeout(self,c,t = None):
    if t is None:
      val = c[QSConstants.DAQ_TOUT_TAG]
      return T.Value(val,'s')
    else:
      c[QSConstants.DAQ_TOUT_TAG] = t['s']
      return t

  @setting(110, 'DAC Channels', returns = ['w'])
  def daq_channels(self,c):
    """
    Retrieve the number of available AWG channels. The number of available AWG c
    hannels is configured through adi_api_mod/v1.0.6/src/helloworld.c and the
    lane information is stored in the registry /Servers/QuBE/possible_links.

    Returns:
        channels : w
            The number of available AWG channels.
    """
    dev = self.selectedDevice(c)
    return dev.number_of_awgs

  @setting(200, 'Upload Parameters', channels=['w','*w'],returns=['b'])
  def upload_parameters(self,c,channels):
    """
    Upload channel parameters.

    Sequence setting.

    Args:
        channels : w, *w
            waveform channel   0 to 2 [The number of waveform channels - 1]
    Returns:
        success  : b
            True if successful.
    """
    dev  = self.selectedDevice(c)
    channels = np.atleast_1d(channels).astype(int)
    if not dev.check_awg_channels(channels):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('awg index', 0, dev.number_of_awgs - 1))
    return self._register_awg_channels(c,dev,channels)

  def _register_awg_channels(self,c,dev,channels):
    """
    Register selected DAC AWG channels

    The method [_register_awg_channels()] register the enabled AWG IDs to the device
    context. This information is used in daq_start() and daq_trigger()

    Data structure:
      qube010: (dev, set{0,1,2,3,...}),
      qube011: (dev, set{0,2,15,..})
    """
    chassis_name = dev.chassis_name

    if chassis_name not in c[QSConstants.DAC_CNXT_TAG].keys():
      c[QSConstants.DAC_CNXT_TAG].update({chassis_name:(dev,set())})

    _to_be_added = list()
    for channel in channels:
      _dev, awgs  = c[QSConstants.DAC_CNXT_TAG][chassis_name]

      _to_be_added = dev.get_awg_id(channel)
      awgs.add(_to_be_added)
      c[QSConstants.DAC_CNXT_TAG][chassis_name] = (_dev,awgs)

    return True

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
    if QSConstants.CNL_READ_VAL != dev.device_role:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.device_name))

    muxchs = np.atleast_1d(muxchs).astype(int)
    for _mux in muxchs:
      if not dev.static_check_mux_channel_range(_mux):
        raise ValueError(QSMessage.ERR_INVALID_RANG.format('muxch', 0, QSConstants.ACQ_MULP - 1))
    resp =  dev.upload_readout_parameters(muxchs)
    if resp:
      resp = self._register_mux_channels(c,dev,muxchs)
    return resp

  def _register_mux_channels(self,c,dev,selected_mux_channels):
    """
    Register selected readout channels

    The method [_register_mux_channels()] register the selected capture module
    IDs and the selected capture units to the device context. This information
    is used in daq_start() and daq_trigger().

    """
    chassis_name = dev.chassis_name
    module_id    = dev.get_capture_module_id()
    unit_ids     = [dev.get_capture_unit_id(_s) for _s in selected_mux_channels]

    if chassis_name not in c[QSConstants.ACQ_CNXT_TAG].keys():
      c[QSConstants.ACQ_CNXT_TAG].update({chassis_name : list()})

    registered_ids = [_id for _d,_id,_u in c[QSConstants.ACQ_CNXT_TAG][chassis_name]]
    try:
      addition = False
      idx = registered_ids.index(module_id)
    except ValueError as e:
      c[QSConstants.ACQ_CNXT_TAG][chassis_name].append((dev,module_id,unit_ids))
    else:
      addition = True

    if addition:
      _dev, _module_id, registered_units = c[QSConstants.ACQ_CNXT_TAG][chassis_name][idx]
      registered_units.extend( [unit_id \
                    for unit_id in unit_ids if unit_id not in registered_units])
      c[QSConstants.ACQ_CNXT_TAG][chassis_name][idx] = (dev,module_id,registered_units)

    return True

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
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('awg index', 0, dev.number_of_awgs - 1))

    resp,number_of_chans,data_length = dev.check_waveform(waveforms,channels)
    if not resp:
      raise ValueError(QSMessage.ERR_INVALID_WAVD.format(number_of_chans))

    return dev.upload_waveform(waveforms,channels)

  @setting(203, 'Download Waveform', muxchs = ['*w','w'], returns = ['*c','*2c'])
  def download_waveform(self,c,muxchs):
    """
    Download acquired waveforms (or processed data points).

    Transfer waveforms or datapoints from Alevo FPGA to a host computer.

    Args:
        muxchs  : *w, w

    Returns:
        data    : *2c,*c
    """
    dev = self.selectedDevice(c)
    if QSConstants.CNL_READ_VAL != dev.device_role:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.device_name))

    muxchs  = np.atleast_1d(muxchs).astype(int)
    for _mux in muxchs:
      if not dev.static_check_mux_channel_range(_mux):
        raise ValueError(QSMessage.ERR_INVALID_RANG.format('muxch', 0, QSConstants.ACQ_MULP - 1))

    data = dev.download_waveform(muxchs)

    return data


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
    if QSConstants.CNL_READ_VAL != dev.device_role:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.device_name))
    elif not dev.static_check_mux_channel_range(muxch):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('muxch', 0, QSConstants.ACQ_MULP - 1))
    elif acqnumb is None:
      return dev.acquisition_number_of_windows[muxch]
    elif 0 < acqnumb and acqnumb <= QSConstants.ACQ_MAXNUMCAPT:
      dev.acquisition_number_of_windows[muxch] = acqnumb
      return acqnumb
    else:
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('Acquisition number of windows', 1, QSConstants.ACQ_MAXNUMCAPT))

  @setting(302, 'Acquisition Window', muxch = ['w'], window = ['*(v[s]v[s])'], returns=['*(v[s]v[s])'])
  def acquisition_window(self,c,muxch,window = None):
    """
    Read and write acquisition windows.

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
    if QSConstants.CNL_READ_VAL != dev.device_role:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.device_name))
    elif not dev.static_check_mux_channel_range(muxch):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('muxch', 0, QSConstants.ACQ_MULP - 1))
    elif window is None:
      return [(T.Value(_s,'ns'),T.Value(_e,'ns')) for _s,_e in dev.acquisition_window[muxch]]

    wl = [(int(_w[0]['ns']+0.5),int(_w[1]['ns']+0.5)) for _w in window]
    if dev.static_check_acquisition_windows(wl):
      dev.set_acquisition_window(muxch,wl)
      return window
    else:
      raise ValueError(QSMessage.ERR_INVALID_WIND)

  @setting(303, 'Acquisition Mode', muxch = ['w'], mode = ['s'], returns=['s'])
  def acquisition_mode(self,c,muxch,mode = None):
    """
    Read and write acquisition mode

    Five (or six) acquisition modes are defined, i.e., 1, 2, 3, A, B, (C) for
    predefined experiments.

    SIGNAL PROCESSING MAP <MODE NUMBER IN THE FOLLOWING TABLES>

      DECIMATION = NO
                          |       Averaging       |
                          |    NO     |   YES     |
            ------+-------+-----------+-----------+--
             SUM |   NO   |           |     1     |
             MAT +--------+-----------|-----------+--
             ION |  YES   |           |           |

      DECIMATION = YES
                          |       Averaging       |
                          |    NO     |   YES     |
            ------+-------+-----------+-----------+--
             SUM |   NO   |     2     |     3     |
             MAT +--------+-----------|-----------+--
             ION |  YES   |     A     |     B     |

      DECIMATION = YES / BINARIZE = YES
                          |       Averaging       |
                          |    NO     |   YES     |
            ------+-------+-----------+-----------+--
             SUM |   NO   |           |           |
             MAT +--------+-----------|-----------+--
             ION |  YES   |     C     |           |

    DEBUG, The mode "C" has not been implemented yet.

    Args:
        muxch    : w
            multiplex channel   0 to 3 [QSConstants.ACQ_MULP-1]

        mode     : s
            Acquisition mode. one of '1', '2', '3', 'A', 'B' can be set.

    Returns:
        mode     : s
    """
    dev = self.selectedDevice(c)
    if QSConstants.CNL_READ_VAL != dev.device_role:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.device_name))
    elif not dev.static_check_mux_channel_range(muxch):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('muxch', 0, QSConstants.ACQ_MULP - 1))
    elif mode is None:
      return dev.acquisition_mode[muxch]
    elif mode in QSConstants.ACQ_MODENUMBER:
      dev.set_acquisition_mode(muxch,mode)
      return mode
    else:
      raise ValueError(QSMessage.ERR_INVALID_ITEM.format( 'Acquisition mode',','.join(QSConstants.ACQ_MODENUMBER)))

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
    if QSConstants.CNL_READ_VAL != dev.device_role:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.device_name))
    elif muxch is not None and not dev.static_check_mux_channel_range(muxch):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('muxch', 0, QSConstants.ACQ_MULP - 1))
    else:
      chassis_name = dev.chassis_name
      resp         = chassis_name in c[QSConstants.ACQ_CNXT_TAG].keys()
      if resp:
        module_enabled = c[QSConstants.ACQ_CNXT_TAG][chassis_name]
        resp           = True
        try:
           idx = [_m for _d, _m, _u in module_enabled].index(dev.get_capture_module_id())
        except ValueError as e:
          resp = False
      if resp:
        _d, _m, unit_enabled = module_enabled[idx]
        if muxch is not None:
          resp = dev.get_capture_unit_id(muxch) in unit_enabled
          result = True if resp else False
        else:
          result = [(dev.get_capture_unit_id(i) in unit_enabled) for i in range(QSConstants.ACQ_MULP)]
      else:
        result = [False for _i in range(QSConstants.ACQ_MULP)] if muxch is None else \
                  False
      return result

  @setting(305, 'Filter Pre Coefficients', muxch = ['w'], coeffs = ['*c'], returns = ['b'])
  def filter_pre_coefficients(self,c,muxch,coeffs):
    """
    Set complex FIR coefficients to a mux channel. (getting obsoleted)
    """
    self.acquisition_fir_coefficients(c,muxch,coeffs)
    raise Exception("Tabuchi wants to rename the API to acquisition_fir_coefficients")

  @setting(306, 'Average Window Coefficients', muxch = ['w'], coeffs = ['*c'], returns = ['b'])
  def set_window_coefficients(self,c,muxch,coeffs):
    """
    Set complex window coefficients to a mux channel. (getting obsoleted)
    """
    self.acquisition_window_coefficients(c,muxch,coeffs)
    raise Exception("Tabuchi wants to rename the API to acquisition_window_coefficients")

  @setting(307, 'Acquisition FIR Coefficients', muxch = ['w'], coeffs = ['*c'], returns = ['b'])
  def acquisition_fir_coefficients(self,c,muxch,coeffs):
    """
    Set complex FIR (finite impulse response) filter coefficients to a mux channel.

    In the decimation DSP logic, a 8-tap FIR filter is applied before decimation.

    Args:
        muxch : w
            Multiplex readout mux channel. 0-3 can be set

        coeffs : *c
            Complex window coefficients. The absolute values of the coeffs has
            to be less than 1.

    Returns:
        success: b
    """
    dev = self.selectedDevice(c)
    if QSConstants.CNL_READ_VAL != dev.device_role:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.device_name))
    elif not dev.static_check_mux_channel_range(muxch):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('muxch', 0, QSConstants.ACQ_MULP - 1))
    elif not dev.static_check_acquisition_fir_coefs(coeffs):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('abs(coeffs)',0,1)  \
                    +  QSMessage.ERR_INVALID_RANG.format('len(coeffs)',1,QSConstants.ACQ_MAX_FCOEF))
    else:
      dev.set_acquisition_fir_coefficient(muxch,coeffs)
    return True

  @setting(308, 'Acquisition Window Coefficients', muxch = ['w'], coeffs = ['*c'], returns = ['b'])
  def acquisition_window_coefficients(self,c,muxch,coeffs):
    """
    Set complex window coefficients to a mux channel.

    In the summation DSP logic, a readout signal is multipled by the window
    coefficients before sum operatation for weighted demodulation.

    Args:
        muxch  : w
            Multiplex readout mux channel. 0-3 can be set

        coeffs : *c
            Complex window coefficients. The absolute values of the coeffs has
            to be less than 1.

    Returns:
        success: b
    """
    dev = self.selectedDevice(c)
    if QSConstants.CNL_READ_VAL != dev.device_role:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.device_name))
    elif not dev.static_check_mux_channel_range(muxch):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('muxch', 0, QSConstants.ACQ_MULP - 1))
    elif not dev.static_check_acquisition_window_coefs(coeffs):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('abs(coeffs)',0,1)  \
                    +  QSMessage.ERR_INVALID_RANG.format('len(coeffs)',1,QSConstants.ACQ_MAX_WCOEF))
    else:
      dev.set_acquisition_window_coefficient(muxch,coeffs)
    return True

  @setting(400, 'Frequency Local', frequency = ['v[Hz]'], returns = ['v[Hz]'])
  def local_frequency(self,c,frequency = None):
    """
    Read and write frequency setting from/to local oscillators.

    The waveform singnals from D/A converters is upconverted using local osci-
    llators (LMX2594).

    Args:
        frequency: v[Hz]
            The mininum frequency resolution of oscillators are 100 MHz [QSCons
            tants.DAC_LO_RESOL].

    Returns:
        frequency: v[Hz]

    """
    dev = self.selectedDevice(c)
    if frequency is None:
      resp = dev.get_lo_frequency()
      frequency = T.Value(resp,'MHz')
    elif dev.static_check_lo_frequency(frequency['MHz']):
      dev.set_lo_frequency(frequency['MHz'])
    else:
      raise ValueError(QSMessage.ERR_FREQ_SETTING.format('LO',QSConstants.DAC_CNCO_RESOL))
    return frequency

  @setting(401, 'Frequency TX NCO', frequency = ['v[Hz]'], returns = ['v[Hz]'])
  def coarse_tx_nco_frequency(self,c,frequency = None):
    """
    Read and write frequency setting from/to coarse NCOs.

    A D/A converter have multiple waveform channels. The channels have a common
    coarse NCO for upconversion. The center center frequency can be tuned with
    the coarse NCO from -6 GHz to 6 GHz.

    Args:
        frequency: v[Hz]
            The minimum resolution of NCO frequencies is 1.46484375 MHz [QSConst
            ants.DAC_CNCO_RESOL].

    Returns:
        frequency: v[Hz]

    """
    dev = self.selectedDevice(c)
    if frequency is None:
      resp = dev.get_dac_coarse_frequency()
      frequency = T.Value(resp,'MHz')
    elif dev.static_check_dac_coarse_frequency(frequency['MHz']):
      dev.set_dac_coarse_frequency(frequency['MHz'])
    else:
      raise ValueError(QSMessage.ERR_FREQ_SETTING.format('TX Corse NCO',QSConstants.DAC_CNCO_RESOL))
    return frequency

  @setting(402, 'Frequency TX Fine NCO', channel = ['w'], frequency = ['v[Hz]'], returns = ['v[Hz]'])
  def fine_tx_nco_frequency(self,c,channel,frequency = None):
    """
    Read and write frequency setting from/to fine NCOs.

    A D/A converter havs multiple waveform channels. Each channel center frequ-
    ency can be tuned using fine NCOs from -1.5 GHz to 1.5 GHz. Note that the
    maximum frequency difference is 1.2 GHz.

    Args:
        channel  : w
            The NCO channel index. The index number corresponds to that of wave-
            form channel index.
        frequency: v[Hz]
            The minimum resolution of NCO frequencies is 0.48828125 MHz [QSConst
            ants.DAC_FNCO_RESOL].

    Returns:
        frequency: v[Hz]

    """
    dev = self.selectedDevice(c)
    if not dev.check_awg_channels([channel]):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('awg index', 0, dev.number_of_awgs - 1))
    elif frequency is None:
      resp = dev.get_dac_fine_frequency(channel)
      frequency = T.Value(resp,'MHz')
    elif dev.static_check_dac_fine_frequency(frequency['MHz']):
      dev.set_dac_fine_frequency(channel,frequency['MHz'])
    else:
      raise ValueError(QSMessage.ERR_FREQ_SETTING.format('TX Fine NCO',QSConstants.DAC_FNCO_RESOL) + '\n'
                     + QSMessage.ERR_INVALID_RANG.format('TX Fine NCO frequency',
                                                         '{} MHz.'.format(-QSConstants.NCO_SAMPLE_F//2),
                                                         '{} MHz.'.format( QSConstants.NCO_SAMPLE_F//2)))
    return frequency

  @setting(403, 'Frequency RX NCO', frequency = ['v[Hz]'], returns = ['v[Hz]'])
  def coarse_rx_nco_frequency(self,c,frequency = None):
    dev = self.selectedDevice(c)
    if QSConstants.CNL_READ_VAL != dev.device_role:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.device_name))
    elif frequency is None:
      resp = dev.get_adc_coarse_frequency()
      frequency = T.Value(resp,'MHz')
    elif dev.static_check_adc_coarse_frequency(frequency['MHz']):
      dev.set_adc_coarse_frequency(frequency['MHz'])
    else:
      raise ValueError(QSMessage.ERR_FREQ_SETTING.format('RX Corse NCO',QSConstants.ADC_CNCO_RESOL))
    return frequency

  @setting(404, 'Frequency Sideband', sideband = ['s'], returns = ['s'])
  def sideband_selection(self,c,sideband = None):
    """
    Read and write the frequency sideband setting to the up- and down-conversion
    mixers.

    Args:
        sideband : s
            The sideband selection string. Either 'usb' or 'lsb' (QSConstants.CN
            L_MXUSB_VAL and QSConstants.CNL_MXLSB_VAL) can be set.

    Returns:
        sideband : s
            The current sideband selection string.
    """
    dev = self.selectedDevice(c)
    if sideband is None:
      sideband = dev.get_mix_sideband()
    elif sideband not in [QSConstants.CNL_MXUSB_VAL, QSConstants.CNL_MXLSB_VAL ]:
      raise Exception(QSMessage.ERR_INVALID_ITEM.format('The sideband string',
                           "{} or {}".format(QSConstants.CNL_MXUSB_VAL, QSConstants.CNL_MXLSB_VAL)))
    else:
      dev.set_mix_sideband(sideband)
    return sideband

class QuBE_Device_debug_otasuke(QuBE_Control_FPGA, QuBE_Control_LSI):

  @inlineCallbacks
  def get_connected(self,*args,**kw):

    yield super(QuBE_Device_debug_otasuke,self).get_connected(*args,**kw)
    self.__initialized = False
    try:
      self.__switch_mask = kw['gsw_mask']
      self.__switch_ctrl = kw['gsw_ctrl']
      self.__initialized = True
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)
    yield

  @inlineCallbacks
  def get_microwave_switch(self):

    mask = self.__switch_mask
    resp = self.__switch_ctrl.read_value()
    output = True
    if resp & mask == mask:
      output = False
    yield
    returnValue(output)

  @inlineCallbacks
  def set_microwave_switch(self,output):

    mask = self.__switch_mask
    resp = self.__switch_ctrl.read_value()
    if True == output:
      resp = resp & (0x3fff ^ mask)
    else:
      resp = resp | mask
    yield self.__switch_ctrl.write_value(resp)


class QuBE_ControlLine_debug_otasuke(QuBE_ControlLine, QuBE_Device_debug_otasuke):

  @inlineCallbacks
  def get_connected(self,*args,**kw ):                      # @inlineCallbacks
    super(QuBE_ControlLine_debug_otasuke,self).get_connected(*args,**kw)
    yield

class QuBE_ReadoutLine_debug_otasuke(QuBE_ReadoutLine, QuBE_Device_debug_otasuke):

  @inlineCallbacks
  def get_connected(self,*args,**kw ):                      # @inlineCallbacks
    super(QuBE_ReadoutLine_debug_otasuke,self).get_connected(*args,**kw)
    yield


class QuBE_Server_debug_otasuke(QuBE_Server):

  deviceWrappers= { QSConstants.CNL_READ_VAL: QuBE_ReadoutLine_debug_otasuke,
                    QSConstants.CNL_CTRL_VAL: QuBE_ControlLine_debug_otasuke }

  def __init__(self,*args,**kw):
    QuBE_Server.__init__(self,*args,**kw)

  def instantiateChannel(self,name,channels,awg_ctrl,cap_ctrl,lsi_ctrl):
    devices = super(QuBE_Server_debug_otasuke,self).\
                    instantiateChannel(name,channels,awg_ctrl,cap_ctrl,lsi_ctrl)
    revised = []
    for device, channel in zip(devices,channels):
      name, args, kw = device
      _kw = dict( gsw_ctrl = lsi_ctrl.gpio,
                  gsw_mask = channel[ QSConstants.CNL_GPIOSW_TAG])
      kw.update(_kw)
      revised.append((name,args,kw))
    return revised

  @setting(502, 'DEBUG AWG REG', addr = ['w'], offset = ['w'], pos = ['w'], bits = ['w'], data = ['w'], returns = ['w'])
  def debug_awg_ctrl_reg(self,c, addr, offset, pos, bits, data = None):
    """
    Read and write to the AWG registers

    It is useful for debug but should not be used for daily operation.

    Args:

        addr  : w   0x00 for master control registers
        offset: w   0x00 [32bits] version
                    0x04 [16bits] control select
                    0x08 [4bits]  awg control. bit0 reset, bit 1 prepare,
                                               bit2 start, bit 3 terminate
                    0x10 [16bits] busy
                    0x14 [16bits] ready
                    0x18 [16bits] done
        pos   : w   Bit location
        bits  : w   Number of bits to read/write
        data  : w   Data. Read operation is performed if None
    """
    dev = self.selectedDevice(c)
    reg = dev._awg_ctrl._AwgCtrl__reg_access                # DEBUG _awg_ctrl is a protected member
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
    if QSConstants.CNL_READ_VAL != dev.device_role:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.device_name))
    reg = dev._cap_ctrl._CaptureCtrl__reg_access            # DEBUG _cap_ctrl is a protected member
    if data is None:
      data = reg.read_bits(addr,offset,pos,bits)
      return data
    else:
      reg.write_bits(addr,offset,pos,bits,data)
    return 0

  @setting(503, 'DEBUG Auto Acquisition FIR Coefficients', muxch = ['w'], bb_frequency = ['v[Hz]'], sigma = ['v[s]'], returns = ['b'])
  def debug_auto_acquisition_fir_coefficients(self,c,muxch,bb_frequency,sigma = None):
    """
    Automatically set finite impulse resoponse filter coefficients.

    Set gauss-envelope FIR coefficients

    Args:
         muxch        : w
            Multiplex readout mux channel. 0-3 can be set

         bb_frequency : v[Hz]
             The base-band frequency of the readout signal. It could be
             f(readout) - f(local oscillator) - f(coase NCO frequency) when
             upper-sideband modu- and demodulations are used.
    Returns:
         success      : b
    """

    if sigma is None:
      sigma = 3.0                                           # nanosecodnds

    freq_in_mhz = bb_frequency['MHz']                       # base-band frequency before decimation.
    if -QSConstants.ADCBB_SAMPLE_R/2. >= freq_in_mhz or    \
        QSConstants.ADCBB_SAMPLE_R/2. <= freq_in_mhz:
        raise Exception( QSMessage.ERR_INVALID_RANG.format('bb_frequency',
                                                           -QSConstants.ADCBB_SAMPLE_R/2.,
                                                            QSConstants.ADCBB_SAMPLE_R/2.))
    n_of_band    = QSConstants.ACQ_MAX_FCOEF
    band_step    = QSConstants.ADCBB_SAMPLE_R/n_of_band
    band_idx     = ( int( freq_in_mhz/band_step+0.5+n_of_band)-n_of_band )
    band_center  = band_step * band_idx

    x            = np.arange(QSConstants.ACQ_MAX_FCOEF)     \
                           -(QSConstants.ACQ_MAX_FCOEF-1)/2 # symmetric in center.
    gaussian     = np.exp(-0.5*x**2/(sigma**2))             # gaussian with sigma of [sigma]
    phase_factor = 2*np.pi*(band_center/QSConstants.ADCBB_SAMPLE_R)*np.arange(QSConstants.ACQ_MAX_FCOEF)
    coeffs       = gaussian*np.exp(1j*phase_factor)*(1-1e-3)

    return self.acquisition_fir_coefficients(c,muxch,coeffs)

  @setting(504, 'DEBUG Auto Acquisition Window Coefficients', muxch = ['w'], bb_frequency = ['v[Hz]'], returns = ['b'])
  def debug_auto_acquisition_window_coefficients(self,c,muxch,bb_frequency):
    """
    Automatically set complex window coefficients

    debug_auto_acquisition_window_coefficients() sets rectangular window as a
    demodulation window. If you want to try windowed demodulation, it is better
    to give the coefs manually.

    *debug_auto_acquisition_window_coefficients() has to be called after
     acquisition_window().

    Args:
         muxch        : w
             Multiplex readout mux channel. 0-3 can be set

         bb_frequency : v[Hz]
             The base-band frequency of the readout signal. It could be
             f(readout) - f(local oscillator) - f(coase NCO frequency) when
             upper-sideband modu- and demodulations are used.
    Returns:
         success      : b
    """
    def _max_window_length(windows):
      def section_length(tuple_section):
        return tuple_section[1]-tuple_section[0]
      return max([section_length(_w) for _w in windows])

    dev         = self.selectedDevice(c)
    freq_in_mhz = bb_frequency['MHz']                       # Base-band frequency before decimation

    if QSConstants.CNL_READ_VAL != dev.device_role:
      raise Exception(QSMessage.ERR_INVALID_DEV.format('readout',dev.device_name))
    elif not dev.static_check_mux_channel_range(muxch):
      raise ValueError(QSMessage.ERR_INVALID_RANG.format('muxch', 0, QSConstants.ACQ_MULP - 1))
    elif -QSConstants.ADCBB_SAMPLE_R/2. >= freq_in_mhz or  \
          QSConstants.ADCBB_SAMPLE_R/2. <= freq_in_mhz:
          raise Exception( QSMessage.ERR_INVALID_RANG.format('bb_frequency',
                                                             -QSConstants.ADCBB_SAMPLE_R/2.,
                                                              QSConstants.ADCBB_SAMPLE_R/2.))

    decim_factor = int(QSConstants.ADCBB_SAMPLE_R / QSConstants.ADCDCM_SAMPLE_R + 0.5)
    nsample      = _max_window_length(dev.acquisition_window[muxch])            \
                                        // (decim_factor * QSConstants.ADC_BBSAMP_IVL)
    phase_factor = 2*np.pi*(freq_in_mhz/QSConstants.ADCDCM_SAMPLE_R)*np.arange(nsample)
    coeffs       = np.exp(-1j*phase_factor)*(1-1e-3)        # Rectangular window

    return self.acquisition_window_coefficients(c,muxch,coeffs)

  @setting(505, 'DEBUG Microwave Switch', output = ['b'], returns = ['b'])
  def debug_microwave_switch(self,c,output = None):
    """
    Enable and disable a microwave switch at the output.

    Args:
        output : b (bool)
            Outputs signal if output = True. Othewise no output or loopback.
    Returns:
        output : b (bool)
            Current status of the switch.
    """
    dev = self.selectedDevice(c)
    if output is not None:
      yield dev.set_microwave_switch(output)
    else:
      output = yield dev.get_microwave_switch()
    returnValue(output)


############################################################
#
# QUBE MANAGER
#

class Qube_Manager_Device(DeviceWrapper):

  @inlineCallbacks
  def connect(self, *args, **kw):                           # @inlineCallbacks
    name, qube_type = args
    print(QSMessage.CONNECTING_CHANNEL.format(name))
    self.name         = name
                                                            # self.qube_type    = qube_type
    self.lsi_ctrl     = kw[  'lsi_ctrl' ]
    self.sync_ctrl    = kw[ 'sync_ctrl' ]
    self.channel_info = kw[ 'channels'  ]
    self._sync_addr   = kw[ 'sync_addr' ]
    self._sync_func   = kw[ 'sync_func' ]
    self._read_func   = kw[ 'read_func' ]
    self._verbose     = False
    yield

  @inlineCallbacks
  def initialize(self):                                     # @inlineCallbacks
    yield self.lsi_ctrl.do_init(rf_type=self.qube_type, message_out=self.verbose)
    mixer_init = [ ( ch[QSConstants.CNL_MIXCH_TAG],
                     ch[QSConstants.CNL_MIXSB_TAG] ) for ch in self.channel_info]

    for ch, usb_lsb in mixer_init:                          # Upper or lower sideband configuration
      if   usb_lsb == QSConstants.CNL_MXUSB_VAL:            #    in the active IQ mixer. The output
        yield self.lsi_ctrl.adrf6780[ ch ].set_usb()        #    become small with a wrong sideband
      elif usb_lsb == QSConstants.CNL_MXLSB_VAL:            #                               setting.
        yield self.lsi_ctrl.adrf6780[ ch ].set_lsb()

  @inlineCallbacks
  def set_microwave_switch(self,value):                     # @inlineCallbacks
    g   = self.lsi_ctrl.gpio
    yield g.write_value(value & 0x3fff)

  @inlineCallbacks
  def read_microwave_switch(self):
    g    = self.lsi_ctrl.gpio
    reps = yield g.read_value()
    returnValue(reps & 0x3fff)

  @property
  def verbose(self):                                        # @property
    return self._verbose
  @verbose.setter
  def verbose(self,x):                                      # @verbose.setter
    if isinstance(x,bool):
      self._verbose = x

  @inlineCallbacks
  def synchronize_with_master(self):                        # @inlineCallbacks
    func, srv = self._sync_func
    yield func(srv,self._sync_addr)

    func, srv = self._read_func
    resp  = yield func(srv)
    print('Qube_Manager_Deice.synchronize_with_master: read value = ',resp)


class Qube_Manager_Server(DeviceServer):
  name          = QSConstants.MNRNAME
  possibleLinks = list()
  adi_api_path  = None
  deviceWrapper = Qube_Manager_Device

  @inlineCallbacks
  def initServer(self):                                     # @inlineCallbacks
    yield DeviceServer.initServer(self)

    cxn = self.client
    reg = cxn[QSConstants.REGSRV]
    try:
      yield reg.cd(QSConstants.REGDIR)
      config = yield reg.get(QSConstants.REGLNK)
      self.possibleLinks = self.extract_links(json.loads(config))
      self.master_link   = yield reg.get(QSConstants.REGMASTERLNK)
      self.adi_api_path  = yield reg.get(QSConstants.REGAPIPATH)

      self._master_ctrl  = yield QuBESequencerMaster(self.master_link)
      self.__is_clock_opened = True
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)

  def extract_links(self,link):
    return [(_name,link[_name][QSConstants.SRV_QUBETY_TAG ],
                   link[_name][QSConstants.SRV_IPLSI_TAG  ],
                   link[_name][QSConstants.SRV_IPCLK_TAG  ],
                   link[_name][QSConstants.SRV_CHANNEL_TAG]) for _name in link.keys()]

  def initContext(self, c):
    DeviceServer.initContext(self,c)

  @inlineCallbacks
  def findDevices(self):                                    # @inlineCallbacks
    cxn   = self.client
    found = list()

    for _name,_type,_iplsi,_ipclk,_channel in self.possibleLinks:
      print(QSMessage.CHECKING_QUBEUNIT.format(_name))
      try:
        res = pingger(_iplsi)
        if 0 == res:
          res = pingger(_ipclk)
        else:
          raise Exception(QSMessage.ERR_HOST_NOTFOUND.format(_name))
      except Exception as e:
        print(sys._getframe().f_code.co_name,e)
        continue

      print(QSMessage.CNCTABLE_QUBEUNIT.format(_name))
      device = yield self.instantiateQube(_name,_type,_iplsi,_ipclk,_channel)
      found.append(device)
      yield

    returnValue(found)

  @inlineCallbacks
  def instantiateQube(self, name, qube_type, iplsi, ipclk, channel_info):    # @inlineCallbacks
    lsi_ctrl  = yield qubelsi.qube.Qube(iplsi, self.adi_api_path)
    sync_ctrl = yield QuBESequencerClient(ipclk)
    args      = (name, qube_type)
    kw        = dict( lsi_ctrl  = lsi_ctrl,
                      sync_ctrl = sync_ctrl,
                      sync_addr = ipclk,
                      sync_func = (Qube_Manager_Server._synchronize_with_master_clock,self),
                      read_func = (Qube_Manager_Server._read_master_clock,self),
                      channels  = channel_info )
    returnValue( (name,args,kw) )

  @setting(100, 'Reset', returns=['b'])
  def device_reinitialize(self,c):
    """
    Reset QuBE units.

    This routine resets ICs in a QuBE unit such as local oscillators, AD/DA
    converters, analog mixers, etc.

    Returns:
        success : Always True
    """
    dev = self.selectedDevice(c)
    yield dev.initialize()
    returnValue(True)

  @setting(101, 'Microwave Switch', value = ['w'], returns = ['w'])
  def microwave_switch(self,c,value = None):
    """
    Read and write the microwave switch settting.

    The on-off setting of the microwave switch at each ports can be set using the following setting
    bits. The logic high '1' = 0b1 makes the switch off or loop back state. The logic AND of the
    settings bits become a value to the argument [value].

        0x0003 - channel 0-1
        0x0004 - channel 2
        0x0020 - channel 5
        0x0040 - channel 6
        0x0080 - channel 7
        0x0100 - channel 8
        0x0800 - channel 11
        0x3000 - channel 12-13

    Args:
        value : w (unsigned int)
            See above.
    Returns:
        value : w (unsigned int)
            Current status of the switch is retrieved.
    """
    dev = self.selectedDevice(c)
    if value is not None:
      yield dev.set_microwave_switch(value)
    else:
      value = yield dev.read_microwave_switch()
    returnValue(value)


  @setting(200, 'Debug Verbose', flag = ['b'], returns=['b'])
  def debug_verbose_message(self,c,flag = None):
    """
    Select debugging mode.

    Set flag = True to see long message output in the console.

    Args:
        flag : b (bool)
    Returns:
        flag : b (bool)
    """
    dev = self.selectedDevice(c)
    if flag is not None:
      dev.verbose = flag
    return dev.verbose

  @setting(301, 'Reconnect Master Clock', returns = ['b'])
  def reconnect_master(self,c):
    """
    Reconnect to the master FPGA board.

    reconnect_master_clock() close the UDP port to the master and reopen the
    socket to the master clock FPGA board.

    Returns:
        flag : b (bool)
            Always True
    """

    if self.__is_clock_opened:
      del self._master_ctrl

    self._master_ctrl = QuBESequencerMaster(self.master_link)
    self.__is_clock_opened = True

    return True

  @setting(302, 'Clear Master Clock', returns = ['b'])
  def clear_master_clock(self,c):
    """
    Reset synchronization clock in the master FPGA board.

    This method reset the syncronization clock in the master FPGA board.

    Returns:
        flag : b (bool)
            Always True
    """

    if not self.__is_clock_opened:
      raise Exception(QSMessage.ERR_DEV_NOT_OPEN)

    resp = False
    try:
      ret  = yield self._master_ctrl.clear_clock()
      resp = True
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)
      raise(e)

    returnValue(resp)

  @setting(303, 'Read Master Clock', returns = ['ww'])
  def read_master_clock(self,c):
    """
    Read synchronization clock in the master FPGA board.

    This method read the value of syncronization clock in the master FPGA board.

    Returns:
        clock : ww (two of 32-bit unsigned int)
            The first and the last 32-bit words corresponds to the high and low
            words of the clock value represented in 64-bit unsigned int.
    """
    resp = yield self._read_master_clock()
    h = (resp & 0xffffffff00000000) >> 32
    l =  resp & 0xffffffff
    returnValue((h,l))

  @setting(304, 'Synchronize Clock', returns = ['b'])
  def synchronize_with_master(self,c):
    """
    Synchronize QuBE unit to the clock in the master FPGA board.

    This method triggers synchronization between the master and the selected
    device clocks.

    Returns:
        flag : b (bool)
            Always True
    """
    dev = self.selectedDevice(c)
    yield dev.synchronize_with_master()
    returnValue(True)

  @inlineCallbacks
  def _synchronize_with_master_clock(self,target_addr):     # @inlineCallbacks

    if not self.__is_clock_opened:
      raise Exception(QSMessage.ERR_DEV_NOT_OPEN)

    resp = False
    try:
      ret = yield self._master_ctrl.kick_clock_synch([target_addr])
      resp = True
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)

    returnValue(resp)

  @inlineCallbacks
  def _read_master_clock(self):                             # @inlineCallbacks

    if not self.__is_clock_opened:
      raise Exception(QSMessage.ERR_DEV_NOT_OPEN)

    resp = 0
    try:
      ret  = yield self._master_ctrl.read_clock()
      resp = ret
    except Exception as e:
      print(sys._getframe().f_code.co_name,e)
      raise(e)

    returnValue(resp)

############################################################
#
# master_ctrl/software wrappers
#
class QuBESequencerMaster(QuBEMasterClient):

  PORT = 16384

  def __init__(self, ip_addr):
    super(QuBESequencerMaster, self).__init__(ip_addr, self.PORT)
    self._QuBEMasterClient__sock.settimeout(2)

  def read_clock(self, value=0):                            # inherited from QuBEMasterClient
    data = struct.pack('BB', 0x30, 0)                       #  <quiet version>
    data += struct.pack('HHH', 0, 0, 0)
    data += struct.pack('<Q', value)
    ret = self.send_recv(data)
    result = struct.unpack('<Q', ret[0][8:])
    return result[0]

class QuBESequencerClient(SequencerClient):

  PORT = 16384

  def __init__(self, ip_addr):
    super(QuBESequencerClient, self).__init__(ip_addr, self.PORT)

  def add_sequencer(self, value, awgs = range(16)):

    select_bits = 0
    for _awg in awgs:
      if 0 <= _awg and _awg < 16:
        select_bits += (1 << _awg)

    data = struct.pack('BB', 0x22, 0)
    data += struct.pack('HH', 0, 0)
    data += struct.pack('>H', 16)                           # 1-command = 16bytes
    data += struct.pack('<Q', value)                        # start time with MSB=1
    data += struct.pack('<H', select_bits)                  # target AWG
    data += struct.pack('BBBBB', 0, 0, 0, 0, 0)             # padding
    data += struct.pack('B', 0)                             # entry id

    return self.send_recv(data)



############################################################
#
# AUX SUBROUTINES FOR EASY SETUP
#

def basic_config():

  _name_tag    = QSConstants.CNL_NAME_TAG
  _type_tag    = QSConstants.CNL_TYPE_TAG
  _control_val = QSConstants.CNL_CTRL_VAL
  _readout_val = QSConstants.CNL_READ_VAL
  mixer_tag    = QSConstants.CNL_MIXCH_TAG
  usb_lsb_tag  = QSConstants.CNL_MIXSB_TAG
  usb_val      = QSConstants.CNL_MXUSB_VAL
  lsb_val      = QSConstants.CNL_MXLSB_VAL
  gpiosw_tag   = QSConstants.CNL_GPIOSW_TAG


  control_qube_500_1500 = \
  [
    {_name_tag  : 'control_0',
     _type_tag  : _control_val,
     'ch_dac'   : [15],                                     # awg id
     'cnco_dac' : (0,0),                                    # chip, main path id
     'fnco_dac' : [(0,0)],                                  # chip, link no
     'lo_dac'   : 0,                                        # local oscillator id
     mixer_tag  : 0,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0003,                                   # switch mask bit(s)
    },
    {_name_tag  : 'control_2',
     _type_tag  : _control_val,
     'ch_dac'   : [14],                                     # awg id
     'cnco_dac' : (0,1),                                    # chip, main path id
     'fnco_dac' : [(0,1)],                                  # chip, link no
     'lo_dac'   : 1,                                        # local oscillator id
     mixer_tag  : 1,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0004,                                   # switch mask bit(s)
    },
    {_name_tag  : 'control_5',
     _type_tag  : _control_val,
     'ch_dac'   : [11,12,13],                               # awg id
     'cnco_dac' : (0,2),                                    # chip, main path id
     'fnco_dac' : [(0,4),(0,3),(0,2)],                      # chip, link no
     'lo_dac'   : 2,                                        # local oscillator id
     mixer_tag  : 2,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0020,                                   # switch mask bit(s)
    },
    {_name_tag  : 'control_6',
     _type_tag  : _control_val,
     'ch_dac'   : [8,9,10],                                 # awg id
     'cnco_dac' : (0,3),                                    # chip, main path id
     'fnco_dac' : [(0,5),(0,6),(0,7)],                      # chip, link no
     'lo_dac'   : 3,                                        # local oscillator id
     mixer_tag  : 3,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0040,                                   # switch mask bit(s)
    },
    {_name_tag  : 'control_7',
     _type_tag  : _control_val,
     'ch_dac'   : [5,6,7],                                  # awg id
     'cnco_dac' : (1,0),                                    # chip, main path id
     'fnco_dac' : [(1,2),(1,1),(1,0)],                      # chip, link no
     'lo_dac'   : 4,                                        # local oscillator id
     mixer_tag  : 4,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0080,                                   # switch mask bit(s)
    },
    {_name_tag  : 'control_8',
     _type_tag  : _control_val,
     'ch_dac'   : [0,3,4],                                  # awg id
     'cnco_dac' : (1,1),                                    # chip, main path id
     'fnco_dac' : [(1,5),(1,4),(1,3)],                      # chip, link no
     'lo_dac'   : 5,                                        # local oscillator id
     mixer_tag  : 5,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0100,                                   # switch mask bit(s)
    },
    {_name_tag  : 'control_b',
     _type_tag  : _control_val,
     'ch_dac'   : [1],                                      # awg id
     'cnco_dac' : (1,2),                                    # chip, main path id
     'fnco_dac' : [(1,6)],                                  # chip, link no
     'lo_dac'   : 6,                                        # local oscillator id
     mixer_tag  : 6,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0800,                                   # switch mask bit(s)
    },
    {_name_tag  : 'control_d',
     _type_tag  : _control_val,
     'ch_dac'   : [2],                                      # awg id
     'cnco_dac' : (1,3),                                    # chip, main path id
     'fnco_dac' : [(1,7)],                                  # chip, link no
     'lo_dac'   : 7,                                        # local oscillator id
     mixer_tag  : 7,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x2000,                                   # switch mask bit(s)
    },
  ]

  readout_control_qube = \
  [
    {_name_tag  : 'readout_01',
     _type_tag  : _readout_val,
     'ch_dac'   : [15],                                     # awg id
     'ch_adc'   : 1,                                        # module id
     'cnco_dac' : (0,0),                                    # chip, main path
     'cnco_adc' : (0,3),                                    # chip, main path
     'fnco_dac' : [(0,0)],                                  # chip, link id
     'lo_dac'   : 0,                                        # local oscillator id
     mixer_tag  : 0,                                        # mixer channel
     usb_lsb_tag: usb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0003,                                   # switch mask bit(s)
    },
    {_name_tag  : 'pump_2',
     _type_tag  : _control_val,
     'ch_dac'   : [14],                                     # awg id
     'cnco_dac' : (0,1),                                    # chip, main path id
     'fnco_dac' : [(0,1)],                                  # chip, link no
     'lo_dac'   : 1,                                        # local oscillator id
     mixer_tag  : 1,                                        # mixer channel
     usb_lsb_tag: usb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0004,                                   # switch mask bit(s)
    },
    {_name_tag  : 'control_5',
     _type_tag  : _control_val,
     'ch_dac'   : [11,12,13],                               # awg id
     'cnco_dac' : (0,2),                                    # chip, main path id
     'fnco_dac' : [(0,4),(0,3),(0,2)],                      # chip, link no
     'lo_dac'   : 2,                                        # local oscillator id
     mixer_tag  : 2,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0020,                                   # switch mask bit(s)
    },
    {_name_tag  : 'control_6',
     _type_tag  : _control_val,
     'ch_dac'   : [8,9,10],                                 # awg id
     'cnco_dac' : (0,3),                                    # chip, main path id
     'fnco_dac' : [(0,5),(0,6),(0,7)],                      # chip, link no
     'lo_dac'   : 3,                                        # local oscillator id
     mixer_tag  : 3,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0040,                                   # switch mask bit(s)
    },
    {_name_tag  : 'control_7',
     _type_tag  : _control_val,
     'ch_dac'   : [5,6,7],                                  # awg id
     'cnco_dac' : (1,0),                                    # chip, main path id
     'fnco_dac' : [(1,2),(1,1),(1,0)],                      # chip, link no
     'lo_dac'   : 4,                                        # local oscillator id
     mixer_tag  : 4,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0080,                                   # switch mask bit(s)
    },
    {_name_tag  : 'control_8',
     _type_tag  : _control_val,
     'ch_dac'   : [0,3,4],                                  # awg id
     'cnco_dac' : (1,1),                                    # chip, main path id
     'fnco_dac' : [(1,5),(1,4),(1,3)],                      # chip, link no
     'lo_dac'   : 5,                                        # local oscillator id
     mixer_tag  : 5,                                        # mixer channel
     usb_lsb_tag: lsb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0100,                                   # switch mask bit(s)
    },
    {_name_tag  : 'pump_b',
     _type_tag  : _control_val,
     'ch_dac'   : [1],                                      # awg id
     'cnco_dac' : (1,2),                                    # chip, main path id
     'fnco_dac' : [(1,6)],                                  # chip, link no
     'lo_dac'   : 6,                                        # local oscillator id
     mixer_tag  : 6,                                        # mixer channel
     usb_lsb_tag: usb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x0800,                                   # switch mask bit(s)
    },
    {_name_tag  : 'readout_cd',
     _type_tag  : _readout_val,
     'ch_dac'   : [2],                                      # awg id
     'ch_adc'   : 0,                                        # module id
     'cnco_dac' : (1,3),                                    # chip, main path
     'cnco_adc' : (1,3),                                    # chip, main path
     'fnco_dac' : [(1,7)],                                  # chip, link no
     'lo_dac'   : 7,                                        # local oscillator id
     mixer_tag  : 7,                                        # mixer channel
     usb_lsb_tag: usb_val,                                  # mixder sideband (initial value)
     gpiosw_tag : 0x3000,                                   # switch mask bit(s)
    },
  ]

  servers = \
   {
     'qube004': { QSConstants.SRV_IPFPGA_TAG: '10.1.0.22', QSConstants.SRV_IPLSI_TAG: '10.5.0.22', QSConstants.SRV_IPCLK_TAG: '10.2.0.22', QSConstants.SRV_QUBETY_TAG: 'A', QSConstants.SRV_CHANNEL_TAG : readout_control_qube  },
     'qube005': { QSConstants.SRV_IPFPGA_TAG: '10.1.0.23', QSConstants.SRV_IPLSI_TAG: '10.5.0.23', QSConstants.SRV_IPCLK_TAG: '10.2.0.23', QSConstants.SRV_QUBETY_TAG: 'A', QSConstants.SRV_CHANNEL_TAG : readout_control_qube  },
     'qube006': { QSConstants.SRV_IPFPGA_TAG: '10.1.0.24', QSConstants.SRV_IPLSI_TAG: '10.5.0.24', QSConstants.SRV_IPCLK_TAG: '10.2.0.24', QSConstants.SRV_QUBETY_TAG: 'B', QSConstants.SRV_CHANNEL_TAG : control_qube_500_1500 },
     'qube010': { QSConstants.SRV_IPFPGA_TAG: '10.1.0.28', QSConstants.SRV_IPLSI_TAG: '10.5.0.28', QSConstants.SRV_IPCLK_TAG: '10.2.0.28', QSConstants.SRV_QUBETY_TAG: 'A', QSConstants.SRV_CHANNEL_TAG : readout_control_qube  },
     'qube011': { QSConstants.SRV_IPFPGA_TAG: '10.1.0.29', QSConstants.SRV_IPLSI_TAG: '10.5.0.29', QSConstants.SRV_IPCLK_TAG: '10.2.0.29', QSConstants.SRV_QUBETY_TAG: 'A', QSConstants.SRV_CHANNEL_TAG : readout_control_qube  },
     'qube012': { QSConstants.SRV_IPFPGA_TAG: '10.1.0.30', QSConstants.SRV_IPLSI_TAG: '10.5.0.30', QSConstants.SRV_IPCLK_TAG: '10.2.0.30', QSConstants.SRV_QUBETY_TAG: 'B', QSConstants.SRV_CHANNEL_TAG : control_qube_500_1500 },
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
# USAGE and for my debugging
#
#  > import QubeServer
#  >  QubeServer.usage()
#

def usage():
  import labrad
  import base64
  import pickle

  from plotly                 import graph_objects as go
  from labrad                 import types as T
  from labrad.units           import ns,us

  cxn= labrad.connect()
  qs = cxn.qube_server

  devices = ['qube004-readout_01']
                                                            # Common settings
  Twaveform = 80*0.128                 # = 10.24            # micro-seconds
  nsample   = int(Twaveform * QSConstants.DACBB_SAMPLE_R + 0.5)
                                                            # points
  freq      =-189/1.024                                     # MHz, baseband signal frequency
  [(qs.select_device(i),qs.shots          (43200))           for i in devices]
  [(qs.select_device(i),qs.daq_timeout    (T.Value(10,'s'))) for i in devices]
  [(qs.select_device(i),qs.daq_length     (Twaveform*us))    for i in devices]
  [(qs.select_device(i),qs.repetition_time(97656*10.24*us))    for i in devices]
  data=np.exp(1j*2*np.pi*(freq/QSConstants.DACBB_SAMPLE_R)*np.arange(nsample))*(1-1e-3)
                                                            # This set spositive frequency shift
                                                            # of {freq} MHz for upper sideband modu-
                                                            # lation. For control, we use lower-
                                                            # side band modulation and it sets
                                                            # {-freq} MHz baseband frequency

                                                            # for pulse operation, try:
                                                            #   data[0:2560]=0.0+1j*0.0

  qs.select_device(devices[0])                              # [Trigger settings] must have
  qs.trigger_board(0)                                       # done before update_parameraters()
                                                            # and update_readout_parameters().

  qs.select_device(devices[0])                              # Readout daq=dac/adc
  """
    Readout setting
      LO         = 8.5 GHz
      NCO        = 1.5 GHz (12000/8 MHz = 15360/10.24 MHz)
      fine NCO0  = 0.0 MHz
      NCO (RX)   = 1.5 GHz
      f(readout) = 8.5 GHz + 1.5 GHz + 0.0 MHz = 10.0 GHz
  """
  dac_chan = 0
  qs.upload_waveform      ([data],dac_chan)
  qs.upload_parameters    ([dac_chan])
  qs.frequency_local      (          T.Value(8500,'MHz'))   # 8.5+1.5=10.2GHz
  qs.frequency_tx_nco     (   T.Value(1599.609375,'MHz'))   # ~ 1.6GHz.
  qs.frequency_rx_nco     (          T.Value(1500,'MHz'))   # better to be the same as tx_nco
                                                            #   but this time is not. = 1.5 GHz = 1536/1.024
  qs.frequency_tx_fine_nco( dac_chan,T.Value(   0,'MHz'))   # better not to use it.
  """
    MUX Window setting
      Enabled channel = 0 & 1
  """
  mux_channels   = list()
  mux_chan       = 0
  readout_window = [(1024*ns,(1024+1024)*ns),               # two sections of 1 us
                    (2224*ns,(2224+1024)*ns)]
  qs.acquisition_window(mux_chan, readout_window)
  qs.debug_auto_acquisition_fir_coefficients   (mux_chan,T.Value(freq,'MHz'))
  qs.debug_auto_acquisition_window_coefficients(mux_chan,T.Value(freq,'MHz'))
  qs.acquisition_mode(mux_chan, 'A' )
  # mux_channels.append(mux_chan)                           # DEBUG, disable mux 0 channel

  mux_chan       = 1
  dT=2.048*us
  readout_window = []
  s =  512*ns    ; readout_window.append(( s, s + dT))
  for i in range(3):
    s += dT + 8*ns ; readout_window.append(( s, s + dT))
    if (s+dT)['us'] > 10.24-0*(2.048+2*0.008):
      raise Exception(None)
  #for i in range(32):
  #  readout_window.append(( (6*i+1)*dT, (6*i+3)*dT))
  #  readout_window.append(( (6*i+4)*dT, (6*i+6)*dT))
  qs.acquisition_window(mux_chan, readout_window)
  qs.acquisition_mode(mux_chan, 'A' )
  qs.debug_auto_acquisition_fir_coefficients   (mux_chan,T.Value(freq+102/1.024,'MHz')) # offset, DEBUG
  qs.debug_auto_acquisition_window_coefficients(mux_chan,T.Value(freq+102/1.024,'MHz')) # offset, DEBUG
  mux_channels.append(mux_chan)
  qs.upload_readout_parameters(mux_channels)

  add_control = False
  if add_control:
    qs.select_device(devices[1])                            # control settings
    """
      Control frequency setting
        LO  = 11 GHz
        NCO =  3 GHz
        fine NCO0 = -24.9 MHz =-255/10.24 (Lower side-band modulation)
        fine NCO1 =   4.9 MHz =  50/10.24 (Lower side-band modulation)
        fine NCO2 =  14.5 MHz = 150/10.24 (Lower side-band modulation)
        f1 = 11 GHz - 3 GHz -   24.9 MHz = 7975.1 with amp = 0.25
        f2 = 11 GHz - 3 GHz -    4.9 MHz = 7995.1 with amp = 0.12
        f3 = 11 GHz - 3 GHz - (-14.6 MHz)= 8014.6 with amp = 0.10
    """
    qs.upload_parameters([0,1,2])
    qs.upload_waveform  ([0.25*data,0.12*data,0.10*data],[0,1,2])
    qs.frequency_local      (          T.Value(  11,'GHz'))
    qs.frequency_tx_nco     (          T.Value(3000,'MHz')) # 3.0GHz
    dac_chan = 0; qs.frequency_tx_fine_nco( dac_chan,T.Value( 24.90234375,'MHz'))
    dac_chan = 1; qs.frequency_tx_fine_nco( dac_chan,T.Value(  4.8828125, 'MHz'))
    dac_chan = 2; qs.frequency_tx_fine_nco( dac_chan,T.Value(-14.6484375, 'MHz'))

  [(qs.select_device(i),qs.daq_start()) for i in devices]   # daq_start
  qs.daq_trigger()                                          # daq_trigger
  qs.daq_stop()                                             # daq_stop waits for done

  #qs.debug_awg_reg(0,0x04,0,16)
  #qs.debug_awg_reg(0,0x10,0,16)
  #qs.debug_awg_reg(0,0x14,0,16)
  #qs.debug_awg_reg(0,0x18,0,16)

  qs.select_device(devices[0])
  dat=qs.download_waveform([mux_chan])
  cxn.disconnect()

  data_view = False
  if data_view:
    mx,length = dat.shape
    tdat=dat[0].reshape((length//10,10))
    dat=np.sum(tdat,axis=1)/10.
    e=np.exp(-1j*2*np.pi*(3.41796875/62.5)*np.arange(length))
                                                            # You can apply fft if need
                                                            #  dat[0]=np.fft.fft(dat[0])
    graph_data = []
    graph_data.append(
      go.Scatter ( x   = np.arange(length),
                   y   = np.real(dat),
                   name= "real")
    )
    graph_data.append(
      go.Scatter ( x   = np.arange(length),
                   y   = np.imag(dat),
                   name= "imag")
    )
                                                            #graph_data.append(
                                                            #  go.Scatter ( x   = np.arange(length),
                                                            #               y   = np.real(dat[1]),
                                                            #               name= "real")
                                                            #)
                                                            #graph_data.append(
                                                            #  go.Scatter ( x   = np.arange(length),
                                                            #               y   = np.imag(dat[1]),
                                                            #               name= "imag")
                                                            #)
    layout = go.Layout( title = 'Spur in RX',
                        xaxis=dict(title='Frequency (GHz)',dtick=0.05),
                        yaxis=dict(title='Dataset',dtick=20)
                       )

    fig = go.Figure( graph_data )
    fig.write_html("1.html")

def test_control_ch(device_name):
  from labrad.units           import ns,us
  import labrad

  cxn= labrad.connect()
  qs = cxn.qube_server

  nsample = 4*5120
  data=np.exp(1j*2*np.pi*(0/QSConstants.DACBB_SAMPLE_R)*np.arange(nsample))*(1-1e-3)

  qs.select_device(device_name)                             # 'qube004-control_6'

  qs.shots(1*25*1000*10)                                    # 10 seconds
  qs.daq_timeout(T.Value(30,'s'))
  qs.daq_length( T.Value(2*nsample,'ns'))
  qs.repetition_time(T.Value(4*10.24,'us'))

  qs.upload_parameters([0,1,2])
  qs.upload_waveform  ([0.5*data,0.3*data,0.1*data],[0,1,2])
  qs.frequency_local      (          T.Value(  11,'GHz'))
  qs.frequency_tx_nco     (          T.Value(3000,'MHz'))   # 3.0GHz ~ 8 GHz
  qs.frequency_tx_fine_nco( 0, T.Value( 29.296875,'MHz'))
  qs.frequency_tx_fine_nco( 1, T.Value( 5.37109375, 'MHz'))
  qs.frequency_tx_fine_nco( 2, T.Value(-14.6484375, 'MHz'))

  qs.daq_start()
  qs.daq_trigger()
  qs.daq_stop()

def test_control_ch_bandwidth(device_name):
  from labrad.units           import ns,us
  import labrad

  cxn= labrad.connect()
  qs = cxn.qube_server

  nsample = 4*5120
  data=np.exp(1j*2*np.pi*(0/QSConstants.DACBB_SAMPLE_R)*np.arange(nsample))*(1-1e-3)

  qs.select_device(device_name)                             # 'qube004-control_6'

  qs.shots          ( 1*1000*25 )                           # 1 seconds
  qs.daq_timeout    ( T.Value(30,'s'))
  qs.daq_length     ( T.Value(2*nsample,'ns'))
  qs.repetition_time( T.Value(4*10.24,'us'))

  qs.upload_parameters    ([0,1,2])
  qs.upload_waveform      ([0.0*data, 1.0*data, 0.0*data], [0,1,2])
  qs.frequency_local      (    T.Value(  11,'GHz'))
  qs.frequency_tx_nco     (    T.Value(3000,'MHz'))         # 3.0GHz ~ 8 GHz
  qs.frequency_tx_fine_nco( 0, T.Value( 29.296875,'MHz'))
  qs.frequency_tx_fine_nco( 1, T.Value( 5.37109375, 'MHz'))
  qs.frequency_tx_fine_nco( 2, T.Value(-14.6484375, 'MHz'))

  if False:                                                 # IF NCO sweep
    for i in range(256):
      fnco = (i*8+1024)*(12000/2**13)
      qs.frequency_tx_nco (    T.Value(fnco,'MHz'))
      qs.daq_start()
      qs.daq_trigger()
      qs.daq_stop()
  else:                                                     # BB AWG waveform sweep
    for i in range(256):
      bbfreq       = (i*20-2560)/10.24
      phase_factor = 2*np.pi*(bbfreq/QSConstants.DACBB_SAMPLE_R)
      data         =   np.exp(1j*phase_factor*np.arange(nsample))*(1-1e-3)
      qs.upload_parameters([1])
      qs.upload_waveform  ([data],[1])
      qs.daq_start()
      qs.daq_trigger()
      qs.daq_stop()

def test_readout_ch_bandwidth_and_spurious(device_name):

  from labrad.units           import ns,us
  import labrad
  import time
  import pickle

  def spectrum_analyzer_get():
    import socket
    import numpy as np
    import pickle
    import struct
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sock.settimeout(1)
    sock.connect(('localhost',19001))
    sock.send(b':TRAC:DATA? 1\n')
    rdat=b''
    while True:
      try:
        r=sock.recv(1024)
        rdat=rdat+r
      except Exception as e:
        break
    sock.close()
    y=np.array(struct.unpack('<501d',rdat[6:]))
    return y

  cxn= labrad.connect()
  qs = cxn.qube_server

  nsample      = 4*5120                                     # 4 x 10.24us = 40.96us
  phase_factor = 2*np.pi*(-189/1.024/QSConstants.DACBB_SAMPLE_R)     # 2pi x normalized frequency
  data         = np.exp(1j*phase_factor*np.arange(nsample))*(1-1e-3)

  qs.select_device(device_name)                             # 'qube004-control_6'

  qs.shots                ( 6*1000*25 )                     # 6 seconds
  qs.daq_timeout          ( T.Value(30,'s'))
  qs.daq_length           ( T.Value(2*nsample,'ns'))
  qs.repetition_time      ( T.Value(4*10.24,'us'))          # identical to the daq_length = CW operation

  qs.upload_parameters    ([0])
  qs.upload_waveform      ([data],[0])
  qs.frequency_local      (    T.Value(8500,'MHz'))
  qs.frequency_tx_nco     (    T.Value(1599.609375,'MHz'))         # 1.5GHz
  qs.frequency_tx_fine_nco( 0, T.Value(   0,'MHz'))

  def experiment_nco_sweep( vault, fnco, file_idx ):
    qs.frequency_tx_nco   (    T.Value(fnco,'MHz'))
    qs.daq_start()
    qs.daq_trigger()
    time.sleep(3.5)
    dat = spectrum_analyzer_get()
    print(file_idx,fnco)
    vault.append(dat)
    with open('data{0:03d}.pkl'.format(file_idx),'wb') as f:
      pickle.dump(np.array(vault), f)
    qs.daq_stop()

  if False:
    vault = []
    for i in range(512):
      fnco = (i*2+512)*(12000/2**13)
      experiment_nco_sweep( vault, fnco, 0 )

  elif False:
    for freq_lo, j in zip(range(9500,8000,-100),range(15)):
      qs.frequency_local( T.Value( freq_lo,'MHz'))
      vault = []
      for i in range(256):
        fnco = (i*4+512)*(12000/2**13)
        experiment_nco_sweep( vault, fnco, j )

  elif False:
    for i in range(256):
      bbfreq       = (i*20-2560)/10.24
      phase_factor = 2*np.pi*(bbfreq/QSConstants.DACBB_SAMPLE_R)
      data         = np.exp(1j*phase_factor*np.arange(nsample))*(1-1e-3)
      qs.upload_waveform  ([data],[0])
      qs.upload_parameters([0])
      qs.daq_start()
      qs.daq_trigger()
      qs.daq_stop()
  else:
    qs.daq_start()
    qs.daq_trigger()
    time.sleep(3.5)
    dat = spectrum_analyzer_get()
    with open('data.pkl','wb') as f:
      pickle.dump(np.array(dat), f)
    qs.daq_stop()


############################################################
#
# SERVER WORKER
#
# In bash, to start QuBE Server w/o debuggin mode
#   $ QUBE_SERVER = 'QuBE Server' python3 QubeServer.py
#
# To start Qube Manager,
#   $ QUBE_SERVER = 'QuBE Manager' python3 QubeServer.py
#
# Otherwise, QuBE Server starts in debugging mode.
#

try:
  server_select = os.environ[ QSConstants.ENV_SRVSEL ]
  if server_select == QSConstants.MNRNAME:
    __server__ = Qube_Manager_Server()
  elif server_select == QSConstants.SRVNAME:
    __server__ = Qube_Server()
  else:
    server_select = None
except KeyError as e:
  server_select    = None

if server_select is None:
    __server__ = QuBE_Server_debug_otasuke()

if __name__ == '__main__':
                                                            ## Import Psyco if available
                                                            #  try:
                                                            #    import psyco
                                                            #    psyco.full()
                                                            #  except ImportError:
                                                            #    pass
                                                            #  print sys.argv
                                                            #  if sys.argv:
                                                            #    del sys.argv[1:]
  util.runServer(__server__)


