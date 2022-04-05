"""
AWG から 50MHz の余弦波を出力して, 信号処理モジュールを全て無効にしてキャプチャします.
"""
import sys
import pathlib
import math
import argparse

lib_path = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(lib_path)
from e7awgsw import *
from e7awgsw.labrad import *

IP_ADDR = '10.0.0.16'

def gen_wave_seq(freq, amp=32760):
    wave_seq = WaveSequence(
        num_wait_words = 16,
        num_repeats = 0xFFFFFFFF)
    
    num_chunks = 1
    for _ in range(num_chunks):
        # int(num_cycles * AwgCtrl.SAMPLING_RATE / freq) を 64 の倍数にすると, 切れ目のない波形が出力される.
        i_wave = SinWave(num_cycles = 8, frequency = freq, amplitude = amp, phase = math.pi / 2)
        q_wave = SinWave(num_cycles = 8, frequency = freq, amplitude = amp)
        iq_samples = IqWave(i_wave, q_wave).gen_samples(
            sampling_rate = AwgCtrl.SAMPLING_RATE, 
            padding_size = WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK)

        wave_seq.add_chunk(
            iq_samples = iq_samples,
            num_blank_words = 0, 
            num_repeats = 0xFFFFFFFF)
    return wave_seq


def set_wave_sequence(awg_ctrl, awgs):
    awg_to_wave_sequence = {}

    freqs = [
        2.5e6, # P8 readout 0
        2.5e6, # P9
        2.5e6, 2.5e6, 2.5e6, # P12, P8, P8  ctrl 0
        2.5e6, 2.5e6, 2.5e6, # P7, P7, P7
        2.51256281e6, 2.51256281e6, 2.51256281e6, # P6
        # 1.953125e6, 1.953125e6, 1.953125e6, # P5
        # 2.47524752e6, 2.5e6, 2.51256281e6, # P5
        # 2.5e6, 2.5e6, 2.51256281e6, # P5 Photo1
        #2.5e6, 2.5e6, 2.7173913e6, # P5 Photo1
        #2.5e6, 2.5e6, 3.52112676e6, # P5 Photo1
        2.5e6, 2.5e6, 2.5e6, # P5 ctrl 1
        2.5e6, # P4
        2.5e6, # P0 readout 1
        ]
    amps = [
            5000, # P8
            5000, # P
            5000, # P13
            5000, # P8
            5000, # P8
            5000, # P7
            5000, # P7
            5000, # P7
            5461, # P6
            5461, # P6
            5461, # P6
            5461, # P5
            5461, # P5
            5461, # P5
            5000, # P
            5000, # P0
            ]

    for awg_id in awgs:
        print("{}: freq={}, amp={}".format(awg_id, freqs[awg_id], amps[awg_id]))
        wave_seq = gen_wave_seq(freqs[awg_id], amps[awg_id]) # 5 MHz  5MHz x 8 周期では切れ目のない波形はできない
        awg_to_wave_sequence[awg_id] = wave_seq
        awg_ctrl.set_wave_sequence(awg_id, wave_seq)
    return awg_to_wave_sequence


def create_awg_ctrl(use_labrad, server_ip_addr):
    if use_labrad:
        return RemoteAwgCtrl(server_ip_addr, IP_ADDR)
    else:
        return AwgCtrl(IP_ADDR)


def main(use_labrad, server_ip_addr, awgs):
    with create_awg_ctrl(use_labrad, server_ip_addr) as awg_ctrl:
        # 初期化
        awg_ctrl.initialize(*awgs)
        # 波形シーケンスの設定
        awg_to_wave_sequence = set_wave_sequence(awg_ctrl, awgs)
        # 波形送信スタート
        awg_ctrl.start_awgs(*awgs)
        print('end')
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipaddr')
    parser.add_argument('--awgs')
    parser.add_argument('--server-ipaddr')
    parser.add_argument('--labrad', action='store_true')
    args = parser.parse_args()

    if args.ipaddr is not None:
        IP_ADDR = args.ipaddr

    server_ip_addr = 'localhost'
    if args.server_ipaddr is not None:
        server_ip_addr = args.server_ipaddr

    awgs = AWG.all()
    if args.awgs is not None:
        awgs = [AWG.of(int(x)) for x in args.awgs.split(',')]

    main(args.labrad, server_ip_addr, awgs)
