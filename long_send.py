
import math
import e7awgsw

#def init_modules(awg_ctrl, cap_ctrl):
#    awg_ctrl.initialize()
#    #awg_ctrl.enable_awgs(*e7awgsw.AWG.all())
#    #cap_ctrl.initialize()

def gen_wave_seq(freq, amp=32767):
    wave_seq = e7awgsw.WaveSequence(
        num_wait_words = 16,
        num_repeats = 0xFFFFFFFF)
    
    num_chunks = 1
    for _ in range(num_chunks):
        # int(num_cycles * AwgCtrl.SAMPLING_RATE / freq) を 64 の倍数にすると, 切れ目のない波形が出力される.
        i_wave = e7awgsw.SinWave(num_cycles = 64, frequency = freq, amplitude = amp, phase = math.pi / 2)
        q_wave = e7awgsw.SinWave(num_cycles = 64, frequency = freq, amplitude = amp)
        iq_samples = e7awgsw.IqWave(i_wave, q_wave).gen_samples(
            sampling_rate = e7awgsw.AwgCtrl.SAMPLING_RATE, 
            padding_size = e7awgsw.WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK)

        wave_seq.add_chunk(
            iq_samples = iq_samples,
            num_blank_words = 0, 
            num_repeats = 0xFFFFFFFF)
    return wave_seq

    
def set_wave_sequence(awg_ctrl, amps, freqs, awgs):
    awg_to_wave_sequence = {}

    # freqs = [
    #     2.5e6, # P8 readout 0
    #     2.5e6, # P9
    #     2.5e6, 2.5e6, 2.5e6, # P12, P8, P8  ctrl 0
    #     2.5e6, 2.5e6, 2.5e6, # P7, P7, P7
    #     2.51256281e6, 2.51256281e6, 2.51256281e6, # P6
    #     # 1.953125e6, 1.953125e6, 1.953125e6, # P5
    #     # 2.47524752e6, 2.5e6, 2.51256281e6, # P5
    #     # 2.5e6, 2.5e6, 2.51256281e6, # P5 Photo1
    #     #2.5e6, 2.5e6, 2.7173913e6, # P5 Photo1
    #     #2.5e6, 2.5e6, 3.52112676e6, # P5 Photo1
    #     2.5e6, 2.5e6, 2.5e6, # P5 ctrl 1
    #     2.5e6, # P4
    #     2.5e6, # P0 readout 1
    #     ]
    # amps = [
    #         5000, # P8
    #         5000, # P
    #         5000, # P13
    #         5000, # P8
    #         5000, # P8
    #         5000, # P7
    #         5000, # P7
    #         5000, # P7
    #         5461, # P6
    #         5461, # P6
    #         5461, # P6
    #         5461, # P5
    #         5461, # P5
    #         5461, # P5
    #         5000, # P
    #         5000, # P0
    #         ]

    for awg_id in awgs:
        print("{}: freq={}, amp={}".format(awg_id, freqs[awg_id], amps[awg_id]))
        wave_seq = gen_wave_seq(freqs[awg_id], amps[awg_id]) # 5 MHz  5MHz x 8 周期では切れ目のない波形はできない
        awg_to_wave_sequence[awg_id] = wave_seq
        awg_ctrl.set_wave_sequence(awg_id, wave_seq)
    return awg_to_wave_sequence

def start(amps, freqs, ipaddr='10.0.0.16'):
    #awg_ctrl = e7awgsw.AwgCtrl(ipaddr)
    #cap_ctrl = e7awgsw.CaptureCtrl(ipaddr)
    awgs = e7awgsw.AWG.all()
    awg_ctrl = e7awgsw.AwgCtrl(ipaddr)
    # 初期化
    awg_ctrl.initialize(*awgs)
    #init_modules(awg_ctrl, cap_ctrl)
    # 波形シーケンスの設定
    awg_to_wave_sequence = set_wave_sequence(awg_ctrl, amps, freqs, awgs)
    # 波形送信スタート
    awg_ctrl.start_awgs(*awgs)

def stop(ipaddr='10.0.0.16'):
    awgs = e7awgsw.AWG.all()
    awg_ctrl = e7awgsw.AwgCtrl(ipaddr)
    awg_ctrl.terminate_awgs(*awgs)
