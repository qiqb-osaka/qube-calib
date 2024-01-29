import time
from concurrent.futures import ThreadPoolExecutor

import e7awgsw
import numpy as np
from e7awgsw import AWG, CaptureModule, IqWave, WaveSequence


class AwgCtrl(e7awgsw.AwgCtrl):
    pass


class CaptureCtrl(e7awgsw.CaptureCtrl):
    def check_err(self, *capu):
        e = super().check_err(*capu)
        if any(e):
            raise IOError("CaptureCtrl error.")

    def get_capture_data(self, *units):
        return CaptureData(super(), *units)


class CaptureData(object):
    def __init__(self, cap_ctrl, *units):
        self.data = v = {}
        for u in units:
            n = cap_ctrl.num_captured_samples(u)
            c = np.array(cap_ctrl.get_capture_data(u, n))
            v[u] = c[:, 0] + 1j * c[:, 1]

    def __getitem__(self, unit):
        return self.data[unit]


class Recv(object):
    def __init__(self, ipaddr: str, module, param=None):
        self._trigger = None
        self.ipaddr = ipaddr
        if isinstance(module, list) or isinstance(module, tuple):
            self.captms = [o if isinstance(o, CaptureModule) else o.id for o in module]
        else:
            m = module
            self.captms = []
            self.captm = m if isinstance(m, CaptureModule) else m.id
        self.param = param
        self.data = None

    def start(self, param=None, timeout=5):
        if param is None:
            param = self.param
        if self.captms:
            units = CaptureModule.get_units(*self.captms)
        else:
            units = CaptureModule.get_units(self.captm)
        with CaptureCtrl(self.ipaddr) as cap_ctrl:
            cap_ctrl.initialize(*units)
            if isinstance(param, list) or isinstance(param, tuple):
                for u, p in zip(units, param):
                    cap_ctrl.set_capture_params(u, p)
            else:
                for i in units:
                    cap_ctrl.set_capture_params(i, param)
            cap_ctrl.start_capture_units(*units)
            cap_ctrl.wait_for_capture_units_to_stop(timeout, *units)
            cap_ctrl.check_err(*units)
            self.data = cap_ctrl.get_capture_data(*units)

    def wait(self, param=None, timeout=5):
        if param is None:
            param = self.param
        if self.captms:
            units = CaptureModule.get_units(*self.captms)
        else:
            units = CaptureModule.get_units(self.captm)
        with CaptureCtrl(self.ipaddr) as cap_ctrl:
            cap_ctrl.initialize(*units)
            if self._trigger is not None:
                for i in self.captms:
                    cap_ctrl.select_trigger_awg(i, self._trigger)
                    cap_ctrl.enable_start_trigger(*CaptureModule.get_units(i))
            if isinstance(param, list) or isinstance(param, tuple):
                for u, p in zip(units, param):
                    cap_ctrl.set_capture_params(u, p)
            else:
                for i in units:
                    cap_ctrl.set_capture_params(i, param)
            cap_ctrl.wait_for_capture_units_to_stop(timeout, *units)
            cap_ctrl.check_err(*units)
            self.data = cap_ctrl.get_capture_data(*units)

    @property
    def trigger(self):
        return self._trigger

    @trigger.setter
    def trigger(self, awg):
        self._trigger = awg if isinstance(awg, AWG) else awg.id


class Send(object):
    def __init__(self, ipaddr, awgs, wave_seqs):
        self.ipaddr = ipaddr
        self.awgs = [o if isinstance(o, AWG) else o.id for o in awgs]
        self.wave_seqs = wave_seqs

    def start(self):
        awgs, wavs = self.awgs, self.wave_seqs
        with AwgCtrl(self.ipaddr) as awg_ctrl:
            awg_ctrl.initialize(*awgs)
            for a, w in zip(awgs, wavs):
                awg_ctrl.set_wave_sequence(a, w)
            awg_ctrl.terminate_awgs(*awgs)
            awg_ctrl.clear_awg_stop_flags(*awgs)
            awg_ctrl.start_awgs(*awgs)

    def terminate(self):
        with AwgCtrl(self.ipaddr) as awg_ctrl:
            awg_ctrl.terminate_awgs(*self.awgs)

    def wait(self, timeout=5):
        awgs, wavs = self.awgs, self.wave_seqs
        with AwgCtrl(self.ipaddr) as awg_ctrl:
            awg_ctrl.initialize(*awgs)
            for a, w in zip(awgs, wavs):
                awg_ctrl.set_wave_sequence(a, w)
            awg_ctrl.terminate_awgs(*awgs)
            awg_ctrl.clear_awg_stop_flags(*awgs)
            print("wait for started by sequencer.")
            awg_ctrl.wait_for_awgs_to_stop(timeout, *self.awgs)


class WaveChunkFactory(object):
    def get_timestamp(self):
        samples = int(self._duration * AwgCtrl.SAMPLING_RATE)
        return np.linspace(0, self._duration, samples)

    def __init__(self, duration=128e-9, amp=32767, blank=0, repeats=1, init=0):
        # duration の初期値は CW 出力を想定して設定した
        # int(duration * AwgCtrl.SAMPLING_RATE) が 64 の倍数だと切れ目のない波形が出力される．
        # 波形チャンクの最小サイズが 128ns (500Msps の繰り返し周期は 2ns)

        self._duration = duration
        self.amp = amp
        self.blank = blank  # [s]
        self.repeats = repeats  # times
        self.init = init  # iq value
        self.iq = np.zeros(*self.get_timestamp().shape).astype(complex)
        self.iq[:] = init

    @property
    def timestamp(self):
        return self.get_timestamp()

    @property
    def chunk(self):
        iq = self.amp * self.iq
        i, q = np.real(iq).astype(int), np.imag(iq).astype(int)
        s = IqWave.convert_to_iq_format(i, q, WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK)

        r = AwgCtrl.SAMPLING_RATE
        b = max(int(r * self.blank), 0)  # 非負の値を取るように
        n = WaveSequence.NUM_SAMPLES_IN_AWG_WORD

        return {"iq_samples": s, "num_blank_words": b // n, "num_repeats": self.repeats}

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, v):
        self._duration = v
        self.iq = np.zeros(*self.get_timestamp().shape).astype(complex)
        self.iq[:] = self.init


class WaveSequenceCW(e7awgsw.WaveSequence):
    def __init__(self, amp=32767):
        super().__init__(num_wait_words=0, num_repeats=0xFFFFFFFF)
        wave_chunk = WaveChunkFactory(amp=amp, repeats=0xFFFFFFFF)
        wave_chunk.iq[:] = 1
        self.add_chunk(**wave_chunk.chunk)


class WaveSequenceFactory(object):
    def __init__(self, num_wait_words=0, num_repeats=1):
        self.num_wait_words = num_wait_words
        self.num_repeats = num_repeats
        self.chunk = []

    def new_chunk(self, duration=128e-9, amp=32767, blank=0, repeats=1, init=0):
        self.chunk.append(
            WaveChunkFactory(
                duration=duration, amp=amp, blank=blank, repeats=repeats, init=init
            )
        )

    @property
    def sequence(self):
        w = WaveSequence(self.num_wait_words, self.num_repeats)
        for c in self.chunk:
            w.add_chunk(**c.chunk)
        return w


def send_recv_single(
    ipfpga, awg_to_wave_sequence, capt_to_capt_param, trigger_awg, sleep=0.5, timeout=20
):
    """
    単体 Qube にて Send と Recv を同期動作させる．
    """

    a, c = awg_to_wave_sequence, capt_to_capt_param
    send = Send(ipfpga, [k for k, v in a.items()], [v.sequence for k, v in a.items()])
    recv = Recv(ipfpga, [k for k, v in c.items()], [v for k, v in c.items()])
    # send = Send(ipfpga, [o.port.awg for o, w in send], [w.sequence for o, w in send])
    # recv = Recv(ipfpga, [v.port.capt for v, p in w.items()], [p for v, p in w.items()])
    # print([v.sum_section_list for k, v in c.items()])
    recv.trigger = trigger_awg

    with ThreadPoolExecutor() as e:
        thread = e.submit(lambda: recv.wait(timeout=timeout))
        time.sleep(sleep)
        send.terminate()
        send.start()
        thread.result()
    send.terminate()

    return send, recv


def _send_recv(qube_to_e7awgsw, trigger_awg):
    result = {}
    d = qube_to_e7awgsw
    if len(d) == 1:
        qube = list(d.keys())[0]
        e = d[qube]
        awg_to_wave_sequence = e["awg_to_wavesequence"]
        capt_to_capt_param = e["capt_to_captparam"]
        s, r = send_recv_single(
            qube.ipfpga,
            awg_to_wave_sequence,
            capt_to_capt_param,
            trigger_awg[qube],
        )
        result[qube] = {}
        result[qube]["send"] = s
        result[qube]["recv"] = r
    else:
        raise ValueError(
            "Multi Qube Sync is not implemented yet in meas.send_recv function."
        )
    return result


def send_recv(ipfpga_to_e7awgsw):
    """
    Send と Recv を同期動作させる．現在は Qube 単体での動作にしか対応していないが，同じ Interface で
    複数台同期に対応させる予定
    """

    result = {}
    d = ipfpga_to_e7awgsw
    if len(d) == 1:
        ipfpga = list(d.keys())[0]
        e = d[ipfpga]
        awg_to_wave_sequence = e["awg_to_wave_sequence"]
        capt_module_to_capt_param = e["capt_module_to_capt_param"]
        trigger_awg = e["trigger_awg"]
        s, r = send_recv_single(
            ipfpga, awg_to_wave_sequence, capt_module_to_capt_param, trigger_awg
        )
        result[ipfpga] = {}
        result[ipfpga]["send"] = s
        result[ipfpga]["recv"] = r
    else:
        raise ValueError(
            "Multi Qube Sync is not implemented yet in meas.send_recv function."
        )

    return result
