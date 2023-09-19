import datetime
import os
import struct
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from typing import Union

import e7awgsw
import numpy as np
from e7awgsw import AwgCtrl, CaptureCtrl, DspUnit, WaveSequence
from quel_clock_master import QuBEMasterClient, SequencerClient

from .meas import CaptureModule, CaptureParam, WaveSequenceFactory
from .pulse import Arbit, Read
from .qube import AWG, CPT, UNIT
from .setupqube import _conv_channel_for_e7awgsw, _conv_to_e7awgsw

PORT = 16384
IPADDR = "10.3.0.255"
REPEAT_WAIT_SEC = 0.1
REPEAT_WAIT_SEC = int(REPEAT_WAIT_SEC * 125000000)  # 125Mcycles = 1sec
CANCEL_STOP_PACKET = struct.pack(8 * "B", 0x2C, *(7 * [0]))


def check_clock(*qubes, ipmaster="10.3.0.255"):
    ipmulti = [q.ipmulti for q in qubes]
    m = QuBEMasterClient(ipmaster, 16384)
    s = [SequencerClient(ip, seqr_port=16384, synch_port=16385) for ip in ipmulti]
    c = [o.read_time() for o in s]
    c.append(m.read_clock())
    return c


def kick(*qubes, delay=1):  # from QubeServer.py
    destinations = [q.ipmulti for q in qubes]
    DAQ_INITSDLY = delay
    cDAQ_SDLY_TAG = DAQ_INITSDLY
    SYNC_CLOCK = 125_000_000  # 125Mcycles = 1sec

    delay = int(cDAQ_SDLY_TAG * SYNC_CLOCK + 0.5)

    seq_cli = {a: SequencerClient(a, seqr_port=16384, synch_port=16385) for a in destinations}
    clock = seq_cli[destinations[0]].read_time() + delay

    for a in destinations:
        seq_cli[a].add_sequencer(16 * (clock // 16 + 1))


def search_qube(o):
    return o.port.qube if isinstance(o, AWG) else o.capt.port.qube if isinstance(o, UNIT) else None


def extract_qubes(*setup):
    return tuple(set([search_qube(o1) for o1, o2 in setup]))


def split_qube(*setup):
    rslt = {q: [] for q in extract_qubes(*setup)}
    for o1, o2 in setup:
        rslt[search_qube(o1)].append((o1, o2))
    return rslt


def split_send_recv(*setup):
    send = tuple((o1, o2) for o1, o2 in setup if isinstance(o1, AWG))
    recv = tuple((o1, o2) for o1, o2 in setup if isinstance(o1, UNIT))
    return send, recv


# WaveSequence と CaptureParam に直接アクセスできること
# 複数台の Qube を扱う場合にも煩雑にならないこと


class PulseSet(object):
    # 複数台の Qube へのパルス設定 (wave_sequence と capture_parameter) を管理する
    # 一括設定変更のサポート関数なども用意
    pass


class PulseSubSet(object):
    # Qube 単位でのパルス管理
    # どの awg へ，どの capture_unit へ．
    pass


class PulseConverter(object):
    Container = namedtuple(
        "Container", ["awg_to_wavesequence", "capt_to_captparam", "capt_to_mergedchannel", "adda_to_channels"]
    )

    # repeats は後から設定できるようにするのが e7awgsw 的に良さそう
    # その後に各種 DSP を設定する
    @classmethod
    def conv(cls, channels, offset=0, interval=0):
        # qube.port に delay を後付け設定しているので忘れていないか確認する
        for k, v in channels.items():
            if isinstance(k, CPT):
                try:
                    k.port.delay
                except AttributeError:
                    raise AttributeError("delay attribute is required for receiver port")
        r = _conv_to_e7awgsw(adda_to_channels=channels, offset=offset, repeats=1, interval=interval, trigger_awg=None)
        # captparam を複製して DSP の設定をクリアする
        func = lambda v: {k2: (cls.duplicate_captparam(v2),) for k2, v2 in v["capt_to_captparam"].items()}  # noqa: E731
        qube_channels = lambda qube: {k: v for k, v in channels.items() if k.port.qube == qube}  # noqa: E731
        return dict(
            [
                (k, cls.Container(v["awg_to_wavesequence"], func(v), v["capt_to_mergedchannel"], qube_channels(k)))
                for k, v in r.items()
            ]
        )

    @classmethod
    def duplicate_captparam(cls, cp, repeats=1, interval=0, delay=0):
        p = CaptureParam()
        p.num_integ_sections = cp.num_integ_sections
        p.capture_delay = cp.capture_delay
        for i in range(repeats):
            for s in cp.sum_section_list:
                p.add_sum_section(*s)
        return p

    @classmethod
    def temp(cls, adda_to_channels):
        channels = adda_to_channels

        print(list(set([k.port.qube for k, v in channels.items()])))

        # チャネルを周波数変換し，時間領域でチャネルを結合し，e7awgsw の制約に合わせてスロットを再配置する
        w2c = dict([(k, _conv_channel_for_e7awgsw(v, k, 0)) for k, v in channels.items()])
        # quantized_channel? merged_channel?

        print([w.port.delay for w, c in w2c.items() if isinstance(w, CPT)])

        return w2c
        # return _conv_to_e7awgsw(adda_to_channels)


class Recv(CaptureCtrl):
    def __init__(self, *unit_cprm_pair):
        # argparse = lambda module, *params: (module, params)
        # cond = lambda o: isinstance(o,tuple) or isinstance(o,list)
        # arg = tuple([argparse(o[0],*(o[1] if cond(o[1]) else (o[1],))) for o in module_params_pair])

        self.parms = arg = unit_cprm_pair

        # typing で書くのがいまどき？
        if not [isinstance(m, UNIT) for m, l in arg] == len(arg) * [True]:
            raise TypeError("1st element of each tuple should be qubecalib.qube.UNIT instance.")

        # if not [qube == m.port.qube for m, l in arg] == len(arg)*[True]:
        #    raise Exception('The qube that owns the CaptureModule candidates in the arguments must all be identical.')

        # if not [len(l) < 5 for m, l in arg] == len(arg)*[True]:
        #    raise Exception('Each CaptureParameter list in the argument must have no longer than 4 elements.')

        qubes = list(set([u.capt.port.qube for u, p in arg]))
        if len(qubes) > 1:
            raise Exception("All awg objects in the argument must belong to a common qube object.")

        super().__init__(qubes[0].ipfpga)

        # self._trigger = None # obsoleted
        # self.modules = [m for m, l in arg]
        # self.units = sum([self.assign_param_to_unit(m, l) for m, l in arg],[])

        self.capts = list(set([u.capt.id for u, p in arg]))
        self.units = units = [u.id if isinstance(u, UNIT) else u for u, p in arg]

        self.initialize(*units)
        for u, p in arg:
            uu = u.id if isinstance(u, UNIT) else u
            self.set_capture_params(uu, p)

    def assign_param_to_unit(self, module, params):  # obsoleted
        m = module
        units = [u for u in CaptureModule.get_units(m if isinstance(m, CaptureModule) else m.id)[: len(params)]]
        self.initialize(*units)
        for u, p in zip(units, params):
            self.set_capture_params(u, p)

        return units

    def start(self, timeout=30):
        u = self.units
        self.start_capture_units(*u)
        self.wait_for_capture_units_to_stop(timeout, *u)
        self.check_err(*u)

    def prepare_for_trigger(self, awg: Union[AWG, e7awgsw.AWG]):
        for c in self.capts:
            self.select_trigger_awg(c, awg.id if isinstance(awg, AWG) else awg)
        self.enable_start_trigger(*self.units)
        # trig = awg if isinstance(awg, e7awgsw.AWG) else awg.id
        # for m in self.modules:
        #    self.select_trigger_awg(m.id, trig)
        # self.enable_start_trigger(*self.units)

    def wait_for_capture(self, timeout=30):
        self.wait_for_capture_units_to_stop(timeout, *self.units)
        self.check_err(*self.units)

    def get(self):
        return {u: self._get(u, p) for u, p in self.parms}

    def _get(self, u, p):
        u = u.id if isinstance(u, UNIT) else u
        l = p.num_integ_sections  # noqa: E741
        m = len(p.sum_section_list)
        n = self.num_captured_samples(u)
        if DspUnit.CLASSIFICATION in p.dsp_units_enabled:
            d = np.array(list(self.get_classification_results(u, n)))
        else:
            c = np.array(self.get_capture_data(u, n))
            d = c[:, 0] + 1j * c[:, 1]
        if DspUnit.INTEGRATION in p.dsp_units_enabled:
            d = d.reshape(1, -1)
        else:
            d = d.reshape(l, -1)
        if DspUnit.SUM in p.dsp_units_enabled:
            d = np.hsplit(d, list(range(m)[1:]))
        else:
            d = np.hsplit(d, [w * p.NUM_SAMPLES_IN_ADC_WORD for w, b in p.sum_section_list[:-1]])
            d = [di.transpose() for di in d]

        return d

    #     def check_err(self, *units):
    #
    #         e = super().check_err(*units)
    #         print(e)
    #         if any(e):
    #             raise IOError('CaptureCtrl error.')

    def wait(self, timeout=30):  # obsoleted
        u = self.units
        self.enable_start_trigger(*u)
        self.wait_for_capture_units_to_stop(timeout, *u)
        self.check_err(*u)

    @property
    def trigger(self):  # obsoleted
        return self._trigger

    @trigger.setter
    def trigger(self, awg):  # obsoleted
        self._trigger = awg if isinstance(awg, e7awgsw.AWG) else awg.id
        for m in self.modules:
            self.select_trigger_awg(m.id, self._trigger)


# class RecvSingleMod(Recv): # for easy access interface

#     def __init__(self, module, params):

#         super().__init__(((module, params),))


class CaptMemory(e7awgsw.CaptureCtrl):  # for data access
    def get_data(self, unit):
        n = self.num_captured_samples(unit)
        c = np.array(self.get_capture_data(unit, n))
        return c[:, 0] + 1j * c[:, 1]


class Send(AwgCtrl):
    def __init__(self, *awg_wseq_pair):
        arg = awg_wseq_pair

        # typing で書くのがいまどき？
        if not [
            (isinstance(a, AWG) or isinstance(a, e7awgsw.AWG))
            and (isinstance(s, WaveSequenceFactory) or isinstance(s, WaveSequence))
            for a, s in arg
        ] == len(arg) * [True]:
            raise TypeError(
                "Element type of each tuple should be (qubecalib.qube.AWG, qubecalib.meas.WaveSequenceFactory)."
            )

        # if not [qube == a.port.qube for a, s in arg] == len(arg)*[True]:
        #    raise Exception('The qube that owns the AWG candidates in the arguments must all be identical.')

        qubes = list(set([a.port.qube for a, w in arg]))
        if len(qubes) > 1:
            raise Exception("All awg objects in the argument must belong to a common qube object.")

        super().__init__(qubes[0].ipfpga)

        lawgs = lambda x: x.id if isinstance(x, AWG) else x  # noqa: E731
        lseqs = lambda x: x.sequence if isinstance(x, WaveSequenceFactory) else x  # noqa: E731
        self.awgs = awgs = [lawgs(a) for a, w in arg]

        self.initialize(*awgs)
        for a, s in arg:
            self.set_wave_sequence(lawgs(a), lseqs(s))

    def start(self):
        # a = [a.id for a in self.awgs]
        # self.terminate_awgs(*a)
        # self.clear_awg_stop_flags(*a)
        self.start_awgs(*self.awgs)

    send = start

    def prepare_for_sequencer(self, timeout=30):
        # a = [a.id for a in self.awgs]
        # self.terminate_awgs(*a)
        # self.clear_awg_stop_flags(*a)
        print("wait:", datetime.datetime.now())
        # print('wait for started by sequencer for {}'.format(self.awgs[0].port.qube.ipfpga))
        self.wait_for_awgs_to_stop(timeout, *self.awgs)
        print("awg done:", datetime.datetime.now())
        print("end")


# class SendSingleAwg(Send):

#     def __init__(self, awg, sequence):

#         super().__init__(((awg, sequence),))


class Terminate(AwgCtrl):
    def __init__(self, awgs):
        super().__init__(awgs.port.qube.ipfpga)

        self.awgs = awgs

    def terminate(self):
        self.terminate_awgs(*self.awgs)


class TerminateSingleAwg(Terminate):
    def __init__(self, awg):
        super().__init__((awg,))


def multishot(adda_to_channels, triggers, repeats=1, timeout=30, interval=50000):
    qube_to_pulse = c = PulseConverter.conv(adda_to_channels, interval)

    if len(c.keys()) == 1:
        qube = tuple(c.keys())[0]
        trigger = [o for o in triggers if o.port.qube() == qube][0]
        units = multishot_single(qube, qube_to_pulse[qube], trigger, repeats, timeout, interval)
    else:
        units = multishot_multi(qube_to_pulse, triggers, repeats, timeout, interval)

    return units, qube_to_pulse


def duplicate_captparam(cp):
    p = CaptureParam()
    p.capture_delay = cp.capture_delay
    for s in cp.sum_section_list:
        p.add_sum_section(*s)
    return p


def multishot_multi(pulse, triggers, repeats=1, timeout=30, interval=50000):
    def set_repeats(w, v):
        w.num_repeats = v
        return w

    def enable_integration(p, v):
        n = duplicate_captparam(p)
        n.num_integ_sections = v
        n.sel_dsp_units_to_enable(e7awgsw.DspUnit.INTEGRATION)
        return n

    arg_send = lambda pulse: tuple(  # noqa: E731
        [(a, set_repeats(w, repeats)) for a, w in pulse.awg_to_wavesequence.items()]
    )
    arg_recv = lambda pulse: tuple(  # noqa: E731
        [(c, tuple([enable_integration(i, repeats) for i in p])) for c, p in pulse.capt_to_captparam.items()]
    )
    t = {awg.port.qube: awg for awg in triggers}

    with redirect_stdout(open(os.devnull, "w")):
        with ThreadPoolExecutor() as e:
            capts = [
                e.submit(lambda: wait_for_awg(qube=k, capt_cparam_pair=arg_recv(v), trigger=t[k], timeout=timeout))
                for k, v in pulse.items()
                if v.capt_to_captparam
            ]
            awgs = [
                e.submit(lambda: wait_for_sequencer(qube=k, awg_wseq_pair=arg_send(v), timeout=timeout))
                for k, v in pulse.items()
                if v.awg_to_wavesequence
            ]
            time.sleep(0.1)

            client = QuBEMasterClient(IPADDR, PORT)
            r, a = client.clear_clock(value=0)
            r, a = client.kick_clock_synch([k.ipmulti for k in pulse.keys()])
            mark = client.read_clock(value=0) + REPEAT_WAIT_SEC
            for qube in pulse.keys():
                a = qube.ipmulti
                s = SequencerClient(a, PORT)
                try:
                    r, a = s.add_sequencer(16 * (mark // 16 + 1) + qube.skew)
                except NameError:
                    raise NameError("qube.skew is required")

            for a in awgs:
                a.result()
            units = [c.result() for c in capts]

        for qube in pulse.keys():
            SequencerClient(qube.ipmulti, PORT).send_recv(CANCEL_STOP_PACKET)

    for qube, v in pulse.items():
        for captm, w in v.capt_to_captparam.items():
            channels = v.adda_to_channels[captm]
            multishot_get_data(captm, channels)

    return units


def multishot_single(qube, pulse, trigger, repeats=1, timeout=30, interval=50000):
    def set_repeats(w, v):
        w.num_repeats = v
        return w

    def enable_integration(p, v):
        n = duplicate_captparam(p)
        n.num_integ_sections = v
        n.sel_dsp_units_to_enable(e7awgsw.DspUnit.INTEGRATION)
        return n

    arg_send = tuple([(a, set_repeats(w, repeats)) for a, w in pulse.awg_to_wavesequence.items()])
    arg_recv = tuple(
        [(c, tuple([enable_integration(i, repeats) for i in p])) for c, p in pulse.capt_to_captparam.items()]
    )

    with ThreadPoolExecutor() as e:
        thread = e.submit(lambda: wait_for_awg(qube, arg_recv, trigger, timeout))
        time.sleep(0.1)
        with Send(qube, *arg_send) as s:
            s.start()
        units = thread.result()

    capt_to_mergedchannel = pulse.capt_to_mergedchannel
    retrieve_data_into_mergedchannel(capt_to_mergedchannel, units)

    capt_to_channels = {cpt: ch for cpt, ch in pulse.adda_to_channels.items() if isinstance(cpt, CPT)}

    # 各 Readin チャネルの読み出しスロットに復調したデータを格納する
    for cpt, chs in capt_to_channels.items():
        for ch in chs:
            is_Read_in_Channel = (
                np.sum([True if isinstance(o, Read) else False for o in ch]) == True  # noqa: E712  # FIXME
            )
            if not is_Read_in_Channel:
                continue
            mch = capt_to_mergedchannel[cpt]

            for slot in ch:
                if not isinstance(slot, Read):
                    continue
                # スロットが含まれる多重化スロットを見つける
                for mslot in mch:
                    mt = mch.get_timestamp(mslot)
                    st = ch.get_timestamp(slot)
                    if mt[0] <= st[0] and st[-1] <= mt[-1]:
                        break
                slot.iq = np.zeros(len(st)).astype(complex)
                slot.iq[:] = mslot.iq[(st[0] <= mt) & (mt <= st[-1])]
                slot.iq *= np.exp(-1j * 2 * np.pi * cpt.modulation_frequency(ch.center_frequency * 1e-6) * 1e-3 * st)

    #    for captm, p in arg_recv:
    #        channels = pulse.adda_to_channels[captm]
    #        multishot_get_data(captm, channels)

    return units


def retrieve_data_into_mergedchannel(capt_to_mergedchannel, units, offset=0):
    cpt2ch = capt_to_mergedchannel
    qubes = list(set([k.port.qube for k, v in cpt2ch.items()]))
    if not len(qubes) == 1:
        raise ValueError("There must be single qube in capt_to_mergedchannel.")
    qube = qubes[0]

    # 各 Channel には Read スロットが単一であると仮定
    # 各 CaptureModule に割り当てられた合成チャネルの時間軸とデータを生成する
    for cpt, ch in cpt2ch.items():
        # 合成チャネルの読み出しスロットにデータを埋め込む
        for slt in ch:
            if not isinstance(slt, Arbit):
                continue
            t_ns = ch.get_timestamp(slt)
            slt.iq = np.zeros(len(t_ns)).astype(complex)
            with CaptMemory(qube.ipfpga) as m:
                for unit in units:
                    slt.iq[:] = m.get_data(unit)


def multishot_get_data(captm, channels):
    units = CaptureModule.get_units(captm.id)
    unit_to_channel = {units[i]: v for i, v in enumerate(channels)}

    with CaptMemory(captm.port.qube.ipfpga) as m:
        for unit, channel in unit_to_channel.items():
            slot = channel.findall(Read)[-1]
            v = m.get_data(unit)
            t = np.arange(0, len(v)) / CaptureCtrl.SAMPLING_RATE
            v *= np.exp(-1j * 2 * np.pi * captm.modulation_frequency(channel.center_frequency * 1e-6) * 1e6 * t)
            slot.iq = v


def _singleshot(adda_to_channels, triggers, repeats, timeout=30, interval=50000):
    c = PulseConverter.conv(adda_to_channels, interval)

    if len(c.keys()) == 1:
        qube = tuple(c.keys())[0]
        trigger = [o for o in triggers if o.port.qube == qube][-1]
        units = singleshot_singleqube(qube, c[qube], trigger, repeats, timeout)
    else:
        units = singleshot_multiqube(c, triggers, repeats, timeout)

    return units


singleshot = _singleshot


def singleshot_singleqube(qube, pulse, trigger, repeats, timeout=30, interval=50000):
    def set_repeats(w, v):
        w.num_repeats = v
        return w

    enable_repeats = lambda p, v: PulseConverter.duplicate_captparam(p, repeats=v)  # noqa: E731
    arg_send = tuple([(a, set_repeats(w, repeats)) for a, w in pulse.awg_to_wavesequence.items()])
    arg_recv = tuple([(c, tuple([enable_repeats(i, repeats) for i in p])) for c, p in pulse.capt_to_captparam.items()])

    # for a, w in arg_send:
    #    w.num_repeats = repeats
    # arg_recv = tuple(
    #     [(c, tuple([PulseConverter.duplicate_captparam(pi, repeats=repeats) for pi in p])) for c, p in arg_recv]
    # )

    # with Send(qube, *arg_send) as s, Recv(qube, *arg_recv) as r:
    #    r.wait_for_trigger(trigger)
    #    s.start()
    #    r.wait_for_capture(timeout=timeout)
    #    units = r.units

    with ThreadPoolExecutor() as e:
        thread = e.submit(lambda: wait_for_awg(qube, arg_recv, trigger, timeout))
        time.sleep(0.1)
        with Send(qube, *arg_send) as s:
            s.start()
        units = thread.result()

    captm = arg_recv[0][0]
    for channel in pulse.adda_to_channels[captm]:
        singleshot_get_data(captm, channel, repeats)

    return units


def wait_for_sequencer(qube, awg_wseq_pair, timeout=30):
    with Send(qube, *awg_wseq_pair) as s:
        s.wait_for_sequencer(timeout=timeout)


def wait_for_awg(qube, capt_cparam_pair, trigger, timeout=30):
    with Recv(qube, *capt_cparam_pair) as r:
        r.wait_for_trigger(trigger)
        r.wait_for_capture(timeout=timeout)
        u = r.units

    return u


def singleshot_multiqube(pulse, triggers, repeats, timeout=30, interval=50000):
    def set_repeats(w, v):
        w.num_repeats = v
        return w

    enable_repeats = lambda p, v: PulseConverter.duplicate_captparam(p, repeats=v)  # noqa: E731
    arg_send = lambda pulse: tuple(  # noqa: E731
        [(a, set_repeats(w, repeats)) for a, w in pulse.awg_to_wavesequence.items()]
    )
    arg_recv = lambda pulse: tuple(  # noqa: E731
        [(c, tuple([enable_repeats(i, repeats) for i in p])) for c, p in pulse.capt_to_captparam.items()]
    )
    t = {awg.port.qube: awg for awg in triggers}

    with redirect_stdout(open(os.devnull, "w")):
        with ThreadPoolExecutor() as e:
            capts = [
                e.submit(lambda: wait_for_awg(qube=k, capt_cparam_pair=arg_recv(v), trigger=t[k], timeout=timeout))
                for k, v in pulse.items()
                if v.capt_to_captparam
            ]
            awgs = [
                e.submit(lambda: wait_for_sequencer(qube=k, awg_wseq_pair=arg_send(v), timeout=timeout))
                for k, v in pulse.items()
                if v.awg_to_wavesequence
            ]
            time.sleep(0.1)

            client = QuBEMasterClient(IPADDR, PORT)
            r, a = client.clear_clock(value=0)
            r, a = client.kick_clock_synch([k.ipmulti for k in pulse.keys()])
            mark = client.read_clock(value=0) + REPEAT_WAIT_SEC
            for qube in pulse.keys():
                a = qube.ipmulti
                s = SequencerClient(a, PORT)
                try:
                    r, a = s.add_sequencer(16 * (mark // 16 + 1) + qube.skew)
                except NameError:
                    raise NameError("qube.skew is required")

            for a in awgs:
                a.result()
            units = [c.result() for c in capts]

        for qube in pulse.keys():
            SequencerClient(qube.ipmulti, PORT).send_recv(CANCEL_STOP_PACKET)

    for qube, v in pulse.items():
        for captm, w in v.capt_to_captparam.items():
            for channel in v.adda_to_channels[captm]:
                singleshot_get_data(captm, channel, repeats)

    return units


def singleshot_get_data(captm, channel, repeats):
    unit = CaptureModule.get_units(captm.id)[0]  # <- とりあえず OK
    slot = channel.findall(Read)[0]
    with CaptMemory(captm.port.qube.ipfpga) as m:
        v = m.get_data(unit)
        v = v.reshape(repeats, int(len(v) / repeats))
        t = np.arange(0, len(v[0])) / CaptureCtrl.SAMPLING_RATE
        v *= np.exp(-1j * 2 * np.pi * captm.modulation_frequency(channel.center_frequency * 1e-6) * 1e6 * t)
        slot.iq = v


def standalone_recv(*setup, timeout=30):
    with Recv(*setup) as c:
        c.start(timeout)
        d = c.get()
    return d


def standalone_send(*setup):
    with Send(*setup) as c:
        c.send()


def standalone_send_recv(*setup, trig=None, timeout=30):
    if len(extract_qubes(*setup)) > 1:
        raise Exception("The standalone_send_recve() only accepts awg/capt belonging to a single qube in setup.")
    send, recv = split_send_recv(*setup)
    if not recv:
        raise Exception("Invalid setup.")
    if trig is None:
        trig, _ = send[0]
    with Send(*send) as ac, Recv(*recv) as cc:
        cc.prepare_for_trigger(trig)
        ac.send()
        cc.wait_for_capture(timeout)
        d = cc.get()
    return d


def sync_send(*setup):
    with Send(*setup) as c:
        c.prepare_for_sequencer(10)
    return None


def sync_send_recv(*setup, trig=None, timeout=30):
    if len(extract_qubes(*setup)) > 1:
        raise Exception("The send_recv_single() only accepts awg/capt belonging to a single qube in setup.")
    send, recv = split_send_recv(*setup)
    with Send(*send) as ac, Recv(*recv) as cc:
        cc.prepare_for_trigger(trig)
        ac.prepare_for_sequencer(timeout)
        cc.wait_for_capture(timeout)
        d = cc.get()
    return d


def send_recv(*setup, trigs={}, delay=1, timeout=30):
    setup_qube = split_qube(*setup)
    if len(setup_qube.keys()) == 1:  # standalone mode
        send, recv = split_send_recv(*setup)
        if send and recv:
            q = list(setup_qube.keys())[0]
            if q in trigs:
                trig = trigs[q]
            else:
                trig, _ = send[0]
            rslt = standalone_send_recv(*setup, trig=trig, timeout=timeout)
        elif send and not recv:
            rslt = standalone_send(*setup)
        elif not send and recv:
            rslt = standalone_recv(*setup, timeout=timeout)
        else:
            raise Exception("Invalid setup.")
        return rslt
    elif len(setup_qube.keys()) == 0:
        raise Exception("Invalid setup.")

    with ThreadPoolExecutor() as e:
        threads = []
        for q, s in setup_qube.items():
            send, recv = split_send_recv(*s)
            if not recv:
                threads.append(e.submit(lambda: sync_send(*send)))
            else:
                if q in trigs:
                    trig = trigs[q]
                else:
                    trig, _ = send[0]
                threads.append(e.submit(lambda: sync_send_recv(*s, trig=trig, timeout=timeout)))

        kick(*tuple(setup_qube.keys()), delay=delay)

        dct = {}
        for d in [o for o in [t.result() for t in threads] if o is not None]:
            dct = dct | d

    return dct
