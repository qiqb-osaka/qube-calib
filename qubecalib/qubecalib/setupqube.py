import e7awgsw
import time
import numpy as np
from typing import Final
from concurrent.futures import ThreadPoolExecutor
from .qube import Readout, Readin, Ctrl, SSB
from .pulse import Read, Schedule, Blank, Rect, Arbit, Arbitrary, SlotWithIQ
from .meas import Send, Recv, WaveSequenceFactory, CaptureParam, CaptureModule, CaptureCtrl, AwgCtrl
from collections import namedtuple
import copy

def words(t): # in ns
    return int(t * 1e-9 * AwgCtrl.SAMPLING_RATE // e7awgsw.hwparam.NUM_SAMPLES_IN_AWG_WORD)

def find_allports(schedule, klass):
    """
    schedule で使われているすべてのポートのに割り当てられたチャネルのリストを返す
    """
    r = []
    for k, v in schedule.items():
        if isinstance(v.wire.port, klass):
            r.append(v)
    return r

def select_band(channel):
    """
    使用するバンドを選択する
    """
    fc = channel.center_frequency
    for b in channel.wire.band:
        if b.range[0] < fc and fc <= b.range[1]:
            return b
    raise ValueError('Invalid center_frequency {}.'.format(fc))

# スロットをマージする際に必要な占有範囲を保持する構造体
# range: (int, int), slots: Slot
RangeWithSlot = namedtuple('RangeWithSlot', ('range', 'slots'))

def conv_to_nonblank_ranges(channel):
    """
    与えられたチャネルに属する非空白スロットの範囲を抽出したリストを返す
    """
    r: Final[List] = []
    cursor = 0
    for slot in channel:
        tail = cursor + slot.duration
        if not isinstance(slot, Blank):
            r.append(
                RangeWithSlot(
                    range=(cursor, tail),
                    slots=[RangeWithSlot(
                        range=(cursor, tail),
                        slots=slot
                    )]
                )
            )
        cursor = tail
    return r

def is_overlap(a, b):
    """
    Slot a, b の占有範囲が重なっているか調べる
    """
    c = (b[0] <= a[0]) * (a[0] < b[1]) # < b [ a ] > ]
    d = (b[0] < a[1]) * (a[1] <= b[1]) # [ < b [ a ] >
    e = (a[0] <= b[0]) * (b[0] < a[1]) # [ a < b > ] >
    f = (a[0] < b[1]) * (b[1] <= a[1]) # < [ a < b > ]
    if c + d + e + f:
        return True
    else:
        return False

def merge_slots(rng, sampling_rate):
    """
    スロットを実際にマージする
    本当なら含まれる最大のサンプリングレートでオーバーサンプルすべきだけど未実装
    """
    # マージする器を作る
    rslt = RangeWithSlot(range=rng.range, slots=Arbitrary(duration=rng.range[1]-rng.range[0], amplitude=1, sampling_rate=sampling_rate))
    # サンプル時間を生成する
    t = np.arange(rng.range[0], rng.range[1], int(1 / sampling_rate * 1e+9)) # [ns]
    # 各スロットの iq の値を合成してゆく
    for s in rng.slots:
        if not isinstance(s.slots, Read):
            idx = (s.range[0] <= t) & (t < s.range[1]) # 占有範囲をインデックス化する
            rslt.slots.iq[idx] += s.slots.iq * s.slots.amplitude # 所定の場所に所定の振幅で加算する
    return rslt

def padding(channel, duration):
    """
    隙間を Blank でパディングする
    channel には非空白スロットのみが与えられていると仮定
    """
    rslt: Final[Channel] = channel.renew()
    cursor = 0
    for o in channel:
        # 次のスロットの前に Blank を埋める
        rslt.append(Blank(duration=o.range[0]-cursor))
        rslt.append(o.slots)
        cursor = o.range[1]
    # お尻の隙間を Blank で埋める
    if duration != cursor:
        rslt.append(Blank(duration=duration-cursor))
    return rslt

Padding = namedtuple('Padding', ('duration'))
Shortage = namedtuple('Shortage', ('duration'))

def quantize_blank(channel):
    r = []
    for i in range(len(channel)):
        slot = channel[i]
        if isinstance(slot, Blank):
            # もしひとつ前がはみ出していたら短くする
            shortage_duration = 0
            if i > 0:
                if isinstance(channel[i - 1], Shortage):
                    shortage_duration = channel[i - 1].duration
            shorten_duration = slot.duration - shortage_duration
            if i > 0 and shorten_duration < 1:
                # Blank が食われてしまう処理は少し面倒なので後回しに
                raise ValueError('Too close pulse in same wire. Not Implemented yet.')
            div, mod = divmod(shorten_duration, 8) # Blank は 8ns の整数倍にしなければいけない
            if mod: # 余りがある
                if not div: # 8ns より短い場合は直後の Slot に吸収してもらわなければいけない
                    r.append(Padding(duration=mod))
                else: # 8ns より長い場合は Blank をちょうど良い長さに切り詰めて余りを吸収してもらう
                    r.append(Blank(duration=8*div))
                    r.append(Padding(duration=mod))
            else: # 丁度いい長さの場合は追加する
                new = copy.copy(slot)
                new.duration = shorten_duration
                r.append(new)
        elif isinstance(slot, Arbitrary): # 任意波形の場合はそのまま追加
            r.append(copy.copy(slot))
        elif isinstance(slot, Padding) or isinstance(slot, Shortage): # Padding と Shortage は次に処理されるはず
            pass
        else:
            raise ValueError('Critical Bug...')
    return r

def quantize_wave(channel):
    r = []
    for i in range(len(channel)):
        slot = channel[i]
        if isinstance(slot, Arbitrary):
            # もしひとつ前に Padding があれば吸収して長くする
            padding_duration = 0
            if i > 0:
                if isinstance(channel[i - 1], Padding):
                    padding_duration = channel[i - 1].duration
                    if divmod(padding_duration, 2)[1]:
                        raise ValueError('Resolution Error... < 2ns')
            extended_duration = slot.duration + padding_duration
            # 新しく slot オブジェクトを作る
            new = copy.copy(slot)
            new.duration = extended_duration
            # 長くした分のブランクデータを頭に追加する
            new.iq = np.concatenate([np.zeros(int(padding_duration/2)).astype(complex), slot.iq])
            r.append(new)
            d, m = divmod(extended_duration, 128)
            if m:
                # 余りがでたらピッタリの長さに調整してブランクデータをお尻に追加する
                extention = 128 - m
                if divmod(extention, 2)[1]:
                    raise ValueError('Resolution Error... < 2ns')
                new.duration += extention
                new.iq = np.concatenate([new.iq, np.zeros(int(extention/2)).astype(complex)])
                # お尻に追加したブランクデータ長分だけはみ出したことを記録する
                r.append(Shortage(duration=new.duration-(slot.duration+padding_duration)))
        elif isinstance(slot, Blank):
            r.append(copy.copy(slot))
        elif isinstance(slot, Padding) or isinstance(slot, Shortage):
            pass
        else:
            raise ValueError('Critical Bug...')
    return r
    
    
def quantize_channel(channel):
    rslt = channel.renew()
    buf = channel
    while True:
        buf = quantize_wave(quantize_blank(buf))
        # Shortage や Padding がなくなるまで繰り返す
        if not list(filter(lambda x: isinstance(x, Shortage) or isinstance(x, Padding), buf)):
            break
    for e in buf:
        rslt.append(e)
    return rslt
    
def calc_modulation_frequency(channel):
    lo_mhz = channel.wire.port.lo.mhz 
    cnco_mhz = channel.wire.port.nco.mhz
    fnco_mhz = select_band(channel).fnco_mhz
    rf_usb = (lo_mhz + (cnco_mhz + fnco_mhz)) * 1e+6
    rf_lsb = (lo_mhz - (cnco_mhz + fnco_mhz)) * 1e+6
    if isinstance(channel.wire.port, Readin):
        fc = channel.center_frequency - rf_usb
    else:
        if channel.wire.port.mix.ssb == SSB.USB:
            fc = channel.center_frequency - rf_usb
        elif channel.wire.port.mix.ssb == SSB.LSB:
            fc = rf_lsb - channel.center_frequency
        else:
            raise ValueError('A port.mix.ssb shuld be instance of SSB(Enum).')
    return fc
    
    
def modulation(channel, offset_time=0):
    rslt = channel.renew()
    hz = calc_modulation_frequency(channel)
    for slot in channel:
        new = copy.copy(slot)
        if isinstance(slot, SlotWithIQ):
            local_offset = channel.get_offset(slot) # <-- get_offset_time
            t = (slot.timestamp + local_offset - offset_time) * 1e-9 # [s]
            new.iq[:] = slot.iq * np.exp(1j * 2 * np.pi * hz * t)
        rslt.append(new)
    return rslt
    

def merge_channels(channels):
    """
    与えられた複数のチャネルを一つのチャネルにマージする
    同じワイヤに割り振られたチャネルのリストが与えられていることを想定している
    """
    slts = [] # スロットをマージした結果を保持するバッファ
    # 全ての非空白スロットから占有範囲を抽出してひとつのリストに集める
    rngs = [r for c in channels for r in conv_to_nonblank_ranges(c)]
    while True:
        #
        # 占有範囲が重なっているスロットを先頭からマージしてゆく
        # マージの際に 128ns の制約から占有範囲が膨らむことが多いので繰り返しチェックする
        #
        def filter_overlpped_slots_with_first(rngs):
            """
            先頭のスロットとオーバーラップしたスロットとその他を分ける
            """
            ovlp = [rngs[0]] # overlapping slots
            oths = [] # others
            for o in rngs[1:]:
                (ovlp if is_overlap(rngs[0].range, o.range) else oths).append(o)
            return ovlp, oths
        ovlp, oths = filter_overlpped_slots_with_first(rngs)
        # 重なったスロットをひとまとまりにして記録する
        slts.append(
            RangeWithSlot(
                range=(
                    min([o.range[0] for o in ovlp]),
                    max([o.range[1] for o in ovlp])
                ),
                slots=sum([rng.slots for rng in ovlp], [])
            )
        )
        # もし処理する残りがひとつ以下になったらマージ作業を終了する
        if len(oths) < 2:
            if oths: # もしひとつだけ残っていればそれを追加（回収）する
                slts.append(oths[0])
            break
        # 処理する残りがあれば oths に対して処理しを続ける
        rngs = oths
    # マルチサンプリングレート対応の試みの名残（現状では不要）
    m = max_sampling_rate = max([slot.sampling_rate for chnl in channels for slot in chnl])
    # マージするチャネルを用意．スロットのみ空の Channel を複製．
    chnl = channels[0].renew()
    # slts に格納された各スロットを開始時刻順にソートする
    sorted_index = np.array([rng.range[0] for rng in slts]).argsort().astype(int)
    # ソートした順にチャネルに追加する
    for rng in [slts[i] for i in sorted_index]:
        chnl.append(merge_slots(rng, sampling_rate=m)) # スロットを実際にマージする
    # マージしたチャネルの継続時間を計算する
    duration = max([sum([slt.duration for slt in chnl]) for chnl in channels])
    # 隙間を Blank でパディングする
    rslt = padding(chnl, duration)
    return rslt
    
def conv_channel_for_e7awgsw(channels, offset_time):
    """
    Blank で始まり Blank で終わる
    """
    modulated_channels = [modulation(channel, offset_time) for channel in channels]
    merged_channel = merge_channels(modulated_channels)
    quantized_channel = quantize_channel(merged_channel)
    if not isinstance(quantized_channel[0], Blank):
        quantized_channel.insert(0, Blank(duration=0))
    if not isinstance(quantized_channel[-1], Blank):
        quantized_channel.append(Blank(duration=0))
    return quantized_channel
    

def run(schedule, repeats=1, interval=100000):
    """
    同一筐体 Qube でのスケジュール実行
    """
    s: Final[Schedule] = schedule
    
    def collect_channel():
        """
        同じ Wire に割り当てられたチャネルのリストを返す
        """
        channels = {}
        for k, c in schedule.items():
            if not len(c):
                continue
            if c.wire in channels:
                channels[c.wire].append(c) # すでに Wire が登録されていれば追加
            else:
                channels[c.wire] = [c] # 新しく登録する Wire ならリストを生成
        return channels
    channels = collect_channel()
    
    # どの ipfpga の qube が使われているかリストする
    qube_ipfpgas = list(set([k.port.capt.ipfpga if isinstance(k.port, Readin) else k.port.awg0.ipfpga for k, v in channels.items()]))
    # このバージョンでは単一筐体を仮定しているので第一要素を ipfpga とする
    ipfpga = qube_ipfpgas[0]
    
    w2c = dict([(k, conv_channel_for_e7awgsw(v, schedule.offset)) for k, v in channels.items()])
    # すべてのチャネルの全長を揃える
    m = max([v.duration for k, v in w2c.items()])
    for k, v in w2c.items():
        v[-1].duration += m - v.duration
    
    def conv_channel_to_e7awgsw_wave_sequence(c):
        w = WaveSequenceFactory(num_wait_words=words(c[0].duration), num_repeats=repeats)
        for i in range(1, len(c), 2):
            w.new_chunk(duration=c[i].duration * 1e-9, amp=int(c[i].amplitude * 32767), blank=0)
            w.chunk[-1].iq[:] = c[i].iq[:]
            w.chunk[-1].blank = c[i+1].duration * 1e-9
        return w
    conv = conv_channel_to_e7awgsw_wave_sequence
    send = [(k, conv(v)) for k, v in w2c.items() if not isinstance(k.port, Readin)]
    o = Send(ipfpga, [o.port.awg for o, w in send], [w.sequence for o, w in send])
    
    def conv_channel_to_e7awgsw_capture_param(channel):
        delay_words = words(channel.wire.delay)
        blank_words = 0
        c = channel.copy()

        if isinstance(c[0], Blank):
            w = words(c[0].duration)
            delay_words += w
            blank_words += w
            c.pop(0)
        
        # 複数の総和区間に対応させる！ 10/21
        # fnco の値を変える
        # モニタ対応
        
        for ci in c:
            if isinstance(ci, Read) or isinstance(ci, Arbitrary):
                capture_words = words(ci.duration)
                continue
            if isinstance(ci, Blank):
                blank_words += words(ci.duration)
                continue
            raise ValueError('Invalid LWC.')

        p = CaptureParam()
        p.num_integ_sections = repeats
        if not (repeats == 1):
            p.add_sum_section(capture_words, blank_words + words(interval))
            p.sum_start_word_no = 0
            p.num_words_to_sum = e7awgsw.CaptureParam.MAX_SUM_SECTION_LEN
            p.sel_dsp_units_to_enable(e7awgsw.DspUnit.INTEGRATION)
        p.capture_delay = delay_words

        return p
    conv = conv_channel_to_e7awgsw_capture_param
    recv = [(w, conv(c)) for w, c in w2c.items() if isinstance(w.port, Readin)]
    
    w = {w: p for w, p in recv}
    r = Recv(ipfpga, [v.port.capt for v, p in w.items()], [p for v, p in w.items()])
    # トリガを設定する
    # Readout_send の Wire のリストを得る
    readout_sends = list(set([k for k, v in channels.items() if isinstance(k.port, Readout)]))
    r.trigger = readout_sends[0].port.awg0
    
    with ThreadPoolExecutor() as e:
        rslt = e.submit(lambda: r.wait(timeout=5))
        time.sleep(0.1)
        o.terminate()
        o.start()
        rslt.result()
        
    o.terminate()
    
    for c in [c for k, c in schedule.items() if isinstance(c.wire.port, Readin)]:
        k = CaptureModule.get_units(c.wire.port.capt.id)[0]
        if [s for s in c if isinstance(s, Read)]:
            c.timestamp = t = (c.get_timestamp(c.findall(Read)[0]) - s.offset) * 1e-9
        else:
            c.timestamp = t = (c.get_timestamp(c.findall(Arbit)[0]) - s.offset) * 1e-9
        c.iq = np.zeros(r.data[k].shape).astype(complex)
        # 復調したデータを格納する
        c.iq[:] = r.data[k] * np.exp(-1j * 2 * np.pi * calc_modulation_frequency(c) * t)
    
    
def maintenance_run(schedule, repeats=1, interval=100000, duration=5242880, capture_delay=0):
    s: Final[Schedule] = schedule
    
    # Search Ctrl
    fm = {}
    for c in find_allports(s, Ctrl):
        if c.wire in fm:
            fm[c.wire].append(c)
        else:
            fm[c.wire] = [c]
    ctrl = [(c, conv_wseq(c, repeats, interval, fm, s)) for c in find_allports(s, Ctrl)]
    
    # Search Readout
    fm = {}
    for c in find_allports(s, Readout):
        if c.wire in fm:
            fm[c.wire].append(c)
        else:
            fm[c.wire] = [c]
    readout = [(c, conv_wseq(c, repeats, interval, fm, s)) for c in find_allports(s, Readout)]

    # Search Readin
    readin = [(c, conv_capt_param(c, repeats, interval)) for c in find_allports(s, Readin)]
    for c, p in readin:
        p.add_sum_section(words(duration), 1)
        p.capture_delay += words(capture_delay)
    
    # [Todo]: Read を複数使えるように拡張（複数の capt, 対応するトリガの設定）
    o = Send(
        readout[0][0].wire.port.awg.ipfpga,
        [o.wire.port.awg for o, w in readout + ctrl],
        [w.sequence for o, w in readout + ctrl]
    )
    w = dict([(c.wire, p) for c, p in readin])
    r = Recv(
        readin[0][0].wire.port.capt.ipfpga,
        [v.port.capt for v, p in w.items()]
    )
    r.trigger = readout[0][0].wire.port.awg
    
    with ThreadPoolExecutor() as e:
        rslt = e.submit(lambda: r.wait(readin[0][1], timeout=5))
        time.sleep(0.1)
        o.terminate()
        o.start()
        rslt.result()
        
    o.terminate()
    
    for c, p in readin:
        k = CaptureModule.get_units(c.wire.port.capt.id)[0]
        c.iq = np.zeros(r.data[k].shape).astype(complex)
        c.timestamp = t = (np.arange(0, r.data[k].shape[0] / AwgCtrl.SAMPLING_RATE, 1 / AwgCtrl.SAMPLING_RATE) - s.offset * 1e-9)
        c.iq[:] = r.data[k] * np.exp(-1j * 2 * np.pi * calc_modulation_frequency(c) * t)
