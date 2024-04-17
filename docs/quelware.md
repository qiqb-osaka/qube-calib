quelware 対応版の参考コード

パルス生成の実行コード
```python
import qubecalib as qc
from e7awgsw import WaveSequence, CaptureParam
w = WaveSequence(num_wait_words=16,num_repeats=1) # 1 word = 4 samples = 8 ns
w.add_chunk(iq_samples=64*[(32767,0),],num_blank_words=0,num_repeats=1)
# 16 word rect pulse

p = CaptureParam()
p.num_integ_sections = 1
p.add_sum_section(num_words=32*16-1, num_post_blank_words=1)
p.capture_delay = 16

qcs = qc.qcsys.QcSystem("qube_riken_1-08.yml","qube_ou_2-10.yml")
box = qcs.box["qube_riken_1-08"]
status, data = qc.QcWaveSubsystem.send_recv((box.port0.channel, w),(box.port1.channel0, p),(box.port1.channel1, p),(box.port12.channel0, p),triggering_channel=[box.port0.channel])
```

データのプロットコード（2.0.1 と同じはず）
```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

cap_data = data[box.port1.channel0][0][:,0]
plt.plot(np.real(cap_data))
plt.plot(np.imag(cap_data))
```

quelware に対応し `send_recv()` の形式で単体筐体でのパルス送受信できる．
2.0.1 では send_recv に対して `(Qube.Port.AWG, WaveSequence)` や `(Qube.Port.Capt.UNIT, CaptParam)` を
与えると送受信同期してパルスが生成される様にしたが，2.1.0 では `(QcBox.Port.TxChannel, WaveSequence)` や `(QcBox.Port.RxChannel)` および
`triggering_channel = [QcBox.Port.TxChannel]` を与えると同様に機能する．送受信同期するためには `triggering_channel` を筐体毎に 1 つだけ指定
しなければならない．指定しなければ AWG と Capture はそれぞれ独自のタイミングで走り出す（送受信は同期しない）．複数台同期はのちに対応する予定．

現場は status は必ず SUCCESS を返すが，近々にきちんとチェックする予定

neopulse の変換結果をこの形式に対応させ adi_api_mod から切り離すのにもう少し作業が必要
