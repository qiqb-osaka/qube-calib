def test_wave() -> None:
    import qubecalib as qc
    from e7awgsw import CaptureParam, WaveSequence

    w = WaveSequence(num_wait_words=16, num_repeats=1)  # 1 word = 4 samples = 8 ns
    w.add_chunk(
        iq_samples=64
        * [
            (32767, 0),
        ],
        num_blank_words=0,
        num_repeats=1,
    )
    # 16 word rect pulse

    p = CaptureParam()
    p.num_integ_sections = 1
    p.add_sum_section(num_words=32 * 16 - 1, num_post_blank_words=1)
    p.capture_delay = 16

    qcs = qc.qcsys.QcSystem("qube_riken_1-08.yml", "qube_ou_2-10.yml")
    box = qcs.box["qube_riken_1-08"]
    status, data = qc.QcWaveSubsystem.send_recv(
        (box.port0.channel, w),
        (box.port1.channel0, p),
        (box.port1.channel1, p),
        (box.port12.channel0, p),
        triggering_channel=[box.port0.channel],
    )
