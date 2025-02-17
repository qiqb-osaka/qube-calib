from __future__ import annotations

from quel_ic_config import Quel1BoxWithRawWss

from .system import BoxPool

DEFAULT_SIDEBAND = "U"


class PortConfigAcquirer:
    def __init__(
        self,
        boxpool: BoxPool,
        box_name: str,
        box: Quel1BoxWithRawWss,
        port: int | tuple[int, int],
        channel: int,
    ):
        # boxpool にキャッシュされている box の設定を取得する
        if box_name not in boxpool._box_config_cache:
            boxpool._box_config_cache[box_name] = box.dump_box()
        dump_box = boxpool._box_config_cache[box_name]["ports"]
        self.dump_config = dp = dump_box[port]
        sideband = dp["sideband"] if "sideband" in dp else DEFAULT_SIDEBAND
        fnco_freq = 0
        if port in box.get_output_ports():
            fnco_freq = dp["channels"][channel]["fnco_freq"]
        if port in box.get_input_ports():
            fnco_freq = dp["runits"][channel]["fnco_freq"]
            if port in box.get_read_input_ports():
                lpbackps = box.get_loopbacks_of_port(port)
                if lpbackps:
                    lpbackp = next(iter(lpbackps))
                    dumped_port = dump_box[lpbackp]
                    sideband = (
                        dumped_port["sideband"]
                        if "sideband" in dumped_port
                        else DEFAULT_SIDEBAND
                    )
            elif port in box.get_monitor_input_ports():
                lpbackps = box.get_loopbacks_of_port(port)
                if lpbackps:
                    lpbackp = next(iter(lpbackps))
                    dumped_port = dump_box[lpbackp]
                    sideband = (
                        dumped_port["sideband"]
                        if "sideband" in dumped_port
                        else DEFAULT_SIDEBAND
                    )
        self.lo_freq: float = dp["lo_freq"]
        self.cnco_freq: float = dp["cnco_freq"]
        self.fnco_freq: float = fnco_freq
        self.sideband: str = sideband
        self._box_name = box_name
        self._port = port
        self._channel = channel

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lo_freq={self.lo_freq}, cnco_freq={self.cnco_freq}, fnco_freq={self.fnco_freq}, sideband={self.sideband})"
