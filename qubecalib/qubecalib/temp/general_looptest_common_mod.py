from __future__ import annotations

import socket
from typing import Any, Mapping

from quel_clock_master import SequencerClient
from quel_ic_config_utils import SimpleBoxIntrinsic, create_box_objects

from .general_looptest_common import BoxPool


class BoxPoolMod:
    @classmethod
    def create_and_add_box_object(
        self, boxpool: BoxPool, boxname: str, setting: Mapping[str, Any]
    ) -> SimpleBoxIntrinsic:
        try:
            _, _, _, _, box = create_box_objects(**setting, refer_by_port=False)
        except socket.timeout:
            raise TimeoutError(f"timeout {boxname}")

        if not isinstance(box, SimpleBoxIntrinsic):
            raise ValueError(f"unsupported boxtype: {setting['boxtype']}")
        sqc = SequencerClient(setting["ipaddr_sss"])
        boxpool._boxes[boxname] = (box, sqc)
        boxpool._linkstatus[boxname] = False
