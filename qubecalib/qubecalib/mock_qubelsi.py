import logging

logger = logging.getLogger(__name__)


class Qube:
    def __init__(self, addr: str, path: str):
        self.addr = addr
        self.ad9082 = []
        self.lmx2594 = []
        self.lmx2594_ad9082 = []
        self.adrf6780 = []
        self.ad5328 = []
        self.gpio = []

    def do_init(self, rf_type: str):
        logger.info(f"do_init() is called for Qube@{self.addr}")
