from __future__ import annotations

import threading
import time
from logging import getLogger

from .boxregistry import BoxRegistry

logger = getLogger(__name__)


class KillTimer(threading.Thread):
    """Background thread that periodically cleans up expired Boxes until stopped."""

    def __init__(self, registry: BoxRegistry, interval_sec: int = 10) -> None:
        super().__init__(daemon=True)
        self.registry = registry
        self.interval = interval_sec
        self._stop_flag = threading.Event()

    def run(self) -> None:
        """Run the background thread."""
        logger.info(f"KillTimer started (interval {self.interval} sec).")
        while not self._stop_flag.is_set():
            self.registry.cleanup_expired()
            time.sleep(self.interval)
        logger.info("KillTimer stopped.")

    def stop(self) -> None:
        """Stop the background thread."""
        self._stop_flag.set()
