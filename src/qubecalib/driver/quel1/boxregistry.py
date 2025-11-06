from __future__ import annotations

import time
from logging import getLogger
from threading import Lock
from typing import cast

from quel_ic_config import Quel1Box

logger = getLogger(__name__)


class BoxHandle:
    """Explicit lifetime-managed Box handle with strong reference."""

    def __init__(self, name: str, box: Quel1Box, deadline: float) -> None:
        self.name = name
        self._ref: Quel1Box | None = box
        self.deadline = deadline

    def get(self) -> Quel1Box | None:
        """Return the Box if still valid."""
        return self._ref

    def is_alive(self) -> bool:
        """Return True if the Box reference is still held."""
        return self._ref is not None

    def refresh(self, new_deadline: float) -> None:
        """Push the expiration deadline forward."""
        self.deadline = new_deadline
        logger.debug(f"Box {self.name} refreshed (new deadline {new_deadline}).")

    def release(self) -> None:
        """Drop the reference to allow GC to release the Box."""
        if self._ref is not None:
            self._ref = None
            logger.debug(f"Box {self.name} released (ref cleared).")


class BoxRegistry:
    """Registry that holds strong references to Boxes and manages their lifetime."""

    def __init__(self, default_deadline: float | None = 3600) -> None:
        self._handles: dict[str, BoxHandle] = {}
        self._lock = Lock()
        self._default_deadline = default_deadline

    def register(
        self, name: str, box: Quel1Box, deadline: float | None = None
    ) -> BoxHandle:
        """Register a Box under lifetime management."""
        with self._lock:
            resolved_deadline = self._resolve_deadline(deadline)
            handle = BoxHandle(name, box, resolved_deadline)
            self._handles[name] = handle
        logger.debug(f"Box {name} registered (deadline {resolved_deadline} sec).")
        return handle

    def _resolve_deadline(self, deadline: float | None) -> float:
        if deadline is not None:
            return deadline
        if self._default_deadline is not None:
            return time.time() + self._default_deadline
        return float("inf")

    def touch(self, name: str, deadline_extension: float | None = None) -> bool:
        """Refresh the expiration deadline of a Box."""
        with self._lock:
            handle = self._handles.get(name)
            if not handle or not handle.is_alive():
                return False
            new_deadline = (
                time.time() + deadline_extension
                if deadline_extension is not None
                else self._resolve_deadline(None)
            )
            handle.refresh(new_deadline)
            return True

    def get(self, name: str) -> Quel1Box:
        """Return the Box instance if it is still alive."""
        with self._lock:
            handle = self._handles.get(name)
            alive = handle and handle.is_alive()
            if not alive:
                raise KeyError(f"Box '{name}' not found or expired.")
        return cast(Quel1Box, cast(BoxHandle, handle).get())

    def release(self, name: str) -> None:
        """Release a specific Box by name."""
        with self._lock:
            handle = self._handles.pop(name, None)
        if handle:
            handle.release()
            logger.debug(f"Box {name} removed from registry.")

    def cleanup_expired(self) -> None:
        """Release all Boxes whose lifetime has expired."""
        now = time.time()
        with self._lock:
            expired = [n for n, h in self._handles.items() if h.deadline < now]
        for name in expired:
            logger.debug(f"Releasing expired Box: {name}")
            self.release(name)

    def list_active(self) -> list[str]:
        """Return the list of all currently alive Box names."""
        with self._lock:
            return [n for n, h in self._handles.items() if h.is_alive()]
