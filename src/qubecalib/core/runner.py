from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger

from .context import ExecutionContext

logger = getLogger(__name__)


@dataclass
class ConcreteDAG:
    pass


# @dataclass
# class ExecutionContext:
#     session_key: str = ""  # Unique identifier for the execution session TODO: required?
#     driver_registry: DriverRegistry = field(default_factory=DriverRegistry)
#     cache: dict[str, Any] = field(default_factory=dict)
#     logdir: str | None = None

#     def get_driver(self, kind: str) -> Driver:
#         return self.driver_registry.get(kind)

#     @deprecated("Use 'get_driver' instead.")
#     def driver(self, kind: str) -> Driver:
#         return self.get_driver(kind)


class DagRunner:
    def run(self, dag: ConcreteDAG, context: ExecutionContext) -> None:
        raise NotImplementedError("DagRunner.run() is not implemented yet.")
