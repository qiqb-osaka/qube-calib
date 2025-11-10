from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from .driverbase import Driver
from .orchestrator import ExecutionContext


class TaskStatus(Enum):
    """Enumeration of possible task statuses."""

    PENDING = auto()  # まだ実行されていない状態
    RUNNING = auto()  # 実行中の状態
    SUCCESS = auto()  # 成功した状態
    FAILED = auto()  # 失敗した状態
    SKIPPED = auto()  # スキップされた状態


@dataclass
class TaskResult:
    task_key: str
    status: TaskStatus
    timestamp: datetime
    payload: Any | None = None  # 実行結果やログなどを格納するためのフィールド
    meta: dict | None = None  # 追加のメタ情報を格納するためのフィールド

    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class ConcreteTaskBase:
    task_key: str
    driver_kind: str
    depends_on: list[ConcreteTaskBase] = field(default_factory=list)
    result: TaskResult | None = None
    name: str | None = None
    description: str | None = None
    _current_context: ExecutionContext | None = field(
        default=None, init=False, repr=False
    )

    def driver(self) -> Driver | None:
        """Get the driver associated with this task."""
        return (
            self._current_context.get_driver(self.driver_kind)
            if self._current_context
            else None
        )

    def run(
        self,
        *,
        context: ExecutionContext | None = None,
        dry_run: bool = False,
    ) -> TaskResult:
        start = datetime.now()
        self._current_context = context

        if not self.ready():
            return TaskResult(
                task_key=self.task_key,
                status=TaskStatus.SKIPPED,
                timestamp=start,
                meta={"reason": "dependency not satisfied"},
            )

        try:
            self.validate()

            if dry_run:
                payload = self.dry_run()
                status = TaskStatus.SKIPPED
            else:
                # put execute the driver from context
                payload = self.execute()
                status = TaskStatus.SUCCESS

            self.finalize()
            end = datetime.now()
            duration = (end - start).total_seconds()

            self.result = TaskResult(
                task_key=self.task_key,
                status=status,
                timestamp=end,
                payload=payload,
                meta={
                    "driver_kind": self.driver_kind,
                    "context": getattr(context, "session_key", None),
                    "duration": duration,
                },
            )
            return self.result

        except Exception as e:
            end = datetime.now()
            self.result = TaskResult(
                task_key=self.task_key,
                status=TaskStatus.FAILED,
                timestamp=end,
                payload=str(e),
                meta={
                    "error": repr(e),
                },
            )
            return self.result

    # Important APIs
    def validate(self) -> None:
        """Validate task parameters before execution. Override if needed."""
        pass

    def execute(self) -> Any:
        """Execute the main logic of the task. Should be overridden."""
        raise NotImplementedError

    def finalize(self) -> None:
        """Finalize task after execution. Override if needed."""
        pass

    def dry_run(self) -> str:
        return f"[DryRun] {self.task_key} executed."

    def ready(self) -> bool:
        """Check if the task dependencies are met and the task is ready to run."""
        return all(t.result and t.result.ok() for t in self.depends_on)


class TaskTemplate:
    pass
