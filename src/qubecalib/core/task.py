from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .context import Driver, TaskResult, TaskStatus
from .runner import ExecutionContext

# class TaskStatus(Enum):
#     """Enumeration of possible task statuses."""

#     PENDING = auto()  # まだ実行されていない状態
#     RUNNING = auto()  # 実行中の状態
#     SUCCESS = auto()  # 成功した状態
#     FAILED = auto()  # 失敗した状態
#     SKIPPED = auto()  # スキップされた状態


# @dataclass
# class TaskResult:
#     task_key: str
#     status: TaskStatus
#     timestamp: datetime
#     payload: Any | None = None  # 実行結果やログなどを格納するためのフィールド
#     meta: dict | None = None  # 追加のメタ情報を格納するためのフィールド

#     def ok(self) -> bool:
#         return self.status == "ok"


@dataclass
class ConcreteTaskBase:
    task_id: str
    device_key: str
    depends_on: list[ConcreteTaskBase] = field(default_factory=list)
    result: TaskResult | None = None
    # name: str | None = None
    # description: str | None = None

    # 実行中のみセットされる ExecutionContext (driver() で使用)
    _current_context: ExecutionContext | None = field(
        default=None, init=False, repr=False
    )

    # ---- Helpers ----
    def driver(self) -> Driver | None:
        if self._current_context is None:
            raise RuntimeError(
                "ExecutionContext is not set for this task (run() with context=...)."
            )
        return self._current_context.get_driver(self.device_key)

    def ready(self) -> bool:
        """Check if the task dependencies are met and the task is ready to run."""
        return all(t.result is not None and t.result.ok() for t in self.depends_on)

    # ---- Lifecycle Methods ----
    def validate(self) -> None:
        """Validate task parameters before execution. Override if needed."""
        pass

    def dry_run(self) -> str:
        """dry_run のときに返す情報（ログ文字列など）。"""
        return f"[DryRun] {self.task_id} executed."

    def execute(self) -> Any:
        """Execute the main logic of the task. Should be overridden."""
        raise NotImplementedError

    # def finalize(self) -> None:
    #     """Finalize task after execution. Override if needed."""
    #     pass

    def run(
        self,
        *,
        context: ExecutionContext | None = None,
        dry_run: bool = False,
        store_in_context: bool = True,
    ) -> TaskResult:
        """Task の実行エントリポイント。Orchestrator から呼ばれる想定。"""
        start = datetime.now()
        self._current_context = context

        if not self.ready():
            res = TaskResult(
                task_id=self.task_id,
                status=TaskStatus.SKIPPED,
                timestamp=start,
                meta={"reason": "dependency not satisfied"},
            )
            self.result = res
            if context and store_in_context:
                context.save_result(res)
            return res

        try:
            self.validate()

            if dry_run:
                payload = self.dry_run()
                status = TaskStatus.SKIPPED
            else:
                # put execute the driver from context
                payload = self.execute()
                status = TaskStatus.SUCCESS

        except Exception as e:
            end = datetime.now()
            res = TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                timestamp=end,
                payload=str(e),
                meta={
                    "error": repr(e),
                },
            )
            self.result = res
            if context and store_in_context:
                context.save_result(res)
            return res

        # self.finalize()
        end = datetime.now()
        duration = (end - start).total_seconds()

        res = TaskResult(
            task_id=self.task_id,
            status=status,
            timestamp=end,
            payload=payload,
            meta={
                "device_key": self.device_key,
                # "context": getattr(context, "session_key", None),
                "duration": duration,
            },
        )
        self.result = res
        if context and store_in_context:
            context.save_result(res)
        return res


@dataclass
class TemplateTaskBase:
    pass
