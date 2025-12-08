# tests/test_concrete_task.py
from datetime import datetime
from typing import Collection, cast

from qubecalib.core.context import Driver, TaskResult, TaskStatus
from qubecalib.core.runner import ExecutionContext
from qubecalib.core.task import ConcreteTaskBase


# ----------------------------------------
# Mock ExecutionContext
# ----------------------------------------
class MockContext(ExecutionContext):
    def __init__(self) -> None:
        self.saved: list[TaskResult] = []
        self.drivers: dict[str, Driver] = {}

    def get_driver(self, key: str) -> Driver | None:
        return self.drivers.get(key, None)

    def save_result(
        self, result: TaskResult, key: str | None = None, append: bool = False
    ) -> None:
        self.saved.append(result)


# ----------------------------------------
# FakeTask for testing
# ----------------------------------------
class FakeTask(ConcreteTaskBase):
    def __init__(
        self,
        task_id: str = "t0",
        device_key: str = "dev",
        depends_on: list[ConcreteTaskBase] | None = None,
        behavior: str = "normal",
    ) -> None:
        super().__init__(
            task_id=task_id, device_key=device_key, depends_on=depends_on or []
        )
        self.behavior = behavior

    def execute(self) -> dict[str, int | str]:
        if self.behavior == "normal":
            return {"value": 123}
        elif self.behavior == "error":
            raise RuntimeError("execute error")
        else:
            return {"value": "unknown"}


# ----------------------------------------
# 1. ready() が依存タスク未完了なら False
# ----------------------------------------
def test_ready_false_when_dependency_not_done() -> None:
    dep = FakeTask("dep", "dev")
    t = FakeTask(depends_on=[dep])

    assert t.ready() is False


# ----------------------------------------
# 2. 依存タスクが成功済みなら ready=True
# ----------------------------------------
def test_ready_true_when_dependency_ok() -> None:
    dep = FakeTask("dep", "dev")
    dep.result = TaskResult(
        task_id="dep",
        status=TaskStatus.SUCCESS,
        timestamp=datetime.now(),
    )
    t = FakeTask(depends_on=[dep])

    assert t.ready() is True


# ----------------------------------------
# 3. 依存不満足 → SKIPPED
# ----------------------------------------
def test_run_skipped_due_to_dependency() -> None:
    dep = FakeTask("dep")
    t = FakeTask(task_id="t1", depends_on=[dep])

    ctx = MockContext()
    result = t.run(context=ctx)

    assert result.status == TaskStatus.SKIPPED
    assert cast(dict, result.meta)["reason"] == "dependency not satisfied"
    assert ctx.saved[0] is result


# ----------------------------------------
# 4. dry_run=True → SKIPPED & dry_run の内容
# ----------------------------------------
def test_run_dry_run() -> None:
    t = FakeTask(task_id="t2")
    # mark dependency success
    t.depends_on = []
    ctx = MockContext()

    result = t.run(context=ctx, dry_run=True)

    assert result.status == TaskStatus.SKIPPED
    assert "[DryRun] t2" in cast(Collection, result.payload)
    assert ctx.saved[0] is result


# ----------------------------------------
# 5. 実行成功 → SUCCESS
# ----------------------------------------
def test_run_success() -> None:
    t = FakeTask(task_id="t3", behavior="normal")
    ctx = MockContext()

    result = t.run(context=ctx)

    assert result.status == TaskStatus.SUCCESS
    assert result.payload == {"value": 123}
    assert cast(dict, result.meta)["device_key"] == "dev"
    assert ctx.saved[0] is result


# ----------------------------------------
# 6. execute() が例外を投げる → FAILED
# ----------------------------------------
def test_run_failed() -> None:
    t = FakeTask(task_id="t4", behavior="error")
    ctx = MockContext()

    result = t.run(context=ctx)

    assert result.status == TaskStatus.FAILED
    assert "execute error" in cast(Collection, result.payload)
    assert "execute error" in cast(dict, result.meta)["error"]
    assert ctx.saved[0] is result


# ----------------------------------------
# 7. store_in_context=False の場合 save されない
# ----------------------------------------
def test_run_no_store_in_context() -> None:
    t = FakeTask(task_id="t5")
    ctx = MockContext()

    result = t.run(context=ctx, store_in_context=False)

    assert result.status == TaskStatus.SUCCESS
    assert ctx.saved == []
