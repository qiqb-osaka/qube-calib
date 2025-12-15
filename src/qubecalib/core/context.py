from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class Driver:
    pass


@dataclass
class BackendEnv:
    pass


class TaskStatus(Enum):
    """Enumeration of possible task statuses."""

    PENDING = auto()  # まだ実行されていない状態
    RUNNING = auto()  # 実行中の状態
    SUCCESS = auto()  # 成功した状態
    FAILED = auto()  # 失敗した状態
    SKIPPED = auto()  # スキップされた状態


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    timestamp: datetime
    payload: Any | None = None  # 実行結果やログなどを格納するためのフィールド
    meta: dict | None = None  # 追加のメタ情報を格納するためのフィールド

    def ok(self) -> bool:
        return self.status == TaskStatus.SUCCESS


@dataclass
class ExecutionContext:
    """
    セッションごとの実行環境。
    - drivers: 物理デバイスとやりとりする Driver 群
    - backend_env: backend名 + namespaceキーで管理する補助オブジェクト
        例: "quel1.deskew", "quel1.binding"
    - task_results: TaskResult のレジストリ
    - data: feed-forward や sweep 用の汎用ストア
    """

    drivers: dict[str, Driver] = field(default_factory=dict)
    backend_env: dict[str, BackendEnv] = field(default_factory=dict)

    task_results: dict[str, TaskResult] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    # --- ヘルパー ---

    def get_driver(self, key: str) -> Any:
        try:
            return self.drivers[key]
        except KeyError:
            raise KeyError(f"Driver '{key}' not found in ExecutionContext.drivers")

    def get_backend_env(self, key: str) -> Any:
        try:
            return self.backend_env[key]
        except KeyError:
            raise KeyError(f"backend_env['{key}'] not found in ExecutionContext")

    def save_result(
        self, result: TaskResult, key: str | None = None, append: bool = False
    ) -> None:
        """TaskResult を保存。task_id と任意キー（scan用など）両方に保存できる。"""
        self.task_results[result.task_id] = result
        if key is not None:
            if append:
                lst = self.data.get(key)
                if lst is None:
                    self.data[key] = [result]
                else:
                    lst.append(result)
            else:
                self.data[key] = result

    def get_result(self, task_id: str) -> TaskResult | None:
        return self.task_results.get(task_id)

    def get_data(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


# @dataclass
# class DriverRegistry:
#     drivers: dict[str, Driver] = field(default_factory=dict)

#     def register(self, kind: str, driver: Driver) -> None:
#         self.drivers[kind] = driver

#     def get(self, kind: str) -> Driver:
#         if kind not in self.drivers:
#             raise KeyError(f"Driver '{kind}' not registered.")
#         return self.drivers[kind]
