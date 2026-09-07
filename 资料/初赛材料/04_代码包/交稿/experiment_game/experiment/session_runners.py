"""v2/v3/v4 会话薄封装类（总册 W6：函数入口迁入 SessionRunnerBase 子类）。

完整状态机仍在 session_v*.py；本模块提供统一类面，供编排层/测试按基类调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_game.experiment.session_base import SessionRunnerBase, SessionServices


class V2SessionRunner(SessionRunnerBase):
    """包装 ``session_v2.run_v2_session``。"""

    def __init__(self, services: SessionServices, **kwargs: Any) -> None:
        super().__init__(services)
        self.kwargs = kwargs
        self.summary: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        from experiment_game.experiment.session_v2 import run_v2_session

        self.summary = run_v2_session(
            self.events,
            self.markers,
            self.bridge,
            on_console=self.services.on_console,
            **self.kwargs,
        )
        return self.summary


class V3SessionRunner(SessionRunnerBase):
    """包装 ``session_v3.run_v3_session``。"""

    def __init__(self, services: SessionServices, **kwargs: Any) -> None:
        super().__init__(services)
        self.kwargs = kwargs
        self.summary: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        from experiment_game.experiment.session_v3 import run_v3_session

        self.summary = run_v3_session(
            self.events,
            self.markers,
            self.bridge,
            on_console=self.services.on_console,
            **self.kwargs,
        )
        return self.summary


class V4SessionRunner(SessionRunnerBase):
    """包装 ``session_v4.run_v4_session``。"""

    def __init__(self, services: SessionServices, buf: Any, **kwargs: Any) -> None:
        super().__init__(services)
        self.buf = buf
        self.kwargs = kwargs
        self.summary: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        from experiment_game.experiment.session_v4 import run_v4_session

        self.summary = run_v4_session(
            self.events,
            self.markers,
            self.bridge,
            self.buf,
            on_console=self.services.on_console,
            **self.kwargs,
        )
        return self.summary
