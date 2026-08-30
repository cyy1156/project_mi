"""FT 策略（默认 all4 + force 晋升 + 告警落盘）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = _REPO / "experiment_game" / "config" / "ft_policy.json"

_DEFAULTS: Dict[str, Any] = {
    "ft_scope": "all4",
    "force_promote_on_gate_fail": True,
    "auto_promote_after_ft": True,
}


def load_ft_policy(path: Path | None = None) -> Dict[str, Any]:
    """读取 FT 策略。

    文件不存在时用内置默认值；文件存在但损坏（JSON 解析失败 / 非 dict）时
    直接抛 ValueError（fail-fast）—— 默认值里 force_promote_on_gate_fail=True
    属于高风险行为，绝不允许在策略文件损坏时静默生效。
    """
    p = Path(path) if path else DEFAULT_POLICY_PATH
    out = dict(_DEFAULTS)
    if p.is_file():
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"FT 策略文件损坏（JSON 解析失败），拒绝回退默认值: {p} -> {exc}"
            ) from exc
        if not isinstance(blob, dict):
            raise ValueError(
                f"FT 策略文件必须是 JSON 对象: {p}（当前为 {type(blob).__name__}）"
            )
        out.update(blob)
    scope = str(out.get("ft_scope") or "all4").strip().lower()
    if scope not in ("all4", "so", "shallow"):
        scope = "all4"
    out["ft_scope"] = scope
    out["force_promote_on_gate_fail"] = bool(out.get("force_promote_on_gate_fail", True))
    out["auto_promote_after_ft"] = bool(out.get("auto_promote_after_ft", True))
    return out
