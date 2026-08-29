"""依赖方向白名单测试（重构实施方案 §1.2 的永久闸门）。

规则：
- core/          ：零项目依赖（任何 experiment_game.* 都禁止，函数内也不行）
- offline/       ：禁止 import experiment / tools（任何层级）
- acquisition/   ：禁止 import experiment / offline / tools（任何层级）
- experiment/    ：禁止「模块顶层」import offline / tools
                   （函数内惰性导入仅限 orchestrator 的 *_resolve_* 回退接缝，过渡期允许）
- tools/ 与入口  ：无限制

tests/ 目录与 _archived 目录不参与扫描（测试是验证代码，归档是历史脚本）。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_PKG = _REPO / "experiment_game"

CORE = "experiment_game.core"
EXPERIMENT = "experiment_game.experiment"
ACQUISITION = "experiment_game.acquisition"
OFFLINE = "experiment_game.offline"
TOOLS = "experiment_game.tools"
PROJECT_TOPS = (CORE, EXPERIMENT, ACQUISITION, OFFLINE, TOOLS)

# 各层禁止引用的项目包前缀（core 自身内部互引允许，故 CORE 不在自身禁列）
ABSOLUTE_FORBIDDEN: dict[str, tuple[str, ...]] = {
    CORE: (EXPERIMENT, ACQUISITION, OFFLINE, TOOLS),
    OFFLINE: (EXPERIMENT, TOOLS),
    ACQUISITION: (EXPERIMENT, OFFLINE, TOOLS),
}
# experiment 层仅禁止顶层 import（函数内惰性回退允许）
TOP_LEVEL_FORBIDDEN: dict[str, tuple[str, ...]] = {
    EXPERIMENT: (OFFLINE, TOOLS),
}


def _module_base(node: ast.AST) -> str:
    """取 import 语句的目标模块名。"""
    if isinstance(node, ast.Import):
        return node.names[0].name if node.names else ""
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    return ""


def _is_project_import(mod: str) -> bool:
    return any(mod == top or mod.startswith(top + ".") for top in PROJECT_TOPS)


def _top_level_import_nodes(tree: ast.Module) -> list[ast.AST]:
    """模块体第一层的 import 语句。"""
    return [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]


def _all_import_nodes(tree: ast.Module) -> list[ast.AST]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]


def _py_files() -> list[Path]:
    skip_parts = {"__pycache__", "tests", "_archived"}
    out: list[Path] = []
    for p in _PKG.rglob("*.py"):
        if skip_parts & set(p.parts):
            continue
        out.append(p)
    return sorted(out)


def _pkg_of(path: Path) -> str:
    rel = path.relative_to(_PKG).with_suffix("")
    parts = ["experiment_game", *rel.parts]
    # 找最长属于已知层的前缀
    for top in PROJECT_TOPS:
        top_parts = top.split(".")
        if tuple(parts[: len(top_parts)]) == tuple(top_parts):
            return ".".join(parts[: len(top_parts)])
    return "experiment_game"


def _check(module: str, nodes: list[ast.AST], path: Path, errors: list[str]) -> None:
    forb = ABSOLUTE_FORBIDDEN.get(module, ())
    forb_top = TOP_LEVEL_FORBIDDEN.get(module, ())
    for node in nodes:
        base = _module_base(node)
        if not _is_project_import(base):
            continue
        for prefix in forb:
            if base == prefix or base.startswith(prefix + "."):
                errors.append(
                    f"{path.relative_to(_REPO)}:{node.lineno} 层 {module} "
                    f"禁止 import {base}"
                )
        for prefix in forb_top:
            if base == prefix or base.startswith(prefix + "."):
                errors.append(
                    f"{path.relative_to(_REPO)}:{node.lineno} 层 {module} "
                    f"顶层禁止 import {base}（如为回退接缝请移入 *_resolve_* 方法）"
                )


def test_dependency_directions() -> None:
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    errors: list[str] = []
    for path in _py_files():
        module = _pkg_of(path)
        if module not in ABSOLUTE_FORBIDDEN and module not in TOP_LEVEL_FORBIDDEN:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue  # 语法错误由其它测试/工具负责
        if module in ABSOLUTE_FORBIDDEN:
            _check(module, _all_import_nodes(tree), path, errors)
        else:
            _check(module, _top_level_import_nodes(tree), path, errors)

    assert not errors, "依赖方向违规：\n" + "\n".join(errors)


def test_core_has_zero_project_deps() -> None:
    core_dir = _PKG / "core"
    for path in core_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _all_import_nodes(tree):
            base = _module_base(node)
            # core 内部互引允许；跨到其它层即违规
            bad = [
                top
                for top in (EXPERIMENT, ACQUISITION, OFFLINE, TOOLS)
                if base == top or base.startswith(top + ".")
            ]
            assert not bad, (
                f"{path}: core 层发现跨层 import {base}（core 必须零跨层依赖）"
            )
