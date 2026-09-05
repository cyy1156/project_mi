"""依赖方向白名单测试（重构实施方案 §1.2 的永久闸门）。

规则：
- core/          ：零项目依赖（任何其它 experiment_game.* 层禁止）
- offline/       ：禁止 import experiment / tools / pipeline / runtime
- acquisition/   ：禁止 import experiment / offline / tools / pipeline
- pipeline/      ：禁止 import experiment / tools（可调 core / offline）
- runtime/       ：禁止 import experiment / offline / tools / pipeline（仅 core）
- experiment/    ：禁止「模块顶层」import offline / tools / pipeline
                   （函数内惰性导入仅限 orchestrator 的 *_resolve_* 回退接缝）
- tools/ 与入口  ：无限制

tests/ 目录与 _archived 目录不参与扫描。
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
PIPELINE = "experiment_game.pipeline"
RUNTIME = "experiment_game.runtime"
TOOLS = "experiment_game.tools"
PROJECT_TOPS = (CORE, EXPERIMENT, ACQUISITION, OFFLINE, PIPELINE, RUNTIME, TOOLS)

ABSOLUTE_FORBIDDEN: dict[str, tuple[str, ...]] = {
    CORE: (EXPERIMENT, ACQUISITION, OFFLINE, PIPELINE, RUNTIME, TOOLS),
    OFFLINE: (EXPERIMENT, TOOLS, PIPELINE, RUNTIME),
    ACQUISITION: (EXPERIMENT, OFFLINE, TOOLS, PIPELINE, RUNTIME),
    # pipeline 可调 core/offline；sim/subject_registry 仅允许函数内惰性导入
    PIPELINE: (TOOLS, RUNTIME),
    RUNTIME: (EXPERIMENT, OFFLINE, TOOLS, PIPELINE, ACQUISITION),
}
TOP_LEVEL_FORBIDDEN: dict[str, tuple[str, ...]] = {
    EXPERIMENT: (OFFLINE, TOOLS, PIPELINE),
    PIPELINE: (EXPERIMENT,),
}


def _module_base(node: ast.AST) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name if node.names else ""
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    return ""


def _is_project_import(mod: str) -> bool:
    return any(mod == top or mod.startswith(top + ".") for top in PROJECT_TOPS)


def _top_level_import_nodes(tree: ast.Module) -> list[ast.AST]:
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
    for top in PROJECT_TOPS:
        top_parts = top.split(".")
        if tuple(parts[: len(top_parts)]) == tuple(top_parts):
            return ".".join(parts[: len(top_parts)])
    return "experiment_game"


def _matches_prefix(base: str, prefix: str) -> bool:
    return base == prefix or base.startswith(prefix + ".")


def _check_forbidden(
    module: str,
    nodes: list[ast.AST],
    forbidden: tuple[str, ...],
    path: Path,
    errors: list[str],
    *,
    kind: str,
) -> None:
    for node in nodes:
        base = _module_base(node)
        if not _is_project_import(base):
            continue
        for prefix in forbidden:
            if _matches_prefix(base, prefix):
                errors.append(
                    f"{path.relative_to(_REPO)}:{node.lineno} 层 {module} "
                    f"{kind}禁止 import {base}"
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
            continue
        if module in ABSOLUTE_FORBIDDEN:
            _check_forbidden(
                module,
                _all_import_nodes(tree),
                ABSOLUTE_FORBIDDEN[module],
                path,
                errors,
                kind="",
            )
        if module in TOP_LEVEL_FORBIDDEN:
            _check_forbidden(
                module,
                _top_level_import_nodes(tree),
                TOP_LEVEL_FORBIDDEN[module],
                path,
                errors,
                kind="顶层",
            )

    assert not errors, "依赖方向违规：\n" + "\n".join(errors)


def test_core_has_zero_project_deps() -> None:
    core_dir = _PKG / "core"
    other = (EXPERIMENT, ACQUISITION, OFFLINE, PIPELINE, RUNTIME, TOOLS)
    for path in core_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _all_import_nodes(tree):
            base = _module_base(node)
            bad = [
                top
                for top in other
                if base == top or base.startswith(top + ".")
            ]
            assert not bad, (
                f"{path}: core 层发现跨层 import {base}（core 必须零跨层依赖）"
            )
