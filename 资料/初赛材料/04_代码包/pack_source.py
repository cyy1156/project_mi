# -*- coding: utf-8 -*-
"""
pack_source.py — XH-202610 源代码交稿目录 / 压缩包

按《README_代码包说明.md》§7 打包清单：
  包含：code/（不含 out/）、collect_data/、experiment_game/（不含 data/ 等）、
        requirements.txt、README、附录与复现指南
  排除：__pycache__/、.git/、.venv/、*.pyc、被试真实数据、中间产物

用法：
  python pack_source.py                 # 写入 交稿/（默认，不打 zip）
  python pack_source.py --dry-run       # 只统计
  python pack_source.py --zip           # 在已有/刚生成的 交稿/ 上打 zip
  python pack_source.py --materialize --zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

REPO = Path(r"D:\MI")
OUT_DIR = Path(r"D:\MI\资料\初赛材料\04_代码包")
JIAOGAO = OUT_DIR / "交稿"

# ---------- 排除规则 ----------
GLOBAL_DIR_EXCLUDE = {
    "__pycache__", ".git", ".venv", ".idea", ".pytest_cache",
    ".ipynb_checkpoints", "node_modules", "sim_subjects", "_analysis",
}
GLOBAL_FILE_SUFFIX = {".pyc", ".pyo", ".log", ".bak", ".tmp", ".swp"}
GLOBAL_FILE_NAME = {
    ".DS_Store", "Thumbs.db", "desktop.ini", ".gitignore",
    "项目计划.md", "代码审查报告.md", "架构重构分析报告.md",
}

CODE_DIR_EXCLUDE = {"out"}
EG_DIR_EXCLUDE = {
    "data", "gui-test-screenshots", "学习复现_从0到1",
}
EG_FILE_EXCLUDE = {
    "项目计划.md", "代码审查报告.md", "架构重构分析报告.md",
    "fnz0828_问题诊断报告.md",
}

ROOTS = [
    ("code", REPO / "code"),
    ("collect_data", REPO / "collect_data"),
    ("experiment_game", REPO / "experiment_game"),
]
TOP_FILES = [
    ("README.md", REPO / "README.md"),
    ("requirements.txt", REPO / "requirements.txt"),
    ("README_代码包说明.md", OUT_DIR / "README_代码包说明.md"),
    ("00_目录结构与复现总览.md", OUT_DIR / "00_目录结构与复现总览.md"),
    ("01_数据集获取说明.md", OUT_DIR / "01_数据集获取说明.md"),
    ("02_附录A_实验证据链索引.md", OUT_DIR / "附录A_实验证据链索引.md"),
    ("03_附录B_自采数据质控样例.md", OUT_DIR / "附录B_自采数据质控样例.md"),
    ("04_附录C_算法公式与算法详细介绍.md", OUT_DIR / "04_附录C_算法公式与算法详细介绍.md"),
    ("code/README_离线代码复现指南.md", OUT_DIR / "README_离线代码复现指南.md"),
    ("experiment_game/README_在线系统运行指南.md", OUT_DIR / "README_在线系统运行指南.md"),
    ("collect_data/README_采集软件说明.md", OUT_DIR / "README_采集软件说明.md"),
]


def dir_excluded(rel_dir: str) -> bool:
    parts = Path(rel_dir).parts
    for p in parts:
        if p in GLOBAL_DIR_EXCLUDE:
            return True
        if parts[0] == "code" and p in CODE_DIR_EXCLUDE:
            return True
        if parts[0] == "experiment_game" and p in EG_DIR_EXCLUDE:
            return True
    return False


def file_excluded(rel_file: str) -> bool:
    p = Path(rel_file)
    if p.suffix.lower() in GLOBAL_FILE_SUFFIX:
        return True
    if p.name in GLOBAL_FILE_NAME:
        return True
    if p.parts[0] == "experiment_game" and p.name in EG_FILE_EXCLUDE:
        return True
    return False


def collect() -> list[tuple[str, Path, int]]:
    """返回 (zip/交稿内相对路径, 绝对路径, 字节数) 列表"""
    items: list[tuple[str, Path, int]] = []
    for arc_root, src_root in ROOTS:
        if not src_root.is_dir():
            print(f"[警告] 缺少目录：{src_root}")
            continue
        for f in sorted(src_root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(src_root).as_posix()
            arc = f"{arc_root}/{rel}"
            parts = Path(arc).parts
            skip = False
            for i in range(1, len(parts)):
                if dir_excluded("/".join(parts[:i])):
                    skip = True
                    break
            if skip or file_excluded(arc):
                continue
            items.append((arc, f, f.stat().st_size))
    for arc, src in TOP_FILES:
        if src.is_file():
            items.append((arc, src, src.stat().st_size))
        else:
            print(f"[警告] 缺少文件：{src}")
    return items


def summarize(items: list[tuple[str, Path, int]]) -> None:
    total = sum(s for _, _, s in items)
    by_top: dict[str, list[int]] = {}
    for arc, _, s in items:
        top = arc.split("/", 1)[0]
        by_top.setdefault(top, [0, 0])
        by_top[top][0] += 1
        by_top[top][1] += s
    print(f"共 {len(items)} 个文件，原始体积 {total / 1024 / 1024:.1f} MB")
    for top, (n, s) in sorted(by_top.items()):
        print(f"  {top:<28} {n:>5} 文件  {s / 1024 / 1024:>8.1f} MB")


def materialize(items: list[tuple[str, Path, int]], dest: Path) -> None:
    if dest.exists():
        print(f"清空旧目录：{dest}")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for arc, src, _ in items:
        out = dest / arc
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    # 交稿目录说明（不入 zip 清单的工作索引，评委也可看）
    readme = dest / "交稿说明.md"
    readme.write_text(
        "# 补充材料 · 交稿目录（XH-202610）\n\n"
        "> 本目录即邮件附件「源代码压缩包」的解压形态；打 zip 时压缩本目录全部内容即可。\n\n"
        "## 内容\n\n"
        "- `code/` · `collect_data/` · `experiment_game/`：可复现源码（不含 DATA、被试落盘、训练 out）\n"
        "- `requirements.txt` · `README.md` · `README_代码包说明.md`：语言 / 环境 / 使用说明\n"
        "- `00`–`04` 与模块内 README：数据集获取、证据链、质控样例、公式详述、分层指南\n\n"
        "## 不含\n\n"
        "- 公开数据集 `DATA/`、真实被试 `experiment_game/data/`、模型权重与 `code/**/out/`\n"
        "- 团队内部排期文件（`补充材料提交计划_*.md`）\n\n"
        f"生成日期：{date.today().isoformat()} · 由 `pack_source.py` 物化。\n",
        encoding="utf-8",
    )
    print(f"已写入交稿目录：{dest}")


def write_zip(src_tree: Path, zip_path: Path) -> None:
    if not src_tree.is_dir():
        raise SystemExit(f"交稿目录不存在，请先 materialize：{src_tree}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(src_tree.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src_tree).as_posix())
    print(f"已生成：{zip_path}  ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="XH-202610 源代码交稿物化 / 打包")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写盘")
    ap.add_argument(
        "--no-materialize",
        action="store_true",
        help="跳过写入交稿/（仅配合 --zip 时需已有交稿）",
    )
    ap.add_argument("--zip", action="store_true", help="从交稿/ 打 zip（默认不打）")
    ap.add_argument(
        "--zip-name",
        default="",
        help="zip 文件名（默认 源代码_XH-202610_YYYYMMDD.zip）",
    )
    args = ap.parse_args()

    items = collect()
    summarize(items)
    if args.dry_run:
        print("(dry-run，未写盘)")
        return

    if not args.no_materialize:
        materialize(items, JIAOGAO)

    if args.zip:
        tag = date.today().strftime("%Y%m%d")
        name = args.zip_name or f"源代码_XH-202610_{tag}.zip"
        write_zip(JIAOGAO, OUT_DIR / name)


if __name__ == "__main__":
    main()
