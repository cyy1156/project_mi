# -*- coding: utf-8 -*-
"""
pack_source.py — XH-202610 源代码压缩包打包脚本
按《README_代码包说明.md》§7 打包清单执行：
  包含：code/（不含 out/）、collect_data/、experiment_game/（不含 data/ 等内部产物）、
        requirements.txt、README.md、README_代码包说明.md、附录A/B
  排除：__pycache__/、.git/、.venv/、*.pyc、被试真实数据、中间产物
用法：
  python pack_source.py            # 打包
  python pack_source.py --dry-run  # 只统计，不写 zip
"""
import sys
import zipfile
from pathlib import Path

REPO = Path(r"D:\MI")
OUT_DIR = Path(r"D:\MI\资料\初赛材料\04_代码包")
DATE_TAG = "20260905"
ZIP_PATH = OUT_DIR / f"源代码_XH-202610_{DATE_TAG}.zip"

# ---------- 排除规则 ----------
GLOBAL_DIR_EXCLUDE = {
    "__pycache__", ".git", ".venv", ".idea", ".pytest_cache",
    ".ipynb_checkpoints", "node_modules", "sim_subjects", "_analysis",
}
GLOBAL_FILE_SUFFIX = {".pyc", ".pyo", ".log", ".bak", ".tmp", ".swp"}
GLOBAL_FILE_NAME = {".DS_Store", "Thumbs.db", "desktop.ini", ".gitignore",
                    "项目计划.md", "代码审查报告.md", "架构重构分析报告.md"}

# 相对仓库根的目录黑名单（按目录名匹配，任意层级生效，但仅限列出的根内）
CODE_DIR_EXCLUDE = {"out"}                       # 训练/预处理缓存（共 ~101G）
EG_DIR_EXCLUDE = {                               # experiment_game 内部产物
    "data", "gui-test-screenshots", "学习复现_从0到1",
}
EG_FILE_EXCLUDE = {                              # 内部过程报告，不随作品提交
    "项目计划.md", "代码审查报告.md", "架构重构分析报告.md",
    "fnz0828_问题诊断报告.md",
}

# 打包根：目标 zip 的顶层目录/文件
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
    # 模块级运行指南：注入到对应模块目录内
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


def collect() -> list:
    """返回 (zip内相对路径, 绝对路径, 字节数) 列表"""
    items = []
    for arc_root, src_root in ROOTS:
        if not src_root.is_dir():
            print(f"[警告] 缺少目录：{src_root}")
            continue
        for f in sorted(src_root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(src_root).as_posix()
            arc = f"{arc_root}/{rel}"
            # 逐级判断是否有目录被排除
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


def main():
    dry = "--dry-run" in sys.argv
    items = collect()
    total = sum(s for _, _, s in items)
    by_top = {}
    for arc, _, s in items:
        top = arc.split("/", 1)[0]
        by_top[top] = by_top.get(top, [0, 0])
        by_top[top][0] += 1
        by_top[top][1] += s
    print(f"共 {len(items)} 个文件，原始体积 {total/1024/1024:.1f} MB")
    for top, (n, s) in sorted(by_top.items()):
        print(f"  {top:<20} {n:>5} 文件  {s/1024/1024:>8.1f} MB")
    if dry:
        print("(dry-run，未写 zip)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for arc, src, _ in items:
            zf.write(src, arc)
    print(f"已生成：{ZIP_PATH}  ({ZIP_PATH.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
