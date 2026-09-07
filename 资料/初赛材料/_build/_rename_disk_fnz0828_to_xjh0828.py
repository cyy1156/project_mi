# -*- coding: utf-8 -*-
"""Rename disk identity fnz0828 -> xjh0828 (dirs, files, text inside subject tree)."""
from __future__ import annotations

import json
import os
from pathlib import Path

OLD = "fnz0828"
NEW = "xjh0828"

ROOTS = [
    Path(r"D:/MI/experiment_game/data/subjects"),
    Path(r"D:/MI/code/train_lab/out/5070_exp42_confound_collapse/A"),
]

TEXT_SUFFIX = {
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".csv",
    ".log",
    ".py",
    ".bat",
    ".ps1",
}


def rename_path(p: Path) -> Path:
    if OLD not in p.name:
        return p
    new = p.with_name(p.name.replace(OLD, NEW))
    if new.exists():
        raise FileExistsError(f"target exists: {new}")
    p.rename(new)
    print(f"REN {p} -> {new.name}")
    return new


def rewrite_text_file(p: Path) -> bool:
    try:
        raw = p.read_bytes()
    except Exception:
        return False
    changed = False
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        if OLD not in t:
            return False
        p.write_text(t.replace(OLD, NEW), encoding="utf-8")
        print(f"TXT {p}")
        return True
    return False


def walk_and_rename(root: Path) -> None:
    if not root.exists():
        print("skip missing", root)
        return
    # deepest first
    all_paths = sorted(root.rglob(f"*{OLD}*"), key=lambda p: len(p.parts), reverse=True)
    for p in all_paths:
        if not p.exists():
            continue
        rename_path(p)
    # also rename root itself if named OLD
    if root.name == OLD and root.exists():
        rename_path(root)


def rewrite_under(root: Path) -> None:
    if not root.exists():
        return
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in TEXT_SUFFIX or p.name.lower() in {"manifest.json", "run_config.json"}:
            rewrite_text_file(p)


def main() -> None:
    # 1) primary subject tree
    subj = Path(r"D:/MI/experiment_game/data/subjects") / OLD
    backup = Path(r"D:/MI/experiment_game/data/subjects/_backup_old_channel_order_20260829") / OLD
    analysis = Path(r"D:/MI/experiment_game/data/subjects/_analysis")
    exp42 = Path(r"D:/MI/code/train_lab/out/5070_exp42_confound_collapse/A") / OLD

    for d in (subj, backup, exp42):
        if d.exists():
            # rename nested session dirs first inside, then the folder
            nested = sorted(d.rglob(f"*{OLD}*"), key=lambda p: len(p.parts), reverse=True)
            for p in nested:
                if p.exists() and OLD in p.name:
                    rename_path(p)
            if d.exists() and d.name == OLD:
                rename_path(d)

    # analysis artifacts with old name in filename
    if analysis.exists():
        for p in sorted(analysis.rglob(f"*{OLD}*"), key=lambda p: len(p.parts), reverse=True):
            if p.exists() and OLD in p.name:
                rename_path(p)

    # 2) rewrite text content under new subject roots
    new_subj = Path(r"D:/MI/experiment_game/data/subjects") / NEW
    new_backup = Path(r"D:/MI/experiment_game/data/subjects/_backup_old_channel_order_20260829") / NEW
    new_exp42 = Path(r"D:/MI/code/train_lab/out/5070_exp42_confound_collapse/A") / NEW
    for d in (new_subj, new_backup, new_exp42, analysis):
        rewrite_under(d)

    print("DONE")
    print("exists old", (Path(r"D:/MI/experiment_game/data/subjects") / OLD).exists())
    print("exists new", new_subj.exists())


if __name__ == "__main__":
    main()
