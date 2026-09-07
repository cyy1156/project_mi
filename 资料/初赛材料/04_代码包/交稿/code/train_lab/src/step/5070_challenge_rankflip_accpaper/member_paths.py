# -*- coding: utf-8 -*-
"""定位 Exp34 A59 / B8 成员 three 目录。"""

from __future__ import annotations

from pathlib import Path

from exp35_config import EXP34_A59_OUT, EXP34_B8_OUT, MEMBER_KEYS, train_lab_out


def _prefer_run(runs: list[Path], prefer_tag: str = "full_20260902_1930") -> Path | None:
    if not runs:
        return None
    for r in runs:
        if prefer_tag in r.name:
            return r
    nonsmoke = [r for r in runs if "smoke" not in r.name.lower()]
    pool = nonsmoke or runs
    return sorted(pool, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def find_a59_member_three(
    member: str,
    *,
    prefer_tag: str = "full_20260902_1930",
) -> Path | None:
    root = (
        train_lab_out()
        / EXP34_A59_OUT
        / f"{member}_challenge_mi_3s_59ch"
        / "challenge_mi_3s_59ch"
    )
    if not root.is_dir():
        return None
    best = _prefer_run(list(root.glob("run_*")), prefer_tag)
    if best is None:
        return None
    three = best / "three"
    return three if three.is_dir() else None


def find_all_a59_members(**kw) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for m in MEMBER_KEYS:
        p = find_a59_member_three(m, **kw)
        if p is not None:
            out[m] = p
    return out


def find_b8_member_three(
    member: str,
    *,
    arm: str,
    prefer_tag: str = "full_20260902_1930",
) -> Path | None:
    suffix = "ft" if arm == "ft" else "scratch"
    root = (
        train_lab_out()
        / EXP34_B8_OUT
        / f"{member}_challenge_mi_3s_8ch_{suffix}"
        / "challenge_mi_3s_8ch"
    )
    if not root.is_dir():
        return None
    best = _prefer_run(list(root.glob("run_*")), prefer_tag)
    if best is None:
        return None
    three = best / "three"
    return three if three.is_dir() else None


def find_all_b8_members(arm: str, **kw) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for m in MEMBER_KEYS:
        p = find_b8_member_three(m, arm=arm, **kw)
        if p is not None:
            out[m] = p
    return out


def fold_has_val(three_dir: Path, fold: int) -> bool:
    fd = three_dir / f"fold{fold}"
    return (fd / "val_prob.npy").is_file() and (fd / "val_y.npy").is_file()


def n_folds_available(member_dirs: dict[str, Path], max_folds: int = 6) -> int:
    n = 0
    for f in range(max_folds):
        if all(fold_has_val(d, f) for d in member_dirs.values()):
            n = f + 1
        else:
            break
    return n
