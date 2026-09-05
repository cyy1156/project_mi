# -*- coding: utf-8 -*-
"""Sync working 原始/截图/xlsx into 交稿/ and write zip for email."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "交稿"
EXP37 = Path(r"D:/MI/code/train_lab/out/5070_challenge_exp37_nested_mcnemar_accpaper")
DATA = Path(r"D:/MI/DATA/挑战杯运动想象赛题数据文件")
ZIP = ROOT / "交稿_离线验证_XH-202610.zip"


def main() -> None:
    if PKG.exists():
        shutil.rmtree(PKG)
    raw = PKG / "原始验证数据"
    shot = PKG / "验证过程截图"
    oof = raw / "oof_N0"
    raw.mkdir(parents=True)
    shot.mkdir(parents=True)
    oof.mkdir(parents=True)

    shutil.copy2(ROOT / "离线性能验证报告_XH-202610.xlsx", PKG / "离线性能验证报告_XH-202610.xlsx")

    usage = ROOT / "数据集使用说明.md"
    if not usage.exists():
        # fallback: copy from previous pack template written beside this script once
        raise FileNotFoundError("缺少 数据集使用说明.md，请先写好再打包")
    shutil.copy2(usage, PKG / "数据集使用说明.md")

    src_raw = ROOT / "原始"
    for name in [
        "submission_QuadFold59.csv",
        "nested_N0_metrics.json",
        "数据说明_使用对照.md",
        "README.md",
        "官方数据说明.md",
        "sample_submission.csv",
    ]:
        p = src_raw / name
        if p.exists():
            shutil.copy2(p, raw / name)

    cm = ROOT / "原始数据_cm_e1f_arms.json"
    # 自采/OpenBMI 混淆矩阵不进入交稿包（本报告仅指定集）
    _ = cm  # keep file on disk for other tooling; do not ship in 交稿

    for n in ["oof_N0_prob.npy", "oof_N0_y.npy", "oof_N0_subjects.npy"]:
        src = EXP37 / "preds" / n
        if src.exists():
            shutil.copy2(src, oof / n)
            (src_raw / "oof_N0").mkdir(exist_ok=True)
            shutil.copy2(src, src_raw / "oof_N0" / n)

    for p in sorted((ROOT / "截图").glob("*.png")):
        shutil.copy2(p, shot / p.name)

    # official copies if missing in 原始
    if DATA.exists():
        for src_name, dst_name in [
            ("数据说明.md", "官方数据说明.md"),
            ("sample_submission.csv", "sample_submission.csv"),
        ]:
            s = DATA / src_name
            if s.exists() and not (raw / dst_name).exists():
                shutil.copy2(s, raw / dst_name)

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(PKG.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(Path("交稿") / p.relative_to(PKG)))

    print("packed", ZIP, "bytes", ZIP.stat().st_size)
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(PKG))


if __name__ == "__main__":
    main()
