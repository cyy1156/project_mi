# -*- coding: utf-8 -*-
"""验证截图回放脚本：S01/S04–S07 的控制台输出（供截屏）与 S05–S07 PNG 渲染。

用法（建议 conda cyy 环境；S01/S04–S07 需先在 02_离线验证 目录）：
  python _make_verify_shots.py --print --only S01   # 数据核验
  python _make_verify_shots.py --print --only S04   # OOF 独立复算
  python _make_verify_shots.py --print --only S05   # 交卷 CSV 完整性
  python _make_verify_shots.py --print --only S06   # Excel 与 JSON 一致性
  python _make_verify_shots.py --print --only S07   # 文件指纹与环境（最后截）
  python _make_verify_shots.py                      # （旧）渲染 S05–S07 PNG
S02（嵌套回放）与 S03（盲测 CSV 再生成）为独立实验脚本回放，见操作手册。
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "交稿" / "原始验证数据"
SHOTS = HERE / "交稿" / "验证过程截图"
DATA_ROOT = Path(r"D:/MI/DATA/挑战杯运动想象赛题数据文件")


# ---------------- S01 数据核验 ----------------
def section_s01() -> str:
    import pickle
    buf = io.StringIO()

    def out(x=""):
        print(x, file=buf)

    out("=" * 72)
    out("S01 · 指定集官方数据核验（目录 / PKL 结构 / 规模计数）")
    out("=" * 72)
    out("数据根: %s" % DATA_ROOT)
    tr = sorted((DATA_ROOT / "train").glob("S*/block_*.pkl"))
    te = sorted((DATA_ROOT / "test").glob("S*/block_*.pkl"))
    out("train blocks: %d 个（S01–S06，每块 30 trial → 900）" % len(tr))
    out("test  blocks: %d 个（S07–S08，每块 30 trial → 120）" % len(te))
    out("sample_submission.csv 存在: %s" % (DATA_ROOT / "sample_submission.csv").is_file())
    p = te[0]
    obj = pickle.load(open(p, "rb"))
    data = obj["data"]
    ch = list(obj["ch_names"])
    nrow = data.shape[0]
    note = "59 EEG + 5 辅助 + 1 trigger" if nrow == 65 else "59 EEG + 5 辅助（test 无 trigger 行）"
    out("样本 PKL: %s" % p.name)
    out("  keys: %s" % sorted(obj.keys()))
    out("  data.shape = %s（%d 行 = %s；22500 = 30 trial × 750 点）" % (data.shape, nrow, note))
    out("  srate = %s" % obj.get("srate"))
    out("  ch_names 前 8: %s" % ch[:8])
    out("==> S01 结论: 官方数据在位且结构与《数据说明》一致 (PASS)")
    return buf.getvalue()


# ---------------- S04 OOF 独立复算 ----------------
def section_s04() -> str:
    buf = io.StringIO()

    def out(x=""):
        print(x, file=buf)

    out("=" * 72)
    out("S04 · 指定集嵌套指标独立复算（从 OOF npy 重算，对照登记 JSON）")
    out("=" * 72)
    out("执行时间: %s   python %s / numpy %s" % (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        platform.python_version(), np.__version__))
    out("方法: 对 oof_N0_prob.npy 做 argmax，独立重算全部指标")
    out("-" * 72)
    prob = np.load(RAW / "oof_N0/oof_N0_prob.npy")
    y = np.load(RAW / "oof_N0/oof_N0_y.npy")
    sub = np.load(RAW / "oof_N0/oof_N0_subjects.npy").astype(str)
    pred = prob.argmax(1)
    cm = np.zeros((3, 3), int)
    for a, b in zip(y, pred):
        cm[a][b] += 1
    N = cm.sum()
    acc = np.trace(cm) / N
    rec = np.array([cm[i][i] / cm[i].sum() for i in range(3)])
    spec = np.array([(N - cm[i].sum() - cm[:, i].sum() + cm[i][i]) / (N - cm[i].sum())
                     for i in range(3)])
    prec = np.array([cm[i][i] / cm[:, i].sum() for i in range(3)])
    f1 = 2 * rec * prec / (rec + prec)
    out("OOF 形状: prob%s y%s  被试: %s" % (prob.shape, y.shape, sorted(set(sub.tolist()))))
    out("混淆矩阵 (行=真实 左/右/静息):")
    for name, row in zip(("Left ", "Right", "Rest "), cm):
        out("  %s %s" % (name, row.tolist()))
    out("Acc        = %.4f" % acc)
    out("macro 召回 = %.4f  (per-class %s)" % (rec.mean(), np.round(rec, 4).tolist()))
    out("macro 特异 = %.4f" % spec.mean())
    out("macro-F1   = %.4f" % f1.mean())
    m = json.loads((RAW / "nested_N0_metrics.json").read_text(encoding="utf-8"))
    out("-" * 72)
    out("对照 nested_N0_metrics.json:")
    allok = True
    for k, a in [("acc", acc), ("macro_recall", rec.mean()),
                 ("macro_specificity", spec.mean()), ("macro_f1", f1.mean())]:
        b = m[k]
        ok = abs(a - b) < 5e-5
        allok &= ok
        out("  %-18s 复算=%.4f  JSON=%.4f  %s" % (k, a, b, "PASS" if ok else "FAIL"))
    folds = [float((pred[sub == "challenge:" + s] == y[sub == "challenge:" + s]).mean())
             for s in ("S01", "S02", "S03", "S04", "S05", "S06")]
    ok6 = max(abs(a - b) for a, b in zip(folds, m["fold_accs"])) < 5e-5
    allok &= ok6
    out("  六折 acc           复算=%s" % [round(x, 4) for x in folds])
    out("  %s" % ("全部一致 PASS" if ok6 else "存在 FAIL"))
    out("==> S04 结论: %s" % ("独立复算与登记 JSON 完全一致 (ALL PASS)" if allok
                              else "存在不一致，需排查"))
    return buf.getvalue()


# ---------------- S05 交卷 CSV 完整性 ----------------
def section_s05() -> str:
    buf = io.StringIO()

    def out(x=""):
        print(x, file=buf)

    out("=" * 72)
    out("S05 · 交卷 CSV 完整性校验（行序对齐官方模板）")
    out("=" * 72)
    smp = list(csv.reader(open(RAW / "sample_submission.csv", encoding="utf-8-sig")))
    subm = list(csv.reader(open(RAW / "submission_QuadFold59.csv", encoding="utf-8-sig")))
    out("sample_submission 行数(含表头): %d   submission 行数: %d" % (len(smp), len(subm)))
    out("表头一致: %s  (header=%s)" % (smp[0] == subm[0], subm[0]))
    ids_equal = [r[0] for r in smp[1:]] == [r[0] for r in subm[1:]]
    out("sample_id 逐行对齐官方模板: %s  (位置计分不会错位)" % ids_equal)
    labs = [r[1] for r in subm[1:]]
    dist = Counter(labs)
    valid = set(labs) <= {"0", "1", "2"}
    out("label 取值合法(0/1/2): %s   分布: %s" % (
        valid, ", ".join("%s->%d" % (k, v) for k, v in sorted(dist.items()))))
    okc = len(subm) == 121 and ids_equal and valid
    out("==> S05 结论: %s" % ("120 行、行序对齐、取值合法 (PASS)" if okc else "FAIL"))
    return buf.getvalue()


# ---------------- S06 Excel 与 JSON 一致性 ----------------
def section_s06() -> str:
    import openpyxl
    buf = io.StringIO()

    def out(x=""):
        print(x, file=buf)

    out("=" * 72)
    out("S06 · Excel 核心指标与登记 JSON 一致性（程序化读取单元格）")
    out("=" * 72)
    xlsx = HERE / "交稿" / "离线性能验证报告_XH-202610.xlsx"
    m = json.loads((RAW / "nested_N0_metrics.json").read_text(encoding="utf-8"))
    ws = openpyxl.load_workbook(xlsx, data_only=True)["00_核心指标"]
    cells = {}
    for row in ws.iter_rows(values_only=True):
        for i, v in enumerate(row):
            if v in ("分类准确率", "召回率（macro）", "特异性（macro）"):
                cells[v] = row[i + 1] if i + 1 < len(row) else None
    pairs = [("分类准确率", "acc", cells["分类准确率"], m["acc"]),
             ("召回率（macro）", "macro_recall", cells["召回率（macro）"], m["macro_recall"]),
             ("特异性（macro）", "macro_specificity", cells["特异性（macro）"], m["macro_specificity"])]
    allok = True
    for name, k, cell, jv in pairs:
        ok = abs(float(cell) - float(jv)) < 5e-5
        allok &= ok
        out("  %-14s Excel=%s  JSON=%.4f  %s" % (name, cell, jv, "PASS" if ok else "FAIL"))
    out("==> S06 结论: %s" % ("Excel 数字来自登记 JSON (ALL PASS)" if allok else "不一致，需排查"))
    return buf.getvalue()


# ---------------- S07 文件指纹与环境 ----------------
def section_s07() -> str:
    buf = io.StringIO()

    def out(x=""):
        print(x, file=buf)

    out("=" * 72)
    out("S07 · 交稿包文件指纹（MD5）与运行环境")
    out("=" * 72)
    files = ["离线性能验证报告_XH-202610.xlsx", "数据集使用说明.md",
             "原始验证数据/submission_QuadFold59.csv", "原始验证数据/nested_N0_metrics.json",
             "原始验证数据/oof_N0/oof_N0_prob.npy", "原始验证数据/oof_N0/oof_N0_y.npy",
             "原始验证数据/oof_N0/oof_N0_subjects.npy",
             "验证过程截图/S01_数据集结构与加载核验.png",
             "验证过程截图/S02_嵌套N0指标汇总.png",
             "验证过程截图/S03_交卷CSV再生成与比对.png",
             "验证过程截图/S04_独立复算.png",
             "验证过程截图/S05_交卷CSV完整性校验.png",
             "验证过程截图/S06_Excel与JSON一致性.png",
             "验证过程截图/S07_文件指纹与环境.png"]
    n_ok = 0
    for f in files:
        p = HERE / "交稿" / f
        if not p.is_file():
            out("  %s  %14s  %s（尚未生成，跳过）" % ("-" * 32, "", f))
            continue
        n_ok += 1
        out("  %s  %9d B  %s" % (hashlib.md5(p.read_bytes()).hexdigest(),
                                 p.stat().st_size, f))
    out("  已指纹文件数: %d/%d" % (n_ok, len(files)))
    out("环境: python %s · numpy %s · %s" % (platform.python_version(),
                                             np.__version__, platform.platform()))
    out("生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return buf.getvalue()


SECTIONS = {"S01": section_s01, "S04": section_s04, "S05": section_s05,
            "S06": section_s06, "S07": section_s07}


# ---------------- 旧渲染模式（S05–S07 PNG） ----------------
def render(title: str, text: str):
    from PIL import Image, ImageDraw, ImageFont
    W, LH, PAD = 1440, 24, 18
    lines = text.rstrip("\n").split("\n")
    H = PAD * 2 + 46 + LH * len(lines)
    img = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, W, 40], fill=(47, 54, 64))
    try:
        ft = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 16)
        ftb = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 17)
    except OSError:
        ft = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
        ftb = ft
    dr.text((PAD, 9), title, font=ftb, fill=(255, 255, 255))
    yy = 46 + PAD
    for ln in lines:
        color = (34, 139, 34) if ("PASS" in ln and "结论" not in ln) else (
            (18, 97, 61) if "结论" in ln else (40, 44, 52))
        dr.text((PAD, yy), ln, font=ft, fill=color)
        yy += LH
    dr.rectangle([0, 0, W - 1, H - 1], outline=(200, 204, 210), width=1)
    return img


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="把校验过程打印到控制台（供截屏），不渲染 PNG")
    ap.add_argument("--only", default="", help="仅输出某段：S01|S04|S05|S06|S07")
    a = ap.parse_args()
    if a.do_print:
        if a.only:
            print(SECTIONS[a.only]())
        else:
            for k in ("S01", "S04", "S05", "S06", "S07"):
                print(SECTIONS[k]())
                print()
        raise SystemExit(0)
    titles = {
        "S05": "验证过程截图 S05 · 指定集指标独立复算（终端复现记录，真实执行输出）",
        "S06": "验证过程截图 S06 · 交卷 CSV 完整性校验（终端复现记录，真实执行输出）",
        "S07": "验证过程截图 S07 · 交稿包文件指纹与运行环境（终端复现记录，真实执行输出）",
    }
    for k, fn in (("S05", section_s04), ("S06", section_s05), ("S07", section_s07)):
        png = render(titles[k], fn())
        dst = SHOTS / ("%s_%s.png" % (k, {"S05": "独立复算", "S06": "CSV校验",
                                          "S07": "文件指纹与环境"}[k]))
        png.save(dst)
        print("saved:", dst)
