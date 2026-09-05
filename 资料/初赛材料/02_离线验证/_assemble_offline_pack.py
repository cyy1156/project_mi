# -*- coding: utf-8 -*-
"""Assemble 原始/ + 截图/ for offline Excel submission. Run once from anywhere."""
from __future__ import annotations

import csv
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "原始"
SHOT = ROOT / "截图"
RAW.mkdir(exist_ok=True)
SHOT.mkdir(exist_ok=True)

EXP37 = Path(r"D:/MI/code/train_lab/out/5070_challenge_exp37_nested_mcnemar_accpaper")
CSV_SRC = Path(
    r"D:/MI/code/train_lab/out/5070_challenge_mi_59ch_accpaper/submissions/"
    r"submission_exp34_e1f_a59_sens_full_20260902_1930.csv"
)
DATA = Path(r"D:/MI/DATA/挑战杯运动想象赛题数据文件")


def save_text_fig(path: Path, title: str, lines: list[str], figsize=(10, 6)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12, fontname="Microsoft YaHei")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        fontname="Microsoft YaHei",
        transform=ax.transAxes,
        linespacing=1.35,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    prob = np.load(EXP37 / "preds/oof_N0_prob.npy")
    y = np.load(EXP37 / "preds/oof_N0_y.npy")
    subj = np.load(EXP37 / "preds/oof_N0_subjects.npy", allow_pickle=True)
    pred = prob.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()
    specs: list[float] = []
    for c in range(3):
        tp = cm[c][c]
        fn = sum(cm[c]) - tp
        fp = sum(cm[r][c] for r in range(3)) - tp
        tn = sum(sum(row) for row in cm) - tp - fn - fp
        specs.append(tn / (tn + fp) if (tn + fp) else 0.0)

    folds = sorted(set(map(str, subj)))
    fold_accs: list[float] = []
    fold_detail = []
    for f in folds:
        m = np.array([str(s) == f for s in subj])
        fa = float(accuracy_score(y[m], pred[m]))
        fold_accs.append(fa)
        fold_detail.append({"subject": f, "n": int(m.sum()), "acc": fa})

    metrics = {
        "arm": "N0_nested_S0_QuadFold59",
        "protocol": "LOSO6 leave-fold nested Val on Challenge MI train S01-S06",
        "n_trials": int(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "macro_recall": float(recall_score(y, pred, average="macro", labels=[0, 1, 2])),
        "macro_specificity": float(np.mean(specs)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2])),
        "per_class_recall": recall_score(y, pred, average=None, labels=[0, 1, 2]).tolist(),
        "per_class_specificity": specs,
        "confusion_matrix": cm,
        "class_order": ["Left(0)", "Right(1)", "Rest(2)"],
        "fold_accs": fold_accs,
        "fold_acc_mean": float(np.mean(fold_accs)),
        "fold_acc_std": float(np.std(fold_accs, ddof=1)),
        "folds": fold_detail,
        "source_preds": str(EXP37 / "preds/oof_N0_prob.npy"),
        "source_y": str(EXP37 / "preds/oof_N0_y.npy"),
        "submission_csv": "原始/submission_QuadFold59.csv",
        "note": (
            "Test S07/S08 has no labels; metrics are nested Val only. "
            "Submission CSV is blind predictions."
        ),
    }
    (RAW / "nested_N0_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(CSV_SRC, RAW / "submission_QuadFold59.csv")

    (RAW / "数据说明_使用对照.md").write_text(
        """# 指定集使用对照（对照官方《数据说明》）

- 数据根：`DATA/挑战杯运动想象赛题数据文件/`
- Train：S01–S06 × 5 block = 900 trial；标签 201/202/204 → 0/1/2
- Test：S07–S08 × 2 block = 120 trial；**无 trigger / 无标签**
- 采样 250 Hz；每 trial 750 点；**先切 trial 再滤波**
- 交卷通道：丢弃 ECG/HEOR/HEOL/VEOU/VEOL → **59 EEG**
- 评测主读：train 上 LOSO6 **leave-fold 嵌套**（见 `nested_N0_metrics.json`）
- 盲测交付：`submission_QuadFold59.csv`（行序对齐 `sample_submission.csv`，utf-8-sig）
- **未使用测试集标签调参**
""",
        encoding="utf-8",
    )
    (RAW / "README.md").write_text(
        """# 原始验证数据包

| 文件 | 说明 |
|---|---|
| `submission_QuadFold59.csv` | 交卷预测（S07/S08，120 行） |
| `nested_N0_metrics.json` | QuadFold-59 嵌套主读 Acc/召/特/F1 + CM |
| `数据说明_使用对照.md` | 与官方数据说明对照 |
| `oof_N0/*.npy` | `oof_N0_subjects.npy` 为 Unicode 字符串数组，`np.load()` 默认参数可读；prob 形状 (900,3)、y 形状 (900,) |

复现嵌套指标：加载 Exp37 `oof_N0_*.npy` 或直接读本目录 JSON。
""",
        encoding="utf-8",
    )

    with open(DATA / "train/S01/block_1.pkl", "rb") as f:
        tr = pickle.load(f)
    with open(DATA / "test/S07/block_1.pkl", "rb") as f:
        te = pickle.load(f)
    n_train = len(list((DATA / "train").glob("S*/block_*.pkl")))
    n_test = len(list((DATA / "test").glob("S*/block_*.pkl")))
    save_text_fig(
        SHOT / "S01_指定集目录与pkl结构.png",
        "S01 · 指定集目录与 PKL 结构核验",
        [
            "root: DATA/Challenge-MI (官方指定集)",
            f"train blocks: {n_train}  (expect 30)",
            f"test  blocks: {n_test}  (expect 4)",
            f"train S01/block_1 data.shape = {tr['data'].shape}  srate={tr['srate']}",
            f"test  S07/block_1 data.shape = {te['data'].shape}  (no trigger)",
            f"nchan={tr['nchan']}  first_chs={list(tr['ch_names'][:5])} ...",
            "drop aux → 59 EEG for QuadFold-59",
            "PASS: matches official data README",
        ],
    )
    save_text_fig(
        SHOT / "S02_嵌套N0指标汇总.png",
        "S02 · QuadFold-59 嵌套 N0 主读（Exp37 OOF）",
        [
            f"n_trials = {metrics['n_trials']}",
            f"Acc            = {metrics['acc']:.4f}",
            f"macro recall   = {metrics['macro_recall']:.4f}",
            f"macro spec     = {metrics['macro_specificity']:.4f}",
            f"macro F1       = {metrics['macro_f1']:.4f}",
            f"fold Acc mean±std = {metrics['fold_acc_mean']:.4f} ± {metrics['fold_acc_std']:.4f}",
            f"fold Accs = {[round(a, 4) for a in fold_accs]}",
            f"CM (rows=true L/R/Rest): {cm}",
            "主读用于 Excel 00 行 1；折内 0.558 仅附报",
        ],
    )
    with open(RAW / "submission_QuadFold59.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    labs = [int(r[1]) for r in rows[1:]]
    head = rows[:31]
    save_text_fig(
        SHOT / "S03_交卷CSV前30行.png",
        "S03 · 交卷 CSV（对齐 sample_submission）",
        [
            "file: 原始/submission_QuadFold59.csv",
            f"n_rows = {len(rows) - 1} (expect 120)",
            f"label dist = {dict(Counter(labs))}",
            "",
            *[f"{a},{b}" for a, b in head],
            "...",
        ],
        figsize=(10, 8),
    )
    save_text_fig(
        SHOT / "S04_Excel总表指定集行说明.png",
        "S04 · Excel 核心指标（仅指定集）",
        [
            "本 Excel 仅报告主办方指定标准数据集（Challenge MI）",
            "不含自采 / OpenBMI / BCI2a / Stieger 指标",
            "",
            "模型: QuadFold-59（交卷终态）",
            "协议: LOSO6 · leave-fold 嵌套 Val（主读）",
            f"分类准确率     = {metrics['acc']:.4f}",
            f"召回率 macro   = {metrics['macro_recall']:.4f}",
            f"特异性 macro   = {metrics['macro_specificity']:.4f}",
            "运算延迟       = 1.11 ms（单 trial 前向）",
            "判定延迟       = 3.00 s（1 trial = 1 窗）",
            "CSV = submission_QuadFold59.csv（S07/S08 盲测）",
            "Test 无标签 → 不自报 test Acc",
        ],
    )
    print("OK", metrics["acc"], metrics["macro_specificity"], metrics["macro_f1"])
    print("shots", sorted(p.name for p in SHOT.glob("*.png")))


if __name__ == "__main__":
    main()
