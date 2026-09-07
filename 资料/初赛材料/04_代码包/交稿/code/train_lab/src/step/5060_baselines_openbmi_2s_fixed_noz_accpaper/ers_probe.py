"""方案 07/08 附报：Test 窗 ERS+/ERS− 分层 Acc（事后探针，不改训练）。

用法（在对应训练包目录，需已有 summary / 或自行扩展）：
  当前为骨架：对 openbmi_*_noz 的 X 算 laterality，按阈值分层统计标签分布。
  完整「模型预测 × ERS 分层」需加载各折 best ckpt 后扩展（见方案 §4）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from data_paths import resolve_data

MU_LO, MU_HI = 8.0, 13.0
FS = 250.0
# OpenBMI 序：Cz,C3,C4,...
I_C3, I_C4 = 1, 2
ERD_THR = -15.0
LAT_THR = 8.0


def _mu_power(x_ct: np.ndarray) -> tuple[float, float]:
    """x_ct: (C,T) → C3/C4 mu band power."""
    f, pxx = welch(x_ct, fs=FS, nperseg=min(256, x_ct.shape[-1]))
    m = (f >= MU_LO) & (f <= MU_HI)
    return float(pxx[I_C3, m].mean()), float(pxx[I_C4, m].mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-tag", default="openbmi_2s_fixed_cue2to4_noz")
    p.add_argument("--limit", type=int, default=5000, help="最多扫描窗数（冒烟）")
    args = p.parse_args()

    data_dir, prefix = resolve_data(args.data_tag)
    X = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
    y3 = np.load(data_dir / f"{prefix}_y_three.npy")
    n = min(int(args.limit), len(X))
    print(f"scan {n}/{len(X)} from {data_dir}")

    ers_plus = ers_minus = rest_n = 0
    for i in range(n):
        yi = int(y3[i])
        if yi == 0:
            rest_n += 1
            continue
        x = np.asarray(X[i, 0], dtype=np.float64)
        p3, p4 = _mu_power(x)
        # 无 REST 基线时用简化「绝对功率偏侧」：右手对侧 C3 应更低 → 用 log 比近似
        if yi == 2:  # right
            contra, ipsi = p3, p4
        else:  # left
            contra, ipsi = p4, p3
        # 相对基线缺失：用 ipsi 作伪基线算伪 ERD
        erd_c = 100.0 * (contra - ipsi) / (ipsi + 1e-12)
        lat = 100.0 * (ipsi - contra) / ((ipsi + contra) / 2 + 1e-12)
        if erd_c <= ERD_THR and lat >= LAT_THR:
            ers_plus += 1
        else:
            ers_minus += 1

    out = {
        "data_tag": args.data_tag,
        "n_scanned": n,
        "n_rest": rest_n,
        "n_mi_ers_plus_proxy": ers_plus,
        "n_mi_ers_minus_proxy": ers_minus,
        "note": (
            "无同 trial REST 的绝对功率代理分层；正式 P-gap 需配对 REST 基线 "
            "+ 模型预测（见方案 07/08 §4）。本脚本仅验证 noz 数据可读。"
        ),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
