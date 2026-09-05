"""数据解析：A0=旧 hop100（含 Rest）；A1+=pf1000 新臂（三类 · protocol_version≥3）。

pf1000 切割（与 `openbmi_pf1000/pipeline.py` / 数据切片说明 §3.3 一致）：
  - Left/Right：从 cue 起切 [cue, cue+5.6s)；段首 0.5s 仅作基线均值；不读 cue 前
  - Rest：Cue 前满 5.6s 同几何（评分 4s + post 1.6s），整段在 Cue 前
  - 窗：past100+cur500+future400；合法 t0∈{0.4…2.0} hop0.1
  - 标签：rest=0 / left=1 / right=2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _paths import HERE, PRE

# 本包优先，避免 OFFICIAL/baselines 的同名 shared_hparams 遮蔽
import sys

if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

from shared_hparams import DATA_TAG_A0, DATA_TAG_PF  # noqa: E402

# 与 preprocess pipeline.PROTOCOL_VERSION 对齐
PF_MIN_PROTOCOL_VERSION = 3
THREE_LABELS = {0, 1, 2}


def resolve_data_dir(tag: str) -> tuple[Path, str]:
    """返回 (dir, npy_prefix)。

    路径相对本仓库 `code/preprocess_lab/out/<tag>/`（`_paths.PRE` 由 __file__ 推导）。
    5090 机即 `F:\\Cyy\\MI\\code\\preprocess_lab\\out\\...`，勿写死 `D:\\cyy\\MI`。
    """
    tag = tag.strip().lower()
    if tag == DATA_TAG_A0:
        return PRE / "out" / DATA_TAG_A0, "openbmi"
    if tag == DATA_TAG_PF:
        return PRE / "out" / DATA_TAG_PF, "openbmi"
    d = PRE / "out" / tag
    return d, "openbmi"


def _load_meta(data_dir: Path) -> dict:
    p = data_dir / "preprocess_meta.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _assert_pf1000_three_class(y_three: np.ndarray, meta: dict, data_dir: Path) -> None:
    """A1+ 必须为三类齐全窗（含空闲）；旧 no_rest 产物拒绝加载。"""
    y = np.asarray(y_three).astype(np.int64).reshape(-1)
    uniq = set(int(u) for u in np.unique(y).tolist())
    ver = int(meta.get("protocol_version") or 0)
    no_rest = meta.get("no_rest")
    counts = {int(u): int(c) for u, c in zip(*np.unique(y, return_counts=True))}

    problems: list[str] = []
    if no_rest is True:
        problems.append("preprocess_meta.no_rest=true")
    if ver and ver < PF_MIN_PROTOCOL_VERSION:
        problems.append(
            f"protocol_version={ver} < {PF_MIN_PROTOCOL_VERSION}（需 Rest + Task 自 cue 起切）"
        )
    if not THREE_LABELS.issubset(uniq):
        problems.append(f"y_three 缺少三类，当前 unique={sorted(uniq)} counts={counts}")
    if 0 not in uniq:
        problems.append("无空闲态 label=0")

    if problems:
        raise RuntimeError(
            "pf1000 数据不符合当前三类协议：\n  - "
            + "\n  - ".join(problems)
            + f"\n目录: {data_dir}\n"
            "请在 preprocess_lab 重跑：\n"
            "  python -m src.datasets.openbmi_pf1000.batch --reset\n"
            "（Left/Right 从 cue 起切；Rest=Cue 前满 5.6s 同几何）"
        )


def _assert_a0_three_class(y_three: np.ndarray, data_dir: Path) -> None:
    uniq = set(int(u) for u in np.unique(np.asarray(y_three)).tolist())
    if not THREE_LABELS.issubset(uniq):
        raise RuntimeError(
            f"A0 数据应含 rest/left/right，当前 unique={sorted(uniq)} @ {data_dir}"
        )


def load_arrays(tag: str, *, require_three: bool = True) -> dict:
    data_dir, prefix = resolve_data_dir(tag)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"数据目录不存在: {data_dir}\n"
            f"A0 需要旧臂 out/{DATA_TAG_A0}；"
            f"A1+ 需要新臂 out/{DATA_TAG_PF}（三类 · 见数据切片与边界过滤说明.md）"
        )

    def _load(name: str):
        p = data_dir / f"{prefix}_{name}.npy"
        if not p.is_file():
            raise FileNotFoundError(p)
        return np.load(p, mmap_mode="r")

    meta = _load_meta(data_dir)
    y_three = np.asarray(_load("y_three"))
    out = {
        "dir": data_dir,
        "prefix": prefix,
        "meta": meta,
        "y_three": y_three,
        "subjects": np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True),
        "trial_id": np.asarray(_load("trial_id")),
    }

    y_task_p = data_dir / f"{prefix}_y_task.npy"
    if y_task_p.is_file():
        out["y_task"] = np.asarray(np.load(y_task_p, mmap_mode="r"))

    xf = data_dir / f"{prefix}_X_full.npy"
    x0 = data_dir / f"{prefix}_X.npy"
    if xf.is_file():
        out["X_full"] = np.load(xf, mmap_mode="r")
        xm = data_dir / f"{prefix}_X_mask.npy"
        if xm.is_file():
            out["X_mask"] = np.load(xm, mmap_mode="r")
    elif x0.is_file():
        out["X"] = np.load(x0, mmap_mode="r")
    else:
        raise FileNotFoundError(f"未找到 {prefix}_X*.npy under {data_dir}")

    t0p = data_dir / f"{prefix}_t0_sec.npy"
    if t0p.is_file():
        out["t0_sec"] = np.asarray(np.load(t0p))

    tag_l = tag.strip().lower()
    is_pf = tag_l == DATA_TAG_PF or data_dir.name == DATA_TAG_PF
    is_a0 = tag_l == DATA_TAG_A0 or data_dir.name == DATA_TAG_A0
    if require_three:
        if is_pf:
            _assert_pf1000_three_class(y_three, meta, data_dir)
        elif is_a0:
            _assert_a0_three_class(y_three, data_dir)

    return out


def summarize_labels(y_three: np.ndarray) -> dict:
    y = np.asarray(y_three).astype(np.int64).reshape(-1)
    u, c = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(u.tolist(), c.tolist())}


def to_bct(x: np.ndarray) -> np.ndarray:
    """统一到 (N, C, T)。"""
    a = np.asarray(x)
    if a.ndim == 4:
        if a.shape[1] == 1:
            a = a[:, 0]
        elif a.shape[-1] == 1:
            a = a[..., 0]
    if a.ndim != 3:
        raise ValueError(f"expect 3D after squeeze, got {a.shape}")
    if a.shape[1] in (500, 600, 800, 1000) and a.shape[2] == 8:
        a = np.transpose(a, (0, 2, 1))
    return a
