"""从 v3 会话 EEG 在当前 OpenBMI 3s 底座上微调被试专用权重。

用法:
  python experiment_game/tools/ft_subject_from_v3.py \\
    --session experiment_game/data/sessions/fnz_ws01_20260826_164149 \\
    --session experiment_game/data/sessions/fnz_ws02_20260826_171537 \\
    --out-dir experiment_game/data/models/fnz

输出:
  {out_dir}/
    best_three.pt · best_task.pt · meta.json · report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "preprocess_lab"))

from adapt_engine.ft import FTRecipe, IncrementalFinetuner, ReplayPool  # noqa: E402
from adapt_engine.registry import load_head  # noqa: E402
from experiment_game.tools.openbmi_replay_pool import (  # noqa: E402
    DEFAULT_REPLAY_RATIO,
    DEFAULT_SEED,
    build_t0_replay_pool,
    build_t0_task_replay_pool,
    resolve_openbmi_root,
)
from experiment_game.experiment.channel_layout import (  # noqa: E402
    DEVICE_CHANNEL_LABELS,
    reorder_device_to_frozen,
)
from src.common.steps.epoch_baseline import task_window_cue_0_to_4  # noqa: E402
from src.common.steps.filter_car import car_reference, notch_and_bandpass  # noqa: E402
from src.common.steps.resample_zscore import trial_zscore  # noqa: E402
from experiment_game.offline.openbmi_align_cut import iter_rest_sources_from_table  # noqa: E402
from src.common.steps.slide_1s import extract_segment_baseline, iter_rest_sources_cue_before  # noqa: E402
from src.common.steps.slide_3s_hop100 import (  # noqa: E402
    WIN_SEC as WIN_SEC_3S,
    segment_to_3s_hop100_windows,
)

FS = 250.0
WIN_S, HOP_S, T0_MIN = 3.0, 0.1, 0.4
N_TIMES = 750
PROTOCOL_OPENBMI_ALIGN = "openbmi_align"
PROTOCOL_LEGACY_V3 = "legacy_v3"
PROTOCOL_SIM_MAT = "sim_mat_cue"

# 默认底座：E1f 四成员 shallow fold0（5090 · 2026-08-23；task 头 2026-08-29 补齐）
DEFAULT_TASK = (
    _REPO
    / "code/train_lab/out/5090_alg_incr_3s_hop100_accpaper"
    / "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100"
    / "run_20260823_095327/task/fold0/best_task.pt"
)
DEFAULT_THREE = (
    _REPO
    / "code/train_lab/out/5090_alg_incr_3s_hop100_accpaper"
    / "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100"
    / "run_20260823_095327/three/fold0/best_three.pt"
)

LABEL_NAMES = {0: "Rest", 1: "Left", 2: "Right"}

# 发布验收（fnz 方案 §7.3）
RELEASE_HELDOUT_ACC_MIN = 0.40
RELEASE_MAX_CLASS_FRAC = 0.60
RELEASE_TRAIN_HELDOUT_GAP_MAX = 0.35

# 采后 FT 默认（heldout 早停 · 可复现）
DEFAULT_FT_MAX_EPOCHS = 20
DEFAULT_FT_PATIENCE = 5
DEFAULT_FT_EPOCHS_FIXED = 5
DEFAULT_FT_DETERMINISTIC = True


def set_training_deterministic(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机性（减轻同配置重复 FT 波动）。"""
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_int(v: Any, default: int = 0) -> int:
    """CSV 空单元格 → pandas NaN；NaN 在 Python 中为真，`int(nan or 0)` 会炸。"""
    if v is None:
        return default
    try:
        if isinstance(v, float) and np.isnan(v):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(v, str) and not v.strip():
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _load_eeg(session_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    for p in (session_dir / "eeg.csv", session_dir / "continuous" / "eeg.csv"):
        if p.is_file():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError(f"无 eeg.csv: {session_dir}")
    t = df["lsl_time"].to_numpy(dtype=np.float64)
    cols = []
    for name in DEVICE_CHANNEL_LABELS:
        if name in df.columns:
            cols.append(name)
        elif name.upper() in df.columns:
            cols.append(name.upper())
        else:
            # CZ/CPZ 大小写
            alt = next((c for c in df.columns if c.upper() == name.upper()), None)
            if alt is None:
                raise KeyError(f"eeg.csv 缺少通道 {name}; cols={list(df.columns)}")
            cols.append(alt)
    x_dev = df[cols].to_numpy(dtype=np.float64)
    return t, reorder_device_to_frozen(x_dev)


def _lsl_to_sample(t_lsl: np.ndarray, t: float) -> int:
    return int(np.searchsorted(t_lsl, float(t)))


def _seg_windows_to_ch_time(wins: List[np.ndarray]) -> List[np.ndarray]:
    """(T,C) → (C,T) float32，与 legacy 切窗输出一致。"""
    return [w.T.astype(np.float32) for w in wins if w.shape[0] == N_TIMES]


def is_sim_session(session_dir: Path) -> bool:
    sd = Path(session_dir)
    if (sd / "sim_script.json").is_file():
        return True
    meta_p = sd / "session.meta.json"
    if meta_p.is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        return bool(meta.get("sim_mode")) or meta.get("phase_mode") == "sim_v3_session"
    return False


def detect_session_protocol(session_dir: Path) -> str:
    """历史 v3：Cue 与 mi_start 间隔 >0.5s → legacy_v3；仿真 → sim_mat_cue；否则 openbmi_align。"""
    if is_sim_session(session_dir):
        return PROTOCOL_SIM_MAT
    table = session_dir / "alignment" / "trial_table.csv"
    if not table.is_file():
        return PROTOCOL_OPENBMI_ALIGN
    rows = pd.read_csv(table).to_dict(orient="records")
    for r in rows:
        if _safe_int(r.get("rejected"), 0) == 1:
            continue
        lab = _safe_int(r.get("label"), -1)
        if lab not in (1, 2):
            continue
        tc, tm = r.get("t_cue"), r.get("t_mi_start")
        if tc != tc or tm != tm:  # NaN
            if tm == tm:
                return PROTOCOL_LEGACY_V3
            continue
        if float(tm) - float(tc) > 0.5:
            return PROTOCOL_LEGACY_V3
    return PROTOCOL_OPENBMI_ALIGN


def _resolve_protocol(session_dirs: List[Path], protocol: str) -> str:
    if protocol != "auto":
        return protocol
    protos = {detect_session_protocol(d) for d in session_dirs}
    if len(protos) > 1:
        raise ValueError(
            f"多 session 协议不一致 {protos}；请显式 --protocol 或分开 FT"
        )
    return protos.pop() if protos else PROTOCOL_OPENBMI_ALIGN


def _cut_windows(x_filt: np.ndarray, t_lsl: np.ndarray, t0: float, t1: float) -> List[np.ndarray]:
    i0 = int(np.searchsorted(t_lsl, t0))
    i1 = int(np.searchsorted(t_lsl, t1))
    dur = (i1 - i0) / FS
    if dur < WIN_S + T0_MIN - 1e-6:
        return []
    outs: List[np.ndarray] = []
    anchor = T0_MIN
    while anchor + WIN_S <= dur + 1e-9:
        a = i0 + int(round(anchor * FS))
        b = i0 + int(round((anchor + WIN_S) * FS))
        if b - a != N_TIMES:
            anchor = round(anchor + HOP_S, 3)
            continue
        w = trial_zscore(x_filt[a:b]).T.astype(np.float32)  # (8,750)
        outs.append(w)
        anchor = round(anchor + HOP_S, 3)
    return outs


def _build_session_windows_legacy(
    session_dir: Path,
    *,
    include_invalid: bool = True,
) -> Dict[str, np.ndarray]:
    """历史 v3：mi_start+T0_MIN 前向切窗，含 Rest 想象试次。"""
    t_lsl, x_raw = _load_eeg(session_dir)
    x_filt = notch_and_bandpass(car_reference(x_raw), FS, l_freq=8.0, h_freq=30.0)
    table = session_dir / "alignment" / "trial_table.csv"
    rows = list(pd.read_csv(table).to_dict(orient="records"))

    wins: List[np.ndarray] = []
    y_three: List[int] = []
    y_task: List[int] = []
    split_ids: List[str] = []
    used_trials = 0
    sess_name = session_dir.name
    for r in rows:
        if _safe_int(r.get("rejected"), 0) == 1:
            continue
        if not include_invalid and _safe_int(r.get("invalid"), 0) == 1:
            continue
        lab = _safe_int(r.get("label"), -1)
        if lab not in (0, 1, 2):
            continue
        if not (r.get("t_mi_start") == r.get("t_mi_start") and r.get("t_mi_end") == r.get("t_mi_end")):
            continue
        t_a, t_b = float(r["t_mi_start"]), float(r["t_mi_end"])
        ws = _cut_windows(x_filt, t_lsl, t_a, t_b)
        if not ws:
            continue
        used_trials += 1
        sid = f"{sess_name}:{_safe_int(r.get('trial_id'), 0)}"
        for w in ws:
            wins.append(w)
            y_three.append(lab)
            y_task.append(0 if lab == 0 else 1)
            split_ids.append(sid)

    if not wins:
        raise RuntimeError(f"未切出任何训练窗：{session_dir}")
    return {
        "X": np.stack(wins, axis=0),
        "y_three": np.asarray(y_three, dtype=np.int64),
        "y_task": np.asarray(y_task, dtype=np.int64),
        "split_id": np.asarray(split_ids),
        "n_trials": used_trials,
        "session": sess_name,
        "protocol": PROTOCOL_LEGACY_V3,
    }


def _cue_time_from_row(r: Dict[str, Any]) -> Optional[float]:
    tc, tm = r.get("t_cue"), r.get("t_mi_start")
    if tc == tc and tm == tm:
        if abs(float(tm) - float(tc)) <= 0.5:
            return float(tc)
        return float(tm)  # legacy fallback
    if tc == tc:
        return float(tc)
    if tm == tm:
        return float(tm)
    return None


def _build_session_windows_openbmi_align(
    session_dir: Path,
    *,
    include_invalid: bool = True,
) -> Dict[str, np.ndarray]:
    """OpenBMI-Align v1：与 preprocess_run_3s_hop100 同构。"""
    t_lsl, x_raw = _load_eeg(session_dir)
    x_filt = notch_and_bandpass(car_reference(x_raw), FS, l_freq=8.0, h_freq=30.0)
    table = session_dir / "alignment" / "trial_table.csv"
    rows = list(pd.read_csv(table).to_dict(orient="records"))

    wins: List[np.ndarray] = []
    y_three: List[int] = []
    y_task: List[int] = []
    split_ids: List[str] = []
    used_trials = 0
    sess_name = session_dir.name
    cue_samples: List[int] = []
    n_left = n_right = n_rest = 0

    for r in rows:
        if _safe_int(r.get("rejected"), 0) == 1:
            continue
        if not include_invalid and _safe_int(r.get("invalid"), 0) == 1:
            continue
        lab = _safe_int(r.get("label"), -1)
        if lab not in (0, 1, 2):
            continue
        t_cue = _cue_time_from_row(r)
        if t_cue is None:
            continue
        cue_idx = _lsl_to_sample(t_lsl, t_cue)
        if lab in (1, 2):
            cue_samples.append(cue_idx)
        if lab == 0:
            n_rest += 1
        elif lab == 1:
            n_left += 1
        else:
            n_right += 1

        seg = task_window_cue_0_to_4(x_filt, cue_idx, FS)
        if seg is None:
            continue
        seg_wins = segment_to_3s_hop100_windows(seg, FS, zscore=True)
        ws = _seg_windows_to_ch_time(seg_wins)
        if not ws:
            continue
        used_trials += 1
        sid = f"{sess_name}:{_safe_int(r.get('trial_id'), 0)}"
        for w in ws:
            wins.append(w)
            y_three.append(lab)
            y_task.append(0 if lab == 0 else 1)
            split_ids.append(sid)

    has_explicit_rest = n_rest > 0
    if not has_explicit_rest and (cue_samples or rows):
        max_rest = min(n_left, n_right) if (n_left + n_right) else 0
        rest_sources = iter_rest_sources_from_table(
            rows,
            t_lsl,
            skip_rejected=True,
            skip_invalid=not include_invalid,
            min_win_sec=WIN_SEC_3S,
        )
        if not rest_sources and cue_samples:
            fb = iter_rest_sources_cue_before(
                np.asarray(sorted(cue_samples), dtype=int),
                FS,
                x_filt.shape[0],
                rest_sec=4.0,
                task_sec=4.0,
                min_win_sec=WIN_SEC_3S,
            )
            rest_sources = [(-(i + 1), int(t0), int(t1)) for i, (t0, t1) in enumerate(fb)]
        for ri, (tid, t0, t1) in enumerate(rest_sources[: int(max_rest)]):
            seg = extract_segment_baseline(x_filt, int(t0), int(t1), FS, baseline_sec=0.5)
            if seg is None:
                continue
            seg_wins = segment_to_3s_hop100_windows(seg, FS, zscore=True)
            ws = _seg_windows_to_ch_time(seg_wins)
            if not ws:
                continue
            sid = f"{sess_name}:rest_{tid if tid > 0 else ri}"
            for w in ws:
                wins.append(w)
                y_three.append(0)
                y_task.append(0)
                split_ids.append(sid)

    if not wins:
        raise RuntimeError(f"未切出任何训练窗（openbmi_align）：{session_dir}")
    return {
        "X": np.stack(wins, axis=0),
        "y_three": np.asarray(y_three, dtype=np.int64),
        "y_task": np.asarray(y_task, dtype=np.int64),
        "split_id": np.asarray(split_ids),
        "n_trials": used_trials,
        "session": sess_name,
        "protocol": PROTOCOL_OPENBMI_ALIGN,
    }


def _build_session_windows_sim(
    session_dir: Path,
    *,
    include_invalid: bool = True,
) -> Dict[str, np.ndarray]:
    """仿真 session：按 mat cue_sample 切窗，不依赖 LSL 对齐。"""
    from experiment_game.experiment.sim.bci2a_mat_loader import load_bci2a_run
    from experiment_game.experiment.sim.sim_script_io import (
        load_sim_script,
        rebuild_sim_script_from_session,
    )

    script = load_sim_script(session_dir) or rebuild_sim_script_from_session(session_dir)
    if script is None:
        raise RuntimeError(f"无法加载仿真脚本：{session_dir}")
    rd = load_bci2a_run(script.mat_path, script.run_id)
    x_filt = notch_and_bandpass(car_reference(rd.x8), FS, l_freq=8.0, h_freq=30.0)

    table_rows: List[Dict[str, Any]] = []
    table_p = session_dir / "alignment" / "trial_table.csv"
    if table_p.is_file():
        table_rows = list(pd.read_csv(table_p).to_dict(orient="records"))

    wins: List[np.ndarray] = []
    y_three: List[int] = []
    y_task: List[int] = []
    split_ids: List[str] = []
    used_trials = 0
    sess_name = session_dir.name

    for ti, tr in enumerate(script.trials):
        trial_id = ti + 1
        if table_rows and ti < len(table_rows):
            row = table_rows[ti]
            if _safe_int(row.get("rejected"), 0) == 1:
                continue
            if not include_invalid and _safe_int(row.get("invalid"), 0) == 1:
                continue
            lab = _safe_int(row.get("label"), int(tr.label))
        else:
            lab = int(tr.label)

        # Rest：任务窗取 Cue 前静息段起点；L/R：取 MI cue
        cue_idx = int(tr.rest_start_sample) if lab == 0 else int(tr.cue_sample)
        seg = task_window_cue_0_to_4(x_filt, cue_idx, FS)
        if seg is None:
            continue
        seg_wins = segment_to_3s_hop100_windows(seg, FS, zscore=True)
        ws = _seg_windows_to_ch_time(seg_wins)
        if not ws:
            continue
        used_trials += 1
        sid = f"{sess_name}:{trial_id}"
        for w in ws:
            wins.append(w)
            y_three.append(lab)
            y_task.append(0 if lab == 0 else 1)
            split_ids.append(sid)

    if not wins:
        raise RuntimeError(f"未切出任何训练窗（sim_mat）：{session_dir}")
    return {
        "X": np.stack(wins, axis=0),
        "y_three": np.asarray(y_three, dtype=np.int64),
        "y_task": np.asarray(y_task, dtype=np.int64),
        "split_id": np.asarray(split_ids),
        "n_trials": used_trials,
        "session": sess_name,
        "protocol": PROTOCOL_SIM_MAT,
    }


def _build_session_windows(
    session_dir: Path,
    *,
    include_invalid: bool = True,
    protocol: str = PROTOCOL_OPENBMI_ALIGN,
) -> Dict[str, np.ndarray]:
    """单会话切窗；split_id = '{session_name}:{trial_id}' 供多会话合并划分。"""
    if protocol == PROTOCOL_SIM_MAT or (
        protocol == "auto" and is_sim_session(session_dir)
    ):
        return _build_session_windows_sim(session_dir, include_invalid=include_invalid)
    if protocol == PROTOCOL_LEGACY_V3:
        return _build_session_windows_legacy(session_dir, include_invalid=include_invalid)
    return _build_session_windows_openbmi_align(session_dir, include_invalid=include_invalid)


def build_dataset(
    session_dir: Path | List[Path],
    *,
    include_invalid: bool = True,
    protocol: str = "auto",
) -> Dict[str, np.ndarray]:
    """协议标签监督；支持单会话或多会话合并。"""
    dirs = [session_dir] if isinstance(session_dir, Path) else list(session_dir)
    resolved = _resolve_protocol([d.resolve() for d in dirs], protocol)
    parts = [
        _build_session_windows(d.resolve(), include_invalid=include_invalid, protocol=resolved)
        for d in dirs
    ]
    if len(parts) == 1:
        p = parts[0]
        return {
            "X": p["X"],
            "y_three": p["y_three"],
            "y_task": p["y_task"],
            "split_id": p["split_id"],
            "n_trials": np.asarray([p["n_trials"]], dtype=np.int64),
            "sessions": [p["session"]],
            "protocol": resolved,
        }
    return {
        "X": np.concatenate([p["X"] for p in parts], axis=0),
        "y_three": np.concatenate([p["y_three"] for p in parts], axis=0),
        "y_task": np.concatenate([p["y_task"] for p in parts], axis=0),
        "split_id": np.concatenate([p["split_id"] for p in parts], axis=0),
        "n_trials": np.asarray([sum(p["n_trials"] for p in parts)], dtype=np.int64),
        "sessions": [p["session"] for p in parts],
        "protocol": resolved,
    }


def _trial_split(
    split_ids: np.ndarray,
    *,
    train_frac: float = 0.7,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(split_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_tr = max(1, int(round(len(uniq) * train_frac)))
    if n_tr >= len(uniq):
        n_tr = max(1, len(uniq) - 1) if len(uniq) > 1 else 1
    tr_set = set(uniq[:n_tr].tolist())
    te_set = set(uniq[n_tr:].tolist()) if len(uniq) > 1 else set(uniq.tolist())
    tr_mask = np.array([t in tr_set for t in split_ids])
    te_mask = np.array([t in te_set for t in split_ids])
    if not te_mask.any():
        te_mask = tr_mask.copy()
    return tr_mask, te_mask


@torch.no_grad()
def _pred_distribution(model, X: np.ndarray, device: str) -> Dict[str, Any]:
    if len(X) == 0:
        return {"pred_counts": {}, "mean_p": []}
    model.eval()
    preds, logits_all = [], []
    bs = 64
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s : s + bs]).to(device)
        try:
            logits = model(xb)
        except RuntimeError:
            logits = model(xb.unsqueeze(1))
        if logits.dim() == 3:
            logits = logits.reshape(logits.shape[0], -1)
        logits_all.append(logits.cpu().numpy())
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    pred = np.concatenate(preds)
    logits = np.concatenate(logits_all)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    uniq, cnt = np.unique(pred, return_counts=True)
    return {
        "pred_counts": {int(k): int(v) for k, v in zip(uniq, cnt)},
        "mean_p": [float(x) for x in probs.mean(axis=0)],
        "max_class_frac": float(cnt.max() / len(pred)) if len(pred) else 0.0,
    }


@torch.no_grad()
def _eval_acc(model, X: np.ndarray, y: np.ndarray, device: str) -> float:
    if len(X) == 0:
        return float("nan")
    model.eval()
    preds = []
    bs = 64
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s : s + bs]).to(device)
        try:
            logits = model(xb)
        except RuntimeError:
            logits = model(xb.unsqueeze(1))
        if logits.dim() == 3:
            logits = logits.reshape(logits.shape[0], -1)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    pred = np.concatenate(preds)
    return float((pred == y).mean())


def _save_ckpt(path: Path, model, *, n_outputs: int, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "n_outputs": int(n_outputs),
            **meta,
        },
        path,
    )


def evaluate_release_gate(
    rep: Dict[str, Any],
    *,
    heldout_acc_min: float = RELEASE_HELDOUT_ACC_MIN,
    max_class_frac: float = RELEASE_MAX_CLASS_FRAC,
    train_gap_max: float = RELEASE_TRAIN_HELDOUT_GAP_MAX,
) -> Dict[str, Any]:
    dist = rep.get("heldout_pred_dist") or {}
    pc = dist.get("pred_counts") or {}
    acc_te = float(rep.get("acc_after_heldout", 0.0))
    acc_tr = float(rep.get("acc_after_train", 0.0))
    mx = float(dist.get("max_class_frac", 1.0))
    gap = acc_tr - acc_te
    classes_present = {int(k) for k in pc.keys()}
    three_ok = classes_present == {0, 1, 2}
    checks = {
        "heldout_acc": acc_te >= heldout_acc_min,
        "max_class_frac": mx < max_class_frac,
        "train_gap": gap < train_gap_max,
        "three_classes_pred": three_ok,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "heldout_acc": acc_te,
        "train_acc": acc_tr,
        "train_minus_heldout": gap,
        "max_class_frac": mx,
        "pred_counts": pc,
        "pred_labels": {LABEL_NAMES.get(k, str(k)): pc[k] for k in sorted(pc)},
    }


def finetune_head(
    ckpt: Path,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    n_outputs: int,
    device: str,
    recipe: FTRecipe,
    replay_pool: Optional[ReplayPool],
    out_path: Path,
    meta: Dict[str, Any],
    early_stop: bool = True,
    max_epochs: int = DEFAULT_FT_MAX_EPOCHS,
    patience: int = DEFAULT_FT_PATIENCE,
    fixed_epochs: int = DEFAULT_FT_EPOCHS_FIXED,
) -> Dict[str, Any]:
    entry = load_head(ckpt, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    acc0_tr = _eval_acc(model, X_tr, y_tr, device)
    acc0_te = _eval_acc(model, X_te, y_te, device)

    fin = IncrementalFinetuner(
        model, recipe, replay_pool=replay_pool, device=device, ckpt_dir=None
    )
    if early_stop:
        rec = fin.train_with_early_stop(
            X_tr,
            y_tr,
            lambda: _eval_acc(model, X_te, y_te, device),
            max_epochs=int(max_epochs),
            patience=int(patience),
            min_epochs=1,
        )
    else:
        recipe_fixed = FTRecipe(
            lr=recipe.lr,
            weight_decay=recipe.weight_decay,
            epochs=int(fixed_epochs),
            batch_size=recipe.batch_size,
            replay_ratio=recipe.replay_ratio,
            seed=recipe.seed,
            balanced_batch=recipe.balanced_batch,
            aug_fn=recipe.aug_fn,
        )
        fin = IncrementalFinetuner(
            model, recipe_fixed, replay_pool=replay_pool, device=device, ckpt_dir=None
        )
        rec = fin.train_round(X_tr, y_tr, frozen=False)
        rec = dict(rec)
        rec["early_stop"] = False
        rec["epochs_run"] = int(fixed_epochs)

    acc1_tr = _eval_acc(model, X_tr, y_tr, device)
    acc1_te = _eval_acc(model, X_te, y_te, device)
    pred_te = _pred_distribution(model, X_te, device)
    ft_meta = {
        **meta,
        "init_ckpt": str(ckpt),
        "ft_record": rec,
        "early_stop": bool(early_stop),
        "acc_before": {"train": acc0_tr, "heldout": acc0_te},
        "acc_after": {"train": acc1_tr, "heldout": acc1_te},
    }
    if early_stop:
        ft_meta["max_epochs"] = int(max_epochs)
        ft_meta["patience"] = int(patience)
        ft_meta["best_epoch"] = rec.get("best_epoch")
        ft_meta["best_heldout_acc"] = rec.get("best_heldout_acc")
    else:
        ft_meta["fixed_epochs"] = int(fixed_epochs)
    _save_ckpt(
        out_path,
        model,
        n_outputs=n_outputs,
        meta=ft_meta,
    )
    return {
        "acc_before_train": acc0_tr,
        "acc_before_heldout": acc0_te,
        "acc_after_train": acc1_tr,
        "acc_after_heldout": acc1_te,
        "heldout_pred_dist": pred_te,
        "ft": rec,
        "out": str(out_path),
    }


def _collect_session_lineage(session_dirs: List[Path]) -> Dict[str, Any]:
    lineages: List[Dict[str, Any]] = []
    for d in session_dirs:
        item: Dict[str, Any] = {"session": d.name}
        meta_p = d / "session.meta.json"
        if meta_p.is_file():
            try:
                m = json.loads(meta_p.read_text(encoding="utf-8"))
                if m.get("campaign_id"):
                    item["campaign_id"] = m.get("campaign_id")
                sr = m.get("source_run") or m.get("session_id")
                if sr:
                    item["source_run"] = sr
                if m.get("session_id"):
                    item["session_id"] = m.get("session_id")
            except Exception:
                pass
        lineages.append(item)
    out: Dict[str, Any] = {"session_lineage": lineages}
    if lineages:
        primary = lineages[0]
        if primary.get("campaign_id"):
            out["campaign_id"] = primary["campaign_id"]
        if primary.get("source_run"):
            out["source_run"] = primary["source_run"]
    return out


def _write_ft_report(
    out_dir: Path,
    *,
    subject_id: str,
    sess_label: str,
    session_dirs: List[Path],
    ds: Dict[str, Any],
    X: np.ndarray,
    y3: np.ndarray,
    three_rep: Dict[str, Any],
    task_rep: Dict[str, Any],
    release: Dict[str, Any],
    args_task_ckpt: Path,
    args_three_ckpt: Path,
    train_session_dirs: Optional[List[Path]] = None,
    heldout_session_dirs: Optional[List[Path]] = None,
    leave_next: bool = False,
    replay_pool: str = "none",
    replay_ratio: float = 0.0,
) -> Dict[str, Any]:
    train_dirs = [Path(p) for p in (train_session_dirs or session_dirs)]
    hold_dirs = [Path(p) for p in (heldout_session_dirs or [])]
    report = {
        "subject_id": subject_id,
        "sessions": ds.get("sessions", []),
        "session": sess_label,
        "out_dir": str(out_dir),
        "n_windows": int(len(X)),
        "n_trials": int(ds["n_trials"][0]),
        "class_counts_three": {int(k): int(v) for k, v in zip(*np.unique(y3, return_counts=True))},
        "three": three_rep,
        "task": task_rep,
        "release_gate": release,
        "base_task_ckpt": str(args_task_ckpt),
        "base_three_ckpt": str(args_three_ckpt),
        "leave_next": bool(leave_next),
        "train_sessions": [d.name for d in train_dirs],
        "heldout_sessions": [d.name for d in hold_dirs],
        "replay_pool": replay_pool,
        "replay_ratio": float(replay_ratio),
        "no_replay": str(replay_pool).lower() in ("", "none"),
    }
    # lineage：训练 + heldout，便于预设标签拼出 ws02+ws03+ws04→ws05
    report.update(_collect_session_lineage(train_dirs + hold_dirs))
    if hold_dirs:
        # source_run 保留首个训练场，避免覆盖；标签走 train/heldout 字段
        train_line = _collect_session_lineage(train_dirs)
        if train_line.get("source_run"):
            report["source_run"] = train_line["source_run"]
    (out_dir / "meta.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    split_title = (
        "准确率（Leave-Next：训练场 vs heldout 场）"
        if leave_next
        else "准确率（按试次划分 heldout）"
    )
    md = "\n".join(
        [
            f"# 被试专用权重 · {subject_id}",
            "",
            f"- 会话：`{sess_label}`",
            f"- 训练：`{[d.name for d in train_dirs]}`",
            f"- heldout：`{[d.name for d in hold_dirs]}`" if hold_dirs else "- heldout：试次划分",
            f"- replay：`{replay_pool}` ratio={replay_ratio}",
            f"- 窗数：{len(X)}（试次 {int(ds['n_trials'][0])}）",
            f"- 底座 three：`{args_three_ckpt}`",
            f"- 底座 task：`{args_task_ckpt}`",
            "",
            f"## {split_title}",
            "",
            "| 头 | train 前→后 | heldout 前→后 |",
            "|----|-------------|---------------|",
            (
                f"| three | {three_rep['acc_before_train']:.3f}→{three_rep['acc_after_train']:.3f} | "
                f"{three_rep['acc_before_heldout']:.3f}→{three_rep['acc_after_heldout']:.3f} |"
            ),
            (
                f"| task（参考） | {task_rep['acc_before_train']:.3f}→{task_rep['acc_after_train']:.3f} | "
                f"{task_rep['acc_before_heldout']:.3f}→{task_rep['acc_after_heldout']:.3f} |"
            ),
            "",
            "## heldout 预测分布（three）",
            "",
            f"- pred_counts: `{three_rep.get('heldout_pred_dist', {}).get('pred_counts', {})}`",
            f"- mean_p Rest/L/R: `{three_rep.get('heldout_pred_dist', {}).get('mean_p', [])}`",
            "",
            "## 发布验收（仅 three · 参考）",
            "",
            f"- **结果**: {'PASS' if release['pass'] else 'FAIL'}",
            f"- checks: `{release.get('checks')}`",
            f"- pred 分布: `{release.get('pred_labels')}`",
            f"- FT 策略: early_stop={three_rep.get('ft', {}).get('early_stop')} "
            f"epochs_run={three_rep.get('ft', {}).get('epochs_run')} "
            f"best_epoch={three_rep.get('ft', {}).get('best_epoch')}",
            "",
            "## 使用方式",
            "",
            f"- `s3_three_ckpt`: `{out_dir / 'best_three.pt'}`",
            f"- `s3_task_ckpt`: `{out_dir / 'best_task.pt'}`",
            "",
        ]
    )
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "release_gate.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _session_leave_split(
    split_ids: np.ndarray,
    *,
    train_session_names: List[str],
    heldout_session_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """按会话名划分：train / heldout（Leave-Next）。split_id 形如 '{session}:{trial}'。"""
    train_set = {str(n) for n in train_session_names}
    hold_set = {str(n) for n in heldout_session_names}

    def _sess(sid: Any) -> str:
        s = str(sid)
        return s.split(":", 1)[0] if ":" in s else s

    tr_mask = np.array([_sess(t) in train_set for t in split_ids], dtype=bool)
    te_mask = np.array([_sess(t) in hold_set for t in split_ids], dtype=bool)
    if not tr_mask.any():
        raise ValueError(
            f"Leave-Next：训练会话无窗（train={sorted(train_set)}）"
        )
    if not te_mask.any():
        raise ValueError(
            f"Leave-Next：heldout 会话无窗（heldout={sorted(hold_set)}）"
        )
    return tr_mask, te_mask


def run_subject_finetune(
    session_dirs: List[Path],
    out_dir: Path,
    *,
    task_ckpt: Path = DEFAULT_TASK,
    three_ckpt: Path = DEFAULT_THREE,
    epochs: int = DEFAULT_FT_EPOCHS_FIXED,
    lr: float = 1e-4,
    batch_size: int = 32,
    train_frac: float = 0.7,
    replay_ratio: float = DEFAULT_REPLAY_RATIO,
    no_replay: bool = False,
    seed: int = DEFAULT_SEED,
    device: Optional[str] = None,
    exclude_invalid: bool = False,
    protocol: str = "auto",
    verbose: bool = True,
    early_stop: bool = True,
    max_epochs: int = DEFAULT_FT_MAX_EPOCHS,
    patience: int = DEFAULT_FT_PATIENCE,
    deterministic: bool = DEFAULT_FT_DETERMINISTIC,
    heldout_session_dirs: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """运行被试 FT；**始终写入 out_dir**，门控 FAIL 不抛错。

    若提供 heldout_session_dirs：Leave-Next —— session_dirs 全作训练，
    heldout_session_dirs 全作 heldout（不再按试次 7:3 切）。

    返回 dict：status, release_gate, report, out_dir, ...
    """
    session_dirs = [Path(p).resolve() for p in session_dirs]
    hold_dirs = [Path(p).resolve() for p in (heldout_session_dirs or [])]
    for session_dir in session_dirs + hold_dirs:
        if not session_dir.is_dir():
            raise FileNotFoundError(f"会话不存在: {session_dir}")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if deterministic:
        set_training_deterministic(seed)

    leave_next = bool(hold_dirs)
    all_dirs = session_dirs + hold_dirs if leave_next else session_dirs
    sess_label = " + ".join(d.name for d in all_dirs)
    if verbose:
        if leave_next:
            print(
                f"[1/4] Leave-Next 切窗 · train={[d.name for d in session_dirs]} · "
                f"heldout={[d.name for d in hold_dirs]}",
                flush=True,
            )
        else:
            print(f"[1/4] 切窗 · {sess_label}", flush=True)

    ds = build_dataset(all_dirs, include_invalid=not exclude_invalid, protocol=protocol)
    X, y3, y2, split_ids = ds["X"], ds["y_three"], ds["y_task"], ds["split_id"]
    resolved_protocol = ds.get("protocol", protocol)
    if verbose:
        print(
            f"  protocol={resolved_protocol} windows={len(X)} trials={int(ds['n_trials'][0])} "
            f"y3={dict(zip(*np.unique(y3, return_counts=True)))}",
            flush=True,
        )

    if leave_next:
        tr_m, te_m = _session_leave_split(
            split_ids,
            train_session_names=[d.name for d in session_dirs],
            heldout_session_names=[d.name for d in hold_dirs],
        )
        if verbose:
            print(
                f"  Leave-Next windows train={int(tr_m.sum())} heldout={int(te_m.sum())} "
                f"trials train={len(np.unique(split_ids[tr_m]))} "
                f"heldout={len(np.unique(split_ids[te_m]))}",
                flush=True,
            )
    else:
        tr_m, te_m = _trial_split(split_ids, train_frac=train_frac, seed=seed)
        if verbose:
            print(
                f"  split trials train={len(np.unique(split_ids[tr_m]))} "
                f"heldout={len(np.unique(split_ids[te_m]))} "
                f"windows {tr_m.sum()}/{te_m.sum()}",
                flush=True,
            )

    rep_ratio = 0.0 if no_replay else float(replay_ratio)
    recipe = FTRecipe(
        lr=lr,
        weight_decay=1e-4,
        epochs=1,
        batch_size=batch_size,
        replay_ratio=rep_ratio,
        seed=seed,
        balanced_batch=True,
    )
    replay_pool: Optional[ReplayPool] = None
    task_replay_pool: Optional[ReplayPool] = None
    if rep_ratio > 0:
        replay_pool = build_t0_replay_pool(seed=seed)
        task_replay_pool = build_t0_task_replay_pool(seed=seed)
        if replay_pool is None or task_replay_pool is None:
            raise RuntimeError(
                f"无法构建 T0 replay 池；检查 {resolve_openbmi_root(prefer_t0=True)}"
            )
        if verbose:
            print(
                f"  replay three: t0 windows={len(replay_pool.windows)} ratio={rep_ratio}",
                flush=True,
            )
            print(
                f"  replay task:  t0 windows={len(task_replay_pool.windows)} ratio={rep_ratio}",
                flush=True,
            )

    subject_id = session_dirs[0].name.split("_")[0]
    base_meta = {
        "subject_id": subject_id,
        "session_id": sess_label,
        "sessions": ds.get("sessions", [session_dirs[0].name]),
        "train_sessions": [d.name for d in session_dirs],
        "heldout_sessions": [d.name for d in hold_dirs] if leave_next else [],
        "leave_next": leave_next,
        "finetune_mode": "leave_next_ws_replay" if leave_next else "full_model_from_v3",
        "cut_protocol": resolved_protocol,
        "replay_pool": "t0" if rep_ratio > 0 else "none",
        "replay_root": str(resolve_openbmi_root(prefer_t0=True)),
        "replay_task_labels": "rest=0, mi=left+right=1",
        "win_sec": WIN_S,
        "hop_sec": HOP_S,
        "fs": FS,
        "n_windows": int(len(X)),
        "n_trials": int(ds["n_trials"][0]),
        "include_invalid": not exclude_invalid,
        "recipe": recipe.to_dict(),
        "early_stop": bool(early_stop),
        "max_epochs": int(max_epochs) if early_stop else int(epochs),
        "patience": int(patience) if early_stop else None,
        "fixed_epochs": int(epochs) if not early_stop else None,
        "deterministic": bool(deterministic),
        "seed": int(seed),
    }

    ft_kw = {
        "early_stop": early_stop,
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "fixed_epochs": int(epochs),
    }

    if verbose:
        mode = (
            f"early_stop max={max_epochs} patience={patience}"
            if early_stop
            else f"fixed_epochs={epochs}"
        )
        print(
            f"[2/4] 微调 three 头 · device={dev} · {mode} · det={deterministic}",
            flush=True,
        )
    three_final = out_dir / "best_three.pt"
    task_final = out_dir / "best_task.pt"
    three_tmp = out_dir / "best_three.pt.tmp"
    task_tmp = out_dir / "best_task.pt.tmp"

    three_rep = finetune_head(
        three_ckpt,
        X[tr_m],
        y3[tr_m],
        X[te_m],
        y3[te_m],
        n_outputs=3,
        device=dev,
        recipe=recipe,
        replay_pool=replay_pool,
        out_path=three_tmp,
        meta=base_meta,
        **ft_kw,
    )
    if verbose:
        print(
            f"  three heldout {three_rep['acc_before_heldout']:.3f} → {three_rep['acc_after_heldout']:.3f}",
            flush=True,
        )

    if verbose:
        print(f"[3/4] 微调 task 头 · device={dev}", flush=True)
    task_rep = finetune_head(
        task_ckpt,
        X[tr_m],
        y2[tr_m],
        X[te_m],
        y2[te_m],
        n_outputs=2,
        device=dev,
        recipe=recipe,
        replay_pool=task_replay_pool,
        out_path=task_tmp,
        meta=base_meta,
        **ft_kw,
    )
    if verbose:
        print(
            f"  task heldout {task_rep['acc_before_heldout']:.3f} → {task_rep['acc_after_heldout']:.3f}",
            flush=True,
        )

    release = evaluate_release_gate(three_rep)
    if verbose:
        print(f"[4/4] 发布验收 three（参考）: {'PASS' if release['pass'] else 'FAIL'}", flush=True)
        for k, ok in release["checks"].items():
            print(f"  - {k}: {'OK' if ok else 'FAIL'}", flush=True)
        print(f"  pred: {release.get('pred_labels')}", flush=True)

    import os

    def _commit_ckpt(tmp: Path, final: Path) -> None:
        """训练快照始终落盘到本轮 ft_runs；门控只作晋升参考，不删权重。"""
        if not tmp.is_file():
            return
        prev = final.with_suffix(final.suffix + ".prev")
        if final.is_file():
            try:
                os.replace(final, prev)
            except OSError:
                pass
        os.replace(tmp, final)

    _commit_ckpt(three_tmp, three_final)
    _commit_ckpt(task_tmp, task_final)
    three_rep["out"] = str(three_final)
    task_rep["out"] = str(task_final)

    report = _write_ft_report(
        out_dir,
        subject_id=subject_id,
        sess_label=sess_label,
        session_dirs=all_dirs if leave_next else session_dirs,
        ds=ds,
        X=X,
        y3=y3,
        three_rep=three_rep,
        task_rep=task_rep,
        release=release,
        args_task_ckpt=task_ckpt,
        args_three_ckpt=three_ckpt,
        train_session_dirs=session_dirs,
        heldout_session_dirs=hold_dirs if leave_next else None,
        leave_next=leave_next,
        replay_pool="t0" if rep_ratio > 0 else "none",
        replay_ratio=rep_ratio,
    )

    return {
        "status": "ok",
        "subject_id": subject_id,
        "out_dir": str(out_dir),
        "release_gate": release,
        "release_pass": bool(release.get("pass")),
        "report": report,
        "three": three_rep,
        "task": task_rep,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--session",
        type=Path,
        action="append",
        required=True,
        help="训练用 v3 会话目录；可重复传入以合并多 session",
    )
    ap.add_argument(
        "--heldout-session",
        type=Path,
        action="append",
        default=None,
        help="Leave-Next：heldout/测试会话目录（可重复）；提供后不再按试次 7:3 切",
    )
    ap.add_argument("--task-ckpt", type=Path, default=DEFAULT_TASK)
    ap.add_argument("--three-ckpt", type=Path, default=DEFAULT_THREE)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=DEFAULT_FT_EPOCHS_FIXED, help="关闭早停时的固定 epoch 数")
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_FT_MAX_EPOCHS, help="早停上限 epoch")
    ap.add_argument("--patience", type=int, default=DEFAULT_FT_PATIENCE, help="早停 patience（heldout acc）")
    ap.add_argument("--no-early-stop", action="store_true", help="关闭 heldout 早停，使用 --epochs")
    ap.add_argument("--no-deterministic", action="store_true", help="不固定随机种子（更快但重复 FT 可能波动）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument(
        "--replay-ratio",
        type=float,
        default=DEFAULT_REPLAY_RATIO,
        help=f"OpenBMI replay 比例（相对 fnz 训练窗；默认 {DEFAULT_REPLAY_RATIO}）",
    )
    ap.add_argument("--no-replay", action="store_true", help="关闭 OpenBMI replay")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--exclude-invalid",
        action="store_true",
        help="排除 trial_table.invalid=1（默认保留，用协议标签训练）",
    )
    ap.add_argument(
        "--protocol",
        choices=("auto", PROTOCOL_OPENBMI_ALIGN, PROTOCOL_LEGACY_V3),
        default="auto",
        help="切窗协议：auto=按 session 检测；openbmi_align=OpenBMI 同构；legacy_v3=历史 v3",
    )
    ap.add_argument(
        "--force-publish",
        action="store_true",
        help="验收未通过仍写入 best_*.pt（默认不通过则不替换）",
    )
    ap.add_argument(
        "--promote",
        action="store_true",
        help="FT 完成后 promote 到 subjects/{id}/models/current/",
    )
    args = ap.parse_args()

    session_dirs = [p.resolve() for p in args.session]
    hold_dirs = [p.resolve() for p in (args.heldout_session or [])]
    if len(session_dirs) == 1 and not hold_dirs:
        out_dir = (args.out_dir or (session_dirs[0] / "subject_ft")).resolve()
    else:
        subject_id = session_dirs[0].name.split("_")[0]
        out_dir = (args.out_dir or (_ROOT / "data" / "models" / subject_id)).resolve()

    result = run_subject_finetune(
        session_dirs,
        out_dir,
        task_ckpt=args.task_ckpt,
        three_ckpt=args.three_ckpt,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        train_frac=args.train_frac,
        replay_ratio=args.replay_ratio,
        no_replay=args.no_replay,
        seed=args.seed,
        device=args.device,
        exclude_invalid=args.exclude_invalid,
        protocol=args.protocol,
        verbose=True,
        early_stop=not args.no_early_stop,
        max_epochs=args.max_epochs,
        patience=args.patience,
        deterministic=not args.no_deterministic,
        heldout_session_dirs=hold_dirs or None,
    )

    release = result["release_gate"]
    if not release["pass"] and not args.force_publish and not args.promote:
        print(
            "  门控未通过（结果已写入 out_dir）；加 --promote 或 --force-publish 晋升 current",
            flush=True,
        )

    if args.promote or (args.force_publish and release["pass"]):
        from experiment_game.experiment.subject_registry import promote_ft_to_current

        sid = result["subject_id"]
        prom = promote_ft_to_current(sid, out_dir, repo_root=_REPO)
        print(f"  已晋升 current: {prom['current_dir']}", flush=True)

    print(f"已写入 {out_dir}")
    print((out_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
