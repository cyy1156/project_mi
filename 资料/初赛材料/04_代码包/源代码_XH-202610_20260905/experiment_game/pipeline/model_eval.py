"""历史模型 × 会话 识别结果网格（需求 2026-08-30 二.4）。

对被试留存的每套权重（current + models/ft_runs/<stamp>）× 每个 v3 会话：
离线切窗（与训练同构 openbmi_align）→ E1f 三分类推理 →
窗级因果平滑（对齐在线 acc_window）/ 试次级因果平滑+多数票（对齐 F5）；
另附窗级 raw 字段便于诊断。
CPU 推理，按模型依次构建、复用评所有会话。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from adapt_engine.e1f import E1fRegistry, E1fStackConfig

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_STACK = "experiment_game/config/e1f_four_member.json"


def _load_overlay_members(overlay_path: Path) -> Dict[str, Any]:
    if not overlay_path.is_file():
        return {}
    try:
        blob = json.loads(overlay_path.read_text(encoding="utf-8"))
        members = blob.get("members") or {}
        return members if isinstance(members, dict) else {}
    except Exception:
        return {}


def _resolve_members_from_root(
    root: Path,
    stack: E1fStackConfig,
) -> Dict[str, Dict[str, Any]]:
    """权重根目录解析：subject models 目录（current/…）或 ft_run 目录（members/…）。"""
    if (root / "current").is_dir():
        cur = root / "current"
        overlay = _load_overlay_members(cur / "e1f_overlay.json")
        members_dir = cur / "members"
        task_fallbacks = (members_dir / "shallow" / "best_task.pt", cur / "best_task.pt")
    else:
        overlay = _load_overlay_members(root / "e1f_overlay.json")
        members_dir = root / "members"
        task_fallbacks = (members_dir / "shallow" / "best_task.pt", root / "best_task.pt")

    out: Dict[str, Dict[str, Any]] = {}
    for m in stack.members:
        three = Path(m.three_ckpt)
        task: Any = Path(m.task_ckpt) if m.task_ckpt else None
        ov = overlay.get(m.name) or {}
        if ov.get("three_ckpt"):
            p = Path(str(ov["three_ckpt"]))
            p = p if p.is_absolute() else (_REPO / p).resolve()
            if p.is_file():
                three = p
        else:
            local = members_dir / m.name / "best_three.pt"
            if local.is_file():
                three = local
        if m.name == "shallow":
            hit = None
            if ov.get("task_ckpt"):
                p = Path(str(ov["task_ckpt"]))
                p = p if p.is_absolute() else (_REPO / p).resolve()
                if p.is_file():
                    hit = p
            if hit is None:
                for cand in task_fallbacks:
                    if cand.is_file():
                        hit = cand
                        break
            task = hit if hit is not None else task
        elif ov.get("task_ckpt"):
            p = Path(str(ov["task_ckpt"]))
            p = p if p.is_absolute() else (_REPO / p).resolve()
            if p.is_file():
                task = p
        out[m.name] = {"three": three, "task": task}
    return out


def _build_registry_from_root(
    root: Path,
    *,
    e1f_config_path: str = _DEFAULT_STACK,
    device: str = "cpu",
) -> E1fRegistry:
    stack = E1fStackConfig.load_json(e1f_config_path, repo_root=_REPO).resolve_paths(
        repo_root=_REPO
    )
    resolved = _resolve_members_from_root(root, stack)
    overrides: Dict[str, Dict[str, Any]] = {}
    for m in stack.members:
        r = resolved.get(m.name) or {}
        three = r.get("three")
        if three is None or not Path(three).is_file():
            raise FileNotFoundError(f"[{m.name}] 缺 three 权重: {three}")
        task = r.get("task")
        overrides[m.name] = {
            "three_ckpt": str(Path(three).resolve()),
            "task_ckpt": (
                str(Path(task).resolve()) if task and Path(task).is_file() else None
            ),
        }
    stack = stack.with_member_overrides(overrides).resolve_paths(repo_root=_REPO)
    return E1fRegistry(stack, device=device)


def list_weight_sets(subject_models_dir: Path | str) -> List[Dict[str, Any]]:
    """current + 各 ft_run（含 e1f_overlay.json 或 members/ 的）。"""
    root = Path(subject_models_dir)
    out: List[Dict[str, Any]] = []
    if (root / "current").is_dir():
        out.append({"id": "current", "label": "current（v3 最终权重）", "root": str(root)})
    runs = root / "ft_runs"
    if runs.is_dir():
        for d in sorted(runs.iterdir()):
            if d.is_dir() and (
                (d / "e1f_overlay.json").is_file() or (d / "members").is_dir()
            ):
                out.append({"id": d.name, "label": f"ft_run {d.name}", "root": str(d)})
    return out


def _window_acc_causal(
    probs: np.ndarray,
    y: np.ndarray,
    split_ids: np.ndarray,
    *,
    lookback: int = 2,
    lr_only: bool = False,
) -> Optional[float]:
    """窗级因果平滑 acc（对齐在线 acc_window / 方案 A）。"""
    from adapt_engine.readout import causal_smooth_pred_sequence

    if len(y) == 0:
        return None
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, sid in enumerate(split_ids):
        groups[str(sid)].append(i)
    pred = np.zeros(len(y), dtype=np.int64)
    for idxs in groups.values():
        seq = [np.asarray(probs[i], dtype=np.float32) for i in idxs]
        smoothed = causal_smooth_pred_sequence(seq, lookback=lookback)
        for j, ix in enumerate(idxs):
            pred[ix] = int(smoothed[j]["pred"])
    mask = (y != 0) if lr_only else np.ones(len(y), dtype=bool)
    if not mask.any():
        return None
    return float((pred[mask] == y[mask]).mean())


def _majority_by_trial(
    preds: np.ndarray,
    y: np.ndarray,
    split_ids: np.ndarray,
    probs: np.ndarray | None = None,
    *,
    causal_lookback: int = 2,
) -> Dict[str, float]:
    """试次级多数票；默认因果平滑后投票（对齐在线 F5 / 方案 A）。"""
    from adapt_engine.readout import causal_smooth_pred_sequence
    from experiment_game.experiment.judge_aggregate import majority_pred_from_votes

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, sid in enumerate(split_ids):
        groups[str(sid)].append(i)
    n_correct = 0
    n = 0
    for _sid, idxs in groups.items():
        labels = y[idxs]
        idx_arr = np.asarray(idxs, dtype=int)
        if probs is not None and len(idx_arr) > 0:
            seq = [np.asarray(probs[i], dtype=np.float32) for i in idx_arr]
            smoothed = causal_smooth_pred_sequence(seq, lookback=causal_lookback)
            preds_use = np.asarray([int(s["pred"]) for s in smoothed], dtype=np.int64)
            p_slice = np.stack(
                [np.asarray(s["p_three"], dtype=np.float64) for s in smoothed],
                axis=0,
            )
        else:
            preds_use = preds[idx_arr]
            p_slice = None
        pred = majority_pred_from_votes(preds_use, p_slice)
        n_correct += int(pred == int(labels[0]))
        n += 1
    return {"acc_trial_majority": round(n_correct / n, 4) if n else None, "n_trials": n}


def evaluate_weight_set(
    weight_root: Path | str,
    session_dirs: List[Path],
    *,
    e1f_config_path: str = _DEFAULT_STACK,
    device: str = "cpu",
    on_progress: Any = None,
) -> Dict[str, Any]:
    """一套权重 × 多个会话 → 每会话窗级/试次级准确率。"""
    from experiment_game.pipeline.finetune import build_dataset

    root = Path(weight_root)
    reg = _build_registry_from_root(root, e1f_config_path=e1f_config_path, device=device)
    rows: List[Dict[str, Any]] = []
    for sd in session_dirs:
        try:
            ds = build_dataset(sd.resolve(), protocol="auto")
            X, y, split_ids = ds["X"], np.asarray(ds["y_three"]), np.asarray(ds["split_id"])
            if len(X) == 0:
                rows.append({"session": sd.name, "n_windows": 0, "error": "no_windows"})
                continue
            probs = reg.forward_three_batch(X)
            preds = probs.argmax(axis=1).astype(np.int64)
            y = y.astype(np.int64)
            # 窗级：展示用因果平滑（对齐在线 acc_window）；另附 raw 便于诊断
            acc_lr = _window_acc_causal(probs, y, split_ids, lr_only=True)
            acc_all = _window_acc_causal(probs, y, split_ids, lr_only=False)
            lr = y != 0
            acc_lr_raw = (
                float((preds[lr] == y[lr]).mean()) if lr.any() else None
            )
            acc_all_raw = float((preds == y).mean())
            maj = _majority_by_trial(preds, y, split_ids, probs)
            rows.append({
                "session": sd.name,
                "n_windows": int(len(X)),
                "acc_window_lr": round(acc_lr, 4) if acc_lr is not None else None,
                "acc_window_all": round(acc_all, 4) if acc_all is not None else None,
                "acc_window_lr_raw": round(acc_lr_raw, 4) if acc_lr_raw is not None else None,
                "acc_window_all_raw": round(acc_all_raw, 4),
                **maj,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"session": sd.name, "error": str(exc)[:160]})
        if on_progress is not None:
            try:
                on_progress(sd.name, len(rows))
            except Exception:
                pass
    return {"weight_set": root.name if root.name != "models" else "current", "root": str(root), "rows": rows}


def evaluate_model_grid(
    subject_models_dir: Path | str,
    session_dirs: List[Path],
    *,
    e1f_config_path: str = _DEFAULT_STACK,
    device: str = "cpu",
    on_progress: Any = None,
) -> Dict[str, Any]:
    """全部权重套 × 全部会话。返回 {models:[…], rows:[…]}。"""
    sets = list_weight_sets(subject_models_dir)
    results: List[Dict[str, Any]] = []
    total = len(sets)
    for i, ws in enumerate(sets):
        if on_progress is not None:
            try:
                on_progress(f"model {i + 1}/{total}: {ws['id']}", i)
            except Exception:
                pass
        try:
            res = evaluate_weight_set(
                ws["root"],
                session_dirs,
                e1f_config_path=e1f_config_path,
                device=device,
            )
            res["id"] = ws["id"]
            res["label"] = ws["label"]
        except Exception as exc:  # noqa: BLE001
            res = {
                "id": ws["id"],
                "label": ws["label"],
                "rows": [
                    {"session": sd.name, "error": str(exc)[:160]} for sd in session_dirs
                ],
            }
        results.append(res)
    return {"models": results, "n_weight_sets": total, "n_sessions": len(session_dirs)}
