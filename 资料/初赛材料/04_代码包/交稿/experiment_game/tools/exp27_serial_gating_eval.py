"""实验 27 延伸：task 串行门控 vs 纯 three 在线评测。

在 exp27 各臂 three 权重基础上，训练匹配的 task 头（同 session / replay 配方），
于下一场 session 上 sweep task_p_on，比较试次级多数票准确率。

用法:
  python experiment_game/tools/exp27_serial_gating_eval.py
  python experiment_game/tools/exp27_serial_gating_eval.py --arms BASE,S_B2,M_B1,CURRENT
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.ft import FTRecipe, IncrementalFinetuner
from adapt_engine.readout import serial_gating
from adapt_engine.registry import load_head
from experiment_game.tools.exp27_fnz_replay_grid import (
    ARMS,
    ArmSpec,
    OUT_DIR,
    SEED,
    TRACKS,
    build_replay_pool,
    finetune_arm,
    freeze_head_only,
)
from experiment_game.tools.ft_subject_from_v3 import (
    DEFAULT_TASK,
    DEFAULT_THREE,
    N_TIMES,
    _eval_acc,
    _trial_split,
    build_dataset,
)
from experiment_game.tools.openbmi_replay_pool import (
    build_task_replay_pool,
    build_t0_task_replay_pool,
    three_labels_to_task,
)

WS02 = _REPO / "experiment_game/data/sessions/fnz_ws02_20260826_171537"
WS03 = _REPO / "experiment_game/data/sessions/fnz_ws03_20260826_174526"
FNZ_CURRENT = _REPO / "experiment_game/data/models/fnz"
OUT_JSON = OUT_DIR / "serial_gating_eval.json"
OUT_MD = OUT_DIR / "serial_gating_eval.md"

LABELS = {0: "Rest", 1: "Left", 2: "Right"}
TASK_P_ON_GRID = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8]


def _task_replay_for_spec(spec: ArmSpec):
    if spec.replay_ratio <= 0:
        return None
    if spec.pool in ("t0", "t0_teach"):
        return build_t0_task_replay_pool(seed=SEED)
    if spec.pool == "all54":
        return build_task_replay_pool(subject_allow=None, seed=SEED, max_per_class=3000)
    if spec.pool == "ob12":
        from experiment_game.tools.openbmi_replay_pool import OPENBMI_ROOT_ALL, T0_SUBJS

        return build_task_replay_pool(
            root=OPENBMI_ROOT_ALL,
            subject_allow=T0_SUBJS,
            seed=SEED,
        )
    return build_t0_task_replay_pool(seed=SEED)


def _forward_probs(model, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    chunks: List[np.ndarray] = []
    bs = 64
    with torch.no_grad():
        for s in range(0, len(X), bs):
            xb = torch.from_numpy(X[s : s + bs]).to(device)
            try:
                logits = model(xb)
            except RuntimeError:
                logits = model(xb.unsqueeze(1))
            if logits.dim() == 3:
                logits = logits.reshape(logits.shape[0], -1)
            chunks.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _train_task_for_spec(
    spec: ArmSpec,
    sessions: List[Path],
    *,
    device: str,
    out_path: Path,
) -> Path:
    if out_path.is_file():
        return out_path

    ds = build_dataset(sessions, include_invalid=True)
    X, y3, split_ids = ds["X"], ds["y_three"], ds["split_id"]
    y2 = three_labels_to_task(y3)
    tr_m, te_m = _trial_split(split_ids, train_frac=0.7, seed=SEED)

    replay_pool = _task_replay_for_spec(spec)
    entry = load_head(DEFAULT_TASK, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    if spec.head_only:
        freeze_head_only(model)

    recipe = FTRecipe(
        lr=1e-4,
        weight_decay=1e-4,
        epochs=5,
        batch_size=32,
        replay_ratio=spec.replay_ratio,
        seed=SEED,
    )
    fin = IncrementalFinetuner(model, recipe, replay_pool=replay_pool, device=device)
    fin.train_round(X[tr_m], y2[tr_m], frozen=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "n_outputs": 2,
            "experiment": "E27_gating",
            "arm": spec.arm_id,
            "head": "task",
            "spec": spec.__dict__,
            "acc_heldout": _eval_acc(model, X[te_m], y2[te_m], device),
        },
        out_path,
    )
    return out_path


def _ensure_three_ckpt(track: str, arm_id: str, *, device: str) -> Path:
    if arm_id == "BASE":
        return DEFAULT_THREE
    ckpt = OUT_DIR / f"{track}_{arm_id}" / "best_three.pt"
    if ckpt.is_file():
        return ckpt
    spec = next(a for a in ARMS if a.arm_id == arm_id)
    sessions = TRACKS[track]["sessions"]
    ds = build_dataset(sessions, include_invalid=True)
    X, y3, split_ids = ds["X"], ds["y_three"], ds["split_id"]
    tr_m, te_m = _trial_split(split_ids, train_frac=0.7, seed=SEED)
    row = finetune_arm(X[tr_m], y3[tr_m], X[te_m], y3[te_m], spec, device=device)
    if row.get("status") != "ok":
        raise RuntimeError(f"three FT failed {track}_{arm_id}: {row}")
    replay_pool = build_replay_pool(spec.pool) if spec.replay_ratio > 0 else None
    entry = load_head(DEFAULT_THREE, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    if spec.head_only:
        freeze_head_only(model)
    recipe = FTRecipe(
        lr=1e-4,
        weight_decay=1e-4,
        epochs=5,
        batch_size=32,
        replay_ratio=spec.replay_ratio,
        seed=SEED,
    )
    fin = IncrementalFinetuner(model, recipe, replay_pool=replay_pool, device=device)
    fin.train_round(X[tr_m], y3[tr_m], frozen=False)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "n_outputs": 3, "experiment": "E27", "track": track, "arm": arm_id},
        ckpt,
    )
    return ckpt


def eval_gating_on_session(
    task_ckpt: Path,
    three_ckpt: Path,
    session: Path,
    *,
    device: str,
    task_p_on_grid: List[float],
) -> Dict[str, Any]:
    ds = build_dataset(session, include_invalid=True)
    X, y = ds["X"], ds["y_three"]
    split_ids = ds["split_id"]

    task_entry = load_head(task_ckpt, n_chans=8, n_times=N_TIMES, device=device)
    three_entry = load_head(three_ckpt, n_chans=8, n_times=N_TIMES, device=device)
    p_task = _forward_probs(task_entry.model, X, device)
    p_three = _forward_probs(three_entry.model, X, device)

    window_rows: Dict[str, Any] = {}
    trial_rows: Dict[str, Any] = {}

    for tpo in task_p_on_grid:
        preds: List[int] = []
        gated_frac = 0.0
        for i in range(len(X)):
            if tpo <= 0.0:
                pred = int(np.argmax(p_three[i]))
                gated = False
            else:
                out = serial_gating(p_task[i], p_three[i], task_p_on=tpo)
                pred = int(out["pred"])
                gated = bool(out["gated"])
            preds.append(pred)
            if gated:
                gated_frac += 1

        preds_arr = np.asarray(preds, dtype=np.int64)
        win_acc = float((preds_arr == y).mean())

        by_trial: Dict[str, List[int]] = defaultdict(list)
        by_label: Dict[str, int] = {}
        for i, sid in enumerate(split_ids):
            sid = str(sid)
            by_trial[sid].append(int(preds_arr[i]))
            by_label[sid] = int(y[i])

        trial_pred, trial_true = [], []
        for sid, ps in by_trial.items():
            cnt = Counter(ps)
            top = cnt.most_common()
            pred = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else top[0][0]
            trial_pred.append(pred)
            trial_true.append(by_label[sid])

        trial_acc = float(np.mean(np.array(trial_pred) == np.array(trial_true)))
        uniq, cnt = np.unique(trial_pred, return_counts=True)
        pred_counts = {LABELS[int(k)]: int(v) for k, v in zip(uniq, cnt)}
        max_frac = float(cnt.max() / len(trial_pred)) if trial_pred else 0.0
        n_right = int(pred_counts.get("Right", 0))

        key = f"{tpo:.1f}"
        window_rows[key] = {
            "task_p_on": tpo,
            "acc_window": win_acc,
            "gated_frac": gated_frac / max(len(X), 1),
        }
        trial_rows[key] = {
            "task_p_on": tpo,
            "acc_trial_majority": trial_acc,
            "pred_counts": pred_counts,
            "max_class_frac": max_frac,
            "n_right_pred_trials": n_right,
        }

    best_tpo = max(
        task_p_on_grid,
        key=lambda t: trial_rows[f"{t:.1f}"]["acc_trial_majority"],
    )
    baseline = trial_rows["0.0"]["acc_trial_majority"]
    best_acc = trial_rows[f"{best_tpo:.1f}"]["acc_trial_majority"]

    return {
        "session": session.name,
        "n_windows": int(len(X)),
        "n_trials": len(set(map(str, split_ids))),
        "task_ckpt": str(task_ckpt),
        "three_ckpt": str(three_ckpt),
        "by_task_p_on": trial_rows,
        "window_by_task_p_on": window_rows,
        "baseline_three_only_trial_acc": baseline,
        "best_task_p_on": best_tpo,
        "best_gated_trial_acc": best_acc,
        "delta_vs_three_only": best_acc - baseline,
    }


def _resolve_scenario(name: str, *, device: str) -> Tuple[Path, Path, Path, str]:
    """返回 (task_ckpt, three_ckpt, test_session, description)。"""
    if name == "BASE":
        return DEFAULT_TASK, DEFAULT_THREE, WS02, "OpenBMI 底座 → ws02"

    if name == "CURRENT":
        return (
            FNZ_CURRENT / "best_task.pt",
            FNZ_CURRENT / "best_three.pt",
            WS03,
            "fnz current (ws01+ws02 FT t0+0.10) → ws03",
        )

    if name.startswith("S_"):
        arm = name.split("_", 1)[1]
        spec = next(a for a in ARMS if a.arm_id == arm)
        d = OUT_DIR / f"S_{arm}"
        three = _ensure_three_ckpt("S", arm, device=device)
        task = _train_task_for_spec(spec, TRACKS["S"]["sessions"], device=device, out_path=d / "best_task.pt")
        return task, three, WS02, f"ws01 FT {arm} → ws02"

    if name.startswith("M_"):
        arm = name.split("_", 1)[1]
        spec = next(a for a in ARMS if a.arm_id == arm)
        d = OUT_DIR / f"M_{arm}"
        three = _ensure_three_ckpt("M", arm, device=device)
        task = _train_task_for_spec(spec, TRACKS["M"]["sessions"], device=device, out_path=d / "best_task.pt")
        return task, three, WS03, f"ws01+ws02 FT {arm} → ws03"

    raise ValueError(name)


def write_md(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 实验 27 · task 串行门控评测",
        "",
        f"生成：{payload['generated_at']}",
        "",
        "口径：下一场 session 全窗推理；pred = serial_gating(p_task, p_three, task_p_on)；"
        "报告 **试次级多数票** acc。task_p_on=0 等价于纯 three argmax。",
        "",
    ]
    for row in payload["scenarios"]:
        lines += [
            f"## {row['scenario']} · {row['description']}",
            "",
            f"- baseline (three only): **{row['baseline_three_only_trial_acc']:.3f}**",
            f"- best gated: **{row['best_gated_trial_acc']:.3f}** @ task_p_on={row['best_task_p_on']:.1f} "
            f"(Δ={row['delta_vs_three_only']:+.3f})",
            "",
            "| task_p_on | trial acc | pred Rest/L/R | max_frac | gated窗占比 |",
            "|-----------|-----------|---------------|----------|-------------|",
        ]
        for tpo in TASK_P_ON_GRID:
            k = f"{tpo:.1f}"
            tr = row["by_task_p_on"][k]
            wr = row["window_by_task_p_on"][k]
            pc = tr["pred_counts"]
            lines.append(
                f"| {tpo:.1f} | {tr['acc_trial_majority']:.3f} | "
                f"{pc.get('Rest', 0)}/{pc.get('Left', 0)}/{pc.get('Right', 0)} | "
                f"{tr['max_class_frac']:.2f} | {wr['gated_frac']:.2f} |"
            )
        lines.append("")

    lines += [
        "## 汇总",
        "",
        "| scenario | three only | best gated | Δ | best task_p_on | 结论 |",
        "|----------|------------|------------|---|----------------|------|",
    ]
    for row in payload["scenarios"]:
        delta = row["delta_vs_three_only"]
        verdict = "有帮助" if delta > 0.02 else ("略有害" if delta < -0.02 else "基本无差")
        lines.append(
            f"| {row['scenario']} | {row['baseline_three_only_trial_acc']:.3f} | "
            f"{row['best_gated_trial_acc']:.3f} | {delta:+.3f} | {row['best_task_p_on']:.1f} | {verdict} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="E27 serial gating eval")
    ap.add_argument(
        "--arms",
        default="BASE,S_B2,S_B3,S_A1,M_B2,M_B1,M_E1,CURRENT",
        help="逗号分隔场景名",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenarios: List[Dict[str, Any]] = []

    for name in [x.strip() for x in args.arms.split(",") if x.strip()]:
        print(f"\n=== {name} ===", flush=True)
        task_ckpt, three_ckpt, session, desc = _resolve_scenario(name, device=device)
        print(f"  task={task_ckpt.name} three={three_ckpt.parent.name}/{three_ckpt.name}", flush=True)
        rep = eval_gating_on_session(
            task_ckpt,
            three_ckpt,
            session,
            device=device,
            task_p_on_grid=TASK_P_ON_GRID,
        )
        rep["scenario"] = name
        rep["description"] = desc
        scenarios.append(rep)
        print(
            f"  three_only={rep['baseline_three_only_trial_acc']:.3f} "
            f"best={rep['best_gated_trial_acc']:.3f} @ {rep['best_task_p_on']:.1f} "
            f"Δ={rep['delta_vs_three_only']:+.3f}",
            flush=True,
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "task_p_on_grid": TASK_P_ON_GRID,
        "scenarios": scenarios,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(payload, OUT_MD)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
