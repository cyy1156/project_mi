"""Leave-Next 爬坡 + F5 试次级读出（因果平滑 + 多数票）。

沿用 2026-08-28 模式：
  syj0828 · 仅 v3 ws01–ws06（排除 v4 ws01_124816）
  fnz0828 · 仅 v3 ws02–ws06（排除 v4 ws01、电极异常 ws07）
  R1–R3 replay=0.1；R4–R5 --no-replay（fnz 至 R4）

每档：
  1) 前序 session FT shallow task+three（heldout 早停）
  2) heldout 上报告窗级 three acc（门控）
  3) heldout 上按 F5 试次级：MI / Rest 因果平滑+多数票

对照：同 heldout 上底座 three、以及 E1f 四成员零样本的 F5 MI acc。

用法:
  python experiment_game/tools/run_leave_next_e1f_task_ramp.py --subject syj0828
  python experiment_game/tools/run_leave_next_e1f_task_ramp.py --subject fnz0828
  python experiment_game/tools/run_leave_next_e1f_task_ramp.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.e1f import E1fRegistry, E1fStackConfig  # noqa: E402
from adapt_engine.registry import load_head  # noqa: E402
from experiment_game.experiment.judge_aggregate import (  # noqa: E402
    primary_judge_from_judgments,
)
from experiment_game.experiment.trial_scoring import (  # noqa: E402
    CAUSAL_LOOKBACK,
    PRE_CUE_REST_POINTS,
)
from experiment_game.tools.ft_subject_from_v3 import (  # noqa: E402
    DEFAULT_TASK,
    DEFAULT_THREE,
    N_TIMES,
    _eval_acc,
    build_dataset,
    run_subject_finetune,
)

SUBJECTS_ROOT = _REPO / "experiment_game" / "data" / "subjects"
E1F_CONFIG = _REPO / "experiment_game" / "config" / "e1f_four_member.json"
LABEL_NAMES = {0: "Rest", 1: "Left", 2: "Right"}

# (train_ws_keys, heldout_ws_key, use_replay)
RAMP_SYJ = [
    (["ws01"], "ws02", True),
    (["ws01", "ws02"], "ws03", True),
    (["ws01", "ws02", "ws03"], "ws04", True),
    (["ws01", "ws02", "ws03", "ws04"], "ws05", False),
    (["ws01", "ws02", "ws03", "ws04", "ws05"], "ws06", False),
]

RAMP_FNZ = [
    (["ws02"], "ws03", True),
    (["ws02", "ws03"], "ws04", True),
    (["ws02", "ws03", "ws04"], "ws05", True),
    (["ws02", "ws03", "ws04", "ws05"], "ws06", False),
]

# cyy0830 / fnz0830 / wzr0830 / xj0830 · v3 w01–w06
RAMP_W = [
    (["w01"], "w02", True),
    (["w01", "w02"], "w03", True),
    (["w01", "w02", "w03"], "w04", True),
    (["w01", "w02", "w03", "w04"], "w05", False),
    (["w01", "w02", "w03", "w04", "w05"], "w06", False),
]

# 兼容旧名
RAMP_CYY = RAMP_W
RAMP_FNZ0830 = RAMP_W

SUBJECTS_W = ("cyy0830", "fnz0830", "wzr0830", "xj0830")
SUBJECTS_ALL = ("syj0828", "fnz0828") + SUBJECTS_W


def _ramp_for_subject(subject_id: str, by_ws: Dict[str, Path]) -> list:
    if subject_id == "syj0828":
        cand = list(RAMP_SYJ)
    elif subject_id == "fnz0828":
        cand = list(RAMP_FNZ)
    elif subject_id in SUBJECTS_W:
        cand = list(RAMP_W)
    else:
        raise ValueError(f"未知被试: {subject_id}")
    out = []
    for train_keys, hold_key, use_replay in cand:
        need = list(train_keys) + [hold_key]
        miss = [k for k in need if k not in by_ws]
        if miss:
            print(f"  [skip] R train={train_keys} hold={hold_key} 缺 {miss}")
            continue
        out.append((train_keys, hold_key, use_replay))
    return out


def _session_key_from_dirname(name: str) -> Optional[str]:
    """从目录名解析 wsNN / wNN。"""
    for part in name.split("_"):
        p = part.lower()
        if p.startswith("ws") and p[2:].isdigit():
            return p
        if p.startswith("w") and not p.startswith("ws") and p[1:].isdigit():
            return p
    return None


def _list_v3_sessions(subject_id: str) -> Dict[str, Path]:
    """session_id(wsNN|wNN) -> 最新一条 v3 目录（排除 record_excluded / 非 v3）。"""
    root = SUBJECTS_ROOT / subject_id / "sessions"
    idx_path = SUBJECTS_ROOT / subject_id / "index.json"
    exclude: set[str] = set()
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        for s in idx.get("sessions") or []:
            if s.get("record_excluded") or s.get("phase_mode") != "v3_session":
                exclude.add(str(s.get("dir") or ""))
    by_ws: Dict[str, Path] = {}
    for d in sorted(root.iterdir() if root.is_dir() else []):
        if not d.is_dir():
            continue
        if d.name in exclude:
            continue
        if subject_id == "syj0828" and "124816" in d.name:
            continue
        if subject_id == "fnz0828" and (
            d.name.endswith("_152231") or "ws07" in d.name
        ):
            continue
        meta = d / "session.meta.json"
        phase = ""
        if meta.is_file():
            try:
                phase = str(json.loads(meta.read_text(encoding="utf-8")).get("phase_mode") or "")
            except Exception:
                phase = ""
        if phase and phase != "v3_session":
            continue
        ws = _session_key_from_dirname(d.name)
        if not ws:
            continue
        prev = by_ws.get(ws)
        if prev is None or d.name > prev.name:
            by_ws[ws] = d
    return by_ws


def _tag_from_ws(keys: Sequence[str]) -> str:
    return "+".join(keys)


@torch.no_grad()
def _probs_from_three_model(model, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    bs = 64
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s : s + bs]).to(device)
        try:
            logits = model(xb)
        except RuntimeError:
            logits = model(xb.unsqueeze(1))
        if logits.dim() == 3:
            logits = logits.reshape(logits.shape[0], -1)
        outs.append(torch.softmax(logits, dim=-1).cpu().numpy()[:, :3])
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 3), dtype=np.float32)


def _f5_from_probs(
    probs: np.ndarray,
    y: np.ndarray,
    split_ids: np.ndarray,
    *,
    lookback: int = CAUSAL_LOOKBACK,
) -> Dict[str, Any]:
    """按 split_id 聚窗 → F5 因果平滑多数票；分 MI / Rest。"""
    by_trial: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    by_label: Dict[str, int] = {}
    for i, sid in enumerate(split_ids):
        key = str(sid)
        by_trial[key].append((i, probs[i]))
        if key not in by_label:
            by_label[key] = int(y[i])

    # per-label: ok / n / points（Left/Right +1；Rest +0.5）
    lab_ok = {0: 0, 1: 0, 2: 0}
    lab_n = {0: 0, 1: 0, 2: 0}
    lab_pts = {0: 0.0, 1: 0.0, 2: 0.0}
    pred_hist: Counter = Counter()
    conf: Counter = Counter()
    score = 0.0

    for sid, items in by_trial.items():
        items = sorted(items, key=lambda t: t[0])
        lab = int(by_label[sid])
        judgments = [
            {
                "t_rel": float(k) * 0.1,
                "p_three": [float(x) for x in p.ravel()[:3]],
                "pred": int(np.argmax(p[:3])),
                "p_max": float(np.max(p[:3])),
            }
            for k, (_, p) in enumerate(items)
        ]
        primary = primary_judge_from_judgments(
            judgments, mode="majority", causal_lookback=lookback
        )
        if primary is None:
            continue
        pred = int(primary["pred"])
        pred_hist[LABEL_NAMES.get(pred, str(pred))] += 1
        conf[f"y{lab}->p{pred}"] += 1
        if lab not in lab_n:
            continue
        lab_n[lab] += 1
        if pred == lab:
            lab_ok[lab] += 1
            pts = PRE_CUE_REST_POINTS if lab == 0 else 1.0
            lab_pts[lab] += pts
            score += pts

    mi_ok = lab_ok[1] + lab_ok[2]
    mi_n = lab_n[1] + lab_n[2]
    rest_ok, rest_n = lab_ok[0], lab_n[0]

    def _lab_block(lab: int) -> Dict[str, Any]:
        n = lab_n[lab]
        ok = lab_ok[lab]
        pts_each = PRE_CUE_REST_POINTS if lab == 0 else 1.0
        return {
            "label": LABEL_NAMES[lab],
            "ok": ok,
            "n": n,
            "acc": (ok / n) if n else float("nan"),
            "points": lab_pts[lab],
            "points_max": float(n) * pts_each,
        }

    by_label_out = {
        "Rest": _lab_block(0),
        "Left": _lab_block(1),
        "Right": _lab_block(2),
    }
    return {
        "rule": "causal_smooth_majority",
        "causal_lookback": int(lookback),
        "mi_ok": mi_ok,
        "mi_n": mi_n,
        "mi_acc": (mi_ok / mi_n) if mi_n else float("nan"),
        "rest_ok": rest_ok,
        "rest_n": rest_n,
        "rest_acc": (rest_ok / rest_n) if rest_n else float("nan"),
        "score": score,
        "score_max": (
            float(lab_n[1] + lab_n[2]) + float(lab_n[0]) * PRE_CUE_REST_POINTS
        ),
        "by_label": by_label_out,
        "pred_hist": dict(pred_hist),
        "confusion": dict(conf),
        "n_trials": mi_n + rest_n,
    }


def eval_f5_three_ckpt(
    ckpt: Path,
    session_dirs: Sequence[Path],
    *,
    device: str,
) -> Dict[str, Any]:
    ds = build_dataset(list(session_dirs), include_invalid=True, protocol="auto")
    X, y, split_ids = ds["X"], ds["y_three"], ds["split_id"]
    entry = load_head(ckpt, n_chans=8, n_times=N_TIMES, device=device)
    probs = _probs_from_three_model(entry.model, X, device)
    win_acc = _eval_acc(entry.model, X, y, device)
    f5 = _f5_from_probs(probs, y, split_ids)
    return {
        "source": "three_ckpt",
        "ckpt": str(ckpt),
        "n_windows": int(len(X)),
        "window_acc": float(win_acc),
        "f5": f5,
    }


def eval_f5_e1f(
    session_dirs: Sequence[Path],
    *,
    device: str,
    e1f_registry: Optional[E1fRegistry] = None,
) -> Dict[str, Any]:
    ds = build_dataset(list(session_dirs), include_invalid=True, protocol="auto")
    X, y, split_ids = ds["X"], ds["y_three"], ds["split_id"]
    reg = e1f_registry
    if reg is None:
        stack = E1fStackConfig.load_json(E1F_CONFIG, repo_root=_REPO).resolve_paths(
            repo_root=_REPO
        )
        missing = stack.missing_paths(repo_root=_REPO)
        if missing:
            raise FileNotFoundError("\n".join(missing[:6]))
        reg = E1fRegistry(stack, device=device)
    probs = reg.forward_three_batch(X)
    preds = probs.argmax(axis=1)
    win_acc = float((preds == y).mean()) if len(y) else float("nan")
    f5 = _f5_from_probs(probs, y, split_ids)
    return {
        "source": "e1f_four_member",
        "n_windows": int(len(X)),
        "window_acc": win_acc,
        "f5": f5,
    }


def run_ramp(subject_id: str, *, promote_final: bool = False) -> Path:
    by_ws = _list_v3_sessions(subject_id)
    ramp = _ramp_for_subject(subject_id, by_ws)
    if not ramp:
        raise ValueError(f"{subject_id}: 无可用 Leave-Next 档（sessions={sorted(by_ws)}）")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ft_root = SUBJECTS_ROOT / subject_id / "models" / "ft_runs"
    ft_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert DEFAULT_TASK.is_file(), f"缺 task 底座: {DEFAULT_TASK}"
    assert DEFAULT_THREE.is_file(), f"缺 three 底座: {DEFAULT_THREE}"
    print(f"[{subject_id}] device={device}")
    print(f"[{subject_id}] E1f shallow base task={DEFAULT_TASK}")
    print(f"[{subject_id}] E1f shallow base three={DEFAULT_THREE}")
    print(f"[{subject_id}] v3 sessions: { {k: v.name for k, v in sorted(by_ws.items())} }")
    print(f"[{subject_id}] Leave-Next 档数={len(ramp)}")
    print(f"[{subject_id}] F5 = causal_smooth(lookback={CAUSAL_LOOKBACK}) + majority")

    e1f_reg: Optional[E1fRegistry] = None
    try:
        stack = E1fStackConfig.load_json(E1F_CONFIG, repo_root=_REPO).resolve_paths(
            repo_root=_REPO
        )
        miss = stack.missing_paths(repo_root=_REPO)
        if not miss:
            e1f_reg = E1fRegistry(stack, device=device)
            print(f"[{subject_id}] E1f 四成员对照已加载")
        else:
            print(f"[{subject_id}] 跳过 E1f 对照（缺权重）: {miss[0]}")
    except Exception as exc:
        print(f"[{subject_id}] 跳过 E1f 对照: {exc}")

    summary: List[Dict[str, Any]] = []
    last_out: Path | None = None

    for i, (train_keys, hold_key, use_replay) in enumerate(ramp, start=1):
        missing = [k for k in list(train_keys) + [hold_key] if k not in by_ws]
        if missing:
            print(f"  [skip] 缺 session {missing}")
            continue
        train_dirs = [by_ws[k] for k in train_keys]
        hold_dirs = [by_ws[hold_key]]
        tag = f"leave_next_{_tag_from_ws(train_keys)}_eval_{hold_key}"
        if not use_replay:
            tag += "_noreplay"
        out_dir = ft_root / f"{stamp}_{tag}"
        print(f"\n=== R{i} {subject_id} train={train_keys} hold={hold_key} replay={use_replay} ===")
        print(f"  out={out_dir}")

        # 底座对照（FT 前）
        print("  [F5] 底座 three → heldout …")
        base_three = eval_f5_three_ckpt(DEFAULT_THREE, hold_dirs, device=device)
        base_e1f = None
        if e1f_reg is not None:
            print("  [F5] E1f 四成员 → heldout …")
            base_e1f = eval_f5_e1f(hold_dirs, device=device, e1f_registry=e1f_reg)

        result = run_subject_finetune(
            train_dirs,
            out_dir,
            task_ckpt=DEFAULT_TASK,
            three_ckpt=DEFAULT_THREE,
            heldout_session_dirs=hold_dirs,
            replay_ratio=0.1 if use_replay else 0.0,
            no_replay=not use_replay,
            early_stop=True,
            verbose=True,
            device=device,
        )
        three = result.get("three") or {}
        release = result.get("release_gate") or {}

        ft_three_path = out_dir / "best_three.pt"
        print("  [F5] FT three → heldout …")
        ft_f5 = eval_f5_three_ckpt(ft_three_path, hold_dirs, device=device)

        def _pack(tag_name: str, blob: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if blob is None:
                return None
            f5 = blob["f5"]
            return {
                "tag": tag_name,
                "window_acc": blob.get("window_acc"),
                "mi_acc": f5.get("mi_acc"),
                "mi_ok": f5.get("mi_ok"),
                "mi_n": f5.get("mi_n"),
                "rest_acc": f5.get("rest_acc"),
                "rest_ok": f5.get("rest_ok"),
                "rest_n": f5.get("rest_n"),
                "score": f5.get("score"),
                "score_max": f5.get("score_max"),
                "by_label": f5.get("by_label"),
                "pred_hist": f5.get("pred_hist"),
                "confusion": f5.get("confusion"),
            }

        row = {
            "r_stage": i,
            "protocol": "leave_next_f5",
            "judge_rule": "causal_smooth_majority",
            "causal_lookback": CAUSAL_LOOKBACK,
            "train": [d.name for d in train_dirs],
            "heldout": hold_dirs[0].name,
            "use_replay": use_replay,
            "replay_ratio": 0.1 if use_replay else 0.0,
            "out_dir": str(out_dir),
            "release_pass": bool(result.get("release_pass")),
            "heldout_acc": three.get("acc_after_heldout"),
            "train_acc": three.get("acc_after_train"),
            "max_class_frac": (three.get("heldout_pred_dist") or {}).get("max_class_frac"),
            "pred_labels": (release.get("pred_labels") if isinstance(release, dict) else None),
            "train_minus_heldout": None,
            "task_heldout_acc": (result.get("task") or {}).get("acc_after_heldout"),
            "base_task_ckpt": str(DEFAULT_TASK),
            "base_three_ckpt": str(DEFAULT_THREE),
            "f5_ft": _pack("ft_three", ft_f5),
            "f5_base_three": _pack("base_three", base_three),
            "f5_base_e1f": _pack("base_e1f", base_e1f),
        }
        if row["train_acc"] is not None and row["heldout_acc"] is not None:
            row["train_minus_heldout"] = float(row["train_acc"]) - float(row["heldout_acc"])
        if isinstance(release, dict) and release.get("checks"):
            row["checks"] = release["checks"]
        summary.append(row)
        last_out = out_dir

        def _lab_line(pack: Optional[Dict[str, Any]]) -> str:
            if not pack or not pack.get("by_label"):
                return "n/a"
            bl = pack["by_label"]
            parts = []
            for name in ("Left", "Right", "Rest"):
                b = bl.get(name) or {}
                n = int(b.get("n") or 0)
                ok = int(b.get("ok") or 0)
                pts = float(b.get("points") or 0.0)
                pmax = float(b.get("points_max") or 0.0)
                acc = (ok / n) if n else float("nan")
                parts.append(f"{name} {ok}/{n}={acc:.1%} pts={pts:.1f}/{pmax:.1f}")
            return " | ".join(parts)

        ft = row["f5_ft"]
        print(
            f"  → window heldout={row['heldout_acc']:.3f} PASS={row['release_pass']} | "
            f"F5 total={ft['score']:.1f}/{ft.get('score_max', 54):.1f}"
        )
        print(f"     FT  {_lab_line(ft)}")
        print(f"     base3 {_lab_line(row.get('f5_base_three'))}")
        if row.get("f5_base_e1f"):
            print(f"     e1f  {_lab_line(row.get('f5_base_e1f'))}")

    sum_path = ft_root / f"{stamp}_{subject_id}_e1f_task_leave_next_f5_summary.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "subject_id": subject_id,
        "protocol": "leave_next_f5",
        "judge_rule": "causal_smooth_majority",
        "causal_lookback": CAUSAL_LOOKBACK,
        "device": device,
        "rows": summary,
    }
    sum_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[{subject_id}] summary → {sum_path}")

    if promote_final and last_out is not None:
        from experiment_game.experiment.subject_registry import promote_ft_to_current

        promote_ft_to_current(subject_id, last_out, repo_root=_REPO)
        print(f"[{subject_id}] promoted {last_out} → models/current/")

    return sum_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--subject",
        choices=SUBJECTS_ALL,
        action="append",
    )
    ap.add_argument("--all", action="store_true", help="syj0828+fnz0828")
    ap.add_argument(
        "--cohort-0828-0830",
        action="store_true",
        help="syj0828 fnz0828 + 全部 *0830 真被试",
    )
    ap.add_argument(
        "--promote-final",
        action="store_true",
        help="最后一档 R 晋升到 models/current（默认只写 ft_runs）",
    )
    args = ap.parse_args()
    subjects = list(args.subject or [])
    if args.cohort_0828_0830:
        subjects = list(SUBJECTS_ALL)
    elif args.all or not subjects:
        subjects = ["syj0828", "fnz0828"]
    for sid in subjects:
        run_ramp(sid, promote_final=bool(args.promote_final))


if __name__ == "__main__":
    main()
