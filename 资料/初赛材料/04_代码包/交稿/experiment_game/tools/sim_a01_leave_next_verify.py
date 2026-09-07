"""A01 Leave-Next Ramp 重验：用修正后的 sim_mat_cue 切窗 + Exp29 B2 策略。

协议（与 ramp.py / 操作建议一致）:
  R0: eval=run3 · T0 · 不 FT
  Rk (k>=1): train=queue[:k] · eval=queue[k] · 每次从 T0 重训
  replay: R1–R3 → 0.10；R≥4 → 关

用法:
  python experiment_game/tools/sim_a01_leave_next_verify.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.registry import load_head  # noqa: E402
from experiment_game.experiment.sim.campaign import load_campaign  # noqa: E402
from experiment_game.experiment.sim.ramp import ft_replay_recommendation  # noqa: E402
from experiment_game.tools.exp27_cross_session_eval import trial_majority_acc  # noqa: E402
from experiment_game.tools.ft_subject_from_v3 import (  # noqa: E402
    DEFAULT_THREE,
    N_TIMES,
    build_dataset,
    run_subject_finetune,
)

DEFAULT_MANIFEST = (
    _REPO
    / "experiment_game/data/sim_subjects/A01/records/campaigns"
    / "sim_20260827_222049/manifest.json"
)
OUT_ROOT = _REPO / "experiment_game/data/sim_subjects/A01/records/leave_next_verify"


def _queue_sessions(manifest: Dict[str, Any]) -> List[tuple[str, Path]]:
    done = {
        str(x.get("run_id") or "").lower(): Path(x["session_dir"])
        for x in (manifest.get("sessions_completed") or [])
        if x.get("run_id") and x.get("session_dir")
    }
    out: List[tuple[str, Path]] = []
    for rid in manifest.get("session_queue") or []:
        rid = str(rid).lower()
        if rid not in done:
            raise FileNotFoundError(f"队列 {rid} 无已完成 session")
        out.append((rid, done[rid]))
    return out


def eval_three_and_lr(model, X: np.ndarray, y: np.ndarray, split_ids: np.ndarray, device: str) -> Dict[str, Any]:
    all_m = trial_majority_acc(model, X, y, split_ids, device)
    # L/R only windows/trials
    mask = np.isin(y, [1, 2])
    if not mask.any():
        lr = {"acc_trial_majority": None, "n_trials": 0, "pred_counts": {}}
    else:
        lr = trial_majority_acc(model, X[mask], y[mask], split_ids[mask], device)
    return {
        "three_trial_acc": all_m["acc_trial_majority"],
        "three_n_trials": all_m["n_trials"],
        "three_pred_counts": all_m["pred_counts"],
        "three_max_class_frac": all_m["max_class_frac"],
        "lr_trial_acc": lr.get("acc_trial_majority"),
        "lr_n_trials": lr.get("n_trials"),
        "lr_pred_counts": lr.get("pred_counts"),
        "lr_max_class_frac": lr.get("max_class_frac"),
    }


def load_three(ckpt: Path, device: str):
    entry = load_head(Path(ckpt), n_chans=8, n_times=N_TIMES, device=device)
    return entry.model


def run_ramp(
    manifest_path: Path,
    *,
    out_dir: Path,
    device: Optional[str] = None,
    max_r: int = 5,
) -> Dict[str, Any]:
    manifest = load_campaign(manifest_path)
    queue = _queue_sessions(manifest)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"a01_leave_next_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    print(f"[verify] device={device} out={run_dir}", flush=True)
    print(f"[verify] queue={[r for r, _ in queue]}", flush=True)

    for R in range(0, min(max_r, len(queue) - 1) + 1):
        eval_run, eval_path = queue[R]
        print(f"\n===== R{R} eval={eval_run} =====", flush=True)
        ds_ev = build_dataset(eval_path, include_invalid=True, protocol="auto")
        X_ev, y_ev, sid_ev = ds_ev["X"], ds_ev["y_three"], ds_ev["split_id"]
        row: Dict[str, Any] = {
            "R": R,
            "eval_run": eval_run,
            "eval_session": str(eval_path),
            "n_windows_eval": int(len(X_ev)),
            "n_trials_eval": int(ds_ev["n_trials"][0]),
            "train_runs": [],
        }

        if R == 0:
            model = load_three(DEFAULT_THREE, device)
            online = eval_three_and_lr(model, X_ev, y_ev, sid_ev, device)
            row.update({"ft": False, "ckpt": str(DEFAULT_THREE), "status": "ok", **online})
            rows.append(row)
            print(
                f"  T0 · three={online['three_trial_acc']:.3f} lr={online['lr_trial_acc']:.3f} "
                f"pred3={online['three_pred_counts']} predLR={online['lr_pred_counts']}",
                flush=True,
            )
            continue

        train_pairs = queue[:R]
        train_dirs = [p for _, p in train_pairs]
        row["train_runs"] = [r for r, _ in train_pairs]
        rec = ft_replay_recommendation(R)
        no_replay = not bool(rec["use_replay"])
        replay_ratio = float(rec["replay_ratio"])
        ft_out_dir = run_dir / f"R{R}_ft"
        print(
            f"  FT train={row['train_runs']} replay={not no_replay}@{replay_ratio} → {ft_out_dir.name}",
            flush=True,
        )
        result = run_subject_finetune(
            train_dirs,
            ft_out_dir,
            no_replay=no_replay,
            replay_ratio=replay_ratio,
            early_stop=True,
            max_epochs=20,
            patience=5,
            deterministic=True,
            seed=42,
            verbose=True,
            device=device,
        )
        three_ckpt = Path(result["out_dir"]) / "best_three.pt"
        model = load_three(three_ckpt, device)
        online = eval_three_and_lr(model, X_ev, y_ev, sid_ev, device)
        gate = result.get("release_gate") or {}
        row.update(
            {
                "ft": True,
                "status": "ok",
                "ckpt": str(three_ckpt),
                "release_pass": result.get("release_pass"),
                "heldout_three": (result.get("three") or {}).get("acc_after_heldout"),
                "gate": gate,
                "use_replay": not no_replay,
                "replay_ratio": 0.0 if no_replay else replay_ratio,
                "n_windows_train": int(
                    build_dataset(train_dirs, include_invalid=True, protocol="auto")["X"].shape[0]
                ),
                **online,
            }
        )
        rows.append(row)
        print(
            f"  FT done · gate={'PASS' if row.get('release_pass') else 'FAIL'} "
            f"heldout={row.get('heldout_three')} · "
            f"eval three={online['three_trial_acc']:.3f} lr={online['lr_trial_acc']:.3f} "
            f"pred3={online['three_pred_counts']} predLR={online['lr_pred_counts']}",
            flush=True,
        )

    # Exp29 offline A01 对照
    exp29_path = _REPO / "experiment_game/data/models/bci2a/exp29/ramp_leave_next.json"
    exp29_a01: List[Dict[str, Any]] = []
    if exp29_path.is_file():
        exp29 = json.loads(exp29_path.read_text(encoding="utf-8"))
        for r in exp29.get("rows") or []:
            if r.get("subject") == "A01" and r.get("status") == "ok":
                exp29_a01.append(
                    {
                        "R": r["R"],
                        "train_runs": r.get("train_runs"),
                        "eval_run": r.get("eval_run"),
                        "online_trial_acc": r.get("online_trial_acc"),
                    }
                )

    payload = {
        "subject_id": "A01",
        "protocol": "leave_next_sim_mat_cue",
        "manifest": str(manifest_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "rows": rows,
        "exp29_offline_a01_ref": exp29_a01,
    }
    (run_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = _format_md(payload)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print("\n" + md, flush=True)
    return payload


def _format_md(payload: Dict[str, Any]) -> str:
    lines = [
        f"# A01 Leave-Next 重验 · {payload.get('created_at')}",
        "",
        "切窗：`sim_mat_cue`（mat cue，不依赖错位 LSL）· 每次从 T0 重训 · Exp29 B2 replay",
        "",
        "| R | train | eval | FT窗 | three试次acc | L/R试次acc | pred(L/R) | gate |",
        "|---|-------|------|------|--------------|------------|-----------|------|",
    ]
    for r in payload.get("rows") or []:
        tr = ",".join(r.get("train_runs") or []) or "—"
        t_acc = r.get("three_trial_acc")
        l_acc = r.get("lr_trial_acc")
        t_s = f"{t_acc*100:.0f}%" if t_acc is not None else "—"
        l_s = f"{l_acc*100:.0f}%" if l_acc is not None else "—"
        pred = r.get("lr_pred_counts") or {}
        pred_s = ",".join(f"{k}:{v}" for k, v in sorted(pred.items())) or "—"
        gate = "PASS" if r.get("release_pass") else ("—" if not r.get("ft") else "FAIL")
        nw = r.get("n_windows_train") or 0
        lines.append(
            f"| R{r['R']} | {tr} | {r['eval_run']} | {nw} | {t_s} | {l_s} | {pred_s} | {gate} |"
        )
    lines.append("")
    if payload.get("exp29_offline_a01_ref"):
        lines.append("## Exp29 离线预处理对照（同被试 Leave-Next B2）")
        lines.append("")
        lines.append("| R | train | eval | online_trial_acc |")
        lines.append("|---|-------|------|------------------|")
        for r in payload["exp29_offline_a01_ref"]:
            acc = r.get("online_trial_acc")
            a_s = f"{acc*100:.0f}%" if acc is not None else "—"
            tr = ",".join(r.get("train_runs") or []) or "—"
            lines.append(f"| R{r['R']} | {tr} | {r.get('eval_run')} | {a_s} |")
        lines.append("")
    # first R with lr>=0.6
    hit = None
    for r in payload.get("rows") or []:
        if r.get("lr_trial_acc") is not None and r["lr_trial_acc"] >= 0.60:
            hit = r["R"]
            break
    lines.append(
        f"**L/R ≥60% 首达：** {'R'+str(hit) if hit is not None else '未达（R0–R5）'}"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--max-r", type=int, default=5)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    run_ramp(args.manifest.resolve(), out_dir=args.out.resolve(), device=args.device, max_r=args.max_r)


if __name__ == "__main__":
    main()
