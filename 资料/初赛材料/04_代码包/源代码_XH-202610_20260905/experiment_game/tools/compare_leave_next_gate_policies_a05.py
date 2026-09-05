"""A05 Leave-Next 双策略对比：门控 FAIL 不晋升 vs FAIL 强制晋升。

按 Campaign 队列 run3→run8：
  R0：底座零样本评 run3
  Rk (k≥1)：train=前 k 场 · heldout=第 k+1 场 · FT → 按策略决定是否写入 current
             再用「当前权重」在 eval 场上报窗级 three acc

策略：
  strict — release_gate FAIL 则不替换 current（本档 FT 作废，继续用上一版）
  force  — 无论 PASS/FAIL 都替换 current

对照列：同 eval 场上底座（E1f three）零样本窗级 acc。

用法：
  python experiment_game/tools/compare_leave_next_gate_policies_a05.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.registry import load_head  # noqa: E402
from experiment_game.pipeline.finetune import (  # noqa: E402
    DEFAULT_TASK,
    DEFAULT_THREE,
    N_TIMES,
    _eval_acc,
    build_dataset,
    run_subject_finetune,
)

SUBJECT = "A05"
CAMPAIGN = (
    _REPO
    / "experiment_game/data/sim_subjects/A05/records/campaigns/sim_20260829_215001/manifest.json"
)
SESSIONS_ROOT = _REPO / "experiment_game/data/sim_subjects/A05/sessions"
OUT_ROOT = (
    _REPO
    / "experiment_game/data/sim_subjects/A05/analysis"
    / f"leave_next_gate_cmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
QUEUE = ["run3", "run4", "run5", "run6", "run7", "run8"]


def _pick_session_dirs(manifest: Dict[str, Any]) -> Dict[str, Path]:
    """优先 Campaign 已完成目录；否则取该 run 最新 session。"""
    out: Dict[str, Path] = {}
    for item in manifest.get("sessions_completed") or []:
        rid = str(item.get("run_id") or "").lower()
        p = Path(str(item.get("session_dir") or ""))
        if rid and p.is_dir():
            out[rid] = p
    for rid in QUEUE:
        if rid in out:
            continue
        cands = sorted(
            [
                d
                for d in SESSIONS_ROOT.iterdir()
                if d.is_dir() and f"_{rid}_" in d.name
            ],
            key=lambda d: d.name,
        )
        if cands:
            out[rid] = cands[-1]
    return out


def _eval_window_acc(three_ckpt: Path, session_dir: Path, *, device: str) -> Dict[str, Any]:
    ds = build_dataset([session_dir], include_invalid=True, protocol="auto")
    X, y = ds["X"], ds["y_three"]
    entry = load_head(three_ckpt, n_chans=8, n_times=N_TIMES, device=device)
    acc = _eval_acc(entry.model, X, y, device)
    # L/R only（与结束页窗级口径接近）
    m = np.isin(y, [1, 2])
    acc_lr = _eval_acc(entry.model, X[m], y[m], device) if m.any() else float("nan")
    return {
        "n_windows": int(len(X)),
        "n_lr": int(m.sum()),
        "acc_window_three": float(acc),
        "acc_window_lr": float(acc_lr),
        "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }


def _replay_on(r_stage: int) -> bool:
    # 与操作台 Leave-Next 提示一致：R1–R3 开 0.10，R4+ 关
    return 1 <= r_stage < 4


def run_policy(
    *,
    name: str,
    force_promote: bool,
    session_by_run: Dict[str, Path],
    device: str,
) -> List[Dict[str, Any]]:
    work = OUT_ROOT / name
    work.mkdir(parents=True, exist_ok=True)
    current_task = Path(DEFAULT_TASK)
    current_three = Path(DEFAULT_THREE)
    rows: List[Dict[str, Any]] = []

    # R0：底座评 run3
    eval0 = session_by_run[QUEUE[0]]
    base0 = _eval_window_acc(Path(DEFAULT_THREE), eval0, device=device)
    cur0 = _eval_window_acc(current_three, eval0, device=device)
    rows.append(
        {
            "R": 0,
            "eval_run": QUEUE[0],
            "train_runs": [],
            "ft": False,
            "gate_pass": None,
            "promoted": False,
            "policy": name,
            "baseline_acc_window": base0["acc_window_three"],
            "baseline_acc_lr": base0["acc_window_lr"],
            "current_acc_window": cur0["acc_window_three"],
            "current_acc_lr": cur0["acc_window_lr"],
            "n_windows_eval": base0["n_windows"],
            "heldout_acc_after": None,
            "train_gap": None,
            "ft_run_dir": None,
            "note": "零样本底座（无 FT）",
        }
    )
    print(f"\n[{name}] R0 eval={QUEUE[0]} base={base0['acc_window_three']:.3f}", flush=True)

    for r in range(1, len(QUEUE)):
        train_runs = QUEUE[:r]
        eval_run = QUEUE[r]
        train_dirs = [session_by_run[x] for x in train_runs]
        eval_dir = session_by_run[eval_run]
        out_dir = work / f"R{r}_train_{'-'.join(train_runs)}_eval_{eval_run}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        use_replay = _replay_on(r)
        print(
            f"\n[{name}] R{r} train={train_runs} heldout/eval={eval_run} "
            f"replay={'0.10' if use_replay else 'off'} init=current",
            flush=True,
        )
        result = run_subject_finetune(
            train_dirs,
            out_dir,
            task_ckpt=current_task,
            three_ckpt=current_three,
            heldout_session_dirs=[eval_dir],
            no_replay=not use_replay,
            replay_ratio=0.10 if use_replay else 0.0,
            early_stop=True,
            max_epochs=20,
            patience=5,
            verbose=True,
            device=device,
        )
        gate = result.get("release_gate") or {}
        gate_pass = bool(result.get("release_pass") if "release_pass" in result else gate.get("pass"))
        three_rep = result.get("three") or {}
        if (out_dir / "release_gate.json").is_file():
            gate = json.loads((out_dir / "release_gate.json").read_text(encoding="utf-8"))
            gate_pass = bool(gate.get("pass"))
        if (out_dir / "meta.json").is_file():
            meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
            three_rep = meta.get("three") or three_rep

        promoted = False
        heldout_raw = float(
            three_rep.get("acc_after_heldout") or gate.get("heldout_acc") or 0
        )
        heldout_smooth = three_rep.get("acc_after_heldout_smooth")
        if heldout_smooth is None:
            heldout_smooth = heldout_raw
        else:
            heldout_smooth = float(heldout_smooth)
        if gate_pass or force_promote:
            current_task = out_dir / "best_task.pt"
            current_three = out_dir / "best_three.pt"
            promoted = True
            print(
                f"  → promote {'FORCE' if (not gate_pass and force_promote) else 'PASS'} "
                f"heldout_smooth={heldout_smooth:.3f} raw={heldout_raw:.3f}",
                flush=True,
            )
        else:
            print(
                f"  → SKIP promote (strict · FAIL) keep previous current",
                flush=True,
            )

        base_ev = _eval_window_acc(Path(DEFAULT_THREE), eval_dir, device=device)
        cur_ev = _eval_window_acc(current_three, eval_dir, device=device)
        row = {
            "R": r,
            "eval_run": eval_run,
            "train_runs": train_runs,
            "ft": True,
            "gate_pass": gate_pass,
            "promoted": promoted,
            "policy": name,
            "baseline_acc_window": base_ev["acc_window_three"],
            "baseline_acc_lr": base_ev["acc_window_lr"],
            "current_acc_window": cur_ev["acc_window_three"],
            "current_acc_lr": cur_ev["acc_window_lr"],
            "n_windows_eval": base_ev["n_windows"],
            "heldout_acc_after": heldout_smooth,
            "heldout_acc_after_raw": heldout_raw,
            "heldout_acc_before": three_rep.get("acc_before_heldout"),
            "heldout_acc_before_smooth": three_rep.get("acc_before_heldout_smooth"),
            "train_acc_after": three_rep.get("acc_after_train") or gate.get("train_acc"),
            "train_gap": gate.get("train_minus_heldout"),
            "gate_checks": gate.get("checks"),
            "pred_labels": gate.get("pred_labels"),
            "ft_run_dir": str(out_dir),
            "replay": use_replay,
            "note": (
                "强制晋升" if promoted and not gate_pass else
                ("门控 PASS 晋升" if promoted else "门控 FAIL 未晋升")
            ),
        }
        rows.append(row)
        print(
            f"  eval {eval_run}: baseline={row['baseline_acc_window']:.3f} "
            f"current={row['current_acc_window']:.3f} ({row['note']})",
            flush=True,
        )

    (work / "rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def _pct(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "—"


def write_compare_md(
    strict_rows: List[Dict[str, Any]],
    force_rows: List[Dict[str, Any]],
    path: Path,
) -> None:
    by_s = {r["R"]: r for r in strict_rows}
    by_f = {r["R"]: r for r in force_rows}
    lines = [
        "# A05 Leave-Next · 门控策略对比",
        "",
        f"- 被试：`{SUBJECT}` · Campaign `sim_20260829_215001`",
        f"- 队列：`{' → '.join(QUEUE)}`",
        "- 底座：E1f / OpenBMI-Align three+task（5090_alg_incr）",
        "- **strict**：门控 FAIL → **不晋升** current（本档 FT 作废）",
        "- **force**：门控 FAIL → **仍强制晋升** current",
        "- 指标：eval 场 **窗级 three acc**（含 Rest）；另附 L/R 窗级",
        "- replay：R1–R3 = 0.10 · R4+ = off",
        "",
        "## 并排对比（同 R / 同 eval）",
        "",
        "| R | eval | 底座窗级 | strict 当前 | force 当前 | strict 门控 | force 门控 | strict 晋升 | force 晋升 |",
        "|---:|---|---:|---:|---:|---|---|---|---|",
    ]
    for r in sorted(set(by_s) | set(by_f)):
        s, f = by_s.get(r, {}), by_f.get(r, {})
        lines.append(
            f"| {r} | {s.get('eval_run') or f.get('eval_run')} | "
            f"{_pct(s.get('baseline_acc_window'))} | "
            f"{_pct(s.get('current_acc_window'))} | "
            f"{_pct(f.get('current_acc_window'))} | "
            f"{'PASS' if s.get('gate_pass') else ('—' if s.get('gate_pass') is None else 'FAIL')} | "
            f"{'PASS' if f.get('gate_pass') else ('—' if f.get('gate_pass') is None else 'FAIL')} | "
            f"{'Y' if s.get('promoted') else 'N'} | "
            f"{'Y' if f.get('promoted') else 'N'} |"
        )
    lines.extend(
        [
            "",
            "## L/R 窗级对照",
            "",
            "| R | eval | 底座 L/R | strict L/R | force L/R |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for r in sorted(set(by_s) | set(by_f)):
        s, f = by_s.get(r, {}), by_f.get(r, {})
        lines.append(
            f"| {r} | {s.get('eval_run') or f.get('eval_run')} | "
            f"{_pct(s.get('baseline_acc_lr'))} | "
            f"{_pct(s.get('current_acc_lr'))} | "
            f"{_pct(f.get('current_acc_lr'))} |"
        )
    lines.extend(["", "## 明细", ""])
    for name, rows in (("strict", strict_rows), ("force", force_rows)):
        lines.append(f"### {name}")
        lines.append("")
        for row in rows:
            lines.append(
                f"- **R{row['R']}** eval=`{row['eval_run']}` train=`{row.get('train_runs')}` · "
                f"gate={row.get('gate_pass')} promote={row.get('promoted')} · "
                f"heldout {_pct(row.get('heldout_acc_after'))} · "
                f"current {_pct(row.get('current_acc_window'))} · {row.get('note')}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    session_by_run = _pick_session_dirs(manifest)
    missing = [r for r in QUEUE if r not in session_by_run]
    if missing:
        print(f"缺少 session: {missing}", flush=True)
        return 1
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"out={OUT_ROOT}", flush=True)
    print("sessions:", {k: str(v) for k, v in session_by_run.items()}, flush=True)

    strict_rows = run_policy(
        name="strict_no_promote_on_fail",
        force_promote=False,
        session_by_run=session_by_run,
        device=device,
    )
    force_rows = run_policy(
        name="force_promote_despite_fail",
        force_promote=True,
        session_by_run=session_by_run,
        device=device,
    )
    write_compare_md(strict_rows, force_rows, OUT_ROOT / "compare.md")
    (OUT_ROOT / "compare.json").write_text(
        json.dumps(
            {"strict": strict_rows, "force": force_rows, "queue": QUEUE},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nDONE → {OUT_ROOT / 'compare.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
