"""实验 32 · BCI2a Leave-Next · 双底座 × 双门控 × FT 范围（so / all4）。

P0 臂：shallow|e1f_so × strict|force
P1 臂：e1f_all4 × strict|force（与 e1f_so 对照）

主指标：三分类窗级 acc（含 Rest）= Fuse/单头 → 因果平滑(lookback=2，按 trial) → argmax

用法：
  python experiment_game/tools/run_exp32_leave_next_dual_model_gate.py --materialize-only
  python experiment_game/tools/run_exp32_leave_next_dual_model_gate.py --phase p0 --all
  python experiment_game/tools/run_exp32_leave_next_dual_model_gate.py --phase p1 --all
  python experiment_game/tools/run_exp32_leave_next_dual_model_gate.py --phase all --subjects A01,A05
  python experiment_game/tools/run_exp32_leave_next_dual_model_gate.py --write-registry --stamp <stamp>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))

from adapt_engine.e1f import (  # noqa: E402
    E1fMemberSpec,
    E1fRegistry,
    E1fStackConfig,
    build_fn_for_arch,
)
from adapt_engine.readout import causal_smooth_pred_sequence  # noqa: E402
from adapt_engine.registry import load_head  # noqa: E402
from experiment_game.experiment.sim.bci2a_catalog import (  # noqa: E402
    list_subject_runs,
    resolve_mat_path,
)
from experiment_game.experiment.sim.bci2a_mat_loader import (  # noqa: E402
    count_run_capacity,
    load_bci2a_run,
)
from experiment_game.experiment.sim.run_to_session_map import build_sim_script  # noqa: E402
from experiment_game.experiment.sim.sim_script_io import write_sim_script  # noqa: E402
from experiment_game.pipeline.finetune import (  # noqa: E402
    N_TIMES,
    build_dataset,
    evaluate_release_gate,
    run_subject_finetune,
)

MAT_ROOT = _REPO / "DATA" / "bci2a"
SIM_ROOT = _REPO / "experiment_game" / "data" / "sim_subjects"
E1F_CONFIG = _REPO / "experiment_game" / "config" / "e1f_four_member.json"
DOC_ROOT = (
    _REPO
    / "资料"
    / "模型训练"
    / "32_旁路_bci2a_LeaveNext_双底座双门控_openbmi_accpaper"
)
CAUSAL_LOOKBACK = 2
SESSION_TRIALS = 36
SCRIPT_SEED = 32

SHALLOW_TASK = (
    _REPO
    / "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper"
    / "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100"
    / "run_20260822_094942/task/fold0/best_task.pt"
)
SHALLOW_THREE = (
    _REPO
    / "code/train_lab/out/5070_baseline_openbmi_3s_hop100_accpaper"
    / "shallow_openbmi_3s_hop100_balbatch_accpaper/openbmi_3s_hop100"
    / "run_20260822_094942/three/fold0/best_three.pt"
)

P0_ARMS = ("shallow_strict", "shallow_force", "e1f_so_strict", "e1f_so_force")
P1_ARMS = ("e1f_all4_strict", "e1f_all4_force")
ALL_ARMS = P0_ARMS + P1_ARMS


def _subject_queue(subject_id: str) -> List[str]:
    runs = list_subject_runs(subject_id)
    ids = [str(r["run_id"]) for r in runs]
    if len(ids) < 6:
        raise RuntimeError(f"{subject_id}: labeled runs < 6: {ids}")
    # 方案：多数 run3→run8；A04 为 run1→run6（mat 内最后 6 个带标签 run）
    return ids[-6:]


def materialize_subject_sessions(subject_id: str, *, force: bool = False) -> Dict[str, Path]:
    """写入轻量仿真 session（仅 sim_script + meta），供 FT 按 mat 切窗。"""
    sid = subject_id.upper()
    mat = resolve_mat_path(sid)
    queue = _subject_queue(sid)
    sess_root = SIM_ROOT / sid / "sessions"
    sess_root.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for i, rid in enumerate(queue):
        name = f"{sid}_{rid}_exp32"
        d = sess_root / name
        script_p = d / "sim_script.json"
        if script_p.is_file() and not force:
            out[rid] = d
            continue
        if d.exists() and force:
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        rd_cap = load_bci2a_run(mat, rid)
        _, _, _, n_max = count_run_capacity(rd_cap)
        n_trials = min(SESSION_TRIALS, int(n_max))
        # 尽量偶数，便于 2 block
        if n_trials >= 2 and n_trials % 2 == 1:
            n_trials -= 1
        script = build_sim_script(
            mat,
            rid,
            session_trials_total=n_trials,
            blocks=2,
            seed=SCRIPT_SEED + i,
        )
        write_sim_script(d, script)
        meta = {
            "subject_id": sid,
            "session_id": rid,
            "sim_mode": True,
            "phase_mode": "sim_v3_session",
            "source_mat": str(mat),
            "source_run": rid,
            "session_trials_total": n_trials,
            "exp32": True,
            "script_seed": SCRIPT_SEED + i,
        }
        (d / "session.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        out[rid] = d
    return out


def _softmax_batch(model, X: np.ndarray, device: str, *, n_out: int = 3) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    bs = 64
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[s : s + bs], dtype=np.float32)).to(
            device
        )
        with torch.no_grad():
            try:
                logits = model(xb)
            except RuntimeError:
                logits = model(xb.unsqueeze(1))
        if logits.dim() == 3:
            logits = logits.reshape(logits.shape[0], -1)
        if int(logits.shape[-1]) != int(n_out):
            raise ValueError(
                f"模型输出类别数={int(logits.shape[-1])} 与 n_out={n_out} 不一致；"
                "禁止静默切片，以免门控准确率系统性偏低"
            )
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds.append(probs.astype(np.float32))
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, n_out), np.float32)


def _window_acc_causal(
    probs: np.ndarray,
    y: np.ndarray,
    split_ids: np.ndarray,
    *,
    lookback: int = CAUSAL_LOOKBACK,
) -> Dict[str, Any]:
    """按 trial（split_id）因果平滑后窗级 argmax。"""
    order = defaultdict(list)
    for i, sid in enumerate(split_ids):
        order[str(sid)].append(i)
    pred = np.zeros(len(y), dtype=np.int64)
    for idxs in order.values():
        seq = [probs[i] for i in idxs]
        smoothed = causal_smooth_pred_sequence(seq, lookback=lookback)
        for j, ix in enumerate(idxs):
            pred[ix] = int(smoothed[j]["pred"])
    acc = float((pred == y).mean()) if len(y) else float("nan")
    m = np.isin(y, [1, 2])
    acc_lr = float((pred[m] == y[m]).mean()) if m.any() else float("nan")
    return {
        "n_windows": int(len(y)),
        "n_lr": int(m.sum()),
        "acc_window_three": acc,
        "acc_window_lr": acc_lr,
        "pred": pred,
    }


def _load_e1f_stack(
    *,
    member_three: Optional[Dict[str, Path]] = None,
    member_task: Optional[Dict[str, Path]] = None,
    shallow_three: Optional[Path] = None,
    shallow_task: Optional[Path] = None,
) -> E1fStackConfig:
    cfg = E1fStackConfig.load_json(E1F_CONFIG, repo_root=_REPO).resolve_paths(
        repo_root=_REPO
    )
    members: List[E1fMemberSpec] = []
    for m in cfg.members:
        three = m.three_ckpt
        task = m.task_ckpt
        if member_three and m.name in member_three:
            three = str(member_three[m.name])
        if member_task and m.name in member_task:
            task = str(member_task[m.name])
        if m.name == "shallow" and shallow_three is not None:
            three = str(shallow_three)
        if m.name == "shallow" and shallow_task is not None:
            task = str(shallow_task)
        members.append(
            E1fMemberSpec(name=m.name, arch=m.arch, three_ckpt=three, task_ckpt=task)
        )
    return E1fStackConfig(
        id=cfg.id,
        label=cfg.label,
        readout_mode=cfg.readout_mode,
        primary_judge_mode=cfg.primary_judge_mode,
        task_ckpt=str(shallow_task or members[0].task_ckpt or cfg.task_ckpt),
        members=members,
        fusion=cfg.fusion,
        test_acc_paper=cfg.test_acc_paper,
    )


def eval_readout(
    mode: str,
    session_dir: Path,
    *,
    device: str,
    shallow_three: Optional[Path] = None,
    shallow_task: Optional[Path] = None,
    member_three: Optional[Dict[str, Path]] = None,
    member_task: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    ds = build_dataset([session_dir], include_invalid=True, protocol="auto")
    X, y, split = ds["X"], ds["y_three"], ds["split_id"]
    if mode == "shallow":
        assert shallow_three is not None
        entry = load_head(shallow_three, n_chans=8, n_times=N_TIMES, device=device)
        probs = _softmax_batch(entry.model, X, device)
    elif mode in ("e1f_so", "e1f_all4"):
        stack = _load_e1f_stack(
            member_three=member_three,
            member_task=member_task,
            shallow_three=shallow_three,
            shallow_task=shallow_task,
        )
        missing = stack.missing_paths(repo_root=_REPO)
        if missing:
            raise FileNotFoundError("; ".join(missing))
        reg = E1fRegistry(stack, device=device)
        probs = reg.forward_three_batch(X)
    else:
        raise ValueError(mode)
    metrics = _window_acc_causal(probs, y, split)
    metrics["mode"] = mode
    metrics["session"] = session_dir.name
    return metrics


def _pred_dist(pred: np.ndarray) -> Dict[str, Any]:
    if len(pred) == 0:
        return {"pred_counts": {}, "max_class_frac": 1.0}
    u, c = np.unique(pred, return_counts=True)
    pc = {int(k): int(v) for k, v in zip(u, c)}
    mx = float(c.max() / c.sum())
    return {"pred_counts": pc, "max_class_frac": mx}


def _fused_gate_from_dirs(
    *,
    member_three: Dict[str, Path],
    member_task: Dict[str, Path],
    train_dirs: List[Path],
    heldout_dir: Path,
    device: str,
) -> Dict[str, Any]:
    """all4：在融合+因果平滑口径上算 heldout/train，再走 release_gate。"""
    stack = _load_e1f_stack(member_three=member_three, member_task=member_task)
    reg = E1fRegistry(stack, device=device)

    ds_tr = build_dataset(train_dirs, include_invalid=True, protocol="auto")
    probs_tr = reg.forward_three_batch(ds_tr["X"])
    m_tr = _window_acc_causal(probs_tr, ds_tr["y_three"], ds_tr["split_id"])

    ds_te = build_dataset([heldout_dir], include_invalid=True, protocol="auto")
    probs_te = reg.forward_three_batch(ds_te["X"])
    m_te = _window_acc_causal(probs_te, ds_te["y_three"], ds_te["split_id"])
    rep = {
        "acc_after_heldout": m_te["acc_window_three"],
        "acc_after_train": m_tr["acc_window_three"],
        "heldout_pred_dist": _pred_dist(m_te["pred"]),
    }
    gate = evaluate_release_gate(rep)
    gate["train_windows"] = m_tr["n_windows"]
    gate["heldout_windows"] = m_te["n_windows"]
    return gate


def _replay_on(r_stage: int) -> bool:
    return 1 <= r_stage < 4


def _parse_arm(arm: str) -> Tuple[str, str, bool]:
    """→ (model_mode, ft_scope, force_promote). model_mode: shallow|e1f_so|e1f_all4"""
    force = arm.endswith("_force")
    if arm.startswith("shallow_"):
        return "shallow", "shallow", force
    if arm.startswith("e1f_so_"):
        return "e1f_so", "so", force
    if arm.startswith("e1f_all4_"):
        return "e1f_all4", "all4", force
    raise ValueError(arm)


def _base_ckpts(mode: str) -> Tuple[Path, Path]:
    if mode == "shallow":
        return SHALLOW_TASK, SHALLOW_THREE
    # e1f shallow member
    cfg = E1fStackConfig.load_json(E1F_CONFIG, repo_root=_REPO).resolve_paths(
        repo_root=_REPO
    )
    m0 = cfg.members[0]
    return Path(m0.task_ckpt), Path(m0.three_ckpt)


def _member_base_ckpts() -> Dict[str, Tuple[Path, Path, str]]:
    """name → (task, three, arch)."""
    cfg = E1fStackConfig.load_json(E1F_CONFIG, repo_root=_REPO).resolve_paths(
        repo_root=_REPO
    )
    return {
        m.name: (Path(m.task_ckpt), Path(m.three_ckpt), m.arch) for m in cfg.members
    }


def run_arm(
    *,
    subject_id: str,
    arm: str,
    session_by_run: Dict[str, Path],
    queue: List[str],
    work_root: Path,
    device: str,
) -> List[Dict[str, Any]]:
    mode, scope, force_promote = _parse_arm(arm)
    work = work_root / arm
    work.mkdir(parents=True, exist_ok=True)
    state_p = work / "state.json"
    rows_p = work / "rows.json"

    if state_p.is_file() and rows_p.is_file():
        state = json.loads(state_p.read_text(encoding="utf-8"))
        if int(state.get("done_R", -1)) >= len(queue) - 1:
            print(f"  [skip] {subject_id}/{arm} already done", flush=True)
            return json.loads(rows_p.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    if rows_p.is_file():
        rows = json.loads(rows_p.read_text(encoding="utf-8"))
    done_R = max((r["R"] for r in rows), default=-1)

    base_task, base_three = _base_ckpts(mode if mode == "shallow" else "e1f_so")
    members_base = _member_base_ckpts()

    if scope == "all4":
        cur_m_task = {k: v[0] for k, v in members_base.items()}
        cur_m_three = {k: v[1] for k, v in members_base.items()}
        current_task, current_three = cur_m_task["shallow"], cur_m_three["shallow"]
    else:
        cur_m_task = cur_m_three = None
        current_task, current_three = base_task, base_three

    # 从已有行恢复 current
    for r in rows:
        if r.get("promoted") and r.get("ft_run_dir"):
            ft = Path(r["ft_run_dir"])
            if scope == "all4":
                for name in members_base:
                    t3 = ft / "members" / name / "best_three.pt"
                    tk = ft / "members" / name / "best_task.pt"
                    if t3.is_file():
                        cur_m_three[name] = t3  # type: ignore
                    if tk.is_file():
                        cur_m_task[name] = tk  # type: ignore
                current_three = cur_m_three["shallow"]  # type: ignore
                current_task = cur_m_task["shallow"]  # type: ignore
            else:
                if (ft / "best_three.pt").is_file():
                    current_three = ft / "best_three.pt"
                    current_task = ft / "best_task.pt"

    def _eval_current(eval_dir: Path) -> Dict[str, Any]:
        if mode == "shallow":
            return eval_readout(
                "shallow", eval_dir, device=device, shallow_three=current_three
            )
        if scope == "all4":
            return eval_readout(
                "e1f_all4",
                eval_dir,
                device=device,
                member_three=cur_m_three,
                member_task=cur_m_task,
            )
        return eval_readout(
            "e1f_so",
            eval_dir,
            device=device,
            shallow_three=current_three,
            shallow_task=current_task,
        )

    def _eval_baseline(eval_dir: Path) -> Dict[str, Any]:
        if mode == "shallow":
            return eval_readout(
                "shallow", eval_dir, device=device, shallow_three=base_three
            )
        return eval_readout(
            "e1f_so",
            eval_dir,
            device=device,
            shallow_three=base_three,
            shallow_task=base_task,
        )

    # R0
    if done_R < 0:
        eval0 = session_by_run[queue[0]]
        base0 = _eval_baseline(eval0)
        cur0 = _eval_current(eval0)
        rows.append(
            {
                "R": 0,
                "eval_run": queue[0],
                "train_runs": [],
                "ft": False,
                "gate_pass": None,
                "promoted": False,
                "arm": arm,
                "baseline_acc_window": base0["acc_window_three"],
                "baseline_acc_lr": base0["acc_window_lr"],
                "current_acc_window": cur0["acc_window_three"],
                "current_acc_lr": cur0["acc_window_lr"],
                "n_windows_eval": cur0["n_windows"],
                "ft_run_dir": None,
                "note": "零样本",
            }
        )
        rows_p.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"  [{arm}] R0 eval={queue[0]} "
            f"base={base0['acc_window_three']:.3f} cur={cur0['acc_window_three']:.3f}",
            flush=True,
        )
        done_R = 0

    for r in range(max(1, done_R + 1), len(queue)):
        train_runs = queue[:r]
        eval_run = queue[r]
        train_dirs = [session_by_run[x] for x in train_runs]
        eval_dir = session_by_run[eval_run]
        out_dir = work / f"R{r}_train_{'-'.join(train_runs)}_eval_{eval_run}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        use_replay = _replay_on(r)
        print(
            f"  [{arm}] R{r} train={train_runs} eval={eval_run} "
            f"replay={'0.10' if use_replay else 'off'}",
            flush=True,
        )

        gate: Dict[str, Any] = {}
        gate_pass = False
        if scope == "all4":
            assert cur_m_three is not None and cur_m_task is not None
            for name, (_bt, _b3, arch) in members_base.items():
                mdir = out_dir / "members" / name
                mdir.mkdir(parents=True, exist_ok=True)
                init_t = cur_m_task[name]
                init_3 = cur_m_three[name]
                print(f"    FT member={name} arch={arch}", flush=True)
                run_subject_finetune(
                    train_dirs,
                    mdir,
                    task_ckpt=init_t,
                    three_ckpt=init_3,
                    heldout_session_dirs=[eval_dir],
                    no_replay=not use_replay,
                    replay_ratio=0.10 if use_replay else 0.0,
                    early_stop=True,
                    max_epochs=20,
                    patience=5,
                    verbose=False,
                    device=device,
                    build_fn=build_fn_for_arch(arch),
                )
            cand_three = {
                n: out_dir / "members" / n / "best_three.pt" for n in members_base
            }
            cand_task = {
                n: out_dir / "members" / n / "best_task.pt" for n in members_base
            }
            gate = _fused_gate_from_dirs(
                member_three=cand_three,
                member_task=cand_task,
                train_dirs=train_dirs,
                heldout_dir=eval_dir,
                device=device,
            )
            gate_pass = bool(gate.get("pass"))
            (out_dir / "release_gate_fused.json").write_text(
                json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            promoted = False
            if gate_pass or force_promote:
                cur_m_three = cand_three
                cur_m_task = cand_task
                current_three = cand_three["shallow"]
                current_task = cand_task["shallow"]
                promoted = True
        else:
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
            if (out_dir / "release_gate.json").is_file():
                gate = json.loads((out_dir / "release_gate.json").read_text(encoding="utf-8"))
            gate_pass = bool(gate.get("pass"))
            promoted = False
            if gate_pass or force_promote:
                current_task = out_dir / "best_task.pt"
                current_three = out_dir / "best_three.pt"
                promoted = True

        base_ev = _eval_baseline(eval_dir)
        cur_ev = _eval_current(eval_dir)
        row = {
            "R": r,
            "eval_run": eval_run,
            "train_runs": train_runs,
            "ft": True,
            "gate_pass": gate_pass,
            "promoted": promoted,
            "arm": arm,
            "baseline_acc_window": base_ev["acc_window_three"],
            "baseline_acc_lr": base_ev["acc_window_lr"],
            "current_acc_window": cur_ev["acc_window_three"],
            "current_acc_lr": cur_ev["acc_window_lr"],
            "n_windows_eval": cur_ev["n_windows"],
            "heldout_acc_after": gate.get("heldout_acc"),
            "train_gap": gate.get("train_minus_heldout"),
            "gate_checks": gate.get("checks"),
            "ft_run_dir": str(out_dir),
            "replay": use_replay,
            "note": (
                "强制晋升"
                if promoted and not gate_pass
                else ("门控 PASS 晋升" if promoted else "门控 FAIL 未晋升")
            ),
        }
        rows.append(row)
        rows_p.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        state_p.write_text(
            json.dumps({"done_R": r, "arm": arm, "subject": subject_id}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(
            f"    → {row['note']} | eval three={row['current_acc_window']:.3f} "
            f"(base={row['baseline_acc_window']:.3f})",
            flush=True,
        )

    state_p.write_text(
        json.dumps(
            {"done_R": len(queue) - 1, "arm": arm, "subject": subject_id, "complete": True},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def _pct(x: Any) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
        if np.isnan(v):
            return "—"
        return f"{v:.3f}"
    except Exception:
        return "—"


def write_summary(out_root: Path, all_results: Dict[str, Dict[str, List[Dict]]]) -> None:
    lines = [
        "# 实验 32 · Leave-Next 结果摘要",
        "",
        f"- 输出：`{out_root}`",
        f"- 主指标：三分类窗级（含 Rest）+ 因果平滑 lookback={CAUSAL_LOOKBACK}",
        "",
    ]
    for sid in sorted(all_results):
        lines.append(f"## {sid}")
        lines.append("")
        arms = all_results[sid]
        # header
        cols = [a for a in ALL_ARMS if a in arms]
        lines.append("| R | eval | " + " | ".join(cols) + " |")
        lines.append("|---|" + "|".join(["---"] * (len(cols) + 1)) + "|")
        rs = sorted({r["R"] for a in cols for r in arms[a]})
        for R in rs:
            eval_run = ""
            cells = []
            for a in cols:
                by = {x["R"]: x for x in arms[a]}
                row = by.get(R, {})
                eval_run = row.get("eval_run") or eval_run
                cells.append(_pct(row.get("current_acc_window")))
            lines.append(f"| {R} | {eval_run} | " + " | ".join(cells) + " |")
        lines.append("")
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_registry_tables(all_results: Dict[str, Dict[str, List[Dict]]], stamp: str) -> Path:
    """回填 资料/模型训练/32/.../总结/结果登记表.md"""
    arm_col = {
        "shallow_strict": "S-st",
        "shallow_force": "S-fo",
        "e1f_so_strict": "E-so-st",
        "e1f_so_force": "E-so-fo",
        "e1f_all4_strict": "E-a4-st",
        "e1f_all4_force": "E-a4-fo",
    }

    def cell(sid: str, arm: str, R: int, key: str = "current_acc_window") -> str:
        rows = (all_results.get(sid) or {}).get(arm) or []
        by = {r["R"]: r for r in rows}
        return _pct(by.get(R, {}).get(key))

    def peak(sid: str, arm: str) -> str:
        rows = (all_results.get(sid) or {}).get(arm) or []
        vals = [float(r["current_acc_window"]) for r in rows if r.get("current_acc_window") is not None]
        return _pct(max(vals) if vals else None)

    def gate_pass_count(sid: str, arm: str) -> str:
        rows = (all_results.get(sid) or {}).get(arm) or []
        ft = [r for r in rows if r.get("ft")]
        if not ft:
            return "_/5"
        n = sum(1 for r in ft if r.get("gate_pass"))
        return f"{n}/5"

    lines = [
        "# 实验 32 · 结果登记表",
        "",
        f"> 数据集：**BCI2a T.mat** · Leave-Next 6 run · stamp=`{stamp}`  ",
        f"> 主指标：**三分类窗级 acc（含 Rest）** · 因果平滑 lookback={CAUSAL_LOOKBACK}  ",
        "> **P0** `e1f_so`；**P1** `e1f_all4`  ",
        f"> 方案：[`方案.md`](../方案.md) · 原始：`experiment_game/data/sim_subjects/_analysis/exp32_{stamp}/`  ",
        f"> 填表时间：{datetime.now().strftime('%Y-%m-%d')}",
        "",
        "**图例：** `S-st/fo`=shallow；`E-so-*`=e1f_so；`E-a4-*`=e1f_all4。",
        "",
        "# P0 · 主表",
        "",
    ]
    subjects = [f"A0{i}" for i in range(1, 10)]
    for sid in subjects:
        lines.append(f"## {sid}")
        lines.append("")
        lines.append("| R | eval | S-st | S-fo | E-so-st | E-so-fo | 底座零样本 |")
        lines.append("|---|------|------|------|---------|---------|------------|")
        q = _subject_queue(sid) if sid in all_results or True else []
        try:
            q = _subject_queue(sid)
        except Exception:
            q = ["run?"] * 6
        for R in range(6):
            ev = q[R] if R < len(q) else "—"
            lines.append(
                f"| R{R} | {ev} | {cell(sid,'shallow_strict',R)} | {cell(sid,'shallow_force',R)} | "
                f"{cell(sid,'e1f_so_strict',R)} | {cell(sid,'e1f_so_force',R)} | "
                f"{cell(sid,'e1f_so_force',R,'baseline_acc_window')} |"
            )
        lines.append(
            f"| 峰值 | — | {peak(sid,'shallow_strict')} | {peak(sid,'shallow_force')} | "
            f"{peak(sid,'e1f_so_strict')} | {peak(sid,'e1f_so_force')} | — |"
        )
        lines.append("")
        lines.append(
            f"门控 PASS（R1–R5）：S-st {gate_pass_count(sid,'shallow_strict')} · "
            f"E-so-st {gate_pass_count(sid,'e1f_so_strict')}"
        )
        lines.append("")

    # P0 means
    lines.extend(["## P0 总表", "", "### 各 R 九人均值 · 三分类窗级", ""])
    lines.append("| R | S-st | S-fo | E-so-st | E-so-fo |")
    lines.append("|---|------|------|---------|---------|")
    for R in range(6):
        cells = []
        for arm in ("shallow_strict", "shallow_force", "e1f_so_strict", "e1f_so_force"):
            vals = []
            for sid in subjects:
                rows = (all_results.get(sid) or {}).get(arm) or []
                by = {r["R"]: r for r in rows}
                v = by.get(R, {}).get("current_acc_window")
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            cells.append(_pct(float(np.mean(vals)) if vals else None))
        lines.append(f"| R{R} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.extend(["# P1 · 对照表", ""])
    for sid in subjects:
        lines.append(f"## {sid} · P1")
        lines.append("")
        lines.append("| R | eval | E-so-st | E-a4-st | Δst | E-so-fo | E-a4-fo | Δfo |")
        lines.append("|---|------|---------|---------|-----|---------|---------|-----|")
        try:
            q = _subject_queue(sid)
        except Exception:
            q = ["—"] * 6
        for R in range(6):
            so_st = cell(sid, "e1f_so_strict", R)
            a4_st = cell(sid, "e1f_all4_strict", R)
            so_fo = cell(sid, "e1f_so_force", R)
            a4_fo = cell(sid, "e1f_all4_force", R)

            def delta(a: str, b: str) -> str:
                try:
                    return _pct(float(b) - float(a)) if a != "—" and b != "—" else "—"
                except Exception:
                    return "—"

            # cell returns formatted floats like 0.123 — parse carefully
            def _f(s: str) -> Optional[float]:
                try:
                    return float(s) if s != "—" else None
                except Exception:
                    return None

            dst = (
                _pct(_f(a4_st) - _f(so_st))
                if _f(a4_st) is not None and _f(so_st) is not None
                else "—"
            )
            dfo = (
                _pct(_f(a4_fo) - _f(so_fo))
                if _f(a4_fo) is not None and _f(so_fo) is not None
                else "—"
            )
            lines.append(
                f"| R{R} | {q[R] if R < len(q) else '—'} | {so_st} | {a4_st} | {dst} | "
                f"{so_fo} | {a4_fo} | {dfo} |"
            )
        lines.append("")

    lines.extend(
        [
            "## P1 总表",
            "",
            "### 各 R 九人均值 · so vs all4",
            "",
            "| R | E-so-st | E-a4-st | Δst | E-so-fo | E-a4-fo | Δfo |",
            "|---|---------|---------|-----|---------|---------|-----|",
        ]
    )
    for R in range(6):
        def mean_arm(arm: str) -> Optional[float]:
            vals = []
            for sid in subjects:
                rows = (all_results.get(sid) or {}).get(arm) or []
                by = {r["R"]: r for r in rows}
                v = by.get(R, {}).get("current_acc_window")
                if v is not None:
                    vals.append(float(v))
            return float(np.mean(vals)) if vals else None

        so_st, a4_st = mean_arm("e1f_so_strict"), mean_arm("e1f_all4_strict")
        so_fo, a4_fo = mean_arm("e1f_so_force"), mean_arm("e1f_all4_force")
        lines.append(
            f"| R{R} | {_pct(so_st)} | {_pct(a4_st)} | "
            f"{_pct(None if so_st is None or a4_st is None else a4_st - so_st)} | "
            f"{_pct(so_fo)} | {_pct(a4_fo)} | "
            f"{_pct(None if so_fo is None or a4_fo is None else a4_fo - so_fo)} |"
        )
    lines.append("")
    lines.append("## 结论（自动草稿）")
    lines.append("")
    r5_so = []
    r5_a4 = []
    for sid in subjects:
        for arm, bucket in (("e1f_so_force", r5_so), ("e1f_all4_force", r5_a4)):
            rows = (all_results.get(sid) or {}).get(arm) or []
            by = {r["R"]: r for r in rows}
            if 5 in by and by[5].get("current_acc_window") is not None:
                bucket.append(float(by[5]["current_acc_window"]))
    if r5_so and r5_a4 and len(r5_so) == len(r5_a4):
        d = float(np.mean(r5_a4) - np.mean(r5_so))
        lines.append(
            f"- force · R5 九人均值：e1f_so={np.mean(r5_so):.3f} · "
            f"e1f_all4={np.mean(r5_a4):.3f} · Δ={d:+.3f}"
            + (" → **过 +2pp 门槛，倾向 all4**" if d >= 0.02 else " → **未过 +2pp，倾向维持 so**")
        )
    else:
        lines.append("- （P1 force 未齐，结论待补）")
    lines.append("")

    out = DOC_ROOT / "总结" / "结果登记表.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def load_results_from_disk(out_root: Path) -> Dict[str, Dict[str, List[Dict]]]:
    all_results: Dict[str, Dict[str, List[Dict]]] = {}
    for sid_dir in sorted(out_root.iterdir()):
        if not sid_dir.is_dir() or not sid_dir.name.startswith("A"):
            continue
        arms: Dict[str, List[Dict]] = {}
        for arm_dir in sid_dir.iterdir():
            rows_p = arm_dir / "rows.json"
            if rows_p.is_file():
                arms[arm_dir.name] = json.loads(rows_p.read_text(encoding="utf-8"))
        if arms:
            all_results[sid_dir.name] = arms
    return all_results


def main() -> int:
    ap = argparse.ArgumentParser(description="Exp32 Leave-Next dual model × gate × FT scope")
    ap.add_argument("--subjects", type=str, default="", help="逗号分隔，如 A01,A05")
    ap.add_argument("--all", action="store_true", help="A01–A09")
    ap.add_argument("--phase", choices=["p0", "p1", "all"], default="all")
    ap.add_argument("--arms", type=str, default="", help="覆盖臂列表")
    ap.add_argument("--materialize-only", action="store_true")
    ap.add_argument("--force-rematerialize", action="store_true")
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--write-registry", action="store_true")
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    if args.all or not args.subjects:
        subjects = [f"A0{i}" for i in range(1, 10)]
    else:
        subjects = [s.strip().upper() for s in args.subjects.split(",") if s.strip()]

    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    elif args.phase == "p0":
        arms = list(P0_ARMS)
    elif args.phase == "p1":
        arms = list(P1_ARMS)
    else:
        arms = list(ALL_ARMS)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = SIM_ROOT / "_analysis" / f"exp32_{stamp}"

    print("materialize sessions…", flush=True)
    for sid in subjects:
        by = materialize_subject_sessions(sid, force=args.force_rematerialize)
        print(f"  {sid}: {list(by)}", flush=True)
    if args.materialize_only:
        return 0

    if args.write_registry and not any(
        (out_root / sid).is_dir() for sid in subjects
    ):
        # allow writing from existing stamp folder
        pass

    out_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "stamp": stamp,
        "subjects": subjects,
        "arms": arms,
        "phase": args.phase,
        "causal_lookback": CAUSAL_LOOKBACK,
        "device": device,
    }
    (out_root / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    all_results: Dict[str, Dict[str, List[Dict]]] = {}
    for sid in subjects:
        print(f"\n==== {sid} ====", flush=True)
        queue = _subject_queue(sid)
        session_by_run = materialize_subject_sessions(sid)
        missing = [r for r in queue if r not in session_by_run]
        if missing:
            print(f"  missing sessions {missing}", flush=True)
            return 1
        for arm in arms:
            print(f"\n-- {sid} / {arm} --", flush=True)
            run_arm(
                subject_id=sid,
                arm=arm,
                session_by_run=session_by_run,
                queue=queue,
                work_root=out_root / sid,
                device=device,
            )

    all_results = load_results_from_disk(out_root)
    write_summary(out_root, all_results)
    reg = write_registry_tables(all_results, stamp)
    (out_root / "all_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nDONE → {out_root}", flush=True)
    print(f"registry → {reg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
