"""E1f all4：四成员各自增量 FT（默认 FT 范围）。

- 每名成员从当前 ckpt（或底座）增量 FT three；shallow 另训 task
- 产物：ft_runs/<stamp>/members/<name>/{best_three,best_task}.pt
- 融合评估：用 FT 后成员拼临时 E1fRegistry，在 heldout 上算 release gate
- 门控 FAIL 时由上层按 force_promote 告警落盘并仍可晋升
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from adapt_engine.e1f import E1fRegistry, E1fStackConfig, build_fn_for_arch
from adapt_engine.ft import FTRecipe, ReplayPool

from .finetune import (
    DEFAULT_FT_DETERMINISTIC,
    DEFAULT_FT_EPOCHS_FIXED,
    DEFAULT_FT_MAX_EPOCHS,
    DEFAULT_FT_PATIENCE,
    DEFAULT_REPLAY_RATIO,
    DEFAULT_SEED,
    build_dataset,
    build_t0_replay_pool,
    build_t0_task_replay_pool,
    evaluate_release_gate,
    finetune_head,
    set_training_deterministic,
    _session_leave_split,
    _trial_split,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_E1F_STACK = REPO_ROOT / "experiment_game" / "config" / "e1f_four_member.json"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rel_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _pred_dist_from_labels(pred: np.ndarray) -> Dict[str, Any]:
    if len(pred) == 0:
        return {"pred_counts": {}, "mean_p": [], "max_class_frac": 0.0}
    uniq, cnt = np.unique(pred.astype(np.int64), return_counts=True)
    return {
        "pred_counts": {int(k): int(v) for k, v in zip(uniq, cnt)},
        "mean_p": [],
        "max_class_frac": float(cnt.max() / len(pred)),
    }


def resolve_member_init_ckpts(
    *,
    subject_models_dir: Path,
    stack: E1fStackConfig,
) -> Dict[str, Dict[str, Optional[Path]]]:
    """优先 subject current/members + overlay；否则底座路径。"""
    overlay_path = subject_models_dir / "current" / "e1f_overlay.json"
    cur_members = subject_models_dir / "current" / "members"
    overlay_members: Dict[str, Any] = {}
    if overlay_path.is_file():
        overlay_members = (_load_json(overlay_path).get("members") or {})

    out: Dict[str, Dict[str, Optional[Path]]] = {}
    for m in stack.members:
        three = Path(m.three_ckpt)
        task: Optional[Path] = Path(m.task_ckpt) if m.task_ckpt else None
        ov = overlay_members.get(m.name) or {}
        if ov.get("three_ckpt"):
            p = Path(str(ov["three_ckpt"]))
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            if p.is_file():
                three = p
        else:
            local = cur_members / m.name / "best_three.pt"
            if local.is_file():
                three = local
        if m.name == "shallow":
            if ov.get("task_ckpt"):
                p = Path(str(ov["task_ckpt"]))
                if not p.is_absolute():
                    p = (REPO_ROOT / p).resolve()
                if p.is_file():
                    task = p
            else:
                local_m = cur_members / m.name / "best_task.pt"
                local_t = subject_models_dir / "current" / "best_task.pt"
                if local_m.is_file():
                    task = local_m
                elif local_t.is_file():
                    task = local_t
        elif ov.get("task_ckpt"):
            p = Path(str(ov["task_ckpt"]))
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            if p.is_file():
                task = p
        else:
            local_m = cur_members / m.name / "best_task.pt"
            if local_m.is_file():
                task = local_m
        out[m.name] = {"three": three, "task": task}
    return out


def _eval_e1f_three(
    registry: E1fRegistry,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    probs = registry.forward_three_batch(X)
    pred = probs.argmax(axis=1).astype(np.int64)
    acc = float((pred == y.astype(np.int64)).mean()) if len(y) else 0.0
    return {"acc": acc, "pred_dist": _pred_dist_from_labels(pred), "n": int(len(y))}


def run_e1f_all4_finetune(
    session_dirs: List[Path],
    out_dir: Path,
    *,
    subject_models_dir: Path,
    stack_json: Path | None = None,
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
    """四成员各自增量 FT；写出 members/ + e1f_overlay.json + release_gate。"""
    session_dirs = [Path(p).resolve() for p in session_dirs]
    hold_dirs = [Path(p).resolve() for p in (heldout_session_dirs or [])]
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_models_dir = Path(subject_models_dir).resolve()

    stack_path = Path(stack_json) if stack_json else DEFAULT_E1F_STACK
    stack = E1fStackConfig.load_json(stack_path, repo_root=REPO_ROOT).resolve_paths(
        repo_root=REPO_ROOT
    )
    init_ckpts = resolve_member_init_ckpts(
        subject_models_dir=subject_models_dir, stack=stack
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if deterministic:
        set_training_deterministic(seed)

    all_dirs = session_dirs + hold_dirs if hold_dirs else session_dirs
    ds = build_dataset(all_dirs, include_invalid=not exclude_invalid, protocol=protocol)
    X = ds["X"]
    y3 = ds["y_three"]
    y_task = ds["y_task"]
    split_id = ds["split_id"]
    resolved_protocol = ds.get("protocol", protocol)
    leave_next = bool(hold_dirs)

    if leave_next:
        tr_mask, te_mask = _session_leave_split(
            split_id,
            train_session_names=[d.name for d in session_dirs],
            heldout_session_names=[d.name for d in hold_dirs],
        )
    else:
        tr_mask, te_mask = _trial_split(split_id, train_frac=train_frac, seed=seed)

    X_tr, y3_tr = X[tr_mask], y3[tr_mask]
    X_te, y3_te = X[te_mask], y3[te_mask]
    yt_tr, yt_te = y_task[tr_mask], y_task[te_mask]

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

    members_out: Dict[str, Any] = {}
    overlay_members: Dict[str, Dict[str, str]] = {}
    members_root = out_dir / "members"
    base_meta = {
        "ft_scope": "all4",
        "protocol": resolved_protocol,
        "leave_next": leave_next,
        "sessions": [d.name for d in session_dirs],
        "heldout_sessions": [d.name for d in hold_dirs],
        "seed": int(seed),
        "deterministic": bool(deterministic),
    }
    ft_kw = {
        "early_stop": early_stop,
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "fixed_epochs": int(epochs),
    }

    for m in stack.members:
        mdir = members_root / m.name
        mdir.mkdir(parents=True, exist_ok=True)
        init = init_ckpts[m.name]
        three_init = init["three"]
        task_init = init.get("task")
        if three_init is None or not Path(three_init).is_file():
            raise FileNotFoundError(f"all4 成员 {m.name} 缺 three 初值: {three_init}")
        build_fn = build_fn_for_arch(m.arch)
        if verbose:
            print(
                f"[all4] FT member={m.name} arch={m.arch} from={Path(three_init).name}",
                flush=True,
            )

        three_tmp = mdir / "best_three.pt.tmp"
        three_rep = finetune_head(
            Path(three_init),
            X_tr,
            y3_tr,
            X_te,
            y3_te,
            n_outputs=3,
            device=device,
            recipe=recipe,
            replay_pool=replay_pool,
            out_path=three_tmp,
            meta={**base_meta, "member": m.name, "arch": m.arch, "head": "three"},
            build_fn=build_fn,
            **ft_kw,
        )
        three_path = mdir / "best_three.pt"
        three_tmp.replace(three_path)

        task_path: Optional[Path] = None
        task_rep = None
        if task_init and Path(task_init).is_file():
            task_tmp = mdir / "best_task.pt.tmp"
            task_rep = finetune_head(
                Path(task_init),
                X_tr,
                yt_tr,
                X_te,
                yt_te,
                n_outputs=2,
                device=device,
                recipe=recipe,
                replay_pool=task_replay_pool if m.name == "shallow" else None,
                out_path=task_tmp,
                meta={**base_meta, "member": m.name, "arch": m.arch, "head": "task"},
                build_fn=build_fn,
                **ft_kw,
            )
            task_path = mdir / "best_task.pt"
            task_tmp.replace(task_path)

        if m.name == "shallow":
            shutil.copy2(three_path, out_dir / "best_three.pt")
            if task_path is not None:
                shutil.copy2(task_path, out_dir / "best_task.pt")

        entry = {"three_ckpt": _rel_repo(three_path)}
        if task_path is not None:
            entry["task_ckpt"] = _rel_repo(task_path)
        overlay_members[m.name] = entry
        members_out[m.name] = {
            "arch": m.arch,
            "three_init": str(three_init),
            "three_out": str(three_path),
            "three_acc_heldout": float(three_rep["acc_after_heldout"]),
            "three_acc_train": float(three_rep["acc_after_train"]),
            "task_out": str(task_path) if task_path else None,
            "task_acc_heldout": (
                float(task_rep["acc_after_heldout"]) if task_rep else None
            ),
        }

    # 融合评估（临时 registry）
    base_reg = E1fRegistry(stack, device=device)
    ft_stack = stack.with_member_overrides(overlay_members).resolve_paths(
        repo_root=REPO_ROOT
    )
    ft_reg = E1fRegistry(ft_stack, device=device)
    before = _eval_e1f_three(base_reg, X_te, y3_te)
    after_tr = _eval_e1f_three(ft_reg, X_tr, y3_tr)
    after_te = _eval_e1f_three(ft_reg, X_te, y3_te)

    fusion_rep = {
        "acc_before_heldout": before["acc"],
        "acc_after_train": after_tr["acc"],
        "acc_after_heldout": after_te["acc"],
        "heldout_pred_dist": after_te["pred_dist"],
        "n_train": after_tr["n"],
        "n_heldout": after_te["n"],
    }
    gate = evaluate_release_gate(fusion_rep)
    status = "PASS" if gate["pass"] else "FAIL"
    release_pass = bool(gate["pass"])

    overlay = {
        "schema": "e1f_overlay_v1",
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "base_stack": _rel_repo(stack_path),
        "ft_scope": "all4",
        "members": overlay_members,
        "source_ft_run": out_dir.name,
    }
    _write_json(out_dir / "e1f_overlay.json", overlay)

    meta = {
        "ft_scope": "all4",
        "readout_mode": "e1f",
        "primary_judge_mode": "majority",
        "status": status,
        "protocol": resolved_protocol,
        "leave_next": leave_next,
        "sessions": [d.name for d in session_dirs],
        "heldout_sessions": [d.name for d in hold_dirs],
        "device": device,
        "members": members_out,
        "fusion_heldout_acc": fusion_rep["acc_after_heldout"],
        "fusion_train_acc": fusion_rep["acc_after_train"],
        "release_gate": gate,
        "created_at": overlay["created_at"],
    }
    _write_json(out_dir / "meta.json", meta)
    _write_json(
        out_dir / "release_gate.json",
        {"status": status, "pass": release_pass, **gate, "fusion": fusion_rep},
    )

    report_lines = [
        f"# E1f all4 FT — {out_dir.name}",
        "",
        f"- status: **{status}**",
        (
            f"- fusion heldout: `{fusion_rep['acc_before_heldout']:.3f}` → "
            f"`{fusion_rep['acc_after_heldout']:.3f}`"
        ),
        (
            f"- train: `{fusion_rep['acc_after_train']:.3f}` | "
            f"max_class_frac: `{gate['max_class_frac']:.3f}`"
        ),
        f"- sessions: `{meta['sessions']}` heldout: `{meta['heldout_sessions']}`",
        "",
        "## Members",
    ]
    for name, info in members_out.items():
        line = (
            f"- **{name}** ({info['arch']}): three heldout "
            f"`{info['three_acc_heldout']:.3f}`"
        )
        if info.get("task_acc_heldout") is not None:
            line += f", task `{info['task_acc_heldout']:.3f}`"
        report_lines.append(line)
    report_lines.append("")
    report = "\n".join(report_lines) + "\n"
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    if verbose:
        print(
            f"[all4] done status={status} fusion_heldout={fusion_rep['acc_after_heldout']:.3f}",
            flush=True,
        )

    # 兼容 orchestrator/operator：对齐 run_subject_finetune 返回形状
    shallow = members_out.get("shallow") or {}
    three_rep_compat = {
        "acc_after_heldout": float(fusion_rep["acc_after_heldout"]),
        "acc_after_train": float(fusion_rep["acc_after_train"]),
        "heldout_pred_dist": fusion_rep["heldout_pred_dist"],
        "ft": {"ft_scope": "all4"},
    }
    task_rep_compat = {
        "acc_after_heldout": float(shallow.get("task_acc_heldout") or 0.0),
        "acc_after_train": 0.0,
        "ft": {},
    }

    return {
        "status": "ok",
        "out_dir": str(out_dir),
        "release_gate": gate,
        "release_pass": release_pass,
        "meta": meta,
        "overlay": overlay,
        "report": report,
        "three": three_rep_compat,
        "task": task_rep_compat,
        "ft_scope": "all4",
        "force_promote_recommended": not release_pass,
    }
