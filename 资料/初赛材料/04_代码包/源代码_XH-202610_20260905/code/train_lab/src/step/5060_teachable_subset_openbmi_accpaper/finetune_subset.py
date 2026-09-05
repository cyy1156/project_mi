"""B2：在 train 折内 teachable(T1) 试次上冻骨干微调分类头（可选）。

仅在 B1 达门槛后使用；不覆盖正式权重。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
CODE_ROOT = HERE.parents[3]
TRAIN_LAB = CODE_ROOT / "train_lab"
PRE_ROOT = CODE_ROOT / "preprocess_lab"
REPO = CODE_ROOT.parent
HOP100 = STEP / "baselines_2s_hop100"
OLD = STEP / "baselines_single"

for p in (STEP, PRE_ROOT, HOP100, OLD):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
_here = str(HERE)
if _here in sys.path:
    sys.path.remove(_here)
sys.path.insert(0, _here)

from data_paths import resolve_data
from perf_loader import apply_runtime_threads, configure_cuda_backends, make_loader
from shared_hparams import FORMAL_SHALLOW_RUN, SHARED, SharedTrainHP
from src.common.steps.split_subjects import iter_subject_kfold
from teachable_io import load_masks, resolve_teachable_paths
from trial_metrics import aggregate_windows_to_trials

# 复用 B1 的模型/预测
from eval_subset import IndexDS, _jsonable, _mean_std, build_model, predict, resolve_run_dir


def freeze_backbone(model: nn.Module, mode: str) -> list[nn.Parameter]:
    """mode=head：只训 final_layer；mode=full：全员可训。"""
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return [p for p in model.parameters() if p.requires_grad]
    for name, p in model.named_parameters():
        p.requires_grad = "final_layer" in name
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("未找到 final_layer 可训参数")
    return trainable


def run_epoch(model, loader, criterion, optimizer, device, train: bool, use_amp: bool):
    model.train(train)
    total, n = 0.0, 0
    amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if train and amp else None
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if amp:
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(x)
                    if logits.ndim > 2:
                        logits = logits.reshape(logits.shape[0], -1)
                    loss = criterion(logits, y)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                logits = model(x)
                if logits.ndim > 2:
                    logits = logits.reshape(logits.shape[0], -1)
                loss = criterion(logits, y)
                if train:
                    loss.backward()
                    optimizer.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


def eval_mask(model, X, y, subjects, trial_ids, mask, device, hp, x_path, n_classes):
    indices = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)
    if len(indices) == 0:
        return {"empty": True, "acc_paper": float("nan"), "n_trials": 0}
    loader = make_loader(
        IndexDS(X, y, indices, x_path=x_path),
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=hp.num_workers,
        pin_memory=hp.pin_memory,
        persistent_workers=hp.persistent_workers and hp.num_workers > 0,
        prefetch_factor=hp.prefetch_factor,
    )
    yt, yp = predict(model, loader, device)
    m = aggregate_windows_to_trials(
        yt, yp, subjects[indices], trial_ids[indices], n_classes=n_classes
    )["metrics"]
    return {
        "empty": False,
        "acc_paper": float(m["acc_paper"]),
        "n_trials": int(m["n_trials"]),
        "n_windows": int(m["n_windows"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="B2 子集冻骨干微调")
    p.add_argument("--model", choices=("shallow",), default="shallow")
    p.add_argument("--task", choices=("task", "three"), default="three")
    p.add_argument("--ft-mode", choices=("head", "full"), default="head")
    p.add_argument("--ft-pool", choices=("teachable", "high_lat"), default="teachable")
    p.add_argument("--run-dir", default="")
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--teachable-mask", default="")
    args = p.parse_args()

    hp = SHARED
    apply_runtime_threads(hp.torch_num_threads)
    configure_cuda_backends(
        cudnn_benchmark=hp.cudnn_benchmark, deterministic=hp.deterministic
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = args.max_epochs or hp.max_epochs_ft
    lr = args.lr or (1e-5 if args.ft_mode == "full" else hp.ft_lr)

    _, mp = resolve_teachable_paths(None, args.teachable_mask or None)
    data_dir, prefix = resolve_data(hp.data_tag)
    x_path = str(data_dir / f"{prefix}_X.npy")
    X = np.load(x_path, mmap_mode="r")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    trial_ids = np.load(data_dir / f"{prefix}_trial_id.npy")
    y = np.load(
        data_dir / (f"{prefix}_y_task.npy" if args.task == "task" else f"{prefix}_y_three.npy")
    )
    masks = load_masks(len(X), mp)
    pool_mask = masks["teachable"] if args.ft_pool == "teachable" else masks["high_lat_eval"]

    run_dir = resolve_run_dir(args.model, args.run_dir or None)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = (
        TRAIN_LAB
        / "out"
        / "5060_teachable_subset_openbmi_accpaper"
        / f"ft_{args.model}_{args.task}_{args.ft_mode}"
        / stamp
    )
    out_run.mkdir(parents=True, exist_ok=True)
    md_dir = REPO / "资料" / "模型训练" / "runs" / "5060_teachable_subset"
    md_dir.mkdir(parents=True, exist_ok=True)

    n_classes = 2 if args.task == "task" else 3
    ckpt_name = "best_task.pt" if args.task == "task" else "best_three.pt"
    fold_iter = list(
        iter_subject_kfold(
            subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
        )
    )
    if args.max_folds > 0:
        fold_iter = fold_iter[: args.max_folds]

    fold_rows = []
    for fd in fold_iter:
        fold = int(fd["fold"])
        tr_m = fd["masks"]["train"] & pool_mask
        va_m = fd["masks"]["val"]
        te_m = fd["masks"]["test"]
        ckpt_path = run_dir / args.task / f"fold{fold}" / ckpt_name
        model = build_model(args.model, n_classes, hp.n_times_expected, hp.drop_prob).to(
            device
        )
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        model.load_state_dict(state)
        params = freeze_backbone(model, args.ft_mode)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=hp.weight_decay)
        crit = nn.CrossEntropyLoss()

        tr_idx = np.flatnonzero(tr_m).astype(np.int64)
        if len(tr_idx) < 32:
            print(f"[fold{fold}] FT 池过小 n={len(tr_idx)}，跳过", flush=True)
            continue
        tr_loader = make_loader(
            IndexDS(X, y, tr_idx, x_path=x_path),
            batch_size=hp.batch_train,
            shuffle=True,
            num_workers=hp.num_workers,
            pin_memory=hp.pin_memory,
            persistent_workers=hp.persistent_workers and hp.num_workers > 0,
            prefetch_factor=hp.prefetch_factor,
        )

        best_val = -1.0
        best_state = None
        bad = 0
        for ep in range(1, max_epochs + 1):
            tr_loss = run_epoch(
                model, tr_loader, crit, opt, device, True, hp.use_amp
            )
            val_m = eval_mask(
                model, X, y, subjects, trial_ids, va_m, device, hp, x_path, n_classes
            )
            val_acc = float(val_m["acc_paper"])
            print(
                f"[fold{fold}] ep{ep} tr_loss={tr_loss:.4f} val_acc_paper={val_acc:.4f}",
                flush=True,
            )
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= hp.patience_ft:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        fold_dir = out_run / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "fold": fold,
                "ft_mode": args.ft_mode,
                "ft_pool": args.ft_pool,
                "base_ckpt": str(ckpt_path),
            },
            fold_dir / f"best_ft_{args.task}.pt",
        )

        row = {"fold": fold, "best_val_acc_paper": best_val, "n_ft_windows": int(len(tr_idx))}
        for tag, mask in (
            ("R0", te_m),
            ("R1", te_m & masks["obvious12"]),
            ("R2", te_m & masks["high_lat_eval"]),
        ):
            row[tag] = eval_mask(
                model, X, y, subjects, trial_ids, mask, device, hp, x_path, n_classes
            )
            print(
                f"[fold{fold}/{tag}] acc_paper={row[tag]['acc_paper']} n={row[tag]['n_trials']}",
                flush=True,
            )
        fold_rows.append(row)

    summary = {
        "model": args.model,
        "task": args.task,
        "ft_mode": args.ft_mode,
        "ft_pool": args.ft_pool,
        "lr": lr,
        "run_dir_base": str(run_dir),
        "folds": fold_rows,
        "mean": {
            k: {
                "acc_paper_mean": _mean_std([r[k]["acc_paper"] for r in fold_rows])[0],
                "acc_paper_std": _mean_std([r[k]["acc_paper"] for r in fold_rows])[1],
            }
            for k in ("R0", "R1", "R2")
            if fold_rows
        },
    }
    (out_run / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# B2 子集微调 · {args.model} · {args.task} · {args.ft_mode}",
        "",
        f"- 时间：`{stamp}`",
        f"- 起点：`{run_dir}`",
        f"- FT 池：`{args.ft_pool}` · lr={lr}",
        "",
        "| 行 | Acc_paper |",
        "|----|-----------|",
    ]
    for k, v in summary.get("mean", {}).items():
        lines.append(f"| {k} | {v['acc_paper_mean']:.4f}±{v['acc_paper_std']:.4f} |")
    md_path = md_dir / f"{stamp}_{args.model}_{args.task}_B2_ft_{args.ft_mode}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {out_run}", flush=True)
    print(f"[done] {md_path}", flush=True)


if __name__ == "__main__":
    main()
