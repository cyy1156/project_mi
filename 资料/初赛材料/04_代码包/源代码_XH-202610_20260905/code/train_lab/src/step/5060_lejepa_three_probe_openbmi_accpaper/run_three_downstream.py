"""Three 下游（LeJEPA）：J0 同骨干监督 / J2 冻住探测 / J3 解冻微调。

用法:
  python run_three_downstream.py --arm j0 --max-folds 1
  python run_three_downstream.py --arm j2 --j1-dir .../run_xxx/j1
  python run_three_downstream.py --arm j2_random --max-folds 1
  python run_three_downstream.py --arm j3 --j1-dir .../run_xxx/j1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import (
    build_backbone,
    eval_acc_paper,
    load_openbmi,
    make_loader,
    materialize_windows,
    out_root,
    save_json,
    seed_everything,
    subject_masks,
    WindowDataset,
)
from lejepa_model import LeJepaClassifier, freeze_backbone, unfreeze_backbone
from shared_hparams import SHARED
from src.common.steps.split_subjects import iter_subject_kfold


def _balanced_sampler(y: np.ndarray, n_classes: int, generator: torch.Generator):
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts[y]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(y),
        replacement=True,
        generator=generator,
    )


def _load_j1_backbone(hp, fold: int, j1_dir: Path, device):
    path = j1_dir / f"fold{fold}_lejepa.pt"
    if not path.is_file():
        # 兼容误命名
        alt = j1_dir / f"fold{fold}_jepa.pt"
        path = alt if alt.is_file() else path
    if not path.is_file():
        raise FileNotFoundError(f"缺少 J1 权重: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    backbone = build_backbone(hp)
    backbone.load_state_dict(ckpt["backbone"], strict=True)
    return backbone.to(device)


def train_one_fold(
    *,
    arm: str,
    fold_info,
    data,
    hp,
    device,
    out_dir: Path,
    j1_dir: Path | None,
    max_train_windows: int = 0,
):
    fold = int(fold_info["fold"])
    X = data["X"]
    y = data["y_three"]
    subjects = data["subjects"]
    trial_ids = data["trial_ids"]
    x_path = data["x_path"]
    masks = subject_masks(
        subjects,
        fold_info["train_subjects"],
        fold_info["val_subjects"],
        fold_info["test_subjects"],
    )
    tr_idx = np.flatnonzero(masks["train"])
    if max_train_windows > 0 and len(tr_idx) > max_train_windows:
        rng = np.random.default_rng(hp.seed + fold + 17)
        tr_idx = np.sort(
            rng.choice(tr_idx, size=int(max_train_windows), replace=False)
        )
        print(f"[{arm}] smoke train subsample → {len(tr_idx)}", flush=True)
    print(
        f"\n======== [{arm}] fold{fold} "
        f"n={len(tr_idx)}/{masks['val'].sum()}/{masks['test'].sum()} ========",
        flush=True,
    )

    seed_everything(hp.seed + fold)
    if arm == "j0":
        backbone = build_backbone(hp).to(device)
        model = LeJepaClassifier(backbone, n_outputs=3, drop_prob=hp.drop_prob).to(device)
        enc_lr = hp.lr
    elif arm == "j2_random":
        backbone = build_backbone(hp).to(device)
        model = LeJepaClassifier(backbone, n_outputs=3, drop_prob=hp.drop_prob).to(device)
        freeze_backbone(model)
        enc_lr = 0.0
    elif arm == "j2":
        assert j1_dir is not None
        backbone = _load_j1_backbone(hp, fold, j1_dir, device)
        model = LeJepaClassifier(backbone, n_outputs=3, drop_prob=hp.drop_prob).to(device)
        freeze_backbone(model)
        enc_lr = 0.0
    elif arm == "j3":
        assert j1_dir is not None
        backbone = _load_j1_backbone(hp, fold, j1_dir, device)
        model = LeJepaClassifier(backbone, n_outputs=3, drop_prob=hp.drop_prob).to(device)
        unfreeze_backbone(model)
        enc_lr = hp.lr_encoder
    else:
        raise ValueError(arm)

    y_tr = y[tr_idx]
    g = torch.Generator().manual_seed(hp.seed + fold)
    print(f"[{arm}] materialize train {len(tr_idx)} → RAM …", flush=True)
    train_ds = materialize_windows(X, tr_idx, y=y)
    train_loader = make_loader(
        train_ds,
        hp.batch_train,
        shuffle=False,
        hp=hp,
        sampler=_balanced_sampler(y_tr, 3, g),
    )
    criterion = nn.CrossEntropyLoss()
    if enc_lr > 0 and any(p.requires_grad for p in model.backbone.parameters()):
        opt = torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.backbone.parameters() if p.requires_grad],
                    "lr": enc_lr,
                },
                {"params": list(model.head.parameters()), "lr": hp.lr},
            ],
            weight_decay=hp.weight_decay,
        )
    else:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

    best_score, best_state, best_ep = -1.0, None, 0
    bad = 0
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, hp.max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            if hp.use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(xb), yb)
                loss.backward()
            else:
                loss = criterion(model(xb), yb)
                loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        val_trial, val_win = eval_acc_paper(
            model, X, y, subjects, trial_ids, masks["val"], device, hp, x_path
        )
        score = float(val_trial["acc_paper"])
        print(
            f"fold{fold} ep {ep:03d}  tr={total/max(n,1):.4f}  "
            f"val_AccPaper={score:.4f}  win_BalAcc={float(val_win['balanced_accuracy']):.4f}"
        )
        if score > best_score:
            best_score = score
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            torch.save(
                {
                    "arm": arm,
                    "fold": fold,
                    "epoch": ep,
                    "val_acc_paper": score,
                    "model": best_state,
                },
                fold_dir / "best_three.pt",
            )
        else:
            bad += 1
            if bad >= hp.patience:
                print(f"  early stop @ ep {ep}")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    te_trial, te_win = eval_acc_paper(
        model, X, y, subjects, trial_ids, masks["test"], device, hp, x_path
    )
    print(
        f"[fold{fold}/test] Acc_paper={te_trial['acc_paper']:.4f}  "
        f"F1m={te_trial['f1_macro']:.4f}"
    )
    return {
        "fold": fold,
        "best_epoch": best_ep,
        "best_val_acc_paper": best_score,
        "test_trial_metrics": te_trial,
        "test_window_metrics": te_win,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=("j0", "j2", "j2_random", "j3"))
    p.add_argument("--j1-dir", type=str, default="", help="含 fold*_jepa.pt 的 j1 目录")
    p.add_argument("--resume-dir", type=str, default="", help="写入已有 run_ 目录")
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument(
        "--max-train-windows",
        type=int,
        default=0,
        help=">0 时训练窗子采样（冒烟；val/test 仍全量）",
    )
    args = p.parse_args()

    hp = SHARED
    repl = {}
    if args.max_epochs > 0:
        repl["max_epochs"] = args.max_epochs
    if args.patience > 0:
        repl["patience"] = args.patience
    if args.num_workers >= 0:
        repl["num_workers"] = args.num_workers
    if repl:
        hp = replace(hp, **repl)

    j1_dir = Path(args.j1_dir).expanduser().resolve() if args.j1_dir else None
    if args.arm in ("j2", "j3"):
        if j1_dir is None or not j1_dir.is_dir():
            raise SystemExit("--arm j2/j3 需要有效 --j1-dir")

    seed_everything(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_openbmi(hp)

    if args.resume_dir:
        root = Path(args.resume_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = out_root()
    out_dir = root / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{args.arm}] out={out_dir} device={device}")

    folds = []
    for i, info in enumerate(
        iter_subject_kfold(
            data["subjects"], n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
        )
    ):
        if args.max_folds > 0 and i >= args.max_folds:
            break
        folds.append(
            train_one_fold(
                arm=args.arm,
                fold_info=info,
                data=data,
                hp=hp,
                device=device,
                out_dir=out_dir,
                j1_dir=j1_dir,
                max_train_windows=int(args.max_train_windows),
            )
        )

    ap = [float(r["test_trial_metrics"]["acc_paper"]) for r in folds]
    f1 = [float(r["test_trial_metrics"]["f1_macro"]) for r in folds]
    summary = {
        "arm": args.arm,
        "shallow_ref_three_acc_paper": 0.5404,
        "test_acc_paper_mean": float(np.mean(ap)),
        "test_acc_paper_std": float(np.std(ap)),
        "test_f1_macro_mean": float(np.mean(f1)),
        "test_f1_macro_std": float(np.std(f1)),
        "delta_vs_shallow": float(np.mean(ap) - 0.5404),
        "folds": folds,
        "hparams": {k: getattr(hp, k) for k in hp.__dataclass_fields__},
        "j1_dir": str(j1_dir) if j1_dir else None,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(out_dir / "summary.json", summary)
    print(
        f"\n[{args.arm}] Test Acc_paper {summary['test_acc_paper_mean']:.4f}"
        f"±{summary['test_acc_paper_std']:.4f}  "
        f"Δvs_shallow={summary['delta_vs_shallow']:+.4f}"
    )


if __name__ == "__main__":
    main()
