"""Exp34 轨 A 训练入口：LOSO6 · 三分类 · 5070 batch/AMP/grad_accum。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_STEP = Path(__file__).resolve().parent
_STEP_PARENT = _STEP.parent
for p in (_STEP, _STEP_PARENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_paths import resolve_data  # noqa: E402
from loso import iter_loso6  # noqa: E402
from metrics_three import three_class_report  # noqa: E402
from shared_hparams import (  # noqa: E402
    OUT_ROOT_TAG,
    SHARED,
    SharedTrainHP,
    shared_as_dict,
)

BuildFn = Callable[[int, int, int, float], nn.Module]


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = np.asarray(self.X[idx], dtype=np.float32)  # (1,C,T) or (C,T)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 2:
            raise ValueError(f"bad x shape {x.shape}")
        return torch.from_numpy(x.copy()), int(self.y[idx])


def _n_chans_of(X: np.ndarray) -> int:
    # (N,1,C,T) or (N,C,T)
    if X.ndim == 4:
        return int(X.shape[2])
    if X.ndim == 3:
        return int(X.shape[1])
    raise ValueError(f"unexpected X.ndim={X.ndim}")


def _balanced_sampler(y: np.ndarray, indices: np.ndarray, seed: int) -> WeightedRandomSampler:
    y_sub = y[indices]
    classes, counts = np.unique(y_sub, return_counts=True)
    freq = {int(c): float(n) for c, n in zip(classes.tolist(), counts.tolist())}
    w = np.asarray([1.0 / freq[int(v)] for v in y_sub.tolist()], dtype=np.float64)
    w = w / w.sum() * len(w)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(w),
        replacement=True,
        generator=gen,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> tuple[dict, float]:
    model.eval()
    losses: list[float] = []
    yt: list[int] = []
    yp: list[int] = []
    crit = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(xb)
            loss = crit(logits, yb)
        losses.append(float(loss.item()))
        pred = logits.argmax(dim=1)
        yt.extend(yb.detach().cpu().tolist())
        yp.extend(pred.detach().cpu().tolist())
    report = three_class_report(np.asarray(yt), np.asarray(yp))
    return report, float(np.mean(losses) if losses else 0.0)


def train_one_fold(
    *,
    fold_info: dict,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    hp: SharedTrainHP,
    out_dir: Path,
    model_name: str,
    build_model: BuildFn,
) -> dict:
    fold = int(fold_info["fold"])
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tr_idx = np.where(fold_info["masks"]["train"])[0]
    va_idx = np.where(fold_info["masks"]["val"])[0]
    n_chans = _n_chans_of(X)
    n_times = int(X.shape[-1])
    if n_chans != hp.n_chans_expected:
        raise RuntimeError(f"n_chans={n_chans} != expected {hp.n_chans_expected}")
    if n_times != hp.n_times_expected:
        raise RuntimeError(f"n_times={n_times} != expected {hp.n_times_expected}")

    print(
        f"\n======== [{model_name}] fold{fold} ========\n"
        f"  train={fold_info['train_subjects']}\n"
        f"  val  ={fold_info['val_subjects']}\n"
        f"  n={len(tr_idx)}/{len(va_idx)}  ch={n_chans} T={n_times} "
        f"batch={hp.batch_train}x{hp.grad_accum}",
        flush=True,
    )

    train_ds = WindowDataset(X, y, tr_idx)
    val_ds = WindowDataset(X, y, va_idx)
    sampler = _balanced_sampler(y, tr_idx, hp.seed + fold)
    train_loader = DataLoader(
        train_ds,
        batch_size=hp.batch_train,
        sampler=sampler,
        num_workers=hp.num_workers,
        pin_memory=hp.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=hp.num_workers,
        pin_memory=hp.pin_memory,
    )

    model = build_model(n_chans, n_times, hp.n_outputs, hp.drop_prob).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=hp.use_amp and device.type == "cuda")
    crit = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_epoch = 0
    best_val = {}
    bad = 0
    ckpt_path = fold_dir / "best_three.pt"

    for epoch in range(1, hp.max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        n_seen = 0
        step_in_accum = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=hp.non_blocking)
            yb = yb.to(device, non_blocking=hp.non_blocking)
            with torch.amp.autocast(
                "cuda", enabled=hp.use_amp and device.type == "cuda"
            ):
                logits = model(xb)
                loss = crit(logits, yb) / float(hp.grad_accum)
            scaler.scale(loss).backward()
            step_in_accum += 1
            running += float(loss.item()) * float(hp.grad_accum) * len(yb)
            n_seen += len(yb)
            if step_in_accum >= hp.grad_accum:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step_in_accum = 0
        if step_in_accum > 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        val_m, val_loss = evaluate(model, val_loader, device, use_amp=hp.use_amp)
        tr_loss = running / max(n_seen, 1)
        acc = float(val_m["acc"])
        print(
            f"  ep{epoch:03d}  train_loss={tr_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={acc:.4f}  "
            f"macroR={val_m['macro_recall']:.4f}",
            flush=True,
        )
        if acc > best_acc + 1e-6:
            best_acc = acc
            best_epoch = epoch
            best_val = val_m
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_m,
                    "n_chans": n_chans,
                    "n_times": n_times,
                    "n_outputs": hp.n_outputs,
                    "model_name": model_name,
                    "fold": fold,
                    "hp": shared_as_dict() if hp is SHARED else dict(hp.__dict__),
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= hp.patience:
                print(f"  early stop @ ep{epoch} (best ep{best_epoch} acc={best_acc:.4f})")
                break

    # dump val probs for E1f later
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            with torch.amp.autocast(
                "cuda", enabled=hp.use_amp and device.type == "cuda"
            ):
                logits = model(xb)
                p = torch.softmax(logits.float(), dim=1).cpu().numpy()
            probs.append(p)
            labels.extend(yb.tolist())
    prob = np.concatenate(probs, axis=0) if probs else np.zeros((0, hp.n_outputs))
    np.save(fold_dir / "val_prob.npy", prob.astype(np.float32))
    np.save(fold_dir / "val_y.npy", np.asarray(labels, dtype=np.int64))
    np.save(fold_dir / "val_idx.npy", va_idx.astype(np.int64))

    result = {
        "fold": fold,
        "val_subjects": fold_info["val_subjects"],
        "train_subjects": fold_info["train_subjects"],
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "best_val_metrics": best_val,
        "ckpt": str(ckpt_path),
    }
    with (fold_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def run_baseline_main(
    *,
    model_name: str,
    build_model: BuildFn,
    structure_note: str = "",
    extra_meta: dict | None = None,
    hp: SharedTrainHP | None = None,
) -> Path:
    hp = hp or SHARED
    parser = argparse.ArgumentParser(description=f"Exp34 {model_name}")
    parser.add_argument("--max-folds", type=int, default=0, help="0=全部；1=只跑 fold0")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--data-tag", type=str, default=hp.data_tag)
    parser.add_argument(
        "--out-root-tag",
        type=str,
        default="",
        help="覆盖 out/ 下实验根目录名（默认 shared_hparams.OUT_ROOT_TAG）",
    )
    args, _unknown = parser.parse_known_args()
    out_root_tag = args.out_root_tag.strip() or OUT_ROOT_TAG

    torch.set_num_threads(int(hp.torch_num_threads))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(hp.cudnn_benchmark)
        device = torch.device("cuda")
        print("device", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("WARN: CUDA 不可用，使用 CPU")

    data_dir, prefix = resolve_data(args.data_tag)
    x_path = data_dir / f"{prefix}_X.npy"
    y_path = data_dir / f"{prefix}_y_three.npy"
    s_path = data_dir / f"{prefix}_subjects.npy"
    for p in (x_path, y_path, s_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"缺少 {p}；请先运行: "
                f"python -m src.datasets.challenge_mi.batch_3s --mode 59"
            )

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    subjects = np.load(s_path, allow_pickle=True)
    print(f"data X={tuple(X.shape)} y={tuple(y.shape)} from {data_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.run_tag.strip() or stamp
    run_dir = (
        Path(__file__).resolve().parents[3]
        / "out"
        / out_root_tag
        / f"{model_name}_challenge_mi_3s_59ch"
        / args.data_tag
        / f"run_{tag}"
        / "three"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model_name": model_name,
        "structure_note": structure_note,
        "hp": shared_as_dict() if hp is SHARED else dict(hp.__dict__),
        "data_dir": str(data_dir),
        "started": stamp,
        "device_label": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "extra": extra_meta or {},
        "experiment": 34,
        "track": "A59",
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    fold_results = []
    t0 = time.time()
    for info in iter_loso6(subjects):
        if args.max_folds > 0 and info["fold"] >= args.max_folds:
            break
        fold_results.append(
            train_one_fold(
                fold_info=info,
                X=X,
                y=y,
                device=device,
                hp=hp,
                out_dir=run_dir,
                model_name=model_name,
                build_model=build_model,
            )
        )

    accs = [float(r["best_val_acc"]) for r in fold_results]
    summary = {
        "model_name": model_name,
        "n_folds": len(fold_results),
        "val_acc_mean": float(np.mean(accs)) if accs else None,
        "val_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "folds": fold_results,
        "elapsed_sec": float(time.time() - t0),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"\n[done] {model_name} folds={len(fold_results)} "
        f"val_acc={summary['val_acc_mean']:.4f}±{summary['val_acc_std']:.4f} "
        f"→ {run_dir}"
    )
    return run_dir
