# -*- coding: utf-8 -*-
"""Exp36 Day2 C1：OpenBMI 45ch 预训练 → 官方 45ch FT（conformer）。

用法：
  python train_c1.py --stage preprocess
  python train_c1.py --stage pretrain
  python train_c1.py --stage ft
  python train_c1.py --stage all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGConformer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_STEP = Path(__file__).resolve().parent
_STEP_PARENT = _STEP.parent
_PREPROCESS = Path(__file__).resolve().parents[4] / "preprocess_lab"
for p in (_STEP, _STEP_PARENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_paths import resolve_data  # noqa: E402
from exp36_config import OUT_ROOT_TAG, exp36_out  # noqa: E402

# reuse LOSO from A59
sys.path.insert(0, str(_STEP_PARENT / "5070_challenge_mi_59ch_accpaper"))
from loso import iter_loso6  # noqa: E402
from metrics_three import three_class_report  # noqa: E402

# label remap: challenge[i] <- openbmi[PERM[i]]
OPENBMI_TO_CHALLENGE_PERM = (1, 2, 0)

CONFORMER_NUM_LAYERS = 2
CONFORMER_NUM_HEADS = 10
CONFORMER_ATT_DROP = 0.5


@dataclass
class C1HP:
    n_chans: int = 45
    n_times: int = 750
    n_outputs: int = 3
    seed: int = 42
    max_epochs_pretrain: int = 80
    max_epochs_ft: int = 200
    patience: int = 15
    batch_train: int = 16
    batch_eval: int = 32
    grad_accum: int = 8
    lr_pretrain: float = 1e-4
    lr_ft: float = 5e-5
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    use_amp: bool = True
    val_subject_ratio: float = 0.15
    remap_labels_on_ft: bool = True


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = np.asarray(self.X[idx], dtype=np.float32)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        return torch.from_numpy(x.copy()), int(self.y[idx])


def build_conformer(n_chans: int, n_times: int, n_outputs: int, drop_prob: float) -> nn.Module:
    return EEGConformer(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        final_fc_length="auto",
        drop_prob=drop_prob,
        num_layers=CONFORMER_NUM_LAYERS,
        num_heads=CONFORMER_NUM_HEADS,
        att_drop_prob=CONFORMER_ATT_DROP,
    )


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
def evaluate(model, loader, device, use_amp: bool) -> tuple[dict, float]:
    model.eval()
    losses, yt, yp = [], [], []
    crit = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(xb)
            loss = crit(logits, yb)
        losses.append(float(loss.item()))
        yt.extend(yb.detach().cpu().tolist())
        yp.extend(logits.argmax(1).detach().cpu().tolist())
    return three_class_report(np.asarray(yt), np.asarray(yp)), float(np.mean(losses) if losses else 0.0)


def remap_classifier_out3(state: dict) -> dict:
    perm = list(OPENBMI_TO_CHALLENGE_PERM)
    out = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        kl = k.lower()
        is_head = any(s in kl for s in ("final", "classif", "fc", "conv_classifier", "linear"))
        if is_head and v.ndim >= 1 and int(v.shape[0]) == 3:
            out[k] = v[perm].clone()
        else:
            out[k] = v
    return out


def train_loop(
    *,
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    device: torch.device,
    hp: C1HP,
    max_epochs: int,
    lr: float,
    out_ckpt: Path,
    seed: int,
) -> dict:
    train_loader = DataLoader(
        WindowDataset(X, y, tr_idx),
        batch_size=hp.batch_train,
        sampler=_balanced_sampler(y, tr_idx, seed),
        num_workers=0,
    )
    val_loader = DataLoader(
        WindowDataset(X, y, va_idx),
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=hp.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=hp.use_amp and device.type == "cuda")
    crit = nn.CrossEntropyLoss()
    best_acc, best_epoch, best_val, bad = -1.0, 0, {}, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running, n_seen, step_in_accum = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=hp.use_amp and device.type == "cuda"):
                loss = crit(model(xb), yb) / float(hp.grad_accum)
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

        val_m, val_loss = evaluate(model, val_loader, device, hp.use_amp)
        acc = float(val_m["acc"])
        print(
            f"  ep{epoch:03d} loss={running/max(n_seen,1):.4f} "
            f"val_loss={val_loss:.4f} val_acc={acc:.4f}",
            flush=True,
        )
        if acc > best_acc + 1e-6:
            best_acc, best_epoch, best_val, bad = acc, epoch, val_m, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_m,
                    "n_chans": hp.n_chans,
                    "n_times": hp.n_times,
                    "n_outputs": hp.n_outputs,
                    "label_space": "openbmi_or_challenge",
                },
                out_ckpt,
            )
        else:
            bad += 1
            if bad >= hp.patience:
                print(f"  early stop @ ep{epoch} best={best_acc:.4f}", flush=True)
                break
    return {"best_acc": best_acc, "best_epoch": best_epoch, "best_val": best_val, "ckpt": str(out_ckpt)}


def subject_holdout_split(subjects: np.ndarray, *, val_ratio: float, seed: int):
    uniq = sorted(set(str(s) for s in subjects.tolist()))
    rng = np.random.default_rng(seed)
    order = list(uniq)
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_ratio)))
    val_set = set(order[:n_val])
    tr_idx = np.asarray([i for i, s in enumerate(subjects.tolist()) if str(s) not in val_set], dtype=np.int64)
    va_idx = np.asarray([i for i, s in enumerate(subjects.tolist()) if str(s) in val_set], dtype=np.int64)
    return tr_idx, va_idx, sorted(val_set)


def stage_preprocess(*, limit: int | None) -> None:
    py = sys.executable
    # slice challenge 59→45
    subprocess.check_call(
        [py, "-m", "src.datasets.challenge_mi.slice_45ch"],
        cwd=str(_PREPROCESS),
    )
    cmd = [py, "-m", "src.datasets.openbmi.batch_3s_fixed_45ch"]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    subprocess.check_call(cmd, cwd=str(_PREPROCESS))


def stage_pretrain(hp: C1HP, run_tag: str) -> Path:
    data_dir, prefix = resolve_data("openbmi_3s_fixed_45ch")
    X = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
    y = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    print(f"pretrain data {tuple(X.shape)} from {data_dir}", flush=True)
    if X.shape[2] != hp.n_chans:
        raise RuntimeError(f"expected {hp.n_chans} ch, got {X.shape}")

    tr_idx, va_idx, val_subs = subject_holdout_split(
        subjects, val_ratio=hp.val_subject_ratio, seed=hp.seed
    )
    print(f"pretrain split n={len(tr_idx)}/{len(va_idx)} val_subjects={val_subs}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("device", torch.cuda.get_device_name(0), flush=True)
    model = build_conformer(hp.n_chans, hp.n_times, hp.n_outputs, hp.drop_prob).to(device)
    out_dir = exp36_out() / "C1" / f"pretrain_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best_openbmi_conformer.pt"
    t0 = time.time()
    summary = train_loop(
        model=model,
        X=X,
        y=y,
        tr_idx=tr_idx,
        va_idx=va_idx,
        device=device,
        hp=hp,
        max_epochs=hp.max_epochs_pretrain,
        lr=hp.lr_pretrain,
        out_ckpt=ckpt,
        seed=hp.seed,
    )
    summary.update(
        {
            "stage": "pretrain",
            "data_tag": "openbmi_3s_fixed_45ch",
            "val_subjects": val_subs,
            "elapsed_sec": round(time.time() - t0, 1),
            "hp": asdict(hp),
            "label_space": "openbmi Rest0 L1 R2",
        }
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("pretrain done", summary["best_acc"], "→", ckpt, flush=True)
    return ckpt


def stage_ft(hp: C1HP, run_tag: str, init_ckpt: Path, max_folds: int = 6) -> Path:
    data_dir, prefix = resolve_data("challenge_mi_3s_45ch")
    X = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
    y = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    print(f"ft data {tuple(X.shape)} from {data_dir}", flush=True)

    init_obj = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    init_state = init_obj["model"] if isinstance(init_obj, dict) and "model" in init_obj else init_obj
    if hp.remap_labels_on_ft:
        init_state = remap_classifier_out3(init_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = (
        exp36_out()
        / "C1"
        / f"ft_conformer_{run_tag}"
        / "challenge_mi_3s_45ch"
        / f"run_{run_tag}"
        / "three"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    t0 = time.time()
    for fold_info in iter_loso6(subjects):
        fold = int(fold_info["fold"])
        if fold >= max_folds:
            break
        fold_dir = run_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        tr_idx = np.where(fold_info["masks"]["train"])[0]
        va_idx = np.where(fold_info["masks"]["val"])[0]
        print(
            f"\n======== [C1/ft] fold{fold} n={len(tr_idx)}/{len(va_idx)} ========",
            flush=True,
        )
        model = build_conformer(hp.n_chans, hp.n_times, hp.n_outputs, hp.drop_prob).to(device)
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        print(f"  init missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}", flush=True)

        ckpt_path = fold_dir / "best_three.pt"
        summary = train_loop(
            model=model,
            X=X,
            y=y,
            tr_idx=tr_idx,
            va_idx=va_idx,
            device=device,
            hp=hp,
            max_epochs=hp.max_epochs_ft,
            lr=hp.lr_ft,
            out_ckpt=ckpt_path,
            seed=hp.seed + fold,
        )
        # dump val probs
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        val_loader = DataLoader(
            WindowDataset(X, y, va_idx), batch_size=hp.batch_eval, shuffle=False
        )
        probs, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                with torch.amp.autocast("cuda", enabled=hp.use_amp and device.type == "cuda"):
                    p = torch.softmax(model(xb).float(), dim=1).cpu().numpy()
                probs.append(p)
                labels.extend(yb.tolist())
        np.save(fold_dir / "val_prob.npy", np.concatenate(probs).astype(np.float32))
        np.save(fold_dir / "val_y.npy", np.asarray(labels, dtype=np.int64))
        np.save(fold_dir / "val_idx.npy", va_idx.astype(np.int64))
        rec = {
            "fold": fold,
            "val_subjects": fold_info["val_subjects"],
            "best_epoch": summary["best_epoch"],
            "best_val_acc": summary["best_acc"],
            "best_val_metrics": summary["best_val"],
            "init_ckpt": str(init_ckpt),
            "ckpt": str(ckpt_path),
        }
        (fold_dir / "summary.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        folds.append(rec)
        print(f"  fold{fold} best={summary['best_acc']:.4f}", flush=True)

    accs = [float(f["best_val_acc"]) for f in folds]
    meta = {
        "experiment": 36,
        "arm": "C1",
        "model": "conformer",
        "init_ckpt": str(init_ckpt),
        "n_folds": len(folds),
        "val_acc_mean": float(np.mean(accs)) if accs else None,
        "val_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "folds": folds,
        "elapsed_sec": round(time.time() - t0, 1),
        "time": datetime.now().isoformat(timespec="seconds"),
        "hp": asdict(hp),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"run_dir": str(run_dir), "arm": "C1"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"C1 FT mean={meta['val_acc_mean']:.4f}±{meta['val_acc_std']:.4f} → {run_dir}",
        flush=True,
    )
    return run_dir


def find_latest_pretrain() -> Path | None:
    root = exp36_out() / "C1"
    if not root.is_dir():
        return None
    cands = sorted(root.glob("pretrain_*/best_openbmi_conformer.pt"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["preprocess", "pretrain", "ft", "all"],
        default="all",
    )
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--limit-mats", type=int, default=None, help="preprocess smoke limit")
    ap.add_argument("--init-ckpt", type=str, default="")
    ap.add_argument("--max-folds", type=int, default=6)
    ap.add_argument("--max-epochs-pretrain", type=int, default=0)
    ap.add_argument("--max-epochs-ft", type=int, default=0)
    args = ap.parse_args()

    hp = C1HP()
    if args.max_epochs_pretrain > 0:
        hp = replace(hp, max_epochs_pretrain=args.max_epochs_pretrain)
    if args.max_epochs_ft > 0:
        hp = replace(hp, max_epochs_ft=args.max_epochs_ft)
    run_tag = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.stage in ("preprocess", "all"):
        stage_preprocess(limit=args.limit_mats)
    init_ckpt = Path(args.init_ckpt) if args.init_ckpt else None
    if args.stage in ("pretrain", "all"):
        init_ckpt = stage_pretrain(hp, run_tag)
    if args.stage in ("ft", "all"):
        if init_ckpt is None or not init_ckpt.is_file():
            init_ckpt = find_latest_pretrain()
        if init_ckpt is None or not init_ckpt.is_file():
            raise SystemExit("需要 --init-ckpt 或先跑 pretrain")
        stage_ft(hp, run_tag, init_ckpt, max_folds=args.max_folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
