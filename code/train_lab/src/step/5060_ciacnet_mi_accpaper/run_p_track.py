"""P-track runner: paper config, 8ch only deviation.

Examples:
  python run_p_track.py --arm P0 --subjects A01 --repeats 1 --max-epochs 2
  python run_p_track.py --arm P1
  python run_p_track.py --arm P2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# package-local imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ciacnet_model import Ablation, build_ciacnet  # noqa: E402
from data_bci2a_p import DEFAULT_CACHE, build_or_load_cache, load_subject, zscore_from_train  # noqa: E402
from eegnet_ref import build_eegnet  # noqa: E402
from shared_hparams import P, p_dict  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
OUT_ROOT = REPO / "code" / "train_lab" / "out" / "5060_ciacnet_mi_accpaper"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[int(t), int(p)] += 1
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / (n * n)
    if abs(1 - pe) < 1e-12:
        return 0.0
    return float((po - pe) / (1 - pe))


def _batch_for_model(xb: torch.Tensor, arm: str) -> torch.Tensor:
    """xb stored as (B,1,C,T). EEGNet wants (B,C,T); CIACNet wants (B,1,C,T)."""
    if arm == "P1":
        return xb.squeeze(1)
    return xb


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, arm: str):
    model.eval()
    total_loss = 0.0
    n = 0
    ys, ps = [], []
    for xb, yb in loader:
        xb = _batch_for_model(xb.to(device, non_blocking=True), arm)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * yb.size(0)
        n += yb.size(0)
        pred = logits.argmax(dim=1)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = float((y_true == y_pred).mean())
    kappa = cohen_kappa(y_true, y_pred)
    return total_loss / max(n, 1), acc, kappa


def train_one_run(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    y_ev: np.ndarray,
    *,
    arm: str,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    lr: float,
    early_stop_patience: int,
    reduce_lr_patience: int,
    reduce_lr_factor: float,
    min_lr: float,
    conv_l2_on: bool,
) -> dict:
    # model expects (B,1,C,T) or (B,C,T); store as (N,1,C,T)
    tr_x = torch.from_numpy(X_tr[:, None, :, :])
    tr_y = torch.from_numpy(y_tr.astype(np.int64))
    ev_x = torch.from_numpy(X_ev[:, None, :, :])
    ev_y = torch.from_numpy(y_ev.astype(np.int64))

    train_loader = DataLoader(
        TensorDataset(tr_x, tr_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        TensorDataset(ev_x, ev_y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=min_lr
    )

    best_acc = -1.0
    best_kappa = 0.0
    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    bad_epochs = 0  # early stop on val_loss (paper)
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = _batch_for_model(xb.to(device, non_blocking=True), arm)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if conv_l2_on and hasattr(model, "loss_reg"):
                loss = loss + model.loss_reg()
            loss.backward()
            opt.step()
            running += float(loss.item()) * yb.size(0)
            n += yb.size(0)

        val_loss, val_acc, val_kappa = evaluate(model, eval_loader, device, criterion, arm)
        train_loss = running / max(n, 1)
        sched.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_kappa": val_kappa,
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )

        # paper: save when accuracy improves
        if val_acc > best_acc + 1e-12:
            best_acc = val_acc
            best_kappa = val_kappa
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # early stop / plateau tracked on validation loss
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"  ep {epoch:4d}  tr_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"acc={val_acc:.4f}  kappa={val_kappa:.4f}  best={best_acc:.4f}@{best_epoch}  "
                f"bad={bad_epochs}",
                flush=True,
            )

        if bad_epochs >= early_stop_patience:
            print(f"  early stop at epoch {epoch} (val_loss patience={early_stop_patience})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    # final eval with best weights
    _, final_acc, final_kappa = evaluate(model, eval_loader, device, criterion, arm)
    return {
        "best_acc": float(best_acc if best_acc >= 0 else final_acc),
        "best_kappa": float(best_kappa if best_acc >= 0 else final_kappa),
        "best_epoch": int(best_epoch),
        "final_acc": float(final_acc),
        "final_kappa": float(final_kappa),
        "epochs_ran": len(history),
        "history_tail": history[-5:],
        "state_dict": best_state,
    }


def build_model(arm: str, n_chans: int, n_times: int, n_outputs: int, dropout: float, ablation: Ablation):
    if arm in ("P0", "P2") or arm.startswith("P3"):
        return build_ciacnet(n_chans, n_times, n_outputs, drop_prob=dropout, ablation=ablation)
    if arm == "P1":
        # braindecode EEGNet expects (B, C, T)
        return build_eegnet(n_chans, n_times, n_outputs, drop_prob=dropout)
    raise ValueError(arm)


def arm_to_ablation(arm: str) -> Ablation:
    return {
        "P0": "full",
        "P1": "full",
        "P2": "full",
        "P3a": "no_cv2",
        "P3b": "no_iat",
        "P3c": "no_tc",
        "P3d": "std_cbam",
    }.get(arm, "full")  # type: ignore


def run_arm(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    cache = build_or_load_cache(Path(args.cache) if args.cache else DEFAULT_CACHE, force=args.rebuild_cache)
    subjects = args.subjects or [f"A0{i}" for i in range(1, 10)]
    ablation = arm_to_ablation(args.arm)
    max_epochs = args.max_epochs if args.max_epochs is not None else (2 if args.arm == "P0" else P.max_epochs)
    repeats = args.repeats if args.repeats is not None else (1 if args.arm == "P0" else P.n_repeats)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_ROOT / args.arm / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "arm": args.arm,
        "ablation": ablation,
        "device": str(device),
        "hparams": p_dict(),
        "overrides": {
            "max_epochs": max_epochs,
            "repeats": repeats,
            "subjects": subjects,
            "dropout": args.dropout,
        },
        "cache": str(cache),
        "note": "8ch paper-config P-track; Z-score from train set per subject",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log_path = run_dir / "run.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"=== {args.arm} start device={device} epochs={max_epochs} repeats={repeats} ===")
    subject_results = []

    for sub in subjects:
        X_tr, y_tr, X_ev, y_ev = load_subject(cache, sub)
        if X_tr.shape[-1] != P.n_times:
            log(f"WARN {sub}: n_times={X_tr.shape[-1]} expected {P.n_times}")
        X_tr, X_ev, zstats = zscore_from_train(X_tr, X_ev)
        n_times = int(X_tr.shape[-1])
        n_chans = int(X_tr.shape[1])
        run_rows = []
        best_row = None
        sub_dir = run_dir / sub
        sub_dir.mkdir(exist_ok=True)

        for r in range(repeats):
            seed = P.seed_base + hash((args.arm, sub, r)) % 100000
            set_seed(seed)
            log(f"\n[{sub}] repeat {r + 1}/{repeats} seed={seed}")
            model = build_model(args.arm, n_chans, n_times, P.n_classes, args.dropout, ablation)
            t0 = time.time()
            out = train_one_run(
                model,
                X_tr,
                y_tr,
                X_ev,
                y_ev,
                arm=args.arm,
                device=device,
                max_epochs=max_epochs,
                batch_size=P.batch_size,
                lr=P.lr,
                early_stop_patience=args.patience if args.patience is not None else P.early_stop_patience,
                reduce_lr_patience=P.reduce_lr_patience,
                reduce_lr_factor=P.reduce_lr_factor,
                min_lr=P.min_lr,
                conv_l2_on=args.arm != "P1",
            )
            elapsed = time.time() - t0
            row = {
                "subject": sub,
                "repeat": r,
                "seed": seed,
                "acc": out["best_acc"],
                "kappa": out["best_kappa"],
                "best_epoch": out["best_epoch"],
                "epochs_ran": out["epochs_ran"],
                "elapsed_sec": elapsed,
            }
            run_rows.append(row)
            log(
                f"  → acc={row['acc']:.4f} kappa={row['kappa']:.4f} "
                f"best_ep={row['best_epoch']} time={elapsed / 60:.1f}m"
            )
            if best_row is None or row["acc"] > best_row["acc"]:
                best_row = row
                if out["state_dict"] is not None:
                    torch.save(out["state_dict"], sub_dir / "best.pt")
            # free
            del model, out
            if device.type == "cuda":
                torch.cuda.empty_cache()

        assert best_row is not None
        # paper: pick best of 10
        sub_summary = {
            "subject": sub,
            "best_acc": best_row["acc"],
            "best_kappa": best_row["kappa"],
            "best_repeat": best_row["repeat"],
            "all_runs": [{k: v for k, v in r.items() if k != "elapsed_sec"} | {"elapsed_sec": r["elapsed_sec"]} for r in run_rows],
            "zscore": zstats,
        }
        (sub_dir / "summary.json").write_text(json.dumps(sub_summary, indent=2), encoding="utf-8")
        subject_results.append(
            {
                "subject": sub,
                "acc": best_row["acc"],
                "kappa": best_row["kappa"],
                "best_repeat": best_row["repeat"],
            }
        )
        log(f"[{sub}] BEST acc={best_row['acc']:.4f} kappa={best_row['kappa']:.4f}")

    accs = [s["acc"] for s in subject_results]
    kappas = [s["kappa"] for s in subject_results]
    summary = {
        "arm": args.arm,
        "mean_acc": float(np.mean(accs)) if accs else None,
        "std_acc": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "mean_kappa": float(np.mean(kappas)) if kappas else None,
        "subjects": subject_results,
        "hparams": p_dict(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(
        f"\n=== DONE {args.arm} mean_acc={summary['mean_acc']:.4f} "
        f"±{summary['std_acc']:.4f} mean_kappa={summary['mean_kappa']:.4f} ==="
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIACNet P-track (paper config, 8ch)")
    p.add_argument("--arm", default="P0", choices=["P0", "P1", "P2", "P3a", "P3b", "P3c", "P3d"])
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--dropout", type=float, default=P.dropout)
    p.add_argument("--cache", type=str, default=None)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_arm(parse_args())
