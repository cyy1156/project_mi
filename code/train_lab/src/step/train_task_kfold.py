"""被试独立五折：分类头1（静息/任务）。读全库 bci2a_*.npy。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet
from torch.utils.data import DataLoader

from dataset import ArrayTaskDataset
from metrics import binary_task_metrics, format_task_metrics

# step → src → train_lab → code
ROOT = Path(__file__).resolve().parents[3]
PRE_ROOT = ROOT / "preprocess_lab"
DATA_DIR = PRE_ROOT / "out" / "bci2a"
DEFAULT_OUT_DIR = ROOT / "train_lab" / "out" / "kfold_task"

sys.path.insert(0, str(PRE_ROOT))
from src.steps.split_subjects import iter_subject_kfold  # noqa: E402


@dataclass
class TaskKFoldConfig:
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 100
    patience: int = 18
    batch_train: int = 32
    batch_eval: int = 64
    lr: float = 0.0007
    weight_decay: float = 0.0001
    drop_prob: float = 0.55
    f1: int = 8
    d: int = 2
    f2: int = 16
    out_dir: str = ""

    def resolved_out_dir(self) -> Path:
        return Path(self.out_dir) if self.out_dir else DEFAULT_OUT_DIR


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    return np.concatenate(ys), np.concatenate(ps)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / max(n, 1)


def make_loader(X, y, cfg: TaskKFoldConfig, train: bool):
    return DataLoader(
        ArrayTaskDataset(X, y),
        batch_size=cfg.batch_train if train else cfg.batch_eval,
        shuffle=train,
        num_workers=0,
    )


def train_one_fold(fold_info, X, y, device, cfg: TaskKFoldConfig) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    out_dir = cfg.resolved_out_dir()
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n======== fold {fold} ========\n"
        f"  train subjects ({len(fold_info['train_subjects'])}): {fold_info['train_subjects']}\n"
        f"  val   subjects ({len(fold_info['val_subjects'])}): {fold_info['val_subjects']}\n"
        f"  test  subjects ({len(fold_info['test_subjects'])}): {fold_info['test_subjects']}\n"
        f"  trials train/val/test = "
        f"{int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}"
    )

    train_loader = make_loader(X[masks["train"]], y[masks["train"]], cfg, train=True)
    val_loader = make_loader(X[masks["val"]], y[masks["val"]], cfg, train=False)
    test_loader = make_loader(X[masks["test"]], y[masks["test"]], cfg, train=False)

    model = EEGNet(
        n_chans=8,
        n_outputs=2,
        n_times=1000,
        F1=cfg.f1,
        D=cfg.d,
        F2=cfg.f2,
        drop_prob=cfg.drop_prob,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_score = -1.0
    best_state = None
    best_ep = 0
    bad_epochs = 0
    best_val_loss = float("inf")

    for ep in range(1, cfg.max_epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        y_true, y_pred = collect_preds(model, val_loader, device)
        m = binary_task_metrics(y_true, y_pred)

        print(
            f"fold{fold} ep {ep:03d}  train_loss={tr_loss:.4f}  "
            f"val_loss={va_loss:.4f}  val_F1={m['f1']:.4f}"
        )

        if m["f1"] > best_score:
            best_score = m["f1"]
            best_ep = ep
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            torch.save(
                {
                    "stage": "A_kfold_task2",
                    "fold": fold,
                    "n_outputs": 2,
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in m.items()},
                    "train_subjects": fold_info["train_subjects"],
                    "val_subjects": fold_info["val_subjects"],
                    "test_subjects": fold_info["test_subjects"],
                    "hparams": asdict(cfg),
                },
                fold_dir / "best_task.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  early stop at ep {ep} (patience={cfg.patience})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    y_te, p_te = collect_preds(model, test_loader, device)
    m_te = binary_task_metrics(y_te, p_te)
    print(format_task_metrics(f"fold{fold}/test", m_te))
    print(f"fold{fold} best val F1={best_score:.4f} @ ep {best_ep}")

    return {
        "fold": fold,
        "best_val_f1": float(best_score),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_ep),
        "stopped_epoch": int(ep),
        "test_metrics": {
            k: (v.tolist() if hasattr(v, "tolist") else float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in m_te.items()
        },
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "train_subjects": fold_info["train_subjects"],
        "val_subjects": fold_info["val_subjects"],
        "test_subjects": fold_info["test_subjects"],
    }


def run_task_kfold(cfg: TaskKFoldConfig | None = None, device: torch.device | None = None) -> dict:
    cfg = cfg or TaskKFoldConfig()
    out_dir = cfg.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR:", out_dir)
    print(
        f"被试独立 {cfg.n_folds} 折 | val_ratio={cfg.val_ratio} | seed={cfg.seed} | "
        f"patience={cfg.patience} | lr={cfg.lr} | wd={cfg.weight_decay} | drop={cfg.drop_prob}"
    )

    for name in ("bci2a_X.npy", "bci2a_y_task.npy", "bci2a_subjects.npy"):
        if not (DATA_DIR / name).exists():
            raise FileNotFoundError(f"缺少 {DATA_DIR / name}（请先跑批处理且 save_full=true）")

    X = np.load(DATA_DIR / "bci2a_X.npy")
    y = np.load(DATA_DIR / "bci2a_y_task.npy")
    subjects = np.load(DATA_DIR / "bci2a_subjects.npy", allow_pickle=True)

    fold_results = []
    for fold_info in iter_subject_kfold(
        subjects, n_folds=cfg.n_folds, val_ratio=cfg.val_ratio, seed=cfg.seed
    ):
        fold_results.append(train_one_fold(fold_info, X, y, device, cfg))

    test_f1s = [r["test_metrics"]["f1"] for r in fold_results]
    test_accs = [r["test_metrics"]["accuracy"] for r in fold_results]
    val_f1s = [r["best_val_f1"] for r in fold_results]

    summary = {
        "task": "task_kfold",
        "hparams": asdict(cfg),
        "out_dir": str(out_dir),
        "folds": fold_results,
        "val_f1_mean": float(np.mean(val_f1s)),
        "val_f1_std": float(np.std(val_f1s)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "test_f1_mean": float(np.mean(test_f1s)),
        "test_f1_std": float(np.std(test_f1s)),
        "mean_best_epoch": float(np.mean([r["best_epoch"] for r in fold_results])),
    }

    print("\n======== 5-fold summary (TEST) ========")
    for r in fold_results:
        m = r["test_metrics"]
        print(
            f"  fold {r['fold']}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
            f"Spe={m['specificity']:.4f} (best_val_F1={r['best_val_f1']:.4f})"
        )
    print(f"  Acc mean±std = {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}")
    print(f"  F1  mean±std = {summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}")
    print(f"  Val F1 mean±std = {summary['val_f1_mean']:.4f} ± {summary['val_f1_std']:.4f}")
    print("done. weights under", out_dir)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> TaskKFoldConfig:
    d = TaskKFoldConfig()
    p = argparse.ArgumentParser(description="被试独立五折：头1 静息/任务")
    p.add_argument("--out-dir", default="", help="输出目录，默认 out/kfold_task")
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--drop-prob", type=float, default=d.drop_prob)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--batch-train", type=int, default=d.batch_train)
    args = p.parse_args()
    return TaskKFoldConfig(
        out_dir=args.out_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        drop_prob=args.drop_prob,
        patience=args.patience,
        max_epochs=args.max_epochs,
        seed=args.seed,
        batch_train=args.batch_train,
    )


def main() -> None:
    run_task_kfold(parse_args())


if __name__ == "__main__":
    main()
