"""被试独立五折：分类头2（空闲/左/右）。读全库 bci2a_*.npy；每折加载同折头1主干。"""

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

from dataset import ArrayThreeDataset
from metrics import format_three_metrics, three_class_metrics

ROOT = Path(__file__).resolve().parents[3]
PRE_ROOT = ROOT / "preprocess_lab"
DATA_DIR = PRE_ROOT / "out" / "bci2a"
DEFAULT_OUT_DIR = ROOT / "train_lab" / "out" / "kfold_three"
DEFAULT_TASK_KFOLD_DIR = ROOT / "train_lab" / "out" / "kfold_task"
FALLBACK_TASK_CKPT = ROOT / "train_lab" / "out" / "best_task.pt"

sys.path.insert(0, str(PRE_ROOT))
from src.steps.split_subjects import iter_subject_kfold  # noqa: E402


@dataclass
class ThreeKFoldConfig:
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 100
    patience: int = 18
    batch_train: int = 32
    batch_eval: int = 64
    lr: float = 0.0007
    weight_decay: float = 0.0001
    drop_prob: float = 0.65
    f1: int = 8
    d: int = 2
    f2: int = 16
    freeze_backbone: bool = False
    out_dir: str = ""
    task_kfold_dir: str = ""

    def resolved_out_dir(self) -> Path:
        return Path(self.out_dir) if self.out_dir else DEFAULT_OUT_DIR

    def resolved_task_kfold_dir(self) -> Path:
        return Path(self.task_kfold_dir) if self.task_kfold_dir else DEFAULT_TASK_KFOLD_DIR


def load_backbone_from_task_ckpt(
    model_three: nn.Module,
    ckpt_path: Path,
    freeze_backbone: bool = False,
) -> None:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    src = ckpt["model"]
    dst = model_three.state_dict()

    new_state = {}
    skipped = []
    for k, v in src.items():
        if k.startswith("final_layer.conv_classifier"):
            skipped.append(k)
            continue
        if k not in dst or dst[k].shape != v.shape:
            skipped.append(k)
            continue
        new_state[k] = v

    missing, unexpected = model_three.load_state_dict(new_state, strict=False)
    if freeze_backbone:
        for name, p in model_three.named_parameters():
            if not name.startswith("final_layer"):
                p.requires_grad = False

    print(f"[init] loaded {len(new_state)} tensors from {ckpt_path}")
    print(f"[init] skipped: {skipped}")
    print(f"[init] missing_keys: {missing}")
    print(f"[init] unexpected_keys: {unexpected}")


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


def make_loader(X, y, cfg: ThreeKFoldConfig, train: bool):
    return DataLoader(
        ArrayThreeDataset(X, y),
        batch_size=cfg.batch_train if train else cfg.batch_eval,
        shuffle=train,
        num_workers=0,
    )


def resolve_task_ckpt(fold: int, cfg: ThreeKFoldConfig) -> Path:
    """优先用同折头1权重，避免用全库 best_task 泄漏测试被试。"""
    fold_ckpt = cfg.resolved_task_kfold_dir() / f"fold{fold}" / "best_task.pt"
    if fold_ckpt.exists():
        return fold_ckpt
    if FALLBACK_TASK_CKPT.exists():
        print(
            f"[warn] 未找到 {fold_ckpt}，回退到 {FALLBACK_TASK_CKPT}\n"
            f"      正式五折请先跑 train_task_kfold（同 seed/划分）。"
        )
        return FALLBACK_TASK_CKPT
    raise FileNotFoundError(
        f"找不到头1权重：既无 {fold_ckpt}，也无 {FALLBACK_TASK_CKPT}"
    )


def train_one_fold(fold_info, X, y, device, cfg: ThreeKFoldConfig) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    out_dir = cfg.resolved_out_dir()
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    task_ckpt = resolve_task_ckpt(fold, cfg)

    print(
        f"\n======== fold {fold} ========\n"
        f"  train subjects ({len(fold_info['train_subjects'])}): {fold_info['train_subjects']}\n"
        f"  val   subjects ({len(fold_info['val_subjects'])}): {fold_info['val_subjects']}\n"
        f"  test  subjects ({len(fold_info['test_subjects'])}): {fold_info['test_subjects']}\n"
        f"  trials train/val/test = "
        f"{int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}\n"
        f"  init from: {task_ckpt}"
    )

    train_loader = make_loader(X[masks["train"]], y[masks["train"]], cfg, train=True)
    val_loader = make_loader(X[masks["val"]], y[masks["val"]], cfg, train=False)
    test_loader = make_loader(X[masks["test"]], y[masks["test"]], cfg, train=False)

    model = EEGNet(
        n_chans=8,
        n_outputs=3,
        n_times=1000,
        F1=cfg.f1,
        D=cfg.d,
        F2=cfg.f2,
        drop_prob=cfg.drop_prob,
    ).to(device)
    load_backbone_from_task_ckpt(model, task_ckpt, freeze_backbone=cfg.freeze_backbone)

    criterion = nn.CrossEntropyLoss()
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_score = -1.0
    best_state = None
    best_ep = 0
    bad_epochs = 0
    best_val_loss = float("inf")

    for ep in range(1, cfg.max_epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        y_true, y_pred = collect_preds(model, val_loader, device)
        m = three_class_metrics(y_true, y_pred)

        print(
            f"fold{fold} ep {ep:03d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
            f"val_F1macro={m['f1_macro']:.4f}"
        )

        if m["f1_macro"] > best_score:
            best_score = m["f1_macro"]
            best_ep = ep
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            m_save = {k: (v.tolist() if k == "cm" else v) for k, v in m.items()}
            torch.save(
                {
                    "stage": "B_kfold_three3",
                    "fold": fold,
                    "n_outputs": 3,
                    "init_from": str(task_ckpt),
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": m_save,
                    "train_subjects": fold_info["train_subjects"],
                    "val_subjects": fold_info["val_subjects"],
                    "test_subjects": fold_info["test_subjects"],
                    "hparams": asdict(cfg),
                },
                fold_dir / "best_three.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  early stop at ep {ep} (patience={cfg.patience})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    y_te, p_te = collect_preds(model, test_loader, device)
    m_te = three_class_metrics(y_te, p_te)
    print(format_three_metrics(f"fold{fold}/test", m_te))
    print(f"fold{fold} best val F1-macro={best_score:.4f} @ ep {best_ep}")

    return {
        "fold": fold,
        "best_val_f1_macro": float(best_score),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_ep),
        "stopped_epoch": int(ep),
        "init_from": str(task_ckpt),
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


def run_three_kfold(cfg: ThreeKFoldConfig | None = None, device: torch.device | None = None) -> dict:
    cfg = cfg or ThreeKFoldConfig()
    out_dir = cfg.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR:", out_dir)
    print("TASK_KFOLD_DIR:", cfg.resolved_task_kfold_dir())
    print(
        f"被试独立 {cfg.n_folds} 折 | val_ratio={cfg.val_ratio} | seed={cfg.seed} | "
        f"patience={cfg.patience} | lr={cfg.lr} | wd={cfg.weight_decay} | "
        f"drop={cfg.drop_prob} | freeze={cfg.freeze_backbone}"
    )

    for name in ("bci2a_X.npy", "bci2a_y_three.npy", "bci2a_subjects.npy"):
        if not (DATA_DIR / name).exists():
            raise FileNotFoundError(f"缺少 {DATA_DIR / name}")

    X = np.load(DATA_DIR / "bci2a_X.npy")
    y = np.load(DATA_DIR / "bci2a_y_three.npy")
    subjects = np.load(DATA_DIR / "bci2a_subjects.npy", allow_pickle=True)

    fold_results = []
    for fold_info in iter_subject_kfold(
        subjects, n_folds=cfg.n_folds, val_ratio=cfg.val_ratio, seed=cfg.seed
    ):
        fold_results.append(train_one_fold(fold_info, X, y, device, cfg))

    test_f1s = [r["test_metrics"]["f1_macro"] for r in fold_results]
    test_accs = [r["test_metrics"]["accuracy"] for r in fold_results]
    val_f1s = [r["best_val_f1_macro"] for r in fold_results]

    summary = {
        "task": "three_kfold",
        "hparams": asdict(cfg),
        "out_dir": str(out_dir),
        "folds": fold_results,
        "val_f1_macro_mean": float(np.mean(val_f1s)),
        "val_f1_macro_std": float(np.std(val_f1s)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "test_f1_macro_mean": float(np.mean(test_f1s)),
        "test_f1_macro_std": float(np.std(test_f1s)),
        "mean_best_epoch": float(np.mean([r["best_epoch"] for r in fold_results])),
    }

    print("\n======== 5-fold summary (TEST) ========")
    for r in fold_results:
        m = r["test_metrics"]
        print(
            f"  fold {r['fold']}: Acc={m['accuracy']:.4f} F1macro={m['f1_macro']:.4f} "
            f"R_idle/left/right="
            f"{m['recall_idle']:.3f}/{m['recall_left']:.3f}/{m['recall_right']:.3f}"
        )
    print(f"  Acc      mean±std = {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}")
    print(f"  F1macro  mean±std = {summary['test_f1_macro_mean']:.4f} ± {summary['test_f1_macro_std']:.4f}")
    print(
        f"  Val F1macro mean±std = {summary['val_f1_macro_mean']:.4f} ± "
        f"{summary['val_f1_macro_std']:.4f}"
    )
    print("done. weights under", out_dir)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> ThreeKFoldConfig:
    d = ThreeKFoldConfig()
    p = argparse.ArgumentParser(description="被试独立五折：头2 空闲/左/右")
    p.add_argument("--out-dir", default="")
    p.add_argument("--task-kfold-dir", default="")
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--drop-prob", type=float, default=d.drop_prob)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--batch-train", type=int, default=d.batch_train)
    p.add_argument("--freeze-backbone", action="store_true")
    args = p.parse_args()
    return ThreeKFoldConfig(
        out_dir=args.out_dir,
        task_kfold_dir=args.task_kfold_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        drop_prob=args.drop_prob,
        patience=args.patience,
        max_epochs=args.max_epochs,
        seed=args.seed,
        batch_train=args.batch_train,
        freeze_backbone=args.freeze_backbone,
    )


def main() -> None:
    run_three_kfold(parse_args())


if __name__ == "__main__":
    main()
