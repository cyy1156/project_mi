"""被试独立五折：分类头1（静息/任务）。读全库 bci2a_*.npy。"""

from __future__ import annotations

import sys
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
OUT_DIR = ROOT / "train_lab" / "out" / "kfold_task"

sys.path.insert(0, str(PRE_ROOT))
from src.steps.split_subjects import iter_subject_kfold  # noqa: E402

N_FOLDS = 5
VAL_RATIO = 0.2
SEED = 42
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_TRAIN = 32
BATCH_EVAL = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROP_PROB = 0.50


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


def make_loader(X, y, train: bool):
    return DataLoader(
        ArrayTaskDataset(X, y),
        batch_size=BATCH_TRAIN if train else BATCH_EVAL,
        shuffle=train,
        num_workers=0,
    )


def train_one_fold(fold_info, X, y, device) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = OUT_DIR / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n======== fold {fold} ========\n"
        f"  train subjects ({len(fold_info['train_subjects'])}): {fold_info['train_subjects']}\n"
        f"  val   subjects ({len(fold_info['val_subjects'])}): {fold_info['val_subjects']}\n"
        f"  test  subjects ({len(fold_info['test_subjects'])}): {fold_info['test_subjects']}\n"
        f"  trials train/val/test = "
        f"{int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}"
    )

    train_loader = make_loader(X[masks["train"]], y[masks["train"]], train=True)
    val_loader = make_loader(X[masks["val"]], y[masks["val"]], train=False)
    test_loader = make_loader(X[masks["test"]], y[masks["test"]], train=False)

    model = EEGNet(
        n_chans=8,
        n_outputs=2,
        n_times=1000,
        F1=8,
        D=2,
        F2=16,
        drop_prob=DROP_PROB,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_score = -1.0
    best_state = None
    best_ep = 0
    bad_epochs = 0

    for ep in range(1, MAX_EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        y_true, y_pred = collect_preds(model, val_loader, device)
        m = binary_task_metrics(y_true, y_pred)

        print(f"fold{fold} ep {ep:03d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_F1={m['f1']:.4f}")

        if m["f1"] > best_score:
            best_score = m["f1"]
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            torch.save(
                {
                    "stage": "A_kfold_task2",
                    "fold": fold,
                    "n_outputs": 2,
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": m,
                    "train_subjects": fold_info["train_subjects"],
                    "val_subjects": fold_info["val_subjects"],
                    "test_subjects": fold_info["test_subjects"],
                },
                fold_dir / "best_task.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  early stop at ep {ep} (patience={PATIENCE})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    # 终评：只在此处碰 test
    y_te, p_te = collect_preds(model, test_loader, device)
    m_te = binary_task_metrics(y_te, p_te)
    print(format_task_metrics(f"fold{fold}/test", m_te))
    print(f"fold{fold} best val F1={best_score:.4f} @ ep {best_ep}")

    return {
        "fold": fold,
        "best_val_f1": best_score,
        "best_epoch": best_ep,
        "test_metrics": m_te,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR:", OUT_DIR)
    print(f"被试独立 {N_FOLDS} 折 | val_ratio={VAL_RATIO} | seed={SEED} | patience={PATIENCE}")

    for name in ("bci2a_X.npy", "bci2a_y_task.npy", "bci2a_subjects.npy"):
        if not (DATA_DIR / name).exists():
            raise FileNotFoundError(f"缺少 {DATA_DIR / name}（请先跑批处理且 save_full=true）")

    X = np.load(DATA_DIR / "bci2a_X.npy")
    y = np.load(DATA_DIR / "bci2a_y_task.npy")
    subjects = np.load(DATA_DIR / "bci2a_subjects.npy", allow_pickle=True)

    fold_results = []
    for fold_info in iter_subject_kfold(
        subjects, n_folds=N_FOLDS, val_ratio=VAL_RATIO, seed=SEED
    ):
        fold_results.append(train_one_fold(fold_info, X, y, device))

    test_f1s = [r["test_metrics"]["f1"] for r in fold_results]
    test_accs = [r["test_metrics"]["accuracy"] for r in fold_results]
    print("\n======== 5-fold summary (TEST) ========")
    for r in fold_results:
        m = r["test_metrics"]
        print(
            f"  fold {r['fold']}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
            f"Spe={m['specificity']:.4f} (best_val_F1={r['best_val_f1']:.4f})"
        )
    print(f"  Acc mean±std = {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"  F1  mean±std = {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print("done. weights under", OUT_DIR)


if __name__ == "__main__":
    main()
