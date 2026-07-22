"""被试独立五折：分类头2（空闲/左/右）。读全库 bci2a_*.npy；每折加载同折头1主干。"""

from __future__ import annotations

import sys
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
OUT_DIR = ROOT / "train_lab" / "out" / "kfold_three"
TASK_KFOLD_DIR = ROOT / "train_lab" / "out" / "kfold_task"
FALLBACK_TASK_CKPT = ROOT / "train_lab" / "out" / "best_task.pt"

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
DROP_PROB = 0.60


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


def make_loader(X, y, train: bool):
    return DataLoader(
        ArrayThreeDataset(X, y),
        batch_size=BATCH_TRAIN if train else BATCH_EVAL,
        shuffle=train,
        num_workers=0,
    )


def resolve_task_ckpt(fold: int) -> Path:
    """优先用同折头1权重，避免用全库 best_task 泄漏测试被试。"""
    fold_ckpt = TASK_KFOLD_DIR / f"fold{fold}" / "best_task.pt"
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


def train_one_fold(fold_info, X, y, device) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = OUT_DIR / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    task_ckpt = resolve_task_ckpt(fold)

    print(
        f"\n======== fold {fold} ========\n"
        f"  train subjects ({len(fold_info['train_subjects'])}): {fold_info['train_subjects']}\n"
        f"  val   subjects ({len(fold_info['val_subjects'])}): {fold_info['val_subjects']}\n"
        f"  test  subjects ({len(fold_info['test_subjects'])}): {fold_info['test_subjects']}\n"
        f"  trials train/val/test = "
        f"{int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}\n"
        f"  init from: {task_ckpt}"
    )

    train_loader = make_loader(X[masks["train"]], y[masks["train"]], train=True)
    val_loader = make_loader(X[masks["val"]], y[masks["val"]], train=False)
    test_loader = make_loader(X[masks["test"]], y[masks["test"]], train=False)

    model = EEGNet(
        n_chans=8,
        n_outputs=3,
        n_times=1000,
        F1=8,
        D=2,
        F2=16,
        drop_prob=DROP_PROB,
    ).to(device)
    load_backbone_from_task_ckpt(model, task_ckpt, freeze_backbone=False)

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
        m = three_class_metrics(y_true, y_pred)

        print(
            f"fold{fold} ep {ep:03d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
            f"val_F1macro={m['f1_macro']:.4f}"
        )

        if m["f1_macro"] > best_score:
            best_score = m["f1_macro"]
            best_ep = ep
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
                },
                fold_dir / "best_three.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  early stop at ep {ep} (patience={PATIENCE})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    y_te, p_te = collect_preds(model, test_loader, device)
    m_te = three_class_metrics(y_te, p_te)
    print(format_three_metrics(f"fold{fold}/test", m_te))
    print(f"fold{fold} best val F1-macro={best_score:.4f} @ ep {best_ep}")

    return {
        "fold": fold,
        "best_val_f1_macro": best_score,
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

    for name in ("bci2a_X.npy", "bci2a_y_three.npy", "bci2a_subjects.npy"):
        if not (DATA_DIR / name).exists():
            raise FileNotFoundError(f"缺少 {DATA_DIR / name}")

    X = np.load(DATA_DIR / "bci2a_X.npy")
    y = np.load(DATA_DIR / "bci2a_y_three.npy")
    subjects = np.load(DATA_DIR / "bci2a_subjects.npy", allow_pickle=True)

    fold_results = []
    for fold_info in iter_subject_kfold(
        subjects, n_folds=N_FOLDS, val_ratio=VAL_RATIO, seed=SEED
    ):
        fold_results.append(train_one_fold(fold_info, X, y, device))

    test_f1s = [r["test_metrics"]["f1_macro"] for r in fold_results]
    test_accs = [r["test_metrics"]["accuracy"] for r in fold_results]
    print("\n======== 5-fold summary (TEST) ========")
    for r in fold_results:
        m = r["test_metrics"]
        print(
            f"  fold {r['fold']}: Acc={m['accuracy']:.4f} F1macro={m['f1_macro']:.4f} "
            f"R_idle/left/right="
            f"{m['recall_idle']:.3f}/{m['recall_left']:.3f}/{m['recall_right']:.3f}"
        )
    print(f"  Acc      mean±std = {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"  F1macro  mean±std = {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print("done. weights under", OUT_DIR)


if __name__ == "__main__":
    main()
