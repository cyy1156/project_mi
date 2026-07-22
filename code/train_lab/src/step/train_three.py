from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet
from torch.utils.data import DataLoader

from dataset import ThreeHeadDataset
from metrics import format_three_metrics, three_class_metrics

# 本文件: MI/code/train_lab/src/step/train_three.py
# parents[0]=step → [1]=src → [2]=train_lab → [3]=code
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "preprocess_lab" / "out" / "bci2a"
OUT_DIR = ROOT / "train_lab" / "out"
TASK_CKPT = OUT_DIR / "best_task.pt"  # 阶段 A 权重（必须先存在）

# 与阶段 A / 文档对齐
DROP_PROB = 0.60
EPOCHS = 100
PATIENCE = 15  # val F1-macro 连续不提升则早停
LR = 1e-3
WEIGHT_DECAY = 1e-4


def _torch_load(path: Path):
    """兼容新旧 PyTorch 的 checkpoint 加载。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_backbone_from_task_ckpt(
    model_three: nn.Module,
    ckpt_path: Path,
    freeze_backbone: bool = False,
) -> None:
    """加载阶段 A 主干；跳过 final_layer.conv_classifier（2→3 类形状不同）。"""
    ckpt = _torch_load(ckpt_path)
    if "model" not in ckpt:
        raise KeyError(f"检查点缺少 'model' 字段: {ckpt_path}，现有键={list(ckpt.keys())}")
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
        logits = model(x.to(device))  # (B, 3)
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("TASK_CKPT:", TASK_CKPT)
    print("基线: braindecode.EEGNet | 阶段 B: n_outputs=3 | 加载头1主干")
    print(f"drop_prob={DROP_PROB} epochs={EPOCHS} patience={PATIENCE}")

    if not (DATA_DIR / "train_X.npy").exists():
        raise FileNotFoundError(f"找不到预处理数据: {DATA_DIR}")
    if not (DATA_DIR / "train_y_three.npy").exists():
        raise FileNotFoundError(f"找不到 train_y_three.npy: {DATA_DIR}")
    if not TASK_CKPT.exists():
        raise FileNotFoundError(
            f"找不到阶段 A 权重: {TASK_CKPT}\n请先运行 train_task.py 并确认已保存 best_task.pt"
        )

    train_loader = DataLoader(
        ThreeHeadDataset(DATA_DIR, "train"),
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ThreeHeadDataset(DATA_DIR, "val"),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    model = EEGNet(
        n_chans=8,
        n_outputs=3,
        n_times=1000,
        F1=8,
        D=2,
        F2=16,
        drop_prob=DROP_PROB,
    ).to(device)

    load_backbone_from_task_ckpt(model, TASK_CKPT, freeze_backbone=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_score = -1.0
    best_state = None
    best_ep = 0
    bad_epochs = 0

    for ep in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        y_true, y_pred = collect_preds(model, val_loader, device)
        m = three_class_metrics(y_true, y_pred)

        print(f"\nep {ep:03d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(format_three_metrics("val", m))

        if m["f1_macro"] > best_score:
            best_score = m["f1_macro"]
            best_ep = ep
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
            m_save = {k: (v.tolist() if k == "cm" else v) for k, v in m.items()}
            torch.save(
                {
                    "stage": "B_braindecode_eegnet_three3",
                    "backend": "braindecode.models.EEGNet",
                    "n_outputs": 3,
                    "init_from": str(TASK_CKPT),
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": m_save,
                },
                OUT_DIR / "best_three.pt",
            )
            print("  ↑ saved", OUT_DIR / "best_three.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  early stop at ep {ep} (patience={PATIENCE}, best_ep={best_ep})")
                break

    if best_state is None:
        raise RuntimeError("未得到任何 best 权重（验证集可能为空或训练未运行）")

    model.load_state_dict(best_state)

    y_va, p_va = collect_preds(model, val_loader, device)
    print(format_three_metrics("val(best)", three_class_metrics(y_va, p_va)))
    y_tr, p_tr = collect_preds(model, train_loader, device)
    print(format_three_metrics("train(best)", three_class_metrics(y_tr, p_tr)))
    print(f"done. best val F1-macro = {best_score:.4f} @ ep {best_ep}")


if __name__ == "__main__":
    main()
