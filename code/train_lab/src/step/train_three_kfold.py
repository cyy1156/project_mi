"""被试独立五折：三分类（空闲/左/右）。默认独立训练、不迁权重；可选 --init-from-task。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_paths import resolve_data
from dataset import ArrayThreeDataset
from metrics import (
    format_three_metrics,
    jsonify_metrics,
    metrics_by_dataset_prefix,
    three_class_metrics,
)
from models import build_model, get_spec, list_models

ROOT = Path(__file__).resolve().parents[3]
PRE_ROOT = ROOT / "preprocess_lab"
DEFAULT_TASK_KFOLD_DIR = ROOT / "train_lab" / "out" / "kfold_task_merged_2s"
FALLBACK_TASK_CKPT = ROOT / "train_lab" / "out" / "best_task.pt"

sys.path.insert(0, str(PRE_ROOT))
from src.common.steps.split_subjects import (  # noqa: E402
    iter_subject_kfold,
    iter_subject_kfold_stratified_by_dataset,
)


@dataclass
class ThreeKFoldConfig:
    model_name: str = "eegnet"
    data_tag: str = "merged_2s"
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 100
    patience: int = 20
    batch_train: int = 32
    batch_eval: int = 64
    lr: float = 0.0015
    weight_decay: float = 0.0001
    drop_prob: float = 0.5
    f1: int = 8
    d: int = 2
    f2: int = 16
    model_kwargs: dict | None = None
    # 新策略默认 False；旧对照才 True
    init_from_task: bool = False
    freeze_backbone: bool = False
    out_dir: str = ""
    task_kfold_dir: str = ""

    def resolved_out_dir(self) -> Path:
        if self.out_dir:
            return Path(self.out_dir)
        return (
            ROOT
            / "train_lab"
            / "out"
            / "baseline"
            / self.model_name
            / self.data_tag
            / "three_default"
        )

    def resolved_task_kfold_dir(self) -> Path:
        return Path(self.task_kfold_dir) if self.task_kfold_dir else DEFAULT_TASK_KFOLD_DIR


def _iter_folds(subjects, cfg: ThreeKFoldConfig):
    if cfg.data_tag.startswith("merged"):
        return iter_subject_kfold_stratified_by_dataset(
            subjects, n_folds=cfg.n_folds, val_ratio=cfg.val_ratio, seed=cfg.seed
        )
    return iter_subject_kfold(
        subjects, n_folds=cfg.n_folds, val_ratio=cfg.val_ratio, seed=cfg.seed
    )


def _build_kwargs(cfg: ThreeKFoldConfig) -> dict:
    kw = {
        "drop_prob": cfg.drop_prob,
        "F1": cfg.f1,
        "D": cfg.d,
        "F2": cfg.f2,
        "f1": cfg.f1,
        "d": cfg.d,
        "f2": cfg.f2,
    }
    if cfg.model_kwargs:
        kw.update(cfg.model_kwargs)
    return kw


def load_backbone_from_task_ckpt(
    model_three: nn.Module,
    ckpt_path: Path,
    freeze_backbone: bool = False,
) -> None:
    """历史对照：迁二分类主干（跳过分类层）。新实验默认不调用。"""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    src = ckpt["model"]
    dst = model_three.state_dict()

    new_state = {}
    skipped = []
    for k, v in src.items():
        # 跳过分类头（EEGNet: final_layer.*；其它含 classifier 的键）
        if k.startswith("final_layer") or "classifier" in k:
            skipped.append(k)
            continue
        if k not in dst or dst[k].shape != v.shape:
            skipped.append(k)
            continue
        new_state[k] = v

    missing, unexpected = model_three.load_state_dict(new_state, strict=False)
    if freeze_backbone:
        for name, p in model_three.named_parameters():
            if name.startswith("final_layer") or "classifier" in name:
                continue
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
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
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
            if logits.ndim > 2:
                logits = logits.reshape(logits.shape[0], -1)
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


def train_one_fold(fold_info, X, y, subjects, device, cfg: ThreeKFoldConfig) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    out_dir = cfg.resolved_out_dir()
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    init_from = None
    if cfg.init_from_task:
        init_from = str(resolve_task_ckpt(fold, cfg))

    print(
        f"\n======== fold {fold} ========\n"
        f"  model={cfg.model_name}  data={cfg.data_tag}  "
        f"weight_transfer={cfg.init_from_task}\n"
        f"  train subjects ({len(fold_info['train_subjects'])}): {fold_info['train_subjects']}\n"
        f"  val   subjects ({len(fold_info['val_subjects'])}): {fold_info['val_subjects']}\n"
        f"  test  subjects ({len(fold_info['test_subjects'])}): {fold_info['test_subjects']}\n"
        f"  trials train/val/test = "
        f"{int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}\n"
        f"  init from: {init_from or '(random / independent)'}"
    )

    train_loader = make_loader(X[masks["train"]], y[masks["train"]], cfg, train=True)
    val_loader = make_loader(X[masks["val"]], y[masks["val"]], cfg, train=False)
    test_loader = make_loader(X[masks["test"]], y[masks["test"]], cfg, train=False)

    model = build_model(
        cfg.model_name,
        n_chans=8,
        n_times=int(X.shape[-1]),
        n_outputs=3,
        **_build_kwargs(cfg),
    ).to(device)

    if cfg.init_from_task:
        load_backbone_from_task_ckpt(
            model, Path(init_from), freeze_backbone=cfg.freeze_backbone
        )

    criterion = nn.CrossEntropyLoss()
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_score = -1.0
    best_state = None
    best_ep = 0
    bad_epochs = 0
    best_val_loss = float("inf")
    ep = 0

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
            m_save = jsonify_metrics(m)
            torch.save(
                {
                    "stage": "B_kfold_three3",
                    "fold": fold,
                    "model_name": cfg.model_name,
                    "n_outputs": 3,
                    "weight_transfer": bool(cfg.init_from_task),
                    "classifier": "native",
                    "init_from": init_from,
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
    subj_te = subjects[masks["test"]]
    by_ds = metrics_by_dataset_prefix(y_te, p_te, subj_te, three_class_metrics)
    m_te = by_ds["overall"]
    print(format_three_metrics(f"fold{fold}/test", {**m_te, "cm": np.asarray(m_te["cm"])}))
    for key in ("bci2a_only", "stieger_only"):
        block = by_ds.get(key)
        if block is None:
            print(f"  [{key}] (no samples)")
        else:
            print(
                f"  [{key}] n={block['n']} Acc={block['accuracy']:.4f} "
                f"F1macro={block['f1_macro']:.4f}"
            )
    print(f"fold{fold} best val F1-macro={best_score:.4f} @ ep {best_ep}")

    return {
        "fold": fold,
        "best_val_f1_macro": float(best_score),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_ep),
        "stopped_epoch": int(ep),
        "init_from": init_from,
        "weight_transfer": bool(cfg.init_from_task),
        "test_metrics": m_te,
        "test_metrics_by_dataset": by_ds,
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "train_subjects": fold_info["train_subjects"],
        "val_subjects": fold_info["val_subjects"],
        "test_subjects": fold_info["test_subjects"],
    }


def run_three_kfold(cfg: ThreeKFoldConfig | None = None, device: torch.device | None = None) -> dict:
    cfg = cfg or ThreeKFoldConfig()
    get_spec(cfg.model_name)
    data_dir, data_prefix = resolve_data(cfg.data_tag)
    out_dir = cfg.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("model:", cfg.model_name, "| family:", get_spec(cfg.model_name).family)
    print("DATA_DIR:", data_dir)
    print("OUT_DIR:", out_dir)
    print("weight_transfer (init_from_task):", cfg.init_from_task)
    if cfg.init_from_task:
        print("TASK_KFOLD_DIR:", cfg.resolved_task_kfold_dir())
    print(
        f"被试独立 {cfg.n_folds} 折 | val_ratio={cfg.val_ratio} | seed={cfg.seed} | "
        f"patience={cfg.patience} | lr={cfg.lr} | wd={cfg.weight_decay} | "
        f"drop={cfg.drop_prob} | freeze={cfg.freeze_backbone}"
    )

    for suffix in ("X", "y_three", "subjects"):
        path = data_dir / f"{data_prefix}_{suffix}.npy"
        if not path.exists():
            raise FileNotFoundError(f"缺少 {path}")

    X = np.load(data_dir / f"{data_prefix}_X.npy")
    y = np.load(data_dir / f"{data_prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{data_prefix}_subjects.npy", allow_pickle=True)
    subjects = np.asarray([str(s) for s in subjects], dtype=object)

    fold_results = []
    for fold_info in _iter_folds(subjects, cfg):
        fold_results.append(train_one_fold(fold_info, X, y, subjects, device, cfg))

    test_f1s = [r["test_metrics"]["f1_macro"] for r in fold_results]
    test_accs = [r["test_metrics"]["accuracy"] for r in fold_results]
    val_f1s = [r["best_val_f1_macro"] for r in fold_results]

    def _ms(ds_key: str, metric: str):
        vals = []
        for r in fold_results:
            block = (r.get("test_metrics_by_dataset") or {}).get(ds_key)
            if block is not None and metric in block:
                vals.append(float(block[metric]))
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    bci_f1_m, bci_f1_s = _ms("bci2a_only", "f1_macro")
    sti_f1_m, sti_f1_s = _ms("stieger_only", "f1_macro")

    summary = {
        "task": "three",
        "model_name": cfg.model_name,
        "family": get_spec(cfg.model_name).family,
        "n_outputs": 3,
        "weight_transfer": bool(cfg.init_from_task),
        "classifier": "native",
        "data": {"dir": str(data_dir), "prefix": data_prefix, "tag": cfg.data_tag},
        "hparams": asdict(cfg),
        "out_dir": str(out_dir),
        "folds": fold_results,
        "val_f1_macro_mean": float(np.mean(val_f1s)),
        "val_f1_macro_std": float(np.std(val_f1s)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "test_f1_macro_mean": float(np.mean(test_f1s)),
        "test_f1_macro_std": float(np.std(test_f1s)),
        "test_f1_macro_bci2a_only_mean": bci_f1_m,
        "test_f1_macro_bci2a_only_std": bci_f1_s,
        "test_f1_macro_stieger_only_mean": sti_f1_m,
        "test_f1_macro_stieger_only_std": sti_f1_s,
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
    print(
        f"  F1macro  mean±std = {summary['test_f1_macro_mean']:.4f} ± "
        f"{summary['test_f1_macro_std']:.4f}"
    )
    print(
        f"  Val F1macro mean±std = {summary['val_f1_macro_mean']:.4f} ± "
        f"{summary['val_f1_macro_std']:.4f}"
    )
    if bci_f1_m is not None:
        print(f"  F1macro bci2a_only  mean±std = {bci_f1_m:.4f} ± {bci_f1_s:.4f}")
    if sti_f1_m is not None:
        print(f"  F1macro stieger_only mean±std = {sti_f1_m:.4f} ± {sti_f1_s:.4f}")
    print("done. weights under", out_dir)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> ThreeKFoldConfig:
    d = ThreeKFoldConfig()
    p = argparse.ArgumentParser(description="被试独立五折：三分类空闲/左/右（默认不迁权重）")
    p.add_argument("--model", default=d.model_name, choices=list_models())
    p.add_argument("--data", default=d.data_tag)
    p.add_argument("--out-dir", default="")
    p.add_argument("--task-kfold-dir", default="", help="仅 --init-from-task 时使用")
    p.add_argument("--init-from-task", action="store_true", help="历史对照：迁移二分类主干")
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--drop-prob", type=float, default=d.drop_prob)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--batch-train", type=int, default=d.batch_train)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--f1", type=int, default=d.f1)
    p.add_argument("--d", type=int, default=d.d)
    p.add_argument("--f2", type=int, default=d.f2)
    args = p.parse_args()
    return ThreeKFoldConfig(
        model_name=args.model,
        data_tag=args.data,
        out_dir=args.out_dir,
        task_kfold_dir=args.task_kfold_dir,
        init_from_task=bool(args.init_from_task),
        lr=args.lr,
        weight_decay=args.weight_decay,
        drop_prob=args.drop_prob,
        patience=args.patience,
        max_epochs=args.max_epochs,
        seed=args.seed,
        batch_train=args.batch_train,
        freeze_backbone=args.freeze_backbone,
        f1=args.f1,
        d=args.d,
        f2=args.f2,
    )


def main() -> None:
    run_three_kfold(parse_args())


if __name__ == "__main__":
    main()
