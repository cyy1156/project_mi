"""被试独立五折：二分类（静息/任务）。默认合并库；模型经注册表构建。"""

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
from dataset import ArrayTaskDataset
from metrics import (
    binary_task_metrics,
    format_task_metrics,
    jsonify_metrics,
    metrics_by_dataset_prefix,
)
from models import build_model, get_spec, list_models

ROOT = Path(__file__).resolve().parents[3]
PRE_ROOT = ROOT / "preprocess_lab"
DEFAULT_OUT_DIR = ROOT / "train_lab" / "out" / "kfold_task_merged_2s"

sys.path.insert(0, str(PRE_ROOT))
from src.common.steps.split_subjects import (  # noqa: E402
    iter_subject_kfold,
    iter_subject_kfold_stratified_by_dataset,
)


@dataclass
class TaskKFoldConfig:
    model_name: str = "eegnet"
    data_tag: str = "merged_2s"
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
    # 其它模型私有超参（可选），JSON 字符串或由代码直接赋 dict
    model_kwargs: dict | None = None
    out_dir: str = ""

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
            / "task_default"
        )


def _iter_folds(subjects, cfg: TaskKFoldConfig):
    if cfg.data_tag.startswith("merged"):
        return iter_subject_kfold_stratified_by_dataset(
            subjects, n_folds=cfg.n_folds, val_ratio=cfg.val_ratio, seed=cfg.seed
        )
    return iter_subject_kfold(
        subjects, n_folds=cfg.n_folds, val_ratio=cfg.val_ratio, seed=cfg.seed
    )


def _build_kwargs(cfg: TaskKFoldConfig) -> dict:
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


def make_loader(X, y, cfg: TaskKFoldConfig, train: bool):
    return DataLoader(
        ArrayTaskDataset(X, y),
        batch_size=cfg.batch_train if train else cfg.batch_eval,
        shuffle=train,
        num_workers=0,
    )


def train_one_fold(fold_info, X, y, subjects, device, cfg: TaskKFoldConfig) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    out_dir = cfg.resolved_out_dir()
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n======== fold {fold} ========\n"
        f"  model={cfg.model_name}  data={cfg.data_tag}\n"
        f"  train subjects ({len(fold_info['train_subjects'])}): {fold_info['train_subjects']}\n"
        f"  val   subjects ({len(fold_info['val_subjects'])}): {fold_info['val_subjects']}\n"
        f"  test  subjects ({len(fold_info['test_subjects'])}): {fold_info['test_subjects']}\n"
        f"  trials train/val/test = "
        f"{int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}"
    )

    train_loader = make_loader(X[masks["train"]], y[masks["train"]], cfg, train=True)
    val_loader = make_loader(X[masks["val"]], y[masks["val"]], cfg, train=False)
    test_loader = make_loader(X[masks["test"]], y[masks["test"]], cfg, train=False)

    model = build_model(
        cfg.model_name,
        n_chans=8,
        n_times=int(X.shape[-1]),
        n_outputs=2,
        **_build_kwargs(cfg),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

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
                    "model_name": cfg.model_name,
                    "n_outputs": 2,
                    "weight_transfer": False,
                    "classifier": "native",
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": jsonify_metrics(m),
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
    subj_te = subjects[masks["test"]]
    by_ds = metrics_by_dataset_prefix(y_te, p_te, subj_te, binary_task_metrics)
    m_te = by_ds["overall"]
    print(format_task_metrics(f"fold{fold}/test", m_te))
    for key in ("bci2a_only", "stieger_only"):
        block = by_ds.get(key)
        if block is None:
            print(f"  [{key}] (no samples)")
        else:
            print(
                f"  [{key}] n={block['n']} Acc={block['accuracy']:.4f} "
                f"F1={block['f1']:.4f} Spe={block['specificity']:.4f}"
            )
    print(f"fold{fold} best val F1={best_score:.4f} @ ep {best_ep}")

    return {
        "fold": fold,
        "best_val_f1": float(best_score),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_ep),
        "stopped_epoch": int(ep),
        "test_metrics": m_te,
        "test_metrics_by_dataset": by_ds,
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "train_subjects": fold_info["train_subjects"],
        "val_subjects": fold_info["val_subjects"],
        "test_subjects": fold_info["test_subjects"],
    }


def _mean_std_by_ds(
    fold_results: list[dict], ds_key: str, metric: str
) -> tuple[float | None, float | None]:
    vals = []
    for r in fold_results:
        block = (r.get("test_metrics_by_dataset") or {}).get(ds_key)
        if block is not None and metric in block:
            vals.append(float(block[metric]))
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def run_task_kfold(cfg: TaskKFoldConfig | None = None, device: torch.device | None = None) -> dict:
    cfg = cfg or TaskKFoldConfig()
    get_spec(cfg.model_name)  # 早失败
    data_dir, data_prefix = resolve_data(cfg.data_tag)
    out_dir = cfg.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("model:", cfg.model_name, "| family:", get_spec(cfg.model_name).family)
    print("DATA_DIR:", data_dir)
    print("OUT_DIR:", out_dir)
    print(
        f"被试独立 {cfg.n_folds} 折 | val_ratio={cfg.val_ratio} | seed={cfg.seed} | "
        f"patience={cfg.patience} | lr={cfg.lr} | wd={cfg.weight_decay} | drop={cfg.drop_prob}"
    )
    for suffix in ("X", "y_task", "subjects"):
        path = data_dir / f"{data_prefix}_{suffix}.npy"
        if not path.exists():
            raise FileNotFoundError(f"缺少 {path}")

    X = np.load(data_dir / f"{data_prefix}_X.npy")
    y = np.load(data_dir / f"{data_prefix}_y_task.npy")
    subjects = np.load(data_dir / f"{data_prefix}_subjects.npy", allow_pickle=True)
    subjects = np.asarray([str(s) for s in subjects], dtype=object)

    fold_results = []
    for fold_info in _iter_folds(subjects, cfg):
        fold_results.append(train_one_fold(fold_info, X, y, subjects, device, cfg))

    test_f1s = [r["test_metrics"]["f1"] for r in fold_results]
    test_accs = [r["test_metrics"]["accuracy"] for r in fold_results]
    val_f1s = [r["best_val_f1"] for r in fold_results]

    bci_f1_m, bci_f1_s = _mean_std_by_ds(fold_results, "bci2a_only", "f1")
    sti_f1_m, sti_f1_s = _mean_std_by_ds(fold_results, "stieger_only", "f1")

    summary = {
        "task": "task",
        "model_name": cfg.model_name,
        "family": get_spec(cfg.model_name).family,
        "n_outputs": 2,
        "weight_transfer": False,
        "classifier": "native",
        "data": {"dir": str(data_dir), "prefix": data_prefix, "tag": cfg.data_tag},
        "hparams": asdict(cfg),
        "out_dir": str(out_dir),
        "folds": fold_results,
        "val_f1_mean": float(np.mean(val_f1s)),
        "val_f1_std": float(np.std(val_f1s)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "test_f1_mean": float(np.mean(test_f1s)),
        "test_f1_std": float(np.std(test_f1s)),
        "test_f1_bci2a_only_mean": bci_f1_m,
        "test_f1_bci2a_only_std": bci_f1_s,
        "test_f1_stieger_only_mean": sti_f1_m,
        "test_f1_stieger_only_std": sti_f1_s,
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
    if bci_f1_m is not None:
        print(f"  F1 bci2a_only  mean±std = {bci_f1_m:.4f} ± {bci_f1_s:.4f}")
    if sti_f1_m is not None:
        print(f"  F1 stieger_only mean±std = {sti_f1_m:.4f} ± {sti_f1_s:.4f}")
    print("done. weights under", out_dir)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> TaskKFoldConfig:
    d = TaskKFoldConfig()
    p = argparse.ArgumentParser(description="被试独立五折：二分类静息/任务")
    p.add_argument("--model", default=d.model_name, choices=list_models())
    p.add_argument("--data", default=d.data_tag, help="merged_2s | bci2a_2s | stieger_2s")
    p.add_argument("--out-dir", default="")
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--drop-prob", type=float, default=d.drop_prob)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--batch-train", type=int, default=d.batch_train)
    p.add_argument("--f1", type=int, default=d.f1)
    p.add_argument("--d", type=int, default=d.d)
    p.add_argument("--f2", type=int, default=d.f2)
    args = p.parse_args()
    return TaskKFoldConfig(
        model_name=args.model,
        data_tag=args.data,
        out_dir=args.out_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        drop_prob=args.drop_prob,
        patience=args.patience,
        max_epochs=args.max_epochs,
        seed=args.seed,
        batch_train=args.batch_train,
        f1=args.f1,
        d=args.d,
        f2=args.f2,
    )


def main() -> None:
    run_task_kfold(parse_args())


if __name__ == "__main__":
    main()
