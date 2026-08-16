"""2s/hop100 Acc_paper 5-fold (standalone Shallow, no braindecode).

Aligned with train_lab/baselines_2s_hop100_accpaper/task_runner.py:
  - data: bci2a_2s_hop100 (Tw=2s @250Hz -> n_times=500, hop=100ms)
  - Train: window CE + batch balance
  - Val early-stop / Test main: Acc_paper
  - Run Task (2-class) first, then Three (3-class); write MD report
  - seed=42, lr=1e-4, wd=1e-4, drop=0.5, patience=18, max_epochs=300

Usage:
  python train_shallow_hop100_accpaper.py
  python train_shallow_hop100_accpaper.py --skip-three
  python train_shallow_hop100_accpaper.py --max-folds 1 --max-epochs 2 --patience 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from shallowfbcsp import build_model  # noqa: E402

SPLIT_DIR = REPO / "code" / "preprocess_lab"
if str(SPLIT_DIR) not in sys.path:
    sys.path.insert(0, str(SPLIT_DIR))
from src.common.steps.split_subjects import iter_subject_kfold  # noqa: E402

DEFAULT_DATA = REPO / "code" / "preprocess_lab" / "out" / "bci2a_2s_hop100"
RECORDS_ROOT = REPO / "\u8d44\u6599" / "\u6a21\u578b\u8bad\u7ec3"


@dataclass(frozen=True)
class HP:
    """Aligned with baselines_2s_hop100_accpaper/shared_hparams.SHARED."""

    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 300
    patience: int = 18
    batch_train: int = 32
    batch_eval: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    n_times_expected: int = 500
    n_chans: int = 8
    protocol: str = "2s-hop100ms-balbatch-accpaper-T-only"
    optimizer: str = "adam"  # "adam" | "adamw"


class WinDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_balanced_sampler(y: np.ndarray, n_classes: int, g: torch.Generator):
    y = np.asarray(y).astype(int).reshape(-1)
    counts = np.maximum(np.bincount(y, minlength=n_classes).astype(np.float64), 1.0)
    w = torch.as_tensor(1.0 / counts[y], dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(y), replacement=True, generator=g)


def _majority_label(preds: np.ndarray) -> int:
    top = Counter(int(p) for p in preds.tolist()).most_common()
    if not top:
        return -1
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return -1
    return int(top[0][0])


def _mean_std(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    if len(a) == 0:
        return 0.0, 0.0
    return float(a.mean()), float(a.std())


def _f(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def clf_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """Window/trial clf metrics aligned with train_lab metrics.py keys."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(accuracy_score(y_true, y_pred))

    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return {
            "confusion_matrix": cm.astype(int).tolist(),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "accuracy": acc,
            "recall": recall,
            "sensitivity": recall,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "balanced_accuracy": float(0.5 * (recall + specificity)),
        }

    recall_macro = float(
        recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )
    precision_macro = float(
        precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )
    f1_macro = float(
        f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )
    recall_per = recall_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    specs = []
    for c in labels:
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return {
        "confusion_matrix": cm.astype(int).tolist(),
        "cm": cm.astype(int).tolist(),
        "accuracy": acc,
        "recall_macro": recall_macro,
        "precision_macro": precision_macro,
        "f1_macro": f1_macro,
        "f1": f1_macro,
        "sensitivity": recall_macro,
        "specificity": float(np.mean(specs)),
        "balanced_accuracy": recall_macro,
        "recall_idle": float(recall_per[0]),
        "recall_left": float(recall_per[1]),
        "recall_right": float(recall_per[2]),
    }


def format_clf(tag: str, m: dict) -> str:
    if "tp" in m:
        cm_s = f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}"
    else:
        cm_s = str(m.get("confusion_matrix", m.get("cm")))
    bal = m.get("balanced_accuracy", float("nan"))
    f1v = m.get("f1", m.get("f1_macro", float("nan")))
    return (
        f"{tag} CM=[{cm_s}] "
        f"Sens={m.get('sensitivity', m.get('recall', float('nan'))):.4f} "
        f"Spec={m.get('specificity', float('nan')):.4f} "
        f"F1={f1v:.4f} BalAcc={bal:.4f}"
    )


def aggregate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    trial_ids: np.ndarray,
    n_classes: int,
) -> tuple[dict, dict]:
    """Return (trial_metrics, window_metrics) like task_runner._eval_split."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    subjects = np.asarray([str(s) for s in subjects])
    trial_ids = np.asarray(trial_ids).astype(np.int64).reshape(-1)

    order, seen = [], set()
    for k in zip(subjects.tolist(), trial_ids.tolist()):
        if k not in seen:
            seen.add(k)
            order.append(k)

    paper_ok, trial_y, trial_pred = [], [], []
    for subj, tid in order:
        m = (subjects == subj) & (trial_ids == tid)
        yt, yp = y_true[m], y_pred[m]
        y = int(np.unique(yt)[0])
        rate = float(np.mean(yp == y))
        paper_ok.append(rate > 0.5)
        maj = _majority_label(yp)
        if maj < 0:
            maj = (y + 1) % n_classes
        trial_y.append(y)
        trial_pred.append(maj)

    yt_arr = np.asarray(trial_y, dtype=np.int64)
    yp_arr = np.asarray(trial_pred, dtype=np.int64)
    win = clf_metrics(y_true, y_pred, n_classes)
    trial_clf = clf_metrics(yt_arr, yp_arr, n_classes)
    trial = {
        "n_trials": int(len(yt_arr)),
        "n_windows": int(len(y_true)),
        "acc_paper": float(np.mean(paper_ok)) if paper_ok else 0.0,
        "acc_majority": float(np.mean(yp_arr == yt_arr)) if len(yt_arr) else 0.0,
        **trial_clf,
    }
    return trial, win


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total, n = 0.0, 0
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
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


def eval_split(model, X, y, subjects, trial_ids, mask, device, hp: HP, n_classes: int):
    loader = DataLoader(
        WinDataset(X[mask], y[mask]),
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=0,
    )
    va_loss = run_epoch(model, loader, nn.CrossEntropyLoss(), None, device, False)
    yt, yp = collect_preds(model, loader, device)
    trial, win = aggregate_metrics(
        yt, yp, subjects[mask], trial_ids[mask], n_classes=n_classes
    )
    return trial, win, float(va_loss)


def log_line(log_path: Path, msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_md(md_path: Path, text: str) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def task_fold_md_lines(folds: list[dict]) -> list[str]:
    lines = ["### Task fold details", ""]
    for r in folds:
        m = r.get("test_trial_metrics") or {}
        mw = r.get("test_window_metrics") or {}
        lines.extend(
            [
                f"#### Fold {r.get('fold', '?')}",
                "",
                f"- stopped_epoch: `{r.get('stopped_epoch')}` | best_epoch: `{r.get('best_epoch')}`",
                f"- Val Acc_paper: `{_f(r.get('best_val_acc_paper'))}`",
                f"- Val BalAcc_maj: `{_f(r.get('best_val_balacc_maj'))}`",
                "",
                "**Test trial**",
                f"- Acc_paper: `{_f(m.get('acc_paper'))}`",
                f"- BalAcc_maj: `{_f(m.get('balanced_accuracy'))}`",
                f"- Acc_majority: `{_f(m.get('acc_majority'))}`",
                f"- Sens/Spec/F1: `{_f(m.get('sensitivity', m.get('recall')))}` / "
                f"`{_f(m.get('specificity'))}` / `{_f(m.get('f1'))}`",
                f"- CM: `{m.get('confusion_matrix')}`",
                f"- n_trials: `{m.get('n_trials')}`",
                "",
                "**Test window**",
                f"- BalAcc: `{_f(mw.get('balanced_accuracy'))}` | F1: `{_f(mw.get('f1'))}` | "
                f"Sens: `{_f(mw.get('sensitivity', mw.get('recall')))}` | "
                f"Spec: `{_f(mw.get('specificity'))}`",
                f"- CM: `{mw.get('confusion_matrix')}`",
                "",
            ]
        )
    return lines


def three_fold_md_lines(folds: list[dict]) -> list[str]:
    lines = ["### Three fold details", ""]
    for r in folds:
        m = r.get("test_trial_metrics") or {}
        mw = r.get("test_window_metrics") or {}
        lines.extend(
            [
                f"#### Fold {r.get('fold', '?')}",
                "",
                f"- stopped_epoch: `{r.get('stopped_epoch')}` | best_epoch: `{r.get('best_epoch')}`",
                f"- Val Acc_paper: `{_f(r.get('best_val_acc_paper'))}`",
                f"- Val BalAcc_maj: `{_f(r.get('best_val_balacc_maj'))}`",
                "",
                "**Test trial**",
                f"- Acc_paper: `{_f(m.get('acc_paper'))}`",
                f"- BalAcc_maj: `{_f(m.get('balanced_accuracy'))}`",
                f"- F1-macro: `{_f(m.get('f1_macro', m.get('f1')))}`",
                f"- Rec idle/left/right: `{_f(m.get('recall_idle'))}` / "
                f"`{_f(m.get('recall_left'))}` / `{_f(m.get('recall_right'))}`",
                f"- Spec(macro): `{_f(m.get('specificity'))}`",
                f"- CM: `{m.get('confusion_matrix', m.get('cm'))}`",
                f"- n_trials: `{m.get('n_trials')}`",
                "",
                "**Test window**",
                f"- BalAcc: `{_f(mw.get('balanced_accuracy'))}` | "
                f"F1m: `{_f(mw.get('f1_macro', mw.get('f1')))}` | "
                f"Spec: `{_f(mw.get('specificity'))}`",
                f"- CM: `{mw.get('confusion_matrix', mw.get('cm'))}`",
                "",
            ]
        )
    return lines


def make_optimizer(model: nn.Module, hp: HP) -> torch.optim.Optimizer:
    name = str(hp.optimizer).lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    raise ValueError(f"unsupported optimizer: {hp.optimizer!r} (use adam|adamw)")


def train_one_fold(
    fold_info,
    X,
    y,
    subjects,
    trial_ids,
    device,
    hp: HP,
    out_dir: Path,
    *,
    n_outputs: int,
    ckpt_name: str,
    stage_tag: str,
) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    n_classes = n_outputs

    print(
        f"\n======== [{stage_tag}] fold {fold} ========\n"
        f"  train={fold_info['train_subjects']}\n"
        f"  val  ={fold_info['val_subjects']}\n"
        f"  test ={fold_info['test_subjects']}\n"
        f"  n_win={int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}"
    )

    g = make_generator(hp.seed + fold)
    y_tr = y[masks["train"]]
    train_loader = DataLoader(
        WinDataset(X[masks["train"]], y_tr),
        batch_size=hp.batch_train,
        sampler=make_balanced_sampler(y_tr, n_classes, g),
        num_workers=0,
    )

    seed_everything(hp.seed + fold)
    model = build_model(hp.n_chans, int(X.shape[-1]), n_outputs, hp.drop_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, hp)

    best_score, best_state, best_ep = -1.0, None, 0
    best_val_loss = float("inf")
    best_val_bal_maj = -1.0
    best_val_trial_metrics: dict | None = None
    bad, ep = 0, 0
    epoch_logs = []

    for ep in range(1, hp.max_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_trial, val_win, va_loss = eval_split(
            model, X, y, subjects, trial_ids, masks["val"], device, hp, n_classes
        )
        score = float(val_trial["acc_paper"])
        bal_maj = float(val_trial["balanced_accuracy"])
        print(
            f"fold{fold} ep{ep:03d} tr={tr:.4f} va={va_loss:.4f} "
            f"val_AccPaper={score:.4f} val_BalAccMaj={bal_maj:.4f} "
            f"win_BalAcc={float(val_win['balanced_accuracy']):.4f}"
        )
        print("  " + format_clf("val_win", val_win))
        print("  " + format_clf("val_trial", val_trial))
        epoch_logs.append(
            {
                "epoch": ep,
                "train_loss": float(tr),
                "val_loss": float(va_loss),
                "val_trial": val_trial,
                "val_window": val_win,
            }
        )
        if score > best_score:
            best_score = score
            best_ep = ep
            best_val_loss = va_loss
            best_val_bal_maj = bal_maj
            best_val_trial_metrics = val_trial
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            torch.save(
                {
                    "stage": stage_tag,
                    "fold": fold,
                    "n_outputs": n_outputs,
                    "protocol": hp.protocol,
                    "early_stop": "acc_paper",
                    "balbatch": True,
                    "model": best_state,
                    "epoch": ep,
                    "val_trial_metrics": val_trial,
                    "val_window_metrics": val_win,
                    "hparams": asdict(hp),
                },
                fold_dir / ckpt_name,
            )
        else:
            bad += 1
            if bad >= hp.patience:
                print(f"  early stop @ ep {ep}")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    te_trial, te_win, _ = eval_split(
        model, X, y, subjects, trial_ids, masks["test"], device, hp, n_classes
    )
    print(
        f"[fold{fold}/test] Acc_paper={te_trial['acc_paper']:.4f}  "
        f"BalAcc_maj={te_trial['balanced_accuracy']:.4f}  "
        f"win_BalAcc={te_win['balanced_accuracy']:.4f}"
    )
    print("  " + format_clf("test_win", te_win))
    print("  " + format_clf("test_trial", te_trial))

    (fold_dir / "epoch_metrics.json").write_text(
        json.dumps(epoch_logs, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return {
        "fold": fold,
        "best_val_acc_paper": float(best_score),
        "best_val_balacc_maj": float(best_val_bal_maj),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_ep),
        "stopped_epoch": int(ep),
        "best_val_trial_metrics": best_val_trial_metrics,
        "test_trial_metrics": te_trial,
        "test_window_metrics": te_win,
        "train_subjects": fold_info["train_subjects"],
        "val_subjects": fold_info["val_subjects"],
        "test_subjects": fold_info["test_subjects"],
    }


def run_kfold(
    X,
    y,
    subjects,
    trial_ids,
    device,
    hp: HP,
    out_dir: Path,
    *,
    n_outputs: int,
    ckpt_name: str,
    stage_tag: str,
    task_key: str,
    max_folds: int = 0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for i, info in enumerate(
        iter_subject_kfold(
            subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
        )
    ):
        if max_folds > 0 and i >= max_folds:
            break
        folds.append(
            train_one_fold(
                info,
                X,
                y,
                subjects,
                trial_ids,
                device,
                hp,
                out_dir,
                n_outputs=n_outputs,
                ckpt_name=ckpt_name,
                stage_tag=stage_tag,
            )
        )

    val_ap = [r["best_val_acc_paper"] for r in folds]
    test_ap = [float(r["test_trial_metrics"]["acc_paper"]) for r in folds]
    test_bm = [float(r["test_trial_metrics"]["balanced_accuracy"]) for r in folds]
    test_wbal = [float(r["test_window_metrics"]["balanced_accuracy"]) for r in folds]
    test_wspec = [float(r["test_window_metrics"]["specificity"]) for r in folds]
    test_tspec = [float(r["test_trial_metrics"]["specificity"]) for r in folds]

    summary = {
        "task": task_key,
        "n_outputs": n_outputs,
        "protocol": hp.protocol,
        "balbatch": True,
        "early_stop": "acc_paper",
        "hparams": asdict(hp),
        "val_acc_paper_mean": _mean_std(val_ap)[0],
        "val_acc_paper_std": _mean_std(val_ap)[1],
        "test_acc_paper_mean": _mean_std(test_ap)[0],
        "test_acc_paper_std": _mean_std(test_ap)[1],
        "test_balacc_maj_mean": _mean_std(test_bm)[0],
        "test_balacc_maj_std": _mean_std(test_bm)[1],
        "test_window_balacc_mean": _mean_std(test_wbal)[0],
        "test_window_balacc_std": _mean_std(test_wbal)[1],
        "test_window_specificity_mean": _mean_std(test_wspec)[0],
        "test_window_specificity_std": _mean_std(test_wspec)[1],
        "test_trial_specificity_mean": _mean_std(test_tspec)[0],
        "test_trial_specificity_std": _mean_std(test_tspec)[1],
        "folds": folds,
        "out_dir": str(out_dir),
    }
    if n_outputs == 2:
        w_f1 = [float(r["test_window_metrics"]["f1"]) for r in folds]
        t_f1 = [float(r["test_trial_metrics"]["f1"]) for r in folds]
        w_sens = [
            float(r["test_window_metrics"].get("sensitivity", r["test_window_metrics"]["recall"]))
            for r in folds
        ]
        summary["test_window_f1_mean"], summary["test_window_f1_std"] = _mean_std(w_f1)
        summary["test_trial_f1_mean"], summary["test_trial_f1_std"] = _mean_std(t_f1)
        summary["test_window_sensitivity_mean"], summary["test_window_sensitivity_std"] = _mean_std(
            w_sens
        )
    else:
        t_f1m = [float(r["test_trial_metrics"].get("f1_macro", r["test_trial_metrics"]["f1"])) for r in folds]
        w_f1m = [float(r["test_window_metrics"].get("f1_macro", r["test_window_metrics"]["f1"])) for r in folds]
        summary["test_f1_macro_maj_mean"], summary["test_f1_macro_maj_std"] = _mean_std(t_f1m)
        summary["test_window_f1_macro_mean"], summary["test_window_f1_macro_std"] = _mean_std(w_f1m)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(
        f"\n[{task_key}] Val Acc_paper "
    )
    return summary


def squeeze_raw_2s(X: np.ndarray) -> np.ndarray:
    """(N,1,8,T) -> (N,8,T) for Shallow/EEGNet-style time models."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0]
    return X


def load_all(data_dir: Path, prefix: str = "bci2a"):
    X = squeeze_raw_2s(np.load(data_dir / f"{prefix}_X.npy"))
    y_task = np.load(data_dir / f"{prefix}_y_task.npy")
    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    trial_ids = np.load(data_dir / f"{prefix}_trial_id.npy")
    return X, y_task, y_three, subjects, trial_ids


def main() -> None:
    p = argparse.ArgumentParser(
        description="standalone Shallow hop100 Acc_paper (task then three; write MD)"
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--prefix", default="bci2a")
    p.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adam",
        help="optimizer: adam (default) or adamw",
    )
    p.add_argument("--skip-three", action="store_true", help="only run Task 2-class")
    p.add_argument("--max-folds", type=int, default=0, help=">0: smoke first N folds")
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    hp = replace(HP(), optimizer=args.optimizer)
    if args.max_epochs > 0 or args.patience > 0:
        hp = replace(
            hp,
            max_epochs=args.max_epochs if args.max_epochs > 0 else hp.max_epochs,
            patience=args.patience if args.patience > 0 else hp.patience,
        )

    data_dir = args.data_dir
    need = [
        f"{args.prefix}_X.npy",
        f"{args.prefix}_y_task.npy",
        f"{args.prefix}_y_three.npy",
        f"{args.prefix}_subjects.npy",
        f"{args.prefix}_trial_id.npy",
    ]
    missing = [n for n in need if not (data_dir / n).is_file()]
    if missing:
        raise SystemExit(
            f"missing files under {data_dir}: {missing}\n"
            f"run preprocess (bci2a_2s_hop100) or pass --data-dir"
        )

    X, y_task, y_three, subjects, trial_ids = load_all(data_dir, args.prefix)
    assert X.ndim == 3 and X.shape[1] == hp.n_chans, X.shape
    assert int(X.shape[-1]) == hp.n_times_expected, X.shape
    assert len(X) == len(y_task) == len(y_three) == len(subjects) == len(trial_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"standalone_shallow_2s_hop100_balbatch_accpaper_{hp.optimizer}"
    out_root = args.out or (HERE / "out" / out_name / f"run_{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"

    md_local = out_root / f"{stamp}_{out_name}\u4e94\u6298\u5b9e\u9a8c\u8bb0\u5f55.md"
    md_records = (
        RECORDS_ROOT / "runs" / f"{stamp}_{out_name}" / f"{out_name}\u4e94\u6298\u5b9e\u9a8c\u8bb0\u5f55.md"
    )

    seed_everything(hp.seed)
    header = "\n".join(
        [
            f"# 5-fold experiment record ({stamp} / {out_name})",
            "",
            f"- start: `{datetime.now().isoformat(timespec='seconds')}`",
            f"- device: `{device}`",
            f"- data: `{data_dir}` (BCI2a T / hop100 only)",
            f"- protocol: `{hp.protocol}` | early_stop=**Acc_paper** | **balbatch**",
            f"- optimizer: `{hp.optimizer}` | lr=`{hp.lr}` | weight_decay=`{hp.weight_decay}`",
            f"- model: `standalone ShallowFBCSPNet` | no braindecode",
            f"- Train: window CE + batch balance; Val/Test: trial Acc_paper",
            f"- pipeline: Task(2-class) then Three(3-class)"
            + (" (skip-three)" if args.skip_three else ""),
            f"- weights: `{out_root}`",
            f"- shared hp: `{asdict(hp)}`",
            "",
            "---",
            "",
        ]
    )
    append_md(md_local, header)
    try:
        append_md(md_records, header)
    except OSError as e:
        log_line(log_path, f"warn: cannot write records md ({e}); local md only")

    log_line(log_path, f"start model={out_name} optimizer={hp.optimizer} device={device} data={data_dir}")

    sum_task = run_kfold(
        X,
        y_task,
        subjects,
        trial_ids,
        device,
        hp,
        out_root / "task",
        n_outputs=2,
        ckpt_name="best_task.pt",
        stage_tag=f"task2_{out_name}",
        task_key="task_kfold_accpaper",
        max_folds=args.max_folds,
    )
    log_line(
        log_path,
        f"TASK done val_AccPaper={sum_task['val_acc_paper_mean']:.4f} "
        f"test_AccPaper={sum_task['test_acc_paper_mean']:.4f}",
    )

    sum_three = None
    if not args.skip_three:
        sum_three = run_kfold(
            X,
            y_three,
            subjects,
            trial_ids,
            device,
            hp,
            out_root / "three",
            n_outputs=3,
            ckpt_name="best_three.pt",
            stage_tag=f"three3_{out_name}",
            task_key="three_kfold_accpaper",
            max_folds=args.max_folds,
        )
        log_line(
            log_path,
            f"THREE done val_AccPaper={sum_three['val_acc_paper_mean']:.4f} "
            f"test_AccPaper={sum_three['test_acc_paper_mean']:.4f}",
        )

    md_tail = [
        "",
        "### Task",
        f"`{_f(sum_task.get('test_window_specificity_mean'))}` / `{_f(sum_task.get('test_window_f1_mean'))}`",
        "",
        *task_fold_md_lines(sum_task["folds"]),
    ]
    if sum_three is not None:
        md_tail.extend(
            [
                "### Three",
                f"`{_f(sum_three.get('test_window_specificity_mean'))}`",
                "",
                *three_fold_md_lines(sum_three["folds"]),
            ]
        )
    else:
        md_tail.extend(["### Three", "- (skipped this run)", ""])

    md_tail.extend(
        [
            "```json",
            json.dumps(asdict(hp), indent=2),
            "```",
            "",
            "",
        ]
    )
    tail = "\n".join(md_tail)
    append_md(md_local, tail)
    try:
        append_md(md_records, tail)
        latest = RECORDS_ROOT / "\u4e94\u6298\u5b9e\u9a8c\u8bb0\u5f55_\u6700\u65b0.md"
        try:
            rel = md_records.relative_to(RECORDS_ROOT).as_posix()
            latest.write_text(
                "# Latest experiment entry\n\n"
                f"Record: [`{rel}`](./{rel})\n\n"
                f"Weights: `{out_root}`\n"
                f"Log: `{log_path}`\n",
                encoding="utf-8",
            )
        except ValueError:
            pass
    except OSError:
        pass

    meta = {
        "model_name": out_name,
        "optimizer": hp.optimizer,
        "stamp": stamp,
        "protocol": hp.protocol,
        "early_stop": "acc_paper",
        "balbatch": True,
        "task": sum_task,
        "three": sum_three,
        "md_local": str(md_local),
        "md_records": str(md_records),
    }
    (out_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log_line(log_path, f"done md={md_local}")
    print(f"\nMD: {md_local}\nout: {out_root}")


if __name__ == "__main__":
    main()
