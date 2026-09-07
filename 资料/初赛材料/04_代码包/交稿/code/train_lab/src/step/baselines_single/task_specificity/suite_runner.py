"""七模型 Task 特异度套件（对齐 步骤_20260801_下午特异度实验）。

臂：
  A22  — 加权CE w0=2.2 + BalAcc（步骤 A 主臂）
  A20  — 加权CE w0=2.0 + BalAcc（步骤 A1，供阈值扫描对照）
  B1   — 普通CE + batch balance
  B2   — w0=2 + batch balance
  S1   — 普通CE + SMOTE

仅 Task，无 Three。用法：
  python suite_runner.py --models eegnet,deep,eegtcnet --arms A22,B1
  python suite_runner.py --models all --arms A22,A20,B1,B2,S1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
BASELINES_DIR = HERE.parent
STEP_DIR = BASELINES_DIR.parent
CODE_ROOT = HERE.parents[4]
TRAIN_LAB = CODE_ROOT / "train_lab"
REPO_ROOT = CODE_ROOT.parent
PRE_ROOT = CODE_ROOT / "preprocess_lab"

for p in (HERE, BASELINES_DIR, STEP_DIR, PRE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared_hparams import SHARED, SharedTrainHP, shared_as_dict
from md_fold_detail import task_fold_md_lines
from data_paths import resolve_data
from dataset import ArrayTaskDataset
from metrics import (
    binary_task_metrics,
    format_task_metrics,
    jsonify_metrics,
    metrics_by_dataset_prefix,
)
from src.common.steps.split_subjects import (
    iter_subject_kfold,
    iter_subject_kfold_stratified_by_dataset,
)
from task_objective import build_task_ce
from task_sampler import make_balanced_sampler
from task_smote import smote_resample_eeg

CNN_MODELS = ("eegnet", "deep", "eegtcnet", "conformer")
GRAPH_MODELS = ("dbn", "gcbnet", "dgcnn")
ALL_MODELS = CNN_MODELS + GRAPH_MODELS


@dataclass(frozen=True)
class ArmSpec:
    name: str
    suffix: str
    use_weighted_ce: bool
    w0: float
    w1: float
    sampler: str  # none | balance | smote
    description: str


ARMS: dict[str, ArmSpec] = {
    "A22": ArmSpec("A22", "wce22_balacc", True, 2.2, 1.0, "none", "加权CE w0=2.2 + BalAcc"),
    "A20": ArmSpec("A20", "wce2_balacc", True, 2.0, 1.0, "none", "加权CE w0=2.0 + BalAcc"),
    "B1": ArmSpec("B1", "balbatch_balacc", False, 1.0, 1.0, "balance", "普通CE + batch balance"),
    "B2": ArmSpec("B2", "wce2_balbatch_balacc", True, 2.0, 1.0, "balance", "w0=2 + batch balance"),
    "S1": ArmSpec("S1", "smote_balacc", False, 1.0, 1.0, "smote", "普通CE + SMOTE"),
}


class ArrayFeatDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        assert X.ndim == 3 and X.shape[1] == 8, X.shape
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def load_baseline_module(name: str):
    path = BASELINES_DIR / f"baseline_{name}.py"
    spec = importlib.util.spec_from_file_location(f"baseline_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def make_builder(name: str, mod) -> Callable:
    if name == "eegnet":
        return lambda n_last, n_out, drop: mod.build_eegnet(8, n_last, n_out, drop)
    if name in GRAPH_MODELS:
        return lambda n_last, n_out, drop: mod.build_model(8, n_last, n_out, drop)
    return lambda n_last, n_out, drop: mod.build_model(8, n_last, n_out, drop)


def load_X(name: str, data_dir: Path, prefix: str, mod) -> tuple[np.ndarray, str]:
    X_raw = np.load(data_dir / f"{prefix}_X.npy")
    if name in GRAPH_MODELS:
        X = mod.raw_to_bandpower(X_raw, sfreq=250.0)
        return X, "bandpower_cube"
    X = np.asarray(X_raw, dtype=np.float32)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]
    assert X.ndim == 3 and X.shape[1] == 8 and X.shape[2] == 500, X.shape
    return X, "raw_temporal"


def make_dataset(X: np.ndarray, y: np.ndarray, kind: str) -> Dataset:
    if kind == "bandpower_cube":
        return ArrayFeatDataset(X, y)
    return ArrayTaskDataset(X, y)


def append_md(md_path: Path, text: str, out_root: Path, log_path: Path) -> None:
    records_root = REPO_ROOT / "资料" / "模型训练"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    rel = md_path.relative_to(records_root).as_posix()
    (records_root / "五折实验记录_最新.md").write_text(
        f"# 最新实验入口\n\n本次记录：[`{rel}`](./{rel})\n\n"
        f"权重目录：`{out_root}`\n日志：`{log_path}`\n",
        encoding="utf-8",
    )


def log_line(log_path: Path, msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        ps.append(logits.argmax(dim=1).cpu().numpy())
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


def iter_folds(subjects, hp, data_tag):
    if data_tag.startswith("merged"):
        return iter_subject_kfold_stratified_by_dataset(
            subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
        )
    return iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    )


def _mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def train_one_fold(
    fold_info, X, y, subjects, device, hp, out_dir, arm: ArmSpec, builder, feat_kind: str, model_name: str
):
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n======== [{model_name}/{arm.name}] fold {fold} ========\n"
        f"  n={int(masks['train'].sum())}/{int(masks['val'].sum())}/{int(masks['test'].sum())}",
        flush=True,
    )
    g = make_generator(hp.seed + fold)
    y_tr = y[masks["train"]]
    X_tr = X[masks["train"]]
    smote_info = None
    if arm.sampler == "smote":
        X_tr, y_tr, smote_info = smote_resample_eeg(
            X_tr, y_tr, random_state=hp.seed + fold, k_neighbors=5
        )
        print(f"  SMOTE {smote_info}", flush=True)

    train_ds = make_dataset(X_tr, y_tr, feat_kind)
    if arm.sampler == "balance":
        sampler = make_balanced_sampler(y_tr, generator=g)
        train_loader = DataLoader(
            train_ds, batch_size=hp.batch_train, sampler=sampler, num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=hp.batch_train,
            shuffle=True,
            num_workers=0,
            generator=g,
        )
    val_loader = DataLoader(
        make_dataset(X[masks["val"]], y[masks["val"]], feat_kind),
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        make_dataset(X[masks["test"]], y[masks["test"]], feat_kind),
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=0,
    )

    seed_everything(hp.seed + fold)
    n_last = int(X.shape[-1])
    model = builder(n_last, 2, hp.drop_prob).to(device)
    if arm.use_weighted_ce:
        criterion = build_task_ce(
            device, mode="fixed", w0=arm.w0, w1=arm.w1, y_train=y_tr
        )
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    best_score, best_state, best_ep, best_val_loss = -1.0, None, 0, float("inf")
    best_val_f1 = -1.0
    bad, ep = 0, 0
    for ep in range(1, hp.max_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, False)
        yt, yp = collect_preds(model, val_loader, device)
        m = binary_task_metrics(yt, yp)
        print(
            f"fold{fold} ep {ep:03d} tr={tr:.4f} va={va:.4f} "
            f"BalAcc={m['balanced_accuracy']:.4f} Spec={m['specificity']:.4f} "
            f"Rec={m['recall']:.4f} F1={m['f1']:.4f}",
            flush=True,
        )
        score = float(m["balanced_accuracy"])
        if score > best_score:
            best_score, best_ep, best_val_loss = score, ep, va
            best_val_f1 = float(m["f1"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            torch.save(
                {
                    "stage": f"task2_{model_name}",
                    "fold": fold,
                    "model_name": model_name,
                    "arm": arm.name,
                    "n_outputs": 2,
                    "input": feat_kind,
                    "model": best_state,
                    "epoch": ep,
                    "val_metrics": jsonify_metrics(m),
                    "smote": smote_info,
                    "hparams": {
                        **shared_as_dict(),
                        "arm": arm.name,
                        "task_sampler": arm.sampler,
                        "task_w0": arm.w0,
                        "task_w1": arm.w1,
                        "task_early_stop": "balanced_accuracy",
                    },
                },
                fold_dir / "best_task.pt",
            )
        else:
            bad += 1
            if bad >= hp.patience:
                print(f"  early stop @ ep {ep}", flush=True)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    y_te, p_te = collect_preds(model, test_loader, device)
    by_ds = metrics_by_dataset_prefix(
        y_te, p_te, subjects[masks["test"]], binary_task_metrics
    )
    m_te = by_ds["overall"]
    print(format_task_metrics(f"fold{fold}/test", m_te), flush=True)
    return {
        "fold": fold,
        "best_val_balanced_accuracy": float(best_score),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_ep),
        "stopped_epoch": int(ep),
        "smote": smote_info,
        "test_metrics": m_te,
        "test_metrics_by_dataset": by_ds,
    }


def run_arm(model_key: str, arm: ArmSpec, data_tag: str, device, hp) -> dict:
    mod = load_baseline_module(model_key)
    builder = make_builder(model_key, mod)
    data_dir, prefix = resolve_data(data_tag)
    X, feat_kind = load_X(model_key, data_dir, prefix, mod)
    y = np.load(data_dir / f"{prefix}_y_task.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    assert len(X) == len(y) == len(subjects)

    model_name = f"{model_key}_{arm.suffix}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = TRAIN_LAB / "out" / "baseline" / model_name / data_tag / f"run_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"
    md_path = (
        REPO_ROOT
        / "资料"
        / "模型训练"
        / "runs"
        / f"{stamp}_{model_name}"
        / f"{model_name}五折实验记录.md"
    )

    append_md(
        md_path,
        "\n".join(
            [
                f"# 特异度套件（{stamp} / {model_name}）",
                "",
                f"- 开始：`{datetime.now().isoformat(timespec='seconds')}`",
                f"- device：`{device}`",
                f"- data：`{data_dir}` feat=`{feat_kind}` shape=`{X.shape}`",
                f"- backbone：`{model_key}` | 臂：`{arm.name}` — {arm.description}",
                f"- 仅 Task；早停=Balanced Acc",
                f"- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65",
                f"- 权重：`{out_root}`",
                "",
                "---",
                "",
            ]
        ),
        out_root,
        log_path,
    )
    log_line(log_path, f"start model={model_name} arm={arm.name} data={data_tag}")

    out_dir = out_root / "task"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for info in iter_folds(subjects, hp, data_tag):
        folds.append(
            train_one_fold(
                info, X, y, subjects, device, hp, out_dir, arm, builder, feat_kind, model_name
            )
        )

    def ms(xs):
        return _mean_std(xs)

    val_bal = [r["best_val_balanced_accuracy"] for r in folds]
    test_spec = [r["test_metrics"]["specificity"] for r in folds]
    test_rec = [r["test_metrics"]["recall"] for r in folds]
    test_bal = [r["test_metrics"]["balanced_accuracy"] for r in folds]
    test_f1s = [r["test_metrics"]["f1"] for r in folds]
    test_accs = [r["test_metrics"]["accuracy"] for r in folds]
    summary = {
        "task": "task_kfold",
        "model_key": model_key,
        "model_name": model_name,
        "arm": arm.name,
        "data_tag": data_tag,
        "feat_kind": feat_kind,
        "X_shape": list(X.shape),
        "val_balanced_accuracy_mean": ms(val_bal)[0],
        "val_balanced_accuracy_std": ms(val_bal)[1],
        "test_specificity_mean": ms(test_spec)[0],
        "test_specificity_std": ms(test_spec)[1],
        "test_recall_mean": ms(test_rec)[0],
        "test_recall_std": ms(test_rec)[1],
        "test_balanced_accuracy_mean": ms(test_bal)[0],
        "test_balanced_accuracy_std": ms(test_bal)[1],
        "test_f1_mean": ms(test_f1s)[0],
        "test_f1_std": ms(test_f1s)[1],
        "test_acc_mean": ms(test_accs)[0],
        "test_acc_std": ms(test_accs)[1],
        "folds": folds,
        "out_dir": str(out_dir),
        "out_root": str(out_root),
        "md": str(md_path),
        "stamp": stamp,
        "pass_gate": bool(
            ms(test_spec)[0] >= 0.40
            and ms(test_rec)[0] >= 0.75
            and ms(test_bal)[0] >= 0.65
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_root / "final_meta.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    log_line(
        log_path,
        f"TASK done Spec={summary['test_specificity_mean']:.4f} "
        f"Rec={summary['test_recall_mean']:.4f} "
        f"BalAcc={summary['test_balanced_accuracy_mean']:.4f} "
        f"pass={summary['pass_gate']}",
    )
    append_md(
        md_path,
        "\n".join(
            [
                "## 最终结论",
                "",
                f"### Task — {arm.name}",
                f"- Val BalAcc：`{summary['val_balanced_accuracy_mean']:.4f} ± {summary['val_balanced_accuracy_std']:.4f}`",
                f"- Test Spec：`{summary['test_specificity_mean']:.4f} ± {summary['test_specificity_std']:.4f}`",
                f"- Test Rec：`{summary['test_recall_mean']:.4f} ± {summary['test_recall_std']:.4f}`",
                f"- Test BalAcc：`{summary['test_balanced_accuracy_mean']:.4f} ± {summary['test_balanced_accuracy_std']:.4f}`",
                f"- Test F1：`{summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}`",
                f"- 过关：`{summary['pass_gate']}`",
                "",
                *task_fold_md_lines(folds),
                f"- 结束：`{datetime.now().isoformat(timespec='seconds')}`",
                "",
            ]
        ),
        out_root,
        log_path,
    )
    print(
        f"\n[{model_name}] Spec={summary['test_specificity_mean']:.4f} "
        f"Rec={summary['test_recall_mean']:.4f} "
        f"BalAcc={summary['test_balanced_accuracy_mean']:.4f} pass={summary['pass_gate']}",
        flush=True,
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="七模型特异度套件")
    p.add_argument("--data", default="merged_2s")
    p.add_argument(
        "--models",
        default="all",
        help="逗号分隔或 all；可选: " + ",".join(ALL_MODELS),
    )
    p.add_argument(
        "--arms",
        default="A22,A20,B1,B2,S1",
        help="逗号分隔；可选: " + ",".join(ARMS),
    )
    p.add_argument(
        "--progress",
        default=str(HERE / "suite_progress.json"),
        help="进度/结果汇总 JSON",
    )
    args = p.parse_args()

    models = list(ALL_MODELS) if args.models.strip() == "all" else [
        x.strip() for x in args.models.split(",") if x.strip()
    ]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    for m in models:
        if m not in ALL_MODELS:
            raise SystemExit(f"未知模型 {m}")
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"未知臂 {a}")

    hp = SHARED
    seed_everything(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress_path = Path(args.progress)
    results = {"started": datetime.now().isoformat(timespec="seconds"), "runs": []}
    if progress_path.is_file():
        try:
            results = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    done = {(r.get("model_key"), r.get("arm")) for r in results.get("runs", []) if r.get("ok")}

    for model_key in models:
        for arm_name in arms:
            if (model_key, arm_name) in done:
                print(f"[skip] {model_key}/{arm_name} already done", flush=True)
                continue
            arm = ARMS[arm_name]
            print(f"\n########## {model_key} / {arm_name} ##########", flush=True)
            try:
                summary = run_arm(model_key, arm, args.data, device, hp)
                results.setdefault("runs", []).append(
                    {
                        "ok": True,
                        "model_key": model_key,
                        "arm": arm_name,
                        "model_name": summary["model_name"],
                        "stamp": summary["stamp"],
                        "out_root": summary["out_root"],
                        "md": summary["md"],
                        "test_specificity_mean": summary["test_specificity_mean"],
                        "test_specificity_std": summary["test_specificity_std"],
                        "test_recall_mean": summary["test_recall_mean"],
                        "test_recall_std": summary["test_recall_std"],
                        "test_balanced_accuracy_mean": summary["test_balanced_accuracy_mean"],
                        "test_balanced_accuracy_std": summary["test_balanced_accuracy_std"],
                        "test_f1_mean": summary["test_f1_mean"],
                        "test_f1_std": summary["test_f1_std"],
                        "test_acc_mean": summary["test_acc_mean"],
                        "test_acc_std": summary["test_acc_std"],
                        "pass_gate": summary["pass_gate"],
                    }
                )
            except Exception as e:
                traceback.print_exc()
                results.setdefault("runs", []).append(
                    {
                        "ok": False,
                        "model_key": model_key,
                        "arm": arm_name,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
            results["updated"] = datetime.now().isoformat(timespec="seconds")
            progress_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    results["finished"] = datetime.now().isoformat(timespec="seconds")
    progress_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAll requested jobs finished. progress={progress_path}", flush=True)


if __name__ == "__main__":
    main()
