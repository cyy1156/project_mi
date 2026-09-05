"""复用 hop100 五折权重，按试次多数票复评 Task / Three。

用法（在 baselines_2s_hop100_trialmaj/）：
  python reeval_kfold.py --model eegnet
  python reeval_kfold.py --model eegnet --smoke-fold 0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
CODE_ROOT = HERE.parents[3]
TRAIN_LAB = CODE_ROOT / "train_lab"
PRE_ROOT = CODE_ROOT / "preprocess_lab"
HOP100 = STEP / "baselines_2s_hop100"

for p in (HERE, HOP100, STEP, PRE_ROOT, STEP / "baselines_single"):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

from data_paths import resolve_data
from dataset import ArrayTaskDataset
from src.common.steps.split_subjects import iter_subject_kfold

from model_registry import get_build_model, get_input_kind, get_prepare_X
from official_runs import hop100_run_dir
from trial_aggregate import aggregate_windows_to_trials

DROP_PROB = 0.50
BATCH_EVAL = 64
N_FOLDS = 5
VAL_RATIO = 0.2
SEED = 42
DATA_TAG = "bci2a_2s_hop100"
N_TIMES = 500
PROTOCOL = "2s-hop100ms-trial_maj>50%-no_retrain"


class ArrayFeatDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


@torch.no_grad()
def predict_all(model, X: np.ndarray, y: np.ndarray, input_kind: str, device) -> np.ndarray:
    if input_kind == "time":
        ds = ArrayTaskDataset(X, y)
    else:
        ds = ArrayFeatDataset(X, y)
    loader = DataLoader(ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)
    model.eval()
    ps = []
    for xb, _ in loader:
        logits = model(xb.to(device))
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        ps.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(ps)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def reeval_one_stage(
    *,
    stage: str,
    n_outputs: int,
    run_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    trial_ids: np.ndarray,
    build_model,
    input_kind: str,
    device: torch.device,
    smoke_fold: int | None,
) -> dict:
    ckpt_name = "best_task.pt" if stage == "task" else "best_three.pt"
    folds_out = []
    for info in iter_subject_kfold(subjects, n_folds=N_FOLDS, val_ratio=VAL_RATIO, seed=SEED):
        fold = int(info["fold"])
        if smoke_fold is not None and fold != smoke_fold:
            continue
        masks = info["masks"]
        ckpt_path = run_dir / stage / f"fold{fold}" / ckpt_name
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(8, int(X.shape[-1]), n_outputs, DROP_PROB).to(device)
        model.load_state_dict(ckpt["model"])

        split_metrics = {}
        for split in ("test", "val"):
            m = masks[split]
            y_win = y[m]
            p_win = predict_all(model, X[m], y_win, input_kind, device)
            agg = aggregate_windows_to_trials(
                y_win,
                p_win,
                subjects[m],
                trial_ids[m],
                n_classes=n_outputs,
            )
            split_metrics[split] = agg["metrics"]
            print(
                f"[{stage}] fold{fold}/{split} n_trials={agg['metrics']['n_trials']} "
                f"Acc_paper={agg['metrics']['acc_paper']:.4f} "
                f"BalAcc={agg['metrics']['balanced_accuracy']:.4f}",
                flush=True,
            )

        folds_out.append(
            {
                "fold": fold,
                "test_subjects": list(map(str, info["test_subjects"])),
                "val_subjects": list(map(str, info["val_subjects"])),
                "test_metrics": split_metrics["test"],
                "val_metrics": split_metrics["val"],
                "ckpt": str(ckpt_path),
            }
        )

    def collect(key: str, split: str = "test") -> list[float]:
        return [float(f[f"{split}_metrics"][key]) for f in folds_out]

    summary = {
        "stage": stage,
        "n_outputs": n_outputs,
        "protocol": PROTOCOL,
        "no_retrain": True,
        "n_folds_evaluated": len(folds_out),
        "test_acc_paper_mean": _mean_std(collect("acc_paper"))[0],
        "test_acc_paper_std": _mean_std(collect("acc_paper"))[1],
        "test_acc_majority_mean": _mean_std(collect("acc_majority"))[0],
        "test_acc_majority_std": _mean_std(collect("acc_majority"))[1],
        "test_balanced_accuracy_mean": _mean_std(collect("balanced_accuracy"))[0],
        "test_balanced_accuracy_std": _mean_std(collect("balanced_accuracy"))[1],
        "val_acc_paper_mean": _mean_std(collect("acc_paper", "val"))[0],
        "val_acc_paper_std": _mean_std(collect("acc_paper", "val"))[1],
        "val_balanced_accuracy_mean": _mean_std(collect("balanced_accuracy", "val"))[0],
        "val_balanced_accuracy_std": _mean_std(collect("balanced_accuracy", "val"))[1],
        "folds": folds_out,
    }
    if n_outputs == 2:
        summary.update(
            {
                "test_f1_mean": _mean_std(collect("f1"))[0],
                "test_f1_std": _mean_std(collect("f1"))[1],
                "test_specificity_mean": _mean_std(collect("specificity"))[0],
                "test_specificity_std": _mean_std(collect("specificity"))[1],
                "test_recall_mean": _mean_std(collect("recall"))[0],
                "test_recall_std": _mean_std(collect("recall"))[1],
            }
        )
    else:
        summary.update(
            {
                "test_f1_macro_mean": _mean_std(collect("f1_macro"))[0],
                "test_f1_macro_std": _mean_std(collect("f1_macro"))[1],
                "test_recall_idle_mean": _mean_std(collect("recall_idle"))[0],
                "test_recall_idle_std": _mean_std(collect("recall_idle"))[1],
                "test_recall_left_mean": _mean_std(collect("recall_left"))[0],
                "test_recall_left_std": _mean_std(collect("recall_left"))[1],
                "test_recall_right_mean": _mean_std(collect("recall_right"))[0],
                "test_recall_right_std": _mean_std(collect("recall_right"))[1],
            }
        )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="hop100 试次级多数票复评")
    p.add_argument("--model", required=True, help="模型名，如 eegnet")
    p.add_argument("--run-dir", default="", help="覆盖官方 hop100 run 目录")
    p.add_argument("--smoke-fold", type=int, default=None, help="仅评该折（调试）")
    p.add_argument("--skip-three", action="store_true")
    args = p.parse_args()

    model_name = args.model.strip()
    run_dir = Path(args.run_dir) if args.run_dir else hop100_run_dir(model_name)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir 不存在: {run_dir}")

    data_dir, prefix = resolve_data(DATA_TAG)
    X_raw = np.load(data_dir / f"{prefix}_X.npy")
    y_task = np.load(data_dir / f"{prefix}_y_task.npy")
    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    trial_ids = np.load(data_dir / f"{prefix}_trial_id.npy")
    assert len(X_raw) == len(y_task) == len(trial_ids)
    assert int(X_raw.shape[-1]) == N_TIMES

    prepare = get_prepare_X(model_name)
    X = prepare(X_raw) if prepare is not None else X_raw
    input_kind = get_input_kind(model_name)
    build_model = get_build_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (
        TRAIN_LAB
        / "out"
        / "baseline_2s_hop100_trialmaj"
        / model_name
        / DATA_TAG
        / f"reeval_{stamp}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "model_name": model_name,
        "hop100_run_dir": str(run_dir),
        "data_tag": DATA_TAG,
        "protocol": PROTOCOL,
        "input_kind": input_kind,
        "device": str(device),
        "smoke_fold": args.smoke_fold,
    }
    (out_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"reeval model={model_name} run={run_dir} out={out_root}", flush=True)
    sum_task = reeval_one_stage(
        stage="task",
        n_outputs=2,
        run_dir=run_dir,
        X=X,
        y=y_task,
        subjects=subjects,
        trial_ids=trial_ids,
        build_model=build_model,
        input_kind=input_kind,
        device=device,
        smoke_fold=args.smoke_fold,
    )
    (out_root / "task_summary.json").write_text(
        json.dumps(sum_task, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    sum_three = None
    if not args.skip_three:
        sum_three = reeval_one_stage(
            stage="three",
            n_outputs=3,
            run_dir=run_dir,
            X=X,
            y=y_three,
            subjects=subjects,
            trial_ids=trial_ids,
            build_model=build_model,
            input_kind=input_kind,
            device=device,
            smoke_fold=args.smoke_fold,
        )
        (out_root / "three_summary.json").write_text(
            json.dumps(sum_three, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    print(
        f"DONE TASK Acc_paper={sum_task['test_acc_paper_mean']:.4f} "
        f"BalAcc={sum_task['test_balanced_accuracy_mean']:.4f}",
        flush=True,
    )
    if sum_three is not None:
        print(
            f"DONE THREE Acc_paper={sum_three['test_acc_paper_mean']:.4f} "
            f"BalAcc={sum_three['test_balanced_accuracy_mean']:.4f}",
            flush=True,
        )
    print(f"out={out_root}", flush=True)


if __name__ == "__main__":
    main()
