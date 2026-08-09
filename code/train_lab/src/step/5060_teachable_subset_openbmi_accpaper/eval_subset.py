"""B1：在 R0/R1/R2(/R3) 子集上评估正式冻结权重（只评不训）。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
CODE_ROOT = HERE.parents[3]
TRAIN_LAB = CODE_ROOT / "train_lab"
PRE_ROOT = CODE_ROOT / "preprocess_lab"
REPO = CODE_ROOT.parent
HOP100 = STEP / "baselines_2s_hop100"
HOP100_ACC = STEP / "baselines_2s_hop100_accpaper"
OLD = STEP / "baselines_single"

for p in (STEP, PRE_ROOT, HOP100, HOP100_ACC, OLD):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
_here = str(HERE)
if _here in sys.path:
    sys.path.remove(_here)
sys.path.insert(0, _here)

try:
    from braindecode.models import EEGNet
except ImportError:
    from braindecode.models import EEGNetv4 as EEGNet

from data_paths import resolve_data
from perf_loader import apply_runtime_threads, configure_cuda_backends, make_loader
from shared_hparams import (
    FORMAL_EEGNET_RUN,
    FORMAL_SHALLOW_RUN,
    SHARED,
    SharedTrainHP,
)
from src.common.steps.split_subjects import iter_subject_kfold
from teachable_io import load_masks, resolve_teachable_paths
from trial_metrics import aggregate_windows_to_trials

EEGNET_F1, EEGNET_D, EEGNET_F2 = 8, 2, 16


class IndexDS(Dataset):
    def __init__(self, X, y, indices, *, x_path: str | None):
        self.x_path = x_path
        self._X_arr = None if x_path else X
        self.y = np.asarray(y, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)

    def _X_view(self):
        if self._X_arr is None:
            self._X_arr = np.load(self.x_path, mmap_mode="r")
        return self._X_arr

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.array(self._X_view()[idx], dtype=np.float32, copy=True)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


def build_model(name: str, n_outputs: int, n_times: int, drop_prob: float) -> nn.Module:
    if name == "shallow":
        return ShallowFBCSPNet(
            n_chans=8, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob
        )
    if name == "eegnet":
        return EEGNet(
            n_chans=8,
            n_outputs=n_outputs,
            n_times=n_times,
            F1=EEGNET_F1,
            D=EEGNET_D,
            F2=EEGNET_F2,
            drop_prob=drop_prob,
        )
    raise ValueError(name)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def resolve_run_dir(model: str, run_override: str | None) -> Path:
    root = TRAIN_LAB / "out" / "5060_baseline_openbmi_2s_hop100_accpaper"
    if run_override:
        p = Path(run_override)
        return p if p.is_absolute() else root / p
    rel = FORMAL_SHALLOW_RUN if model == "shallow" else FORMAL_EEGNET_RUN
    return root / rel


def eval_one(
    *,
    model_name: str,
    task: str,
    fold: int,
    run_dir: Path,
    X,
    y,
    subjects,
    trial_ids,
    test_mask: np.ndarray,
    subset_mask: np.ndarray | None,
    device,
    hp: SharedTrainHP,
    x_path: str,
) -> dict:
    n_classes = 2 if task == "task" else 3
    ckpt_name = "best_task.pt" if task == "task" else "best_three.pt"
    ckpt_path = run_dir / task / f"fold{fold}" / ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    model = build_model(model_name, n_classes, hp.n_times_expected, hp.drop_prob).to(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)

    mask = np.asarray(test_mask, dtype=bool)
    if subset_mask is not None:
        mask = mask & np.asarray(subset_mask, dtype=bool)
    indices = np.flatnonzero(mask).astype(np.int64)
    if len(indices) == 0:
        return {
            "n_windows": 0,
            "n_trials": 0,
            "acc_paper": float("nan"),
            "empty": True,
        }
    loader = make_loader(
        IndexDS(X, y, indices, x_path=x_path),
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=hp.num_workers,
        pin_memory=hp.pin_memory,
        persistent_workers=hp.persistent_workers and hp.num_workers > 0,
        prefetch_factor=hp.prefetch_factor,
    )
    yt, yp = predict(model, loader, device)
    agg = aggregate_windows_to_trials(
        yt, yp, subjects[indices], trial_ids[indices], n_classes=n_classes
    )
    m = agg["metrics"]
    n_subj = len(set(str(s) for s in subjects[indices].tolist()))
    return {
        "empty": False,
        "n_windows": int(m["n_windows"]),
        "n_trials": int(m["n_trials"]),
        "n_subjects": int(n_subj),
        "acc_paper": float(m["acc_paper"]),
        "acc_majority": float(m["acc_majority"]),
        "balanced_accuracy": float(m.get("balanced_accuracy", float("nan"))),
        "recall_left": float(m.get("recall_left", float("nan"))),
        "recall_right": float(m.get("recall_right", float("nan"))),
        "cm": m.get("cm"),
    }


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description="B1 子集评估正式权重")
    p.add_argument("--model", choices=("shallow", "eegnet"), default="shallow")
    p.add_argument("--run-dir", default="", help="覆盖正式 run 目录")
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--tasks", default="task,three", help="task,three")
    p.add_argument("--teachable-json", default="")
    p.add_argument("--teachable-mask", default="")
    p.add_argument("--with-r3", action="store_true", help="额外报 T1 teachable 行")
    args = p.parse_args()

    hp = SHARED
    apply_runtime_threads(hp.torch_num_threads)
    configure_cuda_backends(
        cudnn_benchmark=hp.cudnn_benchmark, deterministic=hp.deterministic
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jp, mp = resolve_teachable_paths(
        args.teachable_json or None, args.teachable_mask or None
    )
    data_dir, prefix = resolve_data(hp.data_tag)
    x_path = str(data_dir / f"{prefix}_X.npy")
    X = np.load(x_path, mmap_mode="r")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    trial_ids = np.load(data_dir / f"{prefix}_trial_id.npy")
    y_task = np.load(data_dir / f"{prefix}_y_task.npy")
    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    masks = load_masks(len(X), mp)

    run_dir = resolve_run_dir(args.model, args.run_dir or None)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = (
        TRAIN_LAB
        / "out"
        / "5060_teachable_subset_openbmi_accpaper"
        / f"eval_{args.model}"
        / stamp
    )
    out_run.mkdir(parents=True, exist_ok=True)
    md_dir = REPO / "资料" / "模型训练" / "runs" / "5060_teachable_subset"
    md_dir.mkdir(parents=True, exist_ok=True)

    rows = ["R0", "R1", "R2"] + (["R3"] if args.with_r3 else [])
    subset_of = {
        "R0": None,
        "R1": masks["obvious12"],
        "R2": masks["high_lat_eval"],
        "R3": masks["teachable"],
    }
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    fold_iter = list(
        iter_subject_kfold(
            subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
        )
    )
    if args.max_folds > 0:
        fold_iter = fold_iter[: args.max_folds]

    results: dict = {
        "model": args.model,
        "run_dir": str(run_dir),
        "teachable_json": str(jp),
        "teachable_mask": str(mp),
        "device": str(device),
        "folds": {},
        "summary": {},
    }

    for task in tasks:
        y = y_task if task == "task" else y_three
        per_row: dict[str, list[dict]] = {r: [] for r in rows}
        for fd in fold_iter:
            fold = int(fd["fold"])
            test_mask = fd["masks"]["test"]
            fold_pack = {}
            for row in rows:
                m = eval_one(
                    model_name=args.model,
                    task=task,
                    fold=fold,
                    run_dir=run_dir,
                    X=X,
                    y=y,
                    subjects=subjects,
                    trial_ids=trial_ids,
                    test_mask=test_mask,
                    subset_mask=subset_of[row],
                    device=device,
                    hp=hp,
                    x_path=x_path,
                )
                per_row[row].append(m)
                fold_pack[row] = m
                print(
                    f"[{args.model}/{task}/fold{fold}/{row}] "
                    f"acc_paper={m.get('acc_paper')} n_trials={m.get('n_trials')} "
                    f"n_subj={m.get('n_subjects')}",
                    flush=True,
                )
            results["folds"].setdefault(task, {})[f"fold{fold}"] = fold_pack

        summ = {}
        r0_mean = _mean_std([x["acc_paper"] for x in per_row["R0"]])[0]
        for row in rows:
            mean, std = _mean_std([x["acc_paper"] for x in per_row[row]])
            n_tr = _mean_std([float(x["n_trials"]) for x in per_row[row]])[0]
            summ[row] = {
                "acc_paper_mean": mean,
                "acc_paper_std": std,
                "n_trials_mean": n_tr,
                "delta_vs_R0": float(mean - r0_mean)
                if np.isfinite(mean) and np.isfinite(r0_mean)
                else float("nan"),
            }
        results["summary"][task] = summ

    (out_run / "summary.json").write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # MD
    lines = [
        f"# B1 子集评估 · {args.model}",
        "",
        f"- 时间：`{stamp}`",
        f"- 权重：`{run_dir}`",
        f"- 清单：`{jp}`",
        f"- mask：`{mp}`",
        "",
    ]
    for task in tasks:
        lines += [f"## {task}", "", "| 行 | Acc_paper | vs R0 Δ | 试次数/折(均) |", "|----|-----------|--------|---------------|"]
        for row in rows:
            s = results["summary"][task][row]
            lines.append(
                f"| {row} | {s['acc_paper_mean']:.4f}±{s['acc_paper_std']:.4f} | "
                f"{s['delta_vs_R0']:+.4f} | {s['n_trials_mean']:.1f} |"
            )
        lines.append("")
        if "three" == task and "R2" in results["summary"][task]:
            d = results["summary"][task]["R2"]["delta_vs_R0"]
            if d >= 0.03:
                advice = "建议开 B2（R2−R0 ≥ +0.03）"
            elif d >= 0.01:
                advice = "可谨慎开 B2（缩小 scope）"
            else:
                advice = "建议不开 B2；转向门控/模板"
            lines += [f"**Three R2 决策**：Δ={d:+.4f} → {advice}", ""]

    md_path = md_dir / f"{stamp}_{args.model}_B1_subset_eval.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {out_run / 'summary.json'}", flush=True)
    print(f"[done] {md_path}", flush=True)


if __name__ == "__main__":
    main()
