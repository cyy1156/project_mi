"""重放已训 ckpt，导出 val/test 窗级 softmax（方案 24 · V/E）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
STEP_DIR = HERE.parent
CODE_ROOT = HERE.parents[3]
PRE_ROOT = CODE_ROOT / "preprocess_lab"
HOP100 = STEP_DIR / "baselines_2s_hop100"
OLD_BASELINES = STEP_DIR / "baselines_single"

for p in (STEP_DIR, PRE_ROOT, HOP100, OLD_BASELINES):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data_paths import resolve_data
from perf_loader import apply_runtime_threads, configure_cuda_backends, make_loader
from prob_dump import dump_rows_to_csv
from shared_hparams import SHARED, SharedTrainHP
from t0_sec import compute_window_t0_sec
from task_runner import (
    IndexArrayDataset,
    _indices_from_mask,
    _loader_kwargs,
    seed_everything,
)
from src.common.steps.split_subjects import iter_subject_kfold

BuildFn = Callable[..., nn.Module]


@torch.no_grad()
def _collect_prob_rows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    fold: int,
    split: str,
    subjects: np.ndarray,
    trial_ids: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    t0_all: np.ndarray,
    non_blocking: bool,
) -> list[dict]:
    model.eval()
    rows: list[dict] = []
    off = 0
    for x, yb in loader:
        x = x.to(device, non_blocking=non_blocking)
        logits = model(x)
        if logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        bs = probs.shape[0]
        for i in range(bs):
            gi = int(indices[off + i])
            p = probs[i]
            rows.append(
                {
                    "subject": str(subjects[gi]),
                    "fold": fold,
                    "split": split,
                    "trial_id": int(trial_ids[gi]),
                    "t0_sec": float(t0_all[gi]),
                    "pred": int(preds[i]),
                    "y": int(yb[i].item()),
                    "p_max": float(p.max()),
                    "p0": float(p[0]),
                    "p1": float(p[1]),
                    "p2": float(p[2]) if len(p) > 2 else 0.0,
                }
            )
        off += bs
    return rows


def _resolve_stage_dir(run_dir: Path, stage: str) -> Path:
    run_dir = run_dir.resolve()
    if run_dir.name == stage and (run_dir / "summary.json").is_file():
        return run_dir
    stage_dir = run_dir / stage
    if (stage_dir / "summary.json").is_file():
        return stage_dir
    raise FileNotFoundError(
        f"no summary.json under {run_dir} or {run_dir / stage}"
    )


def dump_run(
    *,
    run_dir: Path,
    stage: str,
    build_model: BuildFn,
    hp: SharedTrainHP,
    data_tag: str = "openbmi_3s_hop100",
    x_path: str | None = None,
) -> None:
    run_dir = run_dir.resolve()
    stage_dir = _resolve_stage_dir(run_dir, stage)
    summary_path = stage_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_name = summary.get("model_name", run_dir.parent.name)

    data_dir, prefix = resolve_data(data_tag)
    x_npy = data_dir / f"{prefix}_X.npy"
    X = np.load(x_npy, mmap_mode="r")
    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    trial_ids = np.load(data_dir / f"{prefix}_trial_id.npy")
    if x_path is None:
        x_path = str(x_npy)
    t0_all = compute_window_t0_sec(trial_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_times = int(X.shape[-1])
    n_outputs = 3 if stage == "three" else 2
    ckpt_name = "best_three.pt" if stage == "three" else "best_task.pt"

    for info in iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    ):
        fold = info["fold"]
        fold_dir = stage_dir / f"fold{fold}"
        ckpt_path = fold_dir / ckpt_name
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(8, n_times, n_outputs, hp.drop_prob).to(device)
        model.load_state_dict(ckpt["model"])
        rows: list[dict] = []
        for split, key in (("val", "val"), ("test", "test")):
            mask = info["masks"][key]
            indices = _indices_from_mask(mask)
            loader = make_loader(
                IndexArrayDataset(None, y_three, indices, input_kind="time", x_path=x_path),
                batch_size=hp.batch_eval,
                shuffle=False,
                **_loader_kwargs(hp),
            )
            rows.extend(
                _collect_prob_rows(
                    model,
                    loader,
                    device,
                    fold=fold,
                    split=split,
                    subjects=subjects,
                    trial_ids=trial_ids,
                    y=y_three,
                    indices=indices,
                    t0_all=t0_all,
                    non_blocking=hp.non_blocking,
                )
            )
        out_csv = fold_dir / f"prob_dump_{stage}.csv"
        dump_rows_to_csv(out_csv, rows)
        print(f"fold{fold} → {out_csv} ({len(rows)} rows)")


def main(build_model: BuildFn | None = None) -> None:
    p = argparse.ArgumentParser(description="Scheme24 dump-probs replay")
    p.add_argument("--run-dir", type=Path, required=True, help=".../run_*/three")
    p.add_argument("--stage", default="three", choices=("three", "task"))
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-eval", type=int, default=512)
    args = p.parse_args()

    if build_model is None:
        from braindecode.models import ShallowFBCSPNet

        def build_model(n_chans, n_times, n_outputs, drop_prob):
            return ShallowFBCSPNet(
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_times=n_times,
                drop_prob=drop_prob,
            )

    hp = SHARED
    from dataclasses import replace

    hp = replace(hp, num_workers=args.num_workers, batch_eval=args.batch_eval)
    apply_runtime_threads(hp.torch_num_threads)
    seed_everything(hp.seed, cudnn_benchmark=hp.cudnn_benchmark, deterministic=hp.deterministic)
    dump_run(run_dir=args.run_dir, stage=args.stage, build_model=build_model, hp=hp)


if __name__ == "__main__":
    main()
