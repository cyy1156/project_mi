"""方案 19 · V2 五折训练：Val Acc_paper 早停 · balbatch · 对齐方案18 HP。"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
STEP_DIR = HERE.parent
CODE_ROOT = HERE.parents[3]
TRAIN_LAB = CODE_ROOT / "train_lab"
PRE_ROOT = CODE_ROOT / "preprocess_lab"
AUDIT = STEP_DIR / "5070_shallow_impl_audit_accpaper"
HOP100 = STEP_DIR / "baselines_2s_hop100"

# 本包必须最前，否则会命中 baselines_2s_hop100/shared_hparams.py
_here = str(HERE)
if _here in sys.path:
    sys.path.remove(_here)
sys.path.insert(0, _here)

for p in (str(AUDIT), str(STEP_DIR), str(PRE_ROOT)):
    if p not in sys.path:
        sys.path.append(p)
# hop100 仅 append，避免覆盖本包同名模块
_hop = str(HOP100)
if _hop not in sys.path:
    sys.path.append(_hop)

from shared_hparams import SHARED, SharedTrainHP, OUT_ROOT_TAG, SCHEME19_RUNS_TAG, TRAIN_DEVICE_LABEL
from arms import ARMS, V1_REF
from model import DualBandShallowGate, count_params
from paired_data import PairedIndexDataset, assert_paired, load_openbmi_bundle
from trial_metrics import aggregate_windows_to_trials
from metrics import jsonify_metrics
from src.common.steps.split_subjects import iter_subject_kfold
from task_sampler import make_balanced_sampler
from perf_loader import apply_runtime_threads, configure_cuda_backends, make_loader


def seed_everything(seed: int, *, cudnn_benchmark: bool, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_cuda_backends(cudnn_benchmark=cudnn_benchmark, deterministic=deterministic)


def _indices_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)


def _loader_kwargs(hp: SharedTrainHP) -> dict:
    return dict(
        num_workers=hp.num_workers,
        pin_memory=hp.pin_memory,
        persistent_workers=hp.persistent_workers and hp.num_workers > 0,
        prefetch_factor=hp.prefetch_factor if hp.num_workers > 0 else None,
    )


def run_epoch(
    model: DualBandShallowGate,
    loader: DataLoader,
    device: torch.device,
    *,
    train: bool,
    optimizer=None,
    scaler=None,
    use_amp: bool = False,
    non_blocking: bool = True,
) -> float:
    model.train(train)
    total, n = 0.0, 0
    amp_on = bool(use_amp) and device.type == "cuda"
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_mu, x_beta, y in loader:
            x_mu = x_mu.to(device, non_blocking=non_blocking)
            x_beta = x_beta.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if amp_on:
                with torch.amp.autocast("cuda", enabled=True):
                    parts = model(x_mu, x_beta, return_parts=True)
                    loss = model.loss(parts, y)
            else:
                parts = model(x_mu, x_beta, return_parts=True)
                loss = model.loss(parts, y)
            if train and optimizer is not None:
                if amp_on and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total += float(loss.item()) * int(y.size(0))
            n += int(y.size(0))
    return total / max(n, 1)


@torch.no_grad()
def eval_split(
    model: DualBandShallowGate,
    *,
    y: np.ndarray,
    subjects: np.ndarray,
    trial_ids: np.ndarray,
    indices: np.ndarray,
    x_mu_path: str,
    x_beta_path: str,
    device: torch.device,
    hp: SharedTrainHP,
    n_classes: int,
) -> tuple[dict, float, float]:
    ds = PairedIndexDataset(
        y, indices, x_mu_path=x_mu_path, x_beta_path=x_beta_path
    )
    kw = dict(
        num_workers=hp.num_workers,
        pin_memory=hp.pin_memory,
        persistent_workers=hp.persistent_workers and hp.num_workers > 0,
    )
    if hp.num_workers > 0:
        kw["prefetch_factor"] = hp.prefetch_factor
    loader = make_loader(ds, batch_size=hp.batch_eval, shuffle=False, **kw)
    model.eval()
    losses = []
    preds = []
    ys = []
    amp_on = bool(hp.use_amp) and device.type == "cuda"
    for x_mu, x_beta, yb in loader:
        x_mu = x_mu.to(device, non_blocking=hp.non_blocking)
        x_beta = x_beta.to(device, non_blocking=hp.non_blocking)
        yb = yb.to(device, non_blocking=hp.non_blocking)
        if amp_on:
            with torch.amp.autocast("cuda", enabled=True):
                parts = model(x_mu, x_beta, return_parts=True)
                loss = model.loss(parts, yb)
        else:
            parts = model(x_mu, x_beta, return_parts=True)
            loss = model.loss(parts, yb)
        losses.append(float(loss.item()) * int(yb.size(0)))
        preds.append(parts["p_final"].detach().float().cpu().numpy().argmax(axis=1))
        ys.append(yb.detach().cpu().numpy())
    y_win = np.concatenate(ys, axis=0)
    pred_win = np.concatenate(preds, axis=0)
    subs = subjects[indices]
    tids = trial_ids[indices]
    trial = aggregate_windows_to_trials(
        y_win, pred_win, subs, tids, n_classes=n_classes
    )
    metrics = trial["metrics"]
    acc_paper = float(metrics["acc_paper"])
    loss_mean = float(sum(losses) / max(len(y_win), 1))
    return metrics, acc_paper, loss_mean


def train_one_head(
    *,
    arm_id: str,
    fuse: str,
    lambda_aux: float,
    y: np.ndarray,
    subjects: np.ndarray,
    trial_ids: np.ndarray,
    x_mu_path: str,
    x_beta_path: str,
    n_outputs: int,
    hp: SharedTrainHP,
    device: torch.device,
    out_dir: Path,
    max_folds: int = 0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    accs = []
    for fold_info in iter_subject_kfold(
        subjects, n_folds=hp.n_folds, seed=hp.seed, val_ratio=hp.val_ratio
    ):
        fold_i = int(fold_info["fold"])
        if max_folds > 0 and fold_i >= max_folds:
            break
        masks = fold_info["masks"]
        tr_idx = _indices_from_mask(masks["train"])
        va_idx = _indices_from_mask(masks["val"])
        te_idx = _indices_from_mask(masks["test"])
        if len(tr_idx) == 0 or len(va_idx) == 0 or len(te_idx) == 0:
            raise RuntimeError(
                f"fold{fold_i} 空集 train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}"
            )

        model = DualBandShallowGate(
            8,
            hp.n_times_expected,
            n_outputs,
            drop_prob=hp.drop_prob,
            n_filters_branch=hp.n_filters_branch,
            fuse=fuse,
            lambda_aux=lambda_aux,
        ).to(device)
        n_params = count_params(model)
        opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(hp.use_amp) and device.type == "cuda")

        ds_tr = PairedIndexDataset(
            y, tr_idx, x_mu_path=x_mu_path, x_beta_path=x_beta_path
        )
        kw = dict(
            num_workers=hp.num_workers,
            pin_memory=hp.pin_memory,
            persistent_workers=hp.persistent_workers and hp.num_workers > 0,
        )
        if hp.num_workers > 0:
            kw["prefetch_factor"] = hp.prefetch_factor
        if hp.no_balbatch:
            loader_tr = make_loader(
                ds_tr, batch_size=hp.batch_train, shuffle=True, drop_last=True, **kw
            )
        else:
            g = torch.Generator()
            g.manual_seed(hp.seed + fold_i)
            sampler = make_balanced_sampler(
                y[tr_idx], n_classes=n_outputs, generator=g
            )
            loader_tr = make_loader(
                ds_tr,
                batch_size=hp.batch_train,
                shuffle=False,
                sampler=sampler,
                drop_last=True,
                **kw,
            )

        best_val = -1.0
        best_state = None
        bad = 0
        for epoch in range(hp.max_epochs):
            tr_loss = run_epoch(
                model,
                loader_tr,
                device,
                train=True,
                optimizer=opt,
                scaler=scaler,
                use_amp=hp.use_amp,
                non_blocking=hp.non_blocking,
            )
            _, val_acc, val_loss = eval_split(
                model,
                y=y,
                subjects=subjects,
                trial_ids=trial_ids,
                indices=va_idx,
                x_mu_path=x_mu_path,
                x_beta_path=x_beta_path,
                device=device,
                hp=hp,
                n_classes=n_outputs,
            )
            print(
                f"[{arm_id} fold{fold_i} ep{epoch}] "
                f"tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc_paper={val_acc:.4f}"
            )
            if val_acc > best_val + 1e-6:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= hp.patience:
                    print(f"  early stop @ ep{epoch} best_val={best_val:.4f}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        te_metrics, te_acc, te_loss = eval_split(
            model,
            y=y,
            subjects=subjects,
            trial_ids=trial_ids,
            indices=te_idx,
            x_mu_path=x_mu_path,
            x_beta_path=x_beta_path,
            device=device,
            hp=hp,
            n_classes=n_outputs,
        )
        ckpt = out_dir / f"fold{fold_i}_best.pt"
        torch.save(
            {
                "state_dict": best_state,
                "arm": arm_id,
                "fold": fold_i,
                "val_acc_paper": best_val,
                "test_acc_paper": te_acc,
                "n_params": n_params,
                "fuse": fuse,
                "lambda_aux": lambda_aux,
            },
            ckpt,
        )
        row = {
            "fold": fold_i,
            "val_acc_paper": best_val,
            "test_acc_paper": te_acc,
            "test_loss": te_loss,
            "n_params": n_params,
            "metrics": jsonify_metrics(te_metrics),
            "ckpt": str(ckpt),
        }
        fold_rows.append(row)
        accs.append(te_acc)
        print(f"[{arm_id} fold{fold_i}] TEST Acc_paper={te_acc:.4f} params={n_params}")

    arr = np.asarray(accs, dtype=float)
    return {
        "arm": arm_id,
        "fuse": fuse,
        "lambda_aux": lambda_aux,
        "n_outputs": n_outputs,
        "folds": fold_rows,
        "test_acc_paper_mean": float(arr.mean()) if len(arr) else float("nan"),
        "test_acc_paper_std": float(arr.std()) if len(arr) else float("nan"),
        "v1_ref": V1_REF,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="方案19 DualBand Shallow Gate")
    p.add_argument("--arm", default="V2", choices=list(ARMS.keys()))
    p.add_argument(
        "--skip-task",
        action="store_true",
        default=True,
        help="默认跳过 Task（方案19 仅 Three）",
    )
    p.add_argument(
        "--run-task",
        action="store_true",
        help="可选：额外跑 Task 二分类（不进主表）",
    )
    p.add_argument(
        "--three-only",
        action="store_true",
        default=True,
        help="仅 Three（默认开启，与 --skip-task 同义）",
    )
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--batch-train", type=int, default=0)
    p.add_argument("--batch-eval", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--lambda-aux", type=float, default=-1.0)
    args = p.parse_args()

    arm = ARMS[args.arm]
    hp = SHARED
    repl: dict = {}
    if args.max_epochs > 0:
        repl["max_epochs"] = args.max_epochs
    if args.patience > 0:
        repl["patience"] = args.patience
    if args.num_workers >= 0:
        repl["num_workers"] = args.num_workers
    if args.batch_train > 0:
        repl["batch_train"] = args.batch_train
    if args.batch_eval > 0:
        repl["batch_eval"] = args.batch_eval
    if args.no_amp:
        repl["use_amp"] = False
    if args.lambda_aux >= 0:
        repl["lambda_aux"] = float(args.lambda_aux)
    else:
        repl["lambda_aux"] = arm.lambda_aux
    repl["fuse"] = arm.fuse
    if repl:
        hp = replace(hp, **repl)

    apply_runtime_threads(hp.torch_num_threads)
    seed_everything(hp.seed, cudnn_benchmark=hp.cudnn_benchmark, deterministic=hp.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mu = load_openbmi_bundle(hp.data_tag_mu)
    beta = load_openbmi_bundle(hp.data_tag_beta)
    assert_paired(mu, beta)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{stamp}_{arm.arm_id}"
    out_dir = TRAIN_LAB / "out" / OUT_ROOT_TAG / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = (
        CODE_ROOT.parent
        / "资料"
        / "模型训练"
        / "runs"
        / SCHEME19_RUNS_TAG
        / run_name
    )
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "scheme": 19,
        "arm": arm.arm_id,
        "device": str(device),
        "device_label": TRAIN_DEVICE_LABEL,
        "hp": asdict(hp),
        "v1_ref": V1_REF,
        "n_windows": int(len(mu["y_three"])),
        "data_mu": mu["root"].as_posix(),
        "data_beta": beta["root"].as_posix(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results = {}
    skip_task = (not bool(args.run_task)) and bool(args.skip_task or args.three_only)
    if not skip_task:
        print("=== Task (2-class) · 附跑不进主表 ===")
        results["task"] = train_one_head(
            arm_id=arm.arm_id,
            fuse=hp.fuse,
            lambda_aux=hp.lambda_aux,
            y=mu["y_task"],
            subjects=mu["subjects"],
            trial_ids=mu["trial_id"],
            x_mu_path=mu["x_path"],
            x_beta_path=beta["x_path"],
            n_outputs=2,
            hp=hp,
            device=device,
            out_dir=out_dir / "task",
            max_folds=args.max_folds,
        )
        (out_dir / "task" / "summary.json").write_text(
            json.dumps(results["task"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        print("=== skip Task（方案19 仅 Three）===")

    print("=== Three (3-class) · 主表 ===")
    results["three"] = train_one_head(
        arm_id=arm.arm_id,
        fuse=hp.fuse,
        lambda_aux=hp.lambda_aux,
        y=mu["y_three"],
        subjects=mu["subjects"],
        trial_ids=mu["trial_id"],
        x_mu_path=mu["x_path"],
        x_beta_path=beta["x_path"],
        n_outputs=3,
        hp=hp,
        device=device,
        out_dir=out_dir / "three",
        max_folds=args.max_folds,
    )
    (out_dir / "three" / "summary.json").write_text(
        json.dumps(results["three"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary["finished"] = datetime.now().isoformat(timespec="seconds")
    summary["results"] = results
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (runs_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    th = results["three"]
    print(
        f"\n[{arm.arm_id}] Three Acc_paper "
        f"{th['test_acc_paper_mean']:.4f}±{th['test_acc_paper_std']:.4f}  "
        f"vs V1/S0 {V1_REF['three_acc_paper']}"
    )
    print("out →", out_dir)


if __name__ == "__main__":
    main()
