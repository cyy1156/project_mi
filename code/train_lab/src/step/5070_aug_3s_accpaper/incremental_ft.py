"""A0 / G1 / G2 / G3：Stieger 增量 FT 爬坡曲线 + 零样本点。

仿真部署：前半 cue 按时间序每累计 k 个 trial 做一次全量 FT（在已有权重上继续），
在 eval 后半评 Acc_paper。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import _bootstrap  # noqa: F401

from s25_config import (  # noqa: E402
    ANCHOR_09_A0_EVAL_HALF_THREE,
    INCREMENTAL_K_LIST,
    N_FOLDS,
    PROTOCOL,
    RESULTS_ROOT,
)
from data import SubjectStream, iter_subject_streams  # noqa: E402
from data_split import (  # noqa: E402
    assert_no_leakage,
    build_cue_split,
    split_train_val_cues,
    window_mask_for_cues,
)
from dataset import ArrayTaskDataset, ArrayThreeDataset  # noqa: E402
from domain_aug import AugConfig, aug_config_from_spec, apply_domain_aug_np  # noqa: E402
from infer import load_fold_model  # noqa: E402
from stieger_eval import build_shallow, eval_mask_pack  # noqa: E402
from task_sampler import make_balanced_sampler  # noqa: E402
from util_metrics import jsonable, mean_std  # noqa: E402
from s25_weights import resolve_weight_run  # noqa: E402


@dataclass(frozen=True)
class FtHP:
    n_folds: int = 5
    val_ratio: float = 0.2
    seed: int = 42
    max_epochs: int = 300
    patience: int = 20
    batch_train: int = 32
    batch_eval: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    drop_prob: float = 0.50
    replay_ratio: float = 0.0  # G2: 0.10–0.20
    aug: AugConfig | None = None  # G3


SHARED = FtHP()


class ReplayDataset(Dataset):
    """Stieger FT 窗 + 可选 OpenBMI replay 窗（G2）。"""

    def __init__(
        self,
        X_st: np.ndarray,
        y_st: np.ndarray,
        *,
        X_rep: np.ndarray | None = None,
        y_rep: np.ndarray | None = None,
        replay_ratio: float = 0.0,
        n_classes: int = 3,
        aug: AugConfig | None = None,
        seed: int = 42,
    ):
        self.X_st = np.asarray(X_st, dtype=np.float32)
        self.y_st = np.asarray(y_st, dtype=np.int64)
        self.X_rep = None if X_rep is None else np.asarray(X_rep, dtype=np.float32)
        self.y_rep = None if y_rep is None else np.asarray(y_rep, dtype=np.int64)
        self.replay_ratio = float(replay_ratio)
        self.n_classes = n_classes
        self.aug = aug
        self.seed = seed
        self._indices = list(range(len(self.y_st)))
        if self.X_rep is not None and len(self.y_rep) and self.replay_ratio > 0:
            n_rep = max(1, int(round(len(self.y_st) * self.replay_ratio)))
            rng = np.random.default_rng(seed)
            rep_idx = rng.integers(0, len(self.y_rep), size=n_rep)
            self._indices.extend([("r", int(i)) for i in rep_idx])

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int):
        ref = self._indices[i]
        if isinstance(ref, tuple) and ref[0] == "r":
            x = self.X_rep[ref[1]].copy()
            y = int(self.y_rep[ref[1]])
        else:
            x = self.X_st[ref].copy()
            y = int(self.y_st[ref])
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[0]
        if self.aug is not None and self.aug.enabled:
            x = apply_domain_aug_np(x, self.aug, seed=self.seed, index=i, y=y)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def _make_ds(X, y, n_outputs: int):
    return ArrayTaskDataset(X, y) if n_outputs == 2 else ArrayThreeDataset(X, y)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def finetune_on_mask(
    model: nn.Module,
    stream: SubjectStream,
    *,
    mask_tr: np.ndarray,
    mask_va: np.ndarray,
    head: str,
    device: torch.device,
    hp: FtHP,
    fold: int,
    replay_pack: tuple[np.ndarray, np.ndarray] | None = None,
    split=None,
) -> nn.Module:
    n_outputs = 2 if head == "task" else 3
    y = stream.y_task if head == "task" else stream.y_three
    X_tr, y_tr = stream.X[mask_tr], y[mask_tr]
    if len(y_tr) == 0:
        raise RuntimeError("FT 训练窗为空")

    g = torch.Generator()
    g.manual_seed(hp.seed + fold)
    if replay_pack is not None and hp.replay_ratio > 0:
        X_rep, y_rep = replay_pack
        train_ds: Dataset = ReplayDataset(
            X_tr,
            y_tr,
            X_rep=X_rep,
            y_rep=y_rep,
            replay_ratio=hp.replay_ratio,
            n_classes=n_outputs,
            aug=hp.aug,
            seed=hp.seed + fold,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=hp.batch_train,
            shuffle=True,
            num_workers=0,
        )
    elif hp.aug is not None and hp.aug.enabled:
        train_ds = ReplayDataset(
            X_tr, y_tr, aug=hp.aug, seed=hp.seed + fold, n_classes=n_outputs
        )
        train_loader = DataLoader(
            train_ds, batch_size=hp.batch_train, shuffle=True, num_workers=0
        )
    else:
        train_loader = DataLoader(
            _make_ds(X_tr, y_tr, n_outputs),
            batch_size=hp.batch_train,
            sampler=make_balanced_sampler(
                y_tr, n_classes=n_outputs, generator=g
            ),
            num_workers=0,
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
    )
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    mask_va_use = mask_va if int(mask_va.sum()) > 0 else mask_tr
    if split is None:
        split = build_cue_split(stream, val_ratio=hp.val_ratio, seed=hp.seed)
    best_score, best_state, bad = -1.0, None, 0
    for ep in range(1, hp.max_epochs + 1):
        run_epoch(model, train_loader, criterion, optimizer, device, True)
        model.eval()
        agg = eval_mask_pack(
            stream,
            mask_va_use,
            model,
            head=head,
            device=device,
            batch_size=hp.batch_eval,
            use_noz_pipeline=False,
        )
        score = float(agg["segment_metrics"]["acc_paper"])
        if score > best_score:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= hp.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def cues_for_k(train_cues: list[int], k: int) -> list[int]:
    ordered = sorted(int(c) for c in train_cues)
    if k < 0:
        return ordered
    return ordered[: min(k, len(ordered))]


def load_openbmi_replay(
    init_run: Path, head: str, fold: int, *, max_windows: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """从 OpenBMI 训练折 mmap 抽 replay 窗（按类平衡）。"""
    from data_paths import resolve_data  # noqa: WPS433

    data_dir, prefix = resolve_data("openbmi_3s_hop100")
    X = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
    y_key = "y_task" if head == "task" else "y_three"
    y = np.load(data_dir / f"{prefix}_{y_key}.npy")
    n = min(len(y), max_windows)
    idx = np.linspace(0, len(y) - 1, n, dtype=int)
    return np.array(X[idx], dtype=np.float32), np.asarray(y[idx], dtype=np.int64)


def run_incremental_curve(
    stream: SubjectStream,
    *,
    init_run: Path,
    head: str,
    device: torch.device,
    hp: FtHP,
    k_list: tuple[int, ...] = INCREMENTAL_K_LIST,
    replay_pack: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """累积式增量 FT：k 增大时在上一 checkpoint 权重上全量 FT 前 k 个 cue。"""
    split = build_cue_split(stream, val_ratio=hp.val_ratio, seed=hp.seed)
    assert_no_leakage(split)
    mask_eval = window_mask_for_cues(stream.cue_ids, split.eval_cues)
    curve: list[dict] = []
    fold_state: dict[int, nn.Module | None] = {f: None for f in range(hp.n_folds)}

    ordered_k = sorted(k_list, key=lambda x: (x != 0, x if x >= 0 else 10**9))
    for k in ordered_k:
        k_label = "zeroshot" if k == 0 else ("all_half" if k < 0 else str(k))
        print(
            f"  [{stream.subject_id}] {head} k={k_label} ...",
            flush=True,
        )
        fold_results = []
        for fold in range(hp.n_folds):
            if k == 0:
                model = load_fold_model(
                    build_shallow, init_run, head=head, fold=fold, device=device
                )
            else:
                model = fold_state[fold]
                if model is None:
                    model = load_fold_model(
                        build_shallow, init_run, head=head, fold=fold, device=device
                    )
                cal_cues = cues_for_k(split.train_cues, k)
                ft_tr, ft_va = split_train_val_cues(
                    cal_cues, val_ratio=hp.val_ratio, seed=hp.seed
                )
                mask_tr = window_mask_for_cues(stream.cue_ids, ft_tr)
                mask_va = window_mask_for_cues(stream.cue_ids, ft_va)
                model = finetune_on_mask(
                    model,
                    stream,
                    mask_tr=mask_tr,
                    mask_va=mask_va,
                    head=head,
                    device=device,
                    hp=hp,
                    fold=fold,
                    replay_pack=replay_pack,
                    split=split,
                )
                fold_state[fold] = model
            agg = eval_mask_pack(
                stream,
                mask_eval,
                model,
                head=head,
                device=device,
                use_noz_pipeline=True,
            )
            fold_results.append(
                {
                    "fold": fold,
                    "acc_paper": float(agg["segment_metrics"]["acc_paper"]),
                }
            )
            if k == 0:
                del model
                fold_state[fold] = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        accs = [r["acc_paper"] for r in fold_results]
        m, s = mean_std(accs)
        print(
            f"  [{stream.subject_id}] {head} k={k_label} "
            f"Acc_paper={m:.4f}±{s:.4f}",
            flush=True,
        )
        curve.append(
            {
                "k_cues": int(k),
                "k_label": "zeroshot" if k == 0 else ("all_half" if k < 0 else str(k)),
                "folds": fold_results,
                "acc_paper_mean": float(m),
                "acc_paper_std": float(s),
            }
        )
    return {
        "subject_id": stream.subject_id,
        "head": head,
        "init_run": str(init_run),
        "n_eval_cues": split.n_eval,
        "curve": curve,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="方案25 增量 FT 爬坡曲线")
    p.add_argument("--arm", default="A0", choices=("A0", "G1", "G2", "G3"))
    p.add_argument("--run-stamp", default="", help="OpenBMI shallow run 目录名")
    p.add_argument("--subjects", default="")
    p.add_argument("--skip-three", action="store_true")
    p.add_argument("--skip-task", action="store_true", help="仅评 three（主读数）")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-epochs", type=int, default=0, help="覆盖 max_epochs；0=默认 300")
    p.add_argument("--patience", type=int, default=0, help="覆盖 patience；0=默认 20")
    p.add_argument("--replay-ratio", type=float, default=0.15, help="G2 replay 比例")
    p.add_argument("--aug", default="off", help="G3: g3 / ft_light")
    p.add_argument("--k-list", default="", help="逗号分隔；默认 0,10,20,40,80,-1")
    p.add_argument(
        "--train-device",
        default="5070",
        choices=("5070", "5090"),
        help="G1/G2/G3 权重所在机位（A0 始终 5070 S3）",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    hp = SHARED
    if args.smoke:
        hp = replace(hp, max_epochs=2, patience=2, n_folds=1)
    if args.max_epochs > 0:
        hp = replace(hp, max_epochs=int(args.max_epochs))
    if args.patience > 0:
        hp = replace(hp, patience=int(args.patience))
    if args.arm == "G2":
        hp = replace(hp, replay_ratio=float(args.replay_ratio))
    if args.arm == "G3":
        hp = replace(hp, aug=aug_config_from_spec(args.aug or "g3"))

    k_list = INCREMENTAL_K_LIST
    if args.k_list.strip():
        k_list = tuple(int(x) for x in args.k_list.split(",") if x.strip())

    init_run = resolve_weight_run(
        args.arm, run_stamp=args.run_stamp or None, train_device=args.train_device
    )
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    device = torch.device(args.device)
    seed_everything(hp.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_ROOT / f"S25-{args.arm}" / f"{stamp}_incremental_{args.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    meta = {
        "arm": args.arm,
        "protocol": PROTOCOL,
        "init_run": str(init_run),
        "hparams": jsonable(asdict(hp)),
        "k_list": list(k_list),
        "anchor_09_A0_three": ANCHOR_09_A0_EVAL_HALF_THREE,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    all_subj: dict = {}
    md = [
        f"# S25-{args.arm} 增量 FT 爬坡 · {stamp}",
        "",
        f"- 权重：`{init_run}`",
        f"- k 列表：`{list(k_list)}`",
        "",
    ]

    for stream in iter_subject_streams(subjects=subjects):
        print(f"\n=== {stream.subject_id} ===", flush=True)
        sub: dict = {}
        replay = None
        if args.arm == "G2":
            replay = load_openbmi_replay(init_run, "three", fold=0)
        for head in ("task", "three"):
            if head == "three" and args.skip_three:
                continue
            if head == "task" and args.skip_task:
                continue
            if args.arm == "G2" and replay is not None:
                replay_h = load_openbmi_replay(init_run, head, fold=0)
            else:
                replay_h = replay
            row = run_incremental_curve(
                stream,
                init_run=init_run,
                head=head,
                device=device,
                hp=hp,
                k_list=k_list if not args.smoke else (0,),
                replay_pack=replay_h,
            )
            sub[head] = row
        all_subj[stream.subject_id] = sub
        md += [f"## {stream.subject_id}", ""]
        if "three" in sub:
            md += ["| k | Three Acc_paper |", "|---|---|"]
            for pt in sub["three"]["curve"]:
                md.append(
                    f"| {pt['k_label']} | {pt['acc_paper_mean']*100:.2f}% |"
                )
            md.append("")

    summary = {"arm": args.arm, "subjects": all_subj, "meta": meta}
    (out_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
