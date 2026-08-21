"""S08-02 / S08-05：S3 init · Stieger cue 前半 FT · 后半评（含同后半零样本 Δ）。"""

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
from braindecode.models import ShallowFBCSPNet
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
HOP100 = STEP / "baselines_2s_hop100"
for p in (str(HOP100), str(STEP), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(STEP))
sys.path.insert(0, str(HOP100))
sys.path.insert(0, str(HERE))

from config import (  # noqa: E402
    DOCS_07,
    FT_WEIGHT_ROOT,
    N_FOLDS,
    N_TIMES,
    PROTOCOL,
    RESULTS_ROOT,
)
from data import SubjectStream, iter_subject_streams  # noqa: E402
from data_split import (  # noqa: E402
    CueSplit,
    assert_no_leakage,
    build_cue_split,
    window_mask_for_cues,
    write_split_artifacts,
)
from dataset import ArrayTaskDataset, ArrayThreeDataset  # noqa: E402
from infer import load_fold_model, predict_windows  # noqa: E402
from task_sampler import make_balanced_sampler  # noqa: E402
from util_metrics import aggregate_windows_to_segments, jsonable, mean_std  # noqa: E402
from weights import resolve_openbmi_s3_run  # noqa: E402

RESULTS = RESULTS_ROOT / "S08-02_ft_half"
RESULTS_05 = RESULTS_ROOT / "S08-05_ft_half"


@dataclass(frozen=True)
class SharedTrainHP:
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
    early_stop: str = "acc_paper"
    n_times_expected: int = N_TIMES
    finetune_mode: str = "full_model"


SHARED = SharedTrainHP()


def build_shallow(n_chans, n_times, n_outputs, drop_prob):
    return ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, drop_prob=drop_prob
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_line(log_path: Path, msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _make_ds(X, y, n_outputs: int):
    if n_outputs == 2:
        return ArrayTaskDataset(X, y)
    return ArrayThreeDataset(X, y)


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


def _eval_segment_acc_paper(
    model, X, y, seg_keys, mask, device, hp: SharedTrainHP, *, n_classes: int
):
    Xs, ys, keys = X[mask], y[mask], seg_keys[mask]
    if len(ys) == 0:
        empty = {"n_segments": 0, "n_windows": 0, "acc_paper": 0.0, "balanced_accuracy": 0.0}
        return empty, empty, float("inf")
    loader = DataLoader(
        _make_ds(Xs, ys, n_classes),
        batch_size=hp.batch_eval,
        shuffle=False,
        num_workers=0,
    )
    loss = run_epoch(model, loader, nn.CrossEntropyLoss(), None, device, False)
    pred = predict_windows(model, Xs, device, batch_size=hp.batch_eval)
    agg = aggregate_windows_to_segments(ys, pred, keys, n_classes=n_classes)
    return agg["segment_metrics"], agg["window_metrics"], float(loss)


def finetune_one_fold(
    *,
    stream: SubjectStream,
    split: CueSplit,
    init_run: Path,
    head: str,
    fold: int,
    device: torch.device,
    hp: SharedTrainHP,
    ckpt_dir: Path,
    log_path: Path,
    also_zeroshot: bool,
) -> dict:
    n_outputs = 2 if head == "task" else 3
    y = stream.y_task if head == "task" else stream.y_three
    ckpt_name = f"best_{head}.pt"

    mask_ft_tr = window_mask_for_cues(stream.cue_ids, split.ft_train_cues)
    mask_ft_va = window_mask_for_cues(stream.cue_ids, split.ft_val_cues)
    mask_eval = window_mask_for_cues(stream.cue_ids, split.eval_cues)
    if int(mask_eval.sum()) == 0:
        raise RuntimeError(f"{stream.subject_id}: eval half 无窗")
    if int(mask_ft_tr.sum()) == 0:
        raise RuntimeError(f"{stream.subject_id}: ft_train 无窗")
    if np.any(mask_eval & (mask_ft_tr | mask_ft_va)):
        raise RuntimeError("泄漏：eval 与 ft_train/val 重叠")

    zeroshot = None
    if also_zeroshot:
        net0 = load_fold_model(
            build_shallow, init_run, head=head, fold=fold, device=device
        )
        pred0 = predict_windows(
            net0, stream.X[mask_eval], device, batch_size=hp.batch_eval
        )
        zeroshot = aggregate_windows_to_segments(
            y[mask_eval], pred0, stream.seg_keys[mask_eval], n_classes=n_outputs
        )
        del net0

    seed_everything(hp.seed + fold)
    model = load_fold_model(
        build_shallow, init_run, head=head, fold=fold, device=device
    )
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    g = torch.Generator()
    g.manual_seed(hp.seed + fold)
    y_tr = y[mask_ft_tr]
    train_loader = DataLoader(
        _make_ds(stream.X[mask_ft_tr], y_tr, n_outputs),
        batch_size=hp.batch_train,
        sampler=make_balanced_sampler(y_tr, n_classes=n_outputs, generator=g),
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
    )

    best_score, best_state, best_ep = -1.0, None, 0
    best_val_seg = None
    bad = 0
    fold_dir = ckpt_dir / head / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    log_line(
        log_path,
        f"[{stream.subject_id}] {head} fold{fold} "
        f"ft_tr={int(mask_ft_tr.sum())} ft_va={int(mask_ft_va.sum())} "
        f"eval={int(mask_eval.sum())}",
    )

    for ep in range(1, hp.max_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        mon_mask = mask_ft_va if int(mask_ft_va.sum()) > 0 else mask_ft_tr
        val_seg, val_win, va_loss = _eval_segment_acc_paper(
            model,
            stream.X,
            y,
            stream.seg_keys,
            mon_mask,
            device,
            hp,
            n_classes=n_outputs,
        )
        score = float(val_seg["acc_paper"])
        log_line(
            log_path,
            f"fold{fold} ep {ep:03d} tr={tr:.4f} va={va_loss:.4f} "
            f"val_AccPaper={score:.4f}",
        )
        if score > best_score:
            best_score = score
            best_ep = ep
            best_val_seg = val_seg
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
            torch.save(
                {
                    "stage": "stieger_s3_ft_half",
                    "fold": fold,
                    "subject_id": stream.subject_id,
                    "n_outputs": n_outputs,
                    "early_stop": "acc_paper",
                    "init_run": str(init_run),
                    "train_cues": split.ft_train_cues,
                    "val_cues": split.ft_val_cues,
                    "eval_cues": split.eval_cues,
                    "model": best_state,
                    "epoch": ep,
                    "val_segment_metrics": val_seg,
                    "hparams": asdict(hp),
                },
                fold_dir / ckpt_name,
            )
        else:
            bad += 1
            if bad >= hp.patience:
                log_line(log_path, f"  early stop @ ep {ep}")
                break

    if best_state is None:
        raise RuntimeError(f"fold{fold} 未产生 best ckpt")
    model.load_state_dict(best_state)
    model.eval()
    te_seg, te_win, _ = _eval_segment_acc_paper(
        model,
        stream.X,
        y,
        stream.seg_keys,
        mask_eval,
        device,
        hp,
        n_classes=n_outputs,
    )
    out = {
        "fold": fold,
        "best_epoch": best_ep,
        "best_val_acc_paper": float(best_score),
        "val_segment_metrics": best_val_seg,
        "eval_segment_metrics": te_seg,
        "eval_window_metrics": te_win,
        "ckpt": str(fold_dir / ckpt_name),
    }
    if zeroshot is not None:
        out["zeroshot_eval_segment_metrics"] = zeroshot["segment_metrics"]
        zap = float(zeroshot["segment_metrics"]["acc_paper"])
        out["delta_vs_zeroshot"] = float(te_seg["acc_paper"] - zap)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="S08-02/05 Stieger 前半 FT")
    p.add_argument("--subjects", default="")
    p.add_argument("--skip-three", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--run-stamp", default="")
    p.add_argument("--no-zeroshot", action="store_true")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    hp = SHARED
    if args.smoke:
        hp = replace(
            hp,
            max_epochs=args.max_epochs if args.max_epochs > 0 else 2,
            patience=args.patience if args.patience > 0 else 2,
        )
    elif args.max_epochs > 0 or args.patience > 0:
        hp = replace(
            hp,
            max_epochs=args.max_epochs if args.max_epochs > 0 else hp.max_epochs,
            patience=args.patience if args.patience > 0 else hp.patience,
        )

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    device = torch.device(args.device)
    init_run = resolve_openbmi_s3_run("shallow", run_stamp=args.run_stamp or None)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    RESULTS_05.mkdir(parents=True, exist_ok=True)
    result_root = RESULTS / f"{stamp}_shallow_ft_half"
    result_root.mkdir(parents=True, exist_ok=True)
    weight_root = (
        FT_WEIGHT_ROOT / "shallow_stieger_ft_half_balbatch_accpaper" / stamp
    )
    weight_root.mkdir(parents=True, exist_ok=True)
    log_path = result_root / "run.log"

    meta = {
        "arm": "S08-02/S08-05",
        "protocol": PROTOCOL,
        "init_run": str(init_run),
        "weight_root": str(weight_root),
        "hparams": asdict(hp),
        "folds": list(folds),
    }
    (result_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (weight_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        f"# S08-02/05 Stieger 前半 FT · {stamp}",
        "",
        f"- init：`{init_run}`",
        f"- weights：`{weight_root}`",
        f"- 划分：cue 时间对半；Val Acc_paper 早停",
        "",
    ]

    for stream in iter_subject_streams(subjects=subjects):
        split = build_cue_split(stream, val_ratio=hp.val_ratio, seed=hp.seed)
        assert_no_leakage(split)
        write_split_artifacts(stream, split, DOCS_07 / "out" / stream.subject_id)
        log_line(
            log_path,
            f"=== {stream.subject_id} cues={split.n_all} "
            f"train={split.n_train} eval={split.n_eval} ===",
        )
        md += [
            f"## {stream.subject_id}",
            "",
            f"- cues：all={split.n_all} train={split.n_train} eval={split.n_eval}",
            "",
        ]
        subj_ckpt = weight_root / stream.subject_id
        for head in ("task", "three"):
            if head == "three" and args.skip_three:
                continue
            fold_rows = []
            for fold in folds:
                row = finetune_one_fold(
                    stream=stream,
                    split=split,
                    init_run=init_run,
                    head=head,
                    fold=fold,
                    device=device,
                    hp=hp,
                    ckpt_dir=subj_ckpt,
                    log_path=log_path,
                    also_zeroshot=not args.no_zeroshot,
                )
                fold_rows.append(row)
            ap = [float(r["eval_segment_metrics"]["acc_paper"]) for r in fold_rows]
            summ = {
                "acc_paper_mean": mean_std(ap)[0],
                "acc_paper_std": mean_std(ap)[1],
                "folds": fold_rows,
            }
            if "zeroshot_eval_segment_metrics" in fold_rows[0]:
                zap = [
                    float(r["zeroshot_eval_segment_metrics"]["acc_paper"])
                    for r in fold_rows
                ]
                summ["zeroshot_acc_paper_mean"] = mean_std(zap)[0]
                summ["zeroshot_acc_paper_std"] = mean_std(zap)[1]
                summ["delta_mean"] = float(
                    summ["acc_paper_mean"] - summ["zeroshot_acc_paper_mean"]
                )
            (result_root / f"{stream.subject_id}_{head}_summary.json").write_text(
                json.dumps(jsonable(summ), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            md += [
                f"### {head}",
                "",
                f"- FT Acc_paper：`{summ['acc_paper_mean']:.4f}±{summ['acc_paper_std']:.4f}`",
            ]
            if "zeroshot_acc_paper_mean" in summ:
                md.append(
                    f"- 同后半零样本：`{summ['zeroshot_acc_paper_mean']:.4f}±"
                    f"{summ['zeroshot_acc_paper_std']:.4f}` · "
                    f"Δ=`{summ['delta_mean']:+.4f}`"
                )
            md.append("")

    # S08-05 软链说明
    note = (
        f"# S08-05 ≡ S08-02（本套 init 已是 OpenBMI S3）\n\n"
        f"主记录见：`../S08-02_ft_half/{stamp}_shallow_ft_half/`\n"
        f"权重：`{weight_root}`\n"
    )
    (RESULTS_05 / f"{stamp}_see_S08-02.md").write_text(note, encoding="utf-8")

    md_path = result_root / "report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    log_line(log_path, f"done -> {result_root}")


if __name__ == "__main__":
    main()
