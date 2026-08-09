"""全模型微调：OpenBMI Acc_paper init → 通道重排 → 前半训 → 后半伪在线评。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
STEP = HERE.parent
HOP100 = STEP / "baselines_2s_hop100"
PSEUDO = STEP / "game_pseudo_online_hop100"
for p in (STEP, HOP100, PSEUDO, HERE):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
sys.path.insert(0, str(STEP))
sys.path.insert(0, str(HOP100))
sys.path.insert(0, str(PSEUDO))
sys.path.insert(0, str(HERE))

from channel_remap import OPENBMI_CHANS, remap_windows_to_openbmi  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_SESSIONS,
    DOCS_OUT,
    FT_WEIGHT_ROOT,
    MODELS,
    N_FOLDS,
    N_TIMES,
    OPENBMI_SHALLOW_RUN,
    PROTOCOL,
    SESSIONS_ROOT,
)
from data_split import (  # noqa: E402
    TrialSplit,
    assert_no_leakage,
    build_trial_split,
    window_mask_for_trials,
    write_split_artifacts,
)
from dataset import ArrayTaskDataset, ArrayThreeDataset  # noqa: E402
from eval_metrics import aggregate_windows_to_segments, mean_std  # noqa: E402
from infer import load_fold_model, predict_windows  # noqa: E402
from shared_hparams import SHARED, SharedTrainHP  # noqa: E402
from stream import EvalStream, build_eval_stream  # noqa: E402
from task_sampler import make_balanced_sampler  # noqa: E402
from weights import resolve_openbmi_accpaper_run  # noqa: E402

BuildFn = Callable[..., nn.Module]


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


def _make_ds(X: np.ndarray, y: np.ndarray, n_outputs: int):
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


def _assert_full_model_trainable(model: nn.Module) -> None:
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise RuntimeError(
            "finetune_mode=full_model 但存在冻结参数: "
            + ", ".join(frozen[:8])
            + ("..." if len(frozen) > 8 else "")
        )
    n_param = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_param != n_train:
        raise RuntimeError(f"可训参数 {n_train} != 总参数 {n_param}")


def _eval_segment_acc_paper(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    seg_keys: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    hp: SharedTrainHP,
    *,
    n_classes: int,
) -> tuple[dict, dict, float]:
    Xs, ys, keys = X[mask], y[mask], seg_keys[mask]
    if len(ys) == 0:
        empty = {
            "n_segments": 0,
            "n_windows": 0,
            "acc_paper": 0.0,
            "balanced_accuracy": 0.0,
        }
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


def _zeroshot_eval(
    build_model: BuildFn,
    init_run: Path,
    *,
    head: str,
    fold: int,
    stream: EvalStream,
    eval_mask: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
    n_classes: int,
) -> dict:
    net = load_fold_model(
        build_model, init_run, head=head, fold=fold, device=device
    )
    pred = predict_windows(
        net, stream.X[eval_mask], device, batch_size=batch_size
    )
    agg = aggregate_windows_to_segments(
        y[eval_mask], pred, stream.seg_keys[eval_mask], n_classes=n_classes
    )
    del net
    return agg


def finetune_one_fold(
    *,
    model_name: str,
    build_model: BuildFn,
    stream: EvalStream,
    split: TrialSplit,
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

    mask_ft_tr = window_mask_for_trials(stream.trial_ids, split.ft_train_trials)
    mask_ft_va = window_mask_for_trials(stream.trial_ids, split.ft_val_trials)
    mask_eval = window_mask_for_trials(stream.trial_ids, split.eval_trials)

    if int(mask_eval.sum()) == 0:
        raise RuntimeError(f"{stream.subject_id}: eval half 无窗")
    if int(mask_ft_tr.sum()) == 0:
        raise RuntimeError(f"{stream.subject_id}: ft_train 无窗")

    # 泄漏硬检查：eval 与 train/val 无交集
    if np.any(mask_eval & (mask_ft_tr | mask_ft_va)):
        raise RuntimeError("泄漏：eval 窗与 ft_train/val 重叠")

    zeroshot = None
    if also_zeroshot:
        zeroshot = _zeroshot_eval(
            build_model,
            init_run,
            head=head,
            fold=fold,
            stream=stream,
            eval_mask=mask_eval,
            y=y,
            device=device,
            batch_size=hp.batch_eval,
            n_classes=n_outputs,
        )

    seed_everything(hp.seed + fold)
    model = load_fold_model(
        build_model, init_run, head=head, fold=fold, device=device
    )
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    _assert_full_model_trainable(model)

    g = torch.Generator()
    g.manual_seed(hp.seed + fold)
    y_tr = y[mask_ft_tr]
    train_loader = DataLoader(
        _make_ds(stream.X[mask_ft_tr], y_tr, n_outputs),
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

    best_score, best_state, best_ep = -1.0, None, 0
    best_val_seg: dict | None = None
    best_val_win: dict | None = None
    bad = 0

    fold_dir = ckpt_dir / head / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    log_line(
        log_path,
        f"[{stream.subject_id}] {model_name} {head} fold{fold} "
        f"ft_tr={int(mask_ft_tr.sum())} ft_va={int(mask_ft_va.sum())} "
        f"eval={int(mask_eval.sum())} full_model=True",
    )

    for ep in range(1, hp.max_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        if int(mask_ft_va.sum()) > 0:
            val_seg, val_win, va_loss = _eval_segment_acc_paper(
                model,
                stream.X,
                y,
                stream.seg_keys,
                mask_ft_va,
                device,
                hp,
                n_classes=n_outputs,
            )
            score = float(val_seg["acc_paper"])
        else:
            # 极端无 Val：用 train Acc_paper 仅作监控，仍跑满 patience 逻辑会弱
            val_seg, val_win, va_loss = _eval_segment_acc_paper(
                model,
                stream.X,
                y,
                stream.seg_keys,
                mask_ft_tr,
                device,
                hp,
                n_classes=n_outputs,
            )
            score = float(val_seg["acc_paper"])
            log_line(log_path, "WARN: ft_val 为空，早停监控退化为 ft_train")

        log_line(
            log_path,
            f"fold{fold} ep {ep:03d} tr={tr:.4f} va={va_loss:.4f} "
            f"val_AccPaper={score:.4f} "
            f"val_BalAccMaj={float(val_seg.get('balanced_accuracy', 0)):.4f}",
        )
        if score > best_score:
            best_score = score
            best_ep = ep
            best_val_seg = val_seg
            best_val_win = val_win
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
            torch.save(
                {
                    "stage": "openbmi_game_ft_half",
                    "fold": fold,
                    "model_name": model_name,
                    "subject_id": stream.subject_id,
                    "n_outputs": n_outputs,
                    "protocol": hp.protocol,
                    "finetune_mode": "full_model",
                    "freeze_backbone": False,
                    "head_only": False,
                    "no_rap": True,
                    "balbatch": True,
                    "early_stop": "acc_paper",
                    "init_domain": "openbmi",
                    "channel_remap": "game_to_openbmi",
                    "init_run": str(init_run),
                    "init_fold": fold,
                    "train_trials": split.ft_train_trials,
                    "val_trials": split.ft_val_trials,
                    "eval_trials": split.eval_trials,
                    "model": best_state,
                    "epoch": ep,
                    "val_segment_metrics": val_seg,
                    "val_window_metrics": val_win,
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
    log_line(
        log_path,
        f"[eval_half] Acc_paper={te_seg['acc_paper']:.4f} "
        f"BalAcc_maj={te_seg['balanced_accuracy']:.4f} "
        f"win_BalAcc={te_win['balanced_accuracy']:.4f}",
    )

    out = {
        "fold": fold,
        "best_epoch": best_ep,
        "best_val_acc_paper": float(best_score),
        "val_segment_metrics": best_val_seg,
        "val_window_metrics": best_val_win,
        "eval_segment_metrics": te_seg,
        "eval_window_metrics": te_win,
        "n_ft_train_windows": int(mask_ft_tr.sum()),
        "n_ft_val_windows": int(mask_ft_va.sum()),
        "n_eval_windows": int(mask_eval.sum()),
        "ckpt": str(fold_dir / ckpt_name),
        "finetune_mode": "full_model",
        "freeze_backbone": False,
        "head_only": False,
    }
    if zeroshot is not None:
        out["zeroshot_eval_segment_metrics"] = zeroshot["segment_metrics"]
        out["zeroshot_eval_window_metrics"] = zeroshot["window_metrics"]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _summarize_folds(fold_rows: list[dict], *, head: str) -> dict:
    ap = [float(r["eval_segment_metrics"]["acc_paper"]) for r in fold_rows]
    bm = [
        float(r["eval_segment_metrics"]["balanced_accuracy"]) for r in fold_rows
    ]
    wb = [
        float(r["eval_window_metrics"]["balanced_accuracy"]) for r in fold_rows
    ]
    out = {
        "head": head,
        "n_folds": len(fold_rows),
        "acc_paper_mean": mean_std(ap)[0],
        "acc_paper_std": mean_std(ap)[1],
        "balacc_maj_mean": mean_std(bm)[0],
        "balacc_maj_std": mean_std(bm)[1],
        "window_balacc_mean": mean_std(wb)[0],
        "window_balacc_std": mean_std(wb)[1],
        "folds": fold_rows,
    }
    if fold_rows and "zeroshot_eval_segment_metrics" in fold_rows[0]:
        zap = [
            float(r["zeroshot_eval_segment_metrics"]["acc_paper"])
            for r in fold_rows
        ]
        out["zeroshot_acc_paper_mean"] = mean_std(zap)[0]
        out["zeroshot_acc_paper_std"] = mean_std(zap)[1]
    return out


def run_ft_main(
    *,
    model_name: str,
    build_model: BuildFn,
    structure_note: str,
    extra_meta: dict | None = None,
) -> None:
    if model_name not in MODELS:
        raise SystemExit(f"模型 {model_name} 不在本臂名单: {MODELS}")

    p = argparse.ArgumentParser(
        description=f"{model_name} OpenBMI init · 游戏全模型微调（前半训/后半评）"
    )
    p.add_argument("--sessions", default=",".join(DEFAULT_SESSIONS))
    p.add_argument("--subjects", default="", help="过滤 subject_id，逗号分隔")
    p.add_argument("--skip-three", action="store_true")
    p.add_argument("--smoke", action="store_true", help="仅 fold0 + 短 epoch")
    p.add_argument("--max-epochs", type=int, default=0, help=">0 覆盖默认")
    p.add_argument("--patience", type=int, default=0, help=">0 覆盖默认")
    p.add_argument("--no-filter", action="store_true")
    p.add_argument(
        "--run-stamp",
        default="",
        help=f"指定 OpenBMI init run（默认 {OPENBMI_SHALLOW_RUN}）",
    )
    p.add_argument("--no-zeroshot", action="store_true", help="不做后半零样本对照")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
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

    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    subject_filter = {
        s.strip() for s in args.subjects.split(",") if s.strip()
    }
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    device = torch.device(args.device)
    also_zeroshot = not args.no_zeroshot

    init_run = resolve_openbmi_accpaper_run(
        model_name, run_stamp=args.run_stamp or None
    )
    print(f"[init] OpenBMI {model_name} Acc_paper: {init_run}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{model_name}_openbmi_game_ft_half"
    result_root = DOCS_OUT / "results" / f"{stamp}_{out_name}"
    result_root.mkdir(parents=True, exist_ok=True)
    weight_root = (
        FT_WEIGHT_ROOT
        / f"{model_name}_openbmi_game_ft_half_balbatch_accpaper"
        / stamp
    )
    weight_root.mkdir(parents=True, exist_ok=True)
    log_path = result_root / "run.log"

    meta = {
        "stamp": stamp,
        "model_name": model_name,
        "out_name": out_name,
        "arm": "05_openbmi_finetune_first_half_train_second_half_eval",
        "protocol": PROTOCOL,
        "structure_note": structure_note,
        "sessions": sessions,
        "folds": list(folds),
        "init_domain": "openbmi",
        "init_run": str(init_run),
        "channel_remap": list(OPENBMI_CHANS),
        "weight_root": str(weight_root),
        "device": str(device),
        "finetune_mode": "full_model",
        "freeze_backbone": False,
        "head_only": False,
        "hparams": asdict(hp),
        "n_times": N_TIMES,
        "extra_meta": extra_meta or {},
        "also_zeroshot_on_eval_half": also_zeroshot,
    }
    (result_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (weight_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        f"# 05 OpenBMI 游戏微调实验记录（{stamp} / {out_name}）",
        "",
        f"- protocol：`{PROTOCOL}`",
        f"- model：`{model_name}` | {structure_note}",
        f"- **finetune_mode=full_model**（全参数可训；非 head-only）",
        f"- init_domain：`openbmi`",
        f"- init：`{init_run}`",
        f"- channel_remap：游戏序 → `{', '.join(OPENBMI_CHANS)}`",
        f"- weights：`{weight_root}`",
        f"- early_stop=**Val Acc_paper** | max_epochs={hp.max_epochs} | patience={hp.patience}",
        f"- 主报：后半 trial **段级 Acc_paper**",
        "",
    ]

    for name in sessions:
        log_line(log_path, f"=== stream {name} ===")
        stream = build_eval_stream(
            SESSIONS_ROOT / name, apply_filter=not args.no_filter
        )
        if subject_filter and stream.subject_id not in subject_filter:
            log_line(log_path, f"skip {stream.subject_id} (subjects filter)")
            continue
        ch_names = list(stream.meta.get("channels") or [])
        X_remap = remap_windows_to_openbmi(stream.X, ch_names)
        stream.X = X_remap
        stream.meta = {
            **dict(stream.meta),
            "channels_src": ch_names,
            "channels": list(OPENBMI_CHANS),
            "channel_remap": "game_to_openbmi",
        }
        log_line(
            log_path,
            f"remap {ch_names} -> {list(OPENBMI_CHANS)} X={tuple(stream.X.shape)}",
        )
        split = build_trial_split(
            stream, val_ratio=hp.val_ratio, seed=hp.seed
        )
        assert_no_leakage(split)
        write_split_artifacts(stream, split, DOCS_OUT / "out" / name)
        subj = stream.subject_id
        subj_ckpt = weight_root / subj
        log_line(
            log_path,
            f"subject={subj} trials={split.n_all} "
            f"train={split.n_train} eval={split.n_eval}",
        )
        md_lines.append(f"## {subj} / {name}")
        md_lines.append("")
        md_lines.append(
            f"- split：train_trials={split.n_train} eval_trials={split.n_eval} "
            f"| ft_train={len(split.ft_train_trials)} "
            f"ft_val={len(split.ft_val_trials)}"
        )
        md_lines.append(
            f"- remap：`{', '.join(ch_names)}` → `{', '.join(OPENBMI_CHANS)}`"
        )
        md_lines.append("")

        for head in ("task", "three"):
            if head == "three" and args.skip_three:
                md_lines.extend(["### three", "", "- （本次跳过）", ""])
                continue
            fold_rows = []
            for fold in folds:
                row = finetune_one_fold(
                    model_name=model_name,
                    build_model=build_model,
                    stream=stream,
                    split=split,
                    init_run=init_run,
                    head=head,
                    fold=fold,
                    device=device,
                    hp=hp,
                    ckpt_dir=subj_ckpt,
                    log_path=log_path,
                    also_zeroshot=also_zeroshot,
                )
                fold_rows.append(row)
            summ = _summarize_folds(fold_rows, head=head)
            summ["model"] = model_name
            summ["init_run"] = init_run.name
            summ["n_eval_windows"] = int(
                window_mask_for_trials(
                    stream.trial_ids, split.eval_trials
                ).sum()
            )
            summary = {
                "subject_id": subj,
                "session": name,
                "head": head,
                "split": {
                    "train_trials": split.train_trials,
                    "eval_trials": split.eval_trials,
                    "ft_train_trials": split.ft_train_trials,
                    "ft_val_trials": split.ft_val_trials,
                },
                "model": summ,
            }
            (result_root / f"{subj}_{head}_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            md_lines.extend(
                [
                    f"### {head}",
                    "",
                    f"| 模型 | 后半 Acc_paper | 后半 BalAcc_maj | 窗 BalAcc | "
                    f"零样本后半 Acc_paper* |",
                    f"|------|----------------|----------------|-----------|"
                    f"----------------------|",
                    f"| {model_name} | "
                    f"{summ['acc_paper_mean']:.4f} ± {summ['acc_paper_std']:.4f} | "
                    f"{summ['balacc_maj_mean']:.4f} ± {summ['balacc_maj_std']:.4f} | "
                    f"{summ['window_balacc_mean']:.4f} ± {summ['window_balacc_std']:.4f} | "
                    + (
                        f"{summ.get('zeroshot_acc_paper_mean', float('nan')):.4f} ± "
                        f"{summ.get('zeroshot_acc_paper_std', float('nan')):.4f} |"
                        if "zeroshot_acc_paper_mean" in summ
                        else "— |"
                    ),
                    "",
                    "\\*同后半测试集、同 init、未微调对照。",
                    "",
                ]
            )

    md_path = result_root / f"{out_name}实验结果.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    log_line(log_path, f"done -> {result_root}")
