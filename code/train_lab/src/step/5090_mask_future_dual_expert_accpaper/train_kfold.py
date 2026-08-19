"""Acc_paper 五折训练环（A0 自写 500pt / pf1000 自写模型臂）。"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from _paths import OUT_ROOT, PRE
from arms_registry import ArmSpec
from data_io import load_arrays, summarize_labels, to_bct
from feat_index import assert_default_map, assert_future_perturbation
from losses import compute_losses
from md_fold_detail import three_fold_md_lines
from metrics import (
    binary_task_metrics,
    format_task_metrics,
    format_three_metrics,
    jsonify_metrics,
    three_class_metrics,
)
from model import MaskFutureDualExpert
from shared_hparams import OUT_ROOT_TAG, SHARED, SharedTrainHP
from sigreg import SIGReg
from trial_metrics import aggregate_windows_to_trials

sys.path.insert(0, str(PRE))
from src.common.steps.split_subjects import iter_subject_kfold  # noqa: E402


class WinDS(Dataset):
    def __init__(self, x_full, x_mask, y, trial_id, subjects, idx):
        self.xf = x_full
        self.xm = x_mask
        self.y = y
        self.tid = trial_id
        self.subjects = subjects
        self.idx = np.asarray(idx, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        xf = torch.from_numpy(np.asarray(self.xf[j], dtype=np.float32))
        xm = torch.from_numpy(np.asarray(self.xm[j], dtype=np.float32))
        if xf.ndim == 2 and xf.shape[0] in (500, 600, 1000):
            xf = xf.T.contiguous()
            xm = xm.T.contiguous()
        return (
            xf,
            xm,
            int(self.y[j]),
            int(self.tid[j]),
            str(self.subjects[j]),
        )


def _loader_hp(hp: SharedTrainHP) -> SharedTrainHP:
    """Windows spawn 无法 pickle 大 memmap；强制单进程加载。"""
    nw = hp.num_workers
    if sys.platform == "win32" and nw > 0:
        print(
            f"[train_kfold] Windows: num_workers {nw}->0 (memmap+spawn)",
            flush=True,
        )
        nw = 0
    if nw == hp.num_workers:
        return hp
    return replace(
        hp,
        num_workers=nw,
        pin_memory=hp.pin_memory and nw > 0,
        persistent_workers=hp.persistent_workers and nw > 0,
    )


def _balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    freq = {int(c): float(n) for c, n in zip(classes, counts)}
    w = np.array([1.0 / freq[int(t)] for t in y], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(w),
        replacement=True,
    )


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def _window_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    if n_classes == 2:
        return jsonify_metrics(binary_task_metrics(y_true, y_pred))
    return jsonify_metrics(three_class_metrics(y_true, y_pred))


def _resolve_mask_input(
    model: MaskFutureDualExpert,
    xf: torch.Tensor,
    xm: torch.Tensor,
    arm: ArmSpec,
) -> torch.Tensor:
    if arm.leak_eval_full:
        return xf
    if arm.mask_learnable:
        return model.make_mask(xf)
    return xm


@torch.no_grad()
def _eval_split(
    model: MaskFutureDualExpert,
    loader: DataLoader,
    device: torch.device,
    arm: ArmSpec,
    *,
    n_classes: int,
) -> tuple[dict, dict]:
    """返回 (trial_metrics, window_metrics)。"""
    model.eval()
    ys, ps, subs, tids = [], [], [], []
    for xf, xm, y, tid, subj in loader:
        xf = xf.to(device, non_blocking=True)
        xm = xm.to(device, non_blocking=True)
        if arm.a1_600:
            xf = xf[..., :600]
            xm = xm[..., :600]
        x_in = _resolve_mask_input(model, xf, xm, arm)
        out = model(x_in, x_full=None, train_mode=False)
        pred = out["p_final"].argmax(dim=-1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
        if isinstance(subj, (list, tuple)):
            subs.extend(str(s) for s in subj)
        else:
            subs.extend(str(s) for s in np.asarray(subj).tolist())
        tids.append(tid.numpy())

    yt = np.concatenate(ys).astype(np.int64)
    yp = np.concatenate(ps).astype(np.int64)
    sub_arr = np.asarray(subs, dtype=object)
    tid_arr = np.concatenate(tids).astype(np.int64)
    trial = aggregate_windows_to_trials(
        yt, yp, sub_arr, tid_arr, n_classes=n_classes
    )
    win_m = _window_metrics(yt, yp, n_classes)
    return trial["metrics"], win_m


def _run_train_epoch(
    model: MaskFutureDualExpert,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    arm: ArmSpec,
    hp: SharedTrainHP,
    sigreg: SIGReg | None,
    *,
    lam_pred: float,
    lam_sig: float,
    lam_dec: float,
) -> float:
    model.train()
    total, n = 0.0, 0
    for xf, xm, yy, _tid, _subj in loader:
        xf = xf.to(device, non_blocking=True)
        xm = xm.to(device, non_blocking=True)
        if arm.a1_600:
            xf = xf[..., :600]
            xm = xm[..., :600]
        yy = yy.to(device)
        x_in = _resolve_mask_input(model, xf, xm, arm)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            out = model(
                x_in,
                x_full=None if arm.data == "a0" else xf,
                no_grad_target=arm.no_grad_target,
                ema_target=arm.ema_target,
                train_mode=True,
            )
            loss, _meta = compute_losses(
                out,
                yy,
                None if (arm.a1_600 or arm.data == "a0") else xf,
                lambda_cls=hp.lambda_cls,
                lambda_pred=lam_pred,
                lambda_sig=lam_sig,
                lambda_dec=lam_dec,
                cls_cur=arm.cls_cur,
                cls_final=arm.cls_final,
                cls_future=arm.cls_future,
                use_sigreg=arm.use_sigreg,
                sigreg=sigreg,
                dec_no_psd=bool(arm.extra.get("dec_no_psd")),
                dec_no_mubeta=bool(arm.extra.get("dec_no_mubeta")),
                dec_no_time=bool(arm.extra.get("dec_no_time")),
            )
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if arm.ema_target:
            model.update_ema_encoder()
        bs = int(xf.size(0))
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


def _print_test_metrics(
    fold_i: int,
    te_trial: dict,
    te_win: dict,
    n_classes: int,
) -> None:
    f1_key = "f1_macro" if n_classes >= 3 else "f1"
    print(
        f"[fold{fold_i}/test] Acc_paper={te_trial['acc_paper']:.4f}  "
        f"BalAcc_maj={te_trial['balanced_accuracy']:.4f}  "
        f"win_BalAcc={te_win['balanced_accuracy']:.4f}  "
        f"win_{f1_key}={te_win.get(f1_key, float('nan')):.4f}",
        flush=True,
    )
    if n_classes == 2:
        print(format_task_metrics(f"fold{fold_i}/test_window", te_win), flush=True)
    else:
        print(format_three_metrics(f"fold{fold_i}/test_window", te_win), flush=True)


def _log_line(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def build_model_for_arm(arm: ArmSpec, hp: SharedTrainHP, n_times: int, n_outputs: int):
    return MaskFutureDualExpert(
        n_chans=hp.n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        embed_dim=hp.embed_dim,
        drop_prob=hp.drop_prob,
        pred_dropout=hp.pred_dropout,
        use_predictor=arm.use_predictor,
        use_expert_future=arm.use_expert_future,
        use_gate=arm.use_gate,
        use_decoder=arm.use_decoder,
        predictor_temporal=arm.predictor_temporal,
        use_spectral_decoder=arm.use_spectral_decoder,
        gate_entropy=arm.gate_entropy,
        mask_learnable=arm.mask_learnable,
        fixed_alpha=arm.fixed_alpha,
        ema_momentum=hp.ema_momentum,
    )


def _load_xy(arm: ArmSpec, hp: SharedTrainHP):
    if arm.data == "a0":
        raw = load_arrays(hp.data_tag_a0)
        if "X_full" in raw:
            x_full = to_bct(raw["X_full"])
        else:
            x_full = to_bct(raw["X"])
        x_mask = x_full
        n_times = int(x_full.shape[-1])
        if n_times != hp.n_times_a0:
            raise RuntimeError(f"A0 期望 T={hp.n_times_a0}，得到 {x_full.shape}")
    else:
        raw = load_arrays(hp.data_tag_pf)
        if "X_full" in raw:
            x_full = to_bct(raw["X_full"])
            if "X_mask" in raw:
                x_mask = to_bct(raw["X_mask"])
            else:
                x_mask = x_full.copy()
                x_mask[..., -400:] = 0
        else:
            x_full = to_bct(raw["X"])
            if x_full.shape[-1] != 1000 and not arm.a1_600:
                raise RuntimeError(
                    f"pf 数据末维应为 1000，得到 {x_full.shape}；请先跑新预处理臂"
                )
            x_mask = x_full.copy()
            if x_full.shape[-1] >= 1000:
                x_mask[..., -400:] = 0
        n_times = 600 if arm.a1_600 else int(x_full.shape[-1])

    y = np.asarray(raw["y_three"], dtype=np.int64)
    subjects = np.asarray(raw["subjects"])
    trial_id = np.asarray(raw["trial_id"], dtype=np.int64)
    return x_full, x_mask, y, subjects, trial_id, n_times, raw.get("meta") or {}


def run_pf_kfold(
    arm: ArmSpec,
    *,
    hp: SharedTrainHP | None = None,
    max_folds: int = 0,
    out_root: Path | None = None,
) -> dict:
    assert_default_map()
    hp = _loader_hp(hp or SHARED)
    x_full, x_mask, y, subjects, trial_id, n_times, data_meta = _load_xy(arm, hp)
    # Three 协议固定 3 类头（0/1/2）；与 Acc_paper baselines three 一致
    n_outputs = max(3, int(y.max()) + 1)
    n_classes = 3
    y_counts = summarize_labels(y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (out_root or (OUT_ROOT / OUT_ROOT_TAG)) / f"{stamp}_{arm.arm_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    _log_line(
        log_path,
        f"start arm={arm.arm_id} device={device} n_classes={n_classes} "
        f"y_counts={y_counts} pf_protocol_version={data_meta.get('protocol_version')} "
        f"no_rest={data_meta.get('no_rest')}",
    )
    print(
        f"[{arm.arm_id}] labels y_three={y_counts} "
        f"meta_ver={data_meta.get('protocol_version')} no_rest={data_meta.get('no_rest')}",
        flush=True,
    )

    if n_times >= 1000 and not arm.a1_600:
        probe = build_model_for_arm(arm, hp, n_times, n_outputs)
        ratio = assert_future_perturbation(
            probe.encoder,
            i_vis=probe.i_vis,
            i_fut=probe.i_fut,
            n_chans=hp.n_chans,
            n_times=n_times,
        )
        msg = f"[{arm.arm_id}] §3.2.1 future-perturb ratio={ratio:.3f}"
        print(msg, flush=True)
        _log_line(log_path, msg)
        del probe

    fold_rows = []
    max_folds_n = hp.n_folds if max_folds <= 0 else min(max_folds, hp.n_folds)

    for info in iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    ):
        fold_i = int(info["fold"])
        if fold_i >= max_folds_n:
            break
        tr_idx = np.where(info["masks"]["train"])[0]
        va_idx = np.where(info["masks"]["val"])[0]
        te_idx = np.where(info["masks"]["test"])[0]

        print(
            f"\n======== [{arm.arm_id}] fold {fold_i} ========\n"
            f"  train={info['train_subjects']}\n"
            f"  val  ={info['val_subjects']}\n"
            f"  test ={info['test_subjects']}\n"
            f"  n_win={len(tr_idx)}/{len(va_idx)}/{len(te_idx)}",
            flush=True,
        )

        y_tr = y[tr_idx]
        ds_tr = WinDS(x_full, x_mask, y, trial_id, subjects, tr_idx)
        ds_va = WinDS(x_full, x_mask, y, trial_id, subjects, va_idx)
        ds_te = WinDS(x_full, x_mask, y, trial_id, subjects, te_idx)
        dl_kw = dict(
            num_workers=hp.num_workers,
            pin_memory=hp.pin_memory,
            persistent_workers=hp.num_workers > 0,
        )
        if hp.num_workers > 0:
            dl_kw["prefetch_factor"] = hp.prefetch_factor
        dl_tr = DataLoader(
            ds_tr,
            batch_size=hp.batch_train,
            sampler=_balanced_sampler(y_tr),
            **dl_kw,
        )
        dl_va = DataLoader(ds_va, batch_size=hp.batch_eval, shuffle=False, **dl_kw)
        dl_te = DataLoader(ds_te, batch_size=hp.batch_eval, shuffle=False, **dl_kw)

        model = build_model_for_arm(arm, hp, n_times, n_outputs).to(device)
        if arm.ema_target:
            model.init_ema_encoder()
        opt = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )
        use_amp = hp.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        sigreg = (
            SIGReg(num_slices=hp.sigreg_slices).to(device) if arm.use_sigreg else None
        )

        lam_pred = hp.lambda_pred if arm.lambda_pred is None else float(arm.lambda_pred)
        lam_sig = hp.lambda_sig if arm.lambda_sig is None else float(arm.lambda_sig)
        lam_dec = hp.lambda_dec if arm.lambda_dec is None else float(arm.lambda_dec)

        best_score = -1.0
        best_state = None
        best_ep = 0
        best_val_bal = -1.0
        best_val_trial_metrics: dict | None = None
        best_val_window_metrics: dict | None = None
        bad = 0
        ep = 0
        f1_key = "f1_macro" if n_classes >= 3 else "f1"

        for ep in range(1, hp.max_epochs + 1):
            tr_loss = _run_train_epoch(
                model,
                dl_tr,
                opt,
                scaler,
                device,
                arm,
                hp,
                sigreg,
                lam_pred=lam_pred,
                lam_sig=lam_sig,
                lam_dec=lam_dec,
            )
            val_trial, val_win = _eval_split(
                model, dl_va, device, arm, n_classes=n_classes
            )
            score = float(val_trial["acc_paper"])
            bal = float(val_trial["balanced_accuracy"])
            win_f1 = float(val_win.get(f1_key, 0.0))
            line = (
                f"fold{fold_i} ep {ep:03d}  tr={tr_loss:.4f}  "
                f"val_AccPaper={score:.4f}  val_BalAccMaj={bal:.4f}  "
                f"win_BalAcc={float(val_win['balanced_accuracy']):.4f}  "
                f"win_{f1_key}={win_f1:.4f}"
            )
            print(line, flush=True)
            _log_line(log_path, line)

            if score > best_score:
                best_score = score
                best_ep = ep
                best_val_bal = bal
                best_val_trial_metrics = val_trial
                best_val_window_metrics = val_win
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad = 0
                fold_dir = run_dir / f"fold{fold_i}"
                fold_dir.mkdir(exist_ok=True)
                torch.save(
                    {
                        "arm": arm.arm_id,
                        "fold": fold_i,
                        "protocol": hp.protocol,
                        "early_stop": "acc_paper",
                        "model": best_state,
                        "epoch": ep,
                        "val_trial_metrics": val_trial,
                        "val_window_metrics": val_win,
                        "hparams": asdict(hp),
                    },
                    fold_dir / "best.pt",
                )
            else:
                bad += 1
                if bad >= hp.patience:
                    print(f"  early stop @ ep {ep}", flush=True)
                    _log_line(log_path, f"fold{fold_i} early stop @ ep {ep}")
                    break

        assert best_state is not None
        model.load_state_dict(best_state, strict=False)
        te_trial, te_win = _eval_split(
            model, dl_te, device, arm, n_classes=n_classes
        )
        _print_test_metrics(fold_i, te_trial, te_win, n_classes)

        fold_dir = run_dir / f"fold{fold_i}"
        fold_dir.mkdir(exist_ok=True)
        row = {
            "fold": fold_i,
            "best_val_acc_paper": best_score,
            "best_val_balacc_maj": best_val_bal,
            "best_epoch": best_ep,
            "stopped_epoch": ep,
            "best_val_trial_metrics": best_val_trial_metrics,
            "best_val_window_metrics": best_val_window_metrics,
            "test_trial_metrics": te_trial,
            "test_window_metrics": te_win,
            "test_acc_paper": float(te_trial["acc_paper"]),
            "n_test_trials": int(te_trial["n_trials"]),
        }
        (fold_dir / "metrics.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        fold_rows.append(row)

    val_ap = [r["best_val_acc_paper"] for r in fold_rows]
    test_ap = [float(r["test_trial_metrics"]["acc_paper"]) for r in fold_rows]
    test_bm = [float(r["test_trial_metrics"]["balanced_accuracy"]) for r in fold_rows]
    test_wbal = [float(r["test_window_metrics"]["balanced_accuracy"]) for r in fold_rows]
    f1_key = "f1_macro" if n_classes >= 3 else "f1"
    test_wf1 = [float(r["test_window_metrics"][f1_key]) for r in fold_rows]
    test_tm_f1 = [
        float(r["test_trial_metrics"].get(f1_key, r["test_trial_metrics"].get("f1", 0)))
        for r in fold_rows
    ]

    summary = {
        "task": "three_kfold_accpaper",
        "arm": arm.arm_id,
        "note": arm.note,
        "device": str(device),
        "protocol": hp.protocol,
        "no_rap": True,
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
        "test_window_f1_mean": _mean_std(test_wf1)[0],
        "test_window_f1_std": _mean_std(test_wf1)[1],
        "test_f1_macro_maj_mean": _mean_std(test_tm_f1)[0],
        "test_f1_macro_maj_std": _mean_std(test_tm_f1)[1],
        "mean": _mean_std(test_ap)[0],
        "std": _mean_std(test_ap)[1],
        "folds": fold_rows,
        "run_dir": str(run_dir),
        "max_folds": max_folds_n,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    md_lines = [
        f"# {arm.arm_id} · Acc_paper 五折",
        "",
        f"- run_dir: `{run_dir}`",
        f"- Test Acc_paper: `{summary['test_acc_paper_mean']:.4f} ± {summary['test_acc_paper_std']:.4f}`",
        f"- Test BalAcc_maj: `{summary['test_balacc_maj_mean']:.4f} ± {summary['test_balacc_maj_std']:.4f}`",
        f"- Test win F1: `{summary['test_window_f1_mean']:.4f} ± {summary['test_window_f1_std']:.4f}`",
        "",
        *three_fold_md_lines(fold_rows),
    ]
    (run_dir / "fold_detail.md").write_text("\n".join(md_lines), encoding="utf-8")

    done = (
        f"[{arm.arm_id}] done val_AccPaper={summary['val_acc_paper_mean']:.4f} "
        f"test_AccPaper={summary['test_acc_paper_mean']:.4f} "
        f"test_BalAcc={summary['test_balacc_maj_mean']:.4f} "
        f"test_win_F1={summary['test_window_f1_mean']:.4f}"
    )
    print(done, flush=True)
    _log_line(log_path, done)
    return summary
