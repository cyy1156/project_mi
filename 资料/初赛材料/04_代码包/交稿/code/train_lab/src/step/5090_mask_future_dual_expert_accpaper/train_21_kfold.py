"""方案 21 · Acc_paper 五折（F_mi / A2_pt / J1）。"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from _paths import OUT_ROOT, PRE
from arms_registry import ArmSpec
from inwin_jepa import sample_block_starts
from scheme21_data import (
    MI080_TOTAL_PTS,
    crop_pf_mi080,
    filter_indices_by_t0,
    scheme21_mask_future_pts,
)
from shared_hparams import OUT_ROOT_TAG, SHARED, SharedTrainHP
from sigreg import SIGReg

from train_kfold import (  # noqa: E402
    WinDS,
    _balanced_sampler,
    _collate_win,
    _loader_hp,
    _log_line,
    _mean_std,
    _print_test_metrics,
    _resolve_mask_input,
    _run_train_epoch,
    build_model_for_arm,
)
from train_kfold import _load_xy as _load_xy_base  # noqa: E402
from train_kfold import _eval_split as _eval_split_base  # noqa: E402

sys.path.insert(0, str(PRE))
from src.common.steps.split_subjects import iter_subject_kfold  # noqa: E402

from feat_index import assert_future_perturbation  # noqa: E402
from losses import compute_losses  # noqa: E402
from md_fold_detail import three_fold_md_lines  # noqa: E402
from data_io import summarize_labels  # noqa: E402


S1_MAX_EPOCHS = 30
S1_PATIENCE = 5


def _is_scheme21(arm: ArmSpec) -> bool:
    return bool(arm.extra.get("scheme21"))


def _apply_scheme21_xy(
    x_full: np.ndarray,
    x_mask: np.ndarray,
    arm: ArmSpec,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    n_times = int(arm.extra.get("n_times", x_full.shape[-1]))
    mask_pts = scheme21_mask_future_pts(arm.extra)
    if arm.extra.get("pf_mi080"):
        x_full = crop_pf_mi080(x_full)
        x_mask = crop_pf_mi080(x_mask)
        n_times = MI080_TOTAL_PTS
    elif n_times != x_full.shape[-1]:
        n_times = int(x_full.shape[-1])
    return x_full, x_mask, n_times, mask_pts


def _filter_fold_indices(
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    t0_sec: np.ndarray | None,
    arm: ArmSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0_max = arm.extra.get("t0_max")
    if t0_max is None:
        return tr_idx, va_idx, te_idx
    return (
        filter_indices_by_t0(tr_idx, t0_sec, t0_max=float(t0_max)),
        filter_indices_by_t0(va_idx, t0_sec, t0_max=float(t0_max)),
        filter_indices_by_t0(te_idx, t0_sec, t0_max=float(t0_max)),
    )


@torch.no_grad()
def _eval_split_p_cur(
    model,
    loader: DataLoader,
    device: torch.device,
    arm: ArmSpec,
    *,
    n_classes: int,
) -> tuple[dict, dict]:
    """方案 21：分类决策仅用 p_cur。"""
    model.eval()
    ys, ps, subs, tids = [], [], [], []
    for xf, xm, y, tid, subj, t0 in loader:
        xf = xf.to(device, non_blocking=True)
        xm = xm.to(device, non_blocking=True)
        if arm.extra.get("pf_mi080"):
            xf = xf[..., :MI080_TOTAL_PTS]
            xm = xm[..., :MI080_TOTAL_PTS]
        x_in = _resolve_mask_input(model, xf, xm, arm)
        out = model(
            x_in,
            x_full=None,
            t0_sec=t0.to(device),
            train_mode=False,
        )
        pred = out["p_cur"].argmax(dim=-1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
        subs.extend(str(s) for s in subj)
        tids.append(tid.numpy())

    yt = np.concatenate(ys).astype(np.int64)
    yp = np.concatenate(ps).astype(np.int64)
    from trial_metrics import aggregate_windows_to_trials
    from metrics import jsonify_metrics, three_class_metrics

    sub_arr = np.asarray(subs, dtype=object)
    tid_arr = np.concatenate(tids).astype(np.int64)
    trial = aggregate_windows_to_trials(yt, yp, sub_arr, tid_arr, n_classes=n_classes)
    win_m = jsonify_metrics(three_class_metrics(yt, yp))
    return trial["metrics"], win_m


def _run_train_epoch_21(
    model,
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
    lam_cls: float,
    cls_cur: bool,
    inwin: bool = False,
) -> float:
    model.train()
    total, n = 0.0, 0
    for xf, xm, yy, _tid, _subj, t0 in loader:
        xf = xf.to(device, non_blocking=True)
        xm = xm.to(device, non_blocking=True)
        if arm.extra.get("pf_mi080"):
            xf = xf[..., :MI080_TOTAL_PTS]
            xm = xm[..., :MI080_TOTAL_PTS]
        yy = yy.to(device)
        x_in = _resolve_mask_input(model, xf, xm, arm)
        block_starts = None
        if inwin:
            block_starts = sample_block_starts(
                xf.size(0), device=xf.device
            )
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            out = model(
                x_in,
                x_full=None if arm.data == "a0" else xf,
                t0_sec=t0.to(device),
                no_grad_target=True,
                train_mode=True,
                inwin_block_starts=block_starts,
            )
            loss, _meta = compute_losses(
                out,
                yy,
                xf,
                lambda_cls=lam_cls,
                lambda_pred=lam_pred,
                lambda_sig=lam_sig,
                lambda_dec=0.0,
                cls_cur=cls_cur,
                cls_final=False,
                cls_future=False,
                use_sigreg=arm.use_sigreg and lam_sig > 0,
                sigreg=sigreg,
            )
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        bs = int(xf.size(0))
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def _eval_l_pred(
    model,
    loader: DataLoader,
    device: torch.device,
    arm: ArmSpec,
    *,
    inwin: bool = False,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for xf, xm, _yy, _tid, _subj, t0 in loader:
        xf = xf.to(device, non_blocking=True)
        xm = xm.to(device, non_blocking=True)
        if arm.extra.get("pf_mi080"):
            xf = xf[..., :MI080_TOTAL_PTS]
            xm = xm[..., :MI080_TOTAL_PTS]
        x_in = _resolve_mask_input(model, xf, xm, arm)
        block_starts = None
        if inwin:
            block_starts = sample_block_starts(xf.size(0), device=xf.device)
        out = model(
            x_in,
            x_full=xf,
            train_mode=True,
            inwin_block_starts=block_starts,
        )
        if "z_pre_future_seq" in out and "z_target_future_seq" in out:
            l = F.mse_loss(
                out["z_pre_future_seq"],
                out["z_target_future_seq"].detach(),
            )
        elif "z_pre_future" in out and "z_target_future" in out:
            l = F.mse_loss(
                out["z_pre_future"],
                out["z_target_future"].detach(),
            )
        else:
            continue
        bs = xf.size(0)
        total += float(l.item()) * bs
        n += bs
    return total / max(n, 1)


def _pretrain_stage(
    model,
    dl_tr,
    dl_va,
    device,
    arm,
    hp,
    sigreg,
    *,
    lam_pred: float,
    lam_sig: float,
    inwin: bool,
    log_path: Path,
    fold_i: int,
) -> dict:
    params = list(model.encoder.parameters())
    if model.predictor is not None:
        params += list(model.predictor.parameters())
    for p in model.expert_cur.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(params, lr=hp.lr, weight_decay=hp.weight_decay)
    use_amp = hp.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_l = float("inf")
    best_state = None
    bad = 0
    ep = 0
    for ep in range(1, S1_MAX_EPOCHS + 1):
        tr = _run_train_epoch_21(
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
            lam_cls=0.0,
            cls_cur=False,
            inwin=inwin,
        )
        vl = _eval_l_pred(model, dl_va, device, arm, inwin=inwin)
        line = f"fold{fold_i} s1 ep {ep:03d}  tr={tr:.4f}  val_Lpred={vl:.4f}"
        print(line, flush=True)
        _log_line(log_path, line)
        if vl < best_l - 1e-5:
            best_l = vl
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= S1_PATIENCE:
                print(f"  s1 early stop @ ep {ep}", flush=True)
                break
    assert best_state is not None
    return {"state": best_state, "best_val_lpred": best_l, "epochs": ep}


def _finetune_stage(
    model,
    dl_tr,
    dl_va,
    dl_te,
    device,
    arm,
    hp,
    *,
    init_state: dict | None,
    log_path: Path,
    fold_i: int,
    fold_dir: Path,
) -> dict:
    if init_state is not None:
        model.load_state_dict(init_state, strict=False)
    if model.predictor is not None:
        for p in model.predictor.parameters():
            p.requires_grad = False
    for p in model.expert_cur.parameters():
        p.requires_grad = True
    for p in model.encoder.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=hp.lr,
        weight_decay=hp.weight_decay,
    )
    use_amp = hp.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1.0
    best_state = None
    best_ep = 0
    best_val_bal = -1.0
    best_val_trial_metrics = None
    best_val_window_metrics = None
    bad = 0
    ep = 0
    for ep in range(1, hp.max_epochs + 1):
        tr_loss = _run_train_epoch_21(
            model,
            dl_tr,
            opt,
            scaler,
            device,
            arm,
            hp,
            None,
            lam_pred=0.0,
            lam_sig=0.0,
            lam_cls=hp.lambda_cls,
            cls_cur=True,
            inwin=False,
        )
        val_trial, val_win = _eval_split_p_cur(
            model, dl_va, device, arm, n_classes=3
        )
        score = float(val_trial["acc_paper"])
        bal = float(val_trial["balanced_accuracy"])
        line = (
            f"fold{fold_i} s2 ep {ep:03d}  tr={tr_loss:.4f}  "
            f"val_AccPaper={score:.4f}  val_BalAccMaj={bal:.4f}"
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
            torch.save(
                {
                    "arm": arm.arm_id,
                    "fold": fold_i,
                    "stage": "finetune",
                    "model": best_state,
                    "epoch": ep,
                    "val_trial_metrics": val_trial,
                },
                fold_dir / "best.pt",
            )
        else:
            bad += 1
            if bad >= hp.patience:
                print(f"  s2 early stop @ ep {ep}", flush=True)
                break

    assert best_state is not None
    model.load_state_dict(best_state, strict=False)
    te_trial, te_win = _eval_split_p_cur(model, dl_te, device, arm, n_classes=3)
    _print_test_metrics(fold_i, te_trial, te_win, 3)
    return {
        "best_val_acc_paper": best_score,
        "best_val_balacc_maj": best_val_bal,
        "best_epoch": best_ep,
        "stopped_epoch": ep,
        "best_val_trial_metrics": best_val_trial_metrics,
        "best_val_window_metrics": best_val_window_metrics,
        "test_trial_metrics": te_trial,
        "test_window_metrics": te_win,
        "test_acc_paper": float(te_trial["acc_paper"]),
    }


def run_21_kfold(
    arm: ArmSpec,
    *,
    hp: SharedTrainHP | None = None,
    max_folds: int = 0,
    out_root: Path | None = None,
) -> dict:
    if not _is_scheme21(arm):
        raise ValueError(f"非 scheme21 臂: {arm.arm_id}")

    hp = _loader_hp(hp or SHARED)
    x_full, x_mask, y, subjects, trial_id, t0_sec, _nt, mask_future_pts, data_meta = (
        _load_xy_base(arm, hp)
    )
    x_full, x_mask, n_times, mask_future_pts = _apply_scheme21_xy(x_full, x_mask, arm)

    n_outputs = max(3, int(y.max()) + 1)
    y_counts = summarize_labels(y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (out_root or (OUT_ROOT / OUT_ROOT_TAG)) / f"{stamp}_{arm.arm_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    _log_line(
        log_path,
        f"start scheme21 arm={arm.arm_id} n_times={n_times} "
        f"t0_max={arm.extra.get('t0_max')} two_stage={arm.extra.get('two_stage')} "
        f"inwin={arm.extra.get('inwin_jepa')}",
    )
    print(
        f"[{arm.arm_id}] scheme21 n_times={n_times} y={y_counts} "
        f"n_total={len(y)}",
        flush=True,
    )

    if arm.use_predictor and n_times >= 800 and not arm.extra.get("inwin_jepa"):
        probe = build_model_for_arm(arm, hp, n_times, n_outputs)
        fut_pts = 200 if n_times == 800 else 400
        ratio = assert_future_perturbation(
            probe.encoder,
            i_vis=probe.i_vis,
            i_fut=probe.i_fut,
            n_chans=hp.n_chans,
            n_times=n_times,
            future_pts=fut_pts,
        )
        print(f"[{arm.arm_id}] future-perturb ratio={ratio:.3f}", flush=True)
        del probe

    fold_rows = []
    max_folds_n = hp.n_folds if max_folds <= 0 else min(max_folds, hp.n_folds)
    two_stage = bool(arm.extra.get("two_stage"))
    inwin = bool(arm.extra.get("inwin_jepa"))
    eval_fn = _eval_split_p_cur if arm.extra.get("eval_p_cur") else _eval_split_base

    for info in iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    ):
        fold_i = int(info["fold"])
        if fold_i >= max_folds_n:
            break
        tr_idx, va_idx, te_idx = _filter_fold_indices(
            np.where(info["masks"]["train"])[0],
            np.where(info["masks"]["val"])[0],
            np.where(info["masks"]["test"])[0],
            t0_sec,
            arm,
        )
        print(
            f"\n======== [{arm.arm_id}] fold {fold_i} ========\n"
            f"  n_win={len(tr_idx)}/{len(va_idx)}/{len(te_idx)}",
            flush=True,
        )

        ds_kw = dict(mask_future_pts=mask_future_pts, t0_sec=t0_sec)
        ds_tr = WinDS(x_full, x_mask, y, trial_id, subjects, tr_idx, **ds_kw)
        ds_va = WinDS(x_full, x_mask, y, trial_id, subjects, va_idx, **ds_kw)
        ds_te = WinDS(x_full, x_mask, y, trial_id, subjects, te_idx, **ds_kw)
        dl_kw = dict(
            num_workers=hp.num_workers,
            pin_memory=hp.pin_memory,
            persistent_workers=hp.num_workers > 0,
            collate_fn=_collate_win,
        )
        if hp.num_workers > 0:
            dl_kw["prefetch_factor"] = hp.prefetch_factor
        batch_train = int(arm.extra.get("batch_train", hp.batch_train))
        dl_tr = DataLoader(
            ds_tr,
            batch_size=batch_train,
            sampler=_balanced_sampler(y[tr_idx]),
            **dl_kw,
        )
        dl_va = DataLoader(ds_va, batch_size=hp.batch_eval, shuffle=False, **dl_kw)
        dl_te = DataLoader(ds_te, batch_size=hp.batch_eval, shuffle=False, **dl_kw)

        fold_dir = run_dir / f"fold{fold_i}"
        fold_dir.mkdir(exist_ok=True)

        if two_stage:
            model = build_model_for_arm(arm, hp, n_times, n_outputs).to(device)
            sigreg = (
                SIGReg(num_slices=hp.sigreg_slices).to(device)
                if arm.use_sigreg
                else None
            )
            lam_pred = (
                hp.lambda_pred if arm.lambda_pred is None else float(arm.lambda_pred)
            )
            lam_sig = hp.lambda_sig if arm.lambda_sig is None else float(arm.lambda_sig)
            s1 = _pretrain_stage(
                model,
                dl_tr,
                dl_va,
                device,
                arm,
                hp,
                sigreg,
                lam_pred=lam_pred,
                lam_sig=lam_sig,
                inwin=False,
                log_path=log_path,
                fold_i=fold_i,
            )
            (fold_dir / "pretrain_s1.json").write_text(
                json.dumps(
                    {
                        "best_val_lpred": s1["best_val_lpred"],
                        "epochs": s1["epochs"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            model = build_model_for_arm(arm, hp, n_times, n_outputs).to(device)
            row = _finetune_stage(
                model,
                dl_tr,
                dl_va,
                dl_te,
                device,
                arm,
                hp,
                init_state=s1["state"],
                log_path=log_path,
                fold_i=fold_i,
                fold_dir=fold_dir,
            )
            row["fold"] = fold_i
            row["s1_best_val_lpred"] = s1["best_val_lpred"]
        else:
            model = build_model_for_arm(arm, hp, n_times, n_outputs).to(device)
            sigreg = (
                SIGReg(num_slices=hp.sigreg_slices).to(device)
                if arm.use_sigreg
                else None
            )
            lam_pred = (
                hp.lambda_pred if arm.lambda_pred is None else float(arm.lambda_pred)
            )
            lam_sig = hp.lambda_sig if arm.lambda_sig is None else float(arm.lambda_sig)
            opt = torch.optim.Adam(
                model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
            )
            use_amp = hp.use_amp and device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            best_score = -1.0
            best_state = None
            best_ep = 0
            best_val_bal = -1.0
            best_val_trial_metrics = None
            best_val_window_metrics = None
            bad = 0
            ep = 0
            cls_cur = not arm.use_predictor or arm.use_predictor
            for ep in range(1, hp.max_epochs + 1):
                tr_loss = _run_train_epoch_21(
                    model,
                    dl_tr,
                    opt,
                    scaler,
                    device,
                    arm,
                    hp,
                    sigreg,
                    lam_pred=lam_pred if arm.use_predictor else 0.0,
                    lam_sig=lam_sig,
                    lam_cls=hp.lambda_cls,
                    cls_cur=True,
                    inwin=inwin,
                )
                val_trial, val_win = eval_fn(
                    model, dl_va, device, arm, n_classes=3
                )
                score = float(val_trial["acc_paper"])
                bal = float(val_trial["balanced_accuracy"])
                line = (
                    f"fold{fold_i} ep {ep:03d}  tr={tr_loss:.4f}  "
                    f"val_AccPaper={score:.4f}  val_BalAccMaj={bal:.4f}"
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
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    bad = 0
                    torch.save(
                        {"arm": arm.arm_id, "fold": fold_i, "model": best_state},
                        fold_dir / "best.pt",
                    )
                else:
                    bad += 1
                    if bad >= hp.patience:
                        break
            assert best_state is not None
            model.load_state_dict(best_state, strict=False)
            te_trial, te_win = eval_fn(model, dl_te, device, arm, n_classes=3)
            _print_test_metrics(fold_i, te_trial, te_win, 3)
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
            }

        (fold_dir / "metrics.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        fold_rows.append(row)

    val_ap = [float(r["best_val_acc_paper"]) for r in fold_rows]
    test_ap = [float(r["test_trial_metrics"]["acc_paper"]) for r in fold_rows]
    test_bm = [float(r["test_trial_metrics"]["balanced_accuracy"]) for r in fold_rows]
    test_wbal = [float(r["test_window_metrics"]["balanced_accuracy"]) for r in fold_rows]
    f1_key = "f1_macro"
    test_wf1 = [float(r["test_window_metrics"][f1_key]) for r in fold_rows]
    test_tm_f1 = [
        float(r["test_trial_metrics"].get(f1_key, r["test_trial_metrics"].get("f1", 0)))
        for r in fold_rows
    ]

    summary = {
        "task": "scheme21_three_kfold_accpaper",
        "arm": arm.arm_id,
        "note": arm.note,
        "scheme21": True,
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
        "n_train_windows_meta": int(len(y)),
        "data_meta": data_meta,
        "max_folds": max_folds_n,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    md = [
        f"# {arm.arm_id} · scheme21",
        "",
        f"Test Acc_paper: `{summary['test_acc_paper_mean']:.4f} ± {summary['test_acc_paper_std']:.4f}`",
        f"Test BalAcc_maj: `{summary['test_balacc_maj_mean']:.4f} ± {summary['test_balacc_maj_std']:.4f}`",
        f"Test win F1: `{summary['test_window_f1_mean']:.4f} ± {summary['test_window_f1_std']:.4f}`",
        "",
        *three_fold_md_lines(fold_rows),
    ]
    (run_dir / "fold_detail.md").write_text("\n".join(md), encoding="utf-8")
    print(
        f"[{arm.arm_id}] done test_AccPaper={summary['test_acc_paper_mean']:.4f}",
        flush=True,
    )
    return summary
