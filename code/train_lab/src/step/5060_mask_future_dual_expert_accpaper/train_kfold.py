"""Acc_paper 五折训练环（A0 自写 500pt / pf1000 自写模型臂）。"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from _paths import OUT_ROOT, PRE
from arms_registry import ArmSpec
from data_io import load_arrays, to_bct
from feat_index import assert_default_map, assert_future_perturbation
from losses import compute_losses
from model import MaskFutureDualExpert
from shared_hparams import OUT_ROOT_TAG, SHARED, SharedTrainHP
from sigreg import SIGReg

sys.path.insert(0, str(PRE))
from src.common.steps.split_subjects import iter_subject_kfold  # noqa: E402


class WinDS(Dataset):
    def __init__(self, x_full, x_mask, y, trial_id, idx):
        self.xf = x_full
        self.xm = x_mask
        self.y = y
        self.tid = trial_id
        self.idx = np.asarray(idx, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        xf = torch.from_numpy(np.asarray(self.xf[j], dtype=np.float32))
        xm = torch.from_numpy(np.asarray(self.xm[j], dtype=np.float32))
        if xf.ndim == 2 and xf.shape[0] in (500, 600, 1000):
            # (T,C) → (C,T)
            xf = xf.T.contiguous()
            xm = xm.T.contiguous()
        y = int(self.y[j])
        tid = int(self.tid[j])
        return xf, xm, y, tid


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


def _resolve_mask_input(
    model: MaskFutureDualExpert,
    xf: torch.Tensor,
    xm: torch.Tensor,
    arm: ArmSpec,
) -> torch.Tensor:
    """训练/评估统一：B8 用 make_mask(x_full)；B9 评估可偷看 full。"""
    if arm.leak_eval_full:
        return xf
    if arm.mask_learnable:
        return model.make_mask(xf)
    return xm


@torch.no_grad()
def _eval_acc_paper(model, loader, device, arm: ArmSpec):
    model.eval()
    by_trial: dict[int, list[tuple[int, int]]] = {}
    for xf, xm, y, tid in loader:
        xf = xf.to(device, non_blocking=True)
        xm = xm.to(device, non_blocking=True)
        y = y.to(device)
        tid_np = tid.numpy()
        if arm.a1_600:
            xf = xf[..., :600]
            xm = xm[..., :600]
        x_in = _resolve_mask_input(model, xf, xm, arm)
        out = model(x_in, x_full=None, train_mode=False)
        pred = out["p_final"].argmax(dim=-1).cpu().numpy()
        yt = y.cpu().numpy()
        for t, p, yy in zip(tid_np, pred, yt):
            by_trial.setdefault(int(t), []).append((int(p), int(yy)))
    ok = 0
    n = 0
    for pairs in by_trial.values():
        n += 1
        correct = sum(1 for p, yy in pairs if p == yy)
        if (correct / max(len(pairs), 1)) > 0.5:
            ok += 1
    return ok / max(n, 1), n


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
        x_mask = x_full  # A0 无 future mask
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
    return x_full, x_mask, y, subjects, trial_id, n_times


def run_pf_kfold(
    arm: ArmSpec,
    *,
    hp: SharedTrainHP | None = None,
    max_folds: int = 0,
    out_root: Path | None = None,
) -> dict:
    assert_default_map()
    hp = hp or SHARED
    x_full, x_mask, y, subjects, trial_id, n_times = _load_xy(arm, hp)
    n_outputs = int(y.max() + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (out_root or (OUT_ROOT / OUT_ROOT_TAG)) / f"{stamp}_{arm.arm_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # §3.2.1 future 扰动（仅 1000pt 臂；失败停训）
    if n_times >= 1000 and not arm.a1_600:
        probe = build_model_for_arm(arm, hp, n_times, n_outputs)
        ratio = assert_future_perturbation(
            probe.encoder,
            i_vis=probe.i_vis,
            i_fut=probe.i_fut,
            n_chans=hp.n_chans,
            n_times=n_times,
        )
        print(f"[{arm.arm_id}] §3.2.1 future-perturb ratio={ratio:.3f}", flush=True)
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

        y_tr = y[tr_idx]
        ds_tr = WinDS(x_full, x_mask, y, trial_id, tr_idx)
        ds_va = WinDS(x_full, x_mask, y, trial_id, va_idx)
        ds_te = WinDS(x_full, x_mask, y, trial_id, te_idx)
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
        scaler = torch.cuda.amp.GradScaler(
            enabled=hp.use_amp and device.type == "cuda"
        )
        sigreg = (
            SIGReg(num_slices=hp.sigreg_slices).to(device) if arm.use_sigreg else None
        )

        lam_pred = hp.lambda_pred if arm.lambda_pred is None else float(arm.lambda_pred)
        lam_sig = hp.lambda_sig if arm.lambda_sig is None else float(arm.lambda_sig)
        lam_dec = hp.lambda_dec if arm.lambda_dec is None else float(arm.lambda_dec)

        best = -1.0
        best_state = None
        bad = 0
        ep = -1
        for ep in range(hp.max_epochs):
            model.train()
            for xf, xm, yy, _tid in dl_tr:
                xf = xf.to(device, non_blocking=True)
                xm = xm.to(device, non_blocking=True)
                if arm.a1_600:
                    xf = xf[..., :600]
                    xm = xm[..., :600]
                yy = yy.to(device)
                x_in = _resolve_mask_input(model, xf, xm, arm)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
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

            va_acc, _ = _eval_acc_paper(model, dl_va, device, arm)
            if va_acc > best:
                best = va_acc
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
            if bad >= hp.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
        te_acc, n_tr = _eval_acc_paper(model, dl_te, device, arm)
        fold_dir = run_dir / f"fold{fold_i}"
        fold_dir.mkdir(exist_ok=True)
        torch.save(best_state, fold_dir / "best.pt")
        row = {
            "fold": fold_i,
            "val_acc_paper_best": best,
            "test_acc_paper": te_acc,
            "n_test_trials": n_tr,
            "epochs_ran": ep + 1,
        }
        (fold_dir / "metrics.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        fold_rows.append(row)
        print(f"[{arm.arm_id}] fold{fold_i} test_acc_paper={te_acc:.4f}", flush=True)

    scores = [r["test_acc_paper"] for r in fold_rows]
    summary = {
        "arm": arm.arm_id,
        "note": arm.note,
        "device": str(device),
        "mean": float(np.mean(scores)) if scores else None,
        "std": float(np.std(scores)) if scores else None,
        "folds": fold_rows,
        "hp": asdict(hp),
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
