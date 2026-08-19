"""Acc_paper 五折训练环（A0 自写 500pt / pf1000 自写模型臂）。

评估与落盘对齐 5060_baselines Acc_paper 环：
  - 试次级 Acc_paper / BalAcc_maj / F1-macro / 混淆矩阵等（trial_metrics）
  - 窗级 three_class_metrics（含 cm）
  - fold{k}/metrics.json + 臂级 summary.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from _paths import HERE, OUT_ROOT, PRE, STEP, load_official

# 本包必须优先于 OFFICIAL，否则 shared_hparams 会被 baselines 同名模块遮蔽
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

from arms_registry import ArmSpec
from data_io import load_arrays, to_bct
from feat_index import assert_default_map, assert_future_perturbation
from losses import compute_losses
from model import MaskFutureDualExpert
from shared_hparams import OUT_ROOT_TAG, SHARED, SharedTrainHP
from sigreg import SIGReg

# metrics：step/metrics.py；trial_metrics：官方 Acc_paper 包（load_official 避免 path 冲突）
if str(STEP) not in sys.path:
    sys.path.append(str(STEP))
from metrics import (  # noqa: E402
    binary_task_metrics,
    format_task_metrics,
    format_three_metrics,
    jsonify_metrics,
    three_class_metrics,
)

_trial_metrics = load_official("trial_metrics")
aggregate_windows_to_trials = _trial_metrics.aggregate_windows_to_trials

sys.path.insert(0, str(PRE))
from src.common.steps.split_subjects import iter_subject_kfold  # noqa: E402


class WinDS(Dataset):
    """x_mask 可为 None：则从 x_full 现场零填 future400（省一份 ~5.6GB mmap 页缓存）。"""

    def __init__(
        self,
        x_full,
        x_mask,
        y,
        trial_id,
        subjects,
        idx,
        *,
        mask_future_pts: int = 0,
        t0_sec=None,
    ):
        self.xf = x_full
        self.xm = x_mask
        self.y = y
        self.tid = trial_id
        self.subjects = subjects
        self.idx = np.asarray(idx, dtype=np.int64)
        self.mask_future_pts = int(mask_future_pts)
        self.t0_sec = t0_sec

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        # copy=True：mmap/只读视图 → 可写缓冲，避免 non-writable tensor 警告
        xf = torch.from_numpy(np.array(self.xf[j], dtype=np.float32, copy=True))
        if self.xm is None:
            xm = xf.clone()
            if self.mask_future_pts > 0:
                xm[..., -self.mask_future_pts :] = 0
        else:
            xm = torch.from_numpy(np.array(self.xm[j], dtype=np.float32, copy=True))
        if xf.ndim == 2 and xf.shape[0] in (500, 600, 1000):
            # (T,C) → (C,T)
            xf = xf.T.contiguous()
            xm = xm.T.contiguous()
        y = int(self.y[j])
        tid = int(self.tid[j])
        subj = str(self.subjects[j])
        t0 = float(self.t0_sec[j]) if self.t0_sec is not None else 0.0
        return xf, xm, y, tid, subj, t0


def _collate_win(batch):
    """默认 collate 无法堆叠 str；subjects 以 list[str] 返回。"""
    xf = torch.stack([b[0] for b in batch], dim=0)
    xm = torch.stack([b[1] for b in batch], dim=0)
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    tid = torch.tensor([b[3] for b in batch], dtype=torch.long)
    subj = [b[4] for b in batch]
    t0 = torch.tensor([b[5] for b in batch], dtype=torch.float32)
    return xf, xm, y, tid, subj, t0


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


def _mean_std(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))


@torch.no_grad()
def _eval_split(
    model,
    loader,
    device,
    arm: ArmSpec,
    *,
    n_classes: int,
) -> tuple[dict, dict]:
    """对齐 baselines `_eval_split`：返回 (trial_metrics, window_metrics)。"""
    model.eval()
    yt_all: list[int] = []
    yp_all: list[int] = []
    sub_all: list[str] = []
    tid_all: list[int] = []
    for batch in loader:
        xf, xm, y, tid, subj, t0 = batch
        xf = xf.to(device, non_blocking=True)
        xm = xm.to(device, non_blocking=True)
        t0 = t0.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if arm.a1_600:
            xf = xf[..., :600]
            xm = xm[..., :600]
        x_in = _resolve_mask_input(model, xf, xm, arm)
        out = model(
            x_in,
            x_full=None,
            t0_sec=t0,
            y=y,
            train_mode=False,
        )
        pred = out["p_final"].argmax(dim=-1).cpu().numpy().astype(int)
        yt = y.cpu().numpy().astype(int)
        tids = tid.cpu().numpy().astype(np.int64)
        subs = [str(s) for s in subj]
        yt_all.extend(yt.tolist())
        yp_all.extend(pred.tolist())
        tid_all.extend(tids.tolist())
        sub_all.extend(subs)

    yt_arr = np.asarray(yt_all, dtype=np.int64)
    yp_arr = np.asarray(yp_all, dtype=np.int64)
    sub_arr = np.asarray(sub_all, dtype=object)
    tid_arr = np.asarray(tid_all, dtype=np.int64)

    trial = aggregate_windows_to_trials(
        yt_arr, yp_arr, sub_arr, tid_arr, n_classes=n_classes
    )
    if n_classes == 2:
        win_m = jsonify_metrics(binary_task_metrics(yt_arr, yp_arr))
    else:
        win_m = jsonify_metrics(three_class_metrics(yt_arr, yp_arr))
    return trial["metrics"], win_m


def build_model_for_arm(arm: ArmSpec, hp: SharedTrainHP, n_times: int, n_outputs: int):
    embed_dim = int(arm.extra.get("embed_dim", hp.embed_dim))
    return MaskFutureDualExpert(
        n_chans=hp.n_chans,
        n_times=n_times,
        n_outputs=n_outputs,
        embed_dim=embed_dim,
        drop_prob=hp.drop_prob,
        pred_dropout=hp.pred_dropout,
        use_predictor=arm.use_predictor,
        use_expert_future=arm.use_expert_future,
        use_gate=arm.use_gate,
        use_decoder=arm.use_decoder,
        predictor_temporal=arm.predictor_temporal,
        use_spectral_decoder=arm.use_spectral_decoder,
        gate_entropy=arm.gate_entropy,
        predictor_query=arm.predictor_query,
        pred_token_seq=arm.pred_token_seq,
        phase_conditioning=arm.phase_conditioning,
        phase_aux=arm.phase_aux,
        expert_attn_pool=arm.expert_attn_pool,
        mask_learnable=arm.mask_learnable,
        fixed_alpha=arm.fixed_alpha,
        ema_momentum=hp.ema_momentum,
    )


def _load_xy(arm: ArmSpec, hp: SharedTrainHP):
    """返回 (x_full, x_mask|None, y, subjects, trial_id, t0_sec|None, n_times, mask_future_pts)。

    pf 臂只 mmap 一份 X_full；X_mask 在 Dataset 里现场零填，避免双路页缓存打满 16GB。
    """
    mask_future_pts = 0
    if arm.data == "a0":
        raw = load_arrays(hp.data_tag_a0)
        if "X_full" in raw:
            x_full = to_bct(raw["X_full"])
        else:
            x_full = to_bct(raw["X"])
        x_mask = x_full  # A0 无 future mask（同引用，不复制）
        n_times = int(x_full.shape[-1])
        if n_times != hp.n_times_a0:
            raise RuntimeError(f"A0 期望 T={hp.n_times_a0}，得到 {x_full.shape}")
    else:
        raw = load_arrays(hp.data_tag_pf)
        if "X_full" in raw:
            x_full = to_bct(raw["X_full"])
        elif "X" in raw:
            x_full = to_bct(raw["X"])
        else:
            raise RuntimeError(f"pf 数据缺少 X_full/X: {raw.get('dir')}")
        # 故意不加载 X_mask mmap（~5.6GB）；训练时现场零填 future
        x_mask = None
        mask_future_pts = 400
        n_times = 600 if arm.a1_600 else int(x_full.shape[-1])
        if n_times != 1000 and not arm.a1_600:
            raise RuntimeError(
                f"pf 数据末维应为 1000，得到 {x_full.shape}；请先跑新预处理臂"
            )

    y = np.asarray(raw["y_three"], dtype=np.int64)
    subjects = np.asarray(raw["subjects"])
    trial_id = np.asarray(raw["trial_id"], dtype=np.int64)
    t0_sec = raw.get("t0_sec")
    if t0_sec is not None:
        t0_sec = np.asarray(t0_sec, dtype=np.float32)
    return x_full, x_mask, y, subjects, trial_id, t0_sec, n_times, mask_future_pts


def run_pf_kfold(
    arm: ArmSpec,
    *,
    hp: SharedTrainHP | None = None,
    max_folds: int = 0,
    out_root: Path | None = None,
    resume_dir: Path | None = None,
) -> dict:
    assert_default_map()
    hp = hp or SHARED
    x_full, x_mask, y, subjects, trial_id, t0_sec, n_times, mask_future_pts = _load_xy(
        arm, hp
    )
    if arm.predictor_query and t0_sec is None:
        raise RuntimeError(
            f"[{arm.arm_id}] T 系列需要 pf1000 的 openbmi_t0_sec.npy；请重跑 preprocess"
        )
    # Three 协议固定 3 类标签空间（0/1/2）；与 baselines three 头一致
    n_outputs = max(3, int(y.max()) + 1)
    n_classes = 3 if n_outputs >= 3 else 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume_dir 不存在: {run_dir}")
        print(f"[{arm.arm_id}] resume {run_dir}", flush=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (out_root or (OUT_ROOT / OUT_ROOT_TAG)) / f"{stamp}_{arm.arm_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{arm.arm_id}] load n={len(y)} T={n_times} "
        f"mask={'onfly' if x_mask is None else 'array'} "
        f"future_pts={mask_future_pts} device={device}",
        flush=True,
    )

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

    fold_rows: list[dict] = []
    max_folds_n = hp.n_folds if max_folds <= 0 else min(max_folds, hp.n_folds)

    for info in iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    ):
        fold_i = int(info["fold"])
        if fold_i >= max_folds_n:
            break
        prev = run_dir / f"fold{fold_i}" / "metrics.json"
        if prev.is_file():
            row = json.loads(prev.read_text(encoding="utf-8"))
            fold_rows.append(row)
            print(
                f"[{arm.arm_id}] fold{fold_i} skip (resume) "
                f"test_AccPaper={float(row.get('test_acc_paper', float('nan'))):.4f}",
                flush=True,
            )
            continue
        tr_idx = np.where(info["masks"]["train"])[0]
        va_idx = np.where(info["masks"]["val"])[0]
        te_idx = np.where(info["masks"]["test"])[0]

        y_tr = y[tr_idx]
        ds_kw = dict(mask_future_pts=mask_future_pts, t0_sec=t0_sec)
        ds_tr = WinDS(x_full, x_mask, y, trial_id, subjects, tr_idx, **ds_kw)
        ds_va = WinDS(x_full, x_mask, y, trial_id, subjects, va_idx, **ds_kw)
        ds_te = WinDS(x_full, x_mask, y, trial_id, subjects, te_idx, **ds_kw)
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
            collate_fn=_collate_win,
            **dl_kw,
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=hp.batch_eval,
            shuffle=False,
            collate_fn=_collate_win,
            **dl_kw,
        )
        dl_te = DataLoader(
            ds_te,
            batch_size=hp.batch_eval,
            shuffle=False,
            collate_fn=_collate_win,
            **dl_kw,
        )

        model = build_model_for_arm(arm, hp, n_times, n_outputs).to(device)
        if arm.ema_target:
            model.init_ema_encoder()
        opt = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=hp.use_amp and device.type == "cuda"
        )
        sigreg = (
            SIGReg(num_slices=hp.sigreg_slices).to(device) if arm.use_sigreg else None
        )

        lam_pred = hp.lambda_pred if arm.lambda_pred is None else float(arm.lambda_pred)
        lam_sig = hp.lambda_sig if arm.lambda_sig is None else float(arm.lambda_sig)
        lam_dec = hp.lambda_dec if arm.lambda_dec is None else float(arm.lambda_dec)
        lam_phase = float(arm.extra.get("lambda_phase", 0.2 if arm.phase_aux else 0.0))

        best_score = -1.0
        best_val_bal_maj = -1.0
        best_val_trial_metrics: dict | None = None
        best_state = None
        best_ep = -1
        bad = 0
        ep = -1
        for ep in range(hp.max_epochs):
            model.train()
            for xf, xm, yy, _tid, _subj, t0 in dl_tr:
                xf = xf.to(device, non_blocking=True)
                xm = xm.to(device, non_blocking=True)
                t0 = t0.to(device, non_blocking=True)
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
                        t0_sec=t0,
                        y=yy,
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
                        lambda_phase=lam_phase,
                    )
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if arm.ema_target:
                    model.update_ema_encoder()

            va_trial, va_win = _eval_split(
                model, dl_va, device, arm, n_classes=n_classes
            )
            va_acc = float(va_trial["acc_paper"])
            bal_maj = float(va_trial["balanced_accuracy"])
            print(
                f"[{arm.arm_id}] fold{fold_i} ep {ep:03d}  "
                f"val_AccPaper={va_acc:.4f}  val_BalAccMaj={bal_maj:.4f}  "
                f"win_BalAcc={float(va_win['balanced_accuracy']):.4f}",
                flush=True,
            )
            if va_acc > best_score:
                best_score = va_acc
                best_val_bal_maj = bal_maj
                best_val_trial_metrics = va_trial
                best_ep = int(ep)
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
            if bad >= hp.patience:
                print(f"[{arm.arm_id}] fold{fold_i} early stop @ ep {ep}", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
        te_trial, te_win = _eval_split(model, dl_te, device, arm, n_classes=n_classes)

        fold_dir = run_dir / f"fold{fold_i}"
        fold_dir.mkdir(exist_ok=True)
        torch.save(best_state, fold_dir / "best.pt")

        row = {
            "fold": fold_i,
            "best_val_acc_paper": float(best_score),
            "best_val_balacc_maj": float(best_val_bal_maj),
            "best_epoch": int(best_ep),
            "stopped_epoch": int(ep),
            "best_val_trial_metrics": best_val_trial_metrics,
            "test_trial_metrics": te_trial,
            "test_window_metrics": te_win,
            # 兼容旧字段
            "val_acc_paper_best": float(best_score),
            "test_acc_paper": float(te_trial["acc_paper"]),
            "n_test_trials": int(te_trial.get("n_trials", 0)),
            "epochs_ran": int(ep) + 1,
        }
        (fold_dir / "metrics.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        fold_rows.append(row)

        print(
            f"[{arm.arm_id}] fold{fold_i}/test Acc_paper={te_trial['acc_paper']:.4f}  "
            f"BalAcc_maj={te_trial['balanced_accuracy']:.4f}  "
            f"win_BalAcc={te_win['balanced_accuracy']:.4f}",
            flush=True,
        )
        if n_classes == 2:
            print(format_task_metrics(f"{arm.arm_id}/fold{fold_i}/test_window", te_win), flush=True)
        else:
            print(
                format_three_metrics(f"{arm.arm_id}/fold{fold_i}/test_window", te_win),
                flush=True,
            )

    val_ap = [float(r["best_val_acc_paper"]) for r in fold_rows]
    test_ap = [float(r["test_trial_metrics"]["acc_paper"]) for r in fold_rows]
    test_bm = [float(r["test_trial_metrics"]["balanced_accuracy"]) for r in fold_rows]
    test_wbal = [float(r["test_window_metrics"]["balanced_accuracy"]) for r in fold_rows]

    summary: dict = {
        "arm": arm.arm_id,
        "note": arm.note,
        "device": str(device),
        "task": "three_kfold_accpaper",
        "protocol": hp.protocol,
        "early_stop": "acc_paper",
        "n_classes": n_classes,
        "val_acc_paper_mean": _mean_std(val_ap)[0],
        "val_acc_paper_std": _mean_std(val_ap)[1],
        "test_acc_paper_mean": _mean_std(test_ap)[0],
        "test_acc_paper_std": _mean_std(test_ap)[1],
        "test_balacc_maj_mean": _mean_std(test_bm)[0],
        "test_balacc_maj_std": _mean_std(test_bm)[1],
        "test_window_balacc_mean": _mean_std(test_wbal)[0],
        "test_window_balacc_std": _mean_std(test_wbal)[1],
        # 兼容旧字段
        "mean": _mean_std(test_ap)[0] if test_ap else None,
        "std": _mean_std(test_ap)[1] if test_ap else None,
        "folds": fold_rows,
        "hp": asdict(hp),
        "hparams": asdict(hp),
        "run_dir": str(run_dir),
        "out_dir": str(run_dir),
    }
    if n_classes == 2:
        w_f1 = [float(r["test_window_metrics"]["f1"]) for r in fold_rows]
        summary["test_window_f1_mean"] = _mean_std(w_f1)[0]
        summary["test_window_f1_std"] = _mean_std(w_f1)[1]
    else:
        t_f1 = [float(r["test_trial_metrics"]["f1_macro"]) for r in fold_rows]
        w_f1m = [float(r["test_window_metrics"]["f1_macro"]) for r in fold_rows]
        summary["test_f1_macro_maj_mean"] = _mean_std(t_f1)[0]
        summary["test_f1_macro_maj_std"] = _mean_std(t_f1)[1]
        summary["test_window_f1_macro_mean"] = _mean_std(w_f1m)[0]
        summary["test_window_f1_macro_std"] = _mean_std(w_f1m)[1]

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(
        f"\n[{arm.arm_id}] Val Acc_paper "
        f"{summary['val_acc_paper_mean']:.4f}±{summary['val_acc_paper_std']:.4f} | "
        f"Test Acc_paper {summary['test_acc_paper_mean']:.4f}±{summary['test_acc_paper_std']:.4f} | "
        f"Test BalAcc_maj {summary['test_balacc_maj_mean']:.4f}±{summary['test_balacc_maj_std']:.4f}",
        flush=True,
    )
    return summary
