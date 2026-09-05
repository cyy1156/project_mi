"""方案 23 · Acc_paper 五折（机制验证臂）。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from _paths import OUT_ROOT, PRE, load_official
from anchors import eval_indices, prepare_fold_arrays
from arms_registry import (
    CALIBRATION_ARM,
    CALIBRATION_HI,
    CALIBRATION_LO,
    ArmSpec,
    run_arm_folder_name,
)
from data_io import load_arrays, to_bct
from feat_index import assert_default_map, assert_future_perturbation, segment_indices_for_geom
from geometry import get_geom, make_masked, slice_pf1000
from losses import compute_losses
from md_fold_detail import three_fold_md_lines
from model import MaskFutureDualExpert
from shared_hparams import OUT_ROOT_TAG, SHARED, SharedTrainHP
from sigreg import SIGReg

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

STEP = Path(__file__).resolve().parent.parent
if str(STEP) not in sys.path:
    sys.path.append(str(STEP))
from metrics import (  # noqa: E402
    format_three_metrics,
    jsonify_metrics,
    three_class_metrics,
)

_trial_metrics = load_official("trial_metrics")
aggregate_windows_to_trials = _trial_metrics.aggregate_windows_to_trials

sys.path.insert(0, str(PRE))
from src.common.steps.split_subjects import iter_subject_kfold  # noqa: E402


class WinDS23(Dataset):
    def __init__(
        self,
        x_full,
        y,
        trial_id,
        subjects,
        idx,
        *,
        geom_id: str,
        oracle: bool,
        t0_sec=None,
    ):
        self.xf = x_full
        self.y = y
        self.tid = trial_id
        self.subjects = subjects
        self.idx = np.asarray(idx, dtype=np.int64)
        self.geom_id = geom_id
        self.oracle = bool(oracle)
        self.t0_sec = t0_sec
        self.geom = get_geom(geom_id)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        raw = np.array(self.xf[j], dtype=np.float32, copy=True)
        if raw.ndim == 3:
            raw = raw.squeeze(0)
        if raw.shape[0] in (500, 600, 750, 875, 1000) and raw.shape[1] == 8:
            raw = raw.T
        xf_np = slice_pf1000(raw, self.geom_id)
        xf = torch.from_numpy(np.asarray(xf_np, dtype=np.float32, copy=True))
        xm = make_masked(xf, self.geom_id)
        x_in = xf if self.oracle else xm
        y = int(self.y[j])
        tid = int(self.tid[j])
        subj = str(self.subjects[j])
        t0 = float(self.t0_sec[j]) if self.t0_sec is not None else 0.0
        return x_in, xf, y, tid, subj, t0


def _collate_win(batch):
    x_in = torch.stack([b[0] for b in batch], dim=0)
    xf = torch.stack([b[1] for b in batch], dim=0)
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    tid = torch.tensor([b[3] for b in batch], dtype=torch.long)
    subj = [b[4] for b in batch]
    t0 = torch.tensor([b[5] for b in batch], dtype=torch.float32)
    return x_in, xf, y, tid, subj, t0


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


def _mean_std(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))


def build_model_for_arm(arm: ArmSpec, hp: SharedTrainHP, n_outputs: int):
    g = get_geom(arm.geom_id)
    return MaskFutureDualExpert(
        n_chans=hp.n_chans,
        n_times=g.total_pts,
        n_outputs=n_outputs,
        embed_dim=int(hp.embed_dim),
        drop_prob=hp.drop_prob,
        pred_dropout=hp.pred_dropout,
        use_predictor=arm.use_predictor,
        use_expert_future=arm.use_expert_future,
        use_gate=arm.use_gate,
        use_decoder=arm.use_decoder,
        predictor_identity=arm.predictor_identity,
        expert_double=arm.expert_double,
        vis_pts=g.vis_pts,
    )


def _pred_key(arm: ArmSpec) -> str:
    if arm.use_gate or arm.use_expert_future:
        return "p_final"
    return "p_cur"


@torch.no_grad()
def _eval_split(
    model,
    loader,
    device,
    arm: ArmSpec,
    *,
    n_classes: int,
) -> tuple[dict, dict, dict[str, float]]:
    model.eval()
    yt_all: list[int] = []
    yp_all: list[int] = []
    sub_all: list[str] = []
    tid_all: list[int] = []
    pk = _pred_key(arm)
    for x_in, xf, y, tid, subj, t0 in loader:
        x_in = x_in.to(device, non_blocking=True)
        xf = xf.to(device, non_blocking=True)
        t0 = t0.to(device, non_blocking=True)
        out = model(
            x_in,
            x_full=xf if arm.use_predictor else None,
            t0_sec=t0,
            train_mode=False,
        )
        pred = out[pk].argmax(dim=-1).cpu().numpy().astype(int)
        yt = y.numpy().astype(int)
        yt_all.extend(yt.tolist())
        yp_all.extend(pred.tolist())
        tid_all.extend(tid.numpy().astype(np.int64).tolist())
        sub_all.extend(str(s) for s in subj)

    yt_arr = np.asarray(yt_all, dtype=np.int64)
    yp_arr = np.asarray(yp_all, dtype=np.int64)
    sub_arr = np.asarray(sub_all, dtype=object)
    tid_arr = np.asarray(tid_all, dtype=np.int64)
    trial = aggregate_windows_to_trials(
        yt_arr, yp_arr, sub_arr, tid_arr, n_classes=n_classes
    )
    win_m = jsonify_metrics(three_class_metrics(yt_arr, yp_arr))
    per_sub: dict[str, float] = {}
    for sub in np.unique(sub_arr):
        m = sub_arr == sub
        if m.sum() == 0:
            continue
        tr = aggregate_windows_to_trials(
            yt_arr[m], yp_arr[m], sub_arr[m], tid_arr[m], n_classes=n_classes
        )
        per_sub[str(sub)] = float(tr["metrics"]["acc_paper"])
    return trial["metrics"], win_m, per_sub


def _load_data(hp: SharedTrainHP):
    raw = load_arrays(hp.data_tag_pf)
    x_full = to_bct(raw["X_full"] if "X_full" in raw else raw["X"])
    y = np.asarray(raw["y_three"], dtype=np.int64)
    subjects = np.asarray(raw["subjects"])
    trial_id = np.asarray(raw["trial_id"], dtype=np.int64)
    t0_sec = raw.get("t0_sec")
    if t0_sec is None:
        raise RuntimeError("scheme23 需要 openbmi_t0_sec.npy")
    t0_sec = np.asarray(t0_sec, dtype=np.float32)
    return x_full, y, subjects, trial_id, t0_sec, raw.get("meta", {})


def run_23_kfold(
    arm: ArmSpec,
    *,
    hp: SharedTrainHP | None = None,
    max_folds: int = 0,
    out_root: Path | None = None,
    resume_dir: Path | None = None,
    log_path: Path | None = None,
) -> dict:
    assert_default_map()
    hp = hp or SHARED
    x_full, y, subjects, trial_id, t0_sec, data_meta = _load_data(hp)
    n_outputs = max(3, int(y.max()) + 1)
    n_classes = 3
    g = get_geom(arm.geom_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if g.future_pts > 0 and not arm.oracle:
        probe = build_model_for_arm(arm, hp, n_outputs)
        i_vis, i_fut = segment_indices_for_geom(arm.geom_id, probe.t_prime)
        if i_fut:
            ratio = assert_future_perturbation(
                probe.encoder,
                i_vis=i_vis,
                i_fut=i_fut,
                n_chans=hp.n_chans,
                n_times=g.total_pts,
                future_pts=g.future_pts,
            )
            print(f"[{arm.arm_id}] §3.2.1 future-perturb ratio={ratio:.3f}", flush=True)
        del probe

    if resume_dir is not None:
        run_dir = Path(resume_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (out_root or (OUT_ROOT / OUT_ROOT_TAG)) / run_arm_folder_name(arm, stamp)
        run_dir.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_path = run_dir / "run.log"

    def _log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    _log(
        f"start scheme23 arm={arm.arm_id} geom={arm.geom_id} oracle={arm.oracle} "
        f"n={len(y)} device={device}"
    )

    fold_rows: list[dict] = []
    per_subject_all: dict[str, list[float]] = defaultdict(list)
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
            for sub, acc in row.get("per_subject_test_acc_paper", {}).items():
                per_subject_all[str(sub)].append(float(acc))
            continue

        tr_idx = np.where(info["masks"]["train"])[0]
        va_idx = np.where(info["masks"]["val"])[0]
        te_idx = np.where(info["masks"]["test"])[0]
        va_idx = eval_indices(va_idx)
        te_idx = eval_indices(te_idx)

        xf_f, y_f, tid_f, t0_f, subj_f, tr_idx = prepare_fold_arrays(
            arm.arm_id, x_full, y, trial_id, t0_sec, subjects, tr_idx
        )

        ds_kw = dict(geom_id=arm.geom_id, oracle=arm.oracle, t0_sec=t0_f)
        ds_tr = WinDS23(xf_f, y_f, tid_f, subj_f, tr_idx, **ds_kw)
        ds_va = WinDS23(xf_f, y_f, tid_f, subj_f, va_idx, **ds_kw)
        ds_te = WinDS23(xf_f, y_f, tid_f, subj_f, te_idx, **ds_kw)

        dl_kw = dict(
            num_workers=hp.num_workers,
            pin_memory=hp.pin_memory,
            collate_fn=_collate_win,
        )
        dl_tr = DataLoader(
            ds_tr,
            batch_size=hp.batch_train,
            sampler=_balanced_sampler(y_f[tr_idx]),
            **dl_kw,
        )
        dl_va = DataLoader(ds_va, batch_size=hp.batch_eval, shuffle=False, **dl_kw)
        dl_te = DataLoader(ds_te, batch_size=hp.batch_eval, shuffle=False, **dl_kw)

        model = build_model_for_arm(arm, hp, n_outputs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        use_amp = hp.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        sigreg = SIGReg(num_slices=hp.sigreg_slices).to(device) if arm.use_sigreg else None
        lam_pred = hp.lambda_pred if arm.lambda_pred is None else float(arm.lambda_pred)
        lam_sig = hp.lambda_sig if arm.lambda_sig is None else float(arm.lambda_sig)
        lam_dec = hp.lambda_dec if arm.lambda_dec is None else float(arm.lambda_dec)

        best_score = -1.0
        best_state = None
        best_ep = 0
        best_val_trial = None
        bad = 0
        ep = 0
        for ep in range(hp.max_epochs):
            model.train()
            tr_loss = 0.0
            n_b = 0
            for x_in, xf, yy, _tid, _subj, t0 in dl_tr:
                x_in = x_in.to(device, non_blocking=True)
                xf = xf.to(device, non_blocking=True)
                t0 = t0.to(device, non_blocking=True)
                yy = yy.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(
                        x_in,
                        x_full=xf if arm.use_predictor else None,
                        t0_sec=t0,
                        y=yy,
                        train_mode=True,
                    )
                    loss, _ = compute_losses(
                        out,
                        yy,
                        xf if arm.use_predictor else None,
                        lambda_cls=hp.lambda_cls,
                        lambda_pred=lam_pred,
                        lambda_sig=lam_sig,
                        lambda_dec=lam_dec,
                        cls_cur=arm.cls_cur,
                        cls_final=arm.cls_final,
                        cls_future=arm.cls_future,
                        use_sigreg=arm.use_sigreg,
                        sigreg=sigreg,
                    )
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tr_loss += float(loss.detach())
                n_b += 1

            va_trial, va_win, _ = _eval_split(model, dl_va, device, arm, n_classes=n_classes)
            va_acc = float(va_trial["acc_paper"])
            _log(
                f"fold{fold_i} ep {ep+1:03d}  tr={tr_loss/max(n_b,1):.4f}  "
                f"val_AccPaper={va_acc:.4f}  val_BalAccMaj={va_trial['balanced_accuracy']:.4f}"
            )
            if va_acc > best_score:
                best_score = va_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_ep = ep + 1
                best_val_trial = va_trial
                bad = 0
            else:
                bad += 1
            if bad >= hp.patience:
                _log(f"fold{fold_i} early stop @ ep {ep+1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
        te_trial, te_win, te_per_sub = _eval_split(model, dl_te, device, arm, n_classes=n_classes)
        for k, v in te_per_sub.items():
            per_subject_all[k].append(v)

        fold_dir = run_dir / f"fold{fold_i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, fold_dir / "best.pt")
        row = {
            "fold": fold_i,
            "best_val_acc_paper": float(best_score),
            "best_val_balacc_maj": float(best_val_trial["balanced_accuracy"]) if best_val_trial else float("nan"),
            "best_epoch": int(best_ep),
            "stopped_epoch": int(ep + 1),
            "best_val_trial_metrics": best_val_trial,
            "test_trial_metrics": te_trial,
            "test_window_metrics": te_win,
            "test_acc_paper": float(te_trial["acc_paper"]),
            "per_subject_test_acc_paper": te_per_sub,
            "n_train_windows": int(len(tr_idx)),
        }
        (fold_dir / "metrics.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        fold_rows.append(row)
        _log(
            f"fold{fold_i}/test Acc_paper={te_trial['acc_paper']:.4f}  "
            f"BalAcc={te_trial['balanced_accuracy']:.4f}"
        )

    test_ap = [float(r["test_acc_paper"]) for r in fold_rows]
    per_sub_mean = {
        k: float(np.mean(v)) for k, v in per_subject_all.items() if v
    }
    summary = {
        "task": "scheme23_three_kfold_accpaper",
        "arm": arm.arm_id,
        "note": arm.note,
        "scheme23": True,
        "geom_id": arm.geom_id,
        "oracle": arm.oracle,
        "test_acc_paper_mean": _mean_std(test_ap)[0],
        "test_acc_paper_std": _mean_std(test_ap)[1],
        "mean": _mean_std(test_ap)[0],
        "std": _mean_std(test_ap)[1],
        "folds": fold_rows,
        "run_dir": str(run_dir),
        "per_subject_test_acc_paper": per_sub_mean,
        "data_meta": data_meta,
        "hparams": asdict(hp),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = three_fold_md_lines(fold_rows)
    (run_dir / "fold_detail.md").write_text("\n".join(lines), encoding="utf-8")

    if arm.arm_id == CALIBRATION_ARM and max_folds_n >= hp.n_folds:
        m = summary["test_acc_paper_mean"]
        if not (CALIBRATION_LO <= m <= CALIBRATION_HI):
            _log(
                f"WARN calibration gate FAIL: {m:.4f} not in "
                f"[{CALIBRATION_LO}, {CALIBRATION_HI}]"
            )

    return summary
