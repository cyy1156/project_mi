"""方案 26 · 注入 R1/R2/R3 训练配方到 5090 task_runner。"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PKG24 = HERE.parent / "5090_baselines_openbmi_3s_hop100_accpaper"
for p in (str(PKG24), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import shared_hparams as sh24  # noqa: E402  5090 包
import task_runner as tr  # noqa: E402
from recipe_train import (  # noqa: E402
    finalize_swa,
    is_improved,
    make_criterion,
    make_optimizer,
    make_scheduler,
    run_epoch_recipe,
    setup_swa,
)
from s26_hparams import OUT_ROOT_TAG, RecipeTrainHP, recipe_for  # noqa: E402

_PATCHED = False


def install_recipe(arm: str) -> RecipeTrainHP:
    """幂等：替换 OUT_ROOT_TAG、SHARED、train_one_fold、run_epoch。"""
    global _PATCHED
    hp = recipe_for(arm)
    sh24.OUT_ROOT_TAG = OUT_ROOT_TAG
    sh24.SHARED = hp  # type: ignore[misc]
    tr.OUT_ROOT_TAG = OUT_ROOT_TAG
    tr.SHARED = hp  # task_runner 在 import 时绑定了 SHARED，须同步

    if _PATCHED:
        return hp

    _orig_train_one_fold = tr.train_one_fold

    def train_one_fold(*args, **kwargs):
        hp_local: RecipeTrainHP = kwargs.get("hp") or args[5]
        if not isinstance(hp_local, RecipeTrainHP):
            return _orig_train_one_fold(*args, **kwargs)
        return _train_one_fold_recipe(*args, **kwargs)

    tr.train_one_fold = train_one_fold  # type: ignore[assignment]
    tr.run_epoch = _run_epoch_wrapper  # type: ignore[assignment]
    _PATCHED = True
    return hp


def _run_epoch_wrapper(model, loader, criterion, optimizer, device, train, **kw):
    hp = sh24.SHARED
    if isinstance(hp, RecipeTrainHP):
        return run_epoch_recipe(
            model, loader, criterion, optimizer, device, train, hp, **kw
        )
    return tr.run_epoch(model, loader, criterion, optimizer, device, train, **kw)


def _train_one_fold_recipe(
    fold_info,
    X,
    y,
    subjects,
    trial_ids,
    device,
    hp: RecipeTrainHP,
    out_dir: Path,
    *,
    model_name: str,
    build_model,
    input_kind: str,
    n_outputs: int,
    ckpt_name: str,
    stage_tag: str,
    x_path: str | None = None,
    src_box=None,
) -> dict:
    """复制 5090 train_one_fold 训练段，替换 optimizer/scheduler/early-stop/SWA。"""
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    n_classes = n_outputs

    n_tr = int(masks["train"].sum())
    n_va = int(masks["val"].sum())
    n_te = int(masks["test"].sum())
    if n_tr == 0 or n_va == 0 or n_te == 0:
        raise RuntimeError(f"fold{fold} empty split")

    g = torch.Generator()
    g.manual_seed(hp.seed + fold)
    tr_idx = tr._indices_from_mask(masks["train"])
    va_idx = tr._indices_from_mask(masks["val"])
    t0_all = tr.compute_window_t0_sec(trial_ids)
    if float(getattr(hp, "t0_filter_max", 0.0) or 0.0) > 0.0:
        t0_cap = float(hp.t0_filter_max)
        tr_idx = tr_idx[t0_all[tr_idx] <= t0_cap + 1e-6]
    y_tr = y[tr_idx]
    w_tr = None
    if float(getattr(hp, "t0_weight_alpha", 0.0) or 0.0) > 0.0:
        w_tr = tr.t0_train_weight(t0_all[tr_idx], float(hp.t0_weight_alpha))

    cache_paths: list[Path] = []
    pack_tr_path = fold_dir / "_cache_train_X.npy"
    pack_va_path = fold_dir / "_cache_val_X.npy"
    reuse_packs = pack_tr_path.is_file() and pack_va_path.is_file()
    if X is None and not reuse_packs:
        X = __import__("numpy").load(x_path, mmap_mode="r")
        if src_box is not None and not src_box:
            src_box.append(X)
    n_times = int(X.shape[-1]) if X is not None else int(
        __import__("numpy").load(pack_tr_path, mmap_mode="r").shape[-1]
    )

    print(
        f"\n======== [{stage_tag}] [{model_name}] fold {fold} · recipe ========\n"
        f"  n_win={n_tr}/{n_va}/{n_te}"
    )

    try:
        if reuse_packs:
            train_ds = tr.PackedArrayDataset(y_tr, x_path=pack_tr_path, sample_weights=w_tr)
            cache_paths.extend([pack_tr_path, pack_va_path])
            X = None
        else:
            pack_tr_path = tr.materialize_time_pack(X, tr_idx, pack_tr_path)
            pack_va_path = tr.materialize_time_pack(X, va_idx, pack_va_path)
            cache_paths.extend([pack_tr_path, pack_va_path])
            train_ds = tr.PackedArrayDataset(y_tr, x_path=pack_tr_path, sample_weights=w_tr)
            X = None

        train_loader = tr.make_loader(
            train_ds,
            batch_size=hp.batch_train,
            sampler=tr.make_balanced_sampler(y_tr, n_classes=n_classes, generator=g),
            **tr._loader_kwargs(hp),
        )
        tr.seed_everything(hp.seed + fold, cudnn_benchmark=hp.cudnn_benchmark, deterministic=hp.deterministic)
        model = build_model(8, n_times, n_outputs, hp.drop_prob).to(device)
        criterion = make_criterion(hp)
        optimizer = make_optimizer(model, hp)
        scheduler = make_scheduler(optimizer, hp, total_epochs=hp.max_epochs)
        swa_model, swa_meta = setup_swa(model, hp, hp.max_epochs)
        scaler = None
        if hp.use_amp and device.type == "cuda":
            try:
                scaler = torch.amp.GradScaler("cuda")
            except (TypeError, AttributeError):
                scaler = torch.cuda.amp.GradScaler()

        best_score, best_state, best_ep = -1.0, None, 0
        best_val_loss = float("inf")
        best_val_bal_maj = -1.0
        best_val_trial_metrics = None
        bad, ep = 0, 0

        for ep in range(1, hp.max_epochs + 1):
            tr_loss = run_epoch_recipe(
                model, train_loader, criterion, optimizer, device, True, hp,
                non_blocking=hp.non_blocking, use_amp=hp.use_amp, scaler=scaler,
            )
            scheduler.step()
            if swa_model is not None and swa_meta is not None:
                start_ep, _ = swa_meta
                if ep >= start_ep:
                    swa_model.update_parameters(model)

            val_trial, val_win, va_loss = tr._eval_split(
                model, X, y, subjects, trial_ids, masks["val"], device, hp,
                input_kind=input_kind, n_classes=n_classes,
                packed_path=pack_va_path, packed_indices=va_idx, x_path=None,
            )
            score = float(val_trial["acc_paper"])
            bal_maj = float(val_trial["balanced_accuracy"])
            print(
                f"fold{fold} ep {ep:03d} tr={tr_loss:.4f} va={va_loss:.4f} "
                f"val_AccPaper={score:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if is_improved(score, best_score, hp):
                best_score = score
                best_ep = ep
                best_val_loss = va_loss
                best_val_bal_maj = bal_maj
                best_val_trial_metrics = val_trial
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
                torch.save(
                    {
                        "stage": stage_tag,
                        "fold": fold,
                        "model_name": model_name,
                        "n_outputs": n_outputs,
                        "protocol": hp.protocol,
                        "model": best_state,
                        "epoch": ep,
                        "val_trial_metrics": val_trial,
                        "hparams": asdict(hp),
                    },
                    fold_dir / ckpt_name,
                )
            else:
                bad += 1
                if bad >= hp.patience:
                    print(f"  early stop @ ep {ep}")
                    break

        swa_state = finalize_swa(swa_model, model, train_loader, device)
        if swa_state is not None:
            best_state = swa_state
            print(f"  SWA weights applied fold{fold}")

        assert best_state is not None
        model.load_state_dict(best_state)
        te_trial, te_win, _ = tr._eval_split(
            model, X, y, subjects, trial_ids, masks["test"], device, hp,
            input_kind=input_kind, n_classes=n_classes, x_path=x_path,
        )
        print(f"[fold{fold}/test] Acc_paper={te_trial['acc_paper']:.4f}")
        return {
            "fold": fold,
            "best_val_acc_paper": float(best_score),
            "best_val_balacc_maj": float(best_val_bal_maj),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_ep),
            "stopped_epoch": int(ep),
            "best_val_trial_metrics": best_val_trial_metrics,
            "test_trial_metrics": te_trial,
            "test_window_metrics": te_win,
        }
    finally:
        import gc
        gc.collect()
        if not bool(getattr(hp, "keep_fold_packs", False)):
            for pth in cache_paths:
                try:
                    if pth.is_file():
                        pth.unlink()
                except OSError:
                    pass
