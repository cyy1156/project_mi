"""Task + Three 五折：Val Acc_paper 早停；train batch balance；无 RAP；仅 OpenBMI。"""

from __future__ import annotations

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
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
STEP_DIR = HERE.parent
CODE_ROOT = HERE.parents[3]
TRAIN_LAB = CODE_ROOT / "train_lab"
REPO_ROOT = CODE_ROOT.parent
PRE_ROOT = CODE_ROOT / "preprocess_lab"
HOP100 = STEP_DIR / "baselines_2s_hop100"
OLD_BASELINES = STEP_DIR / "baselines_single"

for p in (STEP_DIR, PRE_ROOT, HOP100, OLD_BASELINES):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
_here = str(HERE)
if _here in sys.path:
    sys.path.remove(_here)
sys.path.insert(0, _here)


def _ensure_local_module(name: str) -> None:
    """load_external 可能把 hop100/single 的同名模块留在 sys.modules，须强制用本包版本。"""
    want = (HERE / f"{name}.py").resolve()
    mod = sys.modules.get(name)
    if mod is None:
        return
    got = getattr(mod, "__file__", None)
    if got is None or Path(got).resolve() != want:
        del sys.modules[name]


_ensure_local_module("shared_hparams")
_ensure_local_module("md_fold_detail")
from shared_hparams import (
    OUT_ROOT_TAG,
    SHARED,
    TRAIN_DEVICE_LABEL,
    TRAIN_DEVICE_NOTE,
    SharedTrainHP,
)
from md_fold_detail import task_fold_md_lines, three_fold_md_lines
from trial_metrics import aggregate_windows_to_trials
from data_paths import resolve_data
from metrics import (
    binary_task_metrics,
    format_task_metrics,
    format_three_metrics,
    jsonify_metrics,
    three_class_metrics,
)
from src.common.steps.split_subjects import iter_subject_kfold

# hop100 包内采样器（只读）
from task_sampler import make_balanced_sampler

BuildFn = Callable[..., nn.Module]


class IndexArrayDataset(Dataset):
    """按全局下标从 mmap/数组取窗，避免 X[mask] 整段物化（OpenBMI ~10 GiB）。"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        *,
        input_kind: str,
    ):
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.input_kind = input_kind
        assert len(self.y) == len(X)
        if len(self.indices):
            assert int(self.indices.min()) >= 0
            assert int(self.indices.max()) < len(X)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        # mmap 只读 → copy 成可写 float32，供 from_numpy
        x = np.array(self.X[idx], dtype=np.float32, copy=True)
        if self.input_kind != "feat":
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            assert x.ndim == 2 and x.shape[0] == 8, x.shape
        else:
            assert x.ndim == 2 and x.shape[0] == 8, x.shape
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


class PackedArrayDataset(Dataset):
    """折内已打包的连续数组 X_pack[i] ↔ y_pack[i]（通常 float16 memmap）。"""

    def __init__(self, X_pack: np.ndarray, y_pack: np.ndarray):
        self.X = X_pack
        self.y = np.asarray(y_pack, dtype=np.int64)
        assert len(self.X) == len(self.y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = np.array(self.X[i], dtype=np.float32, copy=True)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)


_GATHER_CHUNK = 1024


def _squeeze_time_windows(block: np.ndarray) -> np.ndarray:
    """(B,1,8,T)|(B,8,T) → (B,8,T)。"""
    block = np.asarray(block)
    if block.ndim == 4 and block.shape[1] == 1:
        return block[:, 0, :, :]
    if block.ndim == 3 and block.shape[1] == 8:
        return block
    raise ValueError(f"unexpected window block shape: {block.shape}")


def materialize_time_pack(
    X_src: np.ndarray,
    indices: np.ndarray,
    out_path: Path,
    *,
    dtype=np.float16,
) -> np.ndarray:
    """
    将全局下标对应的时域窗顺序写入磁盘 memmap，供折内随机采样。
    避免对 10GiB 源文件做整 epoch 随机 IO（16GB 机易抖动）。
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    n = int(len(indices))
    t = int(X_src.shape[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    fp = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dtype, shape=(n, 8, t)
    )
    for s in range(0, n, _GATHER_CHUNK):
        e = min(s + _GATHER_CHUNK, n)
        idxs = indices[s:e]
        block = _squeeze_time_windows(np.array(X_src[idxs]))
        fp[s:e] = block.astype(dtype, copy=False)
    fp.flush()
    return fp


def seed_everything(seed: int, *, cudnn_benchmark: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _cuda_ready() -> bool:
    return torch.cuda.is_available()


def _to_device(x, y, device: torch.device, hp: SharedTrainHP):
    nb = hp.non_blocking and _cuda_ready() and device.type == "cuda"
    return x.to(device, non_blocking=nb), y.to(device, non_blocking=nb)


def _dataloader_kwargs(hp: SharedTrainHP) -> dict:
    nw = int(hp.num_workers)
    kw: dict = {"num_workers": nw}
    if nw > 0:
        kw["pin_memory"] = bool(hp.pin_memory and _cuda_ready())
        if hp.persistent_workers:
            kw["persistent_workers"] = True
    else:
        kw["pin_memory"] = False
    return kw


def _use_amp(hp: SharedTrainHP, device: torch.device) -> bool:
    return bool(hp.use_amp and device.type == "cuda")


def append_md(md_path: Path, text: str, out_root: Path, log_path: Path) -> None:
    records_root = REPO_ROOT / "资料" / "模型训练"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    rel = md_path.relative_to(records_root).as_posix()
    latest = records_root / "五折实验记录_最新.md"
    latest.write_text(
        f"# 最新实验入口\n\n"
        f"本次记录：[`{rel}`](./{rel})\n\n"
        f"权重目录：`{out_root}`\n"
        f"日志：`{log_path}`\n",
        encoding="utf-8",
    )


def log_line(log_path: Path, msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    hp: SharedTrainHP,
) -> tuple[float, np.ndarray, np.ndarray]:
    """单次 forward：同时算 loss 与预测（避免验证扫两遍）。"""
    model.eval()
    use_amp = _use_amp(hp, device)
    total, n = 0.0, 0
    ys, ps = [], []
    for x, y in loader:
        x, y = _to_device(x, y, device, hp)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            if logits.ndim > 2:
                logits = logits.reshape(logits.shape[0], -1)
            loss = criterion(logits, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
        ps.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(y.cpu().numpy())
    return total / max(n, 1), np.concatenate(ys), np.concatenate(ps)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    hp: SharedTrainHP,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train(train)
    use_amp = _use_amp(hp, device)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = _to_device(x, y, device, hp)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                if logits.ndim > 2:
                    logits = logits.reshape(logits.shape[0], -1)
                loss = criterion(logits, y)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / max(n, 1)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def _indices_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)


def _make_ds(X, y, indices: np.ndarray, input_kind: str) -> IndexArrayDataset:
    return IndexArrayDataset(X, y, indices, input_kind=input_kind)


def _eval_split(
    model,
    X,
    y,
    subjects,
    trial_ids,
    mask: np.ndarray,
    device,
    hp: SharedTrainHP,
    *,
    input_kind: str,
    n_classes: int,
    packed_X: np.ndarray | None = None,
    packed_indices: np.ndarray | None = None,
) -> tuple[dict, dict, float]:
    """返回 (trial_metrics, window_metrics, val_loss)。shuffle=False 保证与 subjects/trial_ids 对齐。"""
    if packed_X is not None:
        assert packed_indices is not None
        indices = np.asarray(packed_indices, dtype=np.int64)
        subs, tids = subjects[indices], trial_ids[indices]
        y_pack = y[indices]
        loader = DataLoader(
            PackedArrayDataset(packed_X, y_pack),
            batch_size=hp.batch_eval,
            shuffle=False,
            **_dataloader_kwargs(hp),
        )
    else:
        indices = _indices_from_mask(mask)
        subs, tids = subjects[indices], trial_ids[indices]
        loader = DataLoader(
            _make_ds(X, y, indices, input_kind),
            batch_size=hp.batch_eval,
            shuffle=False,
            **_dataloader_kwargs(hp),
        )
    criterion = nn.CrossEntropyLoss()
    va_loss, yt, yp = eval_epoch(model, loader, criterion, device, hp)
    assert len(yt) == len(subs) == len(tids)
    trial = aggregate_windows_to_trials(yt, yp, subs, tids, n_classes=n_classes)
    if n_classes == 2:
        win_m = jsonify_metrics(binary_task_metrics(yt, yp))
    else:
        win_m = jsonify_metrics(three_class_metrics(yt, yp))
    return trial["metrics"], win_m, float(va_loss)


def train_one_fold(
    fold_info,
    X,
    y,
    subjects,
    trial_ids,
    device,
    hp: SharedTrainHP,
    out_dir: Path,
    *,
    model_name: str,
    build_model: BuildFn,
    input_kind: str,
    n_outputs: int,
    ckpt_name: str,
    stage_tag: str,
) -> dict:
    fold = fold_info["fold"]
    masks = fold_info["masks"]
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    n_classes = n_outputs

    n_tr = int(masks["train"].sum())
    n_va = int(masks["val"].sum())
    n_te = int(masks["test"].sum())
    if n_tr == 0 or n_va == 0 or n_te == 0:
        raise RuntimeError(
            f"[{model_name}] fold{fold} 集合为空: "
            f"train={n_tr} val={n_va} test={n_te}；"
            f"train_subj={fold_info['train_subjects']} "
            f"val_subj={fold_info['val_subjects']} "
            f"test_subj={fold_info['test_subjects']}。"
            "OpenBMI 五折需要足够多被试（建议 ≥10；冒烟至少数人）。"
            "当前若只有 1 人预处理产物，请先跑更多 subject 的 batch。"
        )

    print(
        f"\n======== [{stage_tag}] [{model_name}] fold {fold} ========\n"
        f"  train={fold_info['train_subjects']}\n"
        f"  val  ={fold_info['val_subjects']}\n"
        f"  test ={fold_info['test_subjects']}\n"
        f"  n_win={n_tr}/{n_va}/{n_te}"
    )

    g = torch.Generator()
    g.manual_seed(hp.seed + fold)
    tr_idx = _indices_from_mask(masks["train"])
    y_tr = y[tr_idx]
    cache_paths: list[Path] = []
    X_tr_pack = None
    X_va_pack = None
    try:
        va_idx = _indices_from_mask(masks["val"])
        if input_kind == "feat":
            train_ds: Dataset = _make_ds(X, y, tr_idx, input_kind)
        else:
            pack_tr = fold_dir / "_cache_train_X.npy"
            pack_va = fold_dir / "_cache_val_X.npy"
            print(
                f"  packing train/val windows → float16 "
                f"(n={len(tr_idx)}/{len(va_idx)}) …",
                flush=True,
            )
            X_tr_pack = materialize_time_pack(X, tr_idx, pack_tr)
            X_va_pack = materialize_time_pack(X, va_idx, pack_va)
            cache_paths.extend([pack_tr, pack_va])
            train_ds = PackedArrayDataset(X_tr_pack, y_tr)
            print(
                f"  pack done train={tuple(X_tr_pack.shape)} val={tuple(X_va_pack.shape)}",
                flush=True,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=hp.batch_train,
            sampler=make_balanced_sampler(
                y_tr, n_classes=n_classes, generator=g
            ),
            **_dataloader_kwargs(hp),
        )

        seed_everything(hp.seed + fold, cudnn_benchmark=hp.cudnn_benchmark)
        # time: (N,1,8,T) → T；feat: (N,8,n_band) → n_band
        n_times = int(X.shape[-1])
        model = build_model(8, n_times, n_outputs, hp.drop_prob).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )
        scaler = (
            torch.amp.GradScaler("cuda")
            if _use_amp(hp, device)
            else None
        )

        best_score, best_state, best_ep = -1.0, None, 0
        best_val_loss = float("inf")
        best_val_bal_maj = -1.0
        best_val_trial_metrics: dict | None = None
        bad, ep = 0, 0

        for ep in range(1, hp.max_epochs + 1):
            tr = run_epoch(
                model, train_loader, criterion, optimizer, device, True, hp, scaler
            )
            val_trial, val_win, va_loss = _eval_split(
                model,
                X,
                y,
                subjects,
                trial_ids,
                masks["val"],
                device,
                hp,
                input_kind=input_kind,
                n_classes=n_classes,
                packed_X=X_va_pack if input_kind != "feat" else None,
                packed_indices=va_idx if input_kind != "feat" else None,
            )
            score = float(val_trial["acc_paper"])
            bal_maj = float(val_trial["balanced_accuracy"])
            print(
                f"fold{fold} ep {ep:03d}  tr={tr:.4f}  va={va_loss:.4f}  "
                f"val_AccPaper={score:.4f}  val_BalAccMaj={bal_maj:.4f}  "
                f"win_BalAcc={float(val_win['balanced_accuracy']):.4f}"
            )
            if score > best_score:
                best_score = score
                best_ep = ep
                best_val_loss = va_loss
                best_val_bal_maj = bal_maj
                best_val_trial_metrics = val_trial
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad = 0
                torch.save(
                    {
                        "stage": stage_tag,
                        "fold": fold,
                        "model_name": model_name,
                        "n_outputs": n_outputs,
                        "protocol": hp.protocol,
                        "no_rap": True,
                        "balbatch": True,
                        "early_stop": "acc_paper",
                        "model": best_state,
                        "epoch": ep,
                        "val_trial_metrics": val_trial,
                        "val_window_metrics": val_win,
                        "hparams": asdict(hp),
                    },
                    fold_dir / ckpt_name,
                )
            else:
                bad += 1
                if bad >= hp.patience:
                    print(f"  early stop @ ep {ep}")
                    break

        assert best_state is not None
        model.load_state_dict(best_state)
        te_trial, te_win, _ = _eval_split(
            model,
            X,
            y,
            subjects,
            trial_ids,
            masks["test"],
            device,
            hp,
            input_kind=input_kind,
            n_classes=n_classes,
        )
        if n_classes == 2:
            print(
                f"[fold{fold}/test] Acc_paper={te_trial['acc_paper']:.4f}  "
                f"BalAcc_maj={te_trial['balanced_accuracy']:.4f}  "
                f"win_BalAcc={te_win['balanced_accuracy']:.4f}"
            )
            print(format_task_metrics(f"fold{fold}/test_window", te_win))
        else:
            print(
                f"[fold{fold}/test] Acc_paper={te_trial['acc_paper']:.4f}  "
                f"BalAcc_maj={te_trial['balanced_accuracy']:.4f}  "
                f"win_BalAcc={te_win['balanced_accuracy']:.4f}"
            )
            print(format_three_metrics(f"fold{fold}/test_window", te_win))

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
        # Windows：须先关掉 memmap 句柄才能删缓存文件
        try:
            del train_loader  # type: ignore[name-defined]
        except Exception:
            pass
        try:
            del train_ds  # type: ignore[name-defined]
        except Exception:
            pass
        for pack in (X_tr_pack, X_va_pack):
            if pack is None:
                continue
            try:
                mmap = getattr(pack, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            except Exception:
                pass
        X_tr_pack = None
        X_va_pack = None
        for p in cache_paths:
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass


def run_kfold(
    X,
    y,
    subjects,
    trial_ids,
    device,
    hp: SharedTrainHP,
    out_dir: Path,
    data_tag: str,
    *,
    model_name: str,
    build_model: BuildFn,
    input_kind: str,
    n_outputs: int,
    ckpt_name: str,
    stage_tag: str,
    task_key: str,
    extra_meta: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for info in iter_subject_kfold(
        subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
    ):
        folds.append(
            train_one_fold(
                info,
                X,
                y,
                subjects,
                trial_ids,
                device,
                hp,
                out_dir,
                model_name=model_name,
                build_model=build_model,
                input_kind=input_kind,
                n_outputs=n_outputs,
                ckpt_name=ckpt_name,
                stage_tag=stage_tag,
            )
        )

    def ms(key_trial: str, key_win: str | None = None):
        if key_win is None:
            xs = [float(r["test_trial_metrics"][key_trial]) for r in folds]
        else:
            xs = [float(r["test_window_metrics"][key_win]) for r in folds]
        return _mean_std(xs)

    val_ap = [r["best_val_acc_paper"] for r in folds]
    test_ap = [float(r["test_trial_metrics"]["acc_paper"]) for r in folds]
    test_bm = [float(r["test_trial_metrics"]["balanced_accuracy"]) for r in folds]
    test_wbal = [float(r["test_window_metrics"]["balanced_accuracy"]) for r in folds]

    summary = {
        "task": task_key,
        "model_name": model_name,
        "data_tag": data_tag,
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
        "folds": folds,
        "out_dir": str(out_dir),
    }
    if n_outputs == 2:
        summary["test_window_f1_mean"] = ms("f1", "f1")[0]
        summary["test_window_f1_std"] = ms("f1", "f1")[1]
    else:
        summary["test_f1_macro_maj_mean"] = _mean_std(
            [float(r["test_trial_metrics"]["f1_macro"]) for r in folds]
        )[0]
        summary["test_f1_macro_maj_std"] = _mean_std(
            [float(r["test_trial_metrics"]["f1_macro"]) for r in folds]
        )[1]
        summary["test_window_f1_macro_mean"] = ms("f1_macro", "f1_macro")[0]
        summary["test_window_f1_macro_std"] = ms("f1_macro", "f1_macro")[1]

    if extra_meta:
        summary.update(extra_meta)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(
        f"\n[{task_key} {model_name}] Val Acc_paper "
        f"{summary['val_acc_paper_mean']:.4f}±{summary['val_acc_paper_std']:.4f} | "
        f"Test Acc_paper {summary['test_acc_paper_mean']:.4f}±{summary['test_acc_paper_std']:.4f} | "
        f"Test BalAcc_maj {summary['test_balacc_maj_mean']:.4f}±{summary['test_balacc_maj_std']:.4f}"
    )
    return summary


def run_baseline_main(
    *,
    model_name: str,
    build_model: BuildFn,
    input_kind: str = "time",
    structure_note: str,
    extra_meta: dict | None = None,
    prepare_X: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    import argparse

    p = argparse.ArgumentParser(
        description=f"{model_name} 2s/hop100 Acc_paper 重训（OpenBMI；balbatch）"
    )
    p.add_argument(
        "--data",
        default=SHARED.data_tag,
        choices=("openbmi_2s_hop100",),
        help="仅 OpenBMI hop100；禁用 BCI2a/Stieger 混训",
    )
    p.add_argument("--skip-three", action="store_true")
    p.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help=">0 时只跑前 N 折（冒烟）",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=0,
        help=">0 时覆盖默认 max_epochs（冒烟）",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help=">0 时覆盖默认 patience（冒烟）",
    )
    args = p.parse_args()

    hp = SHARED
    if args.max_epochs > 0 or args.patience > 0:
        hp = replace(
            hp,
            max_epochs=args.max_epochs if args.max_epochs > 0 else hp.max_epochs,
            patience=args.patience if args.patience > 0 else hp.patience,
        )
    seed_everything(hp.seed, cudnn_benchmark=hp.cudnn_benchmark)
    data_tag = args.data
    data_dir, prefix = resolve_data(data_tag)

    # OpenBMI 合并 X ~10 GiB：必须 mmap；标签/被试键可常驻内存
    X = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
    y_task = np.load(data_dir / f"{prefix}_y_task.npy")
    y_three = np.load(data_dir / f"{prefix}_y_three.npy")
    subjects = np.load(data_dir / f"{prefix}_subjects.npy", allow_pickle=True)
    trial_path = data_dir / f"{prefix}_trial_id.npy"
    if not trial_path.is_file():
        raise FileNotFoundError(f"需要 trial_id：{trial_path}")
    trial_ids = np.load(trial_path)
    assert len(X) == len(y_task) == len(y_three) == len(subjects) == len(trial_ids)
    assert int(X.shape[-1]) == hp.n_times_expected, X.shape
    n_subjects = len(set(np.asarray(subjects).tolist()))
    if n_subjects < 3:
        raise RuntimeError(
            f"openbmi_2s_hop100 当前仅 {n_subjects} 名被试（{data_dir}）。"
            "被试独立五折至少需要多名被试；请先跑："
            "python -m src.datasets.openbmi.batch_2s_hop100"
            "（或 --subjects 01,02,... 子集后再训）。"
        )
    if prepare_X is not None:
        from raw_time_openbmi import squeeze_raw_2s as _squeeze_openbmi

        if prepare_X is _squeeze_openbmi or getattr(prepare_X, "__name__", "") == "squeeze_raw_2s":
            cache = data_dir / f"{prefix}_X_squeeze_8x500.npy"
            print(
                f"[load] squeeze_raw_2s mmap cache → {cache} "
                f"(source X{tuple(X.shape)}) …",
                flush=True,
            )
            X = _squeeze_openbmi(X, cache_path=cache)
        else:
            # bandpower 等：分块读 mmap → 产出小特征阵（约几十 MB），勿整表拷贝 raw
            print(f"[load] prepare_X on mmap X{tuple(X.shape)} …", flush=True)
            X = prepare_X(X)
        print(f"[load] prepare_X done → X{tuple(X.shape)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{model_name}_openbmi_2s_hop100_balbatch_accpaper"
    out_root = (
        TRAIN_LAB
        / "out"
        / OUT_ROOT_TAG
        / out_name
        / data_tag
        / f"run_{stamp}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"
    device_meta = {
        "gpu": TRAIN_DEVICE_LABEL,
        "note": TRAIN_DEVICE_NOTE,
        "conda_env": "cyy",
        "pytorch": "2.11.0+cu128",
        "marked_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_root / "DEVICE.json").write_text(
        json.dumps(device_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    records_root = REPO_ROOT / "资料" / "模型训练"
    md_path = records_root / "runs" / f"{stamp}_{out_name}" / f"{out_name}五折实验记录.md"

    # optional smoke: monkey-patch fold iterator via max_folds by wrapping later
    global_max_folds = int(args.max_folds)

    def _iter_folds():
        for i, info in enumerate(
            iter_subject_kfold(
                subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
            )
        ):
            if global_max_folds > 0 and i >= global_max_folds:
                break
            yield info

    # inject fold limiter into run_kfold by temporary patch of iter in closure — cleaner to pass folds
    def run_kfold_limited(**kwargs):
        out_dir = kwargs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        folds = []
        for info in _iter_folds():
            folds.append(
                train_one_fold(
                    info,
                    kwargs["X"],
                    kwargs["y"],
                    kwargs["subjects"],
                    kwargs["trial_ids"],
                    kwargs["device"],
                    kwargs["hp"],
                    out_dir,
                    model_name=kwargs["model_name"],
                    build_model=kwargs["build_model"],
                    input_kind=kwargs["input_kind"],
                    n_outputs=kwargs["n_outputs"],
                    ckpt_name=kwargs["ckpt_name"],
                    stage_tag=kwargs["stage_tag"],
                )
            )
        # rebuild summary like run_kfold
        val_ap = [r["best_val_acc_paper"] for r in folds]
        test_ap = [float(r["test_trial_metrics"]["acc_paper"]) for r in folds]
        test_bm = [float(r["test_trial_metrics"]["balanced_accuracy"]) for r in folds]
        test_wbal = [float(r["test_window_metrics"]["balanced_accuracy"]) for r in folds]
        summary = {
            "task": kwargs["task_key"],
            "model_name": kwargs["model_name"],
            "data_tag": kwargs["data_tag"],
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
            "folds": folds,
            "out_dir": str(out_dir),
            "max_folds": global_max_folds or hp.n_folds,
        }
        if kwargs["n_outputs"] == 2:
            w_f1 = [float(r["test_window_metrics"]["f1"]) for r in folds]
            summary["test_window_f1_mean"] = _mean_std(w_f1)[0]
            summary["test_window_f1_std"] = _mean_std(w_f1)[1]
        else:
            summary["test_f1_macro_maj_mean"] = _mean_std(
                [float(r["test_trial_metrics"]["f1_macro"]) for r in folds]
            )[0]
            summary["test_f1_macro_maj_std"] = _mean_std(
                [float(r["test_trial_metrics"]["f1_macro"]) for r in folds]
            )[1]
            w_f1m = [float(r["test_window_metrics"]["f1_macro"]) for r in folds]
            summary["test_window_f1_macro_mean"] = _mean_std(w_f1m)[0]
            summary["test_window_f1_macro_std"] = _mean_std(w_f1m)[1]
        if kwargs.get("extra_meta"):
            summary.update(kwargs["extra_meta"])
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(
            f"\n[{kwargs['task_key']} {kwargs['model_name']}] Val Acc_paper "
            f"{summary['val_acc_paper_mean']:.4f}±{summary['val_acc_paper_std']:.4f} | "
            f"Test Acc_paper {summary['test_acc_paper_mean']:.4f}±{summary['test_acc_paper_std']:.4f}"
        )
        return summary

    append_md(
        md_path,
        "\n".join(
            [
                f"# 被试独立五折实验记录（{stamp} / {out_name}）",
                "",
                f"- 开始：`{datetime.now().isoformat(timespec='seconds')}`",
                f"- device：`{device}`",
                f"- 训练设备：**{TRAIN_DEVICE_LABEL}**（{TRAIN_DEVICE_NOTE}）",
                f"- data：`{data_dir}`（OpenBMI hop100 · 仅 EEG_MI_train）",
                f"- protocol：`{hp.protocol}` | early_stop=**Acc_paper** | **balbatch** | no_rap",
                f"- 读数口径：`Tw=2s hop=100ms openbmi_sess01+02 subject_key=openbmi:subjNN early_stop=val_acc_paper select=test_acc_paper balbatch patience={hp.patience}`",
                f"- model：`{model_name}` | {structure_note}",
                f"- Train：窗 CE + **batch balance**；Val/Test：整被试块 + trial 聚合 Acc_paper",
                f"- 权重：`{out_root}`",
                f"- shared hp：`{asdict(hp)}`",
                "",
                "---",
                "",
            ]
        ),
        out_root,
        log_path,
    )
    log_line(
        log_path,
        f"start model={out_name} data={data_tag} device={device} "
        f"workers={hp.num_workers} amp={hp.use_amp} cudnn_benchmark={hp.cudnn_benchmark}",
    )

    common = dict(
        X=X,
        subjects=subjects,
        trial_ids=trial_ids,
        device=device,
        hp=hp,
        data_tag=data_tag,
        model_name=out_name,
        build_model=build_model,
        input_kind=input_kind,
        extra_meta=extra_meta,
    )

    sum_task = run_kfold_limited(
        **common,
        y=y_task,
        out_dir=out_root / "task",
        n_outputs=2,
        ckpt_name="best_task.pt",
        stage_tag=f"task2_{out_name}",
        task_key="task_kfold_accpaper",
    )
    log_line(
        log_path,
        f"TASK done val_AccPaper={sum_task['val_acc_paper_mean']:.4f} "
        f"test_AccPaper={sum_task['test_acc_paper_mean']:.4f}",
    )

    sum_three = None
    if not args.skip_three:
        sum_three = run_kfold_limited(
            **common,
            y=y_three,
            out_dir=out_root / "three",
            n_outputs=3,
            ckpt_name="best_three.pt",
            stage_tag=f"three3_{out_name}",
            task_key="three_kfold_accpaper",
        )
        log_line(
            log_path,
            f"THREE done val_AccPaper={sum_three['val_acc_paper_mean']:.4f} "
            f"test_AccPaper={sum_three['test_acc_paper_mean']:.4f}",
        )

    md_tail = [
        "## 最终结论（主报 Acc_paper）",
        "",
        "### Task",
        f"- Val Acc_paper：`{sum_task['val_acc_paper_mean']:.4f} ± {sum_task['val_acc_paper_std']:.4f}`",
        f"- Test Acc_paper：`{sum_task['test_acc_paper_mean']:.4f} ± {sum_task['test_acc_paper_std']:.4f}`",
        f"- Test BalAcc_maj：`{sum_task['test_balacc_maj_mean']:.4f} ± {sum_task['test_balacc_maj_std']:.4f}`",
        f"- Test 窗级 BalAcc（附报）：`{sum_task['test_window_balacc_mean']:.4f} ± {sum_task['test_window_balacc_std']:.4f}`",
        "",
        *task_fold_md_lines(sum_task["folds"]),
    ]
    if sum_three is not None:
        md_tail.extend(
            [
                "### Three",
                f"- Val Acc_paper：`{sum_three['val_acc_paper_mean']:.4f} ± {sum_three['val_acc_paper_std']:.4f}`",
                f"- Test Acc_paper：`{sum_three['test_acc_paper_mean']:.4f} ± {sum_three['test_acc_paper_std']:.4f}`",
                f"- Test BalAcc_maj：`{sum_three['test_balacc_maj_mean']:.4f} ± {sum_three['test_balacc_maj_std']:.4f}`",
                f"- Test 窗级 BalAcc（附报）：`{sum_three['test_window_balacc_mean']:.4f} ± {sum_three['test_window_balacc_std']:.4f}`",
                "",
                *three_fold_md_lines(sum_three["folds"]),
            ]
        )
    else:
        md_tail.extend(["### Three", "- （本次跳过）", ""])

    md_tail.extend(
        [
            "### 共用超参",
            "```json",
            json.dumps(asdict(hp), indent=2),
            "```",
            "",
            f"- 结束：`{datetime.now().isoformat(timespec='seconds')}`",
            "",
        ]
    )
    append_md(md_path, "\n".join(md_tail), out_root, log_path)

    meta = {
        "model_name": out_name,
        "data_tag": data_tag,
        "stamp": stamp,
        "protocol": hp.protocol,
        "early_stop": "acc_paper",
        "balbatch": True,
        "train_device": TRAIN_DEVICE_LABEL,
        "train_device_note": TRAIN_DEVICE_NOTE,
        "out_root_tag": OUT_ROOT_TAG,
        "task": sum_task,
        "three": sum_three,
    }
    (out_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log_line(log_path, "done")
