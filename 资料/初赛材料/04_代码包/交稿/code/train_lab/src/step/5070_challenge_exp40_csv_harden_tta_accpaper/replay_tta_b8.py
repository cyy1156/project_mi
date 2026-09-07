# -*- coding: utf-8 -*-
"""Exp40 T1：已落盘窗向内缩 TTA（边缘复制）+ 嵌套 E1f θ。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_STEP = Path(__file__).resolve().parent
_STEP_PARENT = _STEP.parent
from exp40_config import (  # noqa: E402
    DELTA_TTA_LINE,
    FOLD_OK_MIN,
    MEMBER_KEYS,
    RB8_ANCHOR,
    TTA_DELTAS,
    b8_step,
    exp39_out,
    exp40_out,
)

_B8 = b8_step()
for p in (_STEP_PARENT, _B8, _STEP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_paths import resolve_data  # noqa: E402
from e1f_core import E1fConfig, accuracy, fuse_with_config  # noqa: E402


def inward_shrink_edge_replicate(X: np.ndarray, delta: int) -> np.ndarray:
    """
    X: (N, C, T) 或 (N, 1, C, T)，已 z-score。
    δ>0：取 [:, δ:T]，左侧用首点复制补 δ。
    """
    if delta == 0:
        return X
    if X.ndim == 4:
        # (N,1,C,T)
        body = X[..., delta:]
        edge = X[..., :1]
        pad = np.repeat(edge, delta, axis=-1)
        return np.concatenate([pad, body], axis=-1)
    if X.ndim == 3:
        body = X[:, :, delta:]
        edge = X[:, :, :1]
        pad = np.repeat(edge, delta, axis=-1)
        return np.concatenate([pad, body], axis=-1)
    raise ValueError(f"unsupported X ndim={X.ndim}")


def _builder(name: str):
    if name in ("shallow", "shallow_b"):
        from baseline_shallow import build_model
    elif name == "eegnet":
        from baseline_eegnet import build_model
    else:
        from baseline_conformer import build_model
    return build_model


def _predict(model, X: np.ndarray, device, batch: int = 128) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch):
        xb = np.asarray(X[i : i + batch], np.float32).copy()
        if xb.ndim == 4 and xb.shape[1] == 1:
            xb = xb[:, 0]
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                p = torch.softmax(
                    model(torch.from_numpy(xb).to(device)).float(), 1
                ).cpu().numpy()
        outs.append(p)
    return np.concatenate(outs, axis=0)


def _load_model(member_dir: Path, fold: int, name: str, device):
    ckpt = torch.load(
        member_dir / f"fold{fold}" / "best_three.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = _builder(name)(
        int(ckpt["n_chans"]),
        int(ckpt["n_times"]),
        int(ckpt["n_outputs"]),
        float(ckpt.get("hp", {}).get("drop_prob", 0.5)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model


def fold_ok(arm_accs: list[float], base_accs: list[float]) -> dict:
    n_ge = int(sum(1 for a, b in zip(arm_accs, base_accs) if a + 1e-12 >= b))
    return {"n_folds_ge": n_ge, "fold_ok": bool(n_ge >= FOLD_OK_MIN)}


def run_tta(
    *,
    member_dirs: dict[str, Path],
    rb8_folds: list[dict],
    base_fold_accs: list[float],
    deltas: tuple[int, ...] = TTA_DELTAS,
) -> dict:
    data_dir, prefix = resolve_data("challenge_mi_3s_8ch")
    X_all = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TTA device={device} deltas={deltas} X={tuple(X_all.shape)}", flush=True)

    # 用任一成员折目录取 val_idx / val_y
    ref = next(iter(member_dirs.values()))
    n_folds = len(rb8_folds)
    fold_accs, fold_meta, oof_probs, oof_y = [], [], [], []

    for k in range(n_folds):
        fd = ref / f"fold{k}"
        idx = np.load(fd / "val_idx.npy")
        y = np.load(fd / "val_y.npy").astype(np.int64)
        Xv = np.asarray(X_all[idx], dtype=np.float32)

        cfg_d = rb8_folds[k]["config"]
        cfg = E1fConfig(
            member_names=list(cfg_d["member_names"]),
            temperatures=list(cfg_d["temperatures"]),
            weights=list(cfg_d["weights"]),
        )

        # 每成员：多 δ 概率平均
        member_probs = []
        for name in cfg.member_names:
            model = _load_model(member_dirs[name], k, name, device)
            views = []
            for d in deltas:
                Xd = inward_shrink_edge_replicate(Xv, int(d))
                views.append(_predict(model, Xd, device))
            member_probs.append(np.mean(np.stack(views, axis=0), axis=0).astype(np.float32))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        fused = fuse_with_config(member_probs, cfg)
        acc = float(accuracy(fused, y))
        fold_accs.append(acc)
        oof_probs.append(fused)
        oof_y.append(y)
        fold_meta.append({"fold": k, "acc": acc, "config": cfg.to_dict(), "deltas": list(deltas)})
        print(
            f"  TTA-B8 fold{k} acc={acc:.4f} (base={base_fold_accs[k]:.4f})",
            flush=True,
        )

    fo = fold_ok(fold_accs, base_fold_accs)
    mean = float(np.mean(fold_accs))
    std = float(np.std(fold_accs, ddof=1)) if n_folds > 1 else 0.0
    delta = mean - float(np.mean(base_fold_accs))
    enter = bool(delta + 1e-12 >= DELTA_TTA_LINE and fo["fold_ok"])

    return {
        "arm_id": "TTA-B8",
        "desc": f"inward TTA deltas={list(deltas)} edge-replicate + nested E1f",
        "val_acc_mean": mean,
        "val_acc_std": std,
        "fold_accs": fold_accs,
        "folds": fold_meta,
        "delta_vs_rb8": delta,
        "deltas": list(deltas),
        **fo,
        "enter_candidate": enter,
        "_oof_probs": oof_probs,
        "_oof_y": oof_y,
    }


def main() -> int:
    ranking = json.loads((exp39_out() / "replay" / "ranking_latest.json").read_text(encoding="utf-8"))
    member_dirs = {k: Path(v) for k, v in ranking["member_dirs_b8"].items()}
    rb8_folds = ranking["arms"]["R-B8"]["folds"]
    base_accs = ranking["arms"]["R-B8"]["fold_accs"]
    print("=== T1 TTA-B8 ===", flush=True)
    arm = run_tta(member_dirs=member_dirs, rb8_folds=rb8_folds, base_fold_accs=base_accs)
    out = exp40_out() / "replay"
    out.mkdir(parents=True, exist_ok=True)
    pub = {k: v for k, v in arm.items() if not k.startswith("_")}
    (out / "tta_latest.json").write_text(
        json.dumps(pub, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(out / "tta_b8_oof_prob.npy", np.concatenate(arm["_oof_probs"], axis=0))
    np.save(out / "tta_b8_oof_y.npy", np.concatenate(arm["_oof_y"], axis=0))
    print(
        f"TTA-B8 nested={arm['val_acc_mean']:.4f} Δ={arm['delta_vs_rb8']*100:+.2f}pp "
        f"fold_ok={arm['fold_ok']} enter={arm['enter_candidate']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
