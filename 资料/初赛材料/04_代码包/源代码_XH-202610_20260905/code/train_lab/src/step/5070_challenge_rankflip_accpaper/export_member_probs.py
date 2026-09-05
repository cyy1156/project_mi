# -*- coding: utf-8 -*-
"""导出 / 汇总成员 Val（已有）与 Test 概率到 Exp35 dump 目录。

用法：
  python export_member_probs.py --track a59
  python export_member_probs.py --track b8 --arm ft
  python export_member_probs.py --track a59 --skip-test   # 仅登记 Val 路径
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_STEP = Path(__file__).resolve().parent
_A59 = _STEP.parent / "5070_challenge_mi_59ch_accpaper"
for p in (_A59, _STEP.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(_STEP) not in sys.path:
    sys.path.insert(0, str(_STEP))

from data_paths import resolve_data  # noqa: E402
from exp35_config import MEMBER_KEYS, exp35_out  # noqa: E402
from member_paths import find_all_a59_members, find_all_b8_members, n_folds_available  # noqa: E402


def _builder(name: str):
    if name in ("shallow", "shallow_b"):
        from baseline_shallow import build_model
    elif name == "eegnet":
        from baseline_eegnet import build_model
    elif name == "conformer":
        from baseline_conformer import build_model
    else:
        raise KeyError(name)
    return build_model


def _predict(model, X, device, batch: int) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch):
        xb = np.array(X[i : i + batch], dtype=np.float32, copy=True)
        if xb.ndim == 4 and xb.shape[1] == 1:
            xb = xb[:, 0]
        t = torch.from_numpy(xb).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(t)
                p = torch.softmax(logits.float(), dim=1).cpu().numpy()
        outs.append(p)
    return np.concatenate(outs, axis=0)


def _load_model(ckpt: Path, build_model, device, n_chans_default: int):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    n_chans = int(obj.get("n_chans", n_chans_default))
    n_times = int(obj.get("n_times", 750))
    n_out = int(obj.get("n_outputs", 3))
    drop = float((obj.get("hp") or {}).get("drop_prob", 0.5))
    model = build_model(n_chans, n_times, n_out, drop)
    model.load_state_dict(obj["model"])
    return model.to(device)


def export_track(
    *,
    track: str,
    arm: str | None,
    skip_test: bool,
    batch_eval: int,
    prefer_tag: str,
) -> Path:
    if track == "a59":
        member_dirs = find_all_a59_members(prefer_tag=prefer_tag)
        data_tag = "challenge_mi_3s_59ch"
        n_chans = 59
        dump_name = "a59"
    elif track == "b8":
        if arm not in ("ft", "scratch"):
            raise SystemExit("--arm 须为 ft|scratch")
        member_dirs = find_all_b8_members(arm=arm, prefer_tag=prefer_tag)
        data_tag = "challenge_mi_3s_8ch"
        n_chans = 8
        dump_name = f"b8_{arm}"
    else:
        raise SystemExit("track 须为 a59|b8")

    if len(member_dirs) < 2:
        raise SystemExit(f"成员不足: {member_dirs}")

    n_folds = n_folds_available(member_dirs)
    if n_folds < 1:
        raise SystemExit("无可用 fold val_prob")

    out_root = exp35_out() / "dumps" / dump_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Val：写指针 manifest（不复制大文件）
    val_manifest = {
        "track": track,
        "arm": arm,
        "member_dirs": {k: str(v) for k, v in member_dirs.items()},
        "n_folds": n_folds,
        "prefer_tag": prefer_tag,
        "val_source": "exp34_inline_val_prob",
    }
    with (out_root / "val_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(val_manifest, f, ensure_ascii=False, indent=2)

    # 每折 sanity：对齐 val_y
    for fold in range(n_folds):
        y_ref = None
        idx_ref = None
        for name, d in member_dirs.items():
            y = np.load(d / f"fold{fold}" / "val_y.npy")
            idx_p = d / f"fold{fold}" / "val_idx.npy"
            idx = np.load(idx_p) if idx_p.is_file() else None
            if y_ref is None:
                y_ref = y
                idx_ref = idx
            else:
                if not np.array_equal(y, y_ref):
                    raise RuntimeError(f"fold{fold} {name} val_y 不一致")
                if idx is not None and idx_ref is not None and not np.array_equal(idx, idx_ref):
                    raise RuntimeError(f"fold{fold} {name} val_idx 不一致")

    test_meta = {"skipped": True}
    if not skip_test:
        data_dir, prefix = resolve_data(data_tag)
        Xte = np.load(data_dir / f"{prefix}_test_X.npy", mmap_mode="r")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("test", Xte.shape, "device", device)
        test_dir = out_root / "test_probs"
        test_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(n_folds):
            for name, d in member_dirs.items():
                ckpt = d / f"fold{fold}" / "best_three.pt"
                out_p = test_dir / f"fold{fold}_{name}.npy"
                if out_p.is_file():
                    print("skip existing", out_p.name)
                    continue
                # B8 用 8ch builders：与 A59 同名脚本在不同包；此处用 A59 builders 仅当 n_chans 匹配 ckpt
                # 对 B8：临时把 path 切到 8ch 包
                if track == "b8":
                    _B8 = _STEP.parent / "5070_challenge_mi_8ch_ft_accpaper"
                    if str(_B8) not in sys.path:
                        sys.path.insert(0, str(_B8))
                    if name in ("shallow", "shallow_b"):
                        from baseline_shallow import build_model as bm
                    elif name == "eegnet":
                        from baseline_eegnet import build_model as bm
                    else:
                        from baseline_conformer import build_model as bm
                else:
                    bm = _builder(name)
                model = _load_model(ckpt, bm, device, n_chans)
                probs = _predict(model, Xte, device, batch_eval)
                np.save(out_p, probs.astype(np.float32))
                print("wrote", out_p.name, probs.shape)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        test_meta = {
            "skipped": False,
            "test_dir": str(test_dir),
            "n_test": int(Xte.shape[0]),
            "n_folds": n_folds,
        }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "val_manifest": str(out_root / "val_manifest.json"),
        "test": test_meta,
        "members": list(member_dirs.keys()),
        "n_folds": n_folds,
    }
    with (out_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("dump root", out_root)
    return out_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=("a59", "b8"), default="a59")
    ap.add_argument("--arm", choices=("ft", "scratch"), default="ft")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--batch-eval", type=int, default=64)
    ap.add_argument("--prefer-tag", default="full_20260902_1930")
    args = ap.parse_args()
    export_track(
        track=args.track,
        arm=args.arm if args.track == "b8" else None,
        skip_test=bool(args.skip_test),
        batch_eval=args.batch_eval,
        prefer_tag=args.prefer_tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
