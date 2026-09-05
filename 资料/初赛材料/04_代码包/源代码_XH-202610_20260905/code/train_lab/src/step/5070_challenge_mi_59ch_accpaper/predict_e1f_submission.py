"""E1f-A59 + S-ens 写主 submission CSV。

对每个 LOSO 折：用该折四成员 ckpt + 该折 E1f(T,w) 融合盲测概率；
再对 6 折概率平均 → argmax → sample_submission 顺序。

用法：
  python predict_e1f_submission.py --e1f-json path/to/e1f_*.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_STEP = Path(__file__).resolve().parent
_PARENT = _STEP.parent
for p in (_STEP, _PARENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_paths import resolve_data  # noqa: E402
from e1f_core import E1fConfig, fuse_with_config  # noqa: E402
from shared_hparams import OUT_ROOT_TAG, SHARED  # noqa: E402


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


def _predict(model, X, device, batch: int = 64) -> np.ndarray:
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


def _load_model(ckpt: Path, build_model, device):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    n_chans = int(obj.get("n_chans", SHARED.n_chans_expected))
    n_times = int(obj.get("n_times", SHARED.n_times_expected))
    n_out = int(obj.get("n_outputs", SHARED.n_outputs))
    drop = float((obj.get("hp") or {}).get("drop_prob", SHARED.drop_prob))
    model = build_model(n_chans, n_times, n_out, drop)
    model.load_state_dict(obj["model"])
    return model.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1f-json", type=Path, required=True)
    ap.add_argument("--data-tag", default="challenge_mi_3s_59ch")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--batch-eval", type=int, default=64)
    args = ap.parse_args()

    with args.e1f_json.open("r", encoding="utf-8") as f:
        e1f = json.load(f)
    member_dirs = {k: Path(v) for k, v in e1f["member_dirs"].items()}
    fold_cfgs = {int(rec["fold"]): rec for rec in e1f["folds"]}

    data_dir, prefix = resolve_data(args.data_tag)
    Xte = np.load(data_dir / f"{prefix}_test_X.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("test", Xte.shape, "device", device)

    fold_probs = []
    for fold, rec in sorted(fold_cfgs.items()):
        cfg = E1fConfig(
            member_names=list(rec["config"]["member_names"]),
            temperatures=list(rec["config"]["temperatures"]),
            weights=list(rec["config"]["weights"]),
            smooth_radius=int(rec["config"].get("smooth_radius", 0)),
            val_acc=float(rec["config"].get("val_acc", 0)),
        )
        probs_m = []
        for name in cfg.member_names:
            ckpt = member_dirs[name] / f"fold{fold}" / "best_three.pt"
            model = _load_model(ckpt, _builder(name), device)
            probs_m.append(_predict(model, Xte, device, batch=args.batch_eval))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fused = fuse_with_config(probs_m, cfg)
        fold_probs.append(fused)
        print(f"fold{fold} fused ok")

    ens = np.mean(np.stack(fold_probs, axis=0), axis=0)
    labels = ens.argmax(axis=1).astype(int)

    repo = Path(__file__).resolve().parents[5]
    sub_src = None
    for c in (repo / "DATA").iterdir():
        if (c / "sample_submission.csv").is_file():
            sub_src = c / "sample_submission.csv"
            break
    if sub_src is None:
        raise FileNotFoundError("sample_submission.csv")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_csv or (
        Path(__file__).resolve().parents[3]
        / "out"
        / OUT_ROOT_TAG
        / "submissions"
        / f"submission_exp34_e1f_a59_sens_{stamp}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with sub_src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or ["sample_id", "label"]
        for i, row in enumerate(reader):
            row = dict(row)
            row["label"] = str(int(labels[i]))
            rows.append(row)
    if len(rows) != len(labels):
        raise RuntimeError(f"rows {len(rows)} != preds {len(labels)}")
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    meta = {
        "e1f_json": str(args.e1f_json),
        "n_folds": len(fold_probs),
        "out_csv": str(out),
        "label_counts": {str(i): int((labels == i).sum()) for i in range(3)},
    }
    with out.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("wrote", out, meta["label_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
