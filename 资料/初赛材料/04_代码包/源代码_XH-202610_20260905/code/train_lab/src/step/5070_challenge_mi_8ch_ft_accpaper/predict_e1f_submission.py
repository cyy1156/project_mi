"""E1f-B8 + S-ens → 对照 submission（非主行）。"""

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
sys.path.insert(0, str(_STEP))
sys.path.insert(0, str(_STEP.parent))

from data_paths import resolve_data  # noqa: E402
from e1f_core import E1fConfig, fuse_with_config  # noqa: E402
from shared_hparams import OUT_ROOT_TAG, SHARED  # noqa: E402


def _builder(name: str):
    if name in ("shallow", "shallow_b"):
        from baseline_shallow import build_model
    elif name == "eegnet":
        from baseline_eegnet import build_model
    else:
        from baseline_conformer import build_model
    return build_model


def _predict(model, X, device, batch=128):
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch):
        xb = np.asarray(X[i:i+batch], np.float32)
        if xb.ndim == 4 and xb.shape[1] == 1:
            xb = xb[:, 0]
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                p = torch.softmax(model(torch.from_numpy(xb).to(device)).float(), 1).cpu().numpy()
        outs.append(p)
    return np.concatenate(outs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1f-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()
    e1f = json.loads(args.e1f_json.read_text(encoding="utf-8"))
    member_dirs = {k: Path(v) for k, v in e1f["member_dirs"].items()}
    data_dir, prefix = resolve_data("challenge_mi_3s_8ch")
    Xte = np.load(data_dir / f"{prefix}_test_X.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_probs = []
    for rec in e1f["folds"]:
        fold = int(rec["fold"])
        cfg = E1fConfig(
            member_names=rec["config"]["member_names"],
            temperatures=rec["config"]["temperatures"],
            weights=rec["config"]["weights"],
        )
        probs_m = []
        for name in cfg.member_names:
            ckpt = torch.load(member_dirs[name] / f"fold{fold}" / "best_three.pt",
                              map_location="cpu", weights_only=False)
            model = _builder(name)(int(ckpt["n_chans"]), int(ckpt["n_times"]),
                                   int(ckpt["n_outputs"]), float(ckpt.get("hp", {}).get("drop_prob", 0.5)))
            model.load_state_dict(ckpt["model"])
            model.to(device)
            probs_m.append(_predict(model, Xte, device))
        fold_probs.append(fuse_with_config(probs_m, cfg))
        print("fold", fold, "ok")

    labels = np.mean(np.stack(fold_probs), 0).argmax(1).astype(int)
    repo = Path(__file__).resolve().parents[5]
    src = next(c / "sample_submission.csv" for c in (repo / "DATA").iterdir()
               if (c / "sample_submission.csv").is_file())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_csv or (
        Path(__file__).resolve().parents[3] / "out" / OUT_ROOT_TAG / "submissions"
        / f"submission_exp34_e1f_b8_sens_{stamp}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames
        for i, row in enumerate(r):
            row = dict(row)
            row["label"] = str(int(labels[i]))
            rows.append(row)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
