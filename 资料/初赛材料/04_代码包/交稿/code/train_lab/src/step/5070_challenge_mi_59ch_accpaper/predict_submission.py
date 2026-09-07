"""用各折 best ckpt 对盲测 test 做 S-ens（6 折概率平均）并写 submission CSV。

用法（在本目录）：
  python predict_submission.py --member-dirs DIR1 DIR2 DIR3 DIR4
  # 或先单模：
  python predict_submission.py --run-dir path/to/shallow/.../three

默认读 preprocess out 的 challenge_test_X.npy，顺序与 sample_submission 对齐。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

_STEP = Path(__file__).resolve().parent
_STEP_PARENT = _STEP.parent
for p in (_STEP, _STEP_PARENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_paths import resolve_data
from shared_hparams import SHARED


def _load_model_from_ckpt(ckpt_path: Path, build_model):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_chans = int(ckpt.get("n_chans", SHARED.n_chans_expected))
    n_times = int(ckpt.get("n_times", SHARED.n_times_expected))
    n_outputs = int(ckpt.get("n_outputs", SHARED.n_outputs))
    drop = float(ckpt.get("hp", {}).get("drop_prob", SHARED.drop_prob))
    model = build_model(n_chans, n_times, n_outputs, drop)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _predict_X(model, X: np.ndarray, device, batch: int = 64) -> np.ndarray:
    model.to(device)
    probs = []
    n = X.shape[0]
    for i in range(0, n, batch):
        xb = np.asarray(X[i : i + batch], dtype=np.float32)
        if xb.ndim == 4 and xb.shape[1] == 1:
            xb = xb[:, 0]
        t = torch.from_numpy(xb).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(t)
                p = torch.softmax(logits.float(), dim=1).cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, nargs="*", default=[], help=".../three 目录（含 fold*/best_three.pt）")
    p.add_argument("--data-tag", default="challenge_mi_3s_59ch")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--builder", default="shallow", choices=["shallow", "shallow_b", "eegnet", "conformer"])
    args = p.parse_args()

    if args.builder == "shallow" or args.builder == "shallow_b":
        from baseline_shallow import build_model
    elif args.builder == "eegnet":
        from baseline_eegnet import build_model
    else:
        from baseline_conformer import build_model

    data_dir, prefix = resolve_data(args.data_tag)
    Xte = np.load(data_dir / f"{prefix}_test_X.npy", mmap_mode="r")
    print("test X", Xte.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = list(args.run_dir)
    if not run_dirs:
        raise SystemExit("请提供至少一个 --run-dir")

    all_fold_probs = []
    for rd in run_dirs:
        fold_dirs = sorted(rd.glob("fold*"))
        for fd in fold_dirs:
            ckpt = fd / "best_three.pt"
            if not ckpt.is_file():
                continue
            model = _load_model_from_ckpt(ckpt, build_model)
            prob = _predict_X(model, Xte, device)
            all_fold_probs.append(prob)
            print(f"  {ckpt} ok")

    if not all_fold_probs:
        raise SystemExit("未找到任何 best_three.pt")

    ens = np.mean(np.stack(all_fold_probs, axis=0), axis=0)
    labels = ens.argmax(axis=1).astype(int)

    # sample_submission
    repo = Path(__file__).resolve().parents[5]
    data_root = None
    for c in (repo / "DATA").iterdir():
        if (c / "sample_submission.csv").is_file():
            data_root = c
            break
    if data_root is None:
        raise FileNotFoundError("找不到 sample_submission.csv")

    src = data_root / "sample_submission.csv"
    out = args.out_csv or (
        Path(__file__).resolve().parents[3]
        / "out"
        / "5070_challenge_mi_59ch_accpaper"
        / "submissions"
        / "submission_exp34_e1f_a59_sens_preview.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["sample_id", "label"]
        for i, row in enumerate(reader):
            row = dict(row)
            row["label"] = str(int(labels[i]))
            rows.append(row)
    if len(rows) != len(labels):
        raise RuntimeError(f"submission 行数 {len(rows)} != pred {len(labels)}")

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, "n=", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
