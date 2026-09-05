# -*- coding: utf-8 -*-
"""Exp39 工程选卷 → 可选写 submission CSV。

复用 8ch / 59ch predict 管线；仅当 decision_engineering 要求新文件时调用。
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
_STEP_PARENT = _STEP.parent
from exp39_config import (  # noqa: E402
    a59_step,
    b8_step,
    exp39_out,
    repo_root,
)

_A59 = a59_step()
_B8 = b8_step()
for p in (_STEP_PARENT, _A59, _B8, _STEP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_paths import resolve_data  # noqa: E402
from e1f_core import E1fConfig, fuse_with_config  # noqa: E402


def _builder_factory(step: Path):
    if str(step) not in sys.path:
        sys.path.insert(0, str(step))

    def _builder(name: str):
        if name in ("shallow", "shallow_b"):
            from baseline_shallow import build_model
        elif name == "eegnet":
            from baseline_eegnet import build_model
        else:
            from baseline_conformer import build_model
        return build_model

    return _builder


def _predict(model, X, device, batch=128):
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch):
        xb = np.asarray(X[i : i + batch], np.float32)
        if xb.ndim == 4 and xb.shape[1] == 1:
            xb = xb[:, 0]
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                p = torch.softmax(model(torch.from_numpy(xb).to(device)).float(), 1).cpu().numpy()
        outs.append(p)
    return np.concatenate(outs)


def _find_sample_submission() -> Path:
    data = repo_root() / "DATA"
    for c in data.iterdir():
        if (c / "sample_submission.csv").is_file():
            return c / "sample_submission.csv"
    raise FileNotFoundError("sample_submission.csv")


def _write_labels(labels: np.ndarray, out_csv: Path) -> Path:
    src = _find_sample_submission()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames
        for i, row in enumerate(r):
            row = dict(row)
            row["label"] = str(int(labels[i]))
            rows.append(row)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out_csv


def _predict_e1f_folds(
    *,
    member_dirs: dict[str, Path],
    fold_cfgs: list[dict],
    data_tag: str,
    step_for_builder: Path,
) -> np.ndarray:
    data_dir, prefix = resolve_data(data_tag)
    Xte = np.load(data_dir / f"{prefix}_test_X.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = _builder_factory(step_for_builder)
    fold_probs = []
    for rec in fold_cfgs:
        fold = int(rec["fold"])
        cfg_d = rec["config"]
        cfg = E1fConfig(
            member_names=cfg_d["member_names"],
            temperatures=cfg_d["temperatures"],
            weights=cfg_d["weights"],
        )
        probs_m = []
        for name in cfg.member_names:
            ckpt = torch.load(
                member_dirs[name] / f"fold{fold}" / "best_three.pt",
                map_location="cpu",
                weights_only=False,
            )
            model = builder(name)(
                int(ckpt["n_chans"]),
                int(ckpt["n_times"]),
                int(ckpt["n_outputs"]),
                float(ckpt.get("hp", {}).get("drop_prob", 0.5)),
            )
            model.load_state_dict(ckpt["model"])
            model.to(device)
            probs_m.append(_predict(model, Xte, device))
        fold_probs.append(fuse_with_config(probs_m, cfg))
        print(f"  test fold{fold} ok", flush=True)
    return np.mean(np.stack(fold_probs, 0), 0)


def write_for_decision(doc: dict) -> Path | None:
    eng = doc.get("decision_engineering") or {}
    final = eng.get("decision_engineering")
    if not eng.get("write_new_csv"):
        print(f"engineering={final} → 不写新 CSV（维持 Exp34 S0）", flush=True)
        return None

    arms = doc["arms"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = exp39_out() / "submissions"
    member_b8 = {k: Path(v) for k, v in doc["member_dirs_b8"].items()}
    member_a59 = {k: Path(v) for k, v in doc["member_dirs_a59"].items()}

    if final == "R-B8":
        labels = _predict_e1f_folds(
            member_dirs=member_b8,
            fold_cfgs=arms["R-B8"]["folds"],
            data_tag="challenge_mi_3s_8ch",
            step_for_builder=_B8,
        ).argmax(1)
        out = out_dir / f"submission_exp39_rb8_nested_{stamp}.csv"
    elif final == "R-pool-B8":
        cfg = arms["R-pool-B8"]["pooled_config"]
        fold_cfgs = [
            {"fold": f, "config": cfg} for f in range(int(doc["n_folds"]))
        ]
        labels = _predict_e1f_folds(
            member_dirs=member_b8,
            fold_cfgs=fold_cfgs,
            data_tag="challenge_mi_3s_8ch",
            step_for_builder=_B8,
        ).argmax(1)
        out = out_dir / f"submission_exp39_pool_b8_{stamp}.csv"
    elif final == "V1":
        fold_cfgs = arms["V1"]["folds"]
        labels = _predict_e1f_folds(
            member_dirs=member_b8,
            fold_cfgs=fold_cfgs,
            data_tag="challenge_mi_3s_8ch",
            step_for_builder=_B8,
        ).argmax(1)
        out = out_dir / f"submission_exp39_v1_b8_shallow_b_{stamp}.csv"
    elif final == "R-uni50":
        # 每折：嵌套 A59-E1f 与 B8-E1f 流概率 50/50
        n_folds = int(doc["n_folds"])
        data_dir_a, prefix_a = resolve_data("challenge_mi_3s_59ch")
        data_dir_b, prefix_b = resolve_data("challenge_mi_3s_8ch")
        # test X 应对齐样本数；两轨各自预测再融合
        Xa = np.load(data_dir_a / f"{prefix_a}_test_X.npy", mmap_mode="r")
        Xb = np.load(data_dir_b / f"{prefix_b}_test_X.npy", mmap_mode="r")
        if Xa.shape[0] != Xb.shape[0]:
            raise RuntimeError("A59/B8 test N mismatch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ba = _builder_factory(_A59)
        bb = _builder_factory(_B8)
        fold_probs = []
        for k in range(n_folds):
            ca = arms["R-uni50"]["folds"][k]["stream_e1f_a59"]
            cb = arms["R-uni50"]["folds"][k]["stream_e1f_b8"]
            cfg_a = E1fConfig(
                member_names=ca["member_names"],
                temperatures=ca["temperatures"],
                weights=ca["weights"],
            )
            cfg_b = E1fConfig(
                member_names=cb["member_names"],
                temperatures=cb["temperatures"],
                weights=cb["weights"],
            )
            pa, pb = [], []
            for name in cfg_a.member_names:
                ckpt = torch.load(
                    member_a59[name] / f"fold{k}" / "best_three.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                model = ba(name)(
                    int(ckpt["n_chans"]),
                    int(ckpt["n_times"]),
                    int(ckpt["n_outputs"]),
                    float(ckpt.get("hp", {}).get("drop_prob", 0.5)),
                )
                model.load_state_dict(ckpt["model"])
                model.to(device)
                pa.append(_predict(model, Xa, device))
            for name in cfg_b.member_names:
                ckpt = torch.load(
                    member_b8[name] / f"fold{k}" / "best_three.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                model = bb(name)(
                    int(ckpt["n_chans"]),
                    int(ckpt["n_times"]),
                    int(ckpt["n_outputs"]),
                    float(ckpt.get("hp", {}).get("drop_prob", 0.5)),
                )
                model.load_state_dict(ckpt["model"])
                model.to(device)
                pb.append(_predict(model, Xb, device))
            sa = fuse_with_config(pa, cfg_a)
            sb = fuse_with_config(pb, cfg_b)
            fold_probs.append(0.5 * sa + 0.5 * sb)
            print(f"  uni50 test fold{k} ok", flush=True)
        labels = np.mean(np.stack(fold_probs, 0), 0).argmax(1)
        out = out_dir / f"submission_exp39_uni50_{stamp}.csv"
    else:
        print(f"未实现的工程臂 {final}，跳过写 CSV", flush=True)
        return None

    path = _write_labels(labels.astype(int), out)
    meta = {
        "decision_engineering": final,
        "path": str(path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "label": eng.get("label"),
        "note": "工程选择，非显著性结论；科学主行见 decision_science",
    }
    (out_dir / "decision_csv_latest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote", path, flush=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ranking-json",
        type=Path,
        default=None,
    )
    args = ap.parse_args()
    path = args.ranking_json or (exp39_out() / "replay" / "ranking_latest.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    write_for_decision(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
