# -*- coding: utf-8 -*-
"""Exp40 终态 CSV：按 harden_latest 决策写出或复制 R-B8_raw。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_STEP = Path(__file__).resolve().parent
_STEP_PARENT = _STEP.parent
from exp40_config import (  # noqa: E402
    TTA_DELTAS,
    b8_step,
    exp39_out,
    exp40_out,
    repo_root,
)
from replay_margin_b8 import _apply_bias  # noqa: E402
from replay_tta_b8 import (  # noqa: E402
    _load_model,
    _predict,
    inward_shrink_edge_replicate,
)

_B8 = b8_step()
for p in (_STEP_PARENT, _B8, _STEP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_paths import resolve_data  # noqa: E402
from e1f_core import E1fConfig, fuse_with_config  # noqa: E402


def _find_sample() -> Path:
    for c in (repo_root() / "DATA").iterdir():
        if (c / "sample_submission.csv").is_file():
            return c / "sample_submission.csv"
    raise FileNotFoundError("sample_submission.csv")


def _write_labels(labels: np.ndarray, out_csv: Path) -> Path:
    src = _find_sample()
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


def _predict_rb8_test(ranking: dict, *, use_tta: bool) -> np.ndarray:
    member_dirs = {k: Path(v) for k, v in ranking["member_dirs_b8"].items()}
    folds = ranking["arms"]["R-B8"]["folds"]
    data_dir, prefix = resolve_data("challenge_mi_3s_8ch")
    Xte = np.load(data_dir / f"{prefix}_test_X.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deltas = TTA_DELTAS if use_tta else (0,)
    fold_probs = []
    for rec in folds:
        fold = int(rec["fold"])
        cfg_d = rec["config"]
        cfg = E1fConfig(
            member_names=list(cfg_d["member_names"]),
            temperatures=list(cfg_d["temperatures"]),
            weights=list(cfg_d["weights"]),
        )
        member_probs = []
        for name in cfg.member_names:
            model = _load_model(member_dirs[name], fold, name, device)
            views = []
            for d in deltas:
                Xd = inward_shrink_edge_replicate(np.asarray(Xte, np.float32), int(d))
                views.append(_predict(model, Xd, device))
            member_probs.append(np.mean(np.stack(views, 0), 0).astype(np.float32))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fold_probs.append(fuse_with_config(member_probs, cfg))
        print(f"  test fold{fold} tta={use_tta} ok", flush=True)
    return np.mean(np.stack(fold_probs, 0), 0)


def _margin_counts(labels: np.ndarray) -> dict:
    # 假设 0=L, 1=R, 2=Rest（与挑战杯 three 一致；若不同只作相对诊断）
    c = np.bincount(labels.astype(int), minlength=3)
    return {"L": int(c[0]), "R": int(c[1]), "Rest": int(c[2])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harden-json", type=Path, default=None)
    args = ap.parse_args()
    harden_path = args.harden_json or (exp40_out() / "replay" / "harden_latest.json")
    doc = json.loads(harden_path.read_text(encoding="utf-8"))
    eng = doc["decision_engineering"]
    final = eng["decision_engineering_final"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = exp40_out() / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking = json.loads((exp39_out() / "replay" / "ranking_latest.json").read_text(encoding="utf-8"))
    src39 = exp39_out() / "submissions" / "submission_exp39_rb8_nested_20260904_022230.csv"
    if not src39.is_file():
        cands = sorted((exp39_out() / "submissions").glob("submission_exp39_rb8_*.csv"))
        src39 = cands[-1] if cands else None

    meta = {
        "decision_engineering_final": final,
        "label": eng.get("label"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm_freeze": True,
    }

    if final == "R-B8_raw":
        out = out_dir / f"submission_exp40_rb8_raw_final_{stamp}.csv"
        if src39 is None:
            raise SystemExit("找不到 Exp39 R-B8 CSV")
        shutil.copy2(src39, out)
        meta["path"] = str(out)
        meta["source"] = str(src39)
        meta["note"] = "终态=R-B8_raw；复制 Exp39 工程 CSV"
        print("copied", src39, "->", out, flush=True)
    else:
        use_tta = final in ("TTA-B8", "MC∘TTA")
        probs = _predict_rb8_test(ranking, use_tta=use_tta)
        if final in ("MC-B8", "MC∘TTA"):
            bias = np.asarray(doc["arms"]["MC-B8"]["pooled"]["bias"], dtype=np.float64)
            if final == "MC∘TTA":
                # 组合：对 TTA OOF 拟合的 bias 更贴切；若无 combo bias 则用 MC pooled
                combo = doc["arms"].get("MC∘TTA") or {}
                # 用 combo 各折 bias 平均作为 test 应用
                if combo.get("folds"):
                    bias = np.mean([f["bias"] for f in combo["folds"]], axis=0)
            probs = _apply_bias(probs, bias)
        labels = probs.argmax(1).astype(int)
        tag = final.replace("∘", "x").replace("-", "_")
        out = out_dir / f"submission_exp40_{tag}_final_{stamp}.csv"
        _write_labels(labels, out)
        meta["path"] = str(out)
        meta["test_margin"] = _margin_counts(labels)
        print("wrote", out, "margin", meta["test_margin"], flush=True)

    (out_dir / "final_decision_latest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
