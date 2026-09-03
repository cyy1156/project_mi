# -*- coding: utf-8 -*-
"""轨 S：按预注册规则锁定交卷候选并写 CSV。

用法：
  python make_submission_candidates.py --replay-json path/to/replay_FM_latest.json
  python make_submission_candidates.py --replay-json ... --write-csv
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
_A59 = _STEP.parent / "5070_challenge_mi_59ch_accpaper"
for p in (_A59, _STEP.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(_STEP) not in sys.path:
    sys.path.insert(0, str(_STEP))

from data_paths import resolve_data  # noqa: E402
from e1f_constrained import E1fConfig, fuse_with_config  # noqa: E402
from exp35_config import ANCHOR_E1F_A59, exp35_out, repo_root  # noqa: E402
from member_paths import find_all_a59_members  # noqa: E402


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


def _load_model(ckpt: Path, build_model, device):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    n_chans = int(obj.get("n_chans", 59))
    n_times = int(obj.get("n_times", 750))
    n_out = int(obj.get("n_outputs", 3))
    drop = float((obj.get("hp") or {}).get("drop_prob", 0.5))
    model = build_model(n_chans, n_times, n_out, drop)
    model.load_state_dict(obj["model"])
    return model.to(device)


# v0.2：结构决赛恒进（即使未过旧「晋级线」）
F_CONSTRAINT_ARMS = ("F3", "F4", "F5", "F6", "F7")
M_POOL_PREF = ("M2", "M2c")
M_POOL_FALLBACK = ("M2", "M2c", "M3")


def _best_arm(results: dict, arm_ids: tuple[str, ...]) -> tuple[str | None, float]:
    best_id: str | None = None
    best_val = float("-inf")
    for aid in arm_ids:
        if aid not in results:
            continue
        v = float(results[aid]["val_acc_mean"])
        if v > best_val:
            best_val = v
            best_id = aid
    if best_id is None:
        return None, float("nan")
    return best_id, best_val


def pick_structural_arms(results: dict) -> dict:
    """返回 C_fuse / C_pool / C_conf 对应的源臂。"""
    fuse_id, _ = _best_arm(results, F_CONSTRAINT_ARMS)
    pool_id, _ = _best_arm(results, M_POOL_PREF)
    if pool_id is None:
        pool_id, _ = _best_arm(results, M_POOL_FALLBACK)
    conf_id = "M1" if "M1" in results else None
    return {"C_fuse": fuse_id, "C_pool": pool_id, "C_conf": conf_id, "S0": "F0"}


def build_candidates(replay: dict) -> list[dict]:
    """v0.2：S0 + C_fuse + C_pool + C_conf 强制决赛（报表 id 仍用 S0/S1/S2/S3）。"""
    results = replay.get("results") or {}
    arms = pick_structural_arms(results)

    cands: list[dict] = [
        {
            "id": "S0",
            "structural": "S0",
            "source_arm": "F0",
            "val_acc": float((results.get("F0") or results.get("M0") or {}).get("val_acc_mean", ANCHOR_E1F_A59)),
            "priority": 0,
            "desc": "Exp34 E1f 无约束基线（锚）",
        },
    ]
    if arms["C_conf"]:
        cands.append(
            {
                "id": "S1",
                "structural": "C_conf",
                "source_arm": arms["C_conf"],
                "val_acc": float(results[arms["C_conf"]]["val_acc_mean"]),
                "priority": 1,
                "desc": "conformer 单模（恒进）",
            }
        )
    if arms["C_fuse"]:
        cands.append(
            {
                "id": "S2",
                "structural": "C_fuse",
                "source_arm": arms["C_fuse"],
                "val_acc": float(results[arms["C_fuse"]]["val_acc_mean"]),
                "priority": 2,
                "desc": f"约束融合最优 {arms['C_fuse']}（恒进决赛）",
            }
        )
    if arms["C_pool"]:
        cands.append(
            {
                "id": "S3",
                "structural": "C_pool",
                "source_arm": arms["C_pool"],
                "val_acc": float(results[arms["C_pool"]]["val_acc_mean"]),
                "priority": 3,
                "desc": f"缩池最优 {arms['C_pool']}（恒进决赛）",
            }
        )

    cands = [c for c in cands if c["val_acc"] == c["val_acc"]]
    # 内部读数：argmax Val；并列（差 < 0.005）优先级 C_pool>C_fuse>C_conf>S0
    best = None
    for c in cands:
        if best is None:
            best = c
            continue
        if c["val_acc"] > best["val_acc"] + 0.005:
            best = c
        elif abs(c["val_acc"] - best["val_acc"]) < 0.005 and c["priority"] > best["priority"]:
            best = c
    for c in cands:
        c["is_internal_main"] = best is not None and c["id"] == best["id"]
        # 兼容旧字段：is_main = 内部读数（不等于对外交卷）
        c["is_main"] = c["is_internal_main"]
    return cands


def _arm_fold_cfgs(replay: dict, arm_id: str) -> list[dict]:
    arm = (replay.get("results") or {}).get(arm_id)
    if not arm:
        raise KeyError(arm_id)
    return arm["folds"]


def predict_sens(
    *,
    fold_recs: list[dict],
    member_dirs: dict[str, Path],
    Xte: np.ndarray,
    device: torch.device,
    batch_eval: int,
) -> np.ndarray:
    fold_probs = []
    for rec in fold_recs:
        cfg_d = rec["config"]
        cfg = E1fConfig(
            member_names=list(cfg_d["member_names"]),
            temperatures=list(cfg_d["temperatures"]),
            weights=list(cfg_d["weights"]),
            smooth_radius=0,
            val_acc=float(cfg_d.get("val_acc", 0)),
        )
        probs_m = []
        for name in cfg.member_names:
            ckpt = member_dirs[name] / f"fold{rec['fold']}" / "best_three.pt"
            model = _load_model(ckpt, _builder(name), device)
            probs_m.append(_predict(model, Xte, device, batch_eval))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fold_probs.append(fuse_with_config(probs_m, cfg))
    return np.mean(np.stack(fold_probs, axis=0), axis=0)


def write_csv(labels: np.ndarray, out_csv: Path) -> dict:
    sub_src = None
    data_root = repo_root() / "DATA"
    if data_root.is_dir():
        for c in data_root.iterdir():
            if (c / "sample_submission.csv").is_file():
                sub_src = c / "sample_submission.csv"
                break
    if sub_src is None:
        raise FileNotFoundError("sample_submission.csv")

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
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    counts = {str(i): int((labels == i).sum()) for i in range(3)}
    meta = {"out_csv": str(out_csv), "label_counts": counts, "n": int(len(labels))}
    with out_csv.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _arm_max_w_eegnet(replay: dict, arm_id: str) -> float | None:
    arm = (replay.get("results") or {}).get(arm_id) or {}
    folds = arm.get("folds") or []
    if not folds:
        return None
    mx = 0.0
    for fr in folds:
        cfg = fr.get("config") or {}
        names = list(cfg.get("member_names") or [])
        weights = list(cfg.get("weights") or [])
        for n, w in zip(names, weights):
            if n == "eegnet":
                mx = max(mx, float(w))
    return float(mx)


def _pool_size(replay: dict, arm_id: str) -> int | None:
    arm = (replay.get("results") or {}).get(arm_id) or {}
    folds = arm.get("folds") or []
    if not folds:
        pool = arm.get("pool")
        return len(pool) if isinstance(pool, (list, tuple)) else None
    cfg = folds[0].get("config") or {}
    names = cfg.get("member_names") or []
    return len(names) if names else None


def replace_gate(main: dict, s0_acc: float, replay: dict | None = None) -> dict:
    """方案 §12.1 字面门槛（v0.2.1 收紧）。

    允许替换仅当：
      (A) Val ≥ S0 + 0.010，或
      (B) Val ≥ S0 − 0.005 **且** 成员池更小 **且** max w_eegnet ≤ 0.15
    F3 等「同池约束」不满足 (B)，不得标 robust_simpler。
    """
    val = float(main["val_acc"])
    struct = main.get("structural") or main["id"]
    arm = main.get("source_arm") or ""
    plus = val >= s0_acc + 0.010

    s0_pool = 4
    cand_pool = _pool_size(replay or {}, arm) if replay else None
    smaller_pool = cand_pool is not None and cand_pool < s0_pool
    w_eeg = _arm_max_w_eegnet(replay or {}, arm) if replay else None
    no_pathological_eeg = w_eeg is not None and w_eeg <= 0.15
    robust = (
        val >= s0_acc - 0.005
        and smaller_pool
        and no_pathological_eeg
        and (main["id"] in ("S2", "S3") or struct in ("C_fuse", "C_pool"))
    )
    ok = bool(plus or robust)
    if plus:
        reason = "Val+1pp"
    elif robust:
        reason = "robust_simpler"
    elif (
        val >= s0_acc - 0.005
        and (main["id"] in ("S2", "S3") or struct in ("C_fuse", "C_pool"))
        and not smaller_pool
    ):
        reason = "near_tie_same_pool_not_§12.1"  # 敏感性变体，非替换
    else:
        reason = "not_met"
    return {
        "allowed": ok,
        "reason": reason,
        "main_id": main["id"],
        "structural": struct,
        "main_val": val,
        "s0_val": s0_acc,
        "cand_pool_size": cand_pool,
        "max_w_eegnet": w_eeg,
        "note": "allowed=false 时对外主交卷保持 S0；near_tie 仅作敏感性附报",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-json", type=Path, required=True)
    ap.add_argument("--write-csv", action="store_true")
    ap.add_argument("--batch-eval", type=int, default=64)
    ap.add_argument("--prefer-tag", default="full_20260902_1930")
    args = ap.parse_args()

    replay = json.loads(args.replay_json.read_text(encoding="utf-8"))
    cands = build_candidates(replay)
    s0 = next(c for c in cands if c["id"] == "S0")
    internal = next(c for c in cands if c.get("is_internal_main") or c.get("is_main"))
    gate = replace_gate(internal, float(s0["val_acc"]), replay)
    # 对外主交卷：仅当 §12.1 通过才换；否则锁定 S0 终态
    external = internal if gate["allowed"] else s0
    csv_status = "final_S0" if external["id"] == "S0" else f"final_{external['id']}_replaced"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = exp35_out() / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scheme_version": "v0.2.1",
        "replay_json": str(args.replay_json),
        "structural_arms": pick_structural_arms(replay.get("results") or {}),
        "candidates": cands,
        "internal_main": {k: internal[k] for k in ("id", "structural", "source_arm", "val_acc", "desc")},
        "external_submission": {
            "id": external["id"],
            "structural": external.get("structural"),
            "source_arm": external["source_arm"],
            "val_acc": external["val_acc"],
            "path_hint": "submission_exp34_e1f_a59_sens_full_20260902_1930.csv"
            if external["id"] == "S0"
            else None,
        },
        "replace_gate": gate,
        "exp34_csv_status": csv_status,
        "csv_files": {},
    }

    if args.write_csv:
        member_dirs = find_all_a59_members(prefer_tag=args.prefer_tag)
        data_dir, prefix = resolve_data("challenge_mi_3s_59ch")
        Xte = np.load(data_dir / f"{prefix}_test_X.npy", mmap_mode="r")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 去重：相同 source_arm 只写一次
        written_arms: dict[str, Path] = {}
        label_cache: dict[str, np.ndarray] = {}
        for c in cands:
            arm = c["source_arm"]
            if arm not in (replay.get("results") or {}):
                print("skip CSV, missing arm", arm)
                continue
            if arm in written_arms:
                decision_doc["csv_files"][c["id"]] = str(written_arms[arm])
                continue
            folds = _arm_fold_cfgs(replay, arm)
            ens = predict_sens(
                fold_recs=folds,
                member_dirs=member_dirs,
                Xte=Xte,
                device=device,
                batch_eval=args.batch_eval,
            )
            labels = ens.argmax(axis=1).astype(int)
            label_cache[c["id"]] = labels
            out_csv = out_dir / f"submission_exp35_{c['id']}_{arm}_{stamp}.csv"
            meta = write_csv(labels, out_csv)
            written_arms[arm] = out_csv
            decision_doc["csv_files"][c["id"]] = str(out_csv)
            print("wrote", out_csv, meta["label_counts"])

        # Hamming vs S0
        if "S0" in label_cache:
            base = label_cache["S0"]
            hamm = {}
            for cid, lab in label_cache.items():
                if cid == "S0":
                    continue
                hamm[cid] = float((lab != base).mean())
            decision_doc["hamming_vs_s0"] = hamm

    out_json = out_dir / f"decision_{stamp}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(decision_doc, f, ensure_ascii=False, indent=2)
    latest = out_dir / "decision_latest.json"
    with latest.open("w", encoding="utf-8") as f:
        json.dump(decision_doc, f, ensure_ascii=False, indent=2)
    print("internal_main", decision_doc["internal_main"])
    print("external_submission", decision_doc["external_submission"])
    print("replace_gate", gate)
    print("exp34_csv_status", csv_status)
    print("wrote", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
