"""实验 27 · fnz FT + Replay 双轨 grid（Track S=ws01, M=ws01+ws02）。

用法:
  python experiment_game/tools/exp27_fnz_replay_grid.py
  python experiment_game/tools/exp27_fnz_replay_grid.py --tracks S --arms A0,B2
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "code" / "preprocess_lab"))

from adapt_engine.ft import FTRecipe, IncrementalFinetuner, ReplayPool
from adapt_engine.registry import load_head
from experiment_game.tools.ft_subject_from_v3 import (
    DEFAULT_THREE,
    N_TIMES,
    _eval_acc,
    _trial_split,
    build_dataset,
)

OPENBMI_ROOT = _REPO / "code/preprocess_lab/out/openbmi_3s_hop100"
TEACHABLE_JSON = _REPO / "find_best_trail/out/teachable_trials_v1.json"
WS01 = _REPO / "experiment_game/data/sessions/fnz_ws01_20260826_164149"
WS02 = _REPO / "experiment_game/data/sessions/fnz_ws02_20260826_171537"
OUT_DIR = _REPO / "experiment_game/data/models/fnz/exp27"
REGISTRY_MD = (
    _REPO / "资料/模型训练/27_旁路_fnz被试FT_replay方案对比_openbmi_accpaper/总结/结果登记表.md"
)

LABELS = {0: "Rest", 1: "Left", 2: "Right"}
SEED = 42
MAX_PER_CLASS = 3_000  # 每类上限；共 ~9k 窗，与初轮 ablation 一致
_POOL_CACHE: Dict[str, Optional[ReplayPool]] = {}

T0_SUBJS = {
    f"openbmi:subj{n:02d}"
    for n in (
        1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28,
        29, 30, 32, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47,
    )
}
OB12_SUBJS = {
    f"openbmi:subj{n:02d}" for n in (3, 4, 6, 17, 18, 19, 32, 33, 36, 38, 44, 45)
}

TRACKS: Dict[str, Dict[str, Any]] = {
    "S": {"sessions": [WS01], "label": "ws01 单独"},
    "M": {"sessions": [WS01, WS02], "label": "ws01+ws02 合并"},
}


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    pool: str  # none | all54 | t0 | ob12 | t0_teach
    replay_ratio: float
    head_only: bool = False


ARMS: List[ArmSpec] = [
    ArmSpec("A0", "none", 0.0),
    ArmSpec("A1", "all54", 0.20),
    ArmSpec("B1", "t0", 0.05),
    ArmSpec("B2", "t0", 0.10),
    ArmSpec("B3", "t0", 0.15),
    ArmSpec("B4", "t0", 0.20),
    ArmSpec("C1", "ob12", 0.05),
    ArmSpec("C2", "ob12", 0.10),
    ArmSpec("C3", "ob12", 0.15),
    ArmSpec("D1", "t0_teach", 0.10),
    ArmSpec("D2", "t0_teach", 0.15),
    ArmSpec("E1", "none", 0.0, head_only=True),
    ArmSpec("E2", "t0", 0.10, head_only=True),
]


def _subjects_for_pool(pool: str) -> Optional[Set[str]]:
    if pool in ("none", "all54"):
        return None
    if pool in ("t0", "t0_teach"):
        return T0_SUBJS
    if pool == "ob12":
        return OB12_SUBJS
    raise ValueError(pool)


def _load_teachable_trial_ids() -> Set[int]:
    if not TEACHABLE_JSON.is_file():
        return set()
    data = json.loads(TEACHABLE_JSON.read_text(encoding="utf-8"))
    rows = data.get("trials") or data.get("records") or []
    out: Set[int] = set()
    for rec in rows:
        if rec.get("teachable"):
            out.add(int(rec["hop100_trial_id"]))
    return out


def _collect_pool_indices(
    pool: str,
    *,
    max_per_class: int = MAX_PER_CLASS,
    seed: int = SEED,
) -> Dict[int, np.ndarray]:
    root = OPENBMI_ROOT
    if pool in ("t0", "t0_teach"):
        from experiment_game.tools.openbmi_replay_pool import resolve_openbmi_root

        root = resolve_openbmi_root(prefer_t0=True)
    y = np.load(root / "openbmi_y_three.npy")
    subj = np.load(root / "openbmi_subjects.npy", allow_pickle=True)
    tid = np.load(root / "openbmi_trial_id.npy")
    allowed = _subjects_for_pool(pool)
    teachable_tids = _load_teachable_trial_ids() if pool == "t0_teach" else None

    idx_by_class: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    for i in range(len(y)):
        if allowed is not None and str(subj[i]) not in allowed:
            continue
        c = int(y[i])
        if teachable_tids is not None and c in (1, 2):
            if int(tid[i]) not in teachable_tids:
                continue
        idx_by_class[c].append(i)

    rng = np.random.default_rng(seed)
    out: Dict[int, np.ndarray] = {}
    for c in (0, 1, 2):
        arr = np.asarray(idx_by_class[c], dtype=np.int64)
        if len(arr) > max_per_class:
            rng.shuffle(arr)
            arr = arr[:max_per_class]
        out[c] = arr
    return out


def build_replay_pool(pool: str, *, seed: int = SEED) -> Optional[ReplayPool]:
    if pool == "none":
        return None
    if pool in _POOL_CACHE:
        return _POOL_CACHE[pool]
    idx_map = _collect_pool_indices(pool, seed=seed)
    if pool == "t0_teach" and len(_load_teachable_trial_ids()) == 0:
        _POOL_CACHE[pool] = None
        return None
    if any(len(idx_map[c]) == 0 for c in (0, 1, 2)):
        _POOL_CACHE[pool] = None
        return None
    from experiment_game.tools.openbmi_replay_pool import resolve_openbmi_root

    root = resolve_openbmi_root(prefer_t0=True) if pool in ("t0", "t0_teach") else OPENBMI_ROOT
    pick = np.concatenate([idx_map[c] for c in (0, 1, 2)])
    Xmm = np.load(root / "openbmi_X.npy", mmap_mode="r")
    y = np.load(root / "openbmi_y_three.npy")
    wins = np.stack([Xmm[int(i), 0].astype(np.float32) for i in pick], axis=0)
    labs = y[pick].astype(np.int64)
    rp = ReplayPool(wins, labs, seed=seed)
    _POOL_CACHE[pool] = rp
    return rp


def freeze_head_only(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = "final_layer" in name


@torch.no_grad()
def pred_distribution(model, X: np.ndarray, y: np.ndarray, device: str) -> Dict[str, Any]:
    if len(X) == 0:
        return {}
    model.eval()
    preds, logits_all = [], []
    bs = 64
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s : s + bs]).to(device)
        try:
            logits = model(xb)
        except RuntimeError:
            logits = model(xb.unsqueeze(1))
        if logits.dim() == 3:
            logits = logits.reshape(logits.shape[0], -1)
        logits_all.append(logits.cpu().numpy())
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    pred = np.concatenate(preds)
    logits = np.concatenate(logits_all)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    uniq, cnt = np.unique(pred, return_counts=True)
    counts = {LABELS[int(k)]: int(v) for k, v in zip(uniq, cnt)}
    return {
        "acc": float((pred == y).mean()),
        "pred_counts": counts,
        "mean_p": [float(x) for x in probs.mean(axis=0)],
        "max_class_frac": float(cnt.max() / len(pred)),
    }


def finetune_arm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    spec: ArmSpec,
    *,
    device: str,
) -> Dict[str, Any]:
    replay_pool = build_replay_pool(spec.pool) if spec.replay_ratio > 0 else None
    if spec.replay_ratio > 0 and replay_pool is None:
        return {
            "status": "skipped",
            "reason": f"empty replay pool for {spec.pool}",
        }

    entry = load_head(DEFAULT_THREE, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    if spec.head_only:
        freeze_head_only(model)

    acc0_tr = _eval_acc(model, X_tr, y_tr, device)
    acc0_te = _eval_acc(model, X_te, y_te, device)
    dist0 = pred_distribution(model, X_te, y_te, device)

    recipe = FTRecipe(
        lr=1e-4,
        weight_decay=1e-4,
        epochs=5,
        batch_size=32,
        replay_ratio=spec.replay_ratio,
        seed=SEED,
    )
    fin = IncrementalFinetuner(model, recipe, replay_pool=replay_pool, device=device)
    ft_rec = fin.train_round(X_tr, y_tr, frozen=False)

    acc1_tr = _eval_acc(model, X_tr, y_tr, device)
    acc1_te = _eval_acc(model, X_te, y_te, device)
    dist1 = pred_distribution(model, X_te, y_te, device)

    n_pool = int(len(replay_pool.windows)) if replay_pool is not None else 0
    return {
        "status": "ok",
        "arm": spec.arm_id,
        "pool": spec.pool,
        "replay_ratio": spec.replay_ratio,
        "head_only": spec.head_only,
        "replay_pool_windows": n_pool,
        "acc_before_train": acc0_tr,
        "acc_before_heldout": acc0_te,
        "acc_after_train": acc1_tr,
        "acc_after_heldout": acc1_te,
        "delta_heldout": acc1_te - acc0_te,
        "train_minus_heldout": acc1_tr - acc1_te,
        "heldout_before": dist0,
        "heldout_after": dist1,
        "heldout_max_class_frac": dist1.get("max_class_frac"),
        "ft": ft_rec,
    }


def _gate(row: Dict[str, Any]) -> Dict[str, bool]:
    if row.get("status") != "ok":
        return {"G1": False, "G2": False}
    ho = float(row["acc_after_heldout"])
    mx = float(row.get("heldout_max_class_frac") or 1.0)
    return {"G1": ho >= 0.40, "G2": mx < 0.60}


def rank_track(rows: Dict[str, Dict[str, Any]]) -> List[str]:
    ok = {k: v for k, v in rows.items() if v.get("status") == "ok"}

    def key(arm_id: str) -> Tuple:
        r = ok[arm_id]
        g = _gate(r)
        passed = g["G1"] and g["G2"]
        ho = float(r["acc_after_heldout"])
        mx = float(r.get("heldout_max_class_frac") or 1.0)
        gap = float(r.get("train_minus_heldout") or 0.0)
        simple = {
            "A0": 0, "C1": 1, "C2": 1, "C3": 1, "B1": 2, "B2": 2, "B3": 2, "B4": 2,
            "D1": 3, "D2": 3, "E1": 4, "E2": 4, "A1": 5,
        }.get(arm_id, 9)
        return (0 if passed else 1, -ho, mx, gap, simple)

    return sorted(ok.keys(), key=key)


def _pred_str(dist: Dict[str, Any]) -> str:
    c = dist.get("pred_counts") or {}
    return f"{c.get('Rest', 0)}/{c.get('Left', 0)}/{c.get('Right', 0)}"


def write_report_md(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 实验 27 · fnz Replay grid — 报告",
        "",
        f"生成时间：{payload['generated_at']}",
        f"teachable 试次数：{payload.get('n_teachable_trials', 0)}",
        "",
    ]
    for track in ("S", "M"):
        if track not in payload["tracks"]:
            continue
        block = payload["tracks"][track]
        lines += [
            f"## Track {track} · {block['label']}",
            "",
            f"- 训练窗 / heldout：{block['split']['n_windows_train']} / {block['split']['n_windows_heldout']}",
            "",
            "| arm | pool | ratio | ho before→after | max_frac | pred R/L/Rest | G1 | G2 |",
            "|-----|------|-------|-----------------|----------|---------------|----|----|",
        ]
        for arm_id in [a.arm_id for a in ARMS]:
            row = block["arms"].get(arm_id, {})
            if row.get("status") != "ok":
                reason = row.get("reason", "—")
                lines.append(f"| {arm_id} | — | — | **SKIP** ({reason}) | — | — | — | — |")
                continue
            g = _gate(row)
            hb = row["acc_before_heldout"]
            ha = row["acc_after_heldout"]
            dist = row.get("heldout_after") or {}
            lines.append(
                f"| {arm_id} | {row['pool']} | {row['replay_ratio']:.2f} | "
                f"{hb:.3f}→{ha:.3f} | {row.get('heldout_max_class_frac', 0):.3f} | "
                f"{_pred_str(dist)} | {'✓' if g['G1'] else '✗'} | {'✓' if g['G2'] else '✗'} |"
            )
        winner = block.get("winner")
        lines += ["", f"**Winner：`{track}-{winner}`**", ""]

    lines += ["## 跨轨", ""]
    ws = payload.get("winners", {})
    if "S" in ws:
        lines.append(f"- S winner：`S-{ws.get('S')}`")
    if "M" in ws:
        lines.append(f"- M winner：`M-{ws.get('M')}`")
    if "S" in ws and "M" in ws:
        lines.append(f"- 一致：{ws.get('S') == ws.get('M')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_registry_md(payload: Dict[str, Any]) -> None:
    if not REGISTRY_MD.is_file():
        return
    lines = REGISTRY_MD.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("| S-") or line.startswith("| M-"):
            prefix, arm = line.split("|")[1].strip().split("-", 1)
            track = prefix
            row = payload["tracks"].get(track, {}).get("arms", {}).get(arm, {})
            if row.get("status") == "ok":
                g = _gate(row)
                c = (row.get("heldout_after") or {}).get("pred_counts") or {}
                line = (
                    f"| {track}-{arm} | {row['pool']} | {row['replay_ratio']:.2f} | "
                    f"{row['acc_before_heldout']:.3f} | {row['acc_after_heldout']:.3f} | "
                    f"{row['delta_heldout']:+.3f} | {row.get('heldout_max_class_frac', 0):.3f} | "
                    f"{c.get('Rest', 0)}/{c.get('Left', 0)}/{c.get('Right', 0)} | "
                    f"{'✓' if g['G1'] else '✗'} | {'✓' if g['G2'] else '✗'} | |"
                )
        elif "**Track S winner：**" in line:
            line = f"**Track S winner：** {payload['winners'].get('S', '—')}"
        elif "**Track M winner：**" in line:
            line = f"**Track M winner：** {payload['winners'].get('M', '—')}"
        elif line.startswith("填表时间：—"):
            line = f"填表时间：{payload['generated_at']}"
        out.append(line)
        i += 1
    REGISTRY_MD.write_text("\n".join(out) + "\n", encoding="utf-8")


def run_grid(*, tracks: List[str], arms: List[str], device: str) -> Dict[str, Any]:
    arm_specs = [a for a in ARMS if a.arm_id in arms]
    payload: Dict[str, Any] = {
        "experiment": "E27",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "teachable_json": str(TEACHABLE_JSON),
        "n_teachable_trials": len(_load_teachable_trial_ids()),
        "tracks": {},
        "winners": {},
    }

    for track in tracks:
        sessions = TRACKS[track]["sessions"]
        print(f"\n======== Track {track} ({TRACKS[track]['label']}) ========", flush=True)
        ds = build_dataset(sessions, include_invalid=True)
        X, y3, split_ids = ds["X"], ds["y_three"], ds["split_id"]
        tr_m, te_m = _trial_split(split_ids, train_frac=0.7, seed=SEED)
        split_info = {
            "n_windows_train": int(tr_m.sum()),
            "n_windows_heldout": int(te_m.sum()),
            "n_trials_train": len(np.unique(split_ids[tr_m])),
            "n_trials_heldout": len(np.unique(split_ids[te_m])),
        }
        print(f"  windows train={split_info['n_windows_train']} heldout={split_info['n_windows_heldout']}")

        track_rows: Dict[str, Any] = {}
        for spec in arm_specs:
            run_id = f"{track}-{spec.arm_id}"
            print(
                f"  --- {run_id} pool={spec.pool} ratio={spec.replay_ratio} head={spec.head_only} ---",
                flush=True,
            )
            row = finetune_arm(X[tr_m], y3[tr_m], X[te_m], y3[te_m], spec, device=device)
            track_rows[spec.arm_id] = row
            if row.get("status") == "ok":
                print(
                    f"    heldout {row['acc_before_heldout']:.3f} -> {row['acc_after_heldout']:.3f}  "
                    f"max_frac={row.get('heldout_max_class_frac', 0):.3f}  "
                    f"pred={_pred_str(row.get('heldout_after') or {})}",
                    flush=True,
                )
            else:
                print(f"    SKIP: {row.get('reason')}", flush=True)

        ranked = rank_track(track_rows)
        winner = ranked[0] if ranked else None
        payload["tracks"][track] = {
            "label": TRACKS[track]["label"],
            "sessions": [p.name for p in sessions],
            "split": split_info,
            "arms": track_rows,
            "ranking": ranked,
            "winner": winner,
        }
        payload["winners"][track] = winner

    return payload


def save_winner_ckpt(track: str, winner_arm: str, *, device: str) -> Path:
    spec = next(a for a in ARMS if a.arm_id == winner_arm)
    sessions = TRACKS[track]["sessions"]
    ds = build_dataset(sessions, include_invalid=True)
    X, y3, split_ids = ds["X"], ds["y_three"], ds["split_id"]
    tr_m, te_m = _trial_split(split_ids, train_frac=0.7, seed=SEED)

    replay_pool = build_replay_pool(spec.pool) if spec.replay_ratio > 0 else None
    entry = load_head(DEFAULT_THREE, n_chans=8, n_times=N_TIMES, device=device)
    model = entry.model
    if spec.head_only:
        freeze_head_only(model)

    recipe = FTRecipe(
        lr=1e-4,
        weight_decay=1e-4,
        epochs=5,
        batch_size=32,
        replay_ratio=spec.replay_ratio,
        seed=SEED,
    )
    fin = IncrementalFinetuner(model, recipe, replay_pool=replay_pool, device=device)
    fin.train_round(X[tr_m], y3[tr_m], frozen=False)

    out_dir = OUT_DIR / f"{track}_{winner_arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "n_outputs": entry.n_outputs,
        "experiment": "E27",
        "track": track,
        "arm": winner_arm,
        "spec": spec.__dict__,
        "sessions": [p.name for p in sessions],
    }
    pt_path = out_dir / "best_three.pt"
    torch.save(ckpt, pt_path)
    meta = {
        "experiment": "E27",
        "track": track,
        "arm": winner_arm,
        "acc_heldout": _eval_acc(model, X[te_m], y3[te_m], device),
        "heldout_after": pred_distribution(model, X[te_m], y3[te_m], device),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return pt_path


def main() -> None:
    ap = argparse.ArgumentParser(description="实验 27 fnz replay grid")
    ap.add_argument("--tracks", default="S,M", help="S,M 或 S 或 M")
    ap.add_argument("--arms", default=",".join(a.arm_id for a in ARMS), help="逗号分隔 arm id")
    ap.add_argument("--save-winners", action="store_true", help="保存 report 中各轨 winner 权重")
    ap.add_argument("--merge", action="store_true", help="与已有 report.json 合并（补跑子集时用）")
    args = ap.parse_args()

    tracks = [t.strip().upper() for t in args.tracks.split(",") if t.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_winners:
        report_path = OUT_DIR / "report.json"
        if not report_path.is_file():
            print("先跑 grid 生成 report.json")
            return
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        for track, winner in payload.get("winners", {}).items():
            if winner:
                p = save_winner_ckpt(track, winner, device=device)
                print(f"Saved {track}-{winner} -> {p}")
        return

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  teachable trials: {len(_load_teachable_trial_ids())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "report.json"
    if args.merge and report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        new_payload = run_grid(tracks=tracks, arms=arms, device=device)
        for track in tracks:
            if track in new_payload["tracks"]:
                if track not in payload["tracks"]:
                    payload["tracks"][track] = new_payload["tracks"][track]
                else:
                    payload["tracks"][track]["arms"].update(new_payload["tracks"][track]["arms"])
                    ranked = rank_track(payload["tracks"][track]["arms"])
                    payload["tracks"][track]["ranking"] = ranked
                    payload["tracks"][track]["winner"] = ranked[0] if ranked else None
                payload["winners"][track] = payload["tracks"][track]["winner"]
        payload["generated_at"] = new_payload["generated_at"]
        payload["n_teachable_trials"] = new_payload["n_teachable_trials"]
    else:
        payload = run_grid(tracks=tracks, arms=arms, device=device)

    (OUT_DIR / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_report_md(payload, OUT_DIR / "report.md")

    for track, winner in payload["winners"].items():
        if winner:
            w = {
                "track": track,
                "arm": winner,
                "spec": next(a for a in ARMS if a.arm_id == winner).__dict__,
                "metrics": payload["tracks"][track]["arms"][winner],
            }
            (OUT_DIR / f"winner_{track}.json").write_text(
                json.dumps(w, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )

    update_registry_md(payload)
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    print(f"Winners: {payload['winners']}")


if __name__ == "__main__":
    main()
