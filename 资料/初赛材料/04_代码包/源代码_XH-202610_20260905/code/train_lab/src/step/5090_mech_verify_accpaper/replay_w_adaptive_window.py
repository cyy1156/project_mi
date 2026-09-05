"""方案 24 · W 腿：O1s/O2s/O3s 自适应窗长离线回放。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_PKG3 = HERE.parent / "5090_baselines_openbmi_3s_hop100_accpaper"
for p in (HERE, _PKG3):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from geometry import get_geom  # noqa: E402
from prob_dump import load_prob_dump  # noqa: E402

ARM_GEOM = {
    "O1s_m": "G1s",
    "O2s_m": "G2s",
    "O3s_m": "G3s",
}

TAU_GRID = np.round(np.arange(1.4, 4.01, 0.1), 1)
TAU_FORCE = 4.0


def _vis_sec(geom_id: str) -> float:
    g = get_geom(geom_id)
    return float(g.vis_pts) / 250.0


def _decision_tau(t0_sec: float, geom_id: str) -> float:
    return float(t0_sec) + _vis_sec(geom_id)


def _model_for_tau(tau: float) -> str:
    if tau < 2.4:
        return "O1s_m"
    if tau < 3.4:
        return "O2s_m"
    return "O3s_m"


def _index_by_trial(data: dict[str, np.ndarray]) -> dict[tuple[str, int], list[int]]:
    out: dict[tuple[str, int], list[int]] = {}
    for i in range(len(data["y"])):
        k = (str(data["subject"][i]), int(data["trial_id"][i]))
        out.setdefault(k, []).append(i)
    return out


def _trial_index_maps(
    dumps: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[tuple[str, int], list[int]]]:
    return {arm: _index_by_trial(data) for arm, data in dumps.items()}


def _pick_window_at_tau(
    data: dict[str, np.ndarray],
    idxs: list[int],
    geom_id: str,
    tau: float,
) -> int:
    best_i = idxs[0]
    best_dt = abs(_decision_tau(float(data["t0_sec"][best_i]), geom_id) - tau)
    for i in idxs:
        dt = abs(_decision_tau(float(data["t0_sec"][i]), geom_id) - tau)
        if dt < best_dt:
            best_dt = dt
            best_i = i
    return best_i


def simulate_trial(
    dumps: dict[str, dict[str, np.ndarray]],
    idx_maps: dict[str, dict[tuple[str, int], list[int]]],
    trial_key: tuple[str, int],
    *,
    tau_conf: float,
    split: str,
    fold: int,
) -> tuple[bool, float]:
    y_ref = None
    for arm, data in dumps.items():
        idx_map = idx_maps[arm]
        if trial_key not in idx_map:
            continue
        for i in idx_map[trial_key]:
            if data["split"][i] != split or int(data["fold"][i]) != fold:
                continue
            y_ref = int(data["y"][i])
            break
        if y_ref is not None:
            break
    if y_ref is None:
        return False, TAU_FORCE

    for tau in TAU_GRID:
        arm = _model_for_tau(float(tau))
        data = dumps[arm]
        idx_map = idx_maps[arm]
        if trial_key not in idx_map:
            continue
        idxs = [
            i
            for i in idx_map[trial_key]
            if data["split"][i] == split and int(data["fold"][i]) == fold
        ]
        if not idxs:
            continue
        gi = _pick_window_at_tau(data, idxs, ARM_GEOM[arm], float(tau))
        pred = int(data["pred"][gi])
        pmax = float(data["p_max"][gi])
        if pmax >= tau_conf or float(tau) >= TAU_FORCE - 1e-6:
            return pred == y_ref, float(tau)
    return False, TAU_FORCE


def eval_policy(
    dumps: dict[str, dict[str, np.ndarray]],
    idx_maps: dict[str, dict[tuple[str, int], list[int]]],
    *,
    split: str,
    tau_conf: float,
) -> dict:
    ref = dumps["O2s_m"]
    trials: set[tuple[str, int, int]] = set()
    for i in range(len(ref["y"])):
        if ref["split"][i] != split:
            continue
        trials.add(
            (str(ref["subject"][i]), int(ref["trial_id"][i]), int(ref["fold"][i]))
        )
    ok = 0
    taus: list[float] = []
    for subj, tid, fold in sorted(trials):
        hit, tau = simulate_trial(
            dumps,
            idx_maps,
            (subj, tid),
            tau_conf=tau_conf,
            split=split,
            fold=fold,
        )
        ok += int(hit)
        taus.append(tau)
    n = max(len(trials), 1)
    return {
        "acc": ok / n,
        "n_trials": len(trials),
        "mean_tau": float(np.mean(taus)) if taus else float("nan"),
    }


def grid_tau_conf(
    dumps: dict[str, dict[str, np.ndarray]],
    idx_maps: dict[str, dict[tuple[str, int], list[int]]],
    *,
    split: str,
) -> tuple[float, float]:
    best_conf, best_acc = 0.5, -1.0
    for conf in np.round(np.arange(0.30, 0.91, 0.05), 2):
        acc = eval_policy(dumps, idx_maps, split=split, tau_conf=float(conf))["acc"]
        if acc > best_acc:
            best_acc, best_conf = acc, float(conf)
    return best_conf, best_acc


def load_arm_dumps(run_dirs: dict[str, Path]) -> dict[str, dict[str, np.ndarray]]:
    from prob_dump import merge_prob_dumps

    out = {}
    for arm, rd in run_dirs.items():
        paths = sorted(rd.glob("fold*/prob_dump_three.csv"))
        if not paths:
            raise FileNotFoundError(f"no prob_dump in {rd}; run dump_probs_23.py first")
        out[arm] = merge_prob_dumps(paths)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Scheme24 W · adaptive window replay")
    p.add_argument("--o1-run", type=Path, required=True)
    p.add_argument("--o2-run", type=Path, required=True)
    p.add_argument("--o3-run", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    dumps = load_arm_dumps(
        {"O1s_m": args.o1_run, "O2s_m": args.o2_run, "O3s_m": args.o3_run}
    )
    idx_maps = _trial_index_maps(dumps)
    tau_conf, val_acc = grid_tau_conf(dumps, idx_maps, split="val")
    test = eval_policy(dumps, idx_maps, split="test", tau_conf=tau_conf)
    test_o3_only = float("nan")
    o3_summary = args.o3_run / "summary.json"
    if o3_summary.is_file():
        s3 = json.loads(o3_summary.read_text(encoding="utf-8"))
        test_o3_only = float(s3.get("test_acc_paper_mean", float("nan")))

    out = {
        "tau_conf_val": tau_conf,
        "val_acc_adaptive": val_acc,
        "test_acc_adaptive": test["acc"],
        "test_mean_tau": test["mean_tau"],
        "test_acc_o3s_m_ref": test_o3_only,
        "delta_test_pp_vs_o3": (test["acc"] - test_o3_only) * 100.0
        if test_o3_only == test_o3_only
        else None,
    }
    print(json.dumps(out, indent=2))
    out_path = args.out or (args.o3_run / "replay_w_adaptive.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
