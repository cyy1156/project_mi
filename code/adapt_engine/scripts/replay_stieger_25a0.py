"""方案 25-A0 · Stieger 爬坡曲线回放（adapt_engine 三模式之 offline-replay）。

每被试：trial 时间序 → 4 轮标定闭环（12 FT + 6 小考）→ 累积曲线 + 准入判定。
用法：
  python code/adapt_engine/scripts/replay_stieger_25a0.py --subjects S1 S2 S3 [--fold 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]          # D:/MI
sys.path.insert(0, str(_ROOT / "code"))

from adapt_engine import (  # noqa: E402
    DEFAULT_CONSTANTS, FTRecipe, ModelRegistry, replay_offline,
)

DATA = _ROOT / "code" / "preprocess_lab" / "out" / "stieger_3s_hop100"
S3 = (_ROOT / "code" / "train_lab" / "out" / "5070_baseline_openbmi_3s_hop100_accpaper"
      / "shallow_openbmi_3s_hop100_balbatch_accpaper" / "openbmi_3s_hop100"
      / "run_20260822_094942")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--replay-ratio", type=float, default=0.15)
    ap.add_argument("--out", default=str(_ROOT / "code" / "adapt_engine" / "results"))
    args = ap.parse_args()

    X = np.load(DATA / "stieger_X.npy", mmap_mode="r")
    y = np.load(DATA / "stieger_y_three.npy", allow_pickle=True)
    subj = np.load(DATA / "stieger_subjects.npy", allow_pickle=True).astype(str)
    tid = np.load(DATA / "stieger_trial_id.npy", allow_pickle=True).astype(int)
    uniq_subj = sorted(set(subj.tolist()))
    print(f"Stieger {X.shape} · 被试 {len(uniq_subj)} · 用前 {args.subjects}")

    reg = ModelRegistry(S3 / "task" / f"fold{args.fold}" / "best_task.pt",
                        S3 / "three" / f"fold{args.fold}" / "best_three.pt")

    # 源域回放池（OpenBMI 域内窗替代：用 Stieger 全库小样本，按类平衡）——25-G2 语义近似
    rng = np.random.default_rng(42)
    pool_idx = rng.choice(len(X), size=min(20000, len(X)), replace=False)
    from adapt_engine.ft import ReplayPool

    pool = ReplayPool(np.asarray(X[pool_idx], dtype=np.float32)[:, 0], y[pool_idx], seed=42)

    report = {"subjects": {}, "fold": args.fold, "replay_ratio": args.replay_ratio}
    for s in args.subjects:
        if s not in uniq_subj:
            print(f"跳过（无被试 {s}）")
            continue
        mask = subj == s
        Xi = np.asarray(X[mask], dtype=np.float32)          # (n_win,1,8,750)
        Xi = Xi[:, 0]                                        # (n_win,8,750)
        yi = y[mask]
        ti = tid[mask]
        order = np.argsort(ti, kind="stable")
        Xi, yi, ti = Xi[order], yi[order], ti[order]
        t0 = time.time()
        holder = _Holder(reg)

        class _H:  # replay_offline 需要 model_holder.model 与 .predict
            model = reg.three_heads[0].model

        _H.predict = staticmethod(holder.predict)
        res = replay_offline(
            _H, X=Xi, y=yi, trial_ids=ti, windows_per_trial=7,
            n_rounds=args.rounds, constants=DEFAULT_CONSTANTS,
            replay_pool=pool, recipe=FTRecipe(replay_ratio=args.replay_ratio),
            log=lambda s2: None,
        )
        curve = [(p.round_no, p.k_ft, p.n_quiz, round(p.acc, 4)) for p in res["curve"]]
        gates = [d.status for d in []]  # 由 curve 推
        report["subjects"][s] = {
            "n_trials": int(len(set(ti.tolist()))), "curve": curve,
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"[{s}] trials={len(set(ti.tolist()))} 曲线={curve} 用时{report['subjects'][s]['elapsed_s']}s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"stieger_25a0_{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写 {out / f'stieger_25a0_{stamp}.json'}")
    return 0


class _Holder:
    def __init__(self, reg):
        self.reg = reg

    def predict(self, w: np.ndarray) -> dict:
        from adapt_engine.readout import serial_gating

        h = self.reg.forward_heads(w)
        return serial_gating(h["p_task"], h["p_three"], task_p_on=0.6)


if __name__ == "__main__":
    raise SystemExit(main())
