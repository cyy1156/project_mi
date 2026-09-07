# -*- coding: utf-8 -*-
"""Exp42 · 特征对表校验（方案 v0.2 §6 风险行的落地）

回答红旗 1：syj0828 线上窗级 0.924 但 extract_features 的 probe AUC≈0.47，
到底是 (a) 切窗/锚点错位，还是 (b) 16 维手工特征族本身弱。

方法：对 3 名代表被试的最新会话，用**线上同一 build_dataset**（部署锚点）取窗，
1) 用 extract_features 的同一特征函数按试次聚合 → probe AUC（生产锚点版）；
2) 与 session_features.json 里 extract_features.py 自算的 AUC（自算锚点版）对比；
3) 部署底座四成员 stack 直推同批窗 → 窗级准确率参照。

判定：若 (1)≈(2) 且远低于 (3) → 特征族弱（非锚点 bug），C/D 的 d′ 判据降级；
     若 (1)≫(2) → extract_features 锚点错位，先修管线再谈 C/D。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (_HERE.parents[3], _HERE.parents[3] / "code", _HERE / "../../experiment_game".resolve() if False else _HERE):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

_REPO = _HERE.parents[4]  # D:/MI
for p in (_REPO, _REPO / "code"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from extract_features import _probe_auc, _trial_feat  # noqa: E402

ANALYSIS = _HERE.parents[4] / "资料" / "模型训练" / "42_旁路_真人队列混杂分解与会话特征坍塌诊断_accpaper" / "analysis_42"
CHECK_PEOPLE = ["syj0828", "fnz0828", "zyj0902"]


def _latest_session(member_id: str) -> list[Path]:
    """与线上同源：正式 v3 会话里取最新一个 ws 的全部目录。"""
    from experiment_game.tools.run_leave_next_e1f_task_ramp import (
        _list_v3_sessions,
        _session_dirs,
    )

    by_ws = _list_v3_sessions(member_id)
    key = sorted(by_ws.keys())[-1]
    return _session_dirs(by_ws, key)


def _feat_per_trial(X: np.ndarray, y: np.ndarray, split_ids: np.ndarray):
    """生产窗 X (n,8,750) → 逐试次 16 维特征均值 + 试次标签。"""
    feats: dict[str, list] = {}
    label: dict[str, int] = {}
    for i in range(len(X)):
        sid = str(split_ids[i])
        feats.setdefault(sid, []).append(_trial_feat(X[i]))
        label.setdefault(sid, int(y[i]))
    keys = sorted(feats)
    Xf = np.stack([np.mean(np.stack(feats[k], 0), 0) for k in keys])
    yf = np.array([label[k] for k in keys])
    return Xf, yf


def main() -> Path:
    from experiment_game.tools.ft_subject_from_v3 import build_dataset
    from adapt_engine.e1f import E1fRegistry, E1fStackConfig

    e1f_cfg = _REPO / "experiment_game" / "config" / "e1f_four_member.json"
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    out: dict[str, any] = {"schema": "exp42_feat_anchor_check_v1", "people": {}}

    for pid in CHECK_PEOPLE:
        try:
            hold = _latest_session(pid)
            ds = build_dataset(list(hold), include_invalid=True, protocol="auto")
        except Exception as exc:
            out["people"][pid] = {"error": str(exc)[:200]}
            print(f"[{pid}] SKIP: {exc}")
            continue
        sess_name = Path(str(hold[0])).name
        X, y, split_ids = ds["X"], ds["y_three"], ds["split_id"]
        Xf, yf = _feat_per_trial(X, y, split_ids)
        mi = np.isin(yf, (1, 2))
        auc_prod = _probe_auc(Xf[mi], yf[mi]) if mi.sum() >= 6 else float("nan")

        # 部署底座参照（不加载任何 FT overlay）
        stack = (
            E1fStackConfig.load_json(e1f_cfg, repo_root=_REPO)
            .resolve_paths(repo_root=_REPO)
        )
        reg = E1fRegistry(stack, device=device)
        probs = reg.forward_three_batch(X)
        win_acc = float((probs.argmax(1) == y).mean())

        # 自算锚点版（extract_features.py 已落盘结果，取该成员 probe_auc_lr 的均值）
        sf_path = ANALYSIS / "session_features.json"
        auc_self = float("nan")
        if sf_path.is_file():
            sf = json.loads(sf_path.read_text(encoding="utf-8"))
            aucs = [
                float(r.get("probe_auc_lr"))
                for r in sf.get("rows", [])
                if r.get("member_id") == pid
                and isinstance(r.get("probe_auc_lr"), (int, float))
            ]
            if aucs:
                auc_self = float(np.mean(aucs))

        out["people"][pid] = {
            "session": sess_name,
            "n_windows": int(len(X)),
            "n_trials": int(len(yf)),
            "probe_auc_prod_anchor": round(float(auc_prod), 3),
            "probe_auc_self_anchor": round(auc_self, 3),
            "deployed_window_acc": round(win_acc, 3),
        }
        print(f"[{pid}] {sess_name} prodAUC={auc_prod:.3f} selfAUC={auc_self:.3f} deployAcc={win_acc:.3f}")

    verdict = {
        "anchor_misalignment": any(
            (lambda d: d["probe_auc_prod_anchor"] - d["probe_auc_self_anchor"] > 0.15)(v)
            for v in out["people"].values()
        ),
        "feature_family_weak": all(
            v["probe_auc_prod_anchor"] < 0.65 and v["deployed_window_acc"] > 0.60
            for v in out["people"].values()
        ),
    }
    out["verdict"] = verdict
    path = ANALYSIS / "feat_anchor_check.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[feat-check] wrote {path} verdict={verdict}")
    return path


if __name__ == "__main__":
    main()
