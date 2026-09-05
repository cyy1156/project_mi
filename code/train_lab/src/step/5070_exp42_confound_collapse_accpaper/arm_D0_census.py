# -*- coding: utf-8 -*-
"""Exp42 · D0 坍塌普查臂（方案 v0.2 §2 D0）——纯解析实现。

数据源：每名被试 `models/ft_runs/*_leave_next_*_f5_summary.json`（Leave-Next
逐轮重放落盘）。每行 r_stage 自带 FT 后 heldout 的 F5 读出统计：
  - f5_ft.pred_hist：试次级预测类分布 → heldout max_class_frac
  - f5_ft.confusion：y→p 计数 → MI 试次异侧率
  - f5_ft / f5_base_e1f：FT 后与冻结底座对照
发生率定义（v0.2 冻结）：试次级预测分布熵 <0.5 或 heldout max_class_frac ≥0.8
或 MI 试次异侧率 ≥0.7，任一成立记"坍塌轮"。
（注：熵以试次级 pred_hist 分布计算，非窗级概率熵——免前向的等价替代，偏差已登记。）
决策门：末轮坍塌占比 CI 上界 <20% → H0' 否（D 臂缩个案）；下界 ≥40% → 全量展开。
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[4]
SUBJECTS = _REPO / "experiment_game" / "data" / "subjects"

from cohort_map import build_cohort_map  # noqa: E402

ANALYSIS = _HERE.parents[4] / "资料" / "模型训练" / "42_旁路_真人队列混杂分解与会话特征坍塌诊断_accpaper" / "analysis_42"

ENT_THR = 0.5
MCF_THR = 0.8
WRONGSIDE_THR = 0.7
_LAB = {"Rest": 0, "Left": 1, "Right": 2}


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import beta

        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
        return (lo, hi)
    except Exception:
        p = k / n
        se = math.sqrt(p * (1 - p) / n)
        return (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))


def _active_summary(pid: str) -> Path | None:
    ft = SUBJECTS / pid / "models" / "ft_runs"
    if not ft.is_dir():
        return None
    idx = ft / "FT_RUNS_INDEX.json"
    if idx.is_file():
        try:
            act = json.loads(idx.read_text(encoding="utf-8")).get("active_summary")
            if act:
                p = ft / str(act)
                if p.is_file():
                    return p
        except Exception:
            pass
    cands = sorted(ft.glob("*_summary.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    f5 = row.get("f5_ft") or {}
    hist = f5.get("pred_hist") or {}
    total = sum(hist.values())
    counts = [hist.get(k, 0) for k in ("Rest", "Left", "Right")]
    mcf = (max(counts) / total) if total else float("nan")
    ent = float("nan")
    if total:
        ent = -sum((c / total) * math.log(c / total) for c in counts if c) / math.log(3)
    conf = f5.get("confusion") or {}
    same = conf.get("y1->p1", 0) + conf.get("y2->p2", 0)
    cross = conf.get("y1->p2", 0) + conf.get("y2->p1", 0)
    wrong = (cross / (same + cross)) if (same + cross) else float("nan")
    base = row.get("f5_base_e1f") or {}
    return {
        "r_stage": row.get("r_stage"),
        "heldout": row.get("heldout"),
        "release_pass": row.get("release_pass"),
        "heldout_acc_smooth": row.get("heldout_acc"),
        "mi_acc_f5": f5.get("mi_acc"),
        "train_mcf": row.get("max_class_frac"),
        "norm_entropy_trialdist": round(ent, 3) if ent == ent else None,
        "max_class_frac": round(mcf, 3) if mcf == mcf else None,
        "mi_wrong_side_rate": round(wrong, 3) if wrong == wrong else None,
        "base_e1f_mi_acc": base.get("mi_acc"),
        "collapse": bool(
            (ent == ent and ent < ENT_THR)
            or (mcf == mcf and mcf >= MCF_THR)
            or (wrong == wrong and wrong >= WRONGSIDE_THR)
        ),
    }


def main() -> Path:
    cohort = build_cohort_map()
    rows: list[dict[str, Any]] = []
    for person in cohort["people"]:
        pid = person["primary_id"]
        sp = _active_summary(pid)
        person_rounds: list[dict[str, Any]] = []
        if sp:
            data = json.loads(sp.read_text(encoding="utf-8"))
            for row in data.get("rows", []):
                if not row.get("heldout"):
                    continue
                met = _row_metrics(row)
                met.update({"summary": sp.name})
                person_rounds.append(met)
        rows.append({"person": pid, "summary": sp.name if sp else None, "rounds": person_rounds})

    for r in rows:
        ok = r["rounds"]
        r["n_rounds"] = len(ok)
        r["n_collapse"] = sum(1 for x in ok if x["collapse"])
        r["label"] = "stable" if r["n_collapse"] >= 2 else ("occasional" if r["n_collapse"] == 1 else "none")
        r["last_round_collapse"] = bool(ok[-1]["collapse"]) if ok else None

    last = [r for r in rows if r["last_round_collapse"] is not None]
    k = sum(1 for r in last if r["last_round_collapse"])
    n = len(last)
    lo, hi = _clopper_pearson(k, n)
    gate = {
        "last_round_k": k,
        "last_round_n": n,
        "frac": round(k / n, 3) if n else None,
        "ci95": [round(lo, 3), round(hi, 3)] if n else None,
        "verdict": (
            "H0_rejected_collapse_rare" if n and hi < 0.20
            else ("collapse_common_full_D" if n and lo >= 0.40 else "intermediate_case_by_case")
        ),
    }
    out = {
        "schema": "exp42_arm_D0_v2_offline",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "definition": {"ent_thr": ENT_THR, "mcf_thr": MCF_THR, "wrongside_thr": WRONGSIDE_THR,
                        "entropy_note": "trial-level pred_hist distribution entropy (no-forward surrogate)"},
        "gate": gate,
        "people": rows,
    }
    path = ANALYSIS / "arm_D0.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[D0] wrote {path}")
    print(f"[D0] GATE: {k}/{n} last-round collapse, CI95=[{lo:.3f},{hi:.3f}] -> {gate['verdict']}")
    for r in rows:
        print(f"  {r['person']:<10s} rounds={r['n_rounds']} collapse={r['n_collapse']} "
              f"label={r['label']:<10s} last={r['last_round_collapse']}")
    return path


if __name__ == "__main__":
    main()
