"""Build Exp41 detailed per-run tables + cohort mean/std after Leave-Next."""
from __future__ import annotations

import json
import math
from pathlib import Path

_REPO = Path(r"D:\MI")
SUBJECTS_ROOT = _REPO / "experiment_game" / "data" / "subjects"
EXP41 = (
    _REPO
    / "资料"
    / "模型训练"
    / "41_旁路_真人被试LeaveNext_F5全队列统一_openbmi_accpaper"
    / "总结"
)

SKIP = {
    "_analysis",
    "_backup_old_channel_order_20260829",
    "test",
    "learn_m00",
    "fnz",
    "fnz_1",
    "cyy",
}


def prefer_summary(sid: str, *, min_stamp: str | None = None) -> Path | None:
    ft = SUBJECTS_ROOT / sid / "models" / "ft_runs"
    if not ft.is_dir():
        return None
    cands = sorted(ft.glob("*leave_next*all4*f5_summary.json"))
    if min_stamp:
        cands = [p for p in cands if p.name[:15] >= min_stamp[:15]]
    if cands:
        return cands[-1]
    any_ = sorted(ft.glob("*leave_next*f5_summary.json"))
    return any_[-1] if any_ else None


def rows_of(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, list):
        return d, {}
    for k in ("rows", "stages", "summary", "results", "ramp"):
        if isinstance(d.get(k), list):
            return d[k], d
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "r_stage" in v[0]:
            return v, d
    return [], d


def hold_key(heldout: str) -> str:
    for part in str(heldout).replace("+", "_").split("_"):
        p = part.lower()
        if p.startswith("ws") and p[2:].isdigit():
            return p
        if p.startswith("w") and not p.startswith("ws") and p[1:].isdigit():
            return p
    if "w02+w03" in str(heldout):
        return "w02+w03"
    return str(heldout)[-28:]


def lab(by: dict, name: str) -> str:
    b = by.get(name) or {}
    if not b:
        return "-"
    ok, n, acc = b.get("ok"), b.get("n"), b.get("acc")
    return f"{ok}/{n}={100 * float(acc):.1f}%" if acc is not None else f"{ok}/{n}"


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)  # sample variance
    return m, math.sqrt(var)


def win3_smooth(r: dict) -> float | None:
    v = r.get("heldout_acc_smooth")
    if v is None:
        v = r.get("heldout_acc")
    return float(v) if isinstance(v, (int, float)) else None


def win3_raw(r: dict) -> float | None:
    v = r.get("heldout_acc_raw")
    if v is None:
        f5 = r.get("f5_ft") or {}
        v = f5.get("window_acc")
    return float(v) if isinstance(v, (int, float)) else None


def f5_win3(r: dict) -> float | None:
    f5 = r.get("f5_ft") or {}
    v = f5.get("window_acc")
    return float(v) if isinstance(v, (int, float)) else None


def main(min_stamp: str | None = None) -> None:
    subjects = sorted(
        p.name for p in SUBJECTS_ROOT.iterdir() if p.is_dir() and p.name not in SKIP
    )
    all_last_mi: list[float] = []
    all_last_win_s: list[float] = []
    all_last_win_r: list[float] = []
    all_last_f5win: list[float] = []
    all_last_e1f: list[float] = []
    all_run_mi: list[float] = []
    all_run_win_s: list[float] = []
    all_run_win_r: list[float] = []
    all_run_f5win: list[float] = []
    per_subject_last: list[dict] = []

    lines = [
        "# 实验 41 · 全队列 Leave-Next 详细结果（每档 + 均值/方差）",
        "",
        f"> 生成时间戳过滤：`min_stamp={min_stamp or '最新 all4'}`  ",
        "> **口径对照**：",
        "> - **FT F5 MI** = 试次级、仅 Left+Right（Rest 不进分母）；模型仍为三分类 softmax  ",
        "> - **三分类窗 smooth / raw** = heldout 窗级三分类准确率（Rest+Left+Right）  ",
        "> - **F5 三分类窗** = `f5_ft.window_acc`（F5 读出同源窗级三分类）  ",
        "> - PASS=`release_pass`；方差 = **样本标准差**（n−1）",
        "",
    ]

    detail = ["## 1. 分被试分档明细", ""]
    for sid in subjects:
        path = prefer_summary(sid, min_stamp=min_stamp)
        if path is None:
            detail += [f"### {sid}", "", "_无 summary_", ""]
            continue
        rows, meta = rows_of(path)
        detail.append(f"### {sid}")
        detail.append("")
        detail.append(f"- JSON：`experiment_game/data/subjects/{sid}/models/ft_runs/{path.name}`")
        if meta.get("generated_at"):
            detail.append(f"- generated_at：{meta.get('generated_at')}")
        detail.append("")
        detail.append(
            "| R | hold | 三分类窗smooth | 三分类窗raw | F5三分类窗 | FT F5 MI | E1f零样本 MI | 总分 | Left | Right | Rest | PASS |"
        )
        detail.append(
            "|---|------|----------------|-------------|----------------|----------|--------------|------|------|-------|------|------|"
        )
        sub_mi, sub_ws, sub_wr, sub_fw = [], [], [], []
        for r in rows:
            f5 = r.get("f5_ft") or {}
            be = r.get("f5_base_e1f") or {}
            by = f5.get("by_label") or {}
            ws = win3_smooth(r)
            wr = win3_raw(r)
            fw = f5_win3(r)
            mi = f5.get("mi_acc")
            e1f = be.get("mi_acc")
            if isinstance(mi, (int, float)):
                sub_mi.append(float(mi))
                all_run_mi.append(float(mi))
            if ws is not None:
                sub_ws.append(ws)
                all_run_win_s.append(ws)
            if wr is not None:
                sub_wr.append(wr)
                all_run_win_r.append(wr)
            if fw is not None:
                sub_fw.append(fw)
                all_run_f5win.append(fw)
            if (
                isinstance(mi, (int, float))
                and ws is not None
                and wr is not None
                and fw is not None
                and isinstance(e1f, (int, float))
            ):
                detail.append(
                    f"| {r.get('r_stage')} | {hold_key(r.get('heldout', ''))} | "
                    f"{ws:.3f} | {wr:.3f} | {fw:.3f} | {100 * float(mi):.1f}% | "
                    f"{100 * float(e1f):.1f}% | {f5.get('score')}/{f5.get('score_max')} | "
                    f"{lab(by, 'Left')} | {lab(by, 'Right')} | {lab(by, 'Rest')} | "
                    f"{r.get('release_pass')} |"
                )
            else:
                detail.append(f"| {r.get('r_stage')} | … | incomplete |")
        m_mi, s_mi = mean_std(sub_mi)
        m_ws, s_ws = mean_std(sub_ws)
        m_wr, s_wr = mean_std(sub_wr)
        m_fw, s_fw = mean_std(sub_fw)
        detail.append("")
        detail.append(
            f"- **本被试各档**：MI mean±std = **{100 * m_mi:.1f}% ± {100 * s_mi:.1f}pp**；"
            f"三分类窗smooth = **{m_ws:.3f} ± {s_ws:.3f}**；"
            f"三分类窗raw = **{m_wr:.3f} ± {s_wr:.3f}**；"
            f"F5三分类窗 = **{m_fw:.3f} ± {s_fw:.3f}**（n={len(sub_mi)}）"
        )
        detail.append("")
        if rows:
            last = rows[-1]
            f5 = last.get("f5_ft") or {}
            be = last.get("f5_base_e1f") or {}
            last_mi = float(f5["mi_acc"])
            last_ws = win3_smooth(last)
            last_wr = win3_raw(last)
            last_fw = f5_win3(last)
            last_e1f = float((be or {}).get("mi_acc") or float("nan"))
            all_last_mi.append(last_mi)
            if last_ws is not None:
                all_last_win_s.append(last_ws)
            if last_wr is not None:
                all_last_win_r.append(last_wr)
            if last_fw is not None:
                all_last_f5win.append(last_fw)
            if last_e1f == last_e1f:
                all_last_e1f.append(last_e1f)
            per_subject_last.append(
                {
                    "subject_id": sid,
                    "n_rounds": len(rows),
                    "hold": hold_key(last.get("heldout", "")),
                    "win_smooth": last_ws,
                    "win_raw": last_wr,
                    "f5_win3": last_fw,
                    "win": last_ws,  # alias for older consumers
                    "mi": last_mi,
                    "e1f_mi": last_e1f,
                    "pass": last.get("release_pass"),
                    "score": f5.get("score"),
                    "score_max": f5.get("score_max"),
                    "run_mi_mean": m_mi,
                    "run_mi_std": s_mi,
                    "run_win_smooth_mean": m_ws,
                    "run_win_smooth_std": s_ws,
                    "run_win_raw_mean": m_wr,
                    "run_win_raw_std": s_wr,
                    "summary": path.name,
                }
            )

    # cohort tables
    ranked = sorted(per_subject_last, key=lambda x: -x["mi"])
    m_lmi, s_lmi = mean_std(all_last_mi)
    m_lws, s_lws = mean_std(all_last_win_s)
    m_lwr, s_lwr = mean_std(all_last_win_r)
    m_lfw, s_lfw = mean_std(all_last_f5win)
    m_le, s_le = mean_std(all_last_e1f)
    m_rmi, s_rmi = mean_std(all_run_mi)
    m_rws, s_rws = mean_std(all_run_win_s)
    m_rwr, s_rwr = mean_std(all_run_win_r)
    m_rfw, s_rfw = mean_std(all_run_f5win)

    summary = [
        "## 0. 队列汇总（均值 ± 样本标准差）",
        "",
        "| 统计口径 | n | 均值 | 标准差 |",
        "|----------|---|------|--------|",
        f"| **末档 FT F5 MI**（每被试 1 个，试次 L+R） | {len(all_last_mi)} | **{100 * m_lmi:.1f}%** | **{100 * s_lmi:.1f}pp** |",
        f"| **末档 三分类窗 smooth** | {len(all_last_win_s)} | **{m_lws:.3f}** | **{s_lws:.3f}** |",
        f"| **末档 三分类窗 raw** | {len(all_last_win_r)} | **{m_lwr:.3f}** | **{s_lwr:.3f}** |",
        f"| 末档 F5 三分类窗 | {len(all_last_f5win)} | {m_lfw:.3f} | {s_lfw:.3f} |",
        f"| 末档 E1f 零样本 MI | {len(all_last_e1f)} | {100 * m_le:.1f}% | {100 * s_le:.1f}pp |",
        f"| **全部档次 FT F5 MI**（所有 R 合并） | {len(all_run_mi)} | {100 * m_rmi:.1f}% | {100 * s_rmi:.1f}pp |",
        f"| 全部档次 三分类窗 smooth | {len(all_run_win_s)} | {m_rws:.3f} | {s_rws:.3f} |",
        f"| 全部档次 三分类窗 raw | {len(all_run_win_r)} | {m_rwr:.3f} | {s_rwr:.3f} |",
        f"| 全部档次 F5 三分类窗 | {len(all_run_f5win)} | {m_rfw:.3f} | {s_rfw:.3f} |",
        "",
        f"- 末档 PASS：{sum(1 for x in per_subject_last if x.get('pass'))}/{len(per_subject_last)}",
        "",
        "## 0.1 末档排名",
        "",
        "| 名次 | 被试 | hold | 三分类窗smooth | 三分类窗raw | F5三分类窗 | FT MI | E1f MI | 本被试各档 MI mean±std | PASS |",
        "|------|------|------|----------------|-------------|----------------|-------|--------|-------------------------|------|",
    ]
    for i, x in enumerate(ranked, 1):
        summary.append(
            f"| {i} | **{x['subject_id']}** | {x['hold']} | "
            f"{x['win_smooth']:.3f} | {x['win_raw']:.3f} | {x['f5_win3']:.3f} | "
            f"**{100 * x['mi']:.1f}%** | {100 * x['e1f_mi']:.1f}% | "
            f"{100 * x['run_mi_mean']:.1f}% ± {100 * x['run_mi_std']:.1f}pp | "
            f"{x['pass']} |"
        )
    summary.append("")

    out = EXP41 / "详细结果_全队列分档与均值方差.md"
    text = "\n".join(lines + summary + detail) + "\n"
    out.write_text(text, encoding="utf-8")

    # also json for machine
    payload = {
        "min_stamp": min_stamp,
        "cohort_last_mi_mean": m_lmi,
        "cohort_last_mi_std": s_lmi,
        "cohort_last_win3_smooth_mean": m_lws,
        "cohort_last_win3_smooth_std": s_lws,
        "cohort_last_win3_raw_mean": m_lwr,
        "cohort_last_win3_raw_std": s_lwr,
        "cohort_last_f5_win3_mean": m_lfw,
        "cohort_last_f5_win3_std": s_lfw,
        "cohort_last_win_mean": m_lws,  # alias
        "cohort_last_win_std": s_lws,
        "cohort_all_runs_mi_mean": m_rmi,
        "cohort_all_runs_mi_std": s_rmi,
        "cohort_all_runs_win3_smooth_mean": m_rws,
        "cohort_all_runs_win3_smooth_std": s_rws,
        "cohort_all_runs_win3_raw_mean": m_rwr,
        "cohort_all_runs_win3_raw_std": s_rwr,
        "subjects": per_subject_last,
    }
    (EXP41 / "detailed_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", out)
    print(
        f"last_mi {100 * m_lmi:.1f}% ± {100 * s_lmi:.1f}pp | "
        f"win3_s {m_lws:.3f}±{s_lws:.3f} | win3_r {m_lwr:.3f}±{s_lwr:.3f} | "
        f"n_subj={len(per_subject_last)}"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--min-stamp",
        default=None,
        help="只采纳文件名 stamp >= 此值的 summary，如 20260904_121700",
    )
    args = ap.parse_args()
    main(min_stamp=args.min_stamp)
