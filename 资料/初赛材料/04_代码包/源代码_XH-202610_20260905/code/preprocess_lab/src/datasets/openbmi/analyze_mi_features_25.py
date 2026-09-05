"""OpenBMI · MI 特征显著性分析（与 hop100 切窗几何对齐，无窗内 z-score）。

切段（与 preprocess openbmi_2s_hop100 一致）：
  - MI：Cue 后 0–4 s（Cue 前 0.5 s 基线校正）→ 2 s / 100 ms 滑窗至段尽
  - Rest：Cue 前 4 s（可缩短避让上一 MI）→ 同上滑窗
  - 语料：仅 OpenBMI ``EEG_MI_train``（sess01+sess02；**不含**官方 EEG_MI_test）
  - CAR + notch50 + bp8–30；**不做**窗内 z-score（保绝对功率 ERD）

用法（在 preprocess_lab 下）:
  python -m src.datasets.openbmi.analyze_mi_features_25
  python -m src.datasets.openbmi.analyze_mi_features_25 --subjects 01,02,03
  # 默认：全部 54 名被试；仅 EEG_MI_train（不读 EEG_MI_test）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRE_ROOT = HERE.parents[2]
REPO = PRE_ROOT.parent.parent
if str(PRE_ROOT) not in sys.path:
    sys.path.insert(0, str(PRE_ROOT))

from src.common.steps.epoch_baseline import task_window_cue_0_to_4
from src.common.steps.filter_car import car_reference, notch_and_bandpass
from src.common.steps.select_channels import select_channels
from src.common.steps.slide_1s import extract_segment_baseline, iter_rest_sources_cue_before
from src.common.steps.slide_2s_hop100 import (
    HOP_SEC,
    WIN_SEC,
    segment_to_2s_hop100_windows,
)
from src.datasets.bci2a.labels import filter_left_right_events
from src.datasets.openbmi.load_mat import load_openbmi_mat, parse_sess_subj, subject_key

CHANS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]
IX = {n: i for i, n in enumerate(CHANS)}
FS_OUT = 250.0
# 原始 mat：{REPO}/DATA/openbmi/sess*_subj*_EEG_MI.mat（与 batch_3s_hop100 一致）
MAT_GLOB = REPO / "DATA" / "openbmi"

STANDARDS = {
    "name": "OpenBMI_MI_feature_v2_hop100",
    "epoch": (
        "Rest=cue前4s(可缩短); MI=cue后0–4s; "
        "slide Tw=2s hop=100ms → 窗@250Hz; CAR+notch50+bp8-30; no z-score"
    ),
    "mu_hz": (8.0, 13.0),
    "beta_l_hz": (13.0, 20.0),
    "beta_h_hz": (20.0, 30.0),
    "mu_erd_contra_ok": -15.0,
    "mu_erd_contra_excellent": -35.0,
    "laterality_pp_ok": 8.0,
    "mu_vs_betal_slack": 5.0,
    "rest_mu_frac_ok": 0.40,
    "rest_mu_frac_excellent": 0.55,
    # 时间：按窗起点相对 cue 的 hop 曲线（0,0.1,…s）
    "time_drop_ok": 0.08,
}


@dataclass
class SideMetrics:
    n_mi_trials: int
    n_rest_trials: int
    n_mi_windows: int
    n_rest_windows: int
    mu_erd_c3: float
    mu_erd_c4: float
    mu_erd_cp3: float
    mu_erd_cp4: float
    mu_erd_contra: float
    mu_erd_ipsi: float
    laterality_pp: float
    betal_erd_contra: float
    betah_erd_contra: float
    rest_mu_frac: float
    time_onset_ok: bool
    time_trough_s: float
    time_drop: float
    c3c4_corr_rest: float
    c3c4_corr_mi: float
    corr_drop: float


def _bandpowers_fft(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x (T,C) → Mu / βL / βH 功率 (C,)，rFFT 均值。"""
    t = x.shape[0]
    freqs = np.fft.rfftfreq(t, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x.astype(np.float64), axis=0)) ** 2
    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        if not np.any(m):
            return np.full(x.shape[1], 1e-12)
        return spec[m].mean(axis=0) + 1e-12
    mu = STANDARDS["mu_hz"]
    bl = STANDARDS["beta_l_hz"]
    bh = STANDARDS["beta_h_hz"]
    return band(*mu), band(*bl), band(*bh)


def _erd(p_task: float, p_rest: float) -> float:
    return 100.0 * (p_task - p_rest) / (p_rest + 1e-12)


def _stack_band(wins: list[np.ndarray], fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """list[(T,C)] → (N,C) for mu/bl/bh."""
    mus, bls, bhs = [], [], []
    for w in wins:
        a, b, c = _bandpowers_fft(w, fs)
        mus.append(a)
        bls.append(b)
        bhs.append(c)
    return np.stack(mus, 0), np.stack(bls, 0), np.stack(bhs, 0)


def _side_metrics(
    mi_wins: list[np.ndarray],
    rest_wins: list[np.ndarray],
    mi_trials: list[list[np.ndarray]],
    rest_trials: list[list[np.ndarray]],
    *,
    left: bool,
    fs: float,
) -> SideMetrics | None:
    if len(mi_wins) < 20 or len(rest_wins) < 20:
        return None

    rest_mu, rest_bl, rest_bh = _stack_band(rest_wins, fs)
    mi_mu, mi_bl, mi_bh = _stack_band(mi_wins, fs)
    r_mu, r_bl, r_bh = rest_mu.mean(0), rest_bl.mean(0), rest_bh.mean(0)
    m_mu, m_bl, m_bh = mi_mu.mean(0), mi_bl.mean(0), mi_bh.mean(0)

    def erd_ch(i: int) -> float:
        return _erd(float(m_mu[i]), float(r_mu[i]))

    c3, c4 = IX["C3"], IX["C4"]
    cp3, cp4 = IX["CP3"], IX["CP4"]
    if left:
        contra, ipsi = c4, c3
    else:
        contra, ipsi = c3, c4

    mu_contra = erd_ch(contra)
    mu_ipsi = erd_ch(ipsi)
    laterality = mu_ipsi - mu_contra
    betal_contra = _erd(float(m_bl[contra]), float(r_bl[contra]))
    betah_contra = _erd(float(m_bh[contra]), float(r_bh[contra]))
    rest_tot = r_mu + r_bl + r_bh
    rest_mu_frac = float(np.mean(r_mu / (rest_tot + 1e-12)))

    # 时间：各 trial 按 hop 起点的对侧 Mu 功率 → 跨 trial 平均
    max_h = max((len(tr) for tr in mi_trials), default=0)
    hop_p = []
    for h in range(max_h):
        ps = []
        for tr in mi_trials:
            if h < len(tr):
                mu_p, _, _ = _bandpowers_fft(tr[h], fs)
                ps.append(float(mu_p[contra]))
        if ps:
            hop_p.append(float(np.mean(ps)))
    hop_p = np.asarray(hop_p, dtype=float)
    times = np.arange(len(hop_p)) * float(HOP_SEC)
    if len(hop_p) >= 5:
        pre = float(np.mean(hop_p[times < 0.3]) + 1e-12)
        mid_m = (times >= 0.4) & (times <= 0.9)
        mid = float(np.mean(hop_p[mid_m]) if np.any(mid_m) else hop_p.mean())
        time_drop = (pre - mid) / pre
        trough_s = float(times[int(np.argmin(hop_p))])
        time_onset_ok = time_drop >= float(STANDARDS["time_drop_ok"]) and 0.7 <= trough_s <= 2.0
    else:
        time_drop, trough_s, time_onset_ok = 0.0, float("nan"), False

    def mean_corr(wins: list[np.ndarray]) -> float:
        cs = []
        # 子采样加速
        step = max(1, len(wins) // 400)
        for e in wins[::step]:
            a, b = e[:, c3], e[:, c4]
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                continue
            cs.append(float(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(cs)) if cs else float("nan")

    cr = mean_corr(rest_wins)
    cm = mean_corr(mi_wins)
    return SideMetrics(
        n_mi_trials=len(mi_trials),
        n_rest_trials=len(rest_trials),
        n_mi_windows=len(mi_wins),
        n_rest_windows=len(rest_wins),
        mu_erd_c3=erd_ch(c3),
        mu_erd_c4=erd_ch(c4),
        mu_erd_cp3=erd_ch(cp3),
        mu_erd_cp4=erd_ch(cp4),
        mu_erd_contra=mu_contra,
        mu_erd_ipsi=mu_ipsi,
        laterality_pp=laterality,
        betal_erd_contra=betal_contra,
        betah_erd_contra=betah_contra,
        rest_mu_frac=rest_mu_frac,
        time_onset_ok=bool(time_onset_ok),
        time_trough_s=trough_s,
        time_drop=float(time_drop),
        c3c4_corr_rest=cr,
        c3c4_corr_mi=cm,
        corr_drop=float(cr - cm) if np.isfinite(cr) and np.isfinite(cm) else float("nan"),
    )


def score_side(m: SideMetrics) -> dict:
    s = STANDARDS
    checks = {
        "mu_erd_contra": m.mu_erd_contra <= s["mu_erd_contra_ok"],
        "mu_erd_excellent": m.mu_erd_contra <= s["mu_erd_contra_excellent"],
        "laterality": m.laterality_pp >= s["laterality_pp_ok"],
        "mu_vs_betal": m.mu_erd_contra <= m.betal_erd_contra + s["mu_vs_betal_slack"],
        "rest_mu_frac": m.rest_mu_frac >= s["rest_mu_frac_ok"],
        "time_pattern": m.time_onset_ok,
    }
    core = ["mu_erd_contra", "laterality", "mu_vs_betal", "rest_mu_frac", "time_pattern"]
    passed = sum(1 for k in core if checks[k])
    rate = passed / len(core)
    if rate >= 0.8 and checks["mu_erd_contra"]:
        grade = "明显"
    elif rate >= 0.5:
        grade = "中等"
    else:
        grade = "弱/不明显"
    return {"checks": checks, "passed": passed, "n_core": len(core), "rate": rate, "grade": grade}


def collect_epochs_for_mat(
    mat_path: Path,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[list[np.ndarray]],
    list[list[np.ndarray]],
    list[list[np.ndarray]],
]:
    """left/right/rest 窗列表 + 按 trial 分组的窗列表。"""
    runs = load_openbmi_mat(mat_path, blocks=("EEG_MI_train",))
    left_w, right_w, rest_w = [], [], []
    left_tr, right_tr, rest_tr = [], [], []
    for eeg in runs:
        x = select_channels(eeg.x, eeg.ch_names)
        x = car_reference(x)
        x = notch_and_bandpass(x, eeg.fs)
        fs = float(eeg.fs)
        kept = filter_left_right_events(eeg.events, eeg.artifacts)
        for cue, _lt, lab3, _ in kept:
            seg = task_window_cue_0_to_4(x, int(cue), fs)
            if seg is None:
                continue
            wins = segment_to_2s_hop100_windows(seg, fs, zscore=False)
            if not wins:
                continue
            if int(lab3) == 1:
                left_w.extend(wins)
                left_tr.append(wins)
            elif int(lab3) == 2:
                right_w.extend(wins)
                right_tr.append(wins)

        sources = iter_rest_sources_cue_before(
            kept[:, 0],
            fs,
            x.shape[0],
            rest_sec=4.0,
            task_sec=4.0,
            min_win_sec=WIN_SEC,
        )
        n_left = int(np.sum(kept[:, 2] == 1))
        n_right = int(np.sum(kept[:, 2] == 2))
        max_rest = min(n_left, n_right) if (n_left + n_right) else 0
        sources = sources[: int(max_rest)]
        for t0, t1 in sources:
            seg = extract_segment_baseline(x, int(t0), int(t1), fs, baseline_sec=0.5)
            if seg is None:
                continue
            wins = segment_to_2s_hop100_windows(seg, fs, zscore=False)
            if not wins:
                continue
            rest_w.extend(wins)
            rest_tr.append(wins)
    return left_w, right_w, rest_w, left_tr, right_tr, rest_tr


def analyze_subject(subj: str, mats: list[Path]) -> dict:
    left_w, right_w, rest_w = [], [], []
    left_tr, right_tr, rest_tr = [], [], []
    for mpath in mats:
        L, R, Re, Lt, Rt, Ret = collect_epochs_for_mat(mpath)
        left_w.extend(L)
        right_w.extend(R)
        rest_w.extend(Re)
        left_tr.extend(Lt)
        right_tr.extend(Rt)
        rest_tr.extend(Ret)

    sk = f"subj{subj}" if not str(subj).startswith("subj") else str(subj)
    out: dict = {
        "subject": subject_key(sk),
        "n_mats": len(mats),
        "n_left": len(left_w),
        "n_right": len(right_w),
        "n_rest": len(rest_w),
        "n_left_trials": len(left_tr),
        "n_right_trials": len(right_tr),
        "n_rest_trials": len(rest_tr),
    }
    m_l = _side_metrics(left_w, rest_w, left_tr, rest_tr, left=True, fs=FS_OUT)
    m_r = _side_metrics(right_w, rest_w, right_tr, rest_tr, left=False, fs=FS_OUT)
    for tag, m in (("left", m_l), ("right", m_r)):
        if m is None:
            out[tag] = None
            continue
        out[tag] = {"metrics": asdict(m), "score": score_side(m)}
    rates = [out[t]["score"]["rate"] for t in ("left", "right") if out.get(t)]
    out["mean_pass_rate"] = float(np.mean(rates)) if rates else 0.0
    if out["mean_pass_rate"] >= 0.8:
        out["subject_grade"] = "明显"
    elif out["mean_pass_rate"] >= 0.5:
        out["subject_grade"] = "中等"
    else:
        out["subject_grade"] = "弱/不明显"
    return out


def write_report(results: list[dict], out_md: Path, out_json: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"standards": STANDARDS, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    n = len(results)
    lines = [
        f"# OpenBMI · {n} 被试 MI 特征显著性分析",
        "",
        f"- 生成时间：`{datetime.now().isoformat(timespec='seconds')}`",
        f"- 数据：`{MAT_GLOB}` · **仅 `EEG_MI_train`**（不含 `EEG_MI_test`）· sess01+sess02 同人合并",
        f"- 通道：`{CHANS}`（与 hop100 训练一致）",
        "- 参考文档：`脑电特征提取指标与模板量化分析.docx`",
        f"- 分析标准版本：`{STANDARDS['name']}`",
        "",
        "## 0. 切段与重定标准",
        "",
        f"1. **切段（对齐 hop100）**：{STANDARDS['epoch']}",
        "2. **不用**窗内 z-score；从 mat 独立重切。",
        "3. **合格阈值**：",
        "",
        "| 维度 | OpenBMI 合格线 | 原文档优秀线（对照） |",
        "|------|----------------|----------------------|",
        f"| 对侧 Mu ERD | ≤ {STANDARDS['mu_erd_contra_ok']}%（优秀≤{STANDARDS['mu_erd_contra_excellent']}%） | −50%~−65% |",
        f"| 偏侧性 laterality | ≥ {STANDARDS['laterality_pp_ok']} 百分点 | 对侧远强于同侧 |",
        "| Mu vs 低频 Beta | Mu ERD 不弱于 BetaL（+5pp） | Mu 降幅最大 |",
        f"| 静息 Mu 占比 | ≥ {STANDARDS['rest_mu_frac_ok']} | ≥0.60 |",
        "| 时间形态 | hop 曲线：早段→0.4–0.9s 降≥8%，谷底∈[0.7,2.0]s | 0.5s 起降等 |",
        "",
        "计分：左右各 5 项；≥0.8 明显，≥0.5 中等，否则弱。被试总评=左右平均。",
        "左手核心 **C4/CP4**；右手 **C3/CP3**。`nL/nR/nRest` 为 **窗数**（每 MI 试次约 21 窗）。",
        "",
        f"## 1. 总表（{n} 被试）",
        "",
        "| 被试 | 总评 | 通过率 | 左Mu对侧ERD% | 左偏侧pp | 左评级 | 右Mu对侧ERD% | 右偏侧pp | 右评级 | nL/nR/nRest窗 |",
        "|------|------|--------|--------------|----------|--------|--------------|----------|--------|---------------|",
    ]
    for r in results:
        def cell(side, key, fmt=".1f"):
            if not r.get(side):
                return "NA"
            return format(r[side]["metrics"][key], fmt)

        def grade(side):
            return r[side]["score"]["grade"] if r.get(side) else "NA"

        lines.append(
            f"| {r['subject']} | **{r['subject_grade']}** | {r['mean_pass_rate']:.2f} | "
            f"{cell('left','mu_erd_contra')} | {cell('left','laterality_pp')} | {grade('left')} | "
            f"{cell('right','mu_erd_contra')} | {cell('right','laterality_pp')} | {grade('right')} | "
            f"{r['n_left']}/{r['n_right']}/{r['n_rest']} |"
        )

    gc = Counter(r["subject_grade"] for r in results)
    lines += ["", "### 总评分布", ""]
    for k in ("明显", "中等", "弱/不明显"):
        lines.append(f"- **{k}**：{gc.get(k, 0)} / {n}")
    lines += ["", "## 2. 分被试明细", ""]
    for r in results:
        lines.append(f"### {r['subject']} · **{r['subject_grade']}**（通过率 {r['mean_pass_rate']:.2f}）")
        lines.append("")
        lines.append(
            f"- 窗数 Left={r['n_left']} Right={r['n_right']} Rest={r['n_rest']} · "
            f"试次 L/R/Rest={r.get('n_left_trials')}/{r.get('n_right_trials')}/{r.get('n_rest_trials')} · mats={r['n_mats']}"
        )
        for side, title in (("left", "左手 MI（期望 C4）"), ("right", "右手 MI（期望 C3）")):
            block = r.get(side)
            if not block:
                lines.append(f"- {title}：样本不足")
                continue
            m, sc = block["metrics"], block["score"]
            lines.append(f"- **{title}** → {sc['grade']}（{sc['passed']}/{sc['n_core']}）")
            lines.append(
                f"  - Mu ERD：C3={m['mu_erd_c3']:.1f}% C4={m['mu_erd_c4']:.1f}% | "
                f"对侧={m['mu_erd_contra']:.1f}% 同侧={m['mu_erd_ipsi']:.1f}% | 偏侧pp={m['laterality_pp']:.1f}"
            )
            lines.append(
                f"  - CP3/CP4：{m['mu_erd_cp3']:.1f}% / {m['mu_erd_cp4']:.1f}% | "
                f"βL/βH对侧：{m['betal_erd_contra']:.1f}% / {m['betah_erd_contra']:.1f}%"
            )
            lines.append(
                f"  - 静息Mu占比={m['rest_mu_frac']:.3f} | hop时间降幅={m['time_drop']:.3f} "
                f"谷底={m['time_trough_s']:.2f}s | C3–C4相关 Rest→MI：{m['c3c4_corr_rest']:.2f}→{m['c3c4_corr_mi']:.2f}"
            )
            ck = sc["checks"]
            lines.append(
                "  - 检查："
                + ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in ck.items() if k != "mu_erd_excellent")
                + f" | excellent={'Y' if ck.get('mu_erd_excellent') else 'N'}"
            )
        lines.append("")

    lines += [
        "## 3. 结论摘要",
        "",
        f"OpenBMI {n} 名被试、`{STANDARDS['name']}`："
        f"明显 {gc.get('明显', 0)}、中等 {gc.get('中等', 0)}、弱 {gc.get('弱/不明显', 0)}。",
        "",
        f"- JSON：`{out_json}`",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description="OpenBMI MI 特征显著性：仅 EEG_MI_train；默认全部 54 被试"
    )
    p.add_argument("--subjects", default="", help="逗号分隔；默认 01–54（全部）")
    p.add_argument("--limit-mats-per-subj", type=int, default=0)
    p.add_argument(
        "--out-dir",
        default="",
        help="报告输出目录；默认写 find_best_trail + 04_5060 资料目录",
    )
    args = p.parse_args()
    if args.subjects.strip():
        subjs = [f"{int(x):02d}" for x in args.subjects.split(",") if x.strip()]
    else:
        subjs = [f"{i:02d}" for i in range(1, 55)]

    all_mats = sorted(MAT_GLOB.glob("sess*_subj*_EEG_MI.mat"))
    by_subj: dict[str, list[Path]] = {}
    for mpath in all_mats:
        _, subj = parse_sess_subj(mpath)
        by_subj.setdefault(subj.replace("subj", ""), []).append(mpath)

    results = []
    for nn in subjs:
        mats = sorted(by_subj.get(nn, []))
        if args.limit_mats_per_subj > 0:
            mats = mats[: args.limit_mats_per_subj]
        if not mats:
            print(f"[skip] subj{nn}", flush=True)
            continue
        print(f"[run] subj{nn} mats={len(mats)} …", flush=True)
        results.append(analyze_subject(nn, mats))
        print(
            f"  → {results[-1]['subject_grade']} rate={results[-1]['mean_pass_rate']:.2f} "
            f"wins={results[-1]['n_left']}/{results[-1]['n_right']}/{results[-1]['n_rest']}",
            flush=True,
        )

    outs = []
    if args.out_dir.strip():
        outs.append(Path(args.out_dir))
    else:
        outs.append(REPO / "find_best_trail")
        outs.append(REPO / "资料" / "模型训练" / "04_5060_旁路_2s滑窗100ms_openbmi_accpaper")
    report_stem = f"OpenBMI_{len(results)}被试_MI特征显著性分析"
    for out_dir in outs:
        md = out_dir / f"{report_stem}.md"
        js = out_dir / f"{report_stem}.json"
        write_report(results, md, js)
        print(f"[done] {md}", flush=True)


if __name__ == "__main__":
    main()
