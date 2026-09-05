"""v3 每试次 MI 特征（移植 analyze_mi_features_25 口径，纯 numpy）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from experiment_game.experiment.inference_v2 import CHANNEL_ORDER, FS

IX = {n: i for i, n in enumerate(CHANNEL_ORDER)}
HOP_SEC = 0.1
WIN_SEC = 2.0


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


def bandpowers_fft(x: np.ndarray, fs: float, standards: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x (T,C) → Mu / βL / βH 功率 (C,)，rFFT 均值。"""
    t = x.shape[0]
    freqs = np.fft.rfftfreq(t, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x.astype(np.float64), axis=0)) ** 2

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        if not np.any(m):
            return np.full(x.shape[1], 1e-12)
        return spec[m].mean(axis=0) + 1e-12

    mu = standards["mu_hz"]
    bl = standards["beta_l_hz"]
    bh = standards["beta_h_hz"]
    return band(*mu), band(*bl), band(*bh)


def erd(p_task: float, p_rest: float) -> float:
    return 100.0 * (p_task - p_rest) / (p_rest + 1e-12)


def stack_band(
    wins: List[np.ndarray], fs: float, standards: Dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mus, bls, bhs = [], [], []
    for w in wins:
        a, b, c = bandpowers_fft(w, fs, standards)
        mus.append(a)
        bls.append(b)
        bhs.append(c)
    return np.stack(mus, 0), np.stack(bls, 0), np.stack(bhs, 0)


def segment_to_hop_windows(seg_tc: np.ndarray, fs: float = FS) -> List[np.ndarray]:
    """cue 后 0–4s 段 → 2s/100ms 滑窗列表。"""
    win_n = int(round(WIN_SEC * fs))
    hop_n = int(round(HOP_SEC * fs))
    if seg_tc.shape[0] < win_n:
        return []
    out: List[np.ndarray] = []
    for start in range(0, seg_tc.shape[0] - win_n + 1, hop_n):
        out.append(seg_tc[start : start + win_n])
    return out


def _trial_side_metrics(
    label: int,
    mu_erd: Dict[str, float],
    mu_contra: float,
    mu_ipsi: float,
    laterality: float,
    betal_contra: float,
    betah_contra: float,
    rest_mu_frac: float,
    n_rest_trials: int,
    n_rest_windows: int,
    n_windows: int,
) -> SideMetrics:
    return SideMetrics(
        n_mi_trials=1,
        n_rest_trials=n_rest_trials,
        n_mi_windows=n_windows,
        n_rest_windows=n_rest_windows,
        mu_erd_c3=mu_erd["C3"],
        mu_erd_c4=mu_erd["C4"],
        mu_erd_cp3=mu_erd["CP3"],
        mu_erd_cp4=mu_erd["CP4"],
        mu_erd_contra=mu_contra,
        mu_erd_ipsi=mu_ipsi,
        laterality_pp=laterality,
        betal_erd_contra=betal_contra,
        betah_erd_contra=betah_contra,
        rest_mu_frac=rest_mu_frac,
        time_onset_ok=False,
        time_trough_s=float("nan"),
        time_drop=0.0,
        c3c4_corr_rest=float("nan"),
        c3c4_corr_mi=float("nan"),
        corr_drop=float("nan"),
    )


def verdict_text_mi(
    label: int,
    *,
    mu_contra: float,
    laterality: float,
    contra_ch: str,
    standards: Dict[str, Any],
    trial_grade: Dict[str, Any],
) -> str:
    side = "左手想象" if label == 1 else "右手想象"
    mu_thr = abs(float(standards["mu_erd_contra_ok"]))
    lat_thr = float(standards["laterality_pp_ok"])
    mu_ok = mu_contra <= standards["mu_erd_contra_ok"]
    lat_ok = laterality >= lat_thr
    if mu_contra <= 0:
        mu_phrase = f"{contra_ch} mu 较静息降 {abs(mu_contra):.0f}%"
    else:
        mu_phrase = f"{contra_ch} mu 较静息升 {mu_contra:.0f}%，未出现 ERD"
    parts = [
        f"{side}：{mu_phrase}"
        f"（合格线 {mu_thr:.0f}%{'✓' if mu_ok else '✗'}），"
        f"偏侧 {laterality:.1f}pp（合格线 {lat_thr:.0f}{'✓' if lat_ok else '✗'}）"
    ]
    grade = trial_grade.get("grade", "")
    if grade == "明显":
        parts.append("→ 本试次特征明显")
    elif grade == "中等":
        parts.append("→ 本试次特征中等")
    else:
        parts.append("→ 本试次特征弱；建议检查引导/电极")
    return "".join(parts)


def verdict_text_rest(*, warmup: bool, n_rest_windows_before: int, n_windows: int) -> str:
    if warmup:
        return "ERD 基线预热：块前静息 seed 尚未灌入（不计 ERD 评级）"
    total = n_rest_windows_before + n_windows
    return f"试次间 Rest 灌入基线：本段 +{n_windows} 窗，块内累计 {total} 窗"


def score_side(m: SideMetrics, standards: Dict[str, Any]) -> Dict[str, Any]:
    s = standards
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


def side_metrics(
    mi_wins: List[np.ndarray],
    rest_wins: List[np.ndarray],
    mi_trials: List[List[np.ndarray]],
    rest_trials: List[List[np.ndarray]],
    *,
    left: bool,
    fs: float,
    standards: Dict[str, Any],
) -> Optional[SideMetrics]:
    if len(mi_wins) < 1 or len(rest_wins) < 1:
        return None

    rest_mu, rest_bl, rest_bh = stack_band(rest_wins, fs, standards)
    mi_mu, mi_bl, mi_bh = stack_band(mi_wins, fs, standards)
    r_mu, r_bl, r_bh = rest_mu.mean(0), rest_bl.mean(0), rest_bh.mean(0)
    m_mu, m_bl, m_bh = mi_mu.mean(0), mi_bl.mean(0), mi_bh.mean(0)

    def erd_ch(i: int) -> float:
        return erd(float(m_mu[i]), float(r_mu[i]))

    c3, c4 = IX["C3"], IX["C4"]
    cp3, cp4 = IX["CP3"], IX["CP4"]
    contra, ipsi = (c4, c3) if left else (c3, c4)

    mu_contra = erd_ch(contra)
    mu_ipsi = erd_ch(ipsi)
    laterality = mu_ipsi - mu_contra
    betal_contra = erd(float(m_bl[contra]), float(r_bl[contra]))
    betah_contra = erd(float(m_bh[contra]), float(r_bh[contra]))
    rest_tot = r_mu + r_bl + r_bh
    rest_mu_frac = float(np.mean(r_mu / (rest_tot + 1e-12)))

    max_h = max((len(tr) for tr in mi_trials), default=0)
    hop_p = []
    for h in range(max_h):
        ps = []
        for tr in mi_trials:
            if h < len(tr):
                mu_p, _, _ = bandpowers_fft(tr[h], fs, standards)
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
        time_onset_ok = time_drop >= float(standards["time_drop_ok"]) and 0.7 <= trough_s <= 2.0
    else:
        time_drop, trough_s, time_onset_ok = 0.0, float("nan"), False

    def mean_corr(wins: List[np.ndarray]) -> float:
        cs = []
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


class TrialFeatureExtractor:
    """块内滚动 Rest 基线 + 每试次特征。"""

    LABEL_LEFT = 1
    LABEL_RIGHT = 2
    LABEL_REST = 0

    def __init__(self, standards: Dict[str, Any], *, fs: float = FS, max_rest_windows: int = 63):
        self.standards = standards
        self.fs = fs
        self.max_rest_windows = int(max_rest_windows)
        self._seed_wins: List[np.ndarray] = []
        self.reset_block()

    def reset_block(self) -> None:
        """清空本块 MI 累计；Rest 基线恢复为块前 seed（不要求先做静息试次）。"""
        self._rest_wins = [w.copy() for w in getattr(self, "_seed_wins", [])]
        self._rest_trials = [list(self._rest_wins)] if self._rest_wins else []
        self._mi_wins = []
        self._mi_trials = []
        self._trim_rest_baseline()

    def _trim_rest_baseline(self) -> None:
        """保留最近 N 个 Rest 滑窗作 ERD 基线（块内滚动）。"""
        if self.max_rest_windows <= 0 or len(self._rest_wins) <= self.max_rest_windows:
            return
        drop = len(self._rest_wins) - self.max_rest_windows
        self._rest_wins = self._rest_wins[drop:]
        kept = 0
        new_trials: List[List[np.ndarray]] = []
        for tr in self._rest_trials:
            if kept + len(tr) > drop:
                off = max(0, drop - kept)
                if off < len(tr):
                    new_trials.append(tr[off:])
                break
            kept += len(tr)
        self._rest_trials = new_trials

    def seed_rest_from_segment(
        self,
        seg_filtered_tc: np.ndarray,
        *,
        as_block_seed: bool = True,
    ) -> int:
        """块前/试次间静息：灌入已滤波 Rest 段。as_block_seed=True 时同时写入跨块 seed。"""
        wins = segment_to_hop_windows(seg_filtered_tc, self.fs)
        if not wins:
            return 0
        if as_block_seed:
            self._seed_wins.extend([w.copy() for w in wins])
        self._rest_wins.extend(wins)
        self._rest_trials.append(wins)
        self._trim_rest_baseline()
        return len(wins)

    def add_trial_segment(self, label: int, seg_filtered_tc: np.ndarray) -> None:
        """seg_filtered_tc: cue 后 0–4s，已 8–30Hz 滤波 (T,8)。"""
        wins = segment_to_hop_windows(seg_filtered_tc, self.fs)
        if not wins:
            return
        if label == self.LABEL_REST:
            self._rest_wins.extend(wins)
            self._rest_trials.append(wins)
            self._trim_rest_baseline()
        elif label in (self.LABEL_LEFT, self.LABEL_RIGHT):
            self._mi_wins.extend(wins)
            self._mi_trials.append(wins)

    def compute_trial_features(
        self,
        label: int,
        seg_filtered_tc: np.ndarray,
    ) -> Dict[str, Any]:
        wins = segment_to_hop_windows(seg_filtered_tc, self.fs)
        out: Dict[str, Any] = {"label": label, "n_windows": len(wins)}
        empty_grade = {"grade": "无数据", "checks": {}, "passed": 0, "n_core": 0, "rate": 0.0}
        if not wins:
            out["trial_grade"] = empty_grade
            out["grade"] = empty_grade
            out["verdict_text"] = "本试次数据不足，无法计算特征"
            return out

        # —— Rest 试次：只更新基线，不算对侧 ERD ——
        if label == self.LABEL_REST:
            warmup = len(self._rest_wins) == 0
            out["warmup"] = warmup
            out["is_rest"] = True
            out["n_rest_windows_before"] = len(self._rest_wins)
            if not warmup:
                mu, bl, bh = bandpowers_fft(seg_filtered_tc, self.fs, self.standards)
                tot = mu + bl + bh
                out["rest_mu_frac"] = float(np.mean(mu / (tot + 1e-12)))
            tg = {"grade": "预热", "checks": {}, "passed": 0, "n_core": 0, "rate": 0.0} if warmup else {
                "grade": "基线",
                "checks": {"rest_collected": True},
                "passed": 1,
                "n_core": 1,
                "rate": 1.0,
            }
            out["trial_grade"] = tg
            out["grade"] = tg
            out["verdict_text"] = verdict_text_rest(
                warmup=warmup,
                n_rest_windows_before=len(self._rest_wins),
                n_windows=len(wins),
            )
            return out

        # —— MI 试次：单试次 ERD + 块累计 grade 分离 ——
        if not self._rest_wins:
            out["no_baseline"] = True
            out["verdict_text"] = "尚无 Rest 基线（块前静息未灌入）"
            out["trial_grade"] = empty_grade
            out["grade"] = empty_grade
            return out

        mu, bl, bh = bandpowers_fft(seg_filtered_tc, self.fs, self.standards)
        r_mu, r_bl, r_bh = stack_band(self._rest_wins, self.fs, self.standards)
        rm, rb, rh = r_mu.mean(0), r_bl.mean(0), r_bh.mean(0)
        out["mu_erd"] = {
            ch: erd(float(mu[IX[ch]]), float(rm[IX[ch]])) for ch in ("C3", "C4", "CP3", "CP4")
        }
        if label == self.LABEL_LEFT:
            contra, ipsi = IX["C4"], IX["C3"]
            contra_ch = "C4"
        else:
            contra, ipsi = IX["C3"], IX["C4"]
            contra_ch = "C3"
        mu_contra = erd(float(mu[contra]), float(rm[contra]))
        mu_ipsi = erd(float(mu[ipsi]), float(rm[ipsi]))
        out["mu_erd_contra"] = mu_contra
        out["mu_erd_ipsi"] = mu_ipsi
        out["contra_ch"] = contra_ch
        out["laterality_pp"] = mu_ipsi - mu_contra
        out["betal_erd_contra"] = erd(float(bl[contra]), float(rb[contra]))
        out["betah_erd_contra"] = erd(float(bh[contra]), float(rh[contra]))
        rest_tot = rm + rb + rh
        out["rest_mu_frac"] = float(np.mean(rm / (rest_tot + 1e-12)))

        trial_m = _trial_side_metrics(
            label,
            out["mu_erd"],
            mu_contra,
            mu_ipsi,
            out["laterality_pp"],
            out["betal_erd_contra"],
            out["betah_erd_contra"],
            out["rest_mu_frac"],
            len(self._rest_trials),
            len(self._rest_wins),
            len(wins),
        )
        out["trial_grade"] = score_side(trial_m, self.standards)
        out["grade"] = out["trial_grade"]

        mi_wins_now = self._mi_wins + wins
        mi_trials_now = self._mi_trials + [wins]
        side = side_metrics(
            mi_wins_now,
            self._rest_wins,
            mi_trials_now,
            self._rest_trials,
            left=label == self.LABEL_LEFT,
            fs=self.fs,
            standards=self.standards,
        )
        if side is not None:
            out["side_metrics"] = asdict(side)
            out["block_grade"] = score_side(side, self.standards)
            out["block_n_mi_trials"] = side.n_mi_trials
            out["block_n_rest_segments"] = side.n_rest_trials
            out["block_n_rest_trials"] = side.n_rest_trials  # 兼容旧键；语义=Rest 段数（非 label=0 试次）
        else:
            out["block_grade"] = out["trial_grade"]

        out["verdict_text"] = verdict_text_mi(
            label,
            mu_contra=mu_contra,
            laterality=out["laterality_pp"],
            contra_ch=contra_ch,
            standards=self.standards,
            trial_grade=out["trial_grade"],
        )
        return out
