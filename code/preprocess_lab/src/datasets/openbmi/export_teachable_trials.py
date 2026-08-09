"""导出 teachable_v1 可教试次清单（方案 06 · B0）。

- 质量分：分析臂几何（无窗内 z-score）· 被试池化 Rest 作 ERD 基线
- hop100 映射：按与 batch_2s_hop100 相同的 mat 排序重放 trial 序，用全局 trial_id 反查窗下标
- 禁止：用训练 npy 绝对功率算 ERD；用 05 窗内假质量分

用法（preprocess_lab）:
  python -m src.datasets.openbmi.export_teachable_trials
  python -m src.datasets.openbmi.export_teachable_trials --subjects 03,18 --max-mats-per-subj 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
    N_TIMES_2S,
    WIN_SEC,
    segment_to_2s_hop100_windows,
)
from src.datasets.bci2a.labels import filter_left_right_events
from src.datasets.openbmi.analyze_mi_features_25 import (
    CHANS,
    FS_OUT,
    IX,
    MAT_GLOB,
    STANDARDS,
    _bandpowers_fft,
    _erd,
    _stack_band,
)
from src.datasets.openbmi.load_mat import load_openbmi_mat, parse_sess_subj, subject_key

SCHEMA_VERSION = "teachable_v1"
OBVIOUS_12 = {
    "openbmi:subj03",
    "openbmi:subj04",
    "openbmi:subj06",
    "openbmi:subj17",
    "openbmi:subj18",
    "openbmi:subj19",
    "openbmi:subj32",
    "openbmi:subj33",
    "openbmi:subj36",
    "openbmi:subj38",
    "openbmi:subj44",
    "openbmi:subj45",
}


def _load_subject_tiers(json_path: Path) -> dict[str, dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out = {}
    for r in data["results"]:
        out[str(r["subject"])] = {
            "subject_tier": r["subject_grade"],
            "subject_pass_rate": float(r["mean_pass_rate"]),
        }
    return out


def _rest_pool_powers(mats: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """被试全 mats 的 Rest 窗 → 通道均值 Mu/βL/βH 功率 (C,)。"""
    rest_wins: list[np.ndarray] = []
    for mpath in mats:
        runs = load_openbmi_mat(mpath, blocks=("EEG_MI_train",))
        for eeg in runs:
            x = select_channels(eeg.x, eeg.ch_names)
            x = car_reference(x)
            x = notch_and_bandpass(x, eeg.fs)
            fs = float(eeg.fs)
            kept = filter_left_right_events(eeg.events, eeg.artifacts)
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
            for t0, t1 in sources[: int(max_rest)]:
                seg = extract_segment_baseline(x, int(t0), int(t1), fs, baseline_sec=0.5)
                if seg is None:
                    continue
                wins = segment_to_2s_hop100_windows(seg, fs, zscore=False)
                rest_wins.extend(wins)
    if len(rest_wins) < 20:
        return None
    mu, bl, bh = _stack_band(rest_wins, FS_OUT)
    return mu.mean(0), bl.mean(0), bh.mean(0)


def _trial_metrics(
    wins: list[np.ndarray],
    r_mu: np.ndarray,
    r_bl: np.ndarray,
    *,
    left: bool,
) -> dict:
    mi_mu, mi_bl, _mi_bh = _stack_band(wins, FS_OUT)
    m_mu, m_bl = mi_mu.mean(0), mi_bl.mean(0)
    c3, c4 = IX["C3"], IX["C4"]
    contra, ipsi = (c4, c3) if left else (c3, c4)
    mu_contra = _erd(float(m_mu[contra]), float(r_mu[contra]))
    mu_ipsi = _erd(float(m_mu[ipsi]), float(r_mu[ipsi]))
    laterality = mu_ipsi - mu_contra
    betal_contra = _erd(float(m_bl[contra]), float(r_bl[contra]))

    hop_p = []
    for w in wins:
        mu_p, _, _ = _bandpowers_fft(w, FS_OUT)
        hop_p.append(float(mu_p[contra]))
    hop_p = np.asarray(hop_p, dtype=float)
    times = np.arange(len(hop_p)) * float(HOP_SEC)
    if len(hop_p) >= 5:
        pre = float(np.mean(hop_p[times < 0.3]) + 1e-12)
        mid_m = (times >= 0.4) & (times <= 0.9)
        mid = float(np.mean(hop_p[mid_m]) if np.any(mid_m) else hop_p.mean())
        time_drop = (pre - mid) / pre
        trough_s = float(times[int(np.argmin(hop_p))])
        time_ok = time_drop >= float(STANDARDS["time_drop_ok"]) and 0.7 <= trough_s <= 2.0
    else:
        time_drop, trough_s, time_ok = 0.0, float("nan"), False

    return {
        "mu_erd_contra": float(mu_contra),
        "mu_erd_ipsi": float(mu_ipsi),
        "laterality": float(laterality),
        "betal_erd_contra": float(betal_contra),
        "time_pattern_ok": bool(time_ok),
        "time_drop": float(time_drop),
        "time_trough_s": float(trough_s),
        "n_windows": int(len(wins)),
    }


def _iter_mat_trials(mat_path: Path):
    """与 preprocess_run_2s_hop100 同序产出 trial（含 Rest）；MI 附带无 z-score 窗。"""
    runs = load_openbmi_mat(mat_path, blocks=("EEG_MI_train",))
    sess, subj = parse_sess_subj(mat_path)
    sk = subject_key(subj)
    for run_i, eeg in enumerate(runs):
        x = select_channels(eeg.x, eeg.ch_names)
        x = car_reference(x)
        x = notch_and_bandpass(x, eeg.fs)
        fs = float(eeg.fs)
        kept = filter_left_right_events(eeg.events, eeg.artifacts)
        local_in_run = 0

        for cue, _lt, lab3, _ in kept:
            seg = task_window_cue_0_to_4(x, int(cue), fs)
            if seg is None:
                continue
            wins_z = segment_to_2s_hop100_windows(seg, fs, zscore=True)
            wins_nz = segment_to_2s_hop100_windows(seg, fs, zscore=False)
            kept_z = [w for w in wins_z if w.shape == (N_TIMES_2S, 8)]
            if not kept_z:
                continue
            # 分析窗与训练窗一一对应（条数应同）；取前 len(kept_z) 条无 z 窗
            wins_use = wins_nz[: len(kept_z)]
            if len(wins_use) < len(kept_z):
                # 极少见：退化为仅用已有无 z 窗（映射仍以 hop100 条数为准）
                pass
            hand = "left" if int(lab3) == 1 else "right"
            yield {
                "kind": "mi",
                "subject_id": sk,
                "sess": sess,
                "run_i": run_i,
                "cue": int(cue),
                "hand": hand,
                "lab3": int(lab3),
                "local_in_run": local_in_run,
                "wins_nz": wins_use,
                "n_windows_expect": len(kept_z),
            }
            local_in_run += 1

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
        for t0, t1 in sources[: int(max_rest)]:
            seg = extract_segment_baseline(x, int(t0), int(t1), fs, baseline_sec=0.5)
            if seg is None:
                continue
            wins_z = segment_to_2s_hop100_windows(seg, fs, zscore=True)
            kept_z = [w for w in wins_z if w.shape == (N_TIMES_2S, 8)]
            if not kept_z:
                continue
            yield {
                "kind": "rest",
                "subject_id": sk,
                "sess": sess,
                "run_i": run_i,
                "cue": int(t0),
                "hand": "rest",
                "lab3": 0,
                "local_in_run": local_in_run,
                "wins_nz": None,
                "n_windows_expect": len(kept_z),
            }
            local_in_run += 1


def _build_tid_index(trial_ids: np.ndarray) -> dict[int, tuple[int, int]]:
    """trial_id → (start, count)；窗在 hop100 内连续。"""
    starts: dict[int, int] = {}
    counts: dict[int, int] = {}
    for i, t in enumerate(trial_ids.tolist()):
        t = int(t)
        if t not in starts:
            starts[t] = i
            counts[t] = 1
        else:
            counts[t] += 1
    return {t: (starts[t], counts[t]) for t in starts}


def export_teachable(
    *,
    subjects_filter: set[str] | None,
    max_mats_per_subj: int,
    subject_json: Path,
    hop100_dir: Path,
    out_dir: Path,
    audit_n: int,
) -> Path:
    tiers = _load_subject_tiers(subject_json)
    y3 = np.load(hop100_dir / "openbmi_y_three.npy")
    subjects = np.load(hop100_dir / "openbmi_subjects.npy", allow_pickle=True)
    trial_ids = np.load(hop100_dir / "openbmi_trial_id.npy")
    tid_index = _build_tid_index(np.asarray(trial_ids, dtype=np.int64))

    all_mats = sorted(MAT_GLOB.glob("sess*_subj*_EEG_MI.mat"), key=lambda p: p.name)
    by_subj: dict[str, list[Path]] = defaultdict(list)
    for mpath in all_mats:
        _, subj = parse_sess_subj(mpath)
        by_subj[subject_key(subj)].append(mpath)

    # Rest 功率池（仅打分被试）；为保持 global trial_id，仍按全部 mat 序推进
    norm_filter: set[str] | None = None
    if subjects_filter:
        norm_filter = set()
        for s in subjects_filter:
            if s.startswith("openbmi:"):
                norm_filter.add(s)
            else:
                digits = "".join(ch for ch in s if ch.isdigit())
                norm_filter.add(f"openbmi:subj{int(digits):02d}")
    need_subjects = (
        sorted(norm_filter)
        if norm_filter is not None
        else sorted(by_subj.keys())
    )

    print(f"[rest-pool] subjects={len(need_subjects)} …", flush=True)
    rest_pow: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sk in need_subjects:
        mats = sorted(by_subj[sk], key=lambda p: p.name)
        if max_mats_per_subj > 0:
            mats = mats[:max_mats_per_subj]
        rp = _rest_pool_powers(mats)
        if rp is None:
            print(f"  [warn] no rest pool {sk}", flush=True)
            continue
        rest_pow[sk] = rp
        print(f"  [ok] rest {sk}", flush=True)

    records: list[dict] = []
    audit_ok = 0
    audit_fail = 0
    map_fail = 0
    global_tid = 0
    rank_buckets: dict[tuple[str, str], list[int]] = defaultdict(list)

    mats_walk = all_mats
    if subjects_filter or max_mats_per_subj > 0:
        # 子集模式：仍按全局 mat 序走完以保持 trial_id，但对非目标被试只推进 tid、不算分
        pass

    need_set = set(need_subjects)
    print(f"[walk-mats] n={len(mats_walk)} …", flush=True)
    for mi, mpath in enumerate(mats_walk):
        _, subj = parse_sess_subj(mpath)
        sk = subject_key(subj)
        score_this = sk in need_set
        if max_mats_per_subj > 0 and score_this:
            mats_of = set(sorted(by_subj[sk], key=lambda p: p.name)[:max_mats_per_subj])
            if mpath not in mats_of:
                score_this = False

        n_local = 0
        for tr in _iter_mat_trials(mpath):
            gtid = global_tid + n_local
            n_local += 1
            if tr["kind"] != "mi":
                continue
            if not score_this:
                continue
            if sk not in rest_pow:
                continue
            if gtid not in tid_index:
                map_fail += 1
                continue
            start, cnt = tid_index[gtid]
            wids = list(range(start, start + cnt))
            # 校验标签 / 被试
            lab_h = int(y3[start])
            sub_h = str(subjects[start])
            ok = sub_h == sk and lab_h == int(tr["lab3"]) and cnt == int(tr["n_windows_expect"])
            if ok:
                audit_ok += 1
            else:
                audit_fail += 1
                if audit_fail <= 5:
                    print(
                        f"  [map-mismatch] tid={gtid} expect {sk}/{tr['lab3']}/{tr['n_windows_expect']} "
                        f"got {sub_h}/{lab_h}/{cnt}",
                        flush=True,
                    )
                continue

            r_mu, r_bl, _r_bh = rest_pow[sk]
            met = _trial_metrics(tr["wins_nz"], r_mu, r_bl, left=(tr["hand"] == "left"))
            tier_info = tiers.get(sk, {"subject_tier": "未知", "subject_pass_rate": float("nan")})
            tier = tier_info["subject_tier"]
            t0 = tier in ("明显", "中等")
            t1 = bool(
                t0
                and met["mu_erd_contra"] <= float(STANDARDS["mu_erd_contra_ok"])
                and met["laterality"] >= float(STANDARDS["laterality_pp_ok"])
            )
            t2 = bool(
                t1
                and (
                    met["mu_erd_contra"] <= float(STANDARDS["mu_erd_contra_excellent"])
                    or met["time_pattern_ok"]
                )
            )
            obvious12 = sk in OBVIOUS_12
            high_lat = bool(obvious12 and met["laterality"] >= float(STANDARDS["laterality_pp_ok"]))
            rec = {
                "schema_version": SCHEMA_VERSION,
                "subject_id": sk,
                "subject_tier": tier,
                "subject_pass_rate": tier_info["subject_pass_rate"],
                "trial_uid": f"{sk}|{tr['sess']}|cue{tr['cue']}|{tr['hand']}|r{tr['run_i']}|i{tr['local_in_run']}",
                "sess": tr["sess"],
                "cue": tr["cue"],
                "hand": tr["hand"],
                "mu_erd_contra": met["mu_erd_contra"],
                "mu_erd_ipsi": met["mu_erd_ipsi"],
                "laterality": met["laterality"],
                "time_pattern_ok": met["time_pattern_ok"],
                "time_drop": met["time_drop"],
                "time_trough_s": met["time_trough_s"],
                "teachable": t1,
                "template_grade": t2,
                "in_obvious12": obvious12,
                "high_lat_eval": high_lat,
                "hop100_trial_id": int(gtid),
                "hop100_window_ids": wids,
                "n_windows": int(cnt),
                "map_ok": True,
            }
            records.append(rec)
            rank_buckets[(sk, tr["hand"])].append(len(records) - 1)

        global_tid += n_local
        if (mi + 1) % 10 == 0 or (mi + 1) == len(mats_walk):
            print(
                f"  mats {mi+1}/{len(mats_walk)} global_tid={global_tid} records={len(records)} "
                f"map_ok={audit_ok} fail={audit_fail}",
                flush=True,
            )

    # rank_in_subject_hand by laterality desc
    for (_sk, _hand), idxs in rank_buckets.items():
        idxs_sorted = sorted(idxs, key=lambda i: -float(records[i]["laterality"]))
        for rank, i in enumerate(idxs_sorted, start=1):
            records[i]["rank_in_subject_hand"] = int(rank)

    if global_tid != int(trial_ids.max()) + 1 and not subjects_filter and max_mats_per_subj <= 0:
        print(
            f"[warn] global_tid end={global_tid} vs hop100 max+1={int(trial_ids.max())+1}",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "teachable_trials_v1.json"
    out_csv = out_dir / "teachable_trials_v1.csv"
    out_npz = out_dir / "teachable_window_masks_v1.npz"
    out_readme = out_dir / "teachable_trials_v1_README.md"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "standards": STANDARDS,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "subject_json": str(subject_json),
        "hop100_dir": str(hop100_dir),
        "channels": CHANS,
        "obvious12": sorted(OBVIOUS_12),
        "rules": {
            "T0": "subject_tier in {明显,中等}",
            "T1_teachable": "T0 & mu_erd_contra<=-15 & laterality>=8",
            "T2_template": "T1 & (mu_erd_contra<=-35 | time_pattern_ok)",
            "R2_high_lat": "in_obvious12 & laterality>=8",
        },
        "n_records": len(records),
        "n_teachable": int(sum(1 for r in records if r["teachable"])),
        "n_template_grade": int(sum(1 for r in records if r["template_grade"])),
        "n_high_lat_eval": int(sum(1 for r in records if r["high_lat_eval"])),
        "map_audit_ok": audit_ok,
        "map_audit_fail": audit_fail,
        "map_missing_tid": map_fail,
        "trials": records,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_fields = [
        "subject_id",
        "subject_tier",
        "hand",
        "teachable",
        "template_grade",
        "high_lat_eval",
        "mu_erd_contra",
        "laterality",
        "time_pattern_ok",
        "hop100_trial_id",
        "rank_in_subject_hand",
        "trial_uid",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in csv_fields})

    n = len(y3)
    mask_teach = np.zeros(n, dtype=bool)
    mask_tmpl = np.zeros(n, dtype=bool)
    mask_r1 = np.zeros(n, dtype=bool)
    mask_r2 = np.zeros(n, dtype=bool)
    for r in records:
        wids = r["hop100_window_ids"]
        if r["in_obvious12"]:
            mask_r1[wids] = True
        if r["high_lat_eval"]:
            mask_r2[wids] = True
        if r["teachable"]:
            mask_teach[wids] = True
        if r["template_grade"]:
            mask_tmpl[wids] = True
    np.savez_compressed(
        out_npz,
        teachable=mask_teach,
        template_grade=mask_tmpl,
        obvious12=mask_r1,
        high_lat_eval=mask_r2,
        schema_version=np.array(SCHEMA_VERSION),
    )

    # 抽检
    rng = np.random.default_rng(42)
    sample_n = min(audit_n, len(records))
    sample_idx = rng.choice(len(records), size=sample_n, replace=False) if sample_n else []
    audit_lines = []
    for i in sample_idx:
        r = records[int(i)]
        s, c = tid_index[int(r["hop100_trial_id"])]
        audit_lines.append(
            f"- tid={r['hop100_trial_id']} {r['subject_id']} {r['hand']} "
            f"y3={int(y3[s])} n={c} teachable={r['teachable']} lat={r['laterality']:.1f}"
        )

    readme = "\n".join(
        [
            f"# teachable_trials_v1",
            "",
            f"- 生成：`{payload['generated_at']}`",
            f"- 标准：`{STANDARDS['name']}`",
            f"- 记录数：{payload['n_records']} · teachable(T1)={payload['n_teachable']} · "
            f"template(T2)={payload['n_template_grade']} · high_lat(R2)={payload['n_high_lat_eval']}",
            f"- 映射抽检：ok={audit_ok} fail={audit_fail} missing_tid={map_fail}",
            f"- 被试总评来源：`{subject_json}`",
            f"- hop100：`{hop100_dir}`",
            "",
            "## 规则",
            "",
            "- T0：subject_tier ∈ {明显, 中等}",
            "- T1 teachable：T0 ∧ mu_erd_contra≤−15% ∧ laterality≥8",
            "- T2 template_grade：T1 ∧ (mu_erd_contra≤−35% ∨ time_pattern_ok)",
            "- R2 high_lat_eval：明显12 ∧ laterality≥8",
            "",
            "## 文件",
            "",
            f"- `{out_json.name}`",
            f"- `{out_csv.name}`",
            f"- `{out_npz.name}`（与 openbmi_X 等长 bool mask）",
            "",
            "## 随机抽检",
            "",
            *audit_lines,
            "",
            "## 复跑",
            "",
            "```bash",
            "cd code/preprocess_lab",
            "python -m src.datasets.openbmi.export_teachable_trials",
            "```",
            "",
        ]
    )
    out_readme.write_text(readme, encoding="utf-8")
    print(f"[done] {out_json}", flush=True)
    print(
        f"  teachable={payload['n_teachable']} template={payload['n_template_grade']} "
        f"high_lat={payload['n_high_lat_eval']} map_ok={audit_ok} fail={audit_fail}",
        flush=True,
    )
    return out_json


def main() -> None:
    p = argparse.ArgumentParser(description="导出 teachable_v1 可教试次清单（B0）")
    p.add_argument("--subjects", default="", help="逗号分隔被试号，如 03,18；默认全部")
    p.add_argument("--max-mats-per-subj", type=int, default=0, help=">0 时每被试最多 mats（冒烟）")
    p.add_argument(
        "--subject-json",
        default=str(REPO / "find_best_trail" / "OpenBMI_54被试_MI特征显著性分析.json"),
    )
    p.add_argument(
        "--hop100-dir",
        default=str(PRE_ROOT / "out" / "openbmi_2s_hop100"),
    )
    p.add_argument(
        "--out-dir",
        default=str(REPO / "find_best_trail" / "out"),
    )
    p.add_argument("--audit-n", type=int, default=20)
    args = p.parse_args()
    filt = None
    if args.subjects.strip():
        filt = {x.strip() for x in args.subjects.split(",") if x.strip()}
    export_teachable(
        subjects_filter=filt,
        max_mats_per_subj=int(args.max_mats_per_subj),
        subject_json=Path(args.subject_json),
        hop100_dir=Path(args.hop100_dir),
        out_dir=Path(args.out_dir),
        audit_n=int(args.audit_n),
    )


if __name__ == "__main__":
    main()
