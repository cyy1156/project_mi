"""v3 零样本探针会话：冻结权重 · A/B 引导 · 实时脑电 + 特征卡。

无 FT / 准入 / 游戏；缺模型或 LSL 时拒跑（非演练）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from experiment_game.experiment.eeg_frame_publisher import EegFramePublisher
from experiment_game.experiment.events_log import EventLogger
from experiment_game.experiment.feature_probe import TrialFeatureExtractor, bandpowers_fft
from experiment_game.experiment.inference_v2 import FS, InferenceService, OnlinePreprocessor, RingBuffer, CHANNEL_ORDER
from experiment_game.experiment.markers import MarkerPublisher
from experiment_game.experiment.trial_v2 import (
    TrialContextV2,
    TrialStateMachineV2,
    TrialTimingV2,
    build_calibration_schedule,
)
from experiment_game.experiment.trial_sm import SessionAbort
from experiment_game.experiment.v3_config import V3Config
from experiment_game.experiment.v3_report import build_v3_report, write_v3_report
from experiment_game.experiment.ws_bridge import WsBridge
from experiment_game.experiment.signal_quality import summarize_baseline_hat_check

_ADAPT_ROOT = str(Path(__file__).resolve().parents[2] / "code")
if _ADAPT_ROOT not in sys.path:
    sys.path.insert(0, _ADAPT_ROOT)

_SELF_CHECK_HINT = (
    "v3 探针会话拒跑 — 自检三件套：\n"
    "1. 用 open_operator.bat / open_operator_lan.bat 启动（cyy 环境）\n"
    "2. 开启采集并确认 LSL 流 OpenBCI_EEG 可用\n"
    "3. 确认 config/v3_session.yaml 中 s3_task_ckpt / s3_three_ckpt 路径存在"
)


def block_order(*, seed: Optional[int], subject_id: str) -> List[str]:
    # 用 sha256 而非内置 hash()：后者受进程字符串盐随机化影响，跨进程不可复现
    key = f"{int(seed) if seed is not None else 0}:{subject_id}"
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
    if h % 2 == 0:
        return ["no_guide", "guided"]
    return ["guided", "no_guide"]


def _weight_fingerprint(registry) -> str:
    try:
        import torch

        parts = []
        for p in registry.three_heads[0].model.parameters():
            parts.append(p.detach().cpu().numpy().tobytes()[:4096])
        return hashlib.sha256(b"".join(parts)).hexdigest()[:16]
    except Exception:
        return "unknown"


def diagnose_v3_deps(
    cfg: V3Config,
    *,
    lsl_timeout_s: float = 8.0,
    on_console: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[tuple], List[str]]:
    """v3 硬校验：权重 + LSL 缺一不可。"""
    from experiment_game.experiment.session_v2 import diagnose_v2_online_deps

    return diagnose_v2_online_deps(
        cfg,  # type: ignore[arg-type]
        require_lsl=True,
        lsl_timeout_s=lsl_timeout_s,
        on_console=on_console,
    )


def build_v3_deps_from_buffer(
    cfg: V3Config,
    buf: RingBuffer,
    *,
    on_console: Optional[Callable[[str], None]] = None,
) -> tuple:
    """仿真回放：权重 + 已有 RingBuffer（无 LSL）。"""
    from pathlib import Path

    from adapt_engine.registry import ModelRegistry

    log = on_console or (lambda _m: None)
    task = Path(__file__).resolve().parents[2] / cfg.s3_task_ckpt
    three = Path(__file__).resolve().parents[2] / cfg.s3_three_ckpt
    if not task.is_file():
        raise FileNotFoundError(f"缺 task 权重: {task}")
    if not three.is_file():
        raise FileNotFoundError(f"缺 three 权重: {three}")
    reg = ModelRegistry(task, three)
    pre = OnlinePreprocessor()
    infer = InferenceService(
        buf,
        reg,
        pre,
        task_p_on=cfg.task_p_on,
        signal_quality=cfg.signal_quality_config(),
        window_mode=cfg.online_window_mode,
        mi_task_sec=cfg.imagine_s,
        baseline_before_cue_s=cfg.baseline_before_cue_s,
    )
    log("[v3] 仿真依赖就绪：权重 + Replay RingBuffer")
    return reg, buf, pre, infer


def _extract_segment(buf: RingBuffer, t_start: float, t_end: float) -> Optional[np.ndarray]:
    from pylsl import local_clock

    n = max(1, int(round((t_end - t_start) * FS)))
    return buf.window_ending_at(t_end, n, t_now_lsl=local_clock())


def run_v3_session(
    events: EventLogger,
    markers: MarkerPublisher,
    bridge: WsBridge,
    *,
    on_console: Callable[[str], None] = print,
    config_path: Optional[str] = None,
    v3_overrides: Optional[Dict] = None,
    protocol_locked: bool = True,
    seed: Optional[int] = None,
    subject_id: str = "unknown",
    subject_feedback_mode: str = "none",
    deps: Optional[tuple] = None,
    skip_session_baseline: bool = False,
    skip_block_gap: bool = False,
    block_order_override: Optional[List[str]] = None,
    trial_labels_by_block: Optional[List[List[int]]] = None,
    sim_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import random

    cfg = V3Config.load_yaml(config_path) if config_path else V3Config.load_yaml()
    ignored = cfg.apply_overrides(v3_overrides, protocol_locked=protocol_locked)
    if ignored:
        on_console(f"[v3] 冻结锁忽略 overrides: {ignored}")
    verr = cfg.verify_errors()
    if verr:
        raise ValueError("v3 配置无效: " + "; ".join(verr))

    rng = random.Random(seed) if seed is not None else random.Random()
    if block_order_override is not None:
        order = list(block_order_override)
    else:
        order = block_order(seed=seed, subject_id=subject_id)
    on_console(
        f"[v3] 探针会话：块顺序 {' → '.join(order)} · "
        f"在线窗 {cfg.online_window_mode} ×{len(cfg.judgment_times)} 档"
        f"{f' · seed={seed}' if seed is not None else ''}"
    )

    if deps is not None:
        reg, buf, _, infer = deps
        if reg is None or buf is None or infer is None:
            raise RuntimeError("注入 deps 须含 registry、buffer、infer")
    else:
        deps_pair, reasons = diagnose_v3_deps(cfg, on_console=on_console)
        if deps_pair is None:
            msg = _SELF_CHECK_HINT + "\n原因：" + ("；".join(reasons) if reasons else "未知")
            on_console(f"[v3] ERR {msg}")
            raise RuntimeError(msg)
        reg, buf, _, infer = deps_pair
    assert buf is not None and infer is not None
    infer.window_mode = str(getattr(cfg, "online_window_mode", "openbmi_hop100"))
    infer.mi_task_sec = float(cfg.imagine_s)
    infer.baseline_before_cue_s = float(getattr(cfg, "baseline_before_cue_s", 0.5))
    infer.forward_window = False
    pre_feat = OnlinePreprocessor()
    fp_start = _weight_fingerprint(reg)
    eeg_pub = EegFramePublisher(buf, bridge, cfg, pre=pre_feat, on_console=on_console)
    eeg_pub.start()

    session_dir = Path(events.path).parent if hasattr(events, "path") else Path(".")
    features_path = session_dir / "v3_trial_features.jsonl"
    segments_dir = session_dir / "v3_segments"
    if cfg.save_trial_segments:
        segments_dir.mkdir(parents=True, exist_ok=True)

    extractor = TrialFeatureExtractor(cfg.standards())
    trial_records: List[Dict] = []
    block_records: Dict[str, List[Dict]] = {c: [] for c in order}
    invalid_streak = 0
    invalid_streak_max = 0
    trial_times: Dict[int, Dict[str, float]] = {}
    trial_judgments: Dict[int, List[Dict]] = {}
    progress: Dict[str, Any] = {
        "phase_step": "self_check",
        "block_idx": 0,
        "block_cond": None,
        "trial_in_block": 0,
        "trials_per_block": cfg.trials_per_block,
        "blocks_total": cfg.blocks,
        "session_score": 0,
        "session_trials_done": 0,
        "session_score_max": int(cfg.blocks) * int(cfg.trials_per_block),
    }

    def _ser(data):
        if data is None:
            return None
        if isinstance(data, dict):
            return {
                k: v for k, v in data.items()
                if isinstance(v, (int, float, str, bool, type(None), dict, list))
            }
        return None

    arm_peak_by_trial: Dict[int, int] = {}

    def on_stage(stage: str, ctx, data) -> None:
        from experiment_game.experiment.score_feedback import enrich_stage_data

        if ctx is not None and stage == "cue" and isinstance(data, dict) and "cue_t" in data:
            tid = getattr(ctx, "trial_id", -1)
            trial_times.setdefault(tid, {})["cue_t"] = float(data["cue_t"])
        if ctx is not None and stage == "mi_start" and isinstance(data, dict) and "mi_t" in data:
            mi_t = float(data["mi_t"])
            tid = getattr(ctx, "trial_id", -1)
            cue_t = float(data.get("cue_t", mi_t - cfg.cue_s))
            trial_times[tid] = {
                "mi_t": mi_t,
                "cue_t": cue_t,
                "mi_end_t": mi_t + cfg.imagine_s,
            }
        # 试次间 Rest（Cue 前约 4s）→ 灌入本块滚动 ERD 基线（不要求静息试次）
        if stage == "rest_end" and isinstance(data, dict) and buf is not None:
            rest_t = data.get("rest_t")
            dur = float(data.get("duration_s") or cfg.inter_trial_rest_s)
            if rest_t is not None and dur > 1e-6:
                seg = _extract_segment(buf, float(rest_t), float(rest_t) + dur)
                if seg is not None and seg.shape[0] >= int(FS):
                    filt = pre_feat.process_segment(seg)
                    n_itr = extractor.seed_rest_from_segment(filt, as_block_seed=False)
                    if n_itr:
                        on_console(f"[v3] 试次间 Rest 灌入 ERD 基线 +{n_itr} 窗")
        if ctx is not None:
            progress["trial_in_block"] = getattr(ctx, "trial_id", 0)
            progress["block_cond"] = current_cond
        if subject_feedback_mode == "arm_reach":
            data = enrich_stage_data(
                stage,
                ctx,
                data if isinstance(data, dict) else data,
                peak_by_trial=arm_peak_by_trial,
            )
        score = None
        if isinstance(data, dict) and data.get("score") is not None:
            score = data.get("score")
        # 本场累计：多数票每试次 0/1，满分 = blocks × trials_per_block
        if (
            stage == "trial_end"
            and ctx is not None
            and not getattr(ctx, "rejected", False)
            and isinstance(data, dict)
        ):
            summary = data.get("summary") if isinstance(data.get("summary"), dict) else None
            lab_end = getattr(ctx, "label", None)
            # Rest(0)/L/R 均计分；静息无伸手/拿杯（见 score_feedback.enrich_stage_data）
            if summary is not None and lab_end in (0, 1, 2):
                pts = float(summary.get("score") or 0.0)
                progress["session_score"] = float(progress.get("session_score") or 0) + pts
                progress["session_trials_done"] = int(progress.get("session_trials_done") or 0) + 1
                progress["score"] = pts
        from experiment_game.experiment.class_labels import attach_judge_names, label_name

        lab = getattr(ctx, "label", None) if ctx else None
        if isinstance(data, dict):
            data = attach_judge_names(data, label=lab)
        bridge.broadcast({
            "type": "v2_stage",
            "stage": stage,
            "ctx": {
                "trial_id": getattr(ctx, "trial_id", None) if ctx else None,
                "label": lab,
                "label_name": label_name(lab),
                "mode": getattr(ctx, "mode", None) if ctx else None,
                "round": getattr(ctx, "round_no", None) if ctx else None,
                "subblock": getattr(ctx, "subblock", None) if ctx else None,
            },
            "data": _ser(data),
            "progress": dict(progress),
            "score": score,
            "session_score": progress.get("session_score"),
            "session_score_max": progress.get("session_score_max"),
            "session_trials_done": progress.get("session_trials_done"),
        })

    def judgment_fn(mi_t: float, t_rel: float, ctx) -> Optional[Dict]:
        tid = getattr(ctx, "trial_id", -1)
        times = trial_times.setdefault(tid, {})
        times["mi_t"] = mi_t
        cue_t = times.get("cue_t")
        if cue_t is None:
            cue_t = mi_t if cfg.cue_s <= 1e-6 else mi_t - cfg.cue_s
            times["cue_t"] = cue_t
        try:
            j = infer.judge(float(cue_t), t_rel)
            if j is None:
                return None
            rec = {
                "t_rel": t_rel,
                "win_start_rel": j.get("win_start_rel"),
                "win_end_rel": j.get("win_end_rel", t_rel),
                "pred": int(j.get("pred", 0)),
                "gated": bool(j.get("gated", False)),
                "gated_pred": 0 if j.get("gated") else int(j.get("pred", 0)),
                "p_max": float(j.get("p_max", 0.0)),
                "p_three": j.get("p_three"),
                "margin": j.get("margin"),
                "signal_bad": bool(j.get("signal_bad", False)),
                "reason": j.get("reason"),
            }
            trial_judgments.setdefault(tid, []).append(rec)
            return j
        except Exception:
            return None

    def on_trial_end(ctx, summary: Optional[Dict]) -> Optional[str]:
        nonlocal invalid_streak, invalid_streak_max
        if getattr(ctx, "rejected", False) or summary is None:
            return None
        if summary.get("valid"):
            invalid_streak = 0
        else:
            invalid_streak += 1
            invalid_streak_max = max(invalid_streak_max, invalid_streak)
        _finalize_trial(ctx, summary)
        return None

    timing = TrialTimingV2(
        prep_s=cfg.prep_s,
        cue_s=cfg.cue_s,
        imagine_s=cfg.imagine_s,
        iti_s=cfg.iti_s,
        inter_trial_rest_s=cfg.inter_trial_rest_s,
        judgment_times=cfg.judgment_times,
    )
    time_scale = 1.0
    if sim_meta:
        replay_speed = float(sim_meta.get("replay_speed") or 1.0)
        if replay_speed > 1.0 + 1e-9:
            time_scale = 1.0 / replay_speed
            on_console(
                f"[v3] 仿真：范式等待 ×{time_scale:.4f}（与 {replay_speed}× 回放墙钟对齐）"
            )
    sm = TrialStateMachineV2(
        events,
        markers,
        timing,
        on_stage=on_stage,
        judgment_fn=judgment_fn,
        on_trial_end=on_trial_end,
        is_paused=bridge.is_paused,
        should_abort=bridge.should_abort,
        is_rejected=bridge.is_rejected,
        time_scale=time_scale,
    )

    def _wait_rest(duration_s: float, *, hud: str, sub: str = "") -> None:
        from experiment_game.experiment.trial_sm import wait_until
        from pylsl import local_clock

        bridge.broadcast({
            "type": "hud",
            "text": hud,
            "subtext": sub,
            "show_cross": True,
        })
        wait_until(
            local_clock() + duration_s * time_scale,
            is_paused=bridge.is_paused,
            should_abort=bridge.should_abort,
        )

    def _finalize_trial(ctx: TrialContextV2, summary: Dict) -> None:
        if bridge.should_abort():
            return
        tid = ctx.trial_id
        times = trial_times.get(tid, {})
        cue_t = times.get("cue_t")
        js = trial_judgments.get(tid, [])
        js_ok = [j for j in js if not j.get("signal_bad")]
        # 多数判定窗信号差 → 整试次剔除：不算特征、不进滚动 Rest 基线、不进块统计
        signal_bad = bool(js) and len(js_ok) * 2 <= len(js)

        feat: Dict[str, Any] = {}
        if signal_bad:
            feat = {
                "label": ctx.label,
                "signal_bad": True,
                "verdict_text": "信号质量不足，本试次不计统计（acc/ERD 均已剔除）",
            }
        elif cue_t is not None:
            mi_seg_raw = _extract_segment(buf, cue_t, cue_t + 4.0)
            if mi_seg_raw is not None and mi_seg_raw.shape[0] >= int(FS):
                mi_filt = pre_feat.process_segment(mi_seg_raw)
                feat = extractor.compute_trial_features(ctx.label, mi_filt)
                extractor.add_trial_segment(ctx.label, mi_filt)
                if cfg.save_trial_segments:
                    seg_raw = _extract_segment(
                        buf,
                        cue_t - 2.0,
                        times.get("mi_end_t", cue_t + cfg.imagine_s + cfg.cue_s),
                    )
                    if seg_raw is not None:
                        np.save(segments_dir / f"trial{tid:03d}.npy", seg_raw.astype(np.float32))

        primary = None
        if js_ok:
            from experiment_game.experiment.judge_aggregate import primary_judge_from_judgments

            primary = primary_judge_from_judgments(
                js_ok,
                mode=getattr(cfg, "primary_judge_mode", "majority"),
                primary_s=cfg.primary_judge_s,
            )

        signal_reason = None
        if signal_bad:
            bad_reasons = [j.get("reason") for j in js if j.get("signal_bad") and j.get("reason")]
            if bad_reasons:
                from collections import Counter

                signal_reason = Counter(bad_reasons).most_common(1)[0][0]

        record = {
            "trial_id": tid,
            "label": ctx.label,
            "block": block_idx + 1,
            "cond": current_cond,
            "valid": bool(summary.get("valid")),
            "score": summary.get("score"),
            "signal_bad": signal_bad,
            "signal_reason": signal_reason,
            "judgments": js,
            "primary_judge": primary,
            "features": feat,
        }
        trial_records.append(record)
        block_records[current_cond].append(record)
        with features_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        bridge.broadcast({
            "type": "trial_features",
            "trial_id": tid,
            "label": ctx.label,
            "block": block_idx + 1,
            "cond": current_cond,
            "features": feat,
            "trial_grade": feat.get("trial_grade") or feat.get("grade"),
            "block_grade": feat.get("block_grade"),
            "grade": feat.get("grade"),
            "primary_judge": primary,
            "p_three": primary.get("p_three") if primary else None,
            "valid": record["valid"],
            "signal_bad": signal_bad,
        })
        _broadcast_block_stats()

    def _broadcast_block_stats() -> None:
        recs = block_records[current_cond]
        lr = [
            r for r in recs
            if r.get("label") in (1, 2) and r.get("valid") and not r.get("signal_bad")
        ]
        acc = None
        if lr:
            ok = 0
            for r in lr:
                pj = r.get("primary_judge")
                if pj and int(pj.get("pred", -1)) == int(r["label"]):
                    ok += 1
            acc = ok / len(lr) if lr else None
        erds = [
            float(r["features"]["mu_erd_contra"])
            for r in lr
            if r.get("features") and "mu_erd_contra" in r["features"]
        ]
        lats = [
            float(r["features"]["laterality_pp"])
            for r in lr
            if r.get("features") and "laterality_pp" in r["features"]
        ]
        bridge.broadcast({
            "type": "v3_block",
            "block_idx": block_idx + 1,
            "blocks_total": cfg.blocks,
            "cond": current_cond,
            "trial_done": len(recs),
            "trials_per_block": cfg.trials_per_block,
            "n_valid": len(lr),
            "n_lr": len([r for r in recs if r.get("label") in (1, 2)]),
            "n_signal_bad": sum(1 for r in recs if r.get("signal_bad")),
            "acc_argmax": round(acc, 3) if acc is not None else None,
            "mu_erd_contra_mean": round(float(np.mean(erds)), 1) if erds else None,
            "laterality_pp_mean": round(float(np.mean(lats)), 1) if lats else None,
            "session_score": progress.get("session_score"),
            "session_score_max": progress.get("session_score_max"),
            "session_trials_done": progress.get("session_trials_done"),
            "progress": dict(progress),
        })

    def _run_guidance(block_no: int) -> None:
        progress["phase_step"] = "guidance"
        bridge.broadcast({
            "type": "v2_stage",
            "stage": "guidance_begin",
            "ctx": None,
            "data": {"round": block_no, "timeout_s": cfg.guidance_timeout_s},
            "progress": dict(progress),
        })
        events.emit("v3_guidance_begin", phase="v3", block=block_no)
        markers.push(f"v3_guidance_begin|block={block_no}")
        confirmed = bridge.wait_client_event("v2_guidance_confirm", timeout=cfg.guidance_timeout_s)
        events.emit("v3_guidance_end", phase="v3", block=block_no, passed=bool(confirmed))
        markers.push(f"v3_guidance_end|block={block_no}|passed={int(bool(confirmed))}")
        bridge.broadcast({
            "type": "v2_stage",
            "stage": "guidance_end",
            "ctx": None,
            "data": {"passed": confirmed, "block": block_no},
            "progress": dict(progress),
        })
        if not confirmed:
            raise RuntimeError("动觉引导未确认，v3 会话中止")

    aborted = False
    report: Dict[str, Any] = {}
    baseline_feat: Dict[str, Any] = {}

    try:
        progress["phase_step"] = "baseline"
        events.emit("v3_self_check_ok", phase="v3")
        on_console("[v3] OK 自检通过：权重 + LSL 就绪")

        if skip_session_baseline:
            on_console("[v3] 仿真：跳过 session 静息基线（Cue 前 Rest 作校准）")
            events.emit("v3_baseline_skipped", phase="v3", sim=True)
            markers.push("v3_baseline_skipped|sim")
        else:
            # —— 阶段 1：静息基线 ——
            _wait_rest(
                cfg.baseline_rest_s,
                hud="静息基线",
                sub=f"注视 +，{int(cfg.baseline_rest_s)}s，不做判定",
            )
            tail = buf.snapshot_tail(cfg.baseline_rest_s)
            baseline_mu: List[float] = []
            baseline_beta: List[float] = []
            hat_check: Dict[str, Any] = {}
            if tail is not None and tail.shape[0] >= int(FS):
                filt = pre_feat.process_segment(tail)
                n_erd_seed = extractor.seed_rest_from_segment(filt)
                if n_erd_seed:
                    on_console(f"[v3] 块前静息灌入 ERD 基线 {n_erd_seed} 窗")
                    baseline_feat["erd_seed_windows"] = int(n_erd_seed)
                mu, bl, bh = bandpowers_fft(filt, FS, cfg.standards())
                tot = mu + bl + bh
                baseline_feat = {"rest_mu_frac": float(np.mean(mu / (tot + 1e-12)))}
                baseline_mu = [float(x) for x in mu]
                baseline_beta = [float(x) for x in bl]
                hat_check = summarize_baseline_hat_check(
                    tail,
                    fs=FS,
                    cfg=cfg.signal_quality_config(),
                    channel_names=list(CHANNEL_ORDER),
                )
                baseline_feat["hat_check"] = hat_check
                on_console(f"[v3] {hat_check.get('message', '帽检完成')}")
            events.emit("v3_baseline_end", phase="v3", **baseline_feat)
            if hat_check:
                events.emit("v3_baseline_hat", phase="v3", **hat_check)
            markers.push("v3_baseline_end")
            if baseline_mu:
                bridge.broadcast({
                    "type": "v3_baseline",
                    "baseline_mu": baseline_mu,
                    "baseline_beta": baseline_beta,
                    "rest_mu_frac": baseline_feat.get("rest_mu_frac"),
                    "hat_check": hat_check,
                    "hat_verdict": hat_check.get("verdict"),
                    "hat_message": hat_check.get("message"),
                })

        trial_offset = 0
        current_cond = order[0]
        block_idx = 0

        for block_idx, cond in enumerate(order):
            current_cond = cond
            progress["block_idx"] = block_idx + 1
            progress["block_cond"] = cond
            progress["phase_step"] = "block"
            extractor.reset_block()

            if cond == "guided":
                _run_guidance(block_idx + 1)

            events.emit("v3_block_begin", phase="v3", block=block_idx + 1, cond=cond)
            markers.push(f"v3_block_begin|block={block_idx + 1}|cond={cond}")
            bridge.broadcast({
                "type": "v3_block",
                "block_idx": block_idx + 1,
                "blocks_total": cfg.blocks,
                "cond": cond,
                "trial_done": 0,
                "trials_per_block": cfg.trials_per_block,
                "n_valid": 0,
                "n_lr": 0,
                "phase": "begin",
            })

            if trial_labels_by_block is not None and block_idx < len(trial_labels_by_block):
                labels = [int(x) for x in trial_labels_by_block[block_idx]]
            else:
                labels = build_calibration_schedule(rng)[: cfg.trials_per_block]
                if len(labels) < cfg.trials_per_block:
                    while len(labels) < cfg.trials_per_block:
                        labels.extend(build_calibration_schedule(rng))
                    labels = labels[: cfg.trials_per_block]

            stub = TrialContextV2(trial_id=0, label=0, mode="probe", round_no=block_idx + 1)
            on_stage("round_start", stub, {"mode": "probe", "cond": cond, "block": block_idx + 1, "round": block_idx + 1})
            for i, lab in enumerate(labels):
                if bridge.should_abort():
                    raise SessionAbort("operator abort")
                ctx = TrialContextV2(
                    trial_id=trial_offset + i + 1,
                    label=int(lab),
                    mode="probe",
                    round_no=block_idx + 1,
                    subblock=(i // 6) + 1,
                )
                if cfg.inter_trial_rest_s > 1e-6 and int(lab) != 0:
                    sm.run_inter_trial_rest(ctx)
                sm.run_trial(ctx)
            on_stage("round_end", stub, {"mode": "probe", "cond": cond, "round": block_idx + 1})
            trial_offset += len(labels)

            events.emit("v3_block_end", phase="v3", block=block_idx + 1, cond=cond)
            markers.push(f"v3_block_end|block={block_idx + 1}|cond={cond}")

            if block_idx < len(order) - 1 and not skip_block_gap:
                progress["phase_step"] = "block_gap"
                _wait_rest(
                    cfg.block_gap_s,
                    hud="块间休息",
                    sub=f"{int(cfg.block_gap_s)}s · 下一块：{order[block_idx + 1]}",
                )
            elif block_idx < len(order) - 1 and skip_block_gap:
                on_console("[v3] 仿真：跳过块间休息")

        if not aborted:
            progress["phase_step"] = "report"
            fp_end = _weight_fingerprint(reg)
            frozen = fp_start == fp_end
            report = build_v3_report(
                block_order=order,
                block_records=block_records,
                primary_judge_s=cfg.primary_judge_s,
                frozen=frozen,
                invalid_streak_max=invalid_streak_max,
                baseline=baseline_feat,
            )
            write_v3_report(session_dir, report)
            events.emit("v3_report", phase="v3", quality_tier=report.get("quality_tier"))
            markers.push("v3_report")
            bridge.broadcast({"type": "v3_report", "report": report})

    except SessionAbort:
        aborted = True
        on_console("[v3] 操作者中止")
        events.emit("v3_abort", phase="v3", reason="operator_abort")
        markers.push("v3_abort|operator")
        bridge.broadcast({"type": "v3_abort", "reason": "operator_abort"})
        bridge.broadcast({
            "type": "session",
            "status": "error",
            "message": "操作者已中止",
            "phase": "v3_session",
        })

    finally:
        eeg_pub.stop()
        buf.close()

    summary = {
        "phase_mode": "sim_v3_session" if sim_meta else "v3_session",
        "block_order": order,
        "frozen": fp_start == _weight_fingerprint(reg) if reg else False,
        "invalid_streak_max": invalid_streak_max,
        "n_trials": len(trial_records),
        "session_score": progress.get("session_score"),
        "session_score_max": progress.get("session_score_max"),
        "session_trials_done": progress.get("session_trials_done"),
        "report": report,
        "v3_config_effective": cfg.to_dict(),
        "protocol_locked": protocol_locked,
        "seed": seed,
        "quality_tier": report.get("quality_tier") if not aborted else None,
        "aborted": aborted,
        "sim_meta": sim_meta,
    }
    if aborted:
        on_console(f"[v3] 会话已中止（已完成 {len(trial_records)} 试次）")
    else:
        on_console(f"[v3] 会话完成：quality={summary.get('quality_tier')} frozen={summary['frozen']}")
    return summary
