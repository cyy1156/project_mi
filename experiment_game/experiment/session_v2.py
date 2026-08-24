"""v2 会话模式：动觉引导 → 4 轮标定（12FT+6小考，轮间增量微调）→ 准入 → 游戏协同。

组装 trial_v2 + inference_v2 + adapt_engine；常量从 config/v2_session.yaml 读（可改）。
权重/LSL 缺失时自动降级为"流程演练"（无判定无微调），事件照常落盘，可合成板冒烟。

D8：试次内 0.6s 栅格计分；连续无效熔断 → 可选 FT → v1 诱导/记录降级。

调用：SessionRunner.run_v2_session() 或直接 run_v2_session(events, markers, bridge, on_console)。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from experiment_game.experiment.events_log import EventLogger
from experiment_game.experiment.markers import MarkerPublisher
from experiment_game.experiment.trial_sm import SessionAbort, TrialContext, TrialStateMachine, build_label_schedule
from experiment_game.experiment.trial_v2 import (
    TrialContextV2,
    TrialStateMachineV2,
    TrialTimingV2,
    build_calibration_schedule,
    build_game_schedule,
)
from experiment_game.experiment.v2_config import V2Config
from experiment_game.experiment.prompt_wait import ENV_ADAPT_BODY, wait_prompt_continue
from experiment_game.experiment.ws_bridge import WsBridge

_ADAPT_ROOT = str(Path(__file__).resolve().parents[2] / "code")
_STEP_ROOT = str(Path(__file__).resolve().parents[2] / "code" / "train_lab" / "src" / "step")
if _ADAPT_ROOT not in sys.path:
    sys.path.insert(0, _ADAPT_ROOT)
if _STEP_ROOT not in sys.path:
    sys.path.insert(0, _STEP_ROOT)


def _load_openbmi_replay_pool(
    cfg: V2Config,
    *,
    seed: int = 42,
    max_windows: int = 4096,
    on_console: Callable[[str], None] = print,
):
    """OpenBMI 源域回放池（25-G2 · replay_ratio）。"""
    if cfg.replay_ratio <= 0:
        return None
    try:
        import numpy as np
        from adapt_engine.ft import ReplayPool
        from data_paths import resolve_data

        data_dir, prefix = resolve_data("openbmi_3s_hop100")
        X = np.load(data_dir / f"{prefix}_X.npy", mmap_mode="r")
        y = np.load(data_dir / f"{prefix}_y_three.npy")
        n = min(len(y), max_windows)
        idx = np.linspace(0, len(y) - 1, n, dtype=int)
        pool = ReplayPool(np.array(X[idx], dtype=np.float32), np.asarray(y[idx], dtype=np.int64), seed=seed)
        on_console(f"[v2] replay 池：{n} 窗（openbmi three）")
        return pool
    except Exception as exc:
        on_console(f"[v2] ⚠️ replay 池加载失败：{exc}")
        return None


def diagnose_v2_online_deps(
    cfg: V2Config,
    *,
    require_lsl: bool = True,
    lsl_timeout_s: float = 5.0,
    on_console: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[tuple], List[str]]:
    """检查在线判定/微调依赖。

    返回 (deps_or_None, reasons)。
    deps = (registry, buffer, pre, infer)；失败时 deps 为 None。
    require_lsl=False 时只检查权重（会话开始前、采集尚未推 LSL 时用）。
    """
    log = on_console or (lambda _m: None)
    reasons: List[str] = []
    try:
        import torch  # noqa: F401
    except Exception as exc:
        reasons.append(f"torch 不可用: {exc}")
        return None, reasons

    try:
        from adapt_engine.registry import ModelRegistry
        from experiment_game.experiment.inference_v2 import (
            InferenceService,
            OnlinePreprocessor,
            RingBuffer,
        )
    except Exception as exc:
        reasons.append(f"adapt_engine/inference 导入失败: {exc}")
        return None, reasons

    task = Path(__file__).resolve().parents[2] / cfg.s3_task_ckpt
    three = Path(__file__).resolve().parents[2] / cfg.s3_three_ckpt
    if not task.exists():
        reasons.append(f"缺 task 权重: {task}")
    if not three.exists():
        reasons.append(f"缺 three 权重: {three}")
    if reasons:
        return None, reasons

    try:
        reg = ModelRegistry(task, three)
    except Exception as exc:
        reasons.append(f"ModelRegistry 加载失败: {exc}")
        return None, reasons

    if not require_lsl:
        log("[v2] 权重就绪（LSL 待采集启动后再挂）")
        return (reg, None, None, None), []

    buf = RingBuffer()
    try:
        buf.attach_lsl("OpenBCI_EEG", timeout_s=lsl_timeout_s)
    except Exception as exc:
        reasons.append(f"LSL OpenBCI_EEG 不可用: {exc}")
        try:
            buf.close()
        except Exception:
            pass
        return None, reasons

    try:
        pre = OnlinePreprocessor()
        infer = InferenceService(
            buf,
            reg,
            pre,
            task_p_on=cfg.task_p_on,
            signal_quality=cfg.signal_quality_config(),
        )
    except Exception as exc:
        reasons.append(f"InferenceService 初始化失败: {exc}")
        try:
            buf.close()
        except Exception:
            pass
        return None, reasons

    log("[v2] 在线依赖就绪：权重 + LSL OpenBCI_EEG")
    return (reg, buf, pre, infer), []


def _try_imports(cfg: V2Config, on_console: Optional[Callable[[str], None]] = None):
    """返回 (registry, buffer, pre, infer) 或全 None（降级）。"""
    deps, reasons = diagnose_v2_online_deps(cfg, require_lsl=True, on_console=on_console)
    if deps is None:
        log = on_console or print
        for r in reasons:
            log(f"[v2] ⚠️ 演练降级原因：{r}")
        return None
    return deps


def probe_v2_weights_missing(cfg: V2Config) -> bool:
    """会话开始前：仅查权重（此时尚无 LSL，不能据此判演练）。"""
    deps, _ = diagnose_v2_online_deps(cfg, require_lsl=False)
    return deps is None


def probe_v2_degraded(cfg: V2Config) -> bool:
    """兼容旧调用：含 LSL 检查。采集启动前请用 probe_v2_weights_missing。"""
    return _try_imports(cfg) is None


class _SessionStore:
    """试次 → 判定窗收集（FT/小考数据源）。"""

    def __init__(self):
        self.windows: Dict[int, list] = {}
        self.labels: Dict[int, int] = {}
        self.valid_trials: Set[int] = set()

    def add(self, trial_id: int, label: int, wins: list) -> None:
        if trial_id in self.windows:
            return
        self.windows[trial_id] = wins
        self.labels[trial_id] = int(label)


def _run_v1_fallback_record(
    events: EventLogger,
    markers: MarkerPublisher,
    bridge: WsBridge,
    *,
    n_trials: int = 12,
    on_console: Callable[[str], None] = print,
) -> None:
    """熔断后降级：v1 范式诱导/记录（无在线 judgment_fn）。"""
    from experiment_game.experiment.timing import DEFAULT_TIMING

    on_console(f"[v2] 降级 v1 记录模式（{n_trials} trial，无在线判定）")
    events.emit("v2_fallback_v1_begin", phase="v2_fallback_v1", n_trials=n_trials)
    markers.push("v2_fallback_v1_begin")

    def on_stage(stage: str, ctx, _label) -> None:
        bridge.broadcast({
            "type": "stage",
            "phase": "v2_fallback_v1",
            "stage": stage,
            "trial_id": getattr(ctx, "trial_id", None),
            "label": getattr(ctx, "label", None),
        })

    v1_sm = TrialStateMachine(
        events,
        markers,
        timing=DEFAULT_TIMING,
        on_stage=on_stage,
        is_paused=bridge.is_paused,
        should_abort=bridge.should_abort,
        is_rejected=bridge.is_rejected,
    )
    labels = build_label_schedule(n_trials)
    for i, lab in enumerate(labels, start=1):
        ctx = TrialContext(
            trial_id=9000 + i,
            label=int(lab),
            object="cup",
            scene="home_desk",
            phase="v2_fallback_v1",
        )
        v1_sm.run_trial(ctx)

    events.emit("v2_fallback_v1_end", phase="v2_fallback_v1", n_trials=n_trials)
    markers.push("v2_fallback_v1_end")


def run_v2_session(
    events: EventLogger,
    markers: MarkerPublisher,
    bridge: WsBridge,
    on_console: Callable[[str], None] = print,
    *,
    config_path: Optional[str] = None,
    v2_overrides: Optional[Dict] = None,
    protocol_locked: bool = True,
    seed: Optional[int] = None,
    skip_guidance: bool = False,
    skip_calibration: bool = False,
    skip_gate: bool = False,
    skip_game: bool = False,
) -> Dict:
    import random

    cfg = V2Config.load_yaml(config_path) if config_path else V2Config.load_yaml()
    ignored = cfg.apply_overrides(v2_overrides, protocol_locked=protocol_locked)
    if ignored:
        on_console(f"[v2] 冻结锁忽略 overrides: {ignored}")
    verr = cfg.verify_errors()
    if verr:
        raise ValueError("v2 配置无效: " + "; ".join(verr))

    rng = random.Random(seed) if seed is not None else random.Random()
    on_console(
        f"[v2] 常量加载：轮数{cfg.cal_rounds_min}-{cfg.cal_rounds_max} "
        f"准入{cfg.gate_enter_three} lr={cfg.group_lr} "
        f"D8栅格{len(cfg.judgment_times)}档"
        f"{' · locked' if protocol_locked else ' · debug'}"
        f"{f' · seed={seed}' if seed is not None else ''}"
    )

    deps = _try_imports(cfg, on_console=on_console)
    degraded = deps is None
    if degraded:
        on_console("[v2] ⚠️ 权重或 LSL 不可用 → 流程演练模式（无判定/无微调）")
        bridge.broadcast({
            "type": "v2_online_status",
            "degraded": True,
            "reason": "weights_or_lsl_unavailable",
            "message": "演练模式：权重或 LSL 不可用，无在线判定/微调",
        })
        reg = buf = infer = None
    else:
        reg, buf, _, infer = deps
        on_console("[v2] ✅ 在线判定/微调已启用")
        bridge.broadcast({
            "type": "v2_online_status",
            "degraded": False,
            "reason": "ok",
            "message": "在线判定与微调已启用",
        })

    store = _SessionStore()
    window_cache: Dict[int, list] = {}
    mi_times: Dict[int, float] = {}
    consecutive_invalid = 0
    aborted = False
    abort_reason: Optional[str] = None
    progress: Dict[str, object] = {
        "cal_round": 0,
        "cal_rounds_max": cfg.cal_rounds_max,
        "game_round": 0,
        "game_rounds": cfg.game_rounds,
        "subblock": 0,
        "score": None,
        "ft_status": "idle",
        "phase_step": "guidance",
    }

    def judgment_fn(mi_t: float, t_rel: float, ctx) -> Optional[Dict]:
        tid = getattr(ctx, "trial_id", -1)
        mi_times[tid] = mi_t
        if infer is None:
            return None
        try:
            j = infer.judge(mi_t, t_rel)
            if j is not None and j.get("signal_bad"):
                return j
            if j is not None and "window" in j:
                window_cache.setdefault(tid, []).append(np.asarray(j["window"], dtype=np.float32))
            return j
        except Exception:
            return None

    def on_stage(stage: str, ctx, data) -> None:
        if stage == "mi_start" and isinstance(data, dict) and "mi_t" in data:
            mi_times[getattr(ctx, "trial_id", -1)] = float(data["mi_t"])
        if stage == "round_start" and isinstance(data, dict):
            m = data.get("mode")
            if m == "calibration":
                progress["phase_step"] = "calibration"
            elif m == "game":
                progress["phase_step"] = "game"
        score = None
        if isinstance(data, dict):
            if data.get("score") is not None:
                score = data.get("score")
            elif isinstance(data.get("summary"), dict) and data["summary"].get("score") is not None:
                score = data["summary"].get("score")
        if score is not None:
            progress["score"] = score
        if ctx is not None and getattr(ctx, "subblock", None):
            progress["subblock"] = getattr(ctx, "subblock", 0)
        mode = getattr(ctx, "mode", None) if ctx is not None else None
        if mode == "calibration":
            progress["phase_step"] = "calibration"
        elif mode == "game":
            progress["phase_step"] = "game"
        bridge.broadcast({
            "type": "v2_stage",
            "stage": stage,
            "ctx": {
                "trial_id": getattr(ctx, "trial_id", None) if ctx else None,
                "label": getattr(ctx, "label", None) if ctx else None,
                "mode": mode,
                "round": getattr(ctx, "round_no", None) if ctx else None,
                "subblock": getattr(ctx, "subblock", None) if ctx else None,
            },
            "data": _ser(data),
            "progress": dict(progress),
            "score": score,
        })

    def on_trial_end(ctx, summary: Optional[Dict]) -> Optional[str]:
        nonlocal consecutive_invalid
        if getattr(ctx, "rejected", False):
            return None
        if summary is None:
            return None
        if summary.get("valid"):
            consecutive_invalid = 0
            store.valid_trials.add(ctx.trial_id)
        else:
            consecutive_invalid += 1
            if consecutive_invalid >= cfg.consecutive_invalid_abort:
                return "abort_session"
        return None

    timing = TrialTimingV2(
        prep_s=cfg.prep_s,
        cue_s=cfg.cue_s,
        imagine_s=cfg.imagine_s,
        iti_s=cfg.iti_s,
        judgment_times=cfg.judgment_times,
        scoring=cfg.scoring_config(),
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
    )

    def _commit_round_windows(trial_ids: List[int]) -> List[int]:
        """从 judgment_fn 缓存提交本轮有效试次窗（C1：判定时已处理窗）。"""
        committed: List[int] = []
        for tid in trial_ids:
            if tid not in store.valid_trials or tid in store.windows:
                continue
            wins = window_cache.get(tid)
            if not wins:
                continue
            lab = store.labels.get(tid)
            if lab is None:
                continue
            store.add(tid, lab, wins)
            committed.append(tid)
        return committed

    def _ft_arrays(trial_ids: List[int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        ids = [t for t in trial_ids if t in store.valid_trials and t in store.windows]
        if not ids:
            return None
        X = np.concatenate([store.windows[t] for t in ids], axis=0)
        y = np.concatenate([[store.labels[t]] * len(store.windows[t]) for t in ids])
        return X, y

    def _windows_of_trial(ti: int) -> np.ndarray:
        if ti not in store.valid_trials or ti not in store.windows:
            return np.zeros((0, 8, 750), dtype=np.float32)
        return np.asarray(store.windows[ti], dtype=np.float32)

    def _label_of_trial(ti: int) -> int:
        return int(store.labels[ti])

    def _run_game_trials(
        labels: List[int],
        *,
        round_no: int,
        trial_id_offset: int,
        group_size: int = 3,
    ) -> None:
        """游戏轮：每 group_size 试次一组 FT（H6）。"""
        stub = TrialContextV2(
            trial_id=0, label=0, mode="game", round_no=round_no, subblock=0,
        )
        on_stage("round_start", stub, {"mode": "game", "round": round_no})
        sm.events.emit(
            "round_start", phase="game", round=round_no,
            payload={"phase": "game", "event": "round_start"},
        )
        markers.push("round_start|phase=game")
        for g_start in range(0, len(labels), group_size):
            chunk = labels[g_start : g_start + group_size]
            chunk_ids = [trial_id_offset + g_start + i + 1 for i in range(len(chunk))]
            for i, lab in enumerate(chunk):
                store.labels[chunk_ids[i]] = int(lab)
                ctx = TrialContextV2(
                    trial_id=chunk_ids[i],
                    label=int(lab),
                    mode="game",
                    round_no=round_no,
                    subblock=(g_start // group_size) + 1,
                )
                sm.run_trial(ctx)
            _commit_round_windows(chunk_ids)
            if fin is not None and drift is not None:
                data = _ft_arrays(chunk_ids)
                if data is not None:
                    def _save_pre_state():
                        fin.save_checkpoint("pre")
                        return fin.snapshot_state()

                    pre_state = drift.before_round(_save_pre_state)
                    rec = fin.train_round(*data, frozen=drift.frozen)
                    metric = float(rec.get("loss") or 0.0)
                    action = drift.after_round(
                        round_no * 100 + (g_start // group_size) + 1,
                        metric,
                        rollback_fn=fin.rollback,
                        halve_fn=fin.halve_lr,
                        get_state_fn=lambda: pre_state,
                    )
                    drift_stats.append({
                        "mode": "game",
                        "round": round_no,
                        "group": (g_start // group_size) + 1,
                        "metric": metric,
                        "action": action.value,
                        **rec,
                    })
        on_stage("round_end", stub, {"mode": "game", "round": round_no})
        sm.events.emit(
            "round_end", phase="game", round=round_no,
            payload={"phase": "game", "event": "round_end"},
        )
        markers.push("round_end|phase=game")

    def _handle_v2_abort(reason: str) -> None:
        nonlocal aborted, abort_reason
        aborted = True
        abort_reason = reason
        on_console(f"[v2] 会话熔断：{reason}（连续无效 {consecutive_invalid}）")
        events.emit(
            "v2_abort",
            phase="v2",
            reason=reason,
            consecutive_invalid=consecutive_invalid,
            valid_trials=len(store.valid_trials),
        )
        markers.push(f"v2_abort|reason={reason}")
        bridge.broadcast({
            "type": "v2_abort",
            "reason": reason,
            "consecutive_invalid": consecutive_invalid,
            "valid_trials": len(store.valid_trials),
        })

        if fin is not None and len(store.valid_trials) >= cfg.ft_min_valid_trials:
            import numpy as np

            valid_ids = sorted(store.valid_trials)
            X = np.concatenate([store.windows[t] for t in valid_ids if t in store.windows], axis=0)
            y = np.concatenate([
                [store.labels[t]] * len(store.windows[t])
                for t in valid_ids
                if t in store.windows
            ])
            if len(X):
                rec = fin.train_round(X, y)
                on_console(f"[v2] 熔断后增量 FT：n={len(valid_ids)} trial, loss={rec.get('loss')}")
                events.emit("v2_abort_ft", phase="v2", n_trials=len(valid_ids), **{k: rec.get(k) for k in ("loss", "epochs")})
        else:
            on_console(
                f"[v2] 有效 trial {len(store.valid_trials)} < K={cfg.ft_min_valid_trials}，跳过熔断 FT"
            )

        _run_v1_fallback_record(events, markers, bridge, on_console=on_console)

    def _early_abort_summary(reason: str, *, rounds: int = 0, gstatus: str = "pending") -> Dict:
        nonlocal aborted, abort_reason
        aborted = True
        abort_reason = reason
        progress["phase_step"] = "end"
        if buf:
            buf.close()
        return {
            "gate_status": gstatus,
            "rounds": rounds,
            "curve": [[p.k_ft, round(p.acc, 4)] for p in quiz.curve] if quiz else [],
            "degraded": degraded,
            "aborted": True,
            "abort_reason": reason,
            "valid_trials": len(store.valid_trials),
            "labels": {str(k): v for k, v in store.labels.items() if k in store.valid_trials},
            "drift_stats": drift_stats,
            "v2_config_effective": cfg.to_dict(),
            "protocol_locked": protocol_locked,
            "seed": seed,
            "weak_mi": gstatus == "weak_mi",
            "skips": {
                "guidance": skip_guidance,
                "calibration": skip_calibration,
                "gate": skip_gate,
                "game": skip_game,
            },
        }

    def _abort_guidance(reason: str, *, inter_round: bool = False, rounds: int = 0) -> Dict:
        on_console(f"[v2] 引导未确认（{'轮间' if inter_round else '阶段0'}）——真机模式中止")
        events.emit(
            "v2_guidance_abort",
            phase="v2",
            reason=reason,
            inter_round=inter_round,
        )
        markers.push(f"v2_guidance_abort|reason={reason}")
        bridge.broadcast({
            "type": "v2_abort",
            "reason": reason,
            "kind": "guidance_timeout",
            "inter_round": inter_round,
        })
        bridge.broadcast({
            "type": "session",
            "status": "error",
            "message": "动觉引导未确认，会话已中止",
            "phase": "v2_session",
        })
        return _early_abort_summary(reason, rounds=rounds)

    def _run_round_safe(*args, **kwargs) -> None:
        try:
            sm.run_round(*args, **kwargs)
        except SessionAbort as exc:
            if str(exc) == "consecutive_invalid_abort":
                _handle_v2_abort("consecutive_invalid_5")
            else:
                raise

    # ============ 阶段 0：动觉引导（不采集） ============
    round_no, trial_offset = 0, 0
    gate_status = "degraded" if degraded else "pending"
    drift_stats: List = []
    ctrl = gate = quiz = fin = drift = None

    progress["phase_step"] = "adapt"
    wait_prompt_continue(
        bridge,
        on_console,
        prompt_id="v2_env_adapt",
        title="环境适应",
        body=ENV_ADAPT_BODY,
        button="我明白了",
    )

    progress["phase_step"] = "guidance"
    if skip_guidance:
        on_console("[v2] 调试跳过：动觉引导")
        events.emit("v2_guidance_end", phase="v2", passed=True, skipped=True)
        bridge.broadcast({
            "type": "v2_stage", "stage": "guidance_end", "ctx": None,
            "data": {"passed": True, "skipped": True}, "progress": dict(progress),
        })
    else:
        bridge.broadcast({
            "type": "v2_stage", "stage": "guidance_begin", "ctx": None,
            "data": {"round": 0, "timeout_s": 600}, "progress": dict(progress),
        })
        events.emit("v2_guidance_begin", phase="v2")
        confirmed = bridge.wait_client_event("v2_guidance_confirm", timeout=600.0)
        events.emit("v2_guidance_end", phase="v2", passed=bool(confirmed))
        bridge.broadcast({
            "type": "v2_stage", "stage": "guidance_end", "ctx": None,
            "data": {"passed": confirmed}, "progress": dict(progress),
        })
        if not confirmed:
            if degraded:
                on_console("[v2] 引导未确认（超时）——演练模式继续")
            else:
                summary = _abort_guidance("guidance_timeout_phase0")
                on_console(f"[v2] 会话完成：{summary}")
                return summary

    # ============ 阶段 1+2：标定轮 × N + 准入 ============
    replay_pool = _load_openbmi_replay_pool(cfg, on_console=on_console)
    predict_window = None

    if not degraded:
        from adapt_engine import AdmissionGate, QuizStore, RoundController
        from adapt_engine.drift import DriftGuard
        from adapt_engine.ft import FTRecipe, IncrementalFinetuner

        def _predict_window(w):
            from adapt_engine.readout import serial_gating

            h = reg.forward_heads(w)
            return serial_gating(h["p_task"], h["p_three"], task_p_on=cfg.task_p_on)

        predict_window = _predict_window

        class _H:
            model = reg.three_heads[0].model
            predict = staticmethod(_predict_window)

        fin = IncrementalFinetuner(
            _H.model,
            FTRecipe(
                lr=cfg.group_lr,
                epochs=cfg.ft_epochs,
                batch_size=cfg.ft_batch_size,
                replay_ratio=cfg.replay_ratio,
                weight_decay=cfg.ft_weight_decay,
            ),
            replay_pool=replay_pool,
            ckpt_dir=Path(events.path).parent / "v2_ckpts" if hasattr(events, "path") else None,
        )
        quiz = QuizStore()
        gate = AdmissionGate(_G(cfg))
        ctrl = RoundController(fin, quiz, gate, constants=_C(cfg), logger=on_console)
        drift = DriftGuard(patience=cfg.drift_patience)
        gate_status = "pending"

    if not aborted and not skip_calibration:
        progress["phase_step"] = "calibration"
        while round_no < cfg.cal_rounds_max and not aborted:
            round_no += 1
            progress["cal_round"] = round_no
            labels = build_calibration_schedule(rng)
            round_ids = [trial_offset + i + 1 for i in range(len(labels))]
            for i, lab in enumerate(labels):
                store.labels[round_ids[i]] = int(lab)
            _run_round_safe(labels, mode="calibration", round_no=round_no, trial_id_offset=trial_offset)
            trial_offset += len(labels)
            _commit_round_windows(round_ids)
            if ctrl is not None and predict_window is not None:
                progress["ft_status"] = "running"
                bridge.broadcast({
                    "type": "v2_stage", "stage": "ft_begin", "ctx": None,
                    "data": {"round": round_no}, "progress": dict(progress),
                })
                pre_state = None
                if drift is not None:
                    def _save_pre_state():
                        fin.save_checkpoint("pre")
                        return fin.snapshot_state()

                    pre_state = drift.before_round(_save_pre_state)
                res = ctrl.run_calibration_round(
                    round_ids,
                    windows_of_trial=_windows_of_trial,
                    label_of_trial=_label_of_trial,
                    predict_window=predict_window,
                    frozen=drift.frozen if drift is not None else False,
                )
                progress["ft_status"] = "done"
                gate_status = res["gate"].status
                if drift is not None:
                    action = drift.after_round(
                        round_no,
                        res["curve"].acc,
                        rollback_fn=fin.rollback,
                        halve_fn=fin.halve_lr,
                        get_state_fn=lambda: pre_state,
                    )
                    drift_stats.append({
                        "mode": "calibration",
                        "round": round_no,
                        "metric": res["curve"].acc,
                        "action": action.value,
                    })
                bridge.broadcast({
                    "type": "v2_gate",
                    "round": round_no,
                    "acc": res["curve"].acc,
                    "n_quiz": res["curve"].n_quiz,
                    "status": gate_status,
                    "curve": [[p.k_ft, round(p.acc, 4)] for p in quiz.curve],
                    "progress": dict(progress),
                })
                progress["phase_step"] = "gate"
                if gate_status == "pass":
                    events.emit(
                        "v2_gate_pass",
                        phase="v2",
                        round=round_no,
                        acc=float(res["curve"].acc),
                    )
                    markers.push(f"v2_gate_pass|round={round_no}")
                    bridge.broadcast({
                        "type": "v2_stage",
                        "stage": "gate_pass",
                        "ctx": None,
                        "data": {"round": round_no, "acc": float(res["curve"].acc)},
                        "progress": dict(progress),
                    })
                    break
                if skip_gate:
                    on_console("[v2] 调试跳过：准入门槛，强制继续/结束标定")
                    break
            if aborted:
                break
            if round_no >= cfg.cal_rounds_min and gate_status == "pass":
                break
            bridge.broadcast({
                "type": "v2_stage",
                "stage": "guidance_begin",
                "ctx": None,
                "data": {
                    "round": round_no,
                    "inter_round": True,
                    "gap_s": cfg.cal_round_gap_s,
                    "timeout_s": cfg.cal_round_gap_s,
                },
                "progress": dict(progress),
            })
            confirmed = bridge.wait_client_event("v2_guidance_confirm", timeout=cfg.cal_round_gap_s)
            bridge.broadcast({
                "type": "v2_stage",
                "stage": "guidance_end",
                "ctx": None,
                "data": {"round": round_no, "inter_round": True, "passed": bool(confirmed)},
                "progress": dict(progress),
            })
            if not confirmed:
                if degraded:
                    on_console("[v2] 轮间引导未确认（超时）——演练模式继续")
                else:
                    summary = _abort_guidance(
                        "guidance_timeout_inter_round",
                        inter_round=True,
                        rounds=round_no,
                    )
                    on_console(f"[v2] 会话完成：{summary}")
                    return summary
    elif skip_calibration:
        on_console("[v2] 调试跳过：标定轮")
        gate_status = "skipped"

    if gate_status == "weak_mi":
        on_console(f"[v2] ⚠️ weak_mi：{round_no} 轮标定未达准入，将用当前权重继续游戏")
        events.emit(
            "v2_weak_mi",
            phase="v2",
            rounds=round_no,
            acc=float(quiz.curve[-1].acc) if quiz and quiz.curve else None,
        )
        markers.push(f"v2_weak_mi|rounds={round_no}")
        bridge.broadcast({
            "type": "v2_stage",
            "stage": "weak_mi",
            "ctx": None,
            "data": {"rounds": round_no, "gate_status": gate_status},
            "progress": dict(progress),
        })
        bridge.broadcast({
            "type": "v2_gate",
            "round": round_no,
            "acc": float(quiz.curve[-1].acc) if quiz and quiz.curve else None,
            "n_quiz": quiz.n_trials if quiz else 0,
            "status": "weak_mi",
            "curve": [[p.k_ft, round(p.acc, 4)] for p in quiz.curve] if quiz else [],
            "progress": dict(progress),
        })

    # ============ 阶段 3：游戏协同（试次内推理冻结；组级 FT） ============
    if not aborted and not skip_game:
        progress["phase_step"] = "game"
        on_console(f"[v2] 游戏 {cfg.game_rounds} 轮（可配 {cfg.game_rounds_min}–{cfg.game_rounds_max}）")
        for g_round in range(1, cfg.game_rounds + 1):
            progress["game_round"] = g_round
            labels = build_game_schedule(cfg.game_trials_per_round, rng)
            _run_game_trials(labels, round_no=g_round, trial_id_offset=trial_offset)
            trial_offset += len(labels)
            bridge.broadcast({
                "type": "v2_stage",
                "stage": "round_end",
                "ctx": None,
                "data": {"mode": "game", "round": g_round},
                "progress": dict(progress),
            })
    elif skip_game:
        on_console("[v2] 调试跳过：游戏轮")

    progress["phase_step"] = "end"
    bridge.broadcast({"type": "session", "status": "done" if not aborted else "aborted", "phase": "end"})
    if buf:
        buf.close()
    summary = {
        "gate_status": gate_status,
        "rounds": round_no,
        "curve": [[p.k_ft, round(p.acc, 4)] for p in quiz.curve] if quiz else [],
        "degraded": degraded,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "valid_trials": len(store.valid_trials),
        "labels": {str(k): v for k, v in store.labels.items() if k in store.valid_trials},
        "drift_stats": drift_stats,
        "v2_config_effective": cfg.to_dict(),
        "protocol_locked": protocol_locked,
        "seed": seed,
        "weak_mi": gate_status == "weak_mi",
        "skips": {
            "guidance": skip_guidance,
            "calibration": skip_calibration,
            "gate": skip_gate,
            "game": skip_game,
        },
    }
    on_console(f"[v2] 会话完成：{summary}")
    return summary


def _ser(data):
    if data is None:
        return None
    if isinstance(data, dict) and "summary" in data:
        return data
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, (int, float, str, bool, type(None), dict, list))}
    return None


class _C:  # adapt_engine.constants 兼容壳（V2Config → SystemConstants 协议）
    def __init__(self, cfg: V2Config):
        for k in (
            "cal_rounds_min",
            "cal_rounds_max",
            "trials_per_round",
            "ft_trials_per_round",
            "quiz_trials_per_round",
            "subblock_size",
            "gate_min_quiz_trials",
        ):
            setattr(self, k, getattr(cfg, k))


class _G(_C):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gate_enter_three = cfg.gate_enter_three
