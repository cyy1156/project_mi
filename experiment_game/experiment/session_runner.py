"""会话编排：适应 → 学习 → 准入 → 正式（Phase 3：换物/换景 + 操作者控制）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from pylsl import local_clock

from experiment_game.experiment.content_catalog import (
    schedule_acquire,
    zh_object,
    zh_scene,
)
from experiment_game.experiment.events_log import EventLogger
from experiment_game.experiment.markers import MarkerPublisher, format_payload
from experiment_game.experiment.timing import DEFAULT_TIMING, TrialTiming
from experiment_game.experiment.trial_sm import (
    SessionAbort,
    TrialContext,
    TrialStateMachine,
    build_label_schedule,
    wait_until,
)
from experiment_game.experiment.ws_bridge import WsBridge, hand_from_label


OnConsole = Callable[[str], None]


@dataclass
class Phase2Config:
    acquire_trials: int = 4
    learn_trials_per_step: int = 2
    seed: Optional[int] = None
    object_name: str = "cup"
    scene: str = "home_desk"
    skip_adapt: bool = False
    skip_learn: bool = False
    skip_gate: bool = False
    auto_continue: bool = False
    # Phase 3
    rotate_objects: bool = True
    rotate_scenes: bool = True
    object_pool: Sequence[str] = field(default_factory=lambda: ["cup", "bottle", "apple"])


class SessionRunner:
    def __init__(
        self,
        events: EventLogger,
        markers: MarkerPublisher,
        bridge: WsBridge,
        timing: TrialTiming = DEFAULT_TIMING,
        config: Optional[Phase2Config] = None,
        on_console: Optional[OnConsole] = None,
    ) -> None:
        self.events = events
        self.markers = markers
        self.bridge = bridge
        self.timing = timing
        self.config = config or Phase2Config()
        self.on_console = on_console or (lambda s: print(s, flush=True))
        self._anim = "none"
        self._learn_step: Optional[int] = None
        self._current_object = self.config.object_name
        self._current_scene = self.config.scene
        self._current_trial_id: Optional[int] = None
        self._current_label: Optional[int] = None
        self._reject_count = 0
        self._aborted = False

        self.bridge.set_operator_hook(self._on_operator)

        def on_stage(stage: str, ctx: TrialContext, label: Optional[int]) -> None:
            phase = self._current_phase
            anim = self._anim if stage == "mi" else "none"
            if phase == "acquire":
                anim = "none"
            self._current_trial_id = ctx.trial_id
            self._current_label = label if label is not None else ctx.label
            self._current_object = ctx.object
            self._current_scene = ctx.scene
            self.bridge.broadcast(
                {
                    "type": "stage",
                    "phase": phase,
                    "stage": stage,
                    "trial_id": ctx.trial_id,
                    "label": label,
                    "hand": hand_from_label(label),
                    "anim": anim,
                    "duration_s": self._duration_for(stage),
                    "object": ctx.object,
                    "scene": ctx.scene,
                    "learn_step": self._learn_step,
                    "transition_amp": ctx.transition_amp,
                }
            )
            text, show_cross, sub = self._hud_for(stage, label, phase, ctx.object)
            self.bridge.broadcast(
                {
                    "type": "hud",
                    "text": text,
                    "subtext": sub,
                    "show_cross": show_cross,
                }
            )
            self._broadcast_operator_state()
            lab = "" if label is None else f" L{label}"
            self.on_console(
                f"  [{phase}] trial={ctx.trial_id} {stage}{lab} "
                f"obj={ctx.object} scene={ctx.scene} anim={anim}"
            )

        self.sm = TrialStateMachine(
            events,
            markers,
            timing=timing,
            on_stage=on_stage,
            is_paused=self.bridge.is_paused,
            should_abort=self.bridge.should_abort,
            is_rejected=self.bridge.is_rejected,
        )
        self._current_phase = "idle"

    def _on_operator(self, action: str, _msg: dict) -> None:
        if action in ("pause", "resume", "toggle_pause"):
            state = "paused" if self.bridge.is_paused() else "running"
            self.on_console(f"[operator] {action} → {state}")
            self._broadcast_operator_state()
        elif action == "reject":
            self.on_console(
                f"[operator] reject trial={self._current_trial_id}"
            )
            self._broadcast_operator_state()
        elif action == "abort":
            self.on_console("[operator] ABORT")
            self._aborted = True
            self._broadcast_operator_state()
        elif action in ("gate_ok", "continue"):
            self.on_console(f"[operator] {action}")

    def _broadcast_operator_state(self) -> None:
        self.bridge.broadcast(
            {
                "type": "operator_state",
                "paused": self.bridge.is_paused(),
                "phase": self._current_phase,
                "trial_id": self._current_trial_id,
                "label": self._current_label,
                "object": self._current_object,
                "scene": self._current_scene,
                "reject_count": self._reject_count,
                "aborting": self.bridge.should_abort(),
            }
        )

    def _duration_for(self, stage: str) -> float:
        t = self.timing
        return {
            "fixation": t.fixation_s,
            "cue": t.cue_s,
            "mi": t.mi_s,
            "post_mi_hold": t.post_mi_hold_s,
            "rest": t.rest_s,
            "transition": t.transition_s,
        }.get(stage, 0.0)

    def _hud_for(
        self,
        stage: str,
        label: Optional[int],
        phase: str,
        object_id: str,
    ) -> tuple[str, bool, str]:
        hand = "左手" if label == 1 else ("右手" if label == 2 else "")
        obj = zh_object(object_id)
        if stage == "fixation":
            return ("", True, "注视十字，保持放松")
        if stage == "cue":
            return (f"{hand}抓取{obj}", False, "记住提示，即将开始想象")
        if stage == "mi":
            if phase == "learn" and self._anim == "full_grasp":
                return (f"观察{hand}完整抓取", False, f"抓住{obj}并取走，只需观看")
            if phase == "adapt" and self._anim == "full_grasp":
                return (f"观察{hand}示意", False, f"抓住{obj}并取走，只需观看")
            if phase == "learn" and self._anim == "reach":
                return (f"想象{hand}抓取（弱辅助）", False, "手会前伸，请同步想象")
            return (f"想象{hand}抓取{obj}", False, "身体保持静止")
        if stage == "post_mi_hold":
            return ("保持", False, "")
        if stage == "rest":
            return ("静息", False, "放松，不要想象动作")
        if stage == "transition":
            return ("", False, "")
        return ("", False, "")

    def _wait_continue(
        self,
        prompt_id: str,
        title: str,
        body: str,
        button: str,
        *,
        allow_subject: bool = True,
    ) -> None:
        self.bridge.clear_event("continue")
        self.bridge.clear_event("gate_ok")
        prompt = {
            "type": "prompt",
            "id": prompt_id,
            "title": title,
            "body": body,
            "button": button,
            "allow_subject": allow_subject,
        }
        self.bridge.broadcast(prompt)
        self._broadcast_operator_state()
        who = "被试或操作者" if allow_subject else "操作者（G / 代确认 / 点按钮）"
        self.on_console(f"[prompt] {title} — 等待{who}「{button}」…")
        if self.config.auto_continue:
            self.bridge.clear_pending_prompt()
            wait_until(local_clock() + 0.3)
            return
        deadline = local_clock() + 600.0
        while local_clock() < deadline:
            if self.bridge.should_abort():
                raise SessionAbort("operator abort")
            if self.bridge.is_paused():
                wait_until(
                    local_clock() + 0.2,
                    is_paused=None,
                    should_abort=self.bridge.should_abort,
                )
                continue
            # 准入：接受 continue 或 gate_ok（G）
            if self.bridge.wait_client_event("continue", timeout=0.4):
                self.bridge.clear_pending_prompt()
                self.bridge.clear_event("gate_ok")
                return
            if self.bridge.wait_client_event("gate_ok", timeout=0.1):
                self.bridge.clear_pending_prompt()
                self.bridge.clear_event("continue")
                return
            self.bridge.broadcast(prompt)
        self.bridge.clear_pending_prompt()
        raise TimeoutError(f"等待 continue 超时: {prompt_id}")

    def wait_browser_ready(self, timeout: float = 300.0) -> None:
        self.bridge.broadcast(
            {"type": "hello", "message": "打开诱导页后将自动 ready"}
        )
        self.on_console(f"等待浏览器连接并 ready… ({self.bridge.url})")
        if self.config.auto_continue:
            return
        if self.bridge.wait_client_event("ready", timeout=1.0):
            return
        self.bridge.clear_event("ready")
        self.bridge.broadcast({"type": "hello", "message": "请回传 ready"})
        if not self.bridge.wait_client_event("ready", timeout=timeout):
            raise TimeoutError("浏览器 ready 超时")

    def _run_one_trial(self, ctx: TrialContext) -> None:
        self.bridge.clear_reject()
        try:
            self.sm.run_trial(ctx)
        finally:
            if ctx.rejected:
                self._reject_count += 1
                self._broadcast_operator_state()

    def run_adapt(self) -> None:
        self._current_phase = "adapt"
        self._anim = "none"
        self._learn_step = None
        self._current_object = self.config.object_name
        self._current_scene = self.config.scene
        self.events.emit("phase_start", phase="adapt")
        self.markers.push(format_payload("phase_start", phase="adapt"))
        self.bridge.broadcast({"type": "session", "status": "running", "phase": "adapt"})

        self._wait_continue(
            "adapt_welcome",
            "环境适应",
            "你将看到第一人称双手与桌面上的目标物。任务是：根据提示，在脑中想象用左手或右手去抓取。实验过程请尽量身体静止、不要真实动手。",
            "我明白了",
        )
        self.bridge.broadcast(
            {
                "type": "stage",
                "phase": "adapt",
                "stage": "idle",
                "trial_id": None,
                "label": None,
                "hand": "none",
                "anim": "none",
                "duration_s": 0,
                "object": self._current_object,
                "scene": self._current_scene,
                "learn_step": None,
                "transition_amp": "micro",
            }
        )
        self.bridge.broadcast(
            {
                "type": "hud",
                "text": "熟悉场景",
                "subtext": f"看看双手与{zh_object(self._current_object)}",
                "show_cross": False,
            }
        )
        wait_until(
            local_clock() + 3.0,
            is_paused=self.bridge.is_paused,
            should_abort=self.bridge.should_abort,
        )

        for i, lab in enumerate([1, 2], start=1):
            self._anim = "full_grasp"
            ctx = TrialContext(
                trial_id=i,
                label=lab,
                object=self._current_object,
                scene=self._current_scene,
                phase="adapt",
                transition_amp="micro",
            )
            self._run_one_trial(ctx)

        self._wait_continue(
            "adapt_done",
            "适应结束",
            "若已理解「想象抓取、身体不动」，请继续进入学习阶段。",
            "进入学习",
        )
        self.events.emit("phase_end", phase="adapt")
        self.markers.push(format_payload("phase_end", phase="adapt"))

    def run_learn(self) -> None:
        self._current_phase = "learn"
        n = self.config.learn_trials_per_step
        steps = [
            (1, "full_grasp", "完整动作观察：观看抓取并取走目标物，只需观察"),
            (2, "reach", "弱辅助：手会明显前伸靠近物体但不抓握，请同步想象抓取"),
            (3, "none", "无辅助：画面静止，请自主完成左右手想象"),
        ]
        self.events.emit("phase_start", phase="learn")
        self.markers.push(format_payload("phase_start", phase="learn"))
        self.bridge.broadcast({"type": "session", "status": "running", "phase": "learn"})

        import random

        for step, anim, blurb in steps:
            self._learn_step = step
            self._anim = anim
            self._wait_continue(
                f"learn_step_{step}",
                f"学习 Step {step}",
                blurb,
                "开始本步",
            )
            self.events.emit(
                "learn_step_start",
                phase="learn",
                learn_step=step,
            )
            self.markers.push(
                format_payload("learn_step_start", phase="learn", learn_step=step)
            )
            rng = random.Random(
                None if self.config.seed is None else self.config.seed + step
            )
            labels = build_label_schedule(n, rng)
            for i, lab in enumerate(labels, start=1):
                ctx = TrialContext(
                    trial_id=step * 100 + i,
                    label=lab,
                    object=self._current_object,
                    scene=self._current_scene,
                    phase="learn",
                    transition_amp="micro",
                    learn_step=step,
                )
                self._run_one_trial(ctx)
            self.events.emit(
                "learn_step_end",
                phase="learn",
                learn_step=step,
            )
            self.markers.push(
                format_payload("learn_step_end", phase="learn", learn_step=step)
            )

        self.events.emit("phase_end", phase="learn")
        self.markers.push(format_payload("phase_end", phase="learn"))
        self._learn_step = None
        self._anim = "none"

    def run_gate(self) -> None:
        self.bridge.broadcast({"type": "session", "status": "gate"})
        self.events.emit("phase_start", phase="gate")
        self.markers.push(format_payload("phase_start", phase="gate"))
        self._wait_continue(
            "gate",
            "正式采集准入（人工确认）",
            "请操作者确认受试者已能独立、清晰完成左右手抓取想象后继续。本阶段数据将进入主数据集。\n\n"
            "被试空格无效。请操作者：按 G、点左上角「代确认」，或点本弹窗按钮。",
            "确认准入，开始正式采集",
            allow_subject=False,
        )
        self.events.emit("phase_end", phase="gate")
        self.markers.push(format_payload("phase_end", phase="gate"))

    def run_acquire(self) -> List[int]:
        self._current_phase = "acquire"
        self._anim = "none"
        self._learn_step = None
        self.bridge.broadcast(
            {"type": "session", "status": "running", "phase": "acquire"}
        )
        self.events.emit("phase_start", phase="acquire")
        self.markers.push(format_payload("phase_start", phase="acquire"))

        rows = schedule_acquire(
            self.config.acquire_trials, seed=self.config.seed
        )
        if not self.config.rotate_objects:
            rows = [
                (tid, lab, self.config.object_name, sc) for tid, lab, _, sc in rows
            ]
        if not self.config.rotate_scenes:
            rows = [
                (tid, lab, obj, self.config.scene) for tid, lab, obj, _ in rows
            ]

        enriched: List[TrialContext] = []
        for i, (tid, lab, obj, sc) in enumerate(rows):
            if i + 1 < len(rows):
                _, _, nobj, nsc = rows[i + 1]
                if nsc != sc:
                    tamp = "scene"
                elif nobj != obj:
                    tamp = "swap"
                else:
                    tamp = "micro"
            else:
                tamp = "micro"
            enriched.append(
                TrialContext(
                    trial_id=tid,
                    label=lab,
                    object=obj,
                    scene=sc,
                    phase="acquire",
                    transition_amp=tamp,
                )
            )

        labels_out: List[int] = []
        prev_obj: Optional[str] = None
        prev_scene: Optional[str] = None
        for ctx in enriched:
            if prev_scene is not None and ctx.scene != prev_scene:
                self.events.emit(
                    "scene_change",
                    phase="acquire",
                    trial_id=ctx.trial_id,
                    scene=ctx.scene,
                    object=ctx.object,
                )
                self.markers.push(
                    format_payload(
                        "scene_change", trial_id=ctx.trial_id, phase="acquire"
                    )
                )
                self.on_console(
                    f"[scene_change] → {ctx.scene} ({zh_scene(ctx.scene)})"
                )
            elif prev_obj is not None and ctx.object != prev_obj:
                self.events.emit(
                    "object_change",
                    phase="acquire",
                    trial_id=ctx.trial_id,
                    object=ctx.object,
                    scene=ctx.scene,
                )
                self.markers.push(
                    format_payload(
                        "object_change", trial_id=ctx.trial_id, phase="acquire"
                    )
                )
                self.on_console(
                    f"[object_change] → {ctx.object} ({zh_object(ctx.object)})"
                )
            self._run_one_trial(ctx)
            labels_out.append(ctx.label)
            prev_obj, prev_scene = ctx.object, ctx.scene

        self.events.emit("phase_end", phase="acquire")
        self.markers.push(format_payload("phase_end", phase="acquire"))
        return labels_out

    def run_all(self) -> None:
        self.bridge.clear_event("abort")
        self.bridge.paused = False
        self.bridge.clear_reject()
        self._reject_count = 0
        self.bridge.broadcast({"type": "session", "status": "running"})
        self._broadcast_operator_state()
        try:
            if not self.config.skip_adapt:
                self.run_adapt()
            if not self.config.skip_learn:
                self.run_learn()
            if not self.config.skip_gate:
                self.run_gate()
            schedule = self.run_acquire()
            self.bridge.broadcast({"type": "session", "status": "done"})
            self.bridge.broadcast(
                {
                    "type": "hud",
                    "text": "本会话结束",
                    "subtext": f"感谢配合 · reject={self._reject_count}",
                    "show_cross": False,
                }
            )
            self.on_console(f"正式标签顺序: {schedule}")
            self.on_console(f"reject 计数: {self._reject_count}")
        except SessionAbort:
            self._aborted = True
            self.events.emit("session_abort", phase=self._current_phase)
            self.bridge.broadcast(
                {
                    "type": "session",
                    "status": "error",
                    "message": "操作者已中止会话",
                }
            )
            self.bridge.broadcast(
                {
                    "type": "hud",
                    "text": "会话已中止",
                    "subtext": "操作者按下紧急停止",
                    "show_cross": False,
                }
            )
            self.on_console("会话被操作者中止")
        finally:
            self._broadcast_operator_state()
