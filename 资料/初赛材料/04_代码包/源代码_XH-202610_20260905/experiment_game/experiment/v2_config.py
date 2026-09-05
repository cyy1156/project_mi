"""v2 会话常量：从 config/v2_session.yaml 加载，运行时可改。

默认值保留 SystemConstants 兼容；YAML 优先。无 PyYAML 时回退内置默认（保证 import 不崩）。
UI overrides 经 apply_overrides；采集冻结锁用 PROTOCOL_LOCKED_ALLOW 白名单。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "v2_session.yaml"
_PROD_CONFIG_PATH = _CONFIG_PATH

# 采集冻结锁开启时，允许从 UI overrides 覆盖的字段（G1 任务量）；
# G2 时序 / G3 高级常量不在白名单 → orchestrator 侧忽略。
PROTOCOL_LOCKED_ALLOW: Set[str] = {
    "cal_rounds_min",
    "cal_rounds_max",
    "game_rounds",
    "game_rounds_min",
    "game_rounds_max",
    "game_trials_per_round",
    "trials_per_round",
    "ft_trials_per_round",
    "quiz_trials_per_round",
    "s3_task_ckpt",
    "s3_three_ckpt",
}


def _apply_yaml_dict(cfg: "V2Config", raw: Dict[str, Any]) -> None:
    for k, v in raw.items():
        if not hasattr(cfg, k):
            continue
        if k == "judgment_times" and isinstance(v, list):
            v = tuple(float(x) for x in v)
        try:
            setattr(cfg, k, type(getattr(cfg, k))(v) if not isinstance(v, (list, tuple)) else v)
        except (TypeError, ValueError):
            pass


def _load_yaml_raw(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return _fallback_parse(path)


from experiment_game.experiment.openbmi_align_config import (
    BASELINE_BEFORE_CUE_S,
    HOP_S,
    MI_TASK_SEC_DEFAULT,
    ONLINE_WINDOW_MODE_OPENBMI,
    WIN_S,
    rebuild_judgment_times,
)


@dataclass
class V2Config:
    cal_rounds_min: int = 4
    cal_rounds_max: int = 6
    trials_per_round: int = 18
    ft_trials_per_round: int = 12
    quiz_trials_per_round: int = 6
    subblock_size: int = 6
    cal_round_gap_s: float = 180.0

    gate_enter_three: float = 0.60
    gate_min_quiz_trials: int = 12

    game_rounds: int = 2
    game_rounds_min: int = 2
    game_rounds_max: int = 3
    game_trials_per_round: int = 16
    judgment_times: tuple = field(default_factory=tuple)

    # —— v3 权重游戏测试（game_mode="v3_test"）：跳过标定/准入，直接 20 试次 ——
    game_mode: str = "legacy"  # "v3_test" | "legacy"
    use_v3_weights: bool = True  # 用被试 v3 最终权重（current/members + overlay）
    subject_models_dir: str = ""  # orchestrator 注入 data/subjects/<sid>/models
    v3test_rest_s: float = 5.0
    v3test_cue_s: float = 1.0
    v3test_mi_s: float = 10.0
    v3test_judge_interval_s: float = 0.5
    v3test_consecutive: int = 5
    v3test_n_rest: int = 10
    v3test_n_left: int = 5
    v3test_n_right: int = 5
    v3test_rest_points: float = 0.5
    v3test_mi_points: float = 1.0

    primary_judge_mode: str = "majority"
    ft_min_valid_trials: int = 6

    group_lr: float = 1e-4
    replay_ratio: float = 0.10
    drift_patience: int = 2
    # 0 = 关闭串行门控，three 头预测一律采用
    task_p_on: float = 0.0

    prep_s: float = 2.0
    cue_s: float = 1.0
    imagine_s: float = MI_TASK_SEC_DEFAULT
    iti_s: float = 3.0
    inter_trial_rest_s: float = 4.0
    online_window_mode: str = ONLINE_WINDOW_MODE_OPENBMI
    win_s: float = WIN_S
    win_hop_s: float = HOP_S
    baseline_before_cue_s: float = BASELINE_BEFORE_CUE_S

    ft_epochs: int = 5
    ft_batch_size: int = 32
    ft_weight_decay: float = 1e-4

    s3_task_ckpt: str = ""
    s3_three_ckpt: str = ""

    # 关闭信号质量门控：所有判定窗进入模型
    signal_quality_enabled: bool = False
    signal_min_median_std_uv: float = 0.0
    signal_min_peak_to_peak_uv: float = 0.0
    signal_max_peak_uv: float = 1.0e9
    signal_min_per_channel_std_uv: float = 0.0
    signal_min_active_channels: int = 0
    signal_max_channel_std_ratio: float = 1.0e9
    signal_max_median_std_uv: float = 1.0e9
    signal_max_ptp_uv: float = 1.0e9
    signal_min_car_std_uv: float = 0.0
    signal_max_common_mode_ratio: float = 1.0e9

    def signal_quality_config(self):
        from experiment_game.experiment.signal_quality import SignalQualityConfig

        return SignalQualityConfig(
            enabled=self.signal_quality_enabled,
            min_median_std_uv=self.signal_min_median_std_uv,
            min_peak_to_peak_uv=self.signal_min_peak_to_peak_uv,
            max_peak_uv=self.signal_max_peak_uv,
            min_per_channel_std_uv=self.signal_min_per_channel_std_uv,
            min_active_channels=self.signal_min_active_channels,
            max_channel_std_ratio=self.signal_max_channel_std_ratio,
            max_median_std_uv=self.signal_max_median_std_uv,
            max_ptp_uv=self.signal_max_ptp_uv,
            min_car_std_uv=self.signal_min_car_std_uv,
            max_common_mode_ratio=self.signal_max_common_mode_ratio,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["judgment_times"] = list(self.judgment_times)
        return d

    def trial_total_s(self) -> float:
        return float(
            self.prep_s + self.cue_s + self.imagine_s + self.iti_s + self.inter_trial_rest_s
        )

    def apply_overrides(
        self,
        overrides: Optional[Dict[str, Any]],
        *,
        protocol_locked: bool = False,
    ) -> List[str]:
        """应用 UI overrides；返回被忽略的键（冻结锁白名单外）。"""
        ignored: List[str] = []
        if not overrides:
            return ignored
        known = {f.name for f in fields(self)}
        for k, v in overrides.items():
            if k not in known or k == "judgment_times":
                continue
            if protocol_locked and k not in PROTOCOL_LOCKED_ALLOW:
                ignored.append(k)
                continue
            try:
                cur = getattr(self, k)
                if isinstance(cur, bool):
                    setattr(self, k, bool(v))
                elif isinstance(cur, int) and not isinstance(cur, bool):
                    setattr(self, k, int(v))
                elif isinstance(cur, float):
                    setattr(self, k, float(v))
                elif isinstance(cur, str):
                    setattr(self, k, str(v))
                else:
                    setattr(self, k, v)
            except (TypeError, ValueError):
                ignored.append(k)
        return ignored

    def verify_errors(self) -> List[str]:
        """范围校验，返回错误文案（空=通过）。不抛异常。"""
        errs: List[str] = []
        rebuild_judgment_times(self)
        if self.ft_trials_per_round + self.quiz_trials_per_round != self.trials_per_round:
            errs.append(
                f"ft({self.ft_trials_per_round})+quiz({self.quiz_trials_per_round})"
                f" 须等于 trials_per_round({self.trials_per_round})"
            )
        if self.trials_per_round % self.subblock_size != 0:
            errs.append("trials_per_round 须能被 subblock_size 整除")
        if self.gate_min_quiz_trials < self.quiz_trials_per_round:
            errs.append("gate_min_quiz_trials 须 ≥ quiz_trials_per_round")
        if self.cal_rounds_min > self.cal_rounds_max:
            errs.append("cal_rounds_min 须 ≤ cal_rounds_max")
        if not (self.game_rounds_min <= self.game_rounds <= self.game_rounds_max):
            errs.append(
                f"game_rounds={self.game_rounds} 须在 "
                f"[{self.game_rounds_min}, {self.game_rounds_max}]"
            )
        if not (0.0 < self.gate_enter_three <= 1.0):
            errs.append("gate_enter_three 须在 (0, 1]")
        if not (1e-6 <= self.group_lr <= 1e-2):
            errs.append("group_lr 须在 [1e-6, 1e-2]（推荐 1e-4）")
        if not (0.0 <= self.replay_ratio <= 0.5):
            errs.append("replay_ratio 须在 [0, 0.5]")
        if self.prep_s < 0 or self.cue_s < 0 or self.imagine_s < 1.0 or self.iti_s < 0:
            errs.append("时序无效：prep/cue/iti≥0、MI≥1s")
        if self.inter_trial_rest_s < 0 or self.inter_trial_rest_s > 10:
            errs.append("inter_trial_rest_s 须在 0–10")
        if self.trial_total_s() > 20.0:
            errs.append(f"单 trial 合计 {self.trial_total_s():.1f}s 过长（建议 ≤15s）")
        if self.ft_epochs < 1 or self.ft_epochs > 50:
            errs.append("ft_epochs 须在 1–50")
        if self.ft_batch_size < 1 or self.ft_batch_size > 256:
            errs.append("ft_batch_size 须在 1–256")
        if self.game_mode == "v3_test":
            if self.v3test_rest_s <= 0 or self.v3test_cue_s < 0 or self.v3test_mi_s <= 0:
                errs.append("v3test 时序无效：rest/mi > 0、cue ≥ 0")
            if not (0.1 <= self.v3test_judge_interval_s <= 2.0):
                errs.append("v3test_judge_interval_s 须在 0.1–2.0")
            if self.v3test_consecutive < 1:
                errs.append("v3test_consecutive 须 ≥ 1")
            if min(self.v3test_n_rest, self.v3test_n_left, self.v3test_n_right) < 0:
                errs.append("v3test 试次数不得为负")
            if self.use_v3_weights and not self.subject_models_dir:
                # orchestrator 启动会话时注入；纯配置校验阶段允许为空
                pass
        return errs

    def verify(self) -> None:
        errs = self.verify_errors()
        if errs:
            raise AssertionError("; ".join(errs))

    @classmethod
    def load_yaml(cls, path: Path | str | None = None) -> "V2Config":
        """从 YAML 加载；pilot 等 preset 先合并 prod 再 overlay（保留 ckpt 路径等）。"""
        p = Path(path) if path else _CONFIG_PATH
        cfg = cls()
        if p.resolve() != _PROD_CONFIG_PATH.resolve() and _PROD_CONFIG_PATH.is_file():
            _apply_yaml_dict(cfg, _load_yaml_raw(_PROD_CONFIG_PATH))
        if p.is_file():
            _apply_yaml_dict(cfg, _load_yaml_raw(p))
        rebuild_judgment_times(cfg)
        cfg.verify()
        return cfg


def _fallback_parse(p: Path) -> Dict[str, Any]:
    """无 PyYAML 时的极简 KV 解析（支持 int/float/str/list）。"""
    out: Dict[str, Any] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v.startswith("["):
            out[k] = [float(x.strip()) for x in v[1:-1].split(",") if x.strip()]
        elif v.replace(".", "").replace("-", "").isdigit():
            out[k] = float(v) if "." in v else int(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    c = V2Config.load_yaml()
    print("已加载 v2 会话常量：")
    for k in ["cal_rounds_min", "gate_enter_three", "group_lr", "judgment_times", "trials_per_round"]:
        print(f"  {k} = {getattr(c, k)}")
