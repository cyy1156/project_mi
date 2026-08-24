"""v2 会话常量：从 config/v2_session.yaml 加载，运行时可改。

默认值保留 SystemConstants 兼容；YAML 优先。无 PyYAML 时回退内置默认（保证 import 不崩）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "v2_session.yaml"
_PROD_CONFIG_PATH = _CONFIG_PATH


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


def _build_judgment_times(step_s: float, imagine_s: float) -> Tuple[float, ...]:
    out = []
    t = float(step_s)
    while t <= float(imagine_s) + 1e-9:
        out.append(round(t, 6))
        t += float(step_s)
    return tuple(out)


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
    arm_levels: int = 4  # 遗留；D8 用 score_early_stop

    # D8 · v2.1 在线判定
    judgment_step_s: float = 0.6
    judgment_half_weight_until_s: float = 2.4
    score_early_stop: float = 5.0
    score_invalid_max: float = 3.0
    score_valid_min: float = 4.0
    wrong_class_abort: float = 5.0
    consecutive_invalid_abort: int = 5
    ft_min_valid_trials: int = 6

    group_lr: float = 1e-4
    replay_ratio: float = 0.15
    drift_patience: int = 2
    task_p_on: float = 0.6

    prep_s: float = 2.0
    cue_s: float = 2.0
    imagine_s: float = 6.0
    iti_s: float = 3.0

    ft_epochs: int = 5
    ft_batch_size: int = 32
    ft_weight_decay: float = 1e-4

    s3_task_ckpt: str = ""
    s3_three_ckpt: str = ""

    def scoring_config(self):
        from adapt_engine.scoring_v21 import ScoringConfig
        return ScoringConfig(
            judgment_step_s=self.judgment_step_s,
            judgment_half_weight_until_s=self.judgment_half_weight_until_s,
            imagine_s=self.imagine_s,
            score_early_stop=self.score_early_stop,
            score_invalid_max=self.score_invalid_max,
            score_valid_min=self.score_valid_min,
            wrong_class_abort=self.wrong_class_abort,
        )

    def verify(self) -> None:
        self.judgment_times = _build_judgment_times(self.judgment_step_s, self.imagine_s)
        assert self.ft_trials_per_round + self.quiz_trials_per_round == self.trials_per_round
        assert self.trials_per_round % self.subblock_size == 0
        assert self.gate_min_quiz_trials >= self.quiz_trials_per_round
        assert self.cal_rounds_min <= self.cal_rounds_max
        assert self.game_rounds_min <= self.game_rounds <= self.game_rounds_max
        assert self.judgment_step_s > 0
        assert self.score_valid_min > self.score_invalid_max

    @classmethod
    def load_yaml(cls, path: Path | str | None = None) -> "V2Config":
        """从 YAML 加载；pilot 等 preset 先合并 prod 再 overlay（保留 ckpt 路径等）。"""
        p = Path(path) if path else _CONFIG_PATH
        cfg = cls()
        if p.resolve() != _PROD_CONFIG_PATH.resolve() and _PROD_CONFIG_PATH.is_file():
            _apply_yaml_dict(cfg, _load_yaml_raw(_PROD_CONFIG_PATH))
        if p.is_file():
            _apply_yaml_dict(cfg, _load_yaml_raw(p))
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
