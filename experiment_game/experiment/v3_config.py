"""v3 探针会话常量：从 config/v3_session.yaml 加载。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "v3_session.yaml"
_PROD_CONFIG_PATH = _CONFIG_PATH

PROTOCOL_LOCKED_ALLOW: Set[str] = {
    "blocks",
    "trials_per_block",
    "baseline_rest_s",
    "block_gap_s",
}


def _apply_yaml_dict(cfg: "V3Config", raw: Dict[str, Any]) -> None:
    for k, v in raw.items():
        if not hasattr(cfg, k):
            continue
        if k in ("mu_hz", "beta_l_hz", "beta_h_hz") and isinstance(v, list):
            v = tuple(float(x) for x in v)
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
        return {}


def _build_judgment_times(step_s: float, imagine_s: float) -> Tuple[float, ...]:
    out = []
    t = float(step_s)
    while t <= float(imagine_s) + 1e-9:
        out.append(round(t, 6))
        t += float(step_s)
    return tuple(out)


@dataclass
class V3Config:
    blocks: int = 2
    trials_per_block: int = 18
    baseline_rest_s: float = 60.0
    block_gap_s: float = 90.0
    guidance_timeout_s: float = 600.0

    judgment_step_s: float = 0.6
    judgment_half_weight_until_s: float = 2.4
    score_early_stop: float = 5.0
    score_invalid_max: float = 3.0
    score_valid_min: float = 4.0
    wrong_class_abort: float = 5.0
    primary_judge_s: float = 4.0
    judgment_times: tuple = field(default_factory=tuple)

    mu_hz: tuple = (8.0, 13.0)
    beta_l_hz: tuple = (13.0, 20.0)
    beta_h_hz: tuple = (20.0, 30.0)

    prep_s: float = 2.0
    cue_s: float = 2.0
    imagine_s: float = 6.0
    iti_s: float = 3.0
    task_p_on: float = 0.6

    eeg_frame_interval_s: float = 0.3
    eeg_frame_window_s: float = 2.5
    save_trial_segments: bool = True

    s3_task_ckpt: str = ""
    s3_three_ckpt: str = ""

    signal_quality_enabled: bool = True
    signal_min_median_std_uv: float = 3.0
    signal_min_peak_to_peak_uv: float = 8.0
    signal_max_peak_uv: float = 1000.0
    signal_min_per_channel_std_uv: float = 2.0
    signal_min_active_channels: int = 3
    signal_max_channel_std_ratio: float = 20.0
    signal_max_median_std_uv: float = 60.0
    signal_max_ptp_uv: float = 400.0
    signal_min_car_std_uv: float = 2.0
    signal_max_common_mode_ratio: float = 0.85

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

    def standards(self) -> Dict[str, Any]:
        return {
            "mu_hz": self.mu_hz,
            "beta_l_hz": self.beta_l_hz,
            "beta_h_hz": self.beta_h_hz,
            "mu_erd_contra_ok": -15.0,
            "mu_erd_contra_excellent": -35.0,
            "laterality_pp_ok": 8.0,
            "mu_vs_betal_slack": 5.0,
            "rest_mu_frac_ok": 0.40,
            "rest_mu_frac_excellent": 0.55,
            "time_drop_ok": 0.08,
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["judgment_times"] = list(self.judgment_times)
        d["mu_hz"] = list(self.mu_hz)
        d["beta_l_hz"] = list(self.beta_l_hz)
        d["beta_h_hz"] = list(self.beta_h_hz)
        return d

    def trial_total_s(self) -> float:
        return float(self.prep_s + self.cue_s + self.imagine_s + self.iti_s)

    def apply_overrides(
        self,
        overrides: Optional[Dict[str, Any]],
        *,
        protocol_locked: bool = False,
    ) -> List[str]:
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
                elif isinstance(cur, tuple) and isinstance(v, (list, tuple)):
                    setattr(self, k, tuple(float(x) for x in v))
                else:
                    setattr(self, k, v)
            except (TypeError, ValueError):
                ignored.append(k)
        return ignored

    def verify_errors(self) -> List[str]:
        errs: List[str] = []
        self.judgment_times = _build_judgment_times(self.judgment_step_s, self.imagine_s)
        if self.blocks < 1 or self.blocks > 4:
            errs.append("blocks 须在 1–4")
        if self.trials_per_block < 6 or self.trials_per_block > 36:
            errs.append("trials_per_block 须在 6–36")
        if self.baseline_rest_s < 10 or self.baseline_rest_s > 300:
            errs.append("baseline_rest_s 须在 10–300")
        if self.block_gap_s < 30 or self.block_gap_s > 300:
            errs.append("block_gap_s 须在 30–300")
        if self.prep_s < 0.5 or self.cue_s < 0.5 or self.imagine_s < 1.0 or self.iti_s < 0.5:
            errs.append("时序过短：prep/cue≥0.5s、MI≥1s、ITI≥0.5s")
        task = Path(__file__).resolve().parents[2] / self.s3_task_ckpt
        three = Path(__file__).resolve().parents[2] / self.s3_three_ckpt
        if not task.is_file():
            errs.append(f"缺 task 权重: {task}")
        if not three.is_file():
            errs.append(f"缺 three 权重: {three}")
        return errs

    @classmethod
    def load_yaml(cls, path: Path | str | None = None) -> "V3Config":
        p = Path(path) if path else _CONFIG_PATH
        cfg = cls()
        if p.resolve() != _PROD_CONFIG_PATH.resolve() and _PROD_CONFIG_PATH.is_file():
            _apply_yaml_dict(cfg, _load_yaml_raw(_PROD_CONFIG_PATH))
        _apply_yaml_dict(cfg, _load_yaml_raw(p))
        cfg.judgment_times = _build_judgment_times(cfg.judgment_step_s, cfg.imagine_s)
        return cfg
