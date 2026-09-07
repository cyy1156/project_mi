"""v4 数据质量检测会话配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiment_game.experiment.inference_v2 import CHANNEL_ORDER

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "v4_session.yaml"


def _load_yaml_raw(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


@dataclass
class V4Config:
    duration_s: float = 90.0
    eval_interval_s: float = 3.0
    eval_window_s: float = 3.0
    pass_streak_required: int = 5
    auto_stop_on_pass: bool = False

    eeg_frame_interval_s: float = 0.25
    save_eeg_csv: bool = True
    lsl_stream_name: str = "OpenBCI_EEG"
    lsl_timeout_s: float = 8.0

    channel_labels: Tuple[str, ...] = tuple(CHANNEL_ORDER)
    scoring_channels: Tuple[str, ...] = tuple(CHANNEL_ORDER)
    unused_channels: Tuple[str, ...] = ()
    unused_allow_rail: bool = False
    fs: int = 250

    signal_quality_enabled: bool = True
    signal_min_median_std_uv: float = 0.50
    signal_max_median_std_uv: float = 8.0
    signal_min_peak_to_peak_uv: float = 0.0
    signal_max_ptp_uv: float = 600.0
    signal_max_peak_uv: float = 600.0
    signal_min_per_channel_std_uv: float = 0.40
    signal_max_per_channel_std_uv: float = 150.0
    signal_min_active_channels: int = 6
    signal_max_channel_std_ratio: float = 3.0
    signal_min_car_std_uv: float = 0.10
    signal_max_common_mode_ratio: float = 1.45

    def _label_indices(self, labels: Tuple[str, ...]) -> Tuple[int, ...]:
        idx: List[int] = []
        for name in labels:
            key = str(name)
            for i, ch in enumerate(self.channel_labels):
                if ch.upper() == key.upper():
                    idx.append(i)
                    break
            else:
                raise ValueError(f"通道 {name} 不在 channel_labels {self.channel_labels}")
        return tuple(idx)

    def scoring_indices(self) -> Tuple[int, ...]:
        return self._label_indices(self.scoring_channels)

    def unused_indices(self) -> Tuple[int, ...]:
        if not self.unused_channels:
            return ()
        return self._label_indices(self.unused_channels)

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
            max_per_channel_std_uv=self.signal_max_per_channel_std_uv,
            scoring_channel_indices=self.scoring_indices(),
            unused_channel_indices=self.unused_indices(),
            unused_allow_rail=self.unused_allow_rail,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["channel_labels"] = list(self.channel_labels)
        d["scoring_channels"] = list(self.scoring_channels)
        d["unused_channels"] = list(self.unused_channels)
        return d

    def apply_overrides(self, overrides: Optional[Dict[str, Any]]) -> List[str]:
        ignored: List[str] = []
        if not overrides:
            return ignored
        known = {f.name for f in fields(self)}
        for k, v in overrides.items():
            if k not in known:
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
                    setattr(self, k, tuple(str(x) for x in v))
                else:
                    setattr(self, k, v)
            except (TypeError, ValueError):
                ignored.append(k)
        return ignored

    def verify_errors(self) -> List[str]:
        errs: List[str] = []
        if self.duration_s < 15 or self.duration_s > 600:
            errs.append("duration_s 须在 15–600")
        if self.eval_window_s < 1.0 or self.eval_window_s > 10.0:
            errs.append("eval_window_s 须在 1–10")
        if self.eval_interval_s < self.eval_window_s * 0.5:
            errs.append("eval_interval_s 过小")
        if self.pass_streak_required < 1 or self.pass_streak_required > 20:
            errs.append("pass_streak_required 须在 1–20")
        if self.signal_min_active_channels < 1 or self.signal_min_active_channels > len(self.scoring_channels):
            errs.append("signal_min_active_channels 须在 1–scoring_channels 数量内")
        for name in self.scoring_channels:
            if name not in self.channel_labels and name.upper() not in {c.upper() for c in self.channel_labels}:
                errs.append(f"scoring_channels 含未知通道 {name}")
        return errs

    @classmethod
    def load_yaml(cls, path: Optional[str | Path] = None) -> "V4Config":
        p = Path(path) if path is not None else _CONFIG_PATH
        raw = _load_yaml_raw(p)
        cfg = cls()
        for k, v in raw.items():
            if not hasattr(cfg, k):
                continue
            if k in ("channel_labels", "scoring_channels", "unused_channels") and isinstance(v, list):
                v = tuple(str(x) for x in v)
            try:
                setattr(cfg, k, type(getattr(cfg, k))(v) if not isinstance(v, (list, tuple)) else v)
            except (TypeError, ValueError):
                pass
        return cfg
