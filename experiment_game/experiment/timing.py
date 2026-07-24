"""试次时序常量（秒）— 与 docs/marker_spec.md 一致。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrialTiming:
    fixation_s: float = 2.0
    cue_s: float = 2.0
    mi_s: float = 4.0
    post_mi_hold_s: float = 1.0
    rest_s: float = 4.0
    transition_s: float = 3.0

    @property
    def total_s(self) -> float:
        return (
            self.fixation_s
            + self.cue_s
            + self.mi_s
            + self.post_mi_hold_s
            + self.rest_s
            + self.transition_s
        )


DEFAULT_TIMING = TrialTiming()

# 仅用于联调/界面验收，不用于正式采数
FAST_TIMING = TrialTiming(
    fixation_s=0.6,
    cue_s=0.8,
    mi_s=1.6,
    post_mi_hold_s=0.4,
    rest_s=0.8,
    transition_s=0.6,
)
