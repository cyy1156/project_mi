"""系统常量（v2.0 计划 §3.6 冻结口径；阶段 C 标定后冻结终值，现场禁改）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SystemConstants:
    # —— 标定轮（采集流程 v2.0 §4）——
    cal_rounds_min: int = 4
    cal_rounds_max: int = 6
    trials_per_round: int = 18          # 3 子块 × 6（2L+2R+2Rest 置换）
    subblock_size: int = 6
    ft_trials_per_round: int = 12       # 前 2 子块微调
    quiz_trials_per_round: int = 6      # 末子块进累积小考卷
    cal_round_gap_s: float = 180.0      # 轮间（引导复习 + 后台 FT）

    # —— 准入门槛 ——
    gate_enter_three: float = 0.60
    gate_min_quiz_trials: int = 12      # 小考卷 ≥ 此数起判

    # —— 游戏协同 ——
    game_trials_per_round: int = 16
    judgment_times: tuple = (3.0, 4.0, 5.0, 6.0)   # cue 后秒判定点
    arm_levels: int = 4

    # —— 在线适配与保护 ——
    group_lr: float = 1e-4
    replay_ratio: float = 0.15
    drift_patience: int = 2
    task_p_on: float = 0.6              # 串行门控：Task 概率不足 → 静息

    # —— 范式（6s 试次）——
    prep_s: float = 2.0
    cue_s: float = 2.0
    imagine_s: float = 6.0
    iti_s: float = 3.0

    def verify(self) -> None:
        assert self.ft_trials_per_round + self.quiz_trials_per_round == self.trials_per_round
        assert self.trials_per_round % self.subblock_size == 0
        assert self.gate_min_quiz_trials >= self.quiz_trials_per_round


DEFAULT_CONSTANTS = SystemConstants()
DEFAULT_CONSTANTS.verify()
