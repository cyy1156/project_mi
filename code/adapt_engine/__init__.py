"""adapt_engine · 适配引擎核心库（游戏 v2 开发计划 M3）。

一份代码三处复用：offline-replay（25-A0 爬坡曲线）· batch（采集轮间微调）· runtime（游戏分组微调）。

三个可替换接口（M0 冻结签名）：
  ModelRegistry  —— 24-E 集成 / 24-W 窗长模型组
  ReadoutPolicy  —— 24-V 加权投票（readout.confidence_weighted_majority）
  FTRecipe       —— 25-G2 回放比例 / 25-G3 轻增广（aug_fn）
"""

from .constants import DEFAULT_CONSTANTS, SystemConstants
from .drift import DriftAction, DriftGuard, DriftRecord
from .ft import FTRecipe, IncrementalFinetuner, ReplayPool
from .quiz import AdmissionGate, CurvePoint, GateDecision, QuizStore, QuizTrial
from .readout import confidence_weighted_majority, judge_trial, serial_gating, TrialVerdict
from .scoring_v21 import (
    OnlineScoreTracker,
    ScoringConfig,
    TrialScoreV21,
    build_judgment_times,
    score_trial_from_judgments,
    tick_weight,
)
from .registry import HeadEntry, ModelRegistry, load_head
from .runner import replay_offline, RoundController, RoundSplit, split_round

__all__ = [
    "DEFAULT_CONSTANTS", "SystemConstants",
    "DriftAction", "DriftGuard", "DriftRecord",
    "FTRecipe", "IncrementalFinetuner", "ReplayPool",
    "AdmissionGate", "CurvePoint", "GateDecision", "QuizStore", "QuizTrial",
    "confidence_weighted_majority", "judge_trial", "serial_gating", "TrialVerdict",
    "OnlineScoreTracker", "ScoringConfig", "TrialScoreV21",
    "build_judgment_times", "score_trial_from_judgments", "tick_weight",
    "HeadEntry", "ModelRegistry", "load_head",
    "replay_offline", "RoundController", "RoundSplit", "split_round",
]
