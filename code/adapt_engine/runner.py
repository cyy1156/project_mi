"""轮控制器（三模式共用核心）+ 离线回放入口。

三种部署形态共用 RoundController：
  - offline-replay：方案 25-A0 爬坡曲线（Stieger/自采回放）
  - batch：采集流程轮间后台微调（真实游戏会话调用）
  - runtime：游戏协同轮内分组微调（事件驱动）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .constants import DEFAULT_CONSTANTS, SystemConstants
from .drift import DriftAction, DriftGuard
from .ft import FTRecipe, IncrementalFinetuner, ReplayPool
from .quiz import AdmissionGate, CurvePoint, QuizStore, QuizTrial
from .readout import judge_trial, serial_gating


@dataclass
class RoundSplit:
    ft_trials: List[int]      # trial 索引（微调用）
    quiz_trials: List[int]    # trial 索引（小考卷）


def split_round(trial_indices: List[int], c: SystemConstants) -> RoundSplit:
    """前 2 子块微调 / 末子块小考（采集流程 v2.0 §4.2）。"""
    n = len(trial_indices)
    if n != c.trials_per_round:
        raise ValueError(f"round 须 {c.trials_per_round} 试次，得到 {n}")
    return RoundSplit(
        ft_trials=trial_indices[: c.ft_trials_per_round],
        quiz_trials=trial_indices[c.ft_trials_per_round :],
    )


class RoundController:
    """标定轮闭环：FT（前 12）→ 小考入库（后 6）→ 评估 → 准入判定。

    参数
    ----
    windows_of_trial : Callable[[int], np.ndarray]  → (n_j, 8, 750) 判定窗
    label_of_trial   : Callable[[int], int]
    predict_window   : Callable[[np.ndarray], Dict] → {"pred": int}
    """

    def __init__(
        self,
        finetuner: IncrementalFinetuner,
        quiz: QuizStore,
        gate: AdmissionGate,
        *,
        constants: SystemConstants = DEFAULT_CONSTANTS,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.fin = finetuner
        self.quiz = quiz
        self.gate = gate
        self.c = constants
        self.log = logger or (lambda s: None)
        self.k_ft = 0
        self.round_no = 0

    def run_calibration_round(
        self,
        trial_indices: List[int],
        *,
        windows_of_trial: Callable,
        label_of_trial: Callable,
        predict_window: Callable,
        frozen: bool = False,
    ) -> Dict:
        split = split_round(trial_indices, self.c)

        # 1) 小考入库（先入卷，防任何路径把它喂进 FT；无效/缺窗跳过）
        for ti in split.quiz_trials:
            w = np.asarray(windows_of_trial(ti), dtype=np.float32)
            if len(w) == 0:
                continue
            self.quiz.add(
                QuizTrial(
                    trial_id=int(ti),
                    label=int(label_of_trial(ti)),
                    judgment_windows=w,
                )
            )

        # 2) 微调（前 12；冻结态跳过；无效/缺窗跳过）
        ft_ok = [ti for ti in split.ft_trials if len(np.asarray(windows_of_trial(ti))) > 0]
        if ft_ok and not frozen:
            X = np.concatenate([windows_of_trial(ti) for ti in ft_ok], axis=0)
            y = np.concatenate(
                [np.full(len(windows_of_trial(ti)), int(label_of_trial(ti))) for ti in ft_ok]
            )
            rec = self.fin.train_round(X, y, frozen=False)
        else:
            rec = {"frozen": frozen or not ft_ok, "n": 0}
        self.k_ft += len(ft_ok)
        self.round_no += 1

        # 3) 小考评估（曲线点）+ 准入判定
        pt = self.quiz.evaluate(predict_window, round_no=self.round_no, k_ft=self.k_ft)
        dec = self.gate.update(pt.acc, pt.n_quiz, self.round_no)
        self.log(
            f"[cal R{self.round_no}] k_ft={self.k_ft} quiz_n={pt.n_quiz} acc={pt.acc:.3f} → {dec.status}"
        )
        return {"train": rec, "curve": pt, "gate": dec}


def replay_offline(
    model_holder,
    *,
    X: np.ndarray,
    y: np.ndarray,
    trial_ids: np.ndarray,
    windows_per_trial: int,
    n_rounds: int,
    constants: SystemConstants = DEFAULT_CONSTANTS,
    replay_pool: Optional[ReplayPool] = None,
    recipe: Optional[FTRecipe] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict:
    """离线回放（25-A0 / 采集后复盘）：整库 trial 序列 → 逐轮标定闭环 → 爬坡曲线。

    X: (N_win, 8, 750)，窗按 trial 顺序、每 trial windows_per_trial 个；
    trial_ids: (N_win,) 与 X 对齐；模型三分类头在 model_holder.model。
    """
    import torch  # noqa: F401

    uniq = list(dict.fromkeys(int(t) for t in trial_ids))
    per_trial: Dict[int, List[int]] = {}
    for i, t in enumerate(trial_ids):
        per_trial.setdefault(int(t), []).append(i)

    def windows_of(ti: int) -> np.ndarray:
        return X[per_trial[ti]]

    def label_of(ti: int) -> int:
        return int(y[per_trial[ti][0]])

    fin = IncrementalFinetuner(
        model_holder.model,
        recipe or FTRecipe(),
        replay_pool=replay_pool,
        device="cpu",
    )

    def predict_window(w: np.ndarray) -> Dict:
        return model_holder.predict(w)

    ctrl = RoundController(
        fin, QuizStore(), AdmissionGate(constants), constants=constants, logger=log
    )
    results = []
    n_rounds = min(n_rounds, len(uniq) // constants.trials_per_round)
    for r in range(n_rounds):
        idx = uniq[r * constants.trials_per_round : (r + 1) * constants.trials_per_round]
        results.append(
            ctrl.run_calibration_round(
                idx,
                windows_of_trial=windows_of,
                label_of_trial=label_of,
                predict_window=predict_window,
            )
        )
    return {"controller": ctrl, "rounds": results, "curve": ctrl.quiz.curve}
