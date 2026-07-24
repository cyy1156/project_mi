"""离线适配：experiment_game 会话 → preprocess_lab 兼容张量。"""

from __future__ import annotations

from experiment_game.offline.pipeline import EpochBundle, preprocess_session

__all__ = ["EpochBundle", "preprocess_session"]
