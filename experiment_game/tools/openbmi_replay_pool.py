"""兼容转发：实现见 ``experiment_game.pipeline.openbmi_replay_pool``。"""

from __future__ import annotations

import sys

from experiment_game.pipeline import openbmi_replay_pool as _impl

sys.modules[__name__] = _impl
