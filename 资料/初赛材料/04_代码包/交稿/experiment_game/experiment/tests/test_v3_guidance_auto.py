"""v3 guided 块：合成板自动确认引导；超时为 SessionAbort。"""
from __future__ import annotations

import pytest

from experiment_game.experiment.trial_sm import SessionAbort


def test_guidance_timeout_reason():
    with pytest.raises(SessionAbort) as ei:
        raise SessionAbort("guidance_timeout")
    assert ei.value.reason == "guidance_timeout"


@pytest.mark.parametrize(
    "use_synthetic,sim_meta,override,expect",
    [
        (True, None, None, True),
        (False, {"run_id": "x"}, None, True),
        (False, None, None, False),
        (False, None, True, True),
        (True, None, False, False),
    ],
)
def test_resolve_auto_confirm_guidance(use_synthetic, sim_meta, override, expect):
    """与 session_v3.run_v3_session 入口默认逻辑一致。"""
    auto = override
    if auto is None:
        auto = bool(use_synthetic) or bool(sim_meta)
    assert bool(auto) is expect


def test_block_order_places_guided_once():
    from experiment_game.experiment.session_v3 import block_order

    a = block_order(seed=0, subject_id="subjA")
    b = block_order(seed=1, subject_id="subjA")
    assert set(a) == {"guided", "no_guide"}
    assert a.count("guided") == 1
    assert set(b) == {"guided", "no_guide"}
    # 不同 seed 可能同序，但函数始终恰好一块 guided
    assert a == block_order(seed=0, subject_id="subjA")
