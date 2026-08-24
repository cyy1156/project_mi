"""D8 自测入口：单元测试 + 离线回放 + 降级会话冒烟。

用法（项目根目录）：
  python -m experiment_game.experiment.self_test_d8
  python -m experiment_game.experiment.self_test_d8 --quick   # 跳过会话冒烟
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "code"))

PASS = "PASS"
FAIL = "FAIL"


def _run_case(name: str, fn: Callable[[], None]) -> Tuple[str, str]:
    try:
        fn()
        return name, PASS
    except Exception as exc:
        return name, f"{FAIL}: {exc}\n{traceback.format_exc()}"


def _run_module_tests(mod) -> List[Tuple[str, str]]:
    out = []
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            out.append(_run_case(name, getattr(mod, name)))
    return out


def test_v2_config_load() -> None:
    from experiment_game.experiment.v2_config import V2Config

    prod = V2Config.load_yaml(_ROOT / "experiment_game/config/v2_session.yaml")
    pilot = V2Config.load_yaml(_ROOT / "experiment_game/config/v2_session.pilot.yaml")
    assert len(prod.judgment_times) == 10
    assert prod.judgment_times[0] == 0.6 and prod.judgment_times[-1] == 6.0
    assert pilot.cal_rounds_min == 2
    assert prod.scoring_config().score_early_stop == 5.0


def test_scoring_replay_roundtrip() -> None:
    from experiment_game.experiment.scoring_replay import replay_session

    events = [
        {"event": "trial_start", "trial_id": 1, "label": 1, "phase": "game"},
        {"event": "judge", "trial_id": 1, "t_rel": 0.6, "pred": 1, "score": 0.5},
        {"event": "judge", "trial_id": 1, "t_rel": 1.2, "pred": 1, "score": 1.0},
        {"event": "judge", "trial_id": 1, "t_rel": 1.8, "pred": 1, "score": 1.5},
        {"event": "judge", "trial_id": 1, "t_rel": 2.4, "pred": 1, "score": 2.0},
        {"event": "judge", "trial_id": 1, "t_rel": 3.0, "pred": 1, "score": 3.0},
        {"event": "judge", "trial_id": 1, "t_rel": 3.6, "pred": 1, "score": 4.0},
        {"event": "judge", "trial_id": 1, "t_rel": 4.2, "pred": 1, "score": 5.0},
        {"event": "score_reach", "trial_id": 1, "t_rel": 4.2, "score": 5.0},
        {"event": "mi_end", "trial_id": 1, "early": True, "reason": "score_5"},
        {"event": "trial_end", "trial_id": 1, "label": 1},
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "events.jsonl"
        p.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")
        rows = replay_session(p)
        assert len(rows) == 1
        assert rows[0]["valid"] is True
        assert rows[0]["early_stop"] is True
        assert rows[0]["score"] == 5.0


def test_session_v2_degraded_smoke() -> None:
    """无 LSL/权重：流程演练 + 自动确认引导。"""
    from experiment_game.experiment import trial_v2 as tv2
    from experiment_game.experiment.events_log import EventLogger
    from experiment_game.experiment.markers import MarkerPublisher
    from experiment_game.experiment.session_v2 import run_v2_session

    tv2._wu = lambda t_end, **kw: None

    class _Bridge:
        def broadcast(self, msg):
            pass

        def is_paused(self):
            return False

        def should_abort(self):
            return False

        def is_rejected(self):
            return False

        def wait_client_event(self, name, timeout=600.0):
            if name == "v2_guidance_confirm":
                return True
            return False

    with tempfile.TemporaryDirectory() as td:
        events = EventLogger(Path(td) / "events.jsonl")
        markers = MarkerPublisher(enabled=False)
        logs: List[str] = []

        summary = run_v2_session(
            events,
            markers,
            _Bridge(),
            on_console=logs.append,
            config_path=str(_ROOT / "experiment_game/config/v2_session.pilot.yaml"),
        )
        events.close()

        assert summary["degraded"] is True
        assert summary["aborted"] is False
        rows = [json.loads(l) for l in events.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        ev = [r["event"] for r in rows]
        assert "v2_guidance_begin" in ev
        assert "round_start" in ev
        assert "judge" not in ev  # 降级无判定
        assert "trial_end" in ev
        assert any("会话完成" in x for x in logs)


def test_adapt_engine_core() -> None:
    from adapt_engine.tests import test_adapt_engine as t

    for name in sorted(dir(t)):
        if name.startswith("test_"):
            getattr(t, name)()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过会话冒烟（较慢）")
    args = ap.parse_args()

    results: List[Tuple[str, str]] = []

    from adapt_engine.tests import test_scoring_v21, test_trial_v2_d8

    results += _run_module_tests(test_scoring_v21)
    results += _run_module_tests(test_trial_v2_d8)
    results.append(_run_case("v2_config_load", test_v2_config_load))
    results.append(_run_case("scoring_replay_roundtrip", test_scoring_replay_roundtrip))
    if not args.quick:
        results.append(_run_case("session_v2_degraded_smoke", test_session_v2_degraded_smoke))
    results.append(_run_case("adapt_engine_core", test_adapt_engine_core))

    n_fail = 0
    print("=" * 60)
    print("D8 自测报告")
    print("=" * 60)
    for name, status in results:
        ok = status == PASS
        if not ok:
            n_fail += 1
        mark = "[OK]" if ok else "[FAIL]"
        print(f"  {mark} {name}")
        if not ok:
            print(status)
    print("-" * 60)
    print(f"合计 {len(results)} 项，通过 {len(results) - n_fail}，失败 {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
