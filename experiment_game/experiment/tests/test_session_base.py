"""session_base 公共装配。"""

from __future__ import annotations

from experiment_game.experiment.session_base import SessionRunnerBase, SessionServices, attach_eeg_health


def test_session_runner_base_requires_run():
    class Dummy:
        def emit(self, *a, **k):
            return None

        def push(self, *a, **k):
            return None

        def broadcast(self, *a, **k):
            return None

    svc = SessionServices(Dummy(), Dummy(), Dummy(), lambda _s: None)  # type: ignore[arg-type]
    base = SessionRunnerBase(svc)
    try:
        base.run()
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_attach_eeg_health_none_when_disabled():
    assert attach_eeg_health(None, SessionServices(None, None, None), enabled=False) is None  # type: ignore[arg-type]
