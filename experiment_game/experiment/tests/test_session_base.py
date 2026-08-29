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


def test_session_runner_inherits_services():
    from experiment_game.experiment.session_runner import SessionRunner

    class Dummy:
        def emit(self, *a, **k):
            return None

        def push(self, *a, **k):
            return None

        def broadcast(self, *a, **k):
            return None

        def set_operator_hook(self, *_a, **_k):
            return None

        def is_paused(self):
            return False

        def should_abort(self):
            return False

        def is_rejected(self):
            return False

    ev, mk, br = Dummy(), Dummy(), Dummy()
    runner = SessionRunner(ev, mk, br, on_console=lambda _s: None)  # type: ignore[arg-type]
    assert runner.events is ev
    assert runner.markers is mk
    assert runner.bridge is br
    assert runner.services.events is ev


def test_attach_eeg_health_none_when_disabled():
    assert attach_eeg_health(None, SessionServices(None, None, None), enabled=False) is None  # type: ignore[arg-type]
