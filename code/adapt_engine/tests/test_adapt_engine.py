"""adapt_engine 单元测试（无真实权重依赖；tiny 线性模型验证逻辑）。

运行：C:/Users/yy/.conda/envs/cyy/python.exe -m pytest code/adapt_engine/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adapt_engine import (  # noqa: E402
    AdmissionGate,
    DEFAULT_CONSTANTS,
    DriftAction,
    DriftGuard,
    FTRecipe,
    IncrementalFinetuner,
    QuizStore,
    QuizTrial,
    ReplayPool,
    RoundController,
    judge_trial,
    serial_gating,
    split_round,
)
from adapt_engine.registry import HeadEntry, ModelRegistry  # noqa: E402


class TinyThree(nn.Module):
    """(B,8,750) → 3 类；类相关均值模式，可被 FT 学会。"""

    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.center = nn.Parameter(torch.randn(3, 8, 750, generator=g) * 0.05)
        self.head = nn.Linear(3, 3)

    def forward(self, x):  # (B,8,750)
        if x.dim() == 4:
            x = x.squeeze(1)
        d = torch.stack([((x - c) ** 2).mean(dim=(1, 2)) for c in self.center], dim=1)
        return self.head(-d)


class TinyTask(nn.Module):
    def __init__(self, seed: int = 1):
        super().__init__()
        self.lin = nn.Linear(8 * 750, 2)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        return self.lin(x.flatten(1))


def make_trials(n_trials: int, seed: int = 0) -> tuple:
    """合成试次：label c 的窗 = 类中心 + 噪声。返回 X(N_win,8,750), y(N_win,), trial_ids(N_win,)。"""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (3, 8, 750)).astype(np.float32)
    labels = []
    X = []
    for i in range(n_trials):
        c = i % 3
        for _ in range(4):  # 4 判定窗/试次
            X.append(centers[c] + rng.normal(0, 0.35, (8, 750)).astype(np.float32))
        labels.append(c)
    return np.stack(X), np.repeat(np.array(labels), 4), np.repeat(np.arange(n_trials), 4), centers


class _Holder:
    """predict(w) 供小考评估（用当前模型）。"""

    def __init__(self, model, task_model, task_p_on: float = 0.6):
        self.model = model
        self.task_model = task_model
        self.task_p_on = task_p_on

    def predict(self, w: np.ndarray) -> dict:
        with torch.no_grad():
            p3 = torch.softmax(self.model(torch.from_numpy(w[None])), -1)[0].numpy()
            pt = torch.softmax(self.task_model(torch.from_numpy(w[None])), -1)[0].numpy()
        return serial_gating(pt, p3, task_p_on=self.task_p_on)


def test_serial_gating():
    p_task_off = np.array([0.9, 0.1])
    p_task_on = np.array([0.2, 0.8])
    p3 = np.array([0.1, 0.7, 0.2])
    r = serial_gating(p_task_off, p3, task_p_on=0.6)
    assert r["pred"] == 0 and r["gated"] is True
    r = serial_gating(p_task_on, p3, task_p_on=0.6)
    assert r["pred"] == 1 and r["gated"] is False


def test_judge_trial_majority_and_reach():
    per_j = [
        {"pred": 1, "p_max": 0.8, "t": 3.0, "is_game": True},
        {"pred": 1, "p_max": 0.7, "t": 4.0, "is_game": True},
        {"pred": 1, "p_max": 0.6, "t": 5.0, "is_game": True},
        {"pred": 2, "p_max": 0.9, "t": 6.0, "is_game": True},
    ]
    v = judge_trial(1, per_j, n_levels=4)
    assert v.majority_pred == 1 and v.correct and v.n_correct == 3
    assert v.reach is False            # 3 对 < 4 档
    v2 = judge_trial(1, [dict(j, pred=1) for j in per_j], n_levels=4)
    assert v2.reach is True and v2.reach_time == 6.0   # 4 连对
    v3 = judge_trial(1, [dict(j, pred=1) for j in per_j[:3]], n_levels=3)
    assert v3.reach is True and v3.reach_time == 5.0   # 3 档提前到位


def test_split_round():
    c = DEFAULT_CONSTANTS
    s = split_round(list(range(18)), c)
    assert s.ft_trials == list(range(12)) and s.quiz_trials == list(range(12, 18))


def test_quiz_isolation_and_curve():
    """小考试次永不进 FT；曲线点 k_ft/acc 正确。"""
    X, y, tids, centers = make_trials(36, seed=3)
    model = TinyThree(seed=3)
    task = TinyTask(seed=4)
    holder = _Holder(model, task)
    fin = IncrementalFinetuner(model, FTRecipe(epochs=1, batch_size=16))
    quiz = QuizStore()
    ctrl = RoundController(fin, quiz, AdmissionGate(DEFAULT_CONSTANTS),
                           constants=DEFAULT_CONSTANTS)

    seen_by_ft: list = []

    def windows_of(ti):
        return X[tids == ti]

    def label_of(ti):
        return int(y[tids == ti][0])

    orig_train = fin.train_round

    def spy_train(w, yy, **kw):
        seen_by_ft.extend(np.asarray(yy).tolist())
        return orig_train(w, yy, **kw)

    fin.train_round = spy_train
    uniq = sorted(set(int(t) for t in tids))
    out = []
    for r in range(2):
        idx = uniq[r * 18:(r + 1) * 18]
        out.append(ctrl.run_calibration_round(idx, windows_of_trial=windows_of,
                                              label_of_trial=label_of,
                                              predict_window=holder.predict))
    quiz_ids = {t.trial_id for t in quiz.trials}
    ft_labels_ok = all(lab in seen_by_ft for lab in [0, 1, 2])
    assert len(quiz_ids) == 12 and ft_labels_ok
    # 隔离断言：quiz trial 的标签数 == FT 标签数 - 泄漏检查（用计数近似）
    assert ctrl.k_ft == 24
    assert [p.k_ft for p in quiz.curve] == [12, 24]
    assert [p.n_quiz for p in quiz.curve] == [6, 12]
    assert out[1]["gate"].n_quiz == 12  # 第 2 轮起判


def test_gate_boundaries():
    g = AdmissionGate(DEFAULT_CONSTANTS)
    assert g.update(0.9, 6, 1).status == "fail_pending"    # 规模不够
    assert g.update(0.60, 12, 2).status == "pass"          # 恰过线
    assert g.update(0.59, 12, 2).status == "extend"        # 未过但还有轮次
    assert g.update(0.59, 24, 6).status == "weak_mi"       # 超上限


def test_drift_rollback_and_freeze():
    model = TinyThree()
    fin = IncrementalFinetuner(model, FTRecipe(epochs=1))
    guard = DriftGuard(patience=2, min_rounds=2)
    states = []

    def save():
        s = fin.snapshot_state()
        states.append(s)
        return s

    # R1 升 R2 降 R3 降 → 一档回滚
    guard.before_round(save)
    guard.after_round(1, 0.7)
    guard.before_round(save)
    guard.after_round(2, 0.65)
    guard.before_round(save)
    state_before = fin.snapshot_state()
    act = guard.after_round(3, 0.60, rollback_fn=fin.rollback,
                            halve_fn=fin.halve_lr, get_state_fn=lambda: states[-1])
    assert act == DriftAction.ROLLBACK_LR
    after = fin.snapshot_state()
    ref = states[-1]
    assert all(torch.equal(after[k], ref[k]) for k in ref)  # 逐位回滚
    # 再连续两降 → 冻结
    guard.after_round(4, 0.55)
    act2 = guard.after_round(5, 0.50)
    assert act2 == DriftAction.FREEZE and guard.frozen


def test_replay_pool_balance():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (90, 8, 750)).astype(np.float32)
    y = np.repeat([0, 1, 2], 30)
    pool = ReplayPool(X, y, seed=1)
    Xr, yr = pool.sample(30)
    vals, cnts = np.unique(yr, return_counts=True)
    assert set(vals) == {0, 1, 2} and (cnts == 10).all()


def test_ft_learns_and_frozen_round():
    X, y, tids, centers = make_trials(12, seed=5)
    model = TinyThree(seed=9)
    fin = IncrementalFinetuner(model, FTRecipe(epochs=3, batch_size=16))
    holder = _Holder(model, TinyTask(seed=4))
    acc0 = np.mean([holder.predict(X[i])["pred"] == y[i] for i in range(0, 48, 4)])
    fin.train_round(X, y)
    acc1 = np.mean([holder.predict(X[i])["pred"] == y[i] for i in range(0, 48, 4)])
    assert acc1 >= acc0
    rec = fin.train_round(X[:8], y[:8], frozen=True)
    assert rec["frozen"] is True


def test_registry_ensemble_mean():
    """两个 ckpt 集成 = 概率平均（24-E 接口验证，用 tiny 模型）。"""
    import tempfile

    m1, m2 = TinyThree(seed=1), TinyThree(seed=2)
    with tempfile.TemporaryDirectory() as td:
        p1, p2 = Path(td) / "a.pt", Path(td) / "b.pt"
        torch.save({"model": m1.state_dict(), "n_outputs": 3}, p1)
        torch.save({"model": m2.state_dict(), "n_outputs": 3}, p2)
        reg = ModelRegistry.__new__(ModelRegistry)  # 绕过 braindecode 依赖，手工装头
        reg.task_heads = []
        reg.three_heads = [HeadEntry(m1.eval(), 3, str(p1)), HeadEntry(m2.eval(), 3, str(p2))]
        w = np.random.default_rng(0).normal(0, 1, (8, 750)).astype(np.float32)
        got = reg.forward_heads(w)
        with torch.no_grad():
            q = np.mean([
                torch.softmax(m1(torch.from_numpy(w[None])), -1)[0].numpy(),
                torch.softmax(m2(torch.from_numpy(w[None])), -1)[0].numpy(),
            ], axis=0)
        assert np.allclose(got["p_three"], q, atol=1e-6)


def test_train_with_early_stop():
    model = TinyThree(seed=9)
    fin = IncrementalFinetuner(model, FTRecipe(epochs=1, batch_size=8, seed=42))
    X, y, _, _ = make_trials(6, seed=1)
    acc = 0.0

    def eval_fn():
        nonlocal acc
        acc += 0.05
        return acc

    rec = fin.train_with_early_stop(X, y, eval_fn, max_epochs=8, patience=2)
    assert rec["early_stop"] is True
    assert 1 <= rec["epochs_run"] <= 8
    assert rec["best_epoch"] == rec["epochs_run"]
    assert rec["best_heldout_acc"] >= 0.05


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(fns)} TESTS PASSED")
