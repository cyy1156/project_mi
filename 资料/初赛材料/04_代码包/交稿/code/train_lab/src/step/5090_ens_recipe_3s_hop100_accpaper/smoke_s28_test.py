"""方案 28 · 冒烟：R28 配置构建 + 决策树逻辑（无 prob dump）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from member_runs import normalize_member_name, parse_member_names  # noqa: E402
from replay_r28 import (  # noqa: E402
    build_r28_config,
    evaluate_decision_tree,
    fuse_for_arm,
)
from s28_config import ARM_MEMBERS, R1_ADOPT_THREE, R4_ADOPT_THREE  # noqa: E402


def _fake_member(n: int = 220, *, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    trial_id = np.arange(n) // 11
    y = np.array([rng.integers(0, 3) for _ in range(int(trial_id.max()) + 1)], dtype=np.int64)[
        trial_id
    ]
    probs = rng.dirichlet([1, 1, 1], size=n).astype(np.float32)
    for tid in np.unique(trial_id):
        mask = trial_id == tid
        probs[mask, y[mask][0]] += 0.5
        probs[mask] /= probs[mask].sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1).astype(np.int64)
    return {
        "subject": np.array([f"s{i % 5}" for i in range(n)], dtype=object),
        "fold": np.zeros(n, dtype=np.int64),
        "split": np.array(["val"] * (n // 2) + ["test"] * (n - n // 2)),
        "trial_id": trial_id,
        "t0_sec": (np.arange(n) % 11).astype(np.float32) * 0.1,
        "y": y,
        "pred": pred,
        "p_max": probs.max(axis=1).astype(np.float32),
        "probs": probs,
    }


def test_member_aliases() -> None:
    assert normalize_member_name("t-shallow") == "t_shallow"
    assert parse_member_names("shallow,eegnet") == ["shallow", "eegnet"]


def _fake_members(n_members: int, *, n: int = 220, seed: int = 0) -> list[dict]:
    base = _fake_member(n=n, seed=seed)
    out: list[dict] = []
    rng = np.random.default_rng(seed + 1)
    for i in range(n_members):
        probs = rng.dirichlet([1, 1, 1], size=n).astype(np.float32)
        for tid in np.unique(base["trial_id"]):
            mask = base["trial_id"] == tid
            label = int(base["y"][mask][0])
            probs[mask, label] += 0.4
            probs[mask] /= probs[mask].sum(axis=1, keepdims=True)
        m = dict(base)
        m["probs"] = probs.astype(np.float32)
        m["pred"] = probs.argmax(axis=1).astype(np.int64)
        m["p_max"] = probs.max(axis=1).astype(np.float32)
        out.append(m)
    return out


def test_r28_config_smoke() -> None:
    one = _fake_members(1, seed=1)
    cfg0 = build_r28_config("R0", one)
    assert cfg0.smooth_radius == 0
    fused0 = fuse_for_arm("R0", one, cfg0)
    assert fused0 is one[0]

    for arm in ARM_MEMBERS:
        n = len(ARM_MEMBERS[arm])
        members = _fake_members(n, seed=10 + n)
        cfg = build_r28_config(arm, members)
        fused = fuse_for_arm(arm, members, cfg)
        assert fused["probs"].shape[1] == 3


def test_decision_tree() -> None:
    results = {
        "R1": {"test_acc_paper": R1_ADOPT_THREE + 0.001},
        "R4": {"test_acc_paper": 0.50},
    }
    d = evaluate_decision_tree(results)
    assert d["branch"] == "A"

    results["R1"]["test_acc_paper"] = 0.50
    results["R4"]["test_acc_paper"] = R4_ADOPT_THREE + 0.001
    d = evaluate_decision_tree(results)
    assert d["branch"] == "B"

    results["R4"]["test_acc_paper"] = 0.50
    d = evaluate_decision_tree(results)
    assert d["branch"] == "C"


if __name__ == "__main__":
    test_member_aliases()
    test_r28_config_smoke()
    test_decision_tree()
    print("smoke_s28_test: OK")
