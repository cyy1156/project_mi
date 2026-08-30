"""v3 报告：窗级 acc vs 试次多数票。"""

from __future__ import annotations

from experiment_game.experiment.v3_report import build_v3_report


def _fake_trial(label: int, preds: list[int], *, valid: bool = True) -> dict:
    js = [
        {
            "t_rel": 3.0 + 0.1 * i,
            "pred": int(p),
            "gated_pred": int(p),
            "p_three": [0.1, 0.1, 0.1],
            "p_max": 0.5,
        }
        for i, p in enumerate(preds)
    ]
    # one-hot-ish p_three for causal smooth
    for j, p in zip(js, preds):
        p3 = [0.05, 0.05, 0.05]
        p3[int(p)] = 0.9
        j["p_three"] = p3
        j["p_max"] = 0.9
    return {
        "label": label,
        "valid": valid,
        "signal_bad": False,
        "judgments": js,
        "primary_judge": {
            "pred": max(set(preds), key=preds.count),
            "gated_pred": max(set(preds), key=preds.count),
            "rule": "majority_vote",
        },
        "features": {},
    }


def test_window_acc_differs_from_trial_majority():
    # 5 窗：多数票对，但有错窗 → 窗级 < 1；试次多数票 = 1
    # 因果平滑后约 [1,1,1,1,2] → 4/5
    recs = {
        "sim_b1": [
            _fake_trial(1, [1, 1, 1, 2, 2]),
            _fake_trial(2, [2, 2, 2, 2, 2]),
        ]
    }
    report = build_v3_report(
        block_order=["sim_b1"],
        block_records=recs,
        primary_judge_s=4.0,
    )
    overall = report["overall"]
    assert overall["n_windows"] == 10
    assert overall["acc_window"] is not None
    assert overall["acc_window"] < 1.0
    assert overall["n"] == 2
    assert overall["acc_argmax"] == 1.0
    assert report["blocks"]["sim_b1"]["accuracy"]["acc_window"] == overall["acc_window"]


def test_window_acc_includes_rest():
    # Rest 试次窗也计入窗级（三分类）
    recs = {
        "sim_b1": [
            _fake_trial(0, [0, 0, 0, 0, 1]),
            _fake_trial(1, [1, 1, 1, 1, 1]),
        ]
    }
    report = build_v3_report(
        block_order=["sim_b1"],
        block_records=recs,
        primary_judge_s=4.0,
    )
    overall = report["overall"]
    # 仅 L/R 时 5 窗；含 Rest 后 10 窗
    assert overall["n_windows"] == 10
    assert overall["acc_window"] is not None


def test_trial_majority_includes_rest():
    # Rest 判对 + Left 判对 → 试次多数票 2/2；若只评 L/R 则 n=1
    recs = {
        "sim_b1": [
            _fake_trial(0, [0, 0, 0, 1, 0]),
            _fake_trial(1, [1, 1, 1, 1, 1]),
            _fake_trial(2, [1, 1, 1, 1, 1]),  # Right 错
        ]
    }
    report = build_v3_report(
        block_order=["sim_b1"],
        block_records=recs,
        primary_judge_s=4.0,
    )
    overall = report["overall"]
    assert overall["n"] == 3
    assert overall["acc_argmax"] == round(2 / 3, 4)
