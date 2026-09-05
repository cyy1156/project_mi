import numpy as np


def slice_epoch(x: np.ndarray, cue: int, fs: float) -> np.ndarray | None:
    """旧版：cue-0.5 ~ cue+4s（含基线）。保留供兼容。"""
    t0 = cue + int(-0.5 * fs)
    t1 = cue + int(4.0 * fs)
    if t0 < 0 or t1 > x.shape[0]:
        return None
    return x[t0:t1, :]


def baseline_correct(epoch: np.ndarray, fs: float) -> np.ndarray | None:
    """epoch 从 -0.5s 开始；用前 0.5s 均值归零。"""
    b1 = int(0.5 * fs)
    base = epoch[:b1, :].mean(axis=0, keepdims=True)
    return epoch - base


def classification_window(epoch: np.ndarray, fs: float) -> np.ndarray:
    """去掉基线段，只留 Cue 后 0~4s。"""
    c0 = int(0.5 * fs)
    return epoch[c0:, :]


def epoch_to_class_window(x: np.ndarray, cue: int, fs: float) -> np.ndarray | None:
    ep = slice_epoch(x, cue, fs)
    if ep is None:
        return None
    ep = baseline_correct(ep, fs)
    return classification_window(ep, fs)


def task_window_cue_2_to_4(
    x: np.ndarray,
    cue: int,
    fs: float,
    baseline_sec: float = 0.5,
) -> np.ndarray | None:
    """
    任务态：Cue 后 2~4s（共 2s），用窗起点前 baseline_sec 做基线校正。
    返回 (2*fs, n_ch)；越界则 None。
    """
    n_win = int(round(2.0 * fs))
    n_base = int(round(baseline_sec * fs))
    t0 = cue + int(round(2.0 * fs))
    t1 = cue + int(round(4.0 * fs))
    if t1 - t0 != n_win:
        t1 = t0 + n_win
    base_start = t0 - n_base
    if base_start < 0 or t1 > x.shape[0]:
        return None
    base = x[base_start:t0].mean(axis=0, keepdims=True)
    win = x[t0:t1] - base
    if win.shape[0] != n_win:
        return None
    return win.astype(np.float64)


def task_window_cue_0_to_4(
    x: np.ndarray,
    cue: int,
    fs: float,
    baseline_sec: float = 0.5,
) -> np.ndarray | None:
    """
    任务态：Cue 后 0~4s（共 4s），用 Cue 前 baseline_sec 做基线校正。
    返回 (4*fs, n_ch)；越界则 None。
    """
    n_win = int(round(4.0 * fs))
    n_base = int(round(baseline_sec * fs))
    t0 = cue
    t1 = cue + n_win
    base_start = t0 - n_base
    if base_start < 0 or t1 > x.shape[0]:
        return None
    base = x[base_start:t0].mean(axis=0, keepdims=True)
    win = x[t0:t1] - base
    if win.shape[0] != n_win:
        return None
    return win.astype(np.float64)


def rest_window_with_baseline(
    x: np.ndarray,
    start: int,
    fs: float,
    win_sec: float = 2.0,
    baseline_sec: float = 0.5,
) -> np.ndarray | None:
    """截 [start, start+win_sec)，用开头 baseline_sec 均值减全窗。默认 2s→500@250Hz。"""
    n = int(round(win_sec * fs))
    if start < 0 or start + n > x.shape[0]:
        return None
    win = x[start:start + n, :].copy()
    b = int(round(baseline_sec * fs))
    if b <= 0 or b >= n:
        return None
    win = win - win[:b, :].mean(axis=0, keepdims=True)
    return win


def test_baseline():
    # 假数据：fs=250，cue=125 → 前 0.5s 全是 10，后面全是 20
    x = np.zeros((2000, 8))
    x[0:125, :] = 10
    x[125:1125, :] = 20
    win = epoch_to_class_window(x, cue=125, fs=250.0)
    assert win is not None
    assert win.shape[0] == 1000
    assert np.allclose(win.mean(), 10, atol=1e-6)

    # cue+2~4s
    x2 = np.zeros((3000, 8))
    cue = 500
    x2[cue + 500 - 125: cue + 500, :] = 5   # 基线段
    x2[cue + 500: cue + 1000, :] = 15       # 2~4s
    w2 = task_window_cue_2_to_4(x2, cue, 250.0)
    assert w2 is not None and w2.shape[0] == 500
    assert np.allclose(w2.mean(), 10, atol=1e-5)
    print("OK")


def main() -> None:
    test_baseline()


if __name__ == "__main__":
    main()
